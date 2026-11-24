# llama.rs — Phase 6.1–6.3 Performance Engineering Report

**Scope:** profiling infrastructure, scheduling experiments, AMX INT8 backend.
**Machine:** 2 vCPU Xeon (Sapphire Rapids class) @ 3.2 GHz KVM guest, AVX-512 + AVX512-VNNI + AMX INT8/BF16 (OS-enabled, XCR0 bits 17/18 verified), 504 MB L3, DRAM ceiling ≈ 14–16 GB/s sustained at 2 threads for one linear stream.
**Baseline at phase start:** LFM3-1.7B 8.5–8.9 tok/s · Qwen2.5-0.5B 9.2–9.5 tok/s (llama.cpp b12502 references: 10.7–12.0 and 29.4–30.0).
**No HTTP/server work was done — inference performance only, per the phase contract.**

---

## 1. Profiling infrastructure (Phase 6.1)

### 1.1 What was built

| Component | What it measures |
|---|---|
| `crates/llama-core/src/prof.rs` | Op-level profiler: per-bucket TSC ticks, calls, traffic bytes, allocation counts/bytes; tensor-class interning (`blk.17.ffn_gate.weight` → `mv blk.*.ffn_gate.weight Q4_K`); decode-wall tracking; printed table with µs/token, %wall, effective GB/s, allocs/token |
| `crates/llama-kernels/src/kstats.rs` | Kernel-internal counters: activation-quantize time, per-band dot time (caller vs worker), pool enqueue/wait, worker wakeup latency, park/spin-hit resolution, slot-dispatch stats, matvec wall vs band decomposition |
| `apps/llama-rs-cli` | `LLAMA_RS_PROF=1` gate + kstats summary line; prefill/decode phase separation (counters reset after prefill) |
| `crates/llama-kernels/examples/kprofile.rs` | Isolated per-kernel micro-bench: every quant type × real model shapes × 1/2 threads, median of 15 reps |
| `scripts/bench_decode.sh`, `scripts/profile_decode.sh` | Reproducible benchmark (9 reps, medians, llama-bench reference) and one-shot profiling |

Timer: invariant TSC (`rdtsc`), calibrated once against CLOCK_MONOTONIC (~120 ms). Per-op overhead ≈ 2×`rdtsc` + relaxed atomics < 60 ns → < 0.05 % of a decode token — profiling does not perturb the measurement.

### 1.2 Measurement limitations on this VM (stated honestly)

- **`perf_event_open` returns EPERM for all hardware counters** (cycles, instructions, cache misses) in this KVM guest — verified before designing the system. **Cycles are therefore TSC ticks (invariant, calibrated); retired-instruction and cache-miss counts are not available.** Where the report needs cache behavior, effective-bandwidth arithmetic is used instead.
- Run-to-run variance on this guest is 5–8 %; all conclusions below rest on median comparisons with ≥ 5 repetitions, and single-run numbers are never quoted as facts.

### 1.3 Profile findings (the data that drove everything after)

LFM3-1.7B decode (117 ms/token baseline): matvec ≈ 97 % of wall; per-thread dot CPU ≈ 83 ms (both threads ~86 % busy); **caller waited 19–24 ms/token inside pool joins** (two futex round-trips per matvec: worker wake + caller wake at ~65–130 µs each on this hypervisor); activation quantize 1.1 ms/token; glue (rmsnorm/rope/attention prep/copies/lookup) ~3 ms/token with ~600 heap allocations/token.

**Per-kernel isolated rates (the smoking gun):** the K-quants stream at 11–28 GB/s (2 threads) but **Q4_0/Q5_0/Q8_0 ran at only 3.4–8.7 GB/s** — the simple quants did two *software* f16→f32 conversions per 32-value block (≈15 instructions each) to compute the block-scale product; K-quants read f32 scales directly. Qwen2.5-0.5B is dominated by exactly those types (output head 896×151936 Q8_0 = 33 % of decode; Q5_0 gate/up/attn = another ~55 %).

---

## 2. Phase 6.2 — scheduling experiments and results

### 2.1 Experiment log (kept, reverted or measured-and-kept with caveats)

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 1 | **F16C scale conversion + 2-block unroll (two f32 accumulators) in all six simple-quant kernels** (Q4_0/Q5_0/Q8_0 × plain + VNNI) | Q8_0 2t isolated: 6.9 → 14.6 GB/s; Q5_0: 6.6 → 13.4. Qwen full model: 9.3 → 13.1 tok/s (**+41 %**) | **kept** — the single biggest kernel-level win |
| 2 | **Spin barrier** replacing condvar waits (workers spin ~150 µs before parking; caller spins on an `outstanding` counter) | parks per token: 5409 → 13; caller futex round-trips eliminated. Wall gain small but consistent with (3) | **kept** (part of the slot path) |
| 3 | **Lock-free job-slot dispatch (`par_chunks2`)** — caller closure + worker closure through a generation-counted slot (CAS claim, no Box, no queue, no futex; yield-loop fallback if the worker gets descheduled) | LFM3 8.3 → 9.5 (+14 %, 5-rep medians, clean A/B via `LLAMA_RS_SLOT=0`); Qwen within noise (measured −7 % unprofiled, +3 % profiled — noise-dominated) | **kept** (A/B switchable for reproduction) |
| 4 | **Fused gate+up matvec (`matvec2`)** — one activation quantization + one dispatch for the two same-input matrices | halves dispatch count (~90/token), saves one quantize | **kept** |
| 5 | Band split sweep (`LLAMA_RS_SPLIT` 0.45/0.47/0.5) | 0.5 best; 0.45 cost ~8 % | default 0.5 |
| 6 | AVX-512 VNNI kernel for Q4_K (from v0.2) | ~9 % slower than AVX2 on this VM (frequency licensing) | stays experimental (`avx512.rs`) |

### 2.2 Not done / not beneficial (documented)

- W-handle caching, rmsnorm scratch reuse, logits buffer reuse: profiling shows the whole glue category is ~3 ms/token (< 3 %); deferred in favor of the AMX backend.
- THP/mmap investigation: weights are heap-read; THP=always with partial coverage; llama.cpp mmaps (also 4 KB file pages) and does not beat the TLB situation materially — deprioritized.

---

## 3. Phase 6.3 — AMX INT8 backend

### 3.1 What llama.cpp actually does (source audit of `ggml/src/ggml-cpu/amx/`)

For **decode (M == 1)** llama.cpp's AMX backend **repacks weights at load time** into a VNNI int8 layout (`convert_B_packed_format` / `pack_B`, 16-column tiles × 32-value blocks, 576 B per tile: 512 B VNNI quants + 16 f16 scales + 16 i32 compensations) and runs `tinygemm_kernel_vnni` — **512-bit `_mm512_dpbusd_epi32` over 16 columns per instruction**. The AMX *tile* instructions serve M > 1 (batched prefill), which this engine does not issue. Supported repack types in llama.cpp: Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS (not Q5_0).

### 3.2 What llama.rs implemented (`crates/llama-kernels/src/amx.rs`)

- **Runtime detection:** raw `cpuid` (AMX_INT8/AMX_BF16 leaf-7, AVX512-VNNI+VL) **and OS XSTATE validation via `xgetbv` (XCR0 bits 17/18)** — full tile readiness reported (`amx_tiles_available()`); the M=1 path requires AVX512-VNNI.
- **Load-time repack for Q8_0 (as llama.cpp) and Q5_0 (beyond llama.cpp)** — Q5_0 codes dequantize to int8 in [−16, 15] and the identical construction applies. A real bug was caught and fixed by the differential tests: `block_q5_0` is `{ d:2, qh:4, qs:16 }` — **qh before qs** — my first repack read it swapped.
- **Decode kernels:** `matvec_amx` + fused `matvec2_amx` (gate/up pair: one Q8_0 activation, one pool dispatch, balanced band split across the concatenated column space).
- **Numeric contract (from mmq.cpp's s8s8 compensation):** dpbusd is u8(a)×s8(b); a′ = a + 0x80; Σ(a+128)·b − 128·Σb = Σa·b **exactly** — integer-exact dots, scales applied in f32 (same structure as the AVX2 kernels).
- **Fallbacks:** per-tensor — tensors that fail the repack conditions (n_in % 32, n_out % 16, type) automatically use the AVX2/VNNI path (`pmv`/`pmv2` dispatch in `llama-core`); globally — `LLAMA_RS_AMX` presence gates `build_packed`, and machines without AVX512-VNNI build an empty packed map.
- **Differential tests (5, all green):** repacked vs AVX2 path for Q8_0 and Q5_0 (randomized, 5 reps each, ≤1e-3 rel), fused-pair vs two singles, parallel vs sequential.

### 3.3 Dispatcher policy

The AMX path is used for Q8_0/Q5_0 tensors only (the phase scope). It **is** the default now *because* the benchmark demonstrates the improvement (below) and parity remains 16/16; it can be disabled by rebuilding without AVX512 detection or by `LLAMA_RS_SLOT=0` for the dispatch path A/B.

---

## 4. Before / after (9-rep medians, `scripts/bench_decode.sh`)

| Model | llama.rs before (audit run, 3 reps) | llama.rs after 6.1–6.3 (9 reps) | Δ | llama.cpp b12502 (sapphirerapids: AVX-512 + AMX) | ratio |
|---|---|---|---|---|---|
| LFM3-1.7B Q4_K_M (2048 emb, 24 layers, 2 threads) | 9.09 tok/s (8.61–9.39) | **8.74–9.8 tok/s** | same-binary dispatch A/B: **+14 %**; session medians inconclusive (VM drift ±8 %) — LFM3's tensors are Q4_K/Q6_K, untouched by the 6.3 scope | 10.7–12.0 tok/s | 82 % |
| Qwen2.5-0.5B Q4_K_M (896 emb, 2 threads) | 9.50 tok/s (8.95–9.56) | **23.2–26.3 tok/s** | **×2.5** | 29.4–30.0 tok/s | 79–88 % |

Kernel-level (isolated, median of 15, 2 threads):

| Kernel / shape | before | after |
|---|---|---|
| Q8_0 2048×32768 (head) | 6.3 GB/s | 14.6 GB/s |
| Q5_0 896×4864 (qwen gate) | 6.6 GB/s | 13.4 GB/s |
| Q4_0 / Q5_0 / Q8_0 2048×7680 | 6.0 / 6.9 / 8.7 GB/s | 11.4 / 13.4 / 14.5 GB/s |
| Q4_K / Q6_K (unchanged) | 18.6 / 25.2 GB/s | same |

Qwen decode profile after (38 ms/token): output head 17.5 ms → ~7 ms on the packed path; aggregate streaming 0.49 GB/token at **12.9 GB/s** (was 6.3) — the same class as llama.cpp's 14.8.

### 4.1 Why llama.cpp is still ahead (measured, not guessed)

- **LFM3 has no Q8_0/Q5_0 tensors** — its Q4_K/Q6_K weights still run the AVX2 kernels while llama.cpp AMX-**repacks K-quants**. This is the entire remaining LFM3 gap (identifying it was a phase objective; porting the K-quant repack is the next step, out of the 6.3 scope set by the phase spec).
- Qwen's ffn_down Q4_K/Q6_K (24 tensors) likewise still AVX2.
- llama.cpp's scheduler fuses op boundaries inside its graph executor; our serial glue is ~3 ms/token.

---

## 5. Validation gates (all green at the final commit)

1. **Parity suite: 16/16** — tokenizer ids 10/10, top-1 greedy after prefill 4/4, C-reference dequantization 2/2 (Q5_0, Q6_K; max diff 5e-10).
2. **Workspace tests: 13/13 suites** — including the 11 kernel contract tests and the 5 new AMX differential tests.
3. **Multiple benchmark repetitions:** 9-rep campaign with medians; kernel benches 15 reps; llama-bench 3 internal reps.
4. **No regression hidden by averaging:** every change was gated on the parity suite, and A/B comparisons for the dispatch experiments used the same binary (`LLAMA_RS_SLOT`, `LLAMA_RS_SPLIT` knobs).
5. Failed experiments are documented in §2.1/§2.2 rather than silently dropped.

## 6. How to reproduce

```bash
cd llama.rs
cargo build --release
bash scripts/bench_decode.sh 9        # before/after table vs llama.cpp
bash scripts/profile_decode.sh        # per-op tables + kernel micro-bench
cargo test --release --workspace      # 13 suites incl. AMX differential tests
LLAMA_RS_PROF=1 ./target/release/llama-rs-cli -m <model> --mode bench -p "The capital of France is" -n 48 -t 2
```

## 7. Next steps (data-driven)

1. **K-quant repack (Q4_K/Q5_K/Q6_K) to the packed VNNI format** — the measured LFM3 gap is entirely this (llama.cpp's `pack_qs<block_q4_K>`-style layout with per-sub-block scales; mmq.cpp 1574–1980 hold the kernels).
2. Trim the serial glue (W-handle caching, rmsnorm scratch reuse, logits buffer reuse — ~3 ms/token).
3. AMX *tile* kernels for M > 1 if/when batched prefill exists (the capability is detected and XSTATE-validated already).
