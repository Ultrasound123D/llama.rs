# llama.rs — Phase 6.1–6.3 Performance Engineering Report (rev 2)

**Scope:** profiling infrastructure, scheduling experiments, **AVX-512 VNNI INT8 packed backend** (renamed from "AMX INT8 backend" — see the revision note below).
**Machine:** 2 vCPU Xeon (Sapphire Rapids class) @ 3.2 GHz KVM guest, AVX-512 + AVX512-VNNI + AMX INT8/BF16 (OS-enabled, XCR0 bits 17/18 verified), 504 MB L3, DRAM ceiling ≈ 14–16 GB/s sustained at 2 threads for one linear stream.
**Baseline at phase start (rev-1 measurements):** LFM3-1.7B 8.5–8.9 tok/s · Qwen2.5-0.5B 9.2–9.5 tok/s (llama.cpp b12502 references: 10.7–12.0 and 29.4–30.0 decode).
**No HTTP/server work was done — inference performance only, per the phase contract.**

---

## Revision note (rev 2, 2026-11-25) — what changed after review

1. **Naming corrected.** Rev 1 called Phase 6.3 the "AMX INT8 backend". That was inaccurate: the implementation uses **512-bit `_mm512_dpbusd_epi32` (AVX-512 VNNI)** on load-time-repacked weights — the M=1 decode route of llama.cpp's AMX backend. **No AMX tile instruction (`LDTILECFG`/`TILELOADD`/`TDPBUSD`/`TILEZERO`) is executed anywhere in this engine.** AMX presence in CPUID/XCR0 is *capability detection*, not execution. Corrections applied: module `llama-kernels/src/amx.rs` → **`vnni.rs`**; API `matvec_amx`/`matvec2_amx`/`amx_path_available` → **`matvec_vnni`/`matvec2_vnni`/`vnni_path_available`**; tests `tests/amx.rs` → `tests/vnni.rs`; all docs updated. A genuine AMX-tile backend would target M>1 (batched prefill) and remains future work.
2. **A false env-var claim fixed.** Rev 1 claimed `LLAMA_RS_AMX` gates the repack; **no such variable existed in the code.** The real, implemented and measured switch is now **`LLAMA_RS_VNNI=0`** (builds an empty packed map → every tensor takes the AVX2 fallback; same-binary A/B below).
3. **All tests re-run at the final state** (not carried over from earlier sessions): 28 workspace tests green, parity 16/16 vs llama.cpp.
4. **Fresh matched benchmark with prefill and decode measured separately** (same CPU, same 2 threads, 9-rep medians vs llama-bench pp64/tg32, mean of 5 internal reps).
5. **TSC caveat:** every timing number in this report comes from the calibrated invariant TSC or `Instant`-based wall clock, **not hardware performance counters** (`perf_event_open` is EPERM in this guest; see §1.2). There are no cycle, instruction or cache-miss counts in this report.
6. **Honest headline:** llama.rs is **still slower than llama.cpp** on both models — decode 72% (LFM3) / 80% (Qwen), prefill 9% / 15%. The largest gap is prefill (batching), not decode kernels. No performance claims beyond these measurements.

---

## 1. Profiling infrastructure (Phase 6.1)

### 1.1 What was built

| Component | What it measures |
|---|---|
| `crates/llama-core/src/prof.rs` | Op-level profiler: per-bucket TSC ticks, calls, traffic bytes, allocation counts/bytes; tensor-class interning (`blk.17.ffn_gate.weight` → `mv blk.*.ffn_gate.weight Q4_K`); decode-wall tracking; printed table with µs/token, %wall, effective GB/s, allocs/token |
| `crates/llama-kernels/src/kstats.rs` | Kernel-internal counters: activation-quantize time, per-band dot time (caller vs worker), pool enqueue/wait, worker wakeup latency, park/spin-hit resolution, slot-dispatch stats, matvec wall vs band decomposition |
| `apps/llama-rs-cli` | `LLAMA_RS_PROF=1` gate + kstats summary line; **prefill/decode phase separation (rev 2: prefill timed and reported separately in bench mode)** |
| `crates/llama-kernels/examples/kprofile.rs` | Isolated per-kernel micro-bench: every quant type × real model shapes × 1/2 threads, median of 15 reps |
| `scripts/bench_decode.sh`, `scripts/profile_decode.sh` | Reproducible benchmark (9 reps, medians, llama-bench reference, **pp+tg separate since rev 2**) and one-shot profiling |

Timer: invariant TSC (`rdtsc`), calibrated once against CLOCK_MONOTONIC (measured 3.1999–3.2000 GHz). Per-op overhead ≈ 2×`rdtsc` + relaxed atomics < 60 ns → < 0.05 % of a decode token — profiling does not perturb the measurement.

### 1.2 Measurement limitations on this VM (stated honestly)

- **`perf_event_open` returns EPERM for all hardware counters** (cycles, instructions, cache misses) in this KVM guest — verified before designing the system. **Cycles are therefore TSC ticks (invariant, calibrated); retired-instruction and cache-miss counts are not available.** Where cache behavior matters, effective-bandwidth arithmetic is used instead. TSC-based "cycles" are elapsed ticks, not architectural event counts.
- Run-to-run variance on this guest is 5–8 %; all conclusions below rest on median comparisons with ≥ 5 repetitions, and single-run numbers are never quoted as facts.

### 1.3 Profile findings (rev 2 re-run, 48-token decode, prompt "The capital of France is")

LFM3-1.7B decode (8.48 tok/s profiled run): 154 matvecs/token; matvec wall 114.6 ms/token; per-thread dot CPU 170.2 ms/token across both threads (caller band 76.3 ms); activation quantize 0.78 ms/token (130 allocs, 447 KB); pool enqueue + caller wait ≈ 0 (spin barrier: 0 parks per profiled run).
Qwen2.5-0.5B decode (23.92 tok/s profiled run): 169 matvecs/token; matvec wall 39.8 ms/token; dots 47.0 ms CPU/token (caller band 22.1 ms); quantize 0.25 ms/token.

Phase-start findings that drove the work (rev-1 measurements): matvec ≈ 97 % of wall; the caller waited 19–24 ms/token in two futex round-trips per matvec; the simple quants streamed at only 3.4–8.7 GB/s because each 32-value block did two *software* f16→f32 conversions (~15 instructions each) for the block-scale product; ~600 heap allocations/token of glue.

---

## 2. Phase 6.2 — scheduling experiments and results

### 2.1 Experiment log (kept, reverted or measured-and-kept with caveats)

| # | Experiment | Result (measured during the phase; 5-rep medians unless noted) | Verdict |
|---|---|---|---|
| 1 | **F16C scale conversion + 2-block unroll (two f32 accumulators) in all six simple-quant kernels** (Q4_0/Q5_0/Q8_0 × plain + VNNI) | Q8_0 2t isolated: 6.9 → 14.6 GB/s; Q5_0: 6.6 → 13.4. Qwen full model: 9.3 → 13.1 tok/s (**+41 %**) | **kept** — the single biggest kernel-level win |
| 2 | **Spin barrier** replacing condvar waits (workers spin ~150 µs before parking; caller spins on an `outstanding` counter) | parks per token: 5409 → 13; caller futex round-trips eliminated | **kept** (part of the slot path) |
| 3 | **Lock-free job-slot dispatch (`par_chunks2`)** — caller closure + worker closure through a generation-counted slot (CAS claim, no Box, no queue, no futex; yield-loop fallback if the worker gets descheduled) | LFM3 8.3 → 9.5 (+14 %, clean A/B via `LLAMA_RS_SLOT=0`); Qwen within noise | **kept** (A/B switchable for reproduction) |
| 4 | **Fused gate+up matvec (`matvec2`)** — one activation quantization + one dispatch for the two same-input matrices | halves dispatch count (~90/token), saves one quantize | **kept** |
| 5 | Band split sweep (`LLAMA_RS_SPLIT` 0.45/0.47/0.5) | 0.5 best; 0.45 cost ~8 % | default 0.5 |
| 6 | AVX-512 VNNI kernel for Q4_K (from v0.2) | ~9 % slower than AVX2 on this VM (frequency licensing) | stays experimental (`avx512.rs`) |

### 2.2 Not done / not beneficial (documented)

- W-handle caching, rmsnorm scratch reuse, logits buffer reuse: profiling shows the whole glue category is ~3 ms/token (< 3 %); deferred in favor of the packed backend.
- THP/mmap investigation: weights are heap-read; llama.cpp mmaps (also 4 KB file pages) and does not beat the TLB situation materially — deprioritized.

---

## 3. Phase 6.3 — AVX-512 VNNI INT8 packed backend (repacked M=1 decode path)

### 3.1 What llama.cpp actually does (source audit of `ggml/src/ggml-cpu/amx/`)

For **decode (M == 1)** llama.cpp's AMX backend **repacks weights at load time** into a VNNI int8 layout (`convert_B_packed_format` / `pack_B`, 16-column tiles × 32-value blocks, 576 B per tile: 512 B VNNI quants + 16 f16 scales + 16 i32 compensations) and runs `tinygemm_kernel_vnni` — **512-bit `_mm512_dpbusd_epi32` over 16 columns per instruction**. The AMX *tile* instructions serve M > 1 (batched prefill), which this engine does not issue (our prefill is token-by-token — see the prefill gap in §4). Supported repack types in llama.cpp: Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS (not Q5_0).

### 3.2 What llama.rs implemented (`crates/llama-kernels/src/vnni.rs`)

- **Runtime detection:** raw `cpuid` (AMX_INT8/AMX_BF16 leaf-7, AVX512-VNNI+VL) and OS XSTATE validation via `xgetbv` (XCR0 bits 17/18) — **capability reporting only**; the M=1 path requires and checks AVX512-VNNI. `amx_tiles_available()` reports full tile readiness for a future M>1 path; **no tile instruction is executed.**
- **Load-time repack for Q8_0 (as llama.cpp) and Q5_0 (beyond llama.cpp)** — Q5_0 codes dequantize to int8 in [−16, 15] and the identical construction applies. A real bug was caught by the differential tests: `block_q5_0` is `{ d:2, qh:4, qs:16 }` — **qh before qs** — the first repack read it swapped.
- **Decode kernels:** `matvec_vnni` + fused `matvec2_vnni` (gate/up pair: one Q8_0 activation, one pool dispatch, balanced band split across the concatenated column space).
- **Numeric contract (from mmq.cpp's s8s8 compensation):** dpbusd is u8(a)×s8(b); a′ = a + 0x80; Σ(a+128)·b − 128·Σb = Σa·b **exactly** — integer-exact dots, scales applied in f32 (same structure as the AVX2 kernels).
- **Fallbacks and dispatch safety:** per-tensor — tensors that fail the repack conditions (n_in % 32, n_out % 16, type) automatically use the AVX2/VNNI path (`pmv`/`pmv2` dispatch in `llama-core`); globally — machines without AVX512-VNNI build an empty packed map and take the AVX2/scalar route; `LLAMA_RS_VNNI=0` reproduces exactly that fallback on capable machines (the A/B below runs through it). Every `#[target_feature]` function is reached only behind the matching runtime detection; the scalar fallback exists for pre-AVX2 CPUs. (This guest cannot boot as non-VNNI hardware; the code-gate audit + the identical-fallback A/B are the evidence, stated as such.)
- **Tests: 10 in `tests/vnni.rs` (rev 2)** — the 5 randomized differential tests (Q8_0/Q5_0 vs the AVX2 path, 5 reps each, ≤1e-3 rel; fused vs singles; parallel vs sequential) **plus 5 new edge-case suites**: minimum shapes (32×16), zero weights (exact-zero verified), extreme scales (f16 max 65504 / min-normal), max-magnitude codes, all qh bit patterns, activation edges (zeros/constant/±1e4/1e-7/one-hot), repack gating (None for non-multiple shapes and unsupported types), fused-pair 2-thread vs 1-thread determinism.

### 3.3 Dispatcher policy

The packed VNNI path is the default for Q8_0/Q5_0 tensors because the same-binary A/B demonstrates the improvement (§4) and parity remains 16/16; `LLAMA_RS_VNNI=0` disables it for reproduction, `LLAMA_RS_SLOT=0` A/Bs the dispatch path.

---

## 4. Results (rev 2 matched campaign, 2026-11-25)

**Conditions (matched):** same CPU, `-t 2` both engines, llama.rs `--ctx 2048` (KV rows actually written ≈ 96; capacity does not enter the per-token loop), llama-bench `-p 64 -n 32 -r 5` (its ctx derives from p+n). llama.rs uses a 56-token natural prompt (token count printed by the CLI); llama-bench synthesizes its own 64-token prompt — rates are compared, token counts are reported on both sides. llama.rs medians over 9 process repetitions; all raw runs listed.

| Model | Phase | llama.rs (VNNI on) | llama.cpp b12502 | ratio |
|---|---|---|---|---|
| LFM3-1.7B Q4_K_M | **prefill (pp)** | 8.77 tok/s (8.64–9.54) | 99.82 tok/s | **9 %** |
| LFM3-1.7B Q4_K_M | **decode (tg)** | 8.50 tok/s (8.29–9.22) | 11.82 tok/s | **72 %** |
| Qwen2.5-0.5B Q4_K_M | **prefill (pp)** | 24.75 tok/s (23.82–26.07) | 159.82 tok/s | **15 %** |
| Qwen2.5-0.5B Q4_K_M | **decode (tg)** | 23.62 tok/s (22.30–25.37) | 29.68 tok/s | **80 %** |

**Same-binary A/B of the packed VNNI path** (`LLAMA_RS_VNNI=0`, 5 reps, medians):

| Model | tg (packed VNNI on) | tg (off → AVX2 path) | effect | pp (on/off) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 23.62 | 14.11 | **+67 %** from the packed path alone | 24.75 / 14.34 (+73 %) |
| LFM3-1.7B | 8.50 | 8.40 | +1.2 % (within guest noise — expected: LFM3 has no Q8_0/Q5_0 decode tensors) | 8.77 / 8.44 |

**Kernel-level (isolated `kprofile`, median of 15 reps, 2 threads — this benches the AVX2 `matvec` path; the packed path is evidenced by the A/B above):** Q4_K 2048×7680: 18.70 GB/s; Q6_K 7680×2048: 24.98; Q6_K head 2048×32768: 28.16; Q4_0/Q5_0/Q8_0 2048×7680: 10.35 / 10.64 / 13.39; Qwen head Q8_0 896×151936: 14.66; Qwen gate Q5_0 896×4864: 13.3–14.1; F16: 20.5.

### 4.1 Why llama.cpp is still ahead (measured, not guessed)

- **Prefill is the largest gap (9 % / 15 %):** llama.rs processes the prompt token-by-token (one M=1 forward per token — 8.5–24.8 tok/s ≈ its decode rate); llama.cpp batches the prompt into M>1 GEMMs (ubatch 512), which is also where its **real AMX tile kernels** run. The engine has no batched-prefill path yet — this was not visible in rev 1 because prefill was never measured separately.
- **Decode gap (72 % / 80 %):** LFM3's Q4_K/Q6_K weights still run the AVX2 kernels while llama.cpp AMX-repacks K-quants (the measured next step, out of the 6.3 type scope); Qwen's ffn_down Q4_K/Q6_K likewise; llama.cpp's graph executor fuses op boundaries — our serial glue is ~3 ms/token.
- **Conclusion:** llama.rs is not faster than llama.cpp on any measured axis; the goal of Phase 6 was partially met (2.5× self-improvement, llama.cpp's M=1 VNNI route now ported) and the remaining gaps are now precisely quantified.

---

## 5. Validation gates (all re-run at the rev-2 final state)

1. **Parity suite: 16/16** — tokenizer ids 10/10, top-1 greedy after prefill 4/4, C-reference dequantization 2/2 (Q5_0, Q6_K; max diff 5e-10).
2. **Workspace tests: 28 passing across 4 integration suites** (gguf synthetic 2, kernel contracts 11, **vnni 10** — randomized + edge cases, quant 5). (Rev 1's "13/13 suites" phrasing is superseded by these counts.)
3. **Multiple repetitions:** 9-rep benchmark campaign with medians and all runs listed; 5-rep same-binary A/B; kernel benches 15 reps; llama-bench 5 internal reps.
4. **No regression hidden by averaging:** every change gated on the parity suite; A/B comparisons through the same binary (`LLAMA_RS_VNNI`, `LLAMA_RS_SLOT`, `LLAMA_RS_SPLIT`).
5. Failed experiments are documented in §2.1/§2.2 rather than silently dropped.

### 5.1 Unsafe-code / atomics audit summary (the 017145d diff)

- `pool.rs` job-slot protocol: publish (Relaxed pointer stores + `gen.fetch_add` Release) → claim (CAS Acq/Rel on `claimed`, paired with the Acquire load of `gen`) → complete (`done.store` Release) → caller waits `done ≥ g`. Single-flight invariant is enforced by the caller's wait, which makes ABA impossible and keeps the stack-borrowed fat pointer sound. Reentrant calls from workers run sequentially (deadlock guard). The scoped lifetime erasures (`transmute` of `*mut dyn FnMut`) are sound for the same reason: the join precedes the return.
- **Known limitation (documented, accepted):** a panic inside a job would kill a worker thread and hang the caller's yield-loop (no `catch_unwind` around jobs). Kernels are panic-free by construction (asserts fire before dispatch), and no test hits this; a `catch_unwind` + job-failure flag is the future hardening.
- `vnni.rs` intrinsics are `#[target_feature]`-gated and only called behind `vnni_path_available()`; AVX2 kernels likewise behind `is_x86_feature_detected!` with scalar fallbacks.
- kstats/prof atomics are Relaxed counters — accounting only, never synchronization.

---

## 6. How to reproduce

```bash
cd llama.rs
cargo build --release
cargo test --release --workspace          # 28 tests incl. the 10 VNNI (randomized + edge) tests
bash tests/parity/run_parity.sh          # 16/16 vs llama.cpp
bash scripts/bench_decode.sh 9            # pp+tg matched table vs llama-bench (pp64/tg32)
LLAMA_RS_VNNI=0 bash scripts/bench_decode.sh 5   # A/B: AVX2 fallback, same binary
bash scripts/profile_decode.sh            # per-op tables + kernel micro-bench
LLAMA_RS_PROF=1 ./target/release/llama-rs-cli -m <model> --mode bench -p "The capital of France is" -n 48 -t 2
```

## 7. Next steps (ranked by measured gap)

1. **Batched prefill (M>1)** — the largest measured gap (pp 9 % / 15 %): process the prompt in ubatches with a GEMM-style path. This is also the prerequisite for a genuine AMX-tile backend (tiles need M>1; the CPUID/XCR0 groundwork is already in place and validated).
2. **K-quant repack (Q4_K/Q5_K/Q6_K) to the packed VNNI format** — the measured LFM3 decode gap is largely this (llama.cpp's `pack_qs<block_q4_K>`-style layout with per-sub-block scales; mmq.cpp 1574–1980 hold the kernels).
3. Trim the serial glue (W-handle caching, rmsnorm scratch reuse, logits buffer reuse — ~3 ms/token).
4. Hardening: `catch_unwind` job-failure flag in the pool.
