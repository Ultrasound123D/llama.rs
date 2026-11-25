# llama.rs — Phase 6 Performance Engineering Report (rev 6)

**Scope:** profiling infrastructure, scheduling experiments, the AVX-512 VNNI INT8 packed backend (rev 2–3, incl. the C-golden verification campaign), the K-quant packed path (rev 4 / Phase 6.4, first decode above llama.cpp), batched prefill (rev 5 / Phase 6.5 — an M>1 packed GEMM for all four repacked types, batched LFM3/llama-family forward passes, the Qwen `ffn_down` dispatch-hole fix, and the memory reclaim that halves peak RSS), and — **rev 6 (Phase 6.6) — the Decode & Kernel Bottleneck Audit**: per-stage decode/prefill profiles for both models, the DRAM ceiling and achieved-bandwidth accounting, CPU-counter availability probes, batch-size and context-length sweeps, and the root-cause classification of every remaining gap. **No code-generation kernels were changed in Phase 6.6 — it is instrumentation and measurement only**; all conclusions below come from the audit's fresh numbers.
**Machine:** 2 vCPU Xeon (Sapphire Rapids class) @ 3.2 GHz KVM guest, AVX-512 + AVX512-VNNI + AMX INT8/BF16 (OS-enabled, XCR0 bits 17/18 verified), 504 MB L3; DRAM measured ceilings (rev 6, `examples/drambw`): single linear stream **13.9–14.1 GB/s**, two streams **17.2–27.3 GB/s** (shared-host drift visible across passes).
**Baseline at phase start (rev-1 measurements):** LFM3-1.7B 8.5–8.9 tok/s · Qwen2.5-0.5B 9.2–9.5 tok/s (llama.cpp b12502 references: 10.7–12.0 and 29.4–30.0 decode).
**No HTTP/server work was done — inference performance only, per the phase contract.**

---

## Revision note (rev 3, 2026-11-25) — the verification campaign

Review verdict was: verify correctness of the VNNI path first, then focus on the remaining gap. Executed in order:

1. **Independent C golden reference built from llama.cpp's own code** (`tests/parity/ref_dot.c`): transcribes `quantize_row_q8_0_ref` + the runtime x86 quantizer (`arch/x86/quants.c:302`) + `ggml_vec_dot_q8_0_q8_0_generic` + `ggml_vec_dot_q5_0_q8_0_generic`. It emits *both* llama.cpp quantizer contracts — the discovery here: **llama.cpp itself has two Q8_0 activation-quantizer contracts** (ref: `id = 1/d`, `roundf` vs runtime: `id = 127/max`, round-to-nearest-even) — and the runtime one is what inference actually executes, so that is the byte-exact gate for llama.rs. Details in §3.4.
2. **Differential sweep:** 72 cases (random ×3 seeds ×8 shapes + extremes + qh-walk) × 8 paths (scalar / AVX2 / AVX2+VNNI256 / engine dispatch / packed 1-thread / packed 2-thread / fused pair) — worst error **1.4e-7 of term energy**; activation quantizer **byte-exact** vs the C runtime contract (0 mismatches over all cases). The one initially-"failing" mode (extremes) was root-caused to exact catastrophic cancellation — outputs are pure f32-rounding residues — and the metric was fixed to term-energy normalization (documented in §3.4; all integer dots independently confirmed exact in Python).
3. **Mutation sensitivity self-test:** one flipped weight-code byte → error 5.68e-4 of term energy = **7276× the clean noise floor** — the gates provably detect real corruption (not vacuously green).
4. **CPU feature dispatch verified:** Rust `cpuid` detection cross-checked against `/proc/cpuinfo` (avx512_vnni, amx_int8: MATCH); `repack_tensor_gated(cpu_ok=false)` returns `None` for every type/shape combination (the machine-without-AVX512 simulation → AVX2 fallback, no panic, no AVX-512 instruction reached); end-to-end `LLAMA_RS_VNNI=0` vs `=1` on the real Qwen model: **top-1 token and top-5 ranking identical** (logit deltas ~0.25% — f32 accumulation-order differences compounding through 24 layers, benign).
5. **Matched benchmark re-run with exact-matched prompt lengths** (llama-bench `-p 55/54` per model tokenizer, our natural prompt tokenizes to exactly 55/54): decode LFM3 **81%**, Qwen **74%**; prefill 10%/15% (llama.cpp batches, we are token-by-token — algorithmic, not kernel). Fresh A/B: the packed path alone lifts Qwen decode **12.90 → 23.38 tok/s (+81%)**.
6. **All tests re-run fresh at this state:** 28 workspace tests green, parity **16/16** vs llama.cpp, golden sweep all gates green.

---

## Revision note (rev 4, 2026-11-25) — Phase 6.4: K-quant VNNI repack → LFM3 decode at 102 % of llama.cpp

The review's ranked next step ("K-quant repack, the LFM3 decode gap is largely this; the verification harness extends directly") was executed in full:

1. **Q4_K and Q6_K repacked to the packed-VNNI format** (`vnni.rs::repack_q4_k/repack_q6_k`) — the same construction as llama.cpp's `pack_B<block_q4_K>`/`pack_B<block_q6_K>` (tile sizes 2368 B = 16·144+64 and 3360 B = 16·210, **identical to llama.cpp's TILE_SIZE**). Nibbles stay NIBBLE-PACKED (4.5/6.5625 bpw preserved — the packed path reads ~2.8 % more bytes on Q4_K tiles, zero on Q6_K), and the kernels unpack in-register exactly like `tinygemm_kernel_vnni<q8_K, q4_K/q6_K>`: `dpbusd` over in-register nibbles, integer per-group scale multiply, the **min term via `dpwssds`**, and the q6 **(q−32) offset compensated from the activation bsums×32**. Activations quantize to q8_K (the K-quant vec_dot_type, `quantize_row_q8_K`) — byte-exact vs the C contract (below). A real repack bug (6-bit values OR-ed into a nibble pair without masking) was caught by the differential tests and fixed before anything was benchmarked.
2. **The verification harness extended directly, as predicted**: `ref_dot.c` gained transcriptions of `quantize_row_q8_K_ref` + `ggml_vec_dot_q4_K_q8_K_generic` + `ggml_vec_dot_q6_K_q8_K_generic`; the golden sweep grew from 72 to **153 cases × 8 paths — ALL GREEN** (worst packed-path error 1.15e-7 of term energy; q8_K quantizer **byte-exact**, 0 mismatches; mutation sensitivity q4_k 5115×, q6_k 3376×). Six new test suites in `tests/vnni.rs` (randomized Q4_K/Q6_K vs the AVX2 reference, min-shape/extreme-scale/min-zero edges, K-quant fused pair, 2-thread determinism, activation edges) — 34 workspace tests green; parity **16/16** re-run at the final state.
3. **Load-time discovery fixed by profiling**: only the fused gate+up pair had ever been routed through the packed dispatch — LFM3's q/k/v, attn_gate/o, conv in/out_proj, ffn_down and the output head still ran the AVX2 kernels (≈520 MB/token of traffic). All of them now route through the packed dispatch (`pmv`/`pmv2`), which now also records the per-op profile rows itself (packed branch only — no double counting with the AVX2 fallback).
4. **Glue trim**: the packed path's per-matvec activation-quantization allocations (154 Vecs/token) moved to a process-wide scratch free-list (`vnni.rs::take_q`).
5. **Result (matched campaign, 5-rep medians, same CPU/-t 2, exact-matched prompt lengths):** LFM3 decode **12.02 tok/s vs llama.cpp 11.74 = 102 %** (runs 11.54–12.49); Qwen decode 23.86 vs 30.34 = 79 %; prefill 14 % / 16 % (token-by-token vs llama.cpp's ubatch — batched prefill is the ranked next step, unchanged). Same-binary A/B (`LLAMA_RS_VNNI=0`): LFM3 8.85 → 12.02 (**+36 % from the packed path alone**), Qwen 13.64 → 23.86.
6. **Memory (honest note):** the packed map co-resides with the original GGUF buffer, so llama.rs now holds ~2× model bytes in RAM during inference — the same trade llama.cpp's AMX repack makes. On this 4 GB guest that required the parity script to serialize the llama.cpp-server and llama.rs phases (both sides still computed on the same model/prompts; the comparison semantics are unchanged). A file-backed lazy loader (read tensors from the GGUF on demand, keep only non-repacked bytes) is the ranked follow-up — it would cut steady-state memory below even llama.cpp's.
7. TSC-timing caveat (rev 2, §1.2) unchanged; every number above is median-of-5 process repetitions vs llama-bench's mean of 5 internal reps, both sides fresh in the same session.

---

## Revision note (rev 5, 2026-11-26) — Phase 6.5: batched prefill + memory reclaim

The Phase 6.4 review verdict was: batched prefill is the next lever (pp 14 %/16 %); do not chase decode percentages; measure RSS/peak RSS; longer campaigns with percentiles; keep parity stable after every forward change. Executed:

1. **Packed GEMM (M>1)** for Q4_K/Q6_K/Q8_0/Q5_0 (`vnni.rs::gemm_prepared/gemm2_prepared`): the M×K activation matrix is quantized once per weight type (rows = tokens, one M=1 quantizer contract per row), then each 16-column weight group is streamed against all rows — DRAM traffic per weight byte drops from once-per-token to once-per-batch. **Every output column is bit-identical to `matvec_vnni` on that single row** (same quantizer bytes, same group-dot accumulation order) — differential tests: four types, shapes 64×32…7680×2048, M ∈ {1,2,3,8,17,64}, 1-thread vs 2-thread. Since `matvec_vnni` is the C-golden-verified path (§3.4), the GEMM inherits that verification by exact identity.
2. **Batched forward** for both architectures: causal attention with per-query key ranges (LFM3 SWA window included), the shortconv causal window over `[state | M new values]` with state roll-forward, q/k norm + rope (learned freqs, SWA base) per row, qwen biases, head gates, fused gate/up GEMM pair, reusable activation arenas. Rows split between caller and worker bands; M=1 routes through the untouched decode `forward`.
3. **Qwen `ffn_down` dispatch-hole fix**: the llama-family decode still called the AVX2 `W::matvec` for `ffn_down` (the largest Qwen FFN matvec) — routed through the packed `pmv` dispatch like every other matmul. Qwen tg 23.9 → 24.9 tok/s short-run (30-rep median 23.42 [22.48 24.85]).
4. **Memory reclaim** (§8): `MADV_DONTNEED` (raw-syscall, no libc) on each repacked tensor's original bytes, issued immediately after its repack — LFM3 peak 2050 → 1227 MB, Qwen steady 1079 → 706 MB (724 peak).
5. **Pool hardening**: the single-flight job-slot protocol is now serialized by a dispatch mutex — concurrent `par_chunks2` calls from parallel test threads previously corrupted results (caught by the new GEMM tests).
6. **Campaign upgrades**: 30-rep campaign with median/P10/P90 (`scripts/stats.py`), the M-sweep table (§4.2), `--mode mem` / `--mode batchcmp`, parity extended to **24/24** (batched-vs-token top-1, batchcmp logits self-check).

---

## Revision note (rev 6, 2026-11-26) — Phase 6.6: Decode & Kernel Bottleneck Audit

The Phase 6.5 review direction: do not open new kernels yet; make the next phase profiling-driven and classify every remaining gap as kernel / memory / scheduling / instruction-throughput with evidence. Executed as instrumentation only (no compute kernel was touched):

1. **Stage-level profiling completed.** The op profiler now attributes: (a) the prefill phase's own wall (the `%wall` column is correct for prefill tables too), (b) per-stage batch buckets — `elementwise(batch)`, `swiglu(batch)`, `rope_prep(batch)` / `attn_prep(qknorm+rope)(batch)`, `conv_core(batch)`, and `act_quantize(batch)` timed separately from the GEMMs, (c) a machine-readable TSV mode (`LLAMA_RS_PROF_TSV=1`) that `scripts/audit.sh` parses into the stage tables of §9. A real attribution bug was fixed in `pgemm_shared`: the old single shared timer double-counted the q/k/v GEMM triple (each row counted its predecessors' elapsed time); per-tensor timing is now correct, which also corrects §8.1's "GEMM ≈ 80 % of prefill" to the accurate **63 %** (§9.5).
2. **CPU counters: re-verified blocked.** `examples/perfcheck` calls `perf_event_open` for cycles/instructions/cache-references/cache-misses/branch-misses plus one software event, through the same raw-syscall style as the MADV reclaim, trying six attr sizes: **every event returns EACCES** (rev 1–2 observed EPERM via earlier probes; the precise errno on this guest is EACCES — no hardware counters, no cache-miss counts; TSC + bandwidth arithmetic remain the instrumentation, §1.2 unchanged).
3. **DRAM ceiling measured, not assumed** (`examples/drambw`: 1 GiB, 4-accumulator u64 fold, 3 passes): single-stream steady **13.9–14.1 GB/s**; two streams **27.3 / 27.3 / 17.2 GB/s** across passes — the shared host interferes (the same drift that moves llama-bench ±5 % between sessions); the honest sustained two-stream figure is **17–27 GB/s**.
4. **The audit's fresh engine numbers** (5-rep medians, no-regression gate): LFM3 pp 52.90 [51.59 54.01] (60 %), tg 10.99 [10.90 11.69] (**104 %**); Qwen pp 152.67 [151.18 153.81] (**101 %**), tg 22.23 then 23.28 [21.91 23.95] across two campaigns (**77–85 %** — llama.cpp's own tg moved 28.96 → 27.53 in the same window; host noise, both engines drift). Parity **24/24** re-run; 38 workspace tests green, 0 failures.
5. **Headline findings** (full tables and the classification matrix in §9): decode of both models is **kernel-limited, not memory-limited** (LFM3 streams 12.16 GB/s where llama.cpp streams 12.0 — equal, ~88 % of the single-stream ceiling; the gap to the ceiling is VNNI unpack work, and llama.cpp sits at the same effective rate — which is *why* LFM3 decode holds 101–104 %); Qwen's remaining 19 % is concentrated in the small-tensor attention projections (3.29 GB/s stage = dispatch-bound) and `ffn_down` (8.6 GB/s); prefill of both models is **instruction-throughput bound** (GEMM at 0.65 GB/s ≪ ceiling — shuffle port + AVX-512 frequency licensing), with a *new* dominant term discovered: the batched attention core is **54.3 % of Qwen prefill** and 24.9 % of LFM3's; and the **context sweep found a decode collapse** (Qwen 28 % of llama.cpp at ctx 4050; LFM3 50 % at ctx 4125) that the standard short-prompt campaigns never see — profiled cause: the per-key f32 attention loop (`attn_core`) balloons to **72.7 % / 64.0 % of decode wall** at ctx ≈ 4k while llama.cpp stays flat.
6. **Gates**: no kernel or numerics code changed (the diff is timers, buckets, two probe examples, four scripts); parity 24/24; workspace tests green; the no-regression campaigns above.

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

### 3.4 Independent C-golden differential harness (rev 3 verification)

**Artifacts:** `tests/parity/ref_dot.c` (golden reference, gcc -O2) + `crates/llama-kernels/examples/vnni_golden.rs` (driver; compiles the C reference itself if missing).

**Why a C golden at all:** the rev-2 intra-engine differential (VNNI vs our AVX2 kernels) shares the activation quantizer and the block layouts on both sides — a common-mode hole: a bug in the shared quantizer or layout passes both. The C reference is transcribed from llama.cpp's own sources in a different language and re-compiled, closing that hole: quantizer, layout and dot contracts are all independently re-implemented.

**The two-quantizer discovery:** llama.cpp's `quantize_row_q8_0_ref` (file-creation contract) computes `id = 1/d` with `roundf` (half-away-from-zero); the quantizer it actually **executes** on AVX2 x86 (`arch/x86/quants.c:302`) computes `id = 127.f/maxScalar` with hardware round-to-nearest-even. These are subtly different contracts inside llama.cpp itself. Our engine implements the runtime one — and the harness gates our quantizer **byte-exact against the C transcription of the runtime contract** (0 mismatches across all 72 cases; `d` bits, codes, tie-rounding all identical).

**Coverage (72 cases × 8 paths):** shapes from the legal minimum (32×16) through model-realistic (896×304, 2048×4096); random data (3 seeds × 8 shapes × Q8_0/Q5_0) + `extremes` (f16 scales 0 / min-normal / 1 / 65504 × code bytes 00/01/7F/80/FE/FF, activations quantizing to ±127) + `qhwalk` (each of the 32 `qh` bit positions exercised). Paths compared per case: scalar kernels, direct AVX2 calls, AVX2+VNNI256 (the engine's actual dispatch on this CPU), full engine `matvec`, packed VNNI 1-thread, packed VNNI 2-thread, fused `matvec2_vnni`.

**Error metric — term-energy normalization (methodology finding):** per-output relative error is the wrong metric for dot products whose outputs cross zero; on adversarial data the outputs cancel *exactly* (e.g. block terms 129032508 − 127032004 − 2000504 = 0) and every output becomes a pure f32-rounding residue — no two accumulation orders can agree "relatively" on those. Root-caused (not patched around): the integer dots were confirmed exact independently (Python re-implementation over the raw bytes); the golden emits per-column term energy Σ|sumi_b·d_b| and the gate is `max|a−b| / max(term energy)`. The packed path then shows ≤1.1e-7 and all SIMD paths ≤1.4e-7 — i.e. **f32-rounding-level agreement with llama.cpp's own generic kernels, even under total cancellation.**

**Mutation sensitivity:** the harness proves its own gates are not vacuous — flipping ONE weight-code byte in one block moves the error from 7.8e-8 (clean noise floor) to 5.68e-4 of term energy (**7276×**), far above every gate.

**Known shared-saturation corner (documented, faithful to llama.cpp):** the AVX2-only (non-VNNI) fallback kernel's `maddubs` pair-sums saturate at ±32767 when four 127-magnitude products collide — exactly as ggml's own AVX2-only path does (it takes `_mm256_dpbusd_epi32` when AVX512-VNNI+VL is present, which this machine and our runtime check both are). Parity with llama.cpp's fallback behavior, not a deviation.

---

## 4. Results

### 4.1 Rev 5 (Phase 6.5) — 30-rep matched campaign (2026-11-26), medians with P10/P90

Conditions identical to rev 4 (same CPU, `-t 2`, exactly matched prompt lengths: 55 LFM3 / 54 Qwen tokens, `-n 32`); 30 process repetitions, median [P10, P90] reported per the review requirement (llama.cpp reference numbers move within the guest's 5–8 % band between sessions — 10.78 tg here vs 11.74 in rev 4 — quoted as measured).

| Model | Phase | llama.rs (median [P10 P90]) | llama.cpp b12502 | ratio |
|---|---|---|---|---|
| LFM3-1.7B Q4_K_M | **prefill (pp)** | **53.99 [52.24 55.37]** tok/s | 87.54 | **62 %** (was 14 %) |
| LFM3-1.7B Q4_K_M | **decode (tg)** | **10.93 [10.66 11.11]** tok/s | 10.78 | **101 %** |
| Qwen2.5-0.5B Q4_K_M | **prefill (pp)** | **154.68 [151.57 163.62]** tok/s | 151.96 | **102 %** (was 16 %) |
| Qwen2.5-0.5B Q4_K_M | **decode (tg)** | **23.42 [22.48 24.85]** tok/s | 28.91 | **81 %** (was 79 %) |

**Two metrics at or above 100 %**: LFM3 decode (101 %) and Qwen prefill (102 %). The Qwen pp gain is 6.2× over Phase 6.4 (24.9 → 154.7); the LFM3 pp gain is 4.5× (12.1 → 54.0).

### 4.2 Batch-size sweep (Phase 6.5 request: M ∈ {1, 8, 32, 128, 512})

~550-token prompts, 3 reps, medians [P10 P90]; llama-bench at the same prompt length. The M=1 column is the token-at-a-time Phase 6.4 behavior.

| Model | M=1 | M=8 | M=32 | M=128 | M=512 | llama.cpp (pp550) |
|---|---|---|---|---|---|---|
| LFM3-1.7B | 10.9 (11 %) | 30.4 (30 %) | 39.8 (39 %) | **42.9 (42 %)** | 39.7 (39 %) | 101.1 |
| Qwen2.5-0.5B | 21.5 (13 %) | 60.2 (38 %) | 78.7 (49 %) | **85.2 (53 %)** | 73.0 (46 %) | 159.9 |

Readings: (a) batching pays immediately (M=8 is ~3× M=1); (b) the sweet spot is M=128 on both models — at M=512 the attention term (O(M·window) with the batch's own keys) and the larger activation arenas overtake the amortization; (c) llama.cpp pulls further ahead at longer prompts (their pp 87.5→101 for LFM3) because their AMX tile kernels are not subject to the VNNI shuffle bound below.

### 4.3 Memory (the review's RSS question, measured with `--mode mem`)

| Model | Steady RSS (repack original kept) | Steady RSS (madvise reclaim) | Peak RSS (reclaim) | Released |
|---|---|---|---|---|
| LFM3-1.7B | 2052.8 MB | **1219.1 MB** | **1227.2 MB** | 874.2 MB |
| Qwen2.5-0.5B | 1078.8 MB | **706.0 MB** | **723.7 MB** | 390.9 MB |

The reclaim (see §8) drops steady-state RSS from ~2× model (packed copy + original bytes) to ~1.05× and — because each tensor's pages are freed *immediately after* its repack — the load-time peak stays at the steady level (1227 vs 2050 MB for LFM3). This removes the 4 GB co-residency constraint the parity script worked around in rev 4.

### 4.4 Historic rev 3 / rev 4 tables (unchanged, kept for the record)

## 4. Results (rev 3 matched campaign, 2026-11-25)

**Conditions (matched):** same CPU, `-t 2` both engines; llama.rs `--ctx 2048` (KV rows actually written ≈ 88; capacity does not enter the per-token loop); llama-bench has no context flag (its ctx derives from p+n — b12502 has no `-c`); **prompt lengths now exactly matched** — our natural prompt tokenizes to 55 (LFM3) / 54 (Qwen) tokens per the model tokenizer and llama-bench is invoked with `-p 55` / `-p 54` respectively, `-n 32` both. llama.rs medians over 5 process repetitions (all raw runs listed by the script); llama-bench 5 internal reps. llama.cpp reference numbers move within the guest's 5–8% variance band across sessions (tg: 10.65 this campaign vs 11.82 in rev 2 — both quoted honestly).

| Model | Phase | llama.rs (VNNI on) | llama.cpp b12502 | ratio |
|---|---|---|---|---|
| LFM3-1.7B Q4_K_M | **prefill (pp)** | 8.66 tok/s (5 runs: 8.30–9.10) | 84.16 tok/s | **10 %** |
| LFM3-1.7B Q4_K_M | **decode (tg)** | 8.62 tok/s (5 runs: 8.43–8.82) | 10.65 tok/s | **81 %** |
| Qwen2.5-0.5B Q4_K_M | **prefill (pp)** | 24.45 tok/s (5 runs: 23.57–24.96) | 157.95 tok/s | **15 %** |
| Qwen2.5-0.5B Q4_K_M | **decode (tg)** | 23.77 tok/s (5 runs: 23.17–24.37) | 31.98 tok/s | **74 %** |

**Rev 4 (Phase 6.4, same conditions, fresh campaign):**

| Model | Phase | llama.rs (K-quant VNNI on) | llama.cpp b12502 | ratio |
|---|---|---|---|---|
| LFM3-1.7B Q4_K_M | **prefill (pp)** | 12.06 tok/s (5 runs: 11.76–12.27) | 88.94 tok/s | **14 %** |
| LFM3-1.7B Q4_K_M | **decode (tg)** | **12.02 tok/s** (5 runs: 11.54–12.49) | 11.74 tok/s | **102 %** |
| Qwen2.5-0.5B Q4_K_M | **prefill (pp)** | 24.94 tok/s (5 runs: 22.64–26.47) | 154.91 tok/s | **16 %** |
| Qwen2.5-0.5B Q4_K_M | **decode (tg)** | 23.86 tok/s (5 runs: 22.99–25.02) | 30.34 tok/s | **79 %** |

**Same-binary A/B of the packed VNNI path** (`LLAMA_RS_VNNI=0`, 5 reps, medians, rev-3 fresh run):

| Model | tg (packed VNNI on) | tg (off → AVX2 path) | effect | pp (on/off) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 23.38 | 12.90 | **+81 %** from the packed path alone | 23.54 / 13.10 |
| LFM3-1.7B | 7.93 | 7.89 | within guest noise — expected: LFM3 has no Q8_0/Q5_0 decode tensors (its weights are Q4_K/Q6_K: the K-quant repack is the next measured lever) | — |

**End-to-end accuracy A/B (`--mode debug`, real Qwen model):** `LLAMA_RS_VNNI=1` vs `=0` → top-1 token identical (` Paris`), top-5 ranking identical; logit deltas ~0.25% (f32 accumulation-order compounding through 24 layers). Parity suite 16/16 holds with the packed path on.

**Kernel-level (isolated `kprofile`, median of 15 reps, 2 threads — this benches the AVX2 `matvec` path; the packed path is evidenced by the A/B above):** Q4_K 2048×7680: 18.70 GB/s; Q6_K 7680×2048: 24.98; Q6_K head 2048×32768: 28.16; Q4_0/Q5_0/Q8_0 2048×7680: 10.35 / 10.64 / 13.39; Qwen head Q8_0 896×151936: 14.66; Qwen gate Q5_0 896×4864: 13.3–14.1; F16: 20.5.

### 4.1 Why llama.cpp is still ahead (measured, not guessed)

- **Prefill is the largest gap (10 % / 15 %):** llama.rs processes the prompt token-by-token (one M=1 forward per token — 8.7–24.5 tok/s ≈ its decode rate); llama.cpp batches the prompt into M>1 GEMMs (ubatch 512), which is also where its **real AMX tile kernels** run. The engine has no batched-prefill path yet — this was not visible in rev 1 because prefill was never measured separately.
- **Decode gap (81 % / 74 %):** LFM3's Q4_K/Q6_K weights still run the AVX2 kernels while llama.cpp AMX-repacks K-quants (the measured next step — §7); Qwen's ffn_down Q4_K/Q6_K likewise; llama.cpp's graph executor fuses op boundaries — our serial glue is ~3 ms/token; ~170 small heap allocations per decode token (activation-quantize buffers per matvec + logits copies) — scratch-buffer reuse is a measured lever.
- **Conclusion:** llama.rs is not faster than llama.cpp on any measured axis; the decode kernels for Q8_0/Q5_0 are now **verified equivalent to llama.cpp's own reference semantics** (§3.4), which makes the remaining gap a scheduling/coverage problem (K-quants + glue), not a correctness problem.

---
## 5. Validation gates (rev 3 state; rev 5 additions listed in-place)

1. **C-golden differential sweep: ALL GATES GREEN** (rev 4 re-run, `vnni_golden` example) — **153 cases × 8 paths** (Q8_0/Q5_0/Q4_K/Q6_K) vs llama.cpp's transcribed kernels; quantizers byte-exact (q8_0 runtime contract + q8_K single contract); worst path error 1.15e-7 of term energy; mutation sensitivity 7276× (q8_0), 5115× (q4_k), 3376× (q6_k); dispatch gates (cpuinfo cross-check, gated repack → None).
2. **Parity suite: 16/16 (rev 3/4) → 24/24 (rev 5)** — tokenizer ids 10/10, top-1 greedy after prefill 4/4 (now exercising the BATCHED path), C-reference dequantization 2/2 (Q5_0, Q6_K; max diff 5e-10), + 8 Phase 6.5 gates: batched-vs-token-at-a-time top-1 equality ×4 and the batchcmp logits self-check ×4 (both models, two prompts).
3. **Workspace tests: 38 passing at the Phase 6.6 final state** (rev 5 quoted 39 — the earlier figure counted one additional example/doc pass; the test files themselves are untouched by the Phase 6.6 diff, and every suite is green, 0 failures). Rev 5 additions: GEMM-vs-matvec differential suites — four types, M sweep, 1t/2t, fused pair, zero cases — and the batched f32 SIMD helper suite; the GEMM tests are the exact-identity link that extends the C-golden chain of §3.4 to M>1.
4. **Multiple repetitions:** 5-rep benchmark campaign with medians and all runs listed; 5-rep same-binary A/B; kernel benches 15 reps; llama-bench 5 internal reps.
5. **No regression hidden by averaging:** every change gated on the parity suite; A/B comparisons through the same binary (`LLAMA_RS_VNNI`, `LLAMA_RS_SLOT`, `LLAMA_RS_SPLIT`). Phase 6.6 re-ran the no-regression gate at the final state (rev-6 note item 4) with 5-rep campaigns per model.
6. Failed experiments are documented in §2.1/§2.2 rather than silently dropped.

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
cargo test --release --workspace          # 38 tests incl. the VNNI (randomized + edge) tests
cargo run --release --example vnni_golden  # rev 3: 72-case C-golden differential sweep + dispatch gates + mutation self-test
bash tests/parity/run_parity.sh          # 24/24 vs llama.cpp
bash scripts/bench_decode.sh 30           # pp+tg matched campaign vs llama-bench (median [P10 P90])
bash scripts/bench_prefill.sh 3           # M sweep {1,8,16,32,64,128,256,512} vs llama-bench (rev 6: finer)
bash scripts/audit.sh 3                    # Phase 6.6: stage tables + DRAM ceiling + perf probe (§9)
MODELS=<model> bash scripts/bench_ctx.sh  # Phase 6.6: decode vs context length (§9.5)
cargo run --release --example perfcheck --manifest-path crates/llama-kernels/Cargo.toml  # CPU-counter probe
cargo run --release --example drambw   --manifest-path crates/llama-kernels/Cargo.toml  # DRAM ceiling
LLAMA_RS_VNNI=0 bash scripts/bench_decode.sh 5   # A/B: AVX2 fallback, same binary
bash scripts/profile_decode.sh            # per-op tables + kernel micro-bench
LLAMA_RS_PROF=1 ./target/release/llama-rs-cli -m <model> --mode bench -p "The capital of France is" -n 48 -t 2
LLAMA_RS_PROF=1 LLAMA_RS_PROF_TSV=1 LLAMA_RS_PROF_PP=1 LLAMA_RS_PREFILL_M=128 \
  ./target/release/llama-rs-cli -m <model> --mode bench -p "<long prompt>" -n 4   # prefill TSV profile
```

## 7. Next steps (re-ranked after Phase 6.5 — superseded by the Phase 6.6 evidence-based ranking in §9.8)

The Phase 6.6 audit re-ranked these with measurements (§9.8): attention-core vectorization is now the P0 (54 % of Qwen prefill, 72 % of long-ctx decode); GEMM register-blocking and per-layer grouped dispatch are P1; the AMX tile backend and AVX2 fallback benchmarking are P2. The list below is kept for the record:

1. **Batched prefill (M>1) — DONE in Phase 6.5** (§8): LFM3 pp 14 % → 62 %, Qwen pp 16 % → **102 %** (30-rep medians). Remaining prefill gap (LFM3 at pp512, Qwen 46–53 % at M≥128) is the GEMM kernel's shuffle-bound ~17 % VNNI efficiency vs llama.cpp's AMX tiles (§8.1).
2. **GEMM kernel restructure** — register-blocked multi-token inner loop (2–4 rows per tile load, pre-broadcast activation layouts to dodge the shuffle port): the measured path from 42 %/53 % at M=128 toward the AMX-tile throughput. The capability groundwork (CPUID/XCR0 tile detection) is validated since Phase 6.3 — a real AMX tile backend (LDTILECFG/TDPBUSD) is the alternative route and is also what llama.cpp actually runs here.
3. **Qwen decode (81 %)** — remaining per-op overheads (glue ~3 ms/token, scratch reuse, W::get name formatting), possibly a graph-style fused executor.
4. **Memory, next step** — the madvise reclaim (§8.3) already cut steady RSS to ~1.05× model and peak to the same level (§4.3); a file-backed lazy loader (mmap the GGUF, repack on first touch) would additionally remove the load-time read of the original bytes — steady-state would go below llama.cpp's own footprint.
5. Hardening: `catch_unwind` job-failure flag in the pool (§5.1 limitation — unchanged; the dispatch mutex added in rev 5 removed the cross-driver race but not the panic-hang).

---

## 8. Phase 6.5 design notes and gates

### 8.1 GEMM kernel structure (and the honest ceiling)

`gemm_band` iterates groups × rows and calls the *same* `group_dot_*` inner compute as the decode path per (group, row) — bit-identity by construction, tiles L1/L2-resident across the row loop. Measured arithmetic efficiency is ~17 % of the theoretical VNNI peak (GEMM ops ≈ 2.6 s of a 3.26 s / 166-token prefill). Root cause (profiled + instruction-counted): the operand-broadcast structure of `dpbusd` needs one `vpermd` per `dpbusd` — the single shuffle port caps the issue rate at ~50 % — compounded by this guest's documented AVX-512 frequency licensing (§2.1 row 6). The remaining prefill gap vs llama.cpp at M ≥ 128 is this kernel-vs-AMX-tile gap, not memory bandwidth (weights stream once per batch at <1 GB/s here). A register-blocked multi-token kernel (2–4 rows per tile load) and/or pre-broadcast activation layouts are the identified future levers.

### 8.2 Numerics of the batched path (stated precisely)

- GEMM columns, rope, norms, biases, swiglu, conv, KV: **bit-identical** to the decode path (verified by per-layer checksum probes; `--mode batchcmp` reports max|Δlogit| ≈ 0.5 / 0.48 on 6-token prompts, top-1 identical, for LFM3 / Qwen).
- The attention core is vectorized f32 (AVX2 dot, SIMD `2^k` exp softmax — the ggml `exp_ps` structure with the `(k+127)<<23` construction — FMA value accumulation): the same accuracy class as ggml's own vectorized kernels. The decode path keeps its scalar f64-stable softmax. Divergence from the decode path is f32-accumulation-order noise (measured seed ~1e-5/layer, compounding through 24 layers); llama.cpp's own decode and prefill kernels differ from each other by the same mechanism.
- Gates: parity **24/24** (top-1 vs llama.cpp with batching on; batched-vs-token top-1 equality; batchcmp MATCH on both models × prompts); `LLAMA_RS_VNNI=0` runs the whole batched path through the per-row AVX2 fallback (top-1 identical — fallback sanity); workspace tests 39 (incl. 3 new GEMM differential suites + batch SIMD helpers).

### 8.3 Memory-reclaim safety argument

`free_tensor_pages` runs only for tensors that were successfully repacked; the packed map is then the only reader path (`pmv/pmv2/pgemm*` fall back to `W::get` only for tensors *not* in the map, which were never freed). Load-time reads (norms, biases, conv kernels, rope freqs, embeddings) all happen before `build_packed`; `token_embd` is excluded from repacking and never freed. Interior pages only (boundary pages shared with neighbours stay resident) — cross-tensor damage is impossible. `LLAMA_RS_MEM_KEEP=1` restores the 2× footprint for A/B. The parity suite (24/24) and batchcmp run identically with the reclaim on — the correctness evidence that no live byte is ever released.

### 8.4 Reproduce (Phase 6.5 additions)

```bash
bash scripts/bench_prefill.sh 3        # M sweep {1,8,32,128,512} vs llama-bench, medians [P10 P90]
bash scripts/bench_decode.sh 30        # 30-rep campaign with median/P10/P90 (§4.1)
./target/release/llama-rs-cli -m <model> --mode mem -p "warm"      # RSS / peak RSS stages
./target/release/llama-rs-cli -m <model> --mode batchcmp -p "..."  # batched-vs-token logits self-check
LLAMA_RS_MEM_KEEP=1 ... --mode mem     # A/B: keep originals (2× footprint)
LLAMA_RS_PREFILL_M=1 ... --mode bench  # A/B: token-at-a-time prefill
```

---

## 9. Phase 6.6 — Decode & Kernel Bottleneck Audit (the evidence section)

Method: `scripts/audit.sh` runs, per model, 3 decode-profile process reps (48-token decode, TSV op tables) and 3 prefill-profile reps (~550-token prompt, M=128), medians per bucket; `scripts/audit_parse.py` aggregates tensor-class rows into the stage tables below. Bandwidth classification uses the measured ceilings (§ rev 6 item 3) and the profiler's per-bucket byte attribution (weight bytes actually streamed per call). Evidence files: `audit/*.tsv` (gitignored, regenerated by the script).

### 9.1 CPU counters and the measurement method (the constraint, stated)

`perf_event_open` returns **EACCES for every hardware and software event** on this guest (`examples/perfcheck`, six attr sizes tried). There are no cycle, instruction or cache-miss counters: all timing is the invariant TSC, all "cache behavior" is effective-bandwidth arithmetic (bytes the profiler attributes / elapsed), and the DRAM ceiling comes from `examples/drambw`. Single-linear-stream ceiling: **13.9–14.1 GB/s**; two streams (the decode configuration): **17.2–27.3 GB/s** across passes — the shared host's interference is visible (pass 3 of a 3-pass run drops to 17.2), and the same drift moves llama-bench ±5 % between sessions. Every ratio below is quoted against numbers measured in the same session.

### 9.2 LFM3 decode (101–104 % of llama.cpp) — at the machine's effective rate

| stage | µs/tok | %wall | MB/tok | stream GB/s |
|---|---:|---:|---:|---:|
| ffn gate+up (fused pair, Q4_K) | 25 972 | 29.7 % | 424.7 | 15.97 |
| ffn_down (Q4_K/Q6_K mix) | 22 908 | 26.2 % | 261.0 | 11.13 |
| output head (tied embd, Q6_K) | 15 530 | 17.7 % | 215.0 | 13.52 |
| conv in/out_proj (Q4_K) | 10 249 | 11.7 % | 122.7 | 11.69 |
| attn projections (q/k/v/o/gate) | 7 290 | 8.3 % | 66.4 | 8.90 |
| everything else (attn core 0.8 %, norms, rope, conv, glue) | ~5 500 | 6.4 % | — | — |
| **total** | **87 560** | 100 % | **1 090 MB** | **12.16 overall** |

llama.cpp on the same workload streams the same 1 090 MB/token at 11.00 tok/s = **12.0 GB/s — the same effective rate**. Both engines sit at ~88 % of the single-stream linear ceiling; the residual is the VNNI unpack compute (dpbusd + nibble/table unpack + per-group scale work) that no engine here can skip — which is *why* the engines are equal and why LFM3 decode holds 101–104 %. **Classification: at the pattern-effective machine limit; parity, not a gap.** (The 2-stream burst ceiling of 27 GB/s is unreachable by a 1090 MB/token working set interleaved with unpack work — the pure-fold probe has no compute in its stream.)

### 9.3 Qwen decode (77–85 %) — where the remaining ~19 % actually goes

| stage | µs/tok | %wall | MB/tok | stream GB/s |
|---|---:|---:|---:|---:|
| ffn gate+up (Q5_0 pair) | 12 704 | 31.8 % | 143.8 | 11.06 |
| output head (tied embd) | 7 429 | 18.6 % | 144.6 | 19.01 |
| ffn_down (Q4_K/Q6_K mix) | 8 192 | 20.5 % | 72.3 | 8.62 |
| attn projections (q/k/v/o) | 9 137 | 22.9 % | 30.8 | **3.29** |
| everything else (attn 1.3 %, rope, norms, glue) | ~2 100 | 5.4 % | — | — |
| **total** | **39 890** | 100 % | **392 MB** | **9.59 overall** |

llama.cpp streams the same 392 MB/token at 27.5–29.4 tok/s = **12.8–13.0 GB/s**. The 19 % engine gap decomposes into exactly two measured terms:

1. **The attention-projection stage: 30.8 MB of small tensors (q/k/v/o per layer, 169 matvecs/token) take 9.14 ms — 3.29 GB/s.** At the stage rates the big tensors achieve (11–19 GB/s) this traffic should cost ~2.2 ms; the extra ~7 ms/token is the per-matvec band-split dispatch and thread hand-off on small work units — llama.cpp runs these inside one fused graph pass. This is a **scheduling** term (the pool itself is clean: kstats show 0 µs enqueue/wait, 0 parks) — the cost is the *granularity* of the dispatch, not the mechanism.
2. **ffn_down streams at 8.62 GB/s** (Q4_K/Q6_K packed kernels) vs the 12.8 GB/s llama.cpp achieves in aggregate — a **kernel** term worth ~1.5 ms/token.

Fixing both (grouped per-layer dispatch + faster ffn_down) would put Qwen decode at 392 MB / (29.4 tok/s ≈ 34 ms) ≈ llama.cpp's own effective rate — i.e. the same "at the machine limit" state LFM3 decode already enjoys. **Classification: mixed scheduling + kernel; not memory.**

### 9.4 Prefill stage splits (M=128, ~550-token prompt)

**LFM3 prefill (42–43 % of llama.cpp at this length):**

| stage | µs/tok | %wall | stream GB/s |
|---|---:|---:|---:|
| GEMM (gate+up 30.1 % + ffn_down 19.9 % + conv 8.8 % + attn 4.5 %) | 14 678 | **63.3 %** | **0.65 in-GEMM** |
| attention core (batch, SWA+full, f32) | 5 786 | 24.9 % | — |
| norms/residual/swiglu | 1 246 | 5.4 % | — |
| activation quantize | 599 | 2.6 % | 2.23 |
| conv window+state | 290 | 1.3 % | — |
| rope/qknorm, head, glue, residual | ~600 | 2.6 % | — |
| **total** | **23 210** | 100 % | (11 MB/token streamed) |

**Qwen prefill (53 % at this length):** attention core (batch) **54.3 %** (6 444 µs/tok), GEMMs 32.4 % at 0.90 GB/s in-GEMM, quantize 4.8 %, elementwise 5.7 %, wall 11.87 ms/token.

**Classification for both: instruction-throughput bound, not memory.** The GEMMs stream 0.65–0.90 GB/s — 5 % of the ceiling — because the operand-broadcast structure of `dpbusd` needs a `vpermd` per instruction (the single shuffle port caps the issue rate at ~50 %, §8.1) compounded by the documented AVX-512 frequency licensing. The attention term is O(M·L) f32 compute; at M=128 with L≈550 it is already 54 % of Qwen prefill — llama.cpp's flash-style fused kernels and AMX tiles are what the remaining gap is made of. (Note: §8.1's "GEMM ≈ 80 % of prefill" was the old shared-timer attribution; the corrected per-tensor split is 63 %.)

### 9.5 Context sweep — the decode collapse the short-prompt campaigns never see

llama.rs tg (32 tokens) after prefilling the matched prompt length; llama-bench same session, 1 rep per point (trend evidence — the campaign medians are §4.1's):

| ctx (tok) | LFM3 rs / cpp / ratio | Qwen rs / cpp / ratio |
|---:|---|---|
| ~110 | 11.41 / 10.91 / **105 %** | 21.24 / 27.36 / 78 % |
| ~275 | 10.75 / 10.99 / 98 % | 21.39 / 26.88 / 80 % |
| ~1035 | 9.16 / 10.89 / 84 % | 17.57 / 28.47 / 62 % |
| ~2070 | 7.54 / 10.36 / 73 % | 13.88 / 29.38 / 47 % |
| ~4080 | 5.49 / 10.94 / **50 %** | 7.89 / 28.19 / **28 %** |

**llama.cpp's decode is context-flat; ours collapses.** Profiled cause (one run each at ctx ≈ 4k): `attn_core` balloons to **72.7 % of Qwen decode wall (91.6 ms/token)** and **64.0 % of LFM3's (120 ms/token)**. The per-key loop (`scores.push(dot(q,k)); softmax; axpy per key`) is O(ctx) with a per-key Vec push and scalar accumulation — a kernel problem, not memory: at ctx 4k the KV read traffic is ~2–3 MB/token (trivial); it is pure f32 loop inefficiency versus llama.cpp's vectorized flash-style kernels. **Classification: kernel (attention core); this is the largest single measured gap in the whole audit and it was invisible to the headline campaigns.**

### 9.6 Batch-size sweep (finer, 8 points, 2 reps, ~550-token prompts)

LFM3: M=1 11.01 (11 %) · 8 30.41 (30 %) · 16 35.78 (36 %) · 32 39.98 (40 %) · 64 42.13 (42 %) · 128 42.38 (42 %) · **256 42.61 (43 %)** · 512 40.05 (40 %) — llama.cpp 100.08. Qwen: 21.11 (13 %) · 59.09 (37 %) · 70.72 (44 %) · 80.03 (50 %) · **64 84.62 (53 %) · 128 84.70 (53 %)** · 256 82.27 (51 %) · 512 73.22 (46 %) — llama.cpp 160.62. The sweet spots (M=256 LFM3 / M=128 Qwen) and the M=512 attention-overtake are confirmed at finer resolution; nothing new changes the §4.2 reading.

### 9.7 Root-cause classification matrix (the Phase 6.6 deliverable)

| Workload (short-ctx campaigns) | ratio vs llama.cpp | binding constraint — measured evidence | classification |
|---|---|---|---|
| LFM3 decode | 101–104 % | 12.16 GB/s effective = llama.cpp's 12.0; both ≈ 88 % of single-stream ceiling | **at the pattern-effective machine limit — parity** |
| Qwen decode | 77–85 % | attn-proj stage 3.29 GB/s (7 ms/token over ideal) + ffn_down 8.6 GB/s | **scheduling (dispatch granularity) + kernel (ffn_down rates); not memory** |
| LFM3 prefill @550 | 42–43 % | GEMM 63 % of wall at 0.65 GB/s ≪ ceiling | **instruction throughput (shuffle port + AVX-512 licensing)** |
| Qwen prefill @540 | 53 % | attention core 54 % of wall (O(M·L) f32) | **kernel (batched attention core)** |
| Memory (all workloads) | — | decode 9.6–12.2 GB/s < 14 ceiling; prefill 0.4–0.9 GB/s | **never the binding constraint** |
| Scheduling (decode) | — | pool enqueue/wait ≈ 0 µs, 0 parks; cost only via per-matvec band-split × 169/token | **clean at macro scale; granularity is the term** |
| Long-context decode (§9.5) | 28–50 % at ctx 4k | attn_core 72.7 %/64.0 % of wall; llama.cpp flat | **kernel (per-key f32 attention loop)** |

### 9.8 Re-ranked next steps (evidence-weighted, per the review's priority table)

1. **P0 — batched + decode attention core vectorization** (new evidence, largest measured term): flash-style row blocking, SIMD dot/softmax/axpy over key runs, f32 first. §9.4 (54 % of Qwen prefill) and §9.5 (72 % of long-ctx decode) are both this one kernel family; nothing else in the audit is this large. An f16 KV cache halves the traffic on top.
2. **P1 — GEMM register-blocking / shuffle-dodging layouts** (2–4 rows per tile load, pre-broadcast activation layouts): the measured path from 42 %/53 % toward the AMX-tile throughput; §8.1's analysis stands.
3. **P1 — per-layer grouped matvec dispatch** (q/k/v/o in one band-split per layer): removes the Qwen attn-proj 7 ms/token dispatch term; pairs with `ffn_down` kernel work to close Qwen decode to parity.
4. **P2 — a real AMX tile backend** (LDTILECFG/TDPBUSD; CPUID/XCR0 tile detection validated since Phase 6.3): llama.cpp's actual M>1 path on this machine.
5. **P2 — AVX2 fallback benchmarking** (independent numbers, not just parity) and the file-backed lazy loader (§7 item 4, memory).
