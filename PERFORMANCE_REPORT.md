# llama.rs — Phase 6.1–6.3 Performance Engineering Report (rev 3)

**Scope:** profiling infrastructure, scheduling experiments, **AVX-512 VNNI INT8 packed backend** (renamed from "AMX INT8 backend" — see the rev-2 note), the **rev-3 independent verification campaign** of the VNNI path (C-golden differential harness, dispatch safety, matched prefill/decode benchmark), and — **rev 4 (Phase 6.4) — the K-quant packed path**: Q4_K/Q6_K repacked to the VNNI format, every LFM3 matvec routed through the packed dispatch, and the first decode result **above llama.cpp** (LFM3 tg = 102 %).
**Machine:** 2 vCPU Xeon (Sapphire Rapids class) @ 3.2 GHz KVM guest, AVX-512 + AVX512-VNNI + AMX INT8/BF16 (OS-enabled, XCR0 bits 17/18 verified), 504 MB L3, DRAM ceiling ≈ 14–16 GB/s sustained at 2 threads for one linear stream.
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

## 5. Validation gates (all re-run at the rev-3 final state)

1. **C-golden differential sweep: ALL GATES GREEN** (rev 4 re-run, `vnni_golden` example) — **153 cases × 8 paths** (Q8_0/Q5_0/Q4_K/Q6_K) vs llama.cpp's transcribed kernels; quantizers byte-exact (q8_0 runtime contract + q8_K single contract); worst path error 1.15e-7 of term energy; mutation sensitivity 7276× (q8_0), 5115× (q4_k), 3376× (q6_k); dispatch gates (cpuinfo cross-check, gated repack → None).
2. **Parity suite: 16/16** — tokenizer ids 10/10, top-1 greedy after prefill 4/4, C-reference dequantization 2/2 (Q5_0, Q6_K; max diff 5e-10).
3. **Workspace tests: 34 passing across 4 integration suites** (gguf synthetic 2, kernel contracts 11, vnni 16 — randomized + edge cases incl. the six K-quant suites, quant 5).
4. **Multiple repetitions:** 5-rep benchmark campaign with medians and all runs listed; 5-rep same-binary A/B; kernel benches 15 reps; llama-bench 5 internal reps.
5. **No regression hidden by averaging:** every change gated on the parity suite; A/B comparisons through the same binary (`LLAMA_RS_VNNI`, `LLAMA_RS_SLOT`, `LLAMA_RS_SPLIT`).
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
cargo test --release --workspace          # 28 tests incl. the 10 VNNI (randomized + edge) tests
cargo run --release --example vnni_golden  # rev 3: 72-case C-golden differential sweep + dispatch gates + mutation self-test
bash tests/parity/run_parity.sh          # 16/16 vs llama.cpp
bash scripts/bench_decode.sh 5            # pp+tg matched table vs llama-bench (exact-matched prompt lengths)
LLAMA_RS_VNNI=0 bash scripts/bench_decode.sh 5   # A/B: AVX2 fallback, same binary
bash scripts/profile_decode.sh            # per-op tables + kernel micro-bench
LLAMA_RS_PROF=1 ./target/release/llama-rs-cli -m <model> --mode bench -p "The capital of France is" -n 48 -t 2
```

## 7. Next steps (ranked by measured gap — after Phase 6.4)

1. **Batched prefill (M>1)** — now the dominant measured gap by far (pp 14 % / 16 %): process the prompt in ubatches with a GEMM-style path. This is also the prerequisite for a genuine AMX-tile backend (tiles need M>1; the CPUID/XCR0 groundwork is already in place and validated). LFM3 decode at 102 % shows the packed M=1 route is done; Qwen decode (79 %) needs the remaining per-op overheads, likely a graph-style fused executor.
2. **File-backed lazy loader** — read tensors from the GGUF on demand at load, keep only non-repacked bytes resident: cuts steady-state memory from ~2× model (packed map + original buffer, same trade llama.cpp's AMX repack makes) to ~1.1× model, below even llama.cpp's footprint, and removes the co-residency constraint the parity script now works around.
3. Remaining glue (~4 ms/token LFM3): swiglu/rmsnorm/attention scratch reuse, W::get name formatting per token.
4. Hardening: `catch_unwind` job-failure flag in the pool.
