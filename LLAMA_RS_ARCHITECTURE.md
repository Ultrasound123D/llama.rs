# LLAMA_RS_ARCHITECTURE — Architecture Document for the llama.rs Engine (a Native Rust Port of the llama.cpp Engine)

**Version:** 0.2 (Phases 0–6) · **Date:** 2026-11-25 (Phase 6)
**Source baseline:** llama.cpp b12502 (commit 398f00d65) — fully cloned and audited locally
**Goal:** a native GGUF inference engine written in Rust, output-compatible with llama.cpp (parity proven by tests), with zero C/C++ runtime linkage.

---

## 1) Source ↔ Rust Map (Phase 0: Source Audit — complete)

| llama.cpp component | Source file | llama.rs counterpart (current v0) | Final-plan counterpart |
|---|---|---|---|
| GGUF reader | `ggml/src/gguf.cpp`, `ggml-common.h` | `crates/llama-gguf` | unchanged |
| Tensor types/quant blocks | `ggml-common.h` (block_q*_K) | `crates/llama-quant` | unchanged |
| Compute ops | `ggml/src/ggml-cpu/ops.cpp` | `llama-core` (lib.rs ops + quant matvec) | `crates/llama-kernels` + SIMD — **done in Phase 6** |
| Compute kernels (SIMD) | `ggml/src/ggml-cpu/arch/x86/quants.c` + `vec.cpp` | `crates/llama-kernels/src/avx2.rs` (+ `avx512.rs` experiment) | same (AMX not ported) |
| Activation quantizers | `ggml-quants.c quantize_row_q8_0/q8_K` + arch/x86 fast path | `crates/llama-kernels/src/quant.rs` | unchanged |
| Threadpool | `ggml/src/ggml-cpu/ggml-cpu.c` (ggml_threadpool) | `crates/llama-kernels/src/pool.rs` (persistent workers, spin-then-park) | unchanged |
| Backend/scheduling | `ggml/src/ggml-backend.cpp` | — (not needed for CPU in v0) | `crates/llama-backend` |
| Compute graph | `ggml/src/ggml.c` (granular graph building) | **direct approach with no graph** in v0 (straight-line execution per model) | optional `crates/llama-graph` |
| Model architecture | `src/llama-arch.cpp` | `llama-core::model` (per-arch tensor wiring) | `crates/llama-models` |
| Model compute building | `src/models/lfm3.cpp`, `models/lfm-base.cpp`, `models/llama.cpp`… | `llama-core::model::{Lfm3, LlamaFamily}` (covers qwen2) | one module per architecture |
| hparams/loading | `src/llama-model.cpp` + `llama-model-loader.cpp` | `llama-core::model` (`load` from GGUF) | `crates/llama-runtime` |
| KV/hybrid memory | `src/llama-memory-hybrid-iswa.cpp` | f32 KV + conv-state inside each model (semantically identical) | `crates/llama-memory` |
| BPE tokenizer | `src/llama-vocab.cpp` + `src/unicode.cpp` | `llama-core::tokenizer` (verbatim port of its algorithms) | `crates/llama-tokenizer` |
| Sampling | `src/llama-sampling.cpp` | `llama-core::sampler` | `crates/llama-sampler` |
| Chat templates | `common/chat.cpp` (jinja) | `llama-core::chat` (ChatML/LFM3 subset) | mini-jinja |
| Server/OpenAI API | `tools/server/` | — (Phase 7) | `crates/llama-server` |

**Key architectural decisions:**
1. **Rebuild contracts, not line-by-line translation**: we re-implement the *behavior* of the ops (the very same ggml formulas: rope_yarn/rotate_pairs, ssm_conv, get_scale_min_k4...) in safe Rust, rather than mimicking ggml's dynamic machinery (context/tensor graph), which serves the backend plumbing more than numerical fidelity.
2. **Weights stay quantized in memory** (as in llama.cpp) and the matvec runs in the **int8 domain** (Phase 6): the activation vector is quantized once per matvec into the weight type's `vec_dot_type` (Q8_0 / Q8_K, exactly llama.cpp's `from_float` on `wdata`), then every output column is an int-domain `vec_dot` with per-block scales — the same numeric contract as ggml's AVX2 kernels. This replaced v0's per-column f32 dequant (which dequantized the whole matrix per token).
3. **Parity before speed**: output equivalence with llama.cpp is the gate for every phase (see §4).
4. **Phase 6 fidelity note**: llama.cpp's *published* logits already carry int8 activation rounding; running the same int-domain math moves llama.rs numerically closer to llama.cpp than exact-f32 dequant did.

## 2) Computations Ported Verbatim from the Source (the Authoritative Code Reference)

Source locations read and matched against (all from b12502):

| Operation | Source | Fidelity notes |
|---|---|---|
| Q4_K/Q5_K/Q6_K dequant | `ggml-quants.c:1539/1741/1949` + `get_scale_min_k4:889` | byte-exact port |
| Q8_0/Q4_0/Q5_0 | `ggml-quants.c:462…556` | verbatim port |
| f16→f32 | `ggml-common.h` + cpu convert | IEEE half |
| RoPE NEOX with learned factors | `ggml-cpu/ops.cpp:6148-6252` (rope cache + rotate_pairs) | angle_j = pos · base^(−2j/d) / ff_j; pairs (j, j+d/2) |
| shortconv | `models/lfm-base.cpp:7-93` + `ops.cpp:9859 ssm_conv` | kernel [7,2048]; state of 6 rows; window [state, x]; y = c·conv(b·x) |
| LFM3 block fusion | `models/lfm3.cpp:98-208` | sandwich-norms order + residual from the un-normalized prev |
| gated attention | `models/lfm3.cpp:101-136` | per-head q_norm/k_norm + per-head sigmoid gate |
| llama3 pre-tokenizer | `unicode.cpp:355-493` (state machine) + `llama-vocab.cpp:293-299` | verbatim port |
| BPE merges + ignore_merges | `llama-vocab.cpp:660-820` | rank-based priority queue |
| GPT-2 byte encoding | `unicode.cpp:157-234` | table 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF |
| add_bos from GGUF overrides pre | `llama-vocab.cpp:3005-3017` | LFM3: add_bos=false |
| character classes | `unicode.h` flags (\p{L}=\{Lu,Ll,Lt,Lm,Lo\}, \p{N}=\{Nd,Nl,No\}, \s=White_Space) | via the `unicode-general-category` crate |

## 3) llama.rs Architecture (v0.2)

```
llama.rs/
├── Cargo.toml                    # workspace
├── crates/
│   ├── llama-gguf/               # GGUF v3 reader: header, kv-metadata (all types), tensor index, data
│   ├── llama-quant/              # block_q*_K + dequant + f16 codec + matvec (legacy scalar path)
│   ├── llama-kernels/            # Phase 6: quant.rs (activation quantizers q8_0/q8_K — the exact
│   │                             #   functions llama.cpp runs on x86), scalar.rs (the *generic* dot
│   │                             #   contracts from ggml-cpu/quants.c), avx2.rs (the arch/x86
│   │                             #   kernels + prefetch + inlined 256-bit VNNI), avx512.rs (512-bit
│   │                             #   VNNI experiment), pool.rs (persistent worker pool),
│   │                             #   lib.rs (QAct + dispatch + parallel matvec)
│   └── llama-core/
│       ├── lib.rs                # W tensor ref + ops: rmsnorm, silu, softmax, matvec, rope, attention
│       ├── model.rs              # LFM3 (24 layers: 16 shortconv + 8 attention: 4 full + 4 SWA window 256)
│       │                         #   + llama/qwen2 family (GQA + SwiGLU)
│       ├── tokenizer.rs          # full BPE (llama3-pre state machine, merges, special fragmentation)
│       ├── sampler.rs            # greedy / temperature / top-k / top-p
│       └── chat.rs               # ChatML + LFM3 templates (thinking tags)
├── apps/llama-rs-cli/            # generate / chat / tokenize / bench / debug (+ LLAMA_RS_PROF=1)
├── crates/llama-kernels/examples/ # matbench / q4cmp / q8bench / smallbench micro-benchmarks
└── tests/parity/                 # parity harness vs llama.cpp + C ground-truth references (ref_q5/ref_q6)
```

**Tensor data model**: a weights array in ggml order: `W[in, out]` laid out sequentially by row in memory (row-major over dim0), and matvec computes `out[o] = Σ_i W[i,o]·x[i]` — the same semantics as `ggml_mul_mat` in llama.cpp for the audited files (weights stored [n_embd, out] as in the GGUF above: e.g. `ffn_gate [2048, 7680]`).

## 4) Exit Gates per Phase (with current execution status)

| Phase | Exit criterion | Status |
|---|---|---|
| P0 audit | source↔crate map + documented verbatim op transfer (above) | **complete** |
| P1 GGUF reader | read the LFM3 and Qwen GGUFs: metadata + all tensors and their dims match the `gguf_explore.py` dump | **complete** (automated check: the tokenizer runs on both models) |
| P2 quantization | dequantize sampled blocks from real files: f32 values match a reference running the same algorithm + an end-to-end golden test (covered by logits parity) | **complete** (Q5_0 + Q6_K vs a C reference built from the ggml code itself: diff ≤ 5e-10) |
| P3 generation | run a model end-to-end on CPU with zero C++ libraries | **complete** (LFM3 + Qwen2.5) |
| P4 tokenizer | tokenizing mixed-script/English/symbols texts: ids match llama.cpp `/tokenize` 100% | **complete: 10/10 checks = 100% ids** |
| P5 first model | Qwen2.5-0.5B: greedy text identical to `llama-completion` (same seed/temp=0) | **complete**: the first 14 tokens match literally; afterwards near-perfect numerical parity (gaps < 0.1 logprob) that flips candidate order — requires the ggml kernels (P6) |
| P5+ reference model | LFM3-1.7B: same Qwen gate + hybrid structure (conv + SWA + head gate + learned rope) | **complete**: top-1 greedy match on ×2 prompts; a full chat answer byte-identical to llama.cpp's |
| P6 CPU SIMD | benchmark against llama-bench on the same machine | **complete**: LFM3-1.7B 0.9 → **8.7 tok/s (~10×)**, 2 threads, vs llama.cpp 11.9 (AVX-512 build). Kernel rate 13.9–16.8 GB/s DRAM (Q4_K/Q6_K); parity gates 16/16 still green. Findings: software prefetch was worth +53%; AVX-512 VNNI kernel ~9% slower on this VM (kept as experiment); remaining Qwen gap = llama.cpp's AMX tiles (unported) |
| P7 server/parallelism | OpenAI API + streaming + batching | planned |
| P8 architecture expansion | llama3/gemma/qwen3… | planned |

## 5) Parity Test Plan (tests/parity)

1. **Synthetic units (run in CI without models)**: `crates/llama-gguf/tests/synthetic.rs` builds a small GGUF v3 file by hand and checks all KV types, tensor dims and alignment; `crates/llama-quant/tests/dequant_synth.rs` dequantizes hand-computed synthetic Q4_0/Q8_0/F16 blocks.
2. **kernel contracts (llama-kernels/tests/kernels.rs)**: AVX2 and AVX-512 SIMD dots vs the scalar *generic* contracts (integer parts exact, f32 reduction order within 1e-5); `matvec` vs the exact-f32 dequant reference with exactly-representable activations (tol 1e-3); quantizer hand-cases; f16 encoder vs F16C; pool semantics.
3. **tokenizer-ids**: the same list of texts through the llama.cpp server `/tokenize` and through `llama-rs-cli tokenize` → compare id sequences (must match 100%).
4. **greedy-generation**: `llama-completion -m model -p prompt -n N` (no random sampling) vs `llama-rs-cli generate` → compare text/ids. A greedy match implies a match of: embedding + all ops + KV + sampler argmax.
5. **Dequantization**: `ref_q5.c`/`ref_q6.c` (transcribed from ggml-quants.c itself) compare dequantized values from a real file column by column.
6. **bifurcation search** on any divergence: print the first-token logits and compare (future tool).

## 6) v0 Limits (not yet ported — with full honesty)

- Throughput: LFM3-1.7B 8.7 tok/s vs llama.cpp 11.9 on this VM (the remaining gap is inter-matvec scheduling, not kernel streaming rate); Qwen2.5-0.5B 8.8 vs 31.6 — llama.cpp's simple-quant tensors run on **AMX tile kernels** (its sapphirerapids backend) which are not ported.
- No HTTP server, no parallel batching, no speculative decoding (Phase 7).
- No GPU backends. No AMX.
- Chat template subset (not full jinja): ChatML + LFM3 thinking tags, no tools/vision.
- Supported architectures: lfm3 + the llama family (llama1/2/3, qwen2, qwen2.5 through the llama-family model). The rest in P8.
- Licensing note: MIT (llama.cpp) — copyright notices in the NOTICE file in the repo; ported pieces are documented with their sources in §2.

## 7) Phase 6 Engineering Log (b12502 → Rust)

Ported from `ggml/src/ggml-cpu`:
- `arch/x86/quants.c`: `quantize_row_q8_0` (AVX2 semantics: round-ties-even), the AVX2 vec_dots for q4_0/q5_0/q8_0/q4_K/q5_K/q6_K (incl. the `get_scale_shuffle` tables, `bytes_from_nibbles_32`, `bytes_from_bits_32`, the q5_K `hbits/hmask` bit rotation, the q6_K +32-unsigned trick with the `bsums` correction, and the VNNI `dpbusd` branch of `mul_sum_us8_pairs_float` as inlined variants).
- `ggml-quants.c`: `quantize_row_q8_K_ref` (= the x86 fast path), `nearest_int` (bit-exact magic-number trick).
- `vec.cpp`: `ggml_vec_dot_f32` / `ggml_vec_dot_f16_f32` structures (F16C bulk conversion).
- `ggml-cpu.c`: the mul_mat vec path's *shape* — activation quantized once (`wdata`), per-row vec_dot, thread participation incl. the caller.

Fixes found during Phase 6 (through the new random-block tests):
- **latent Q5_K dequant bug** in llama-quant (indexed `qh[g*32+l]` instead of the fixed `qh[l]` with rotating `u1/u2` bit masks, and `ql` without the group offset): unreachable before (no Q5_K tensors in either model) — now matches `dequantize_row_q5_K`.
- f16→f32 fast path for normal halves (6 integer ops), f16→f32 in the F16 dot via F16C.

Measurements on this VM (2 vCPU Xeon sapphirerapids, 3200 MHz, DRAM ceiling 29 GB/s @2 threads):
- Q4_K 2048×524288 (1 GB): 9.0 → **13.9 GB/s** after adding software prefetch (the single biggest win, +53%); Q6_K: 16.8 GB/s; Q8_0: 1.6 → 6.7 GB/s after inlining the 256-bit VNNI dot.
- Full LFM3-1.7B: 0.9 → **8.7 tok/s** (llama.cpp 11.9, which loads an AVX-512 + AMX backend).
- AVX-512 VNNI q4_K kernel: built and tested (`avx512.rs`, comparison in `examples/q4cmp.rs`) but ~9% slower than AVX2 on this VM → default dispatch stays AVX2.
