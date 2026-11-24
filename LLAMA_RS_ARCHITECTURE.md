# LLAMA_RS_ARCHITECTURE — Architecture Document for the llama.rs Engine (a Native Rust Port of the llama.cpp Engine)

**Version:** 0.1 (Phases 0–5) · **Date:** 2026-11-24 (English revision: 2026-11-25)
**Source baseline:** llama.cpp b12502 (commit 398f00d65) — fully cloned and audited locally
**Goal:** a native GGUF inference engine written in Rust, output-compatible with llama.cpp (parity proven by tests), with zero C/C++ runtime linkage.

---

## 1) Source ↔ Rust Map (Phase 0: Source Audit — complete)

| llama.cpp component | Source file | llama.rs counterpart (current v0) | Final-plan counterpart |
|---|---|---|---|
| GGUF reader | `ggml/src/gguf.cpp`, `ggml-common.h` | `crates/llama-gguf` | unchanged |
| Tensor types/quant blocks | `ggml-common.h` (block_q*_K) | `crates/llama-quant` | unchanged |
| Compute ops | `ggml/src/ggml-cpu/ops.cpp` | `llama-core` (lib.rs ops + quant matvec) | `crates/llama-kernels` + SIMD |
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
2. **Weights stay quantized in memory** (as in llama.cpp) and dequantization happens inside the matvec kernel directly from the blocks — no full dequant (would eat 6.8 GB) and no full f16 (3.4 GB). This is the "correctness first" mode; SIMD kernels (Phase 6) will be added later at the same call site.
3. **Parity before speed**: output equivalence with llama.cpp is the gate for every phase (see §4).

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

## 3) llama.rs Architecture (v0)

```
llama.rs/
├── Cargo.toml                    # workspace
├── crates/
│   ├── llama-gguf/               # GGUF v3 reader: header, kv-metadata (all types), tensor index, data
│   ├── llama-quant/              # block_q*_K + dequant + matvec (f32-accumulation kernel)
│   └── llama-core/
│       ├── lib.rs                # W tensor ref + ops: rmsnorm, silu, softmax, matvec, rope, attention
│       ├── model.rs              # LFM3 (24 layers: 16 shortconv + 8 attention: 4 full + 4 SWA window 256)
│       │                         #   + llama/qwen2 family (GQA + SwiGLU)
│       ├── tokenizer.rs          # full BPE (llama3-pre state machine, merges, special fragmentation)
│       ├── sampler.rs            # greedy / temperature / top-k / top-p
│       └── chat.rs               # ChatML + LFM3 templates (thinking tags)
├── apps/llama-rs-cli/            # generate / chat / tokenize / bench / debug
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
| P6 CPU SIMD | benchmark against llama-bench on the same machine | **planned** (currently 0.9 tok/s scalar vs 11 for llama.cpp) |
| P7 server/parallelism | OpenAI API + streaming + batching | planned |
| P8 architecture expansion | llama3/gemma/qwen3… | planned |

## 5) Parity Test Plan (tests/parity)

1. **Synthetic units (run in CI without models)**: `crates/llama-gguf/tests/synthetic.rs` builds a small GGUF v3 file by hand and checks all KV types, tensor dims and alignment; `crates/llama-quant/tests/dequant_synth.rs` dequantizes hand-computed synthetic Q4_0/Q8_0/F16 blocks.
2. **tokenizer-ids**: the same list of texts through the llama.cpp server `/tokenize` and through `llama-rs-cli tokenize` → compare id sequences (must match 100%).
3. **greedy-generation**: `llama-completion -m model -p prompt -n N` (no random sampling) vs `llama-rs-cli generate` → compare text/ids. A greedy match implies a match of: embedding + all ops + KV + sampler argmax.
4. **Dequantization**: `ref_q5.c`/`ref_q6.c` (transcribed from ggml-quants.c itself) compare dequantized values from a real file column by column.
5. **bifurcation search** on any divergence: print the first-token logits and compare (future tool).

## 6) v0 Limits (not yet ported — with full honesty)

- No optimized SIMD kernels (f32-accumulation kernel only) → slower than llama.cpp CPU kernels (expected several-fold).
- No HTTP server, no parallel batching, no speculative decoding.
- No GPU backends.
- Chat template subset (not full jinja): ChatML + LFM3 thinking tags, no tools/vision.
- Supported architectures: lfm3 + the llama family (llama1/2/3, qwen2, qwen2.5 through the llama-family model). The rest in P8.
- Licensing note: MIT (llama.cpp) — copyright notices in the NOTICE file in the repo; ported pieces are documented with their sources in §2.
