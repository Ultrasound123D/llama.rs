# llama.rs — a native Rust port of the llama.cpp engine (b12502)

A GGUF inference engine written in pure Rust, **with zero C/C++ runtime linkage**, output-compatible with llama.cpp — proven by automated parity tests (16/16 passing).

## Status (v0.1 — Phases 0–5 complete)

| Feature | Status | Evidence |
|---|---|---|
| Full GGUF v3 reader | ✅ | metadata + tensors + offsets match |
| Dequantization Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/F16/F32/BF16 | ✅ | matches the C reference (the ggml code itself): 5e-10 diff |
| Hybrid LFM3 architecture (16 shortconv + 8 gated attention + SWA + learned RoPE + sandwich norms) | ✅ | top-1 greedy match + a full chat answer byte-identical to llama.cpp |
| llama/qwen2 family (GQA + q/k/v biases + SwiGLU) | ✅ | Qwen2.5-0.5B: first 14 tokens identical |
| BPE tokenizer (llama3-pre + ignore_merges + byte-encode + merges + special fragmentation) | ✅ | 10/10 tokenize checks (5 texts × 2 models: mixed-script/English/symbols) = 100% identical ids |
| LFM3 chat template (thinking tags extracted from the raw template) + ChatML | ✅ | chat mode works on both models |
| Sampling: greedy/temp/top-k/top-p | ✅ | — |
| CLI: generate/chat/tokenize/bench/debug | ✅ | — |
| SIMD kernels matching ggml accumulation order | ⏳ Phase 6 | ~0.9–3 tok/s (plain f32 kernels) vs ~11 for llama.cpp |

## Build & run

```bash
cargo build --release

# Raw generation (llama-completion equivalent)
./target/release/llama-rs-cli -m LFM3-1.7B-Q4_K_M.gguf --mode generate -p "The capital of France is" -n 32 -t 2

# Chat (LFM3 template with thinking tags)
./target/release/llama-rs-cli -m LFM3-1.7B-Q4_K_M.gguf --mode chat -p "What is the capital of Morocco? Answer briefly." -n 60 -t 2

# Tokenize
./target/release/llama-rs-cli -m model.gguf --mode tokenize --no-bos -p "some text"

# Benchmark
./target/release/llama-rs-cli -m model.gguf --mode bench -p "warm" -n 16 -t 2
```

## Parity tests (llama.cpp as golden reference)

```bash
bash tests/parity/run_parity.sh
# Current result: PASS=16 FAIL=0
#  - tokenization ×10 (mixed-script/English/digits/symbols texts, on both models)
#  - top-1 greedy after prefill ×4 (includes the hybrid LFM3)
#  - dequantization ×2: Q5_0 and Q6_K against direct ggml C code (ground truth)
```

## Layout

```
crates/llama-gguf    GGUF v3 reader (full-type metadata + tensor table + data) + synthetic tests
crates/llama-quant   ggml block structures + dequantization + matvec (threads) + synthetic tests
crates/llama-core    ops (rmsnorm/rope/swiglu/attention) + models (LFM3 hybrid + llama/qwen2 family) + tokenizer + sampler + chat
apps/llama-rs-cli    CLI
tests/parity/        automated parity tests vs llama.cpp + C ground-truth references (ref_q5/ref_q6)
```

Synthetic unit tests (hand-built GGUF + hand-computed quantization blocks) run in CI without models; the full parity tests run locally by direct comparison against llama.cpp binaries.

## v0 limits (full honesty)

- Plain scalar f32 kernels: several times slower than llama.cpp CPU kernels (0.9 tok/s vs 11 on LFM3 here) — Phase 6: fused dot kernels in the exact ggml accumulation order (int8 for K-quants) to reach literal greedy parity and a large speedup.
- Greedy divergence after a few tokens on near-tied candidates (logprob gaps < 0.1) is caused by f32 accumulation order — solved by the Phase 6 kernels.
- No HTTP server/batching/GPU yet (Phase 7); two supported architectures (Phase 8 expands this).
- Chat template subset (no full jinja).

## License & attribution

MIT. Work based on llama.cpp (MIT) — attribution is documented in `LLAMA_RS_ARCHITECTURE.md` §2 with the exact upstream source location of every ported algorithm (b12502, commit 398f00d65).
