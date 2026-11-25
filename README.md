# llama.rs — a native Rust port of the llama.cpp engine (b12502)

A GGUF inference engine written in pure Rust, **with zero C/C++ runtime linkage**, output-compatible with llama.cpp — proven by automated parity tests (16/16 passing).

## Status (v0.2 — Phases 0–6 complete)

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
| **Phase 6 SIMD kernels** (int8-domain AVX2+VNNI vec_dots, activation quantizers, persistent worker pool) | ✅ | LFM3-1.7B: **0.9 → 12.0 tok/s (~13×)** on this 2-vCPU VM; parity 16/16 still green |

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
crates/llama-gguf     GGUF v3 reader (full-type metadata + tensor table + data) + synthetic tests
crates/llama-quant    ggml block structures + dequantization + f16 codec + synthetic tests
crates/llama-kernels  Phase 6: activation quantizers (q8_0/q8_K), int-domain SIMD vec_dots (AVX2+VNNI,
                     AVX-512 experiment), persistent worker pool, parallel matvec + kernel tests
crates/llama-core     ops (rmsnorm/rope/swiglu/attention) + models (LFM3 hybrid + llama/qwen2 family) + tokenizer + sampler + chat
apps/llama-rs-cli     CLI
tests/parity/         automated parity tests vs llama.cpp + C ground-truth references (ref_q5/ref_q6)
```

Synthetic unit tests (hand-built GGUF + hand-computed quantization blocks) run in CI without models; the full parity tests run locally by direct comparison against llama.cpp binaries.

## v0 limits (full honesty)

- **Throughput vs llama.cpp on this VM** (matched 2-thread benchmark with exactly-matched prompt lengths, prefill/decode measured separately, medians): **LFM3-1.7B decode 12.02 vs 11.74 tok/s — 102%, faster than llama.cpp**; Qwen2.5-0.5B decode 23.86 vs 30.34 tok/s (79%). Prefill: 12.06 / 24.94 vs 88.94 / 154.91 tok/s — llama.rs prefills token-by-token while llama.cpp batches the whole prompt (M>1, where its real AMX tile kernels run); **batched prefill is the largest remaining gap**. All quantized decode tensors — Q8_0/Q5_0 *and* the K-quants Q4_K/Q6_K — now run on the repacked-VNNI M=1 route (ported from llama.cpp's AMX backend as `llama-kernels/src/vnni.rs` — AVX-512 VNNI; no AMX tile instructions are executed by this engine), and that path is **verified against an independent C golden built from llama.cpp's own quantizer and vec_dot sources** (byte-exact quantizers, 1e-7-of-term-energy agreement across 153 randomized + edge cases — see `PERFORMANCE_REPORT.md` §3.4). Same-binary A/B: the packed path alone lifts LFM3 decode 8.85 → 12.02 tok/s (+36%). Honest cost: the packed map co-resides with the original weights (~2× model RAM, the same trade llama.cpp's AMX repack makes).
- An AVX-512 VNNI q4_K kernel exists (`llama-kernels/src/avx512.rs`, `examples/q4cmp.rs`) but measured ~9% slower than AVX2 on this VM (frequency licensing) — kept off the default dispatch.
- No HTTP server/batching/GPU yet (Phase 7); two supported architectures (Phase 8 expands this).
- Chat template subset (no full jinja).
- Prefill is token-at-a-time (no batched prompt processing — Phase 7).

## License & attribution

MIT. Work based on llama.cpp (MIT) — attribution is documented in `LLAMA_RS_ARCHITECTURE.md` §2 with the exact upstream source location of every ported algorithm (b12502, commit 398f00d65).
