# llama.rs — a native Rust port of the llama.cpp engine (b12502)

A GGUF inference engine written in pure Rust, **with zero C/C++ runtime linkage**, output-compatible with llama.cpp — proven by automated parity tests (24/24 passing).

**Releases:** [v0.1.0](https://github.com/Ultrasound123D/llama.rs/releases/tag/v0.1.0) (engine complete, parity 16/16) → [v0.2.0](https://github.com/Ultrasound123D/llama.rs/releases/tag/v0.2.0) (SIMD kernels, Qwen decode ~2.5×) → [v0.3.0](https://github.com/Ultrasound123D/llama.rs/releases/tag/v0.3.0) (packed VNNI + batched prefill — at/above llama.cpp on two axes; linux-x64 binary attached) → [v0.4.0](https://github.com/Ultrasound123D/llama.rs/releases/tag/v0.4.0) (flash attention + register-blocked GEMM + grouped q/k/v dispatch — Qwen decode 97 %, LFM3 prefill 91 %, Qwen prefill 141 % of llama.cpp; linux-x64 binary attached). Release notes carry the measured evidence for each milestone.

## Status (v0.4 — Phases 0–6.8 complete)

| Feature | Status | Evidence |
|---|---|---|
| Full GGUF v3 reader | ✅ | metadata + tensors + offsets match |
| Dequantization Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/F16/F32/BF16 | ✅ | matches the C reference (the ggml code itself): 5e-10 diff |
| Hybrid LFM3 architecture (16 shortconv + 8 gated attention + SWA + learned RoPE + sandwich norms) | ✅ | top-1 greedy match + a full chat answer byte-identical to llama.cpp |
| llama/qwen2 family (GQA + q/k/v biases + SwiGLU) | ✅ | Qwen2.5-0.5B: first 14 tokens identical |
| BPE tokenizer (llama3-pre + ignore_merges + byte-encode + merges + special fragmentation) | ✅ | 10/10 tokenize checks (5 texts × 2 models: mixed-script/English/symbols) = 100% identical ids |
| LFM3 chat template (thinking tags extracted from the raw template) + ChatML | ✅ | chat mode works on both models |
| Sampling: greedy/temp/top-k/top-p | ✅ | — |
| CLI: generate/chat/tokenize/bench/debug/batchcmp/mem | ✅ | — |
| **Phase 6 SIMD kernels** (int8-domain AVX2+VNNI vec_dots, activation quantizers, persistent worker pool) | ✅ | LFM3-1.7B: **0.9 → 12.0 tok/s (~13×)** on this 2-vCPU VM; parity 16/16 still green |
| **Phase 6.3–6.4 packed VNNI decode path** (load-time repack of Q8_0/Q5_0/Q4_K/Q6_K) | ✅ | LFM3 decode at **101–102 % of llama.cpp** (30-rep campaign, median [P10 P90]) |
| **Phase 6.5 batched prefill (M>1 packed GEMM)** | ✅ | LFM3 pp **12 → 54 tok/s (4.5×)**; **Qwen pp at 102 % of llama.cpp** (154.7 vs 152.0); Qwen pp512 sweep M∈{1,8,32,128,512}; parity 24/24 |
| **Phase 6.5 memory reclaim** (madvise of repacked originals) | ✅ | LFM3 peak RSS **2050 → 1227 MB**; Qwen **1079 → 706 MB** steady / 724 peak |
| **Phase 6.6 Decode & Kernel Bottleneck Audit** (stage profiling, DRAM-ceiling accounting, CPU-counter probe, ctx/batch sweeps) | ✅ | every remaining gap classified with evidence (`PERFORMANCE_REPORT.md` §9): decode is kernel-limited (not memory — LFM3 streams 12.2 GB/s, equal to llama.cpp); prefill is instruction-throughput bound; **new: the per-key f32 attention core is 54 % of Qwen prefill and up to 72 % of long-context decode** |
| **Phase 6.7 attention flash kernel** (blocked online-softmax, register-blocked V, SWA-aware; + `target_feature` shim-call fix, + FTZ/DAZ guard) | ✅ | LFM3 long-ctx decode **50 % → 79 %** of llama.cpp at ctx 4k (curve near-flat: 103/97/93/79 %); Qwen ctx 4k **28 % → 43 %**; **Qwen prefill 115 %** of llama.cpp (179 vs 156); LFM3 tg holds **105 %**; attention share of Qwen prefill **54 % → 11–13 %**; parity 24/24, 42 tests |
| **Phase 6.8 register-blocked GEMM + grouped q/k/v dispatch** (RB=4-row tile kernels for all repacked types; one quantization + one band-split for the q/k/v triple; small-tensor single-band policy) | ✅ | K-quant GEMM **1.7–2.1×** at M=128 (gate+up pair 297→150 µs/tok, 0.47→0.92 GB/s); **LFM3 prefill 43 % → 91 %** of llama.cpp (M=128, 90.1 tok/s, 2.1×); **Qwen prefill 141 %** (215 tok/s at M=256); **Qwen decode 77–85 % → 97 %** (25.1 vs 25.8 tok/s); LFM3 tg 106–108 %; bit-identity to `matvec_vnni` proven per column; parity 24/24, 43 tests |

## Build & run

```bash
cargo build --release

# Raw generation (llama-completion equivalent)
./target/release/llama-rs-cli -m LFM3-1.7B-Q4_K_M.gguf --mode generate -p "The capital of France is" -n 32 -t 2

# Chat (LFM3 template with thinking tags)
./target/release/llama-rs-cli -m LFM3-1.7B-Q4_K_M.gguf --mode chat -p "What is the capital of Morocco? Answer briefly." -n 60 -t 2

# Tokenize
./target/release/llama-rs-cli -m model.gguf --mode tokenize --no-bos -p "some text"

# Benchmark (batched prefill + decode, with RSS report)
./target/release/llama-rs-cli -m model.gguf --mode bench -p "warm" -n 16 -t 2

# Batched-prefill self-check (token-at-a-time vs batched logits + top-1)
./target/release/llama-rs-cli -m model.gguf --mode batchcmp -p "The capital of France is" -t 2

# Memory study (RSS / peak RSS at load → prefill → decode)
./target/release/llama-rs-cli -m model.gguf --mode mem -p "warm" -t 2
```

### Knobs (A/B switches, same binary)

- `LLAMA_RS_PREFILL_M` — batched-prefill chunk size (default 512; `1` = token-at-a-time)
- `LLAMA_RS_MEM_KEEP=1` — keep the original tensor bytes resident (disable the madvise reclaim)
- `LLAMA_RS_VNNI=0` — disable the packed VNNI path (AVX2 fallback everywhere)
- `LLAMA_RS_SLOT=0` / `LLAMA_RS_SPLIT` — dispatch-path and band-split knobs
- `LLAMA_RS_PROF=1` — op-level TSC profile; `LLAMA_RS_PROF_PP=1` — prefill-phase profile (bench mode); `LLAMA_RS_PROF_TSV=1` — machine-readable rows (`scripts/audit.sh` parses them)

### Audit tooling (Phase 6.6)

```bash
bash scripts/audit.sh 3                      # stage tables (decode + prefill) for both models + DRAM ceiling + perf probe
MODELS=<model> bash scripts/bench_ctx.sh      # decode vs context length (llama.rs vs llama-bench)
cargo run --release --example perfcheck --manifest-path crates/llama-kernels/Cargo.toml  # perf_event_open availability probe
cargo run --release --example drambw   --manifest-path crates/llama-kernels/Cargo.toml  # DRAM streaming ceiling
```

## Parity tests (llama.cpp as golden reference)

```bash
bash tests/parity/run_parity.sh
# Current result: PASS=24 FAIL=0
#  - tokenization ×10 (mixed-script/English/digits/symbols texts, on both models)
#  - top-1 greedy after prefill ×4 (includes the hybrid LFM3)
#  - batched-prefill gates ×8 (Phase 6.5): batched top-1 == token-at-a-time top-1,
#    and the batchcmp logits self-check, on both models × 2 prompts
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

- **Throughput vs llama.cpp on this VM** (matched 2-thread benchmark with exactly-matched prompt lengths, prefill/decode measured separately, medians): **LFM3-1.7B decode 101–105 % of llama.cpp**; **Qwen2.5-0.5B prefill 115 %** (after the Phase 6.7 attention flash kernel); remaining gaps (Qwen decode 77–85 %, LFM3 prefill 42–64 %) are classified with measurements in `PERFORMANCE_REPORT.md` §9 — the largest single terms are now the shuffle-bound GEMM at M ≥ 64 (63 % of LFM3 prefill wall) and the Qwen attn-projection dispatch granularity. Long-context decode after Phase 6.7: LFM3 79 % / Qwen 43 % of llama.cpp at ctx ≈ 4k (was 50 %/28 % — the flash kernel fixed the collapse; §9.5 re-measured). All quantized decode tensors — Q8_0/Q5_0 *and* the K-quants Q4_K/Q6_K — now run on the repacked-VNNI M=1 route (ported from llama.cpp's AMX backend as `llama-kernels/src/vnni.rs` — AVX-512 VNNI; no AMX tile instructions are executed by this engine), and that path is **verified against an independent C golden built from llama.cpp's own quantizer and vec_dot sources** (byte-exact quantizers, 1e-7-of-term-energy agreement across 153 randomized + edge cases — see `PERFORMANCE_REPORT.md` §3.4). Same-binary A/B: the packed path alone lifts LFM3 decode 8.85 → 12.02 tok/s (+36%). Memory: the madvise reclaim keeps RSS at ~1.05× model (peak 1227 MB LFM3 / 724 MB Qwen).
- An AVX-512 VNNI q4_K kernel exists (`llama-kernels/src/avx512.rs`, `examples/q4cmp.rs`) but measured ~9% slower than AVX2 on this VM (frequency licensing) — kept off the default dispatch.
- No HTTP server/batching/GPU yet (Phase 7); two supported architectures (Phase 8 expands this).
- Chat template subset (no full jinja).
- Long-context decode: the Phase 6.6 collapse is fixed (Phase 6.7 flash kernel: LFM3 79 % / Qwen 43 % at ctx ≈ 4k, near-flat for LFM3); the remaining long-ctx gap is the O(L) strided KV scan, measured at ~131 cycles/key ≈ DRAM-latency bound — a kv-layout change is the ranked follow-up.

## License & attribution

MIT. Work based on llama.cpp (MIT) — attribution is documented in `LLAMA_RS_ARCHITECTURE.md` §2 with the exact upstream source location of every ported algorithm (b12502, commit 398f00d65).
