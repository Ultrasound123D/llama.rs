#!/usr/bin/env bash
# =============================================================
# llama.rs per-op decode profile (Phase 6.1 instrumentation)
# One profiled decode per model + the isolated kernel micro-bench.
# Usage: bash scripts/profile_decode.sh
# =============================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_RS=${LLAMA_RS:-$ROOT/target/release/llama-rs-cli}
QWEN=${QWEN:-/home/z/my-project/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
LFM=${LFM:-/home/z/my-project/models/LFM3-1.7B-Q4_K_M.gguf}

for m in "$LFM" "$QWEN"; do
    echo "==== $(basename "$m") decode profile ===="
    LLAMA_RS_PROF=1 "$LLAMA_RS" -m "$m" --mode bench -p "The capital of France is" -n 48 -t 2 2>&1 \
        | grep -vE "^\[llama.rs\]"
    echo
done

echo "==== isolated per-kernel micro-benchmarks ===="
cd "$ROOT" || exit 1
cargo run --release --example kprofile --manifest-path crates/llama-kernels/Cargo.toml 2>/dev/null || \
    cargo run --release --example kprofile 2>/dev/null || true
