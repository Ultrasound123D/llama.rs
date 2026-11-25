#!/usr/bin/env bash
# =============================================================
# Phase 6.6 — Decode & Kernel Bottleneck Audit
# Per-stage decode + prefill profiles (median over REPS process reps,
# TSV op tables parsed by scripts/audit_parse.py), the DRAM streaming
# ceiling (examples/drambw), and the perf_event_open availability
# probe (examples/perfcheck). Output is the raw evidence for the
# Phase 6.6 section of PERFORMANCE_REPORT.md.
# Usage: bash scripts/audit.sh [reps]
# Env:   LLAMA_RS / QWEN / LFM / THREADS
# =============================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_RS=${LLAMA_RS:-$ROOT/target/release/llama-rs-cli}
QWEN=${QWEN:-/home/z/my-project/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
LFM=${LFM:-/home/z/my-project/models/LFM3-1.7B-Q4_K_M.gguf}
REPS=${1:-3}
THREADS=${THREADS:-2}
BASE="The quick brown fox jumps over the lazy dog while the sun sets behind the distant hills and the river flows quietly through the valley carrying leaves and small branches toward the sea where waves meet the shore again and again as birds circle above searching for fish in the fading light of evening"
PROMPT="$BASE $BASE $BASE $BASE $BASE $BASE $BASE $BASE $BASE $BASE" # ~550 tokens
mkdir -p "$ROOT/audit"

echo "== Phase 6.6 audit: reps=$REPS threads=$THREADS =="
echo "== 1. hardware-counter availability probe =="
cargo run --release --example perfcheck --manifest-path crates/llama-kernels/Cargo.toml 2>/dev/null | grep -v "^ *Compiling\|^ *Finished\|^ *Running" || true

echo
echo "== 2. DRAM streaming ceiling (linear read; L3 504 MB) =="
dram=$(cargo run --release --example drambw --manifest-path crates/llama-kernels/Cargo.toml 2>/dev/null | grep "#audit")
echo "$dram"
CEIL_1T=$(echo "$dram" | sed -n 's/.*dram_ceiling_1t=\([0-9.]*\).*/\1/p')
CEIL_2T=$(echo "$dram" | sed -n 's/.*dram_ceiling_2t=\([0-9.]*\).*/\1/p')

echo
echo "== 3. per-stage profiles (median of $REPS process reps) =="
for m in "$LFM" "$QWEN"; do
    name=$(basename "$m" .gguf)
    # decode: natural short prompt, 48 tokens (single kstats+TSV block)
    for i in $(seq "$REPS"); do
        LLAMA_RS_PROF=1 LLAMA_RS_PROF_TSV=1 "$LLAMA_RS" -m "$m" --mode bench \
            -p "The capital of France is" -n 48 -t "$THREADS" --ctx 2048 2>/dev/null \
            | awk '/^kstats:/{blk++} blk==1 && (/^prof-tsv/ || /^#prof-tsv-wall/){print}' \
            > "audit/${name}_decode_$i.tsv"
    done
    # prefill: ~550 tokens at the M=128 sweet spot (first kstats+TSV block)
    for i in $(seq "$REPS"); do
        LLAMA_RS_PROF=1 LLAMA_RS_PROF_TSV=1 LLAMA_RS_PROF_PP=1 LLAMA_RS_PREFILL_M=128 \
            "$LLAMA_RS" -m "$m" --mode bench -p "$PROMPT" -n 4 -t "$THREADS" --ctx 4096 2>/dev/null \
            | awk '/^kstats:/{blk++} blk==1 && (/^prof-tsv/ || /^#prof-tsv-wall/){print}' \
            > "audit/${name}_prefill_$i.tsv"
    done
    echo "---- $name ----"
    export DRAM_CEILING_1T=${CEIL_1T:-14.0} DRAM_CEILING_2T=${CEIL_2T:-17.0}
    python3 "$ROOT/scripts/audit_parse.py" decode "$ROOT/audit/${name}_decode_"*.tsv
    python3 "$ROOT/scripts/audit_parse.py" prefill "$ROOT/audit/${name}_prefill_"*.tsv
done

echo
echo "== 4. llama.cpp reference (same session, pp/tg medians of 2 llama-bench reps) =="
"$ROOT/../llama.cpp-bin/llama-b12502/llama-bench" -m "$LFM" -t "$THREADS" -p 55 -n 32 -r 2 2>/dev/null | grep -E "pp55|tg32" || true
"$ROOT/../llama.cpp-bin/llama-b12502/llama-bench" -m "$QWEN" -t "$THREADS" -p 54 -n 32 -r 2 2>/dev/null | grep -E "pp54|tg32" || true
echo "(llama.cpp per-op profiles are not exposed by llama-bench; the engine-level numbers above are the reference for the gap arithmetic)"
echo "audit complete — TSV evidence in audit/*.tsv"
