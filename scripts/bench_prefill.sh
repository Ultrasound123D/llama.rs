#!/usr/bin/env bash
# =============================================================
# Phase 6.5 prefill benchmark: batch-size sweep vs llama-bench.
#
# For every model: llama.rs prefill tok/s at M in {1, 8, 32, 128, 512}
# (LLAMA_RS_PREFILL_M; the prompt is processed in chunks of M) against
# llama-bench at the same prompt length. Prompts are synthesized to be
# >= 512 tokens so every batch size actually applies; the M=1 column is
# the token-at-a-time baseline (Phase 6.4 behavior).
# Repetitions: REPS per cell (default 5), median / P10 / P90 printed.
# Usage: bash scripts/bench_prefill.sh [reps]
# Env: LLAMA_RS / LLAMA_BIN / QWEN / LFM / THREADS / CTX
# =============================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_RS=${LLAMA_RS:-$ROOT/target/release/llama-rs-cli}
LLAMA_BIN=${LLAMA_BIN:-/home/z/my-project/llama.cpp-bin/llama-b12502}
QWEN=${QWEN:-/home/z/my-project/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
LFM=${LFM:-/home/z/my-project/models/LFM3-1.7B-Q4_K_M.gguf}
REPS=${1:-5}
THREADS=${THREADS:-2}
CTX=${CTX:-4096}
BASE="The quick brown fox jumps over the lazy dog while the sun sets behind the distant hills and the river flows quietly through the valley carrying leaves and small branches toward the sea where waves meet the shore again and again as birds circle above searching for fish in the fading light of evening"
PROMPT="$BASE $BASE $BASE $BASE $BASE $BASE $BASE $BASE $BASE $BASE" # ~550 tokens

pp_series() { # model -> prefill tok/s series (one per rep) at current LLAMA_RS_PREFILL_M
    local out series=()
    for _ in $(seq "$REPS"); do
        out=$("$LLAMA_RS" -m "$1" --mode bench -p "$PROMPT" -n 4 -t "$THREADS" --ctx "$CTX" 2>/dev/null)
        pp=$(printf '%s\n' "$out" | grep -oE 'bench-pp: [0-9]+ prompt tokens in [0-9.]+s = [0-9.]+' | grep -oE '[0-9.]+$')
        [ -n "$pp" ] && series+=("$pp")
    done
    echo "${series[*]:-0}"
}

echo "prefill sweep: reps=$REPS threads=$THREADS ctx=$CTX (prompt ~550 tok; llama.rs medians [P10 P90] vs llama-bench mean)"
for m in "$LFM" "$QWEN"; do
    name=$(basename "$m" .gguf)
    # matched prompt length for llama-bench (exact token count per model tokenizer)
    n_pp=$("$LLAMA_RS" -m "$m" --mode tokenize --no-bos -p "$PROMPT" -t 1 2>/dev/null | grep -oE 'tokens\([0-9]+\)' | grep -oE '[0-9]+')
    cpp=$("$LLAMA_BIN/llama-bench" -m "$m" -t "$THREADS" -p "${n_pp:-512}" -n 4 -r 2 2>/dev/null |
        awk -F'|' -v tag="pp${n_pp:-512} " '$0 ~ tag {gsub(/ /,"",$8); n=split($8,a,"±"); print a[1]}' | head -1)
    cpp=${cpp:-0}
    echo "-- $name (llama.cpp b12502 pp = ${cpp} tok/s, prompt ${n_pp} tokens)"
    for M in 1 8 32 128 512; do
        s=$(LLAMA_RS_PREFILL_M=$M pp_series "$m")
        read -r med p10 p90 <<< "$(printf '%s\n' $s | python3 "$ROOT/scripts/stats.py")"
        ratio=$(python3 -c "print(f'{100*float(\"${med:-0}\")/float(\"${cpp:-1}\"):.0f}')")
        printf '   M=%-3s llama.rs %7.2f tok/s  [P10 %7.2f  P90 %7.2f]  = %5s%% of llama.cpp\n' "$M" "$med" "$p10" "$p90" "$ratio"
    done
done
