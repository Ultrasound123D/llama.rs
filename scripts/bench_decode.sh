#!/usr/bin/env bash
# =============================================================
# llama.rs reproducible decode benchmark (Phase 6.1-6.3)
# Multiple repetitions, both models, llama.cpp references.
# Usage: bash scripts/bench_decode.sh [reps]
# Env:   LLAMA_RS=..., LLAMA_BIN=..., QWEN=..., LFM=...
# =============================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_RS=${LLAMA_RS:-$ROOT/target/release/llama-rs-cli}
LLAMA_BIN=${LLAMA_BIN:-/home/z/my-project/llama.cpp-bin/llama-b12502}
QWEN=${QWEN:-/home/z/my-project/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
LFM=${LFM:-/home/z/my-project/models/LFM3-1.7B-Q4_K_M.gguf}
REPS=${1:-9}
N=${N:-64}
THREADS=${THREADS:-2}
PROMPT="The capital of France is"

bench_rs() { # model -> "v1 v2 ..." all run values
    local vals=()
    for _ in $(seq "$REPS"); do
        v=$("$LLAMA_RS" -m "$1" --mode bench -p "$PROMPT" -n "$N" -t "$THREADS" 2>/dev/null \
            | grep -oE '= [0-9.]+ tok/s' | grep -oE '[0-9.]+')
        [ -n "$v" ] && vals+=("$v")
    done
    echo "${vals[*]}"
}

median_of() { # space-separated numbers
    printf '%s\n' $1 | sort -n | awk '{a[NR]=$1} END {print (NR%2==1)? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'
}

bench_cpp() { # model -> llama-bench tg tok/s
    "$LLAMA_BIN/llama-bench" -m "$1" -t "$THREADS" -p 64 -n 32 -r 3 2>/dev/null \
        | awk -F'|' '/tg32/ {gsub(/ /,"",$8); n=split($8,a,"±"); print a[1]}' | head -1
}

echo "reps=$REPS n=$N threads=$THREADS  (all runs shown, medians compared)"
echo
for m in "$LFM" "$QWEN"; do
    name=$(basename "$m" .gguf)
    runs=$(bench_rs "$m")
    rs_med=$(median_of "$runs")
    cpp_med=$(bench_cpp "$m")
    ratio=$(python3 -c "print(f'{100*float(\"${rs_med:-0}\")/float(\"${cpp_med:-1}\"):.1f}')")
    printf '%-32s llama.rs %6.2f tok/s | llama.cpp %6.2f tok/s | ratio %5s%%\n' \
        "$name" "$rs_med" "$cpp_med" "$ratio"
    echo "  runs: $runs"
done
