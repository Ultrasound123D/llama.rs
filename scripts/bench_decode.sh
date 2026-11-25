#!/usr/bin/env bash
# =============================================================
# llama.rs reproducible benchmark (Phase 6.1-6.3, revised 2026-11-25)
# Prefill (pp) and decode (tg) are measured SEPARATELY, under matched
# conditions vs llama-bench: same CPU, same thread count (-t 2), and
# comparable prompt lengths (llama.rs: 56-token natural prompt printed
# by the CLI; llama-bench: synthesizes its own 64-token prompt).
# A/B switch: run with LLAMA_RS_VNNI=0 to disable the packed VNNI
# path in the SAME binary (load-time repack is skipped, AVX2 fallback).
# Usage: bash scripts/bench_decode.sh [reps]
# Env:   LLAMA_RS / LLAMA_BIN / QWEN / LFM / THREADS / CTX / N
# =============================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_RS=${LLAMA_RS:-$ROOT/target/release/llama-rs-cli}
LLAMA_BIN=${LLAMA_BIN:-/home/z/my-project/llama.cpp-bin/llama-b12502}
QWEN=${QWEN:-/home/z/my-project/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
LFM=${LFM:-/home/z/my-project/models/LFM3-1.7B-Q4_K_M.gguf}
REPS=${1:-9}
N=${N:-32}
THREADS=${THREADS:-2}
CTX=${CTX:-2048}
PROMPT="The quick brown fox jumps over the lazy dog while the sun sets behind the distant hills and the river flows quietly through the valley carrying leaves and small branches toward the sea where waves meet the shore again and again as birds circle above searching for fish in the fading light of evening"

bench_rs() { # model -> "pp1 pp2 ...|tg1 tg2 ..." (one series per phase)
    local pps=() tgs=()
    for _ in $(seq "$REPS"); do
        out=$("$LLAMA_RS" -m "$1" --mode bench -p "$PROMPT" -n "$N" -t "$THREADS" --ctx "$CTX" 2>/dev/null)
        pp=$(printf '%s\n' "$out" | grep -oE 'bench-pp: [0-9]+ prompt tokens in [0-9.]+s = [0-9.]+' | grep -oE '[0-9.]+$')
        tg=$(printf '%s\n' "$out" | grep -oE 'bench: [0-9]+ tokens in [0-9.]+s = [0-9.]+' | grep -oE '[0-9.]+$')
        [ -n "$pp" ] && pps+=("$pp")
        [ -n "$tg" ] && tgs+=("$tg")
    done
    echo "${pps[*]}|${tgs[*]}"
}

median_of() { # space-separated numbers -> median
    printf '%s\n' $1 | sort -n | awk '{a[NR]=$1} END {print (NR%2==1)? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'
}
stats_of() { # space-separated numbers -> "median P10 P90" (Phase 6.5: percentile reporting)
    printf '%s\n' $1 | python3 "$ROOT/scripts/stats.py"
}

bench_cpp_both() { # model n_pp -> "pp tg" (llama-bench mean over its internal reps)
    local out pp tg
    out=$("$LLAMA_BIN/llama-bench" -m "$1" -t "$THREADS" -p "$2" -n "$N" -r 5 2>/dev/null)
    pp=$(printf '%s\n' "$out" | awk -F'|' -v tag="pp$2 " '$0 ~ tag {gsub(/ /,"",$8); n=split($8,a,"±"); print a[1]}' | head -1)
    tg=$(printf '%s\n' "$out" | awk -F'|' -v tag="tg$N " '$0 ~ tag {gsub(/ /,"",$8); n=split($8,a,"±"); print a[1]}' | head -1)
    echo "${pp:-0} ${tg:-0}"
}

echo "reps=$REPS n=$N threads=$THREADS ctx=$CTX (prefill and decode measured separately; all runs shown, medians compared)"
echo
for m in "$LFM" "$QWEN"; do
    name=$(basename "$m" .gguf)
    # exact matched prompt length: our natural prompt tokenizes to NPP tokens
    # per model tokenizer; llama-bench synthesizes exactly the same count
    n_pp=$("$LLAMA_RS" -m "$m" --mode tokenize --no-bos -p "$PROMPT" -t 1 2>/dev/null | grep -oE 'tokens\([0-9]+\)' | grep -oE '[0-9]+')
    series=$(bench_rs "$m")
    rs_pp=$(median_of "${series%%|*}")
    rs_tg=$(median_of "${series##*|}")
    read -r cpp_pp cpp_tg <<< "$(bench_cpp_both "$m" "${n_pp:-54}")"
    pp_ratio=$(python3 -c "print(f'{100*float(\"${rs_pp:-0}\")/float(\"${cpp_pp:-1}\"):.0f}')")
    tg_ratio=$(python3 -c "print(f'{100*float(\"${rs_tg:-0}\")/float(\"${cpp_tg:-1}\"):.0f}')")
    read -r pp_med pp_p10 pp_p90 <<< "$(stats_of "${series%%|*}")"
    read -r tg_med tg_p10 tg_p90 <<< "$(stats_of "${series##*|}")"
    pp_ratio=$(python3 -c "print(f'{100*float("${pp_med:-0}")/float("${cpp_pp:-1}"):.0f}')")
    tg_ratio=$(python3 -c "print(f'{100*float("${tg_med:-0}")/float("${cpp_tg:-1}"):.0f}')")
    printf '%-34s pp: llama.rs %6.2f [P10 %6.2f P90 %6.2f] | llama.cpp %6.2f tok/s | %5s%%\n' "$name" "$pp_med" "$pp_p10" "$pp_p90" "$cpp_pp" "$pp_ratio"
    printf '%-34s tg: llama.rs %6.2f [P10 %6.2f P90 %6.2f] | llama.cpp %6.2f tok/s | %5s%%\n' "$name" "$tg_med" "$tg_p10" "$tg_p90" "$cpp_tg" "$tg_ratio"
    echo "  pp runs: ${series%%|*}"
    echo "  tg runs: ${series##*|}"
done
