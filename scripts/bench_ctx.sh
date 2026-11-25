#!/usr/bin/env bash
# =============================================================
# Phase 6.6 — context-length sweep: decode (tg) as the KV cache
# grows. llama.rs and llama-bench both decode -n 32 AFTER prefilling
# a matched prompt length (the context the decoder attends to).
# LFM3's attention is SWA-windowed (256) + hybrid conv, Qwen2.5 is
# full causal attention — the contrast is the point of the sweep.
# llama.rs results are appended incrementally to audit/ctx_<model>.rs
# (a timeout never loses completed rows); llama-bench needs one
# invocation per length (its multi -p collapses tg to a single row).
# Usage: MODELS="path" bash scripts/bench_ctx.sh
# Env:   L_LIST (default "64 256 1024 2048 4096"), CTX_REPS (default 1)
#        LLAMA_RS / LLAMA_BIN / QWEN / LFM / THREADS
# =============================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_RS=${LLAMA_RS:-$ROOT/target/release/llama-rs-cli}
LLAMA_BIN=${LLAMA_BIN:-/home/z/my-project/llama.cpp-bin/llama-b12502}
QWEN=${QWEN:-/home/z/my-project/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
LFM=${LFM:-/home/z/my-project/models/LFM3-1.7B-Q4_K_M.gguf}
THREADS=${THREADS:-2}
N=${N:-32}
CTX_REPS=${CTX_REPS:-1}
L_LIST=${L_LIST:-"64 256 1024 2048 4096"}
BASE="The quick brown fox jumps over the lazy dog while the sun sets behind the distant hills and the river flows quietly through the valley carrying leaves and small branches toward the sea where waves meet the shore again and again as birds circle above searching for fish in the fading light of evening"

median_of() { printf '%s\n' $1 | sort -n | awk '{a[NR]=$1} END {print (NR%2==1)? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }

mkdir -p "$ROOT/audit"
echo "context sweep: reps=$CTX_REPS threads=$THREADS n=$N — decode AFTER prefill of the matched prompt length (KV rows = prompt + generated)"

for m in ${MODELS:-"$LFM" "$QWEN"}; do
    name=$(basename "$m" .gguf)
    RSLOG="$ROOT/audit/ctx_${name%%-*}.rs"
    LOG="$ROOT/audit/ctx_${name%%-*}.log"
    : > "$RSLOG"
    {
    echo "-- $name (llama.rs tg per ctx; llama.cpp appended after) --"
    printf '%8s %12s %12s %8s\n' "ctx(tok)" "llama.rs" "llama.cpp" "ratio"
    } > "$LOG"

    lengths=()   # actual token counts
    tgs_rs=()    # llama.rs tg medians, parallel to lengths

    # ---- llama.rs: one process per length ----
    for L in $L_LIST; do
        k=$(( (L + 54) / 55 )); [ "$k" -lt 1 ] && k=1
        PROMPT=""
        for ((j = 0; j < k; j++)); do PROMPT="$PROMPT $BASE"; done
        n=$("$LLAMA_RS" -m "$m" --mode tokenize --no-bos -p "$PROMPT" -t 1 2>/dev/null | grep -oE 'tokens\([0-9]+\)' | grep -oE '[0-9]+')
        [ -z "$n" ] && continue
        tgs=()
        for _ in $(seq "$CTX_REPS"); do
            out=$("$LLAMA_RS" -m "$m" --mode bench -p "$PROMPT" -n "$N" -t "$THREADS" --ctx 8192 2>/dev/null)
            tg=$(printf '%s\n' "$out" | grep -oE 'bench: [0-9]+ tokens in [0-9.]+s = [0-9.]+' | grep -oE '[0-9.]+$')
            [ -n "$tg" ] && tgs+=("$tg")
        done
        rs=$(median_of "${tgs[*]:-0}")
        printf '%8s %12s\n' "$n" "$rs" >> "$RSLOG"   # incremental evidence
        lengths+=("$n")
        tgs_rs+=("$rs")
    done

    # ---- llama.cpp: one llama-bench invocation per length ----
    for i in "${!lengths[@]}"; do
        v=${lengths[$i]}
        out=$("$LLAMA_BIN/llama-bench" -m "$m" -t "$THREADS" -p "$v" -n "$N" -r 1 2>/dev/null)
        cpp=$(printf '%s\n' "$out" | awk -F'|' -v tag="tg$N " '$0 ~ tag {gsub(/ /,"",$8); split($8,a,"±"); print a[1]}' | head -1)
        cpp=${cpp:-0}
        rs=${tgs_rs[$i]:-0}
        ratio=$(python3 -c "print(f'{100*float(\"${rs:-0}\")/float(\"${cpp:-1}\"):.0f}')")
        printf '%8s %12s %12s %7s%%\n' "$v" "${rs:-0}" "$cpp" "$ratio" >> "$LOG"
    done
    cat "$LOG"
done
