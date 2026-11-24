#!/usr/bin/env bash
# =============================================================
# llama.rs parity tests against llama.cpp (golden reference)
# Requires: llama.cpp binaries + llama.rs built + the models
# Usage: bash tests/parity/run_parity.sh [port_qwen] [port_lfm3]
# Paths overridable via environment variables:
#   LLAMA_BIN (llama.cpp binaries dir) / LLAMA_RS / QWEN / LFM
# =============================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_BIN=${LLAMA_BIN:-/home/z/my-project/llama.cpp-bin/llama-b12502}
LLAMA_RS=${LLAMA_RS:-$ROOT/target/release/llama-rs-cli}
EXAMPLES=${EXAMPLES:-$ROOT/target/release/examples}
QWEN=${QWEN:-/home/z/my-project/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}
LFM=${LFM:-/home/z/my-project/models/LFM3-1.7B-Q4_K_M.gguf}
PORT1=${1:-8081}; PORT2=${2:-8082}

pass=0; fail=0
check() { # name, ok
  if [ "$2" = "1" ]; then pass=$((pass+1)); echo "  PASS  $1"; else fail=$((fail+1)); echo "  FAIL  $1"; fi
}

echo "== [1] tokenize: llama.rs vs llama.cpp /tokenize (mixed-script/English/symbols) =="
declare -a TEXTS=(
  "Hello world"
  "人工智能是一门构建智能系统的科学"
  "The quick brown fox jumps over 42 lazy dogs!"
  "count 123 and numbers 9876543 with symbols !@#\$%^&*"
  "mixed 中文 with English words and 42 numbers"
)
for model in "$QWEN" "$LFM"; do
  name=$(basename "$model")
  port=$PORT1
  ( "$LLAMA_BIN/llama-server" -m "$model" --host 127.0.0.1 --port $port -c 2048 -t 2 >/dev/null 2>&1 & )
  sleep 15
  for txt in "${TEXTS[@]}"; do
    ref=$(curl -s http://127.0.0.1:$port/tokenize -H "Content-Type: application/json" -d "$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1]}))" "$txt")" | python3 -c "import sys,json; print(json.load(sys.stdin)['tokens'])")
    ours=$("$LLAMA_RS" -m "$model" --mode tokenize --no-bos -p "$txt" -t 1 2>/dev/null | grep -o '\[.*\]')
    if [ "$ref" == "$ours" ]; then check "tokenize [$name] '${txt:0:25}…'" 1; else check "tokenize [$name] '${txt:0:25}…' (ref=$ref ours=$ours)" 0; fi
  done
  pkill -f "llama-server.*port $port" 2>/dev/null; sleep 2
done

echo "== [2] distribution after prefill: top-1 match with llama.cpp (greedy) =="
for model in "$QWEN" "$LFM"; do
  name=$(basename "$model")
  port=$PORT2
  ( "$LLAMA_BIN/llama-server" -m "$model" --host 127.0.0.1 --port $port -c 2048 -t 2 >/dev/null 2>&1 & )
  sleep 15
  for prompt in "The capital of France is" "The" ; do
    ref_tok=$(curl -s http://127.0.0.1:$port/v1/completions -H "Content-Type: application/json" -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'max_tokens': 1, 'temperature': 0, 'logprobs': 1}))" "$prompt")" | python3 -c "import sys,json; print(repr(json.load(sys.stdin)['choices'][0]['text']))")
    ours_tok=$("$LLAMA_RS" -m "$model" --mode debug --no-bos -p "$prompt" -t 1 2>/dev/null | grep top1 | python3 -c "import sys; line=sys.stdin.read(); import re; m=re.search(r'text=\"(.*?)\"', line); print(repr(m.group(1)) if m else 'NONE')")
    if [ "$ref_tok" == "$ours_tok" ]; then check "top1-greedy [$name] after '$prompt'" 1; else check "top1-greedy [$name] after '$prompt' (cpp=$ref_tok rs=$ours_tok)" 0; fi
  done
  pkill -f "llama-server.*port $port" 2>/dev/null; sleep 2
done

echo "== [3] dequantization: llama.rs vs C reference (original ggml code) =="
gcc -O2 -o /tmp/ref_q5 "$SCRIPT_DIR/ref_q5.c"
gcc -O2 -o /tmp/ref_q6 "$SCRIPT_DIR/ref_q6.c"

# Q5_0 — column 785 of token_embd.weight in Qwen (offset extracted automatically by dumpcol)
"$EXAMPLES/dumpcol" "$QWEN" token_embd.weight 785 2>/tmp/meta_q5.txt >/tmp/rs_q5.txt
off=$(grep -o 'offset_in_file=[0-9]*' /tmp/meta_q5.txt | head -1 | cut -d= -f2)
cb=$(grep -o 'col_bytes=[0-9]*' /tmp/meta_q5.txt | head -1 | cut -d= -f2)
col_off=$(( off + 785 * cb ))
awk '{print $2}' /tmp/rs_q5.txt > /tmp/rs_q5_vals.txt
/tmp/ref_q5 "$QWEN" "$col_off" "$cb" | tr ' ' '\n' > /tmp/c_q5_vals.txt
maxd=$(paste /tmp/rs_q5_vals.txt /tmp/c_q5_vals.txt | awk '{d=$1-$2; if(d<0)d=-d; if(d>m)m=d} END {print m+0}')
ok=$(python3 -c "print(1 if float('$maxd') < 1e-6 else 0)")
check "dequant Q5_0 column 785 of token_embd (max diff=$maxd)" $ok

# Q6_K — column 0 of blk.0.ffn_down.weight in Qwen
"$EXAMPLES/dumpcol" "$QWEN" blk.0.ffn_down.weight 0 2>/tmp/meta_q6.txt >/tmp/rs_q6.txt
off=$(grep -o 'offset_in_file=[0-9]*' /tmp/meta_q6.txt | head -1 | cut -d= -f2)
cb=$(grep -o 'col_bytes=[0-9]*' /tmp/meta_q6.txt | head -1 | cut -d= -f2)
awk '{print $2}' /tmp/rs_q6.txt > /tmp/rs_q6_vals.txt
/tmp/ref_q6 "$QWEN" "$off" "$cb" | tr ' ' '\n' > /tmp/c_q6_vals.txt
maxd=$(paste /tmp/rs_q6_vals.txt /tmp/c_q6_vals.txt | awk '{d=$1-$2; if(d<0)d=-d; if(d>m)m=d} END {print m+0}')
ok=$(python3 -c "print(1 if float('$maxd') < 1e-6 else 0)")
check "dequant Q6_K column 0 of blk.0.ffn_down (max diff=$maxd)" $ok

echo ""
echo "======================================"
echo "Result: PASS=$pass FAIL=$fail"
[ $fail -eq 0 ] && echo "All parity tests passed" || echo "Review the failures above"
