#!/bin/bash
# Real harness matrix: one guarded server, then for each harness turn 1 = file-read tool call (note.txt) + answer,
# turn 2 = continue the session and recall the word. Pass = "MANGO" in the harness output.
# usage: scripts/harness/real_matrix.sh <ornith|qwen> TAG [harness...]
# env: CONTEXT (262144)  GROW swap-growth MiB (5120)  CEIL swap ceiling MiB (4096)  MAXS guard seconds (5400)
#      TURN_TIMEOUT s per turn (1200)  ENVX="VAR=V ..." server env  PROXY=1 log requests via scripts/harness/logproxy.py
#      ORNITH_CKPT / FLASHNEXT_CKPT checkpoint dirs  PORT (18140)
# Results: results/cpp/harness-matrix/TAG/ (summary.txt, guard.jsonl, server.log, per-turn output/status/health).
set -u
WHICH=$1; TAG=$2; shift 2
HARNESSES=${*:-"opencode codex claude-code qwen-code pi kimi hermes"}
PORT=${PORT:-18140}
WT=$(cd "$(dirname "$0")/../.." && pwd); cd $WT
if [ "$WHICH" = ornith ]; then MODEL=ornith; CKPT=${ORNITH_CKPT:-$HOME/models/Ornith-1.5-35B-A3B-MLX-4bit}
else MODEL=flashnext; CKPT=${FLASHNEXT_CKPT:-$HOME/models/Qwen3.8-Flash-Next-MLX-4bit-MTP}; fi
OUT=$WT/results/cpp/harness-matrix/$TAG; mkdir -p $OUT
for p in $(pgrep -f "model_router|veystrix start|real_run_guard"); do
  ps -o command= -p $p | grep -q "/Veystrix" && { echo "LEFTOVER process $p: refusing"; exit 3; }
done
echo "swap before: $(sysctl -n vm.swapusage)" | tee $OUT/swap-before.txt
.venv/bin/python scripts/flashnext/real_run_guard.py --log $OUT/guard.jsonl --floor-gib 1.5 --swap-growth-mib ${GROW:-5120} \
  --swap-ceiling-mib ${CEIL:-4096} --compressor-growth-mib 8192 --thermal-level 10 --max-seconds ${MAXS:-5400} \
  -- env MallocLargeCache=0 ${ENVX:-A=1} .venv/bin/python -m veystrix start --foreground --engine cpp --model $MODEL \
  --checkpoint "$CKPT" --port $PORT --context ${CONTEXT:-262144} > $OUT/guard-stdout.log 2>$OUT/server.log < /dev/null &
GPID=$!
READY=0
for i in $(seq 1 300); do
  curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ready": true' && { READY=1; break; }
  kill -0 $GPID 2>/dev/null || { echo "engine exited early"; break; }
  sleep 2
done
curl -s http://127.0.0.1:$PORT/veystrix/status > $OUT/status.json 2>/dev/null
curl -s http://127.0.0.1:$PORT/health > $OUT/health.json 2>/dev/null
.venv/bin/python -m veystrix endpoint --json --port $PORT < /dev/null > $OUT/endpoint.json 2>&1
MID=$(python3 -c "import json;print(json.load(open('$OUT/health.json'))['installed_model_id'])")
echo "ready=$READY model=$MID"
HPORT=$PORT
if [ -n "${PROXY:-}" ]; then HPORT=$((PORT+1)); python3 $WT/scripts/harness/logproxy.py $HPORT $PORT $OUT/requests.jsonl & PXPID=$!; sleep 1; fi
T1="Read the file note.txt with your file-reading tool and tell me the secret word it contains."
T2="Which file did you read in this conversation, and what was the secret word? Answer in one line."
HH=$(mktemp -d "${TMPDIR:-/tmp}/vx-harness.XXXXXX")
[ -d "$HH" ] || { echo "no HH"; exit 4; }
args1() { case $1 in
  opencode) echo run;; codex) echo exec --skip-git-repo-check;; claude-code) echo -p;; qwen-code) echo --yolo;;
  pi) echo -p;; kimi) echo -p;; hermes) echo chat --oneshot -q;; esac; }
args2() { case $1 in
  opencode) echo run --continue;; codex) echo exec resume --last --skip-git-repo-check;; claude-code) echo -p --continue;;
  qwen-code) echo --yolo --continue;; pi) echo -c -p;; kimi) echo -c -p;; hermes) echo chat --continue --oneshot -q;; esac; }
for h in $HARNESSES; do
  [ $READY = 1 ] || break
  P=$HH/$h; mkdir -p $P; echo "the secret word is MANGO" > $P/note.txt
  ( cd $P && git init -q . 2>/dev/null )
  PYTHONPATH=$WT HOME=$HH VEYSTRIX_HOME=$HH VEYSTRIX_NO_KEYCHAIN=1 .venv/bin/python -m veystrix connect $h --port $HPORT --model $MID > $OUT/connect-$h.json 2>&1 || echo "connect $h failed"
  for turn in 1 2; do
    if [ $turn = 1 ]; then A=$(args1 $h); MSG=$T1; else A=$(args2 $h); MSG=$T2; fi
    t0=$(date +%s)
    ( cd $P && PYTHONPATH=$WT HOME=$HH VEYSTRIX_HOME=$HH VEYSTRIX_NO_KEYCHAIN=1 timeout ${TURN_TIMEOUT:-1200} $WT/.venv/bin/python -m veystrix run $h --port $HPORT --model $MID -- $A "$MSG" < /dev/null > $OUT/$h-t$turn.out 2>&1 )
    rc=$?
    curl -s http://127.0.0.1:$PORT/veystrix/status > $OUT/status-$h-t$turn.json 2>/dev/null
    curl -s http://127.0.0.1:$PORT/health > $OUT/health-$h-t$turn.json 2>/dev/null
    txt=$(tr -d '\033' < $OUT/$h-t$turn.out | tail -c 4000)
    if [ $turn = 1 ]; then ok=$(echo "$txt" | grep -q MANGO && echo PASS || echo FAIL); else ok=$(echo "$txt" | grep -q MANGO && echo PASS || echo FAIL); fi
    echo "$h turn$turn rc=$rc $(( $(date +%s)-t0 ))s $ok :: $(echo "$txt" | tail -c 200 | tr '\n' ' ')" | tee -a $OUT/summary.txt
  done
done
[ -n "${PXPID:-}" ] && kill $PXPID
pkill -INT -f "model_router.*--port $PORT"
for i in $(seq 1 40); do pgrep -f "model_router.*--port $PORT" >/dev/null || break; sleep 1; done
pkill -TERM -f "model_router.*--port $PORT"; wait $GPID 2>/dev/null
echo "--- guard end"; grep '"event": "end"' $OUT/guard.jsonl | cut -c1-400 | tee -a $OUT/summary.txt
