#!/bin/bash
# 起某个 run 的后台全池 Elo 评测队列。
#
# 用法: bash run_pool_elo.sh <run_id> <game> [parallel] [headroom]
#
# 做成脚本而不是内联 ssh 命令：多层引号在 ssh + fish + bash 之间会被反复剥离，
# nohup/setsid 与后台符号极易失效（实测踩过多次）。
set -euo pipefail

RUN_ID="$1"
GAME="$2"
PARALLEL="${3:-8}"
HEADROOM="${4:-2}"

HL=/home/qingle/agentbench/AgentBenchHL
AB=/home/qingle/agentbench/AgentBench
VENV=/home/qingle/agentbench/.venv/bin
RUNS=/home/qingle/agentbench/runs

cd "$HL"
setsid nohup "$VENV/python" -u scripts/pool_elo_worker.py \
  --run-root "$RUNS/$RUN_ID" \
  --agentbench-root "$AB" \
  --game "$GAME" \
  --parallel "$PARALLEL" \
  --cpus-per-match 2 \
  --headroom "$HEADROOM" \
  --poll-s 60 \
  > "$RUNS/pool-elo-$RUN_ID.log" 2>&1 < /dev/null &

echo "launched pool-elo worker run=$RUN_ID game=$GAME parallel=$PARALLEL pid=$!"
