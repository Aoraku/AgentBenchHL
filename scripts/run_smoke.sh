#!/bin/bash
# 起一个游戏的 4 轮烟测（后台）。
#
# 单独做成脚本而不是一行 ssh 命令：多层引号在 ssh + fish + bash 之间会被反复
# 剥离，环境变量和 & 极易失效（实测踩过多次）。脚本文件传上去执行最可靠。
#
# 用法: bash run_smoke.sh <game> <opponent_rank> <run_id>
set -euo pipefail

GAME="$1"
RANK="$2"
RUN_ID="$3"

HL=/home/qingle/agentbench/AgentBenchHL
AB=/home/qingle/agentbench/AgentBench
VENV=/home/qingle/agentbench/.venv/bin
RUNS=/home/qingle/agentbench/runs

cd "$HL"
export AGENTBENCH_ROOT="$AB"
# 临时改用 teamorouter 中转做烟测；主线实验仍走 tsinghua 中转。
export ABHL_API_KEY=sk-teamo-3776c539c97f060096a414a44eda613d394f42c1a97bf736

setsid nohup "$VENV/python" -u scripts/smoke_game.py \
  --game "$GAME" \
  --agentbench-root "$AB" \
  --runs-root "$RUNS" \
  --abhl "$VENV/abhl" \
  --iterations 4 \
  --opponent-rank "$RANK" \
  --run-id "$RUN_ID" \
  --report "$RUNS/$RUN_ID.report.json" \
  > "$RUNS/$RUN_ID.out" 2>&1 < /dev/null &

echo "launched $GAME rank=$RANK run_id=$RUN_ID pid=$!"
