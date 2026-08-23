#!/bin/bash
# 给一个**正在跑**的 run 挂上慢评测（后台全池评测）。
#
# 为什么需要它
# ------------
# `evaluation.background_pool` 是在 run **启动时**读的，所以已经在跑的 run
# 不会因为改了配置就自动获得慢评测。而重启 run 会丢掉已经跑完的轮次，
# 代价远大于手动挂一个 worker。新起的 run 不需要这个脚本（配置里打开即可）。
#
# 用法: bash attach_slow_eval.sh <run-id> [stride] [game]
set -euo pipefail

RUN_ID="$1"
STRIDE="${2:-3}"
GAME="${3:-antwar2}"

HL=/home/qingle/agentbench/AgentBenchHL
AB=/home/qingle/agentbench/AgentBench
VENV=/home/qingle/agentbench/.venv/bin
RUN_ROOT="$HL/runs/$RUN_ID"
LOG="/home/qingle/agentbench/runs/$RUN_ID.pool-elo.log"

if [ ! -d "$RUN_ROOT" ]; then
  echo "no such run: $RUN_ROOT" >&2
  exit 1
fi

# 幂等：一个 run 只能有一个 worker。
#
# 不能写 `pgrep -f -- "--run-root $RUN_ROOT"`：pgrep 把 `--` 之后的第一个词当
# pattern、其余当"多余的 pattern"，报 "only one pattern can be provided" 并退出
# 非零。而它在 if 条件里，非零被当成"没找到"，于是这道防线形同不存在——
# 实测因此给两个 run 各起了 2 个 worker，它们并发写同一个 pool-elo/ 目录。
# 改成先全量列出再自己 grep，pattern 就不会以 - 开头。
if pgrep -af pool_elo_worker.py | grep -qF -- "$RUN_ROOT"; then
  echo "slow-eval already attached to $RUN_ID (skipped)"
  exit 0
fi

cd "$HL"
# parallel=2 / headroom=10：四个 run 的慢评测同时跑也只占 8 局并发，
# 主迭代的对局与 agent 思考优先（思考占全程约 84%，抢它的机时最不划算）。
# worker 自己还会按 load average 再退让一层。
setsid nohup "$VENV/python" -u scripts/pool_elo_worker.py \
  --run-root "$RUN_ROOT" \
  --agentbench-root "$AB" \
  --game "$GAME" \
  --seeds 7 \
  --best-only \
  --iteration-stride "$STRIDE" \
  --parallel 2 \
  --headroom 10 \
  > "$LOG" 2>&1 < /dev/null &
echo "attached slow-eval run=$RUN_ID stride=$STRIDE pid=$! log=$LOG"
