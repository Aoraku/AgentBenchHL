#!/bin/bash
# 只测**终局**：把每个 run 末轮选中的那一版拉去打完整个冻结人类池，
# 得到它的真实池内 Elo 与名次，然后退出。
#
# 为什么这是默认路径
# ------------------
# 绝大多数实验要回答的是一句话："同样迭代 N 轮之后，这一版在 229 人池里排第几"。
# 那只需要**一个**版本，不需要整条曲线。成本差一个数量级：
#
#   --best-only --iteration-stride 3   →  50 版 / 22900 局 / 约 17 小时
#   --last-n-best 1                    →   4 版 /  1832 局 / 约 1.6 小时
#
# 而且它跑在**迭代已经结束之后**，机器是空的，所以 --parallel 可以开大，
# 不像 background_pool 那样和迭代抢机时（实测 16 个 worker 一起下水把
# load 打到 40/32，迭代慢 10~20 倍，最后一个版本都没测完）。
#
# 需要曲线的实验才用 attach_slow_eval.sh（或配置里 background_pool: true）。
#
# 用法: bash final_elo.sh <game> <run-id> [run-id ...]
#   例: bash final_elo.sh antwar2 vk4-glm-5.2 vk4-glm-5.3 vk4-kimi-k3
#
# 分轨游戏要额外指定挑战者轨（rollman 是目前唯一一个）：
#   CHALLENGER_TRACK=rollman bash final_elo.sh rollman s8k4-rollman
set -euo pipefail

GAME="${1:?usage: final_elo.sh <game> <run-id> [run-id ...]}"
shift
[ "$#" -ge 1 ] || { echo "至少给一个 run-id" >&2; exit 1; }

HL=/home/qingle/agentbench/AgentBenchHL
AB=/home/qingle/agentbench/AgentBench
VENV=/home/qingle/agentbench/.venv/bin
# 迭代已经结束，机器是空的，所以并发按"核数 - 留给别人的" 给足。
# 机器是共用的（agentlab 上有别人的进程），所以仍然留 6 核并让 worker
# 自己按 load average 退让 —— 别人的 run 也在这台机器上。
PARALLEL="${PARALLEL:-8}"
HEADROOM="${HEADROOM:-6}"

cd "$HL"
for RUN_ID in "$@"; do
  RUN_ROOT="$HL/runs/$RUN_ID"
  LOG="/home/qingle/agentbench/runs/$RUN_ID.final-elo.log"
  if [ ! -d "$RUN_ROOT" ]; then
    echo "SKIP 没有这个 run: $RUN_ROOT" >&2
    continue
  fi
  # 幂等：一个 run 只能有一个 worker。两个 worker 并发写同一个 pool-elo/
  # 会重复调度、结果文件互相覆盖，而且**不报错**。
  # 注意不能用 `pgrep -f -- "--run-root X"`：pattern 以 - 开头会让 pgrep 报
  # "only one pattern can be provided" 并退出非零，而非零在 if 里被当成
  # "没找到"，这道防线就形同不存在（实测因此起过双 worker）。
  if pgrep -af pool_elo_worker.py | grep -qF -- "$RUN_ROOT"; then
    echo "SKIP $RUN_ID 已经有 worker 在跑"
    continue
  fi
  ARGS=(
    --run-root "$RUN_ROOT"
    --agentbench-root "$AB"
    --game "$GAME"
    --seeds 7
    --once
    --parallel "$PARALLEL"
    --headroom "$HEADROOM"
  )
  if [ -n "${ONLY_CANDIDATE:-}" ]; then
    # 监督器在所有迭代进程退出后冻结末轮候选，并显式传入，避免 worker
    # 重新读取后来变化的账本而评测错版本。
    ARGS+=(--only-candidates "$ONLY_CANDIDATE")
  else
    # 人工调用仍保持原来的默认语义：只测末轮选中的一版。
    ARGS+=(--last-n-best 1)
  fi
  if [ -n "${CHALLENGER_TRACK:-}" ]; then
    # 分轨游戏必须指定，否则会同轨互殴（ghost 打 ghost），
    # 那种对局在协议层就没意义，锚点 Elo 不适用。
    ARGS+=(--challenger-track "$CHALLENGER_TRACK")
  fi
  setsid nohup "$VENV/python" -u scripts/pool_elo_worker.py "${ARGS[@]}" \
    > "$LOG" 2>&1 < /dev/null &
  echo "final-elo run=$RUN_ID pid=$! log=$LOG"
done
