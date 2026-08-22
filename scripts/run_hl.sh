#!/bin/bash
# 起一个 HL run（后台、脱离终端）。首跑与续跑用的是同一条命令。
#
# 为什么做成脚本而不是一行 ssh
# --------------------------
# 多层引号在 ssh + fish + bash 之间会被反复剥离，环境变量与 `&` 极易失效
# （实测踩过多次：key 变成空串、后台符号被吃掉导致 ssh 一断进程就死）。
# 脚本文件传上去执行最可靠。
#
# 用法
# ----
#   bash run_hl.sh <config-path> <run-id> [iterations]
#
# iterations 不给则用配置里的 runtime.max_iterations。
#
# ★ 续跑语义（32 → 64 就靠这个）
# ------------------------------
# driver 的 `completed` 计数**每个进程从 0 开始**，而"这个 run 是否已经开始过"
# 是从磁盘状态文件判断的（runs/<id>/.../state）。所以：
#
#   第一次: bash run_hl.sh <cfg> ab32-random 32   → 跑第 1..32 轮
#   第二次: bash run_hl.sh <cfg> ab32-random 32   → 接着跑第 33..64 轮
#
# 也就是说 `iterations` 是"本次再跑多少轮"，不是"总共跑到第几轮"。
# 想续到 64 就再执行一次同样的命令，**不要**把它改成 64（那会再跑 64 轮到 96）。
#
# 幂等保护：如果同一个 run-id 已有活着的进程，直接拒绝启动。
# 两个进程同时写一份 events.jsonl 会把账本搅乱（对局记录交错、
# thread 状态互相覆盖），而且从曲线上看不出来。
set -euo pipefail

CONFIG="${1:?usage: run_hl.sh <config> <run-id> [iterations]}"
RUN_ID="${2:?usage: run_hl.sh <config> <run-id> [iterations]}"
ITERATIONS="${3:-}"

HL=/home/qingle/agentbench/AgentBenchHL
AB=/home/qingle/agentbench/AgentBench
VENV=/home/qingle/agentbench/.venv/bin
RUNS=/home/qingle/agentbench/runs

cd "$HL"

if pgrep -f "goal-led run .*--run-id $RUN_ID( |\$)" > /dev/null; then
  echo "REFUSED: run-id '$RUN_ID' already has a live process (pgrep matched)."
  echo "         两个进程写同一份 events.jsonl 会静默污染账本。"
  echo "         先确认: pgrep -af \"--run-id $RUN_ID\""
  exit 1
fi

export AGENTBENCH_ROOT="$AB"
# key 从 .env 读。configs/models/*.yaml 里只写 api_key_env 的**名字**，
# 绝不写 key 本身，这样配置可以安全进 git。
set -a
. "$HL/.env"
set +a

ARGS=(goal-led run --config "$CONFIG" --run-id "$RUN_ID")
if [ -n "$ITERATIONS" ]; then
  ARGS+=(--iterations "$ITERATIONS")
fi

mkdir -p "$RUNS"
LOG="$RUNS/$RUN_ID.out"
# 追加而不是覆盖：续跑时上一段日志是排查依据，覆盖掉就没法回溯
# "第 32 轮到底怎么收尾的"。
setsid nohup "$VENV/abhl" "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
PID=$!

{
  echo "=== launched $(date -Iseconds) pid=$PID"
  echo "    config=$CONFIG iterations=${ITERATIONS:-<from config>}"
} >> "$RUNS/$RUN_ID.launches"

echo "launched run-id=$RUN_ID pid=$PID log=$LOG"
