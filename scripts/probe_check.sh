#!/bin/bash
# 用真实回放测一个游戏的决策探针，打印决策数与支持集统计。
#
# 用法: bash probe_check.sh <game> <candidate_dir> <replay_path> <role>
#
# 为什么做成脚本：多层引号在 ssh + fish + bash 之间会被反复剥离，
# 内联 python -c 里的引号几乎无法保持（实测踩过多次）。
set -euo pipefail

GAME="$1"
CANDIDATE="$2"
REPLAY="$3"
ROLE="$4"

HL=/home/qingle/agentbench/AgentBenchHL
PY=/home/qingle/agentbench/.venv/bin/python
WORKER="$HL/src/agentbench_hl/adapters/$GAME/policy_trace_worker.py"

cd "$CANDIDATE"
OUT=$(timeout 900 "$PY" "$WORKER" \
  --candidate "$CANDIDATE" --replay "$REPLAY" \
  --match-id probe-check --role "$ROLE" 2>&1) || {
    echo "PROBE FAILED:"
    echo "$OUT" | tail -5
    exit 1
  }

echo "$OUT" | grep AGENTBENCH_POLICY_TRACE | "$PY" -c '
import collections
import json
import sys

line = sys.stdin.read()
payload = json.loads(line.split("=", 1)[1])
decisions = payload["decisions"]
sizes = [len(support) for item in decisions for support in item["legal_supports"]]
print(f"决策数 {len(decisions)}")
if sizes:
    ordered = sorted(sizes)
    print(f"支持集 min/中位/max = {ordered[0]}/{ordered[len(ordered) // 2]}/{ordered[-1]}")
actions = collections.Counter(a for item in decisions for a in item["actions"])
print(f"动作种类 {len(actions)}，最常见 {actions.most_common(4)}")
hold = sum(v for k, v in actions.items() if k in ("HOLD", "END"))
total = sum(actions.values())
if total:
    print(f"空动作占比 {hold / total:.1%}")
'
