#!/bin/bash
# 打印某个 run 里 behavioral IG 的口径与失败原因。
#
# 用来诊断"IG 有值但 support_mode 是 opcode_alphabet"这类静默降级：
# 精确探针崩了会退回字母表近似，指标还在、只是不再有意义
# （见 docs/LESSONS_LEARNED.md B 条）。
set -uo pipefail

RUNS=/home/qingle/agentbench/runs
PY=/home/qingle/agentbench/.venv/bin/python

for RUN_ID in "$@"; do
  EVENTS="$RUNS/$RUN_ID/events.jsonl"
  [ -f "$EVENTS" ] || { echo "$RUN_ID: 无 events.jsonl"; continue; }
  echo "===== $RUN_ID"
  "$PY" - "$EVENTS" <<'PYEOF'
import json
import sys

rows = []
for line in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        continue

for row in rows:
    if row.get("event_type") != "IterationMetricsFinalized":
        continue
    payload = row.get("payload") or {}
    print(f"  iter {payload.get('research_iteration')}:")
    for key in sorted(payload):
        low = key.lower()
        if "ig" in low or "support" in low or "probe" in low:
            value = payload[key]
            if isinstance(value, str) and len(value) > 220:
                value = value[:220] + "..."
            print(f"    {key} = {value!r}")
PYEOF
  echo
done
