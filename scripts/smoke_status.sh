#!/bin/bash
# 汇报所有烟测 run 的实时进度：事件数、最近事件类型、最后更新时间。
#
# 为什么做成脚本：ssh 到远端时本地是 fish、远端是 bash，for/do/done 与引号
# 会被反复重解析（实测踩过多次）。脚本文件传上去执行最可靠。
set -uo pipefail

RUNS=/home/qingle/agentbench/runs
PY=/home/qingle/agentbench/.venv/bin/python

echo "时间: $(date '+%F %T')  负载: $(cut -d' ' -f1-3 /proc/loadavg)"
echo

for RUN_ID in "$@"; do
  DIR="$RUNS/$RUN_ID"
  EVENTS="$DIR/events.jsonl"
  if [ ! -f "$EVENTS" ]; then
    echo "$RUN_ID: 尚无 events.jsonl"
    continue
  fi
  COUNT=$(wc -l < "$EVENTS")
  MTIME=$(stat -c %y "$EVENTS" | cut -d. -f1)
  AGE=$(( $(date +%s) - $(stat -c %Y "$EVENTS") ))
  ALIVE=$(pgrep -fc "run-id $RUN_ID" || echo 0)
  printf '%-22s 事件 %-5s 最后更新 %s (%ss 前) 进程 %s\n' \
    "$RUN_ID" "$COUNT" "$MTIME" "$AGE" "$ALIVE"

  "$PY" - "$EVENTS" <<'PYEOF'
import collections
import json
import sys

path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8", errors="ignore"):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        continue

counts = collections.Counter(str(r.get("event_type")) for r in rows)
metrics = [
    r["payload"] for r in rows
    if r.get("event_type") == "IterationMetricsFinalized" and isinstance(r.get("payload"), dict)
]
matches = [
    r["payload"] for r in rows
    if r.get("event_type") == "GoalMatchCompleted" and isinstance(r.get("payload"), dict)
]
done = [m for m in matches if m.get("status") == "complete"]
print(f"      轮次 {len(metrics)} | 对局 {len(done)}/{len(matches)} 完成", end="")
if metrics:
    last = metrics[-1]
    print(
        f" | 末轮 win={last.get('win_rate')} ig={last.get('behavioral_ig')}"
        f" mode={last.get('behavioral_ig_support_mode')}"
    )
else:
    print("")
if matches and not done:
    errs = collections.Counter(str(m.get("error"))[:70] for m in matches if m.get("error"))
    for detail, n in errs.most_common(2):
        print(f"      ✗ {n}x {detail}")
recent = [str(r.get("event_type")) for r in rows[-3:]]
print(f"      最近事件 {recent}")
PYEOF
  echo
done
