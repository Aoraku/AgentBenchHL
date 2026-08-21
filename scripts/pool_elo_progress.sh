#!/bin/bash
# 汇报某个 run 的全池 Elo 评测进度：完成多少版本、峰值是谁。
#
# 与 pool_elo_status.py 的区别：这个只看"已完整跑完"的版本（partial=false），
# 因为半份数据画出的曲线会被误读（见 docs/LESSONS_LEARNED.md K 条）。
set -uo pipefail

RUNS=/home/qingle/agentbench/runs
PY=/home/qingle/agentbench/.venv/bin/python

for RUN_ID in "$@"; do
  DIR="$RUNS/$RUN_ID/pool-elo"
  if [ ! -d "$DIR" ]; then
    echo "$RUN_ID: 无 pool-elo 目录"
    continue
  fi
  echo "===== $RUN_ID"
  "$PY" - "$DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
done, partial, failed = [], [], 0
for directory in sorted(root.iterdir()):
    summary = directory / "challenger-elo.json"
    if not summary.is_file():
        if directory.is_dir():
            failed += 1
        continue
    try:
        doc = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    (partial if doc.get("partial") else done).append(doc)

print(f"  完整跑完 {len(done)} 版 | 未跑完 {len(partial)} 版 | 无结果 {failed} 版")
if done:
    fingerprints = {d.get("pool_fingerprint") for d in done}
    print(f"  池指纹 {fingerprints}")
    rated = [d for d in done if d.get("elo") is not None]
    if rated:
        peak = max(rated, key=lambda d: d["elo"])
        print(
            f"  峰值 {peak['challenger_id']} (iter {peak.get('iteration')}) "
            f"elo={peak['elo']} 名次=#{peak['pool_rank']} "
            f"胜率={peak.get('win_rate')} 局数={peak['complete_matches']}"
        )
        with_iter = sorted(
            (d for d in rated if d.get("iteration") is not None),
            key=lambda d: d["iteration"],
        )
        print(f"  有迭代序号的 {len(with_iter)} 版：")
        for d in with_iter[:12]:
            print(
                f"    iter {d['iteration']:>3} {str(d['challenger_id'])[:30]:<30} "
                f"elo={d['elo']:<9} #{d['pool_rank']:<4} win={d.get('win_rate')}"
            )
if partial:
    print(f"  未跑完的前 3 个：{[d.get('challenger_id') for d in partial[:3]]}")
PYEOF
  echo
done
