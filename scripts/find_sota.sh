#!/bin/bash
# 找出某个 run 里全池实测 Elo 最高的版本（只看当前池指纹，跨指纹不可比）。
#
# 用法: bash find_sota.sh <run_id> <game>
set -euo pipefail

RUN_ID="$1"
GAME="$2"

HL=/home/qingle/agentbench/AgentBenchHL
AB=/home/qingle/agentbench/AgentBench
PY=/home/qingle/agentbench/.venv/bin/python
RUNS=/home/qingle/agentbench/runs

cd "$HL"
"$PY" - "$RUNS/$RUN_ID/pool-elo" "$AB" "$GAME" <<'PYEOF'
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from agentbench_hl.application.challenger_eval import load_frozen_pool

queue_root = Path(sys.argv[1])
pool = load_frozen_pool(sys.argv[2], sys.argv[3])
print(f"当前池指纹 {pool.fingerprint}（{pool.size} 人，榜首 {pool.top_elo:.1f}）")

rows = []
for directory in sorted(queue_root.iterdir()) if queue_root.is_dir() else []:
    summary = directory / "challenger-elo.json"
    if not summary.is_file():
        continue
    try:
        doc = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if doc.get("elo") is None or doc.get("partial"):
        continue
    # 跨池指纹的 Elo 不可比：池子重测过之后旧结果的尺子已经变了。
    if doc.get("pool_fingerprint") != pool.fingerprint:
        continue
    rows.append(doc)

if not rows:
    print("当前指纹下还没有完整跑完的版本")
    raise SystemExit(1)

rows.sort(key=lambda d: -d["elo"])
print(f"\n当前指纹下已实测 {len(rows)} 版，按 Elo 排序：")
for doc in rows[:10]:
    print(
        f"  elo={doc['elo']:<9} #{doc['pool_rank']:<4} "
        f"胜率={doc.get('win_rate')} 局数={doc['complete_matches']:<4} "
        f"iter={doc.get('iteration')} {doc['challenger_id']}"
    )

best = rows[0]
print(f"\nSOTA = {best['challenger_id']}")
print(f"  快照路径 {best['challenger_root']}")
print(
    f"  elo {best['elo']} / 池内 #{best['pool_rank']} / "
    f"距榜首 {best.get('elo_gap_to_top')} / {best['complete_matches']} 局"
)
print(f"  W-D-L {best['wins']}-{best['draws']}-{best['losses']}")
PYEOF
