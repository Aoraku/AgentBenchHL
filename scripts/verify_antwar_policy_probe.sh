#!/usr/bin/env bash
# 在服务器上验证 antwar policy probe：跑完整一局，输出成本与 |A(s)| 分布摘要。
# 只读，不修改任何 run 数据。
set -euo pipefail

V=/home/qingle/agentbench/.venv/bin/python
HL=/home/qingle/agentbench/AgentBenchHL
RUN=/home/qingle/agentbench/runs/exp2-antwar-4iter
CAND_ID="${1:-v002_cross_mortar_evasion}"
ROLE="${2:-P0}"
MAX_ROUNDS="${3:-}"

REPLAY=$(find "$RUN/workspace/feedback" -path "*/$CAND_ID/$ROLE-seed-*/replay.json" | head -1)
if [ -z "$REPLAY" ]; then echo "找不到 $CAND_ID/$ROLE 的回放"; exit 1; fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp -R "$HL/gamepacks/antwar/candidate_support/." "$WORK/"
cp "$RUN/workspace/.agentbench/rollouts/$CAND_ID/ai.py" "$WORK/ai.py"

ARGS=(--candidate "$WORK" --replay "$REPLAY" --match-id probe --role "$ROLE")
if [ -n "$MAX_ROUNDS" ]; then ARGS+=(--max-rounds "$MAX_ROUNDS"); fi

cd "$HL"
if ! timeout 900 "$V" src/agentbench_hl/adapters/antwar/policy_trace_worker.py "${ARGS[@]}" \
     > /tmp/pt.txt 2>/tmp/pt.err; then
  echo "worker 失败："; tail -c 1200 /tmp/pt.err; exit 1
fi

"$V" - <<'PY'
import json, statistics
raw = open("/tmp/pt.txt").read().strip()
prefix = "AGENTBENCH_POLICY_TRACE="
if not raw.startswith(prefix):
    print("输出格式不对：", raw[:200]); raise SystemExit(1)
d = json.loads(raw[len(prefix):])
decisions = d["decisions"]
sizes = [len(s) for x in decisions for s in x["legal_supports"]]
print(f"回合 {d['rounds_replayed']}/{d['rounds_total']}  truncated={d['truncated']}  耗时 {d['elapsed_s']}s")
print(f"决策点 {len(decisions)}  原子动作 {len(sizes)}")
print(f"每决策点 {d['elapsed_s']/max(len(decisions),1):.4f}s")
print(f"|A(s)| min={min(sizes)} 中位={statistics.median(sizes)} mean={statistics.mean(sizes):.1f} max={max(sizes)}")
below = 100 * sum(1 for s in sizes if s < 10) / len(sizes)
above = 100 * sum(1 for s in sizes if s > 10) / len(sizes)
print(f"相对字母表近似 |A|=10：真实更小 {below:.1f}%，真实更大 {above:.1f}%")
print(f"唯一 occupancy_id {len({x['occupancy_id'] for x in decisions})}")
print(f"提交过动作的决策点 {sum(1 for x in decisions if x['actions'])}")
PY
