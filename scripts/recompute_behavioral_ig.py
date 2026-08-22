#!/usr/bin/env python3
"""离线重算 behavioral IG：把字母表近似的 |A| 换成逐点精确的 |A(s)|。

为什么可以纯计算，不必重跑对局
------------------------------
在线测量把**逐决策点的 KL** 完整落盘在 ``behavioral-ig/trace-NNNN.json`` 的
``kl_trace`` 里。而 ε-smoothing 下的 KL 有闭式解：

* 两边动作相同        → KL = 0
* 两边动作不同        → KL = (m − u)·ln(m/u)，其中 u = ε/|A|、m = 1 − ε + u

也就是说 KL **只取决于"这一点是否分歧"和"这一点的 |A|"**。前者可以直接从
``kl_trace`` 读出来（>0 即分歧），后者用状态探针在冻结回放上重算。于是整件事退化成
纯算术：不需要重放对局，不需要 transcript（那个目录本来就没保留）。

修的是什么
----------
在线测量曾经因为"状态探针给 246 个决策点、线协议记 247 个"而整局回落到
``|A| = 10`` 的操作码字母表。KL 的尺度逐点由 |A| 决定，所以近似值与精确值不是
同一个量纲的数。这个脚本把已完成轮次的 IG 按精确 |A(s)| 重算，并**同时保留**
两个数——不是覆盖历史，是给出可对照的第二列。

口径一致性
----------
KL 直接调用在线那份 ``epsilon_regularized_kl``，|A(s)| 直接调用在线那份
``support_probe``，不另写一套公式。

用法::

    python3 scripts/recompute_behavioral_ig.py <run_root>
    python3 scripts/recompute_behavioral_ig.py <run_root> --write   # 落盘 json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from agentbench_hl.application.support_probe import support_sizes as probe_support_sizes
from agentbench_hl.domain.metrics import epsilon_regularized_kl

DEFAULT_EPSILON = 0.01


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _events(run_root: Path) -> list[dict]:
    out: list[dict] = []
    for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _epsilon(run_root: Path) -> float:
    config = run_root / "run-config.json"
    if config.is_file():
        document = _load(config)
        for section in ("measurement", "metrics", "information_gain"):
            value = (document.get(section) or {}).get("epsilon")
            if isinstance(value, (int, float)) and 0 < float(value) < 1:
                return float(value)
    return DEFAULT_EPSILON


def _replay_for(run_root: Path, request_id: str, baseline_id: str, role: str, seed: int) -> Path:
    """IG 用的是 baseline 的**影子局**回放（与在线测量同一条）。"""

    return (
        run_root
        / "workspace"
        / "feedback"
        / request_id
        / "shadow"
        / baseline_id
        / f"{role}-seed-{seed}"
        / "replay.json"
    )


def recompute_case(
    kl_trace: list[float], sizes: list[int], epsilon: float
) -> tuple[list[float], int, int]:
    """返回 (重算后的逐点 KL, 用上精确 |A| 的点数, 无法重算的分歧点数)。"""

    out: list[float] = []
    exact = 0
    stranded = 0
    for index, value in enumerate(kl_trace):
        if value <= 0:
            out.append(0.0)
            continue
        size = sizes[index] if index < len(sizes) else None
        if size is None or size < 2:
            # 分歧点却只有 <2 个合法动作 —— 说明这一点没对齐，保留原值并计数，
            # 不能悄悄按 |A|=1 算出一个 0。
            out.append(value)
            stranded += 1
            continue
        legal = tuple(str(item) for item in range(int(size)))
        out.append(epsilon_regularized_kl("0", "1", legal, epsilon))
        exact += 1
    return out, exact, stranded


def main() -> int:
    parser = argparse.ArgumentParser(description="离线重算 behavioral IG（精确 |A(s)|）")
    parser.add_argument("run_root")
    parser.add_argument("--write", action="store_true", help="把结果写进 run 目录")
    parser.add_argument(
        "--agentbench-root",
        default=None,
        help="A 仓路径；省略时从 run-config.json 的 paths.agentbench_root 读。"
        "rollman 这类候选包不自带后端 core/ 的游戏必须有它，否则探针失败并静默退回近似口径",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    config = _load(run_root / "run-config.json") if (run_root / "run-config.json").is_file() else {}
    game = (config.get("goal") or {}).get("game") or config.get("game")
    if not game:
        print("无法从 run-config.json 判断游戏名")
        return 1

    # 探针定位后端要用它。之前这里完全没传，导致 rollman 的重算全轮失败
    # （"cannot locate rollman backend core/ package"），而失败在汇总里
    # 只表现为"没有可重算的轮次"，很容易被当成数据缺失而不是配置缺失。
    agentbench_root = args.agentbench_root or (
        (config.get("paths") or {}).get("agentbench_root")
    )
    agentbench_root = Path(agentbench_root).expanduser().resolve() if agentbench_root else None
    if agentbench_root is not None and not agentbench_root.is_dir():
        print(f"⚠ agentbench_root 不存在：{agentbench_root}（探针可能因此失败）")

    epsilon = _epsilon(run_root)
    print(f"run={run_root.name} game={game} epsilon={epsilon} agentbench_root={agentbench_root}\n")

    header = (
        f"{'it':>3} {'candidate':<26} {'ig_近似':>9} {'ig_精确':>9} {'Δ':>8} "
        f"{'点数':>6} {'精确点':>7} {'|A|中位':>7} {'|A|均值':>8} {'失配':>5}"
    )
    print(header)
    print("-" * len(header))

    results: list[dict] = []
    for event in _events(run_root):
        if event.get("event_type") != "InformationGainMeasured":
            continue
        payload = event["payload"]
        iteration = payload.get("research_iteration")
        trace_path = run_root / "behavioral-ig" / f"trace-{int(iteration):04d}.json"
        if not trace_path.is_file():
            continue
        trace = _load(trace_path)
        baseline_id = payload.get("baseline_candidate_id")
        request_id = payload.get("request_id")
        baseline_root = run_root / "snapshots" / str(baseline_id)

        weighted: list[tuple[float, int]] = []
        all_sizes: list[int] = []
        exact_total = 0
        stranded_total = 0
        for case in trace.get("cases") or []:
            kl_trace = case.get("kl_trace")
            if not kl_trace:
                continue
            replay = _replay_for(
                run_root, str(request_id), str(baseline_id), str(case["role"]), int(case["seed"])
            )
            if not replay.is_file() or not baseline_root.is_dir():
                continue
            try:
                sizes = [int(item) for item in probe_support_sizes(
                    str(game),
                    baseline_root,
                    replay,
                    str(case["role"]),
                    agentbench_root=agentbench_root,
                )]
            except Exception as error:  # noqa: BLE001 - 探针失败要如实跳过并说明
                print(f"  it{iteration} {case['role']}: 探针失败 {type(error).__name__}: {error}")
                continue
            recomputed, exact, stranded = recompute_case(list(kl_trace), sizes, epsilon)
            weighted.append((statistics.fmean(recomputed), len(recomputed)))
            all_sizes.extend(sizes[: len(recomputed)])
            exact_total += exact
            stranded_total += stranded

        if not weighted:
            continue
        total = sum(count for _, count in weighted)
        exact_ig = sum(mean * count for mean, count in weighted) / total
        online = payload.get("behavioral_ig")
        delta = None if online is None else exact_ig - float(online)
        row = {
            "research_iteration": iteration,
            "candidate_id": payload.get("candidate_id"),
            "baseline_candidate_id": baseline_id,
            "behavioral_ig_online": online,
            "behavioral_ig_exact": round(exact_ig, 6),
            "decisions": total,
            "exact_decisions": exact_total,
            "stranded_disagreements": stranded_total,
            "support_median": statistics.median(all_sizes) if all_sizes else None,
            "support_mean": round(statistics.fmean(all_sizes), 2) if all_sizes else None,
            "support_max": max(all_sizes) if all_sizes else None,
            "epsilon": epsilon,
        }
        results.append(row)
        print(
            f"{iteration:>3} {str(payload.get('candidate_id'))[:26]:<26} "
            f"{(f'{online:.4f}' if online is not None else '-'):>9} "
            f"{exact_ig:>9.4f} {(f'{delta:+.4f}' if delta is not None else '-'):>8} "
            f"{total:>6} {exact_total:>7} "
            f"{(row['support_median'] if row['support_median'] is not None else '-'):>7} "
            f"{(row['support_mean'] if row['support_mean'] is not None else '-'):>8} "
            f"{stranded_total:>5}"
        )

    if not results:
        print("没有可重算的轮次（缺 trace / 影子局回放 / 候选快照）")
        return 1

    online_values = [
        float(row["behavioral_ig_online"])
        for row in results
        if row["behavioral_ig_online"] is not None
    ]
    exact_values = [float(row["behavioral_ig_exact"]) for row in results]
    print(
        f"\n{len(results)} 轮：近似均值 {statistics.fmean(online_values):.4f} → "
        f"精确均值 {statistics.fmean(exact_values):.4f}"
    )
    stranded = sum(int(row["stranded_disagreements"]) for row in results)
    if stranded:
        print(f"⚠ 有 {stranded} 个分歧点无法用精确 |A| 重算（已保留原值），说明那些点没对齐")

    if args.write:
        out = run_root / "recomputed-behavioral-ig.json"
        out.write_text(
            json.dumps(
                {"game": game, "epsilon": epsilon, "rows": results}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
