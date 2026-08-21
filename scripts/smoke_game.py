"""烟测一个游戏：跑 4 轮迭代，然后逐条验收。

为什么需要一个专门的验收脚本
----------------------------
"跑起来没报错"不是验收标准。有几类失败**不会**让 run 崩掉，只会让指标静默失去意义：

* 精确 IG 探针崩了 → ``behavioral_ig`` 退回字母表近似（LESSONS_LEARNED B 条）；
* 对局全部 incomplete → ``win_rate`` 是 0，看起来像"没赢"而不是"没跑起来"；
* 候选包格式不对 → 每轮 protocol_error，迭代照样往下走。

所以本脚本在 run 结束后读 ``events.jsonl``，逐条判定：

1. 迭代轮数是否达到要求（中途死掉就是不合格）；
2. 每轮是否真的产出了候选，且候选之间有差异；
3. 对局的 complete 率；
4. ``behavioral_ig`` 是否非 null，且 ``support_mode`` 是否为精确口径；
5. ``score_margin`` 是否存在（没有它，全败轮就没有梯度，见 G 条）；
6. 是否出现过任何 fatal / 未捕获异常。

任何一条不过就报不合格并指出原因，**不给"大致可以"这种结论**。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "configs" / "experiments" / "smoke-template.yaml"


def render_config(game: str, *, opponent_rank: int, iterations: int, out_dir: Path) -> Path:
    """把模板渲染成某个游戏的烟测配置。"""

    document = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    document["game"] = game
    document["curriculum"]["opponent_rank"] = opponent_rank
    document["runtime"]["max_iterations"] = iterations
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"smoke-{game}.yaml"
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    return path


def _events(run_root: Path) -> list[dict]:
    path = run_root / "events.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def verify(run_root: Path, *, expected_iterations: int) -> dict[str, object]:
    """逐条验收，返回结构化结论。"""

    events = _events(run_root)
    by_type = Counter(str(event.get("event_type")) for event in events)

    metrics = [
        event["payload"]
        for event in events
        if event.get("event_type") == "IterationMetricsFinalized"
        and isinstance(event.get("payload"), dict)
    ]
    matches = [
        event["payload"]
        for event in events
        if event.get("event_type") == "GoalMatchCompleted"
        and isinstance(event.get("payload"), dict)
    ]

    complete = [row for row in matches if row.get("status") == "complete"]
    with_margin = [
        row for row in complete if isinstance(row.get("score_margin"), (int, float))
    ]
    ig_values = [
        row.get("behavioral_ig") for row in metrics if row.get("behavioral_ig") is not None
    ]
    support_modes = Counter(
        str(row.get("behavioral_ig_support_mode"))
        for row in metrics
        if row.get("behavioral_ig_support_mode")
    )
    protocol_errors = sum(1 for row in metrics if row.get("protocol_error"))
    candidates = {
        str(row.get("candidate_id")) for row in matches if row.get("candidate_id")
    }

    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check(
        "迭代轮数达标",
        len(metrics) >= expected_iterations,
        f"完成 {len(metrics)}/{expected_iterations} 轮",
    )
    check(
        "每轮都产出候选",
        protocol_errors == 0,
        f"protocol_error 轮数 {protocol_errors}",
    )
    check(
        "候选有多样性",
        len(candidates) >= max(2, len(metrics)),
        f"不同候选 {len(candidates)} 个（{len(metrics)} 轮）",
    )
    rate = len(complete) / len(matches) if matches else 0.0
    check(
        "对局完成率 ≥ 0.8",
        rate >= 0.8,
        f"{len(complete)}/{len(matches)} = {rate:.2%}",
    )
    check(
        "有连续奖励 score_margin",
        bool(complete) and len(with_margin) == len(complete),
        f"{len(with_margin)}/{len(complete)} 局带分差",
    )
    check(
        "behavioral_ig 非 null",
        len(ig_values) >= 1,
        f"{len(ig_values)}/{len(metrics)} 轮有 IG 值",
    )
    exact = sum(
        count for mode, count in support_modes.items() if mode not in ("opcode_alphabet",)
    )
    check(
        "IG 用精确支撑集",
        exact >= 1,
        f"support_mode 分布 {dict(support_modes)}",
    )
    fatal = by_type.get("GoalLedFailed", 0) + by_type.get("RunFailed", 0)
    check("无 fatal 事件", fatal == 0, f"fatal 事件 {fatal} 个")

    return {
        "run_root": str(run_root),
        "iterations": len(metrics),
        "matches": len(matches),
        "complete_matches": len(complete),
        "candidates": len(candidates),
        "ig_rounds": len(ig_values),
        "support_modes": dict(support_modes),
        "event_types": dict(by_type),
        "checks": checks,
        "passed": all(bool(item["ok"]) for item in checks),
    }


def render_verdict(report: dict[str, object]) -> str:
    lines = [f"===== 验收 {report['run_root']}"]
    for item in report["checks"]:  # type: ignore[union-attr]
        mark = "✓" if item["ok"] else "✗"
        lines.append(f"  {mark} {item['name']:<24} {item['detail']}")
    lines.append(f"  → {'合格' if report['passed'] else '不合格'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--agentbench-root", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs" / "experiments")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--opponent-rank", type=int, default=30)
    parser.add_argument("--abhl", default="abhl")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="不跑 run，只对已有的 run 目录做验收",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--report", type=Path, default=None)
    arguments = parser.parse_args(argv)

    run_id = arguments.run_id or f"smoke-{arguments.game}"
    run_root = (arguments.runs_root / run_id).resolve()

    if not arguments.verify_only:
        config = render_config(
            arguments.game,
            opponent_rank=arguments.opponent_rank,
            iterations=arguments.iterations,
            out_dir=arguments.config_dir,
        )
        print(f"[smoke] {arguments.game} 配置 {config}", flush=True)
        environment = dict(os.environ)
        environment["AGENTBENCH_ROOT"] = str(arguments.agentbench_root.resolve())
        log_path = run_root.parent / f"{run_id}.driver.log"
        run_root.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(f"[smoke] {arguments.game} 开始，日志 {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(  # noqa: S603 - 参数由本脚本构造
                [
                    arguments.abhl,
                    "goal-led",
                    "run",
                    "--config",
                    str(config),
                    "--run-id",
                    run_id,
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=environment,
                check=False,
            )
        print(
            f"[smoke] {arguments.game} 结束 rc={completed.returncode} "
            f"{(time.time() - started) / 60:.1f} 分钟",
            flush=True,
        )

    report = verify(run_root, expected_iterations=arguments.iterations)
    print(render_verdict(report))
    if arguments.report is not None:
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
