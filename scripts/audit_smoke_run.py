"""烟测严格验收：不只看"有没有崩"，而是逐条查证迭代**真的发生了**。

为什么需要比"跑完 4 轮"更严的判据
--------------------------------
"进程没崩 + 有 4 条指标"完全可以在下面这些情况下成立，而它们都意味着链路是坏的：

* 对局全部 incomplete —— ``win_rate`` 是 0，看起来像"没赢"而不是"没跑起来"。
  实测 rollman 就栽在这里：座次名传成 P0/P1 而它的角色叫 rollman/ghost，
  12/12 局失败，但 run 照样"跑完"了；
* agent 从没打开规则或回放 —— 它在凭先验瞎写，迭代等于随机搜索；
* 每轮候选只改了个注释 —— 有"多样性"但没有真实策略差异；
* IG 静默退回字母表近似 —— 指标还在，只是不再有意义。

所以这里按**六个维度**验收，每个维度都要有正面证据：

1. 迭代完整性：轮数、无 fatal、无 protocol_error
2. 对局有效性：完成率、有连续奖励（score_margin）
3. 读规则：agent 是否真的读过 gamepack 里的规则/决策空间文件
4. 读回放：agent 是否真的读过本轮下发的回放叙述
5. 真实策略：候选代码是否有实质差异（不只是注释/空白）
6. 精确 IG：support_mode 是否为精确枚举而非字母表近似

另外报告"是否有提升"，但**不作为合格判据**——4 轮烟测的目的是验证链路，
样本量根本不足以证明学习效果，把它当门槛会得出错误结论。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

#: agent 读过这些文件才算"读了规则"。名字取自 gamepack 与 A 的游戏资产。
RULE_FILE_HINTS = (
    "rules.md",
    "decision_space",
    "GOAL_CHARTER",
    "sdk_interface",
    "replay_format",
    "replay_skill",
    "game.yaml",
)

#: 读回放的证据：反馈目录里的叙述文件。
REPLAY_HINTS = ("all-replays.md", "replay", "narration", "feedback")


def _events(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "events.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _payloads(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [
        event["payload"]
        for event in events
        if event.get("event_type") == event_type and isinstance(event.get("payload"), dict)
    ]


def _agent_transcript(run_root: Path) -> str:
    """把 agent 侧留下的文本痕迹拼起来，用于检查它读过什么。

    来源有三处，都要看：
      * ``events.jsonl`` 里 agent 相关事件的 payload（工具调用、命令）；
      * ``*.driver.log``（codex CLI 的 stderr，含工具调用摘要）；
      * ``workspace/`` 下 agent 自己写的笔记。
    """

    chunks: list[str] = []
    events_path = run_root / "events.jsonl"
    if events_path.is_file():
        chunks.append(events_path.read_text(encoding="utf-8", errors="ignore"))
    for log in sorted(run_root.parent.glob(f"{run_root.name}*.log")):
        chunks.append(log.read_text(encoding="utf-8", errors="ignore"))
    workspace = run_root / "workspace"
    if workspace.is_dir():
        for note in sorted(workspace.rglob("*.md"))[:40]:
            chunks.append(note.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


#: 候选包里可能承载策略的文件名。
#
# ⚠️ 不能只看 main.py。多数游戏的 main.py 是**通信壳**（读 stdin / 写 stdout /
# 调 run_ai），真正的策略在 ai.py 里。实测 antwar 的 8 个候选快照 main.py
# 全是同一个 md5，而 ai.py 各不相同——只看 main.py 会得出"候选没有差异"
# 的错误结论。
STRATEGY_FILENAMES = ("ai.py", "strategy.py", "policy.py", "main.py")

#: 明确**不算**策略的文件：框架注入的脚手架与官方模板。
NON_STRATEGY_FILENAMES = (
    "ai_example.py",
    "official_main.py",
    "_bootstrap.py",
    "selfcheck.py",
    "selfcheck_lib.py",
)


def _candidate_bodies(run_root: Path) -> dict[str, str]:
    """读每个候选快照的**策略**代码，用来判断策略是否有实质差异。

    取 ``STRATEGY_FILENAMES`` 里第一个存在的文件；找不到就退回把该快照下
    所有非脚手架 ``*.py`` 拼起来——宁可多读，也不要因为文件名不符预期
    而误判成"没有策略"。
    """

    bodies: dict[str, str] = {}
    snapshots = run_root / "snapshots"
    if not snapshots.is_dir():
        return bodies
    for directory in sorted(snapshots.iterdir()):
        if not directory.is_dir():
            continue
        for filename in STRATEGY_FILENAMES:
            path = directory / filename
            if path.is_file():
                bodies[directory.name] = path.read_text(encoding="utf-8", errors="ignore")
                break
        else:
            merged = [
                path.read_text(encoding="utf-8", errors="ignore")
                for path in sorted(directory.glob("*.py"))
                if path.name not in NON_STRATEGY_FILENAMES
            ]
            if merged:
                bodies[directory.name] = "\n".join(merged)
    return bodies


def _strip_cosmetics(source: str) -> str:
    """去掉注释、空白与文档字符串，只留下真正会执行的代码。

    这样才能区分"改了策略"和"只加了注释"。
    """

    without_docstrings = re.sub(r'"""(?:.|\n)*?"""', "", source)
    without_docstrings = re.sub(r"'''(?:.|\n)*?'''", "", without_docstrings)
    lines: list[str] = []
    for line in without_docstrings.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def audit(run_root: Path, *, expected_iterations: int) -> dict[str, Any]:
    events = _events(run_root)
    by_type = Counter(str(event.get("event_type")) for event in events)
    metrics = _payloads(events, "IterationMetricsFinalized")
    matches = _payloads(events, "GoalMatchCompleted")
    transcript = _agent_transcript(run_root)
    bodies = _candidate_bodies(run_root)

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str, *, gate: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "gate": gate})

    # ---------------------------------------------------------- 1 迭代完整性
    check(
        "迭代轮数达标",
        len(metrics) >= expected_iterations,
        f"完成 {len(metrics)}/{expected_iterations} 轮",
    )
    fatal = by_type.get("GoalLedFailed", 0) + by_type.get("RunFailed", 0)
    check("无 fatal 事件", fatal == 0, f"{fatal} 个")
    protocol_errors = sum(1 for row in metrics if row.get("protocol_error"))
    check("每轮都产出候选", protocol_errors == 0, f"protocol_error {protocol_errors} 轮")

    # ---------------------------------------------------------- 2 对局有效性
    complete = [row for row in matches if row.get("status") == "complete"]
    rate = len(complete) / len(matches) if matches else 0.0
    errors = Counter(
        str(row.get("error"))[:80]
        for row in matches
        if row.get("status") != "complete" and row.get("error")
    )
    detail = f"{len(complete)}/{len(matches)} = {rate:.1%}"
    if errors:
        top, count = errors.most_common(1)[0]
        detail += f"；主要失败：{count}x {top}"
    check("对局完成率 ≥ 0.8", rate >= 0.8, detail)

    with_margin = [
        row for row in complete if isinstance(row.get("score_margin"), (int, float))
    ]
    check(
        "有连续奖励 score_margin",
        bool(complete) and len(with_margin) == len(complete),
        f"{len(with_margin)}/{len(complete)} 局带分差",
    )

    # ---------------------------------------------------------- 3 读规则
    read_rules = sorted({hint for hint in RULE_FILE_HINTS if hint in transcript})
    check(
        "agent 读过规则/决策空间",
        len(read_rules) >= 2,
        f"命中 {read_rules}" if read_rules else "痕迹里找不到任何规则文件",
    )

    # ---------------------------------------------------------- 4 读回放
    narration_files = list((run_root / "workspace" / "feedback").rglob("*.md")) if (
        run_root / "workspace" / "feedback"
    ).is_dir() else []
    read_replay = [hint for hint in REPLAY_HINTS if hint in transcript]
    check(
        "agent 读过回放叙述",
        bool(narration_files) and bool(read_replay),
        f"叙述文件 {len(narration_files)} 份，痕迹命中 {read_replay}",
    )

    # ---------------------------------------------------------- 5 真实策略
    normalised = {name: _strip_cosmetics(body) for name, body in bodies.items()}
    distinct = len(set(normalised.values()))
    sizes = sorted(len(body.splitlines()) for body in normalised.values())
    median = sizes[len(sizes) // 2] if sizes else 0
    duplicates = [
        name
        for name, body in normalised.items()
        if list(normalised.values()).count(body) > 1
    ]
    # 判据分两条：
    #   * 去掉注释/空白后**互不相同**。多样性约束要求每轮 k 个候选走不同路径，
    #     所以重复就是失败——哪怕文件名不同；
    #   * 代码量用**中位数**而不是最小值。
    #
    # 为什么不能用最小值：agent 有时会**故意**写一个极短的诊断候选来隔离问题。
    # 实测 miracle 第 4 轮的 v003_protocol_probe 只有 25 行，docstring 写明
    # "isolate startup, init framing, turn framing from all strategy code"——
    # 那是合理的科学行为（先排除协议层问题再谈策略），不是"没写策略"。
    # 用最小值会把这种诊断轮误判为不合格。
    check(
        "候选是真实策略且互不相同",
        distinct == len(normalised) and median >= 30,
        f"{len(normalised)} 个快照 / {distinct} 份不同代码 / "
        f"行数中位 {median}（最短 {sizes[0] if sizes else 0}）"
        + (f"；重复：{duplicates}" if duplicates else ""),
    )

    # ---------------------------------------------------------- 6 精确 IG
    ig_rounds = [row for row in metrics if row.get("behavioral_ig") is not None]
    modes = Counter(
        str(row.get("behavioral_ig_support_mode"))
        for row in metrics
        if row.get("behavioral_ig_support_mode")
    )
    reasons = Counter(
        str(row.get("behavioral_ig_reason"))
        for row in metrics
        if row.get("behavioral_ig") is None and row.get("behavioral_ig_reason")
    )
    # IG 需要前后两轮配对，第 1 轮天然没有；所以判据是"除首轮外都有"。
    expected_ig = max(0, len(metrics) - 1)
    detail = f"{len(ig_rounds)}/{expected_ig} 轮（首轮无配对基线）"
    if reasons:
        detail += f"；未出值原因 {dict(reasons)}"
    check("behavioral_ig 有值", len(ig_rounds) >= expected_ig and expected_ig > 0, detail)
    exact = sum(count for mode, count in modes.items() if mode != "opcode_alphabet")
    check(
        "IG 用精确支撑集",
        exact >= 1,
        f"support_mode {dict(modes)}" if modes else "没有任何 support_mode 记录",
    )

    # ------------------------------------------------- 提升（只报告，不作门槛）
    trend = [
        {
            "iteration": row.get("research_iteration"),
            "win_rate": row.get("win_rate"),
            "margin_mean": row.get("margin_mean"),
            "behavioral_ig": row.get("behavioral_ig"),
        }
        for row in metrics
    ]
    first = next((row for row in trend if row["win_rate"] is not None), None)
    last = next((row for row in reversed(trend) if row["win_rate"] is not None), None)
    improvement = None
    if first and last and first is not last:
        improvement = {
            "win_rate_delta": round(float(last["win_rate"]) - float(first["win_rate"]), 4),
            "margin_delta": (
                round(float(last["margin_mean"]) - float(first["margin_mean"]), 2)
                if isinstance(last["margin_mean"], (int, float))
                and isinstance(first["margin_mean"], (int, float))
                else None
            ),
        }

    gates = [item for item in checks if item["gate"]]
    return {
        "run_root": str(run_root),
        "iterations": len(metrics),
        "matches": len(matches),
        "complete_matches": len(complete),
        "candidates": len(bodies),
        "checks": checks,
        "trend": trend,
        "improvement": improvement,
        "passed": all(item["ok"] for item in gates),
    }


def render(report: dict[str, Any]) -> str:
    lines = [f"===== 严格验收 {report['run_root']}"]
    for item in report["checks"]:
        lines.append(f"  {'✓' if item['ok'] else '✗'} {item['name']:<26} {item['detail']}")
    lines.append("  --- 逐轮走势（仅报告，不作门槛）")
    for row in report["trend"]:
        lines.append(
            f"      iter {row['iteration']}: win={row['win_rate']} "
            f"margin={row['margin_mean']} ig={row['behavioral_ig']}"
        )
    if report["improvement"]:
        lines.append(f"      首末差：{report['improvement']}")
    lines.append(f"  → {'合格' if report['passed'] else '不合格'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", nargs="+", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--report", type=Path, default=None)
    arguments = parser.parse_args(argv)

    reports = []
    for run_root in arguments.run_root:
        report = audit(run_root.resolve(), expected_iterations=arguments.iterations)
        reports.append(report)
        print(render(report))
        print()
    if arguments.report is not None:
        arguments.report.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if all(item["passed"] for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
