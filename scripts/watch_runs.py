"""一眼看清若干 HL run 的健康度。

这个脚本存在的理由：撤掉人工监督之后，需要一个**能判断"跑得动"还是
"在空转"**的入口。只看轮数是不够的——实测踩过的坑是轮数在涨、
但每轮的候选代码没变（agent 在原地重交同一份），或者对局全是 0 回合判负
（协议错），这两种情况轮数曲线都很正常。

所以每个 run 报四类信号：

1. **进度**：完成轮数 / 目标轮数、是否还有活进程。
2. **迭代信号**：本 run 出现过几个不同的候选 id、最近一轮胜率与分差。
   候选 id 一直不变 = agent 在空转。
3. **对手策略是否真的生效**：累计打过多少个不同对手、最近一轮打了谁。
   ``fix`` 应该始终是同一批 4 个；``random`` 应该每轮都在换；
   ``progress`` 应该随胜率推进；``self`` 由 agent 决定。
   这是 ablation 的核心校验——如果四个 run 打的对手集合看起来一样，
   那这个消融就白跑了。
4. **健康度**：0 回合对局占比、协议纠正次数、失败事件、token 花费。

用法::

    python scripts/watch_runs.py --runs-root /home/qingle/agentbench/runs \\
        --run-id ab32-antwar2-random ab32-antwar2-self ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METRICS_EVENT = "IterationMetricsFinalized"


@dataclass
class RunView:
    run_id: str
    iterations: int = 0
    candidates: set[str] = field(default_factory=set)
    opponents: set[str] = field(default_factory=set)
    last_opponents: list[str] = field(default_factory=list)
    last_win_rate: float | None = None
    last_margin: float | None = None
    first_win_rate: float | None = None
    matches: int = 0
    zero_round: int = 0
    corrections: int = 0
    failures: list[str] = field(default_factory=list)
    tokens: int = 0
    policy: str | None = None
    batch: int = 0
    stop_reason: str | None = None
    alive: bool = False
    #: 上游退避重试次数（503/429 等）。它不是错误，但**直接决定吞吐**：
    #: 四个 run 并发打同一个中转站时，503 会让每轮的墙钟从几分钟涨到几十分钟，
    #: 表现是"进度看起来卡住了"而日志里毫无报错。不显示出来会被误判成挂死。
    retries: int = 0


def _events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _alive(run_id: str) -> bool:
    """这个 run 的 driver 进程还在不在。

    ``--`` 是必需的：pattern 是 ``--run-id <id>``，以 ``-`` 开头。
    不加 ``--`` 的话 pgrep 会把它当成自己的选项解析，于是**永远返回空**，
    脚本把每个 run 都报成"停"。这个误判很危险——看到"停了"的人会去重启，
    而那会造成两个进程写同一份 events.jsonl（对局记录交错、thread 状态互相
    覆盖），并且从曲线上看不出来。
    """

    try:
        found = subprocess.run(  # noqa: S603,S607 - 固定参数
            ["pgrep", "-f", "--", f"--run-id {run_id}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return bool(found.stdout.strip())


def inspect(runs_root: Path, run_id: str, *, log_dirs: tuple[Path, ...] = ()) -> RunView:
    view = RunView(run_id=run_id, alive=_alive(run_id))
    # 退避重试次数只在 driver 的 stdout 日志里（不进事件账本，因为它是
    # 传输层噪声而不是实验事实）。但它决定吞吐，所以要报出来。
    #
    # 日志与 run 目录可能不在同一处：配置里 runs_root 是相对路径，
    # 解析结果（AgentBenchHL/runs/）和启动脚本写日志的地方
    # （~/agentbench/runs/）未必一致。所以候选路径要多给几个。
    for directory in (*log_dirs, runs_root, runs_root.parent, runs_root.parent.parent / "runs"):
        candidate = directory / f"{run_id}.out"
        if candidate.is_file():
            try:
                view.retries = candidate.read_text(
                    encoding="utf-8", errors="ignore"
                ).count("[llm-retry]")
            except OSError:
                pass
            break
    for event in _events(runs_root / run_id / "events.jsonl"):
        kind = event.get("event_type")
        payload = event.get("payload") or {}
        if kind == "AgentTokenUsage":
            value = payload.get("total_tokens")
            if isinstance(value, int):
                # 逐次相加：这是"一次请求"的花费，不是会话累计值。
                view.tokens += value
        elif kind == "GoalMatchRequested":
            view.policy = payload.get("opponent_policy") or view.policy
            if isinstance(payload.get("batch"), int):
                view.batch = payload["batch"]
            assignment = payload.get("opponent_assignment") or {}
            chosen: list[str] = []
            for values in assignment.values():
                chosen.extend(values if isinstance(values, list) else [values])
            if not chosen and payload.get("opponent_id"):
                chosen = [str(payload["opponent_id"])]
            if chosen:
                view.last_opponents = sorted(set(chosen))
                view.opponents.update(chosen)
        elif kind == "GoalMatchCompleted":
            view.matches += 1
            if not payload.get("rounds"):
                view.zero_round += 1
        elif kind == "GoalLedCorrectionRequested":
            view.corrections += 1
        elif kind in ("GoalLedIterationFailed", "GoalLedFailed"):
            view.failures.append(str(payload.get("error"))[:160])
        elif kind == "GoalLedFinished":
            view.stop_reason = payload.get("stop_reason")
        elif kind == METRICS_EVENT:
            view.iterations += 1
            candidate = payload.get("best_candidate_id")
            if isinstance(candidate, str):
                view.candidates.add(candidate)
            rate = payload.get("win_rate")
            if isinstance(rate, (int, float)):
                view.last_win_rate = float(rate)
                if view.first_win_rate is None:
                    view.first_win_rate = float(rate)
            margin = payload.get("margin_mean")
            if isinstance(margin, (int, float)):
                view.last_margin = float(margin)
    return view


def verdict(view: RunView, *, target: int | None) -> list[str]:
    """把"看起来正常但其实在空转"的情况点出来。"""

    notes: list[str] = []
    if view.iterations == 0:
        notes.append("还没有完成任何一轮（可能仍在第 1 轮思考，agent 思考占全程 ~84%）")
        return notes
    # 空转检测：轮数在涨但候选 id 不变，说明 agent 在重交同一份代码。
    if len(view.candidates) <= 1 and view.iterations >= 3:
        notes.append(
            f"⚠️ {view.iterations} 轮只出现 {len(view.candidates)} 个候选 id —— "
            "疑似空转（agent 在重交同一份代码）"
        )
    if view.matches and view.zero_round / view.matches > 0.3:
        notes.append(
            f"⚠️ 0 回合对局占 {view.zero_round}/{view.matches} —— "
            "候选大概率协议格式错（0 回合=直接判负，学不到策略信息）"
        )
    # 对手策略生效性：这是 ablation 的命门。
    #
    # progress 故意不在这里检查："只打过 b 个对手"是它的**正确**行为——
    # 窗口只在稳定打赢（得分率 > advance_win_rate）之后才推进，
    # 所以胜率还低的时候窗口本来就不该动。实测 17 轮胜率 0.25 时累计
    # 4 个对手，那是策略在如实工作，不是没生效。
    # 判断 progress 有没有生效要看**胜率上去之后窗口是否跟着推进**，
    # 那需要跨轮对比，由 --target 跑完后的曲线回答。
    if view.policy in ("random", "self") and len(view.opponents) <= view.batch:
        notes.append(
            f"⚠️ policy={view.policy} 但累计只打过 {len(view.opponents)} 个对手 —— "
            "对手策略可能没生效（那样四组消融会退化成同一组）"
        )
    if view.policy == "fix" and view.batch and len(view.opponents) > view.batch + 2:
        notes.append(
            f"⚠️ policy=fix 却打过 {len(view.opponents)} 个对手 —— fix 应当固定打前 b 名"
        )
    if view.corrections > 0:
        notes.append(f"协议纠正 {view.corrections} 次（偶发正常，连续 3 次以上会终止 run）")
    for failure in view.failures[-2:]:
        notes.append(f"✗ 失败事件: {failure}")
    if target and view.iterations >= target and view.alive:
        notes.append("已达目标轮数但进程仍在（正常：最后一轮的收尾）")
    if not view.alive and (target is None or view.iterations < target):
        # stop_reason 为空 = driver 没走到正常收尾（被杀 / OOM / 未捕获异常）。
        # 有 stop_reason 说明是它自己决定停的（预算耗尽、协议纠正超限…）。
        cause = view.stop_reason or "未知（driver 没写 stop_reason，可能是被杀或崩溃）"
        notes.append(
            f"✗ 进程已退出但只完成 {view.iterations}"
            + (f"/{target}" if target else "")
            + f" 轮，原因={cause}；续跑：bash scripts/run_hl.sh <config> {view.run_id} "
            + str((target or view.iterations) - view.iterations)
        )
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--run-id", nargs="+", required=True)
    parser.add_argument("--target", type=int, default=None, help="目标轮数，用于判断是否跑完")
    parser.add_argument(
        "--log-dir",
        nargs="*",
        type=Path,
        default=[],
        help="driver stdout 日志（<run-id>.out）所在目录；不给则在 runs-root 附近找",
    )
    arguments = parser.parse_args(argv)

    problems = 0
    for run_id in arguments.run_id:
        view = inspect(
            arguments.runs_root.resolve(),
            run_id,
            log_dirs=tuple(item.resolve() for item in arguments.log_dir),
        )
        head = (
            f"[{run_id}] policy={view.policy} 轮数={view.iterations}"
            + (f"/{arguments.target}" if arguments.target else "")
            + f" 进程={'活' if view.alive else '停'}"
        )
        print(head)
        trend = ""
        if view.first_win_rate is not None and view.last_win_rate is not None:
            trend = f" (首轮 {view.first_win_rate} → 最近 {view.last_win_rate})"
        print(
            f"    候选={len(view.candidates)} 个  对局={view.matches} "
            f"(0 回合 {view.zero_round})  胜率{trend}  分差={view.last_margin}"
        )
        print(
            f"    对手: 累计 {len(view.opponents)} 个，最近一轮 "
            f"{[item.split('__')[0] for item in view.last_opponents] or '—'}"
        )
        print(f"    token={view.tokens / 1_000_000:.2f}M", end="")
        if view.retries:
            print(f"  上游退避重试={view.retries} 次（503/429，拖慢吞吐但不是错误）")
        else:
            print()
        for note in verdict(view, target=arguments.target):
            print(f"    {note}")
            if note.startswith(("⚠️", "✗")):
                problems += 1
        print()
    print(f"总计 {problems} 项需要关注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
