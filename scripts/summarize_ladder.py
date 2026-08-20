#!/usr/bin/env python3
"""爬梯进展汇总：这个 run 到底把人类池里的**第几名**打穿了。

和 ``watch_progress.py`` 的分工
-------------------------------
那个是"现在怎么样"的体检表（逐轮、含进行中）。这个是"到目前为止爬到第几名"的
**成果单**：按对手分组，给出对手的真实名次与 elo、耗了几轮、最好战绩、是否攻克。

为什么必须显示真实名次
----------------------
事件里的 ``conquest.target_index`` 只是**征服序列的位置**（0,1,2,…），而
``ladder_up`` 的序列是 ``rank10 → rank9 → … → rank1``，所以 index 越大对手越强。
早先这里把 index+1 标成"梯级"，看上去像名次，于是读出来像是"梯级在下降"——方向正好反了。
名次不在事件里，必须回 A 仓的 ``players/measured_elo.json`` 查（BT-MLE 算出的
选手池 elo，按 elo 降序即名次）。

对手 id 里的字样也不能当名次用：``rank11__eve__Mv2Lv1AI__v46`` 的真实名次是 **6**，
``rank11`` 只是那个选手的名字。

按对手分组而不是逐轮看，是因为 conquest 下 ``win_rate`` 会随对手升级周期性归零，
逐轮就是一条锯齿；按对手才能读出"爬了几级、卡在谁身上"。

用法::

    python3 scripts/summarize_ladder.py <run_root> [<run_root> ...]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path


def _ranks(agentbench_root: Path, game: str) -> dict[str, tuple[int, float]]:
    """opponent_id → (名次, elo)；按 measured_elo 降序即名次。"""

    path = agentbench_root / "games" / game / "players" / "measured_elo.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))

    def _elo(item: dict) -> float:
        value = item.get("measured_elo")
        return float(value) if value is not None else -1e9

    ratings = sorted(document.get("ratings") or [], key=lambda item: -_elo(item))
    return {
        str(item.get("player_id")): (index + 1, _elo(item))
        for index, item in enumerate(ratings)
    }


def _metrics(run_root: Path) -> tuple[list[dict], float]:
    rows: list[dict] = []
    tokens = 0
    path = run_root / "events.jsonl"
    if not path.is_file():
        return rows, 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("event_type")
        if kind == "IterationMetricsFinalized":
            rows.append(event.get("payload") or {})
        elif kind == "AgentTokenUsage":
            value = (event.get("payload") or {}).get("total_tokens")
            if isinstance(value, int):
                tokens += value
    rows.sort(key=lambda item: item.get("research_iteration") or 0)
    return rows, tokens / 1e6


def _short(opponent_id: str) -> str:
    return opponent_id.split("__")[-1] if "__" in opponent_id else opponent_id[:10]


def report(run_root: Path, agentbench_root: Path) -> None:
    metrics, tokens = _metrics(run_root)
    if not metrics:
        print(f"[{run_root.name}] 没有已定稿的轮次\n")
        return

    elos = [item.get("pool_elo") for item in metrics if item.get("pool_elo") is not None]
    first, last = metrics[0], metrics[-1]
    game = str(last.get("game") or "?")
    ranks = _ranks(agentbench_root, game)
    print(
        f"■ {run_root.name}  {game}  {len(metrics)} 轮 · tokens {tokens:.1f}M\n"
        f"  pool_elo: 起 {first.get('pool_elo')} → 峰 {max(elos) if elos else '-'} "
        f"→ 末 {last.get('pool_elo')}"
        f"  (净 {float(last.get('pool_elo', 0)) - float(first.get('pool_elo', 0)):+.1f})"
    )

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for item in metrics:
        opponent = (item.get("opponent_ids") or ["?"])[0]
        groups.setdefault(opponent, []).append(item)

    print(
        f"  {'名次':<6} {'对手elo':>8} {'对手':<8} {'轮数':>4} {'最佳战绩':>9} "
        f"{'末轮':>6} {'我方elo':>8} {'状态':<6}"
    )
    for opponent, items in groups.items():
        best = max(items, key=lambda item: item.get("win_rate") or 0.0)
        matches = best.get("matches") or 0
        rate = best.get("win_rate") or 0.0
        wins = int(round(rate * matches))
        tail = items[-1]
        # 攻克判据：这个对手已不是当前目标（后面换人了）＝ 打穿；仍是当前目标 = 进行中。
        finished = opponent != (metrics[-1].get("opponent_ids") or [""])[0]
        rank_info = ranks.get(opponent)
        rank_cell = f"rank{rank_info[0]}" if rank_info else "?"
        elo_cell = f"{rank_info[1]:.0f}" if rank_info else "-"
        print(
            f"  {rank_cell:<6} {elo_cell:>8} {_short(opponent):<8} {len(items):>4} "
            f"{f'{wins}/{matches}':>9} {f'{(tail.get("win_rate") or 0.0) * 100:.0f}%':>6} "
            f"{tail.get('pool_elo'):>8} {'已打穿' if finished else '进行中':<6}"
        )

    cleared = (last.get("conquest") or {}).get("cleared")
    if cleared is not None:
        # 说明：cleared 是"已攻克的对手个数"，序列是 rank10→rank9→…→rank1，
        # 所以 cleared=9 意味着 rank10..rank2 全部打穿，当前目标是榜首。
        print(f"  已攻克对手数: {cleared}（序列 rank10 → rank1，数字越小越强）")
    experience = sorted(run_root.glob("workspace/research/EXPERIENCE.md"))
    if experience:
        lines = len(experience[0].read_text(encoding="utf-8", errors="replace").splitlines())
        print(f"  EXPERIENCE.md {lines} 行")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="爬梯成果单")
    parser.add_argument("run_roots", nargs="+")
    parser.add_argument(
        "--agentbench-root",
        default=os.environ.get("AGENTBENCH_ROOT", "../AgentBench"),
        help="A 仓根目录（名次来自 games/<game>/players/measured_elo.json）",
    )
    args = parser.parse_args()
    root = Path(args.agentbench_root).expanduser().resolve()
    for item in args.run_roots:
        report(Path(item).expanduser().resolve(), root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
