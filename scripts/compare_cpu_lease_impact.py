"""比较不同 ``cpus_per_match`` 下的对局误判率（自然实验）。

背景
----
A 的对战器用**墙钟**判每步超时。若多局对局共享核，计算重的选手会因为邻居抢占
而被误判"超时"——"慢但合法"的策略被系统性误杀，跑分随并行度漂移，基准失去可比性
（见 ``cpu_leases.py`` 记录的实测事故）。所以 ``cpus_per_match`` 不能随便降。

但它也不能设得过大：实测一局（两个选手进程轮流走子 + 判题器）只吃约 0.65 核，
而 3 核租约让 32 核机器只能跑 10 路，利用率仅 21%，白扔 2/3 算力。

本脚本不另跑校准，而是直接利用**已有的自然实验**：批量重测过程中，
不同游戏用过不同的 ``cpus_per_match``（早期 3 核、后来 2 核）。
把各自的 ``incomplete`` 率摆在一起，就能看出降核是否引入了误判。

判据：``incomplete`` 率不随 ``cpus_per_match`` 下降而升高，且失败原因里
没有出现新的超时类错误。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def summarise(path: Path, cpus: int | None) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    bad = [row for row in rows if row.get("status") != "complete"]
    reasons = collections.Counter()
    for row in bad:
        detail = row.get("error") or row.get("diagnostic") or "unknown"
        reasons[str(detail)[:70]] += 1
    rounds = [row["rounds"] for row in rows if isinstance(row.get("rounds"), int)]
    return {
        "game": path.parent.name.replace("abhl-ladder-", ""),
        "cpus_per_match": cpus,
        "matches": len(rows),
        "incomplete": len(bad),
        "incomplete_rate": round(len(bad) / len(rows), 4) if rows else None,
        "rounds_mean": round(sum(rounds) / len(rounds), 1) if rounds else None,
        "top_reasons": reasons.most_common(3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob-root", default="/tmp", help="含 abhl-ladder-* 目录的根")
    parser.add_argument(
        "--cpus-map",
        default="",
        help="game=cpus 逗号分隔，例如 generals=3,antwar=3,miracle=2",
    )
    arguments = parser.parse_args(argv)

    mapping: dict[str, int] = {}
    for item in arguments.cpus_map.split(","):
        if "=" in item:
            name, _, value = item.partition("=")
            mapping[name.strip()] = int(value)

    rows: list[dict[str, object]] = []
    for path in sorted(Path(arguments.glob_root).glob("abhl-ladder-*/matches.jsonl")):
        game = path.parent.name.replace("abhl-ladder-", "")
        rows.append(summarise(path, mapping.get(game)))

    header = (
        f"{'game':<12}{'cpus':>5}{'matches':>9}{'incomplete':>12}"
        f"{'rate':>8}{'rounds_mean':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['game']:<12}{row['cpus_per_match'] or '?':>5}{row['matches']:>9}"
            f"{row['incomplete']:>12}{row['incomplete_rate']:>8}"
            f"{row['rounds_mean'] or '-':>13}"
        )
        for reason, count in row["top_reasons"]:  # type: ignore[union-attr]
            if reason and reason != "None":
                print(f"      {count}x {reason}")

    grouped: dict[int, list[float]] = collections.defaultdict(list)
    for row in rows:
        cpus = row["cpus_per_match"]
        rate = row["incomplete_rate"]
        if isinstance(cpus, int) and isinstance(rate, float):
            grouped[cpus].append(rate)
    if len(grouped) > 1:
        print("\n按 cpus_per_match 汇总（越低越省核，误判率不应上升）：")
        for cpus in sorted(grouped, reverse=True):
            values = grouped[cpus]
            print(
                f"  {cpus} 核: {len(values)} 个游戏，"
                f"incomplete 率 平均 {sum(values) / len(values):.4f}，"
                f"最高 {max(values):.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
