"""一眼看清各游戏人类池评分的口径与样本量。

为什么需要它
------------
``measured_elo.json`` 是所有 Elo 的**尺子**：候选版本的静态池 Elo 全靠它做锚点。
尺子换了（degree 从 6 提到 24、局数翻四倍），旧结果就不可比。所以在替换池数据
前后都要能一眼核对 degree / 选手数 / 实测局数，否则很容易把不可信的旧版本
当成权威版本继续用——那会让所有下游曲线静默地建立在错误刻度上。

``degree`` 是每位选手在 Swiss/循环配对里打的场次数：它直接决定标准误
（``s.e. ∝ 1/√n``）。degree=6 时实测标准误 ±106，而池内前十相邻分差只有 5~21，
名次顺序根本不可辨识；degree=24 才把标准误压到 ±52~±96。

用法
----
    python scripts/show_pool_state.py <agentbench_root>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GAMES = (
    "antwar",
    "antwar2",
    "generals",
    "miracle",
    "rollman",
    "lostspace",
    "aquawar",
    "snakego",
    "deepclue",
)


def describe(root: Path, game: str) -> str:
    path = root / "games" / game / "players" / "measured_elo.json"
    if not path.is_file():
        return f"{game:<12} 缺失 {path}"
    document = json.loads(path.read_text(encoding="utf-8"))
    ratings = [
        row
        for row in (document.get("ratings") or [])
        if isinstance(row, dict) and row.get("measured_elo") is not None
    ]
    values = sorted(float(row["measured_elo"]) for row in ratings)
    top = f"{values[-1]:.1f}" if values else "-"
    # 前十相邻分差的最小值：它必须显著大于标准误，名次才是可辨识的。
    gaps = [values[i] - values[i - 1] for i in range(len(values) - 1, max(len(values) - 10, 0), -1)]
    min_gap = f"{min(gaps):.1f}" if gaps else "-"
    return (
        f"{game:<12} degree={document.get('degree')!s:<4} "
        f"选手={len(ratings):<5} 局数={document.get('played_matches')!s:<6} "
        f"榜首={top:<9} 前十最小相邻分差={min_gap}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agentbench_root", type=Path)
    args = parser.parse_args()
    for game in GAMES:
        print(describe(args.agentbench_root.resolve(), game))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
