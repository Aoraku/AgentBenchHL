"""批量重测所有游戏的人类池 Elo。

为什么要重测
------------
现有 9 个池全部是 ``degree=6``（每人只打 12 局）跑出来的，审计结果显示这个样本量
撑不起"排第几"这句话：

* Elo 标准误中位 **±106**（BT 粗估 400/ln10 / sqrt(n·p·(1-p))）；
* 而前十名相邻分差中位只有 **5~30 分** —— 比误差小一个数量级，
  意味着榜首那几名的**具体顺序完全不可辨识**；
* antwar2 前十里 8 人胜率 1.000、miracle 前十里 10 人胜率 1.000。
  对"从未失败"的选手，BT 只能给出下界，真实强度不可辨识；
* deepclue 更是 **0 个有分选手**（它的动作是自由文本，决策空间不可枚举）。

把 ``degree`` 提到 24 能把标准误压到约 ±53（∝1/sqrt(n)），同时让胜率饱和的
人数显著下降（对手更多 ⇒ 更可能碰到强手而输一局 ⇒ 强度变得可辨识）。

为什么用一个批量脚本而不是手敲 8 条命令
--------------------------------------
* 池子重测会**改变锚点尺子**，所有依赖它的挑战者评测都必须在之后重跑。
  一个脚本把顺序固定下来，避免人工漏掉某个游戏；
* 8 个游戏总计约 3 万局，需要串行排队（每个游戏内部并发），
  否则会互相抢核并把 load 顶穿；
* 每个游戏跑完立刻落盘 + 审计，中途挂掉可以从下一个游戏继续。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# 池子规模差异很大（generals 81 人 ~ miracle 305 人），但赛程是 degree 驱动的，
# 所以逐个游戏的对局数正比于人数。按对局数从小到大排：先出结果的先看到。
GAMES = (
    "generals",
    "antwar",
    "rollman",
    "snakego",
    "lostspace",
    "aquawar",
    "antwar2",
    "miracle",
)


def run_one(
    game: str,
    *,
    abhl: str,
    agentbench_root: Path,
    degree: int,
    seeds: str,
    parallel: int,
    cpus_per_match: int,
    timeout: float,
    log_dir: Path,
) -> dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ladder-{game}.log"
    command = [
        abhl,
        "ladder",
        "eval",
        game,
        "--agentbench-root",
        str(agentbench_root),
        "--degree",
        str(degree),
        "--seeds",
        seeds,
        "--parallel",
        str(parallel),
        "--cpus-per-match",
        str(cpus_per_match),
        "--timeout",
        str(timeout),
    ]
    started = time.time()
    print(f"[ladder] {game} 开始：{' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(  # noqa: S603 - 参数全部由本脚本构造
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.time() - started
    measured = agentbench_root / "games" / game / "players" / "measured_elo.json"
    summary: dict[str, object] = {
        "game": game,
        "returncode": completed.returncode,
        "elapsed_s": round(elapsed, 1),
        "log": str(log_path),
    }
    if measured.is_file():
        document = json.loads(measured.read_text(encoding="utf-8"))
        rated = [
            row
            for row in document.get("ratings") or []
            if isinstance(row, dict) and row.get("measured_elo") is not None
        ]
        saturated = sum(1 for row in rated if float(row.get("winrate") or 0.0) >= 1.0)
        summary.update(
            {
                "degree": document.get("degree"),
                "rated_players": len(rated),
                "played_matches": document.get("played_matches"),
                "saturated": saturated,
            }
        )
    print(
        f"[ladder] {game} 完成（rc={completed.returncode}，{elapsed / 60:.1f} 分钟）："
        f"{summary.get('rated_players')} 人有分，"
        f"{summary.get('played_matches')} 局，胜率饱和 {summary.get('saturated')} 人",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abhl", default="abhl", help="abhl 可执行文件路径")
    parser.add_argument("--agentbench-root", required=True, type=Path)
    parser.add_argument("--games", nargs="*", default=list(GAMES))
    parser.add_argument(
        "--degree",
        type=int,
        default=24,
        help="每人对手数。6 → 标准误约 ±106（当前）；24 → 约 ±53",
    )
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--cpus-per-match", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    arguments = parser.parse_args(argv)

    results: list[dict[str, object]] = []
    for game in arguments.games:
        results.append(
            run_one(
                game,
                abhl=arguments.abhl,
                agentbench_root=arguments.agentbench_root.resolve(),
                degree=arguments.degree,
                seeds=arguments.seeds,
                parallel=arguments.parallel,
                cpus_per_match=arguments.cpus_per_match,
                timeout=arguments.timeout,
                log_dir=arguments.log_dir.resolve(),
            )
        )
        if arguments.report is not None:
            arguments.report.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    failed = [row for row in results if row["returncode"] != 0]
    print(f"\n[ladder] 全部完成：{len(results)} 个游戏，{len(failed)} 个失败", flush=True)
    for row in failed:
        print(f"  失败 {row['game']} rc={row['returncode']} 见 {row['log']}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
