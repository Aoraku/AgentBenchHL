#!/usr/bin/env python3
"""候选脚手架验收：**用真实对战器跑一局**，跑通才算数。

为什么必须真跑
--------------
脚手架的作用是让候选（LLM 写的 ``ai.py``）能被对战器当成合法选手启动。
如果脚手架有问题，症状是**静默的**：该游戏所有候选都"0 回合判负"，
主表上会表现成"这个模型在这个游戏完全不行"——一个彻底的假结论。
复用票数、静态 import 检查都拦不住这种错误，只有真跑一局能。

做法
----
对每个（游戏, 角色轨）：

1. 用 ``candidate_support/`` 装一个候选包，并放入
   ``_shared/candidate_probes/`` 里的**最小合法 ai.py**（探针只保证动作合法，
   不含策略，多数直接复用官方 SDK 自带的示例）；
2. 从**审计通过**的人类选手里挑一个当对手（非对称游戏挑对位角色轨）；
3. 用 ``ContractArena`` 真跑一局；
4. 判定：``status == complete`` 且 ``rounds != 0``。
   （``rounds is None`` 也算通过：rollman 的对战器压根不上报回合数，
   把"没上报字段"当成"第一帧就判负"曾经让整个池子 0 通过。）

用法
----
    python scripts/verify_candidate_support.py                       # 全部
    python scripts/verify_candidate_support.py --game snakego
    python scripts/verify_candidate_support.py --parallel 4 --report out.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentbench_hl.adapters.contract.arena import ContractArena  # noqa: E402
from agentbench_hl.adapters.contract.factory import (  # noqa: E402
    _supports_compiled_players,
    game_roles,
)
from agentbench_hl.adapters.contract.pool import (  # noqa: E402
    PoolPlayer,
    load_pool,
    opposing_track,
    players_in_track,
    runnable_players,
)
from agentbench_hl.adapters.isolation import select_candidate_isolation  # noqa: E402
from agentbench_hl.ports.arena import MatchCase  # noqa: E402
from agentbench_hl.ports.isolation import IsolationRequest  # noqa: E402

GAMEPACKS = REPO_ROOT / "gamepacks"
PROBES = GAMEPACKS / "_shared" / "candidate_probes"
AGENTBENCH_ROOT = Path(os.environ.get("AGENTBENCH_ROOT", REPO_ROOT.parent / "AgentBench"))

CANDIDATE_ID = "candidate-support-probe"


@dataclass
class Target:
    """一个待验收的（游戏, 角色轨）。"""

    game: str
    track: str | None

    @property
    def label(self) -> str:
        return self.game if self.track is None else f"{self.game}[{self.track}]"

    @property
    def support_root(self) -> Path:
        """候选包目录。

        **所有游戏（含非对称的 rollman）都只有一个包。** 一次 run 里同一份候选快照
        要打完所有角色（``goal_led_service`` 是 ``for role in roles``），所以按轨发
        多个包没有意义；多套官方 SDK 放在包内的 ``<track>_sdk/`` 子目录里。
        """

        return GAMEPACKS / self.game / "candidate_support"

    @property
    def probe_path(self) -> Path:
        """最小合法 ``ai.py``。优先按轨取，没有就用该游戏通用的那份。

        rollman 用通用那份：它按 ``self.role`` 分支，正好顺带验证角色注入链路。
        """

        if self.track is not None:
            per_track = PROBES / f"{self.game}-{self.track}.py"
            if per_track.is_file():
                return per_track
        return PROBES / f"{self.game}.py"


def discover_targets(games: list[str] | None) -> list[Target]:
    """列出待验收目标。

    非对称游戏要**每个座位各验一局**——一个座位跑通不代表另一个也行（rollman 的
    吃豆人交 1 个方向、幽灵交 3 个，用错长度整局作废）。角色轨从候选包里的
    ``<track>_sdk/`` 子目录推断，与 ``gen_candidate_support.py`` 的约定一致，
    不需要额外维护一张表。
    """

    targets: list[Target] = []
    for pack in sorted(GAMEPACKS.iterdir()):
        if not pack.is_dir() or pack.name.startswith("_"):
            continue
        if games and pack.name not in games:
            continue
        support = pack / "candidate_support"
        if not support.is_dir():
            continue
        tracks = sorted(
            item.name.removesuffix("_sdk")
            for item in support.iterdir()
            if item.is_dir() and item.name.endswith("_sdk")
        )
        if tracks:
            targets.extend(Target(pack.name, track) for track in tracks)
        else:
            targets.append(Target(pack.name, None))
    return targets


def stage_candidate(target: Target, work_root: Path) -> Path:
    """装一个可运行的候选包：脚手架 + 最小合法 ai.py。"""

    candidate = work_root / "candidate"
    if candidate.exists():
        shutil.rmtree(candidate)
    shutil.copytree(target.support_root, candidate)
    if not target.probe_path.is_file():
        raise FileNotFoundError(
            f"{target.label}: 缺少验收探针 {target.probe_path}。"
            "每个（游戏, 轨）都要有一份最小合法 ai.py，否则无法证明脚手架可用。"
        )
    shutil.copy2(target.probe_path, candidate / "ai.py")
    return candidate


def pick_opponents(
    target: Target, roles: tuple[str, ...], *, limit: int
) -> tuple[list[PoolPlayer], str]:
    """挑若干个审计通过的人类对手（按排名），并决定候选该坐哪个座位。

    返回**列表**而不是单个：个别选手即使审计通过，换个座次/换个对手组合仍可能起不来
    （实测 snakego 的 rank01 在对手席上会被判 ``no runnable main.py``）。
    验收要检验的是**脚手架**，不该被某个对手的问题带崩，所以依次尝试。
    """

    supports_compiled = _supports_compiled_players(AGENTBENCH_ROOT, target.game)
    pool = load_pool(AGENTBENCH_ROOT, target.game, supports_compiled=supports_compiled)
    runnable = runnable_players(pool)
    if not runnable:
        raise RuntimeError(
            f"{target.game}: 没有审计通过的选手可当对手（players/runnable.json 为空？）"
        )

    def by_rank(players: list[PoolPlayer]) -> list[PoolPlayer]:
        return sorted(players, key=lambda p: (p.rank is None, p.rank or 0, p.player_id))

    if target.track is None:
        # 对称游戏：候选坐 roles[0]。
        return by_rank(list(runnable))[:limit], roles[0]

    # 非对称游戏：候选坐自己那一轨，对手必须是对位轨的人。
    other = opposing_track(target.track)
    if other is None:
        raise RuntimeError(f"{target.game}: 轨 {target.track} 没有对位轨定义")
    peers = players_in_track(runnable, other)
    if not peers:
        raise RuntimeError(
            f"{target.game}: 对位轨 {other} 没有审计通过的选手，无法为 {target.track} 找陪练"
        )
    if target.track not in roles:
        raise RuntimeError(f"{target.game}: 轨 {target.track} 不在 roles={roles} 里")
    return by_rank(list(peers))[:limit], target.track


def verify(
    target: Target,
    *,
    work_base: Path,
    timeout_s: float,
    cpus_per_match: int,
    opponent_attempts: int,
) -> dict:
    started = time.time()
    record: dict[str, object] = {"game": target.game, "track": target.track, "verified": False}
    work_root = work_base / target.label.replace("[", "-").replace("]", "")
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        roles = game_roles(AGENTBENCH_ROOT, target.game)
        candidate = stage_candidate(target, work_root)
        opponents, seat = pick_opponents(target, roles, limit=opponent_attempts)
    except Exception as error:  # noqa: BLE001 - 验收要如实记录任何失败
        record["diagnostic"] = f"{type(error).__name__}: {error}"
        record["elapsed_s"] = round(time.time() - started, 1)
        return record

    record["seat"] = seat
    attempts: list[dict[str, object]] = []

    for index, opponent in enumerate(opponents):
        attempt_root = work_root / f"attempt-{index}"
        attempt_root.mkdir(parents=True, exist_ok=True)

        def factory(request: IsolationRequest, _root: Path = attempt_root):
            return select_candidate_isolation(
                request, backend="auto", profile_path=_root / "verify.sb"
            )

        arena = ContractArena(
            game=target.game,
            agentbench_root=AGENTBENCH_ROOT,
            roles=roles,
            artifact_root=attempt_root / "matches",
            build_root=work_root / "build",
            isolation_factory=factory,
            opponents={opponent.player_id: opponent},
            timeout_s=timeout_s,
            cpus_per_match=cpus_per_match,
        )

        try:
            result = arena.run_case(
                MatchCase(CANDIDATE_ID, opponent.player_id, seat, 7), candidate
            )
        except Exception as error:  # noqa: BLE001
            attempts.append(
                {
                    "opponent_id": opponent.player_id,
                    "diagnostic": f"{type(error).__name__}: {error}",
                }
            )
            continue

        outcome = {
            "opponent_id": opponent.player_id,
            "status": result.status,
            "result": result.result,
            "rounds": result.rounds,
            "replay": str(result.replay_path) if result.replay_path else None,
            "returncodes": list(result.process_returncodes or ()),
            "diagnostic": result.error,
        }
        attempts.append(outcome)
        # 判据与池子审计一致：complete 且不是"第一帧就判负"。
        # rounds is None 属于"该游戏不上报回合数"，不能算失败。
        if result.status == "complete" and result.rounds != 0:
            record.update(outcome)
            record["verified"] = True
            break

    if not record["verified"] and attempts:
        record.update(attempts[-1])
        record["verified"] = False
    record["attempts"] = attempts
    record["elapsed_s"] = round(time.time() - started, 1)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", help="只验收指定游戏（可重复）")
    parser.add_argument("--parallel", type=int, default=1, help="并行验收数")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--cpus-per-match", type=int, default=4)
    parser.add_argument(
        "--opponent-attempts",
        type=int,
        default=3,
        help="最多尝试几个对手（个别对手自身起不来时不该判脚手架失败）",
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "artifacts" / "candidate-verify"),
        help="工作目录（对局产物落这里）",
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "artifacts" / "candidate_support_verification.json"),
    )
    args = parser.parse_args(argv)

    targets = discover_targets(args.game)
    if not targets:
        print("没有找到任何 candidate_support 目录，先跑 scripts/gen_candidate_support.py")
        return 1

    work_base = Path(args.work_root).resolve()
    work_base.mkdir(parents=True, exist_ok=True)
    print(f"验收 {len(targets)} 个目标：" + ", ".join(t.label for t in targets))

    def run(target: Target) -> dict:
        record = verify(
            target,
            work_base=work_base,
            timeout_s=args.timeout_s,
            cpus_per_match=args.cpus_per_match,
            opponent_attempts=args.opponent_attempts,
        )
        mark = "PASS" if record.get("verified") else "FAIL"
        detail = (
            f"status={record.get('status')} rounds={record.get('rounds')} "
            f"vs {record.get('opponent_id')} seat={record.get('seat')} "
            f"{record.get('elapsed_s')}s"
        )
        print(f"  [{mark}] {target.label}: {detail}")
        if not record.get("verified") and record.get("diagnostic"):
            print(f"         诊断：{record['diagnostic']}")
        return record

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            records = list(pool.map(run, targets))
    else:
        records = [run(target) for target in targets]

    passed = sum(1 for record in records if record.get("verified"))
    report = {
        "schema_version": "1.0",
        "generated_at": time.time(),
        "agentbench_root": str(AGENTBENCH_ROOT),
        "passed": passed,
        "total": len(records),
        "rows": records,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n{passed}/{len(records)} 通过；报告：{report_path}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
