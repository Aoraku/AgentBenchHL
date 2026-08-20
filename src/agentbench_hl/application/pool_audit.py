"""选手池真实可用性审计 —— "文件存在"不等于"能打"。

**为什么必须有**：A 的选手池里有历史提交是**实际不可用**的。实测（有/无沙箱结果一致，
排除隔离副作用）：

- ``rank14__ChenYQ__YINN__v10``  → 每步超时（``player timed out``）
- ``rank20__Hqc__greedy__v21``   → 协议垃圾（``frame exceeds 64 MiB: 3801784559``）

如果把它们当对手，候选会凭空"赢"下不存在的对局；如果把它们的 Elo 当锚点，
表现分反解会被系统性带偏。所以对手池与 Elo 锚点都必须只用**验证过能打**的选手。

**判定方式**：让选手**自己和自己打一局**（self-play smoke）。这是最中立的判据——
不引入第三方选手的强弱与兼容性；只要 ``status=complete`` 且 ``rounds > 0`` 就算可用。

产出写到 A 仓（因为这是 A 选手池的事实）：``games/<game>/players/runnable.json``。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.adapters.contract.arena import ContractArena
from agentbench_hl.adapters.contract.pool import (
    PoolPlayer,
    load_pool,
    opposing_track,
    players_in_track,
    ranked_ladder,
    tracks_of,
)
from agentbench_hl.adapters.isolation import select_candidate_isolation
from agentbench_hl.ports.arena import MatchCase
from agentbench_hl.ports.isolation import IsolationRequest

RUNNABLE_FILENAME = "runnable.json"
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AuditRow:
    player_id: str
    rank: int | None
    elo: float | None
    verified: bool
    rounds: int | None
    elapsed_s: float
    diagnostic: str | None
    # 非对称游戏的角色天梯（rollman / ghost）；对称游戏为 None。
    track: str | None = None
    # 本次审计用的陪练（非对称游戏才有），用于复核"通过率是否被陪练强度带偏"。
    sparring_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "player_id": self.player_id,
            "rank": self.rank,
            "elo": self.elo,
            "verified": self.verified,
            "rounds": self.rounds,
            "elapsed_s": round(self.elapsed_s, 1),
            "diagnostic": self.diagnostic,
            "track": self.track,
            "sparring_id": self.sparring_id,
        }


def _arena_for(
    game: str,
    agentbench_root: Path,
    work_root: Path,
    player: PoolPlayer,
    roles: Sequence[str],
    *,
    isolation_backend: str,
    timeout_s: float,
    cpus_per_match: int,
    sparring: PoolPlayer | None = None,
) -> ContractArena:
    """构造只含"被审选手 + 陪练"的最小 arena。

    ``sparring`` 用于**非对称游戏**：rollman 的每份提交只实现一个角色，
    对称 self-play 会让两个座位都演同一个角色（实测：双方 0 回合、各 -1000、
    全池 0 通过）。这时必须放一个**对位角色**的陪练进去。
    """

    def factory(request: IsolationRequest):
        return select_candidate_isolation(
            request,
            backend=isolation_backend,
            profile_path=work_root / "audit.sb",
        )

    opponents = {player.player_id: player}
    if sparring is not None:
        opponents[sparring.player_id] = sparring

    return ContractArena(
        game=game,
        agentbench_root=agentbench_root,
        roles=tuple(roles),
        artifact_root=work_root / "matches",
        build_root=work_root / "build",
        isolation_factory=factory,
        opponents=opponents,
        timeout_s=timeout_s,
        cpus_per_match=cpus_per_match,
    )


def audit_pool(
    game: str,
    agentbench_root: str | Path,
    *,
    work_root: str | Path,
    ranked_only: bool = True,
    parallel: int = 4,
    seed: int = 7,
    attempts: int = 2,
    cpus_per_match: int = 4,
    timeout_s: float = 900.0,
    isolation_backend: str = "auto",
    write: bool = True,
    only: Sequence[str] | None = None,
    on_row: Callable[[AuditRow, int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """对选手池做 self-play smoke 审计，返回（并可写入）结果。

    ``attempts`` > 1 时任一次成功即判可用：每步超时判定本身有抖动（计算重的选手在
    复杂局面才会超限），单次判定会把"边缘可用"误杀成"不可用"。
    ``cpus_per_match`` 默认给 4 核（两个选手 + 后端 + 余量），self-play 两边都是同一份
    重计算程序，核太少会人为制造超时。

    ``only`` 限定要审计的 player_id 子集（补审/续跑）；``on_row`` 每审完一个选手回调
    一次（供服务端实时进度）；``should_stop`` 允许外部取消，已完成部分照常写盘。
    """

    from agentbench_hl.adapters.contract.factory import _supports_compiled_players, game_roles

    root = Path(agentbench_root).resolve()
    work = Path(work_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    roles = game_roles(root, game)
    # ★ 审计**不能**读自己上一次的结论：``load_pool`` 默认会用 runnable.json 把
    # "审过且失败"的选手降级为不可运行，如果审计再拿这个结果当输入，重审就只会
    # 重审幸存者，失败者永远没有翻案机会，池子随着每次审计单调缩小
    # （实测：antwar 第二次审计的候选从 193 掉到 103）。审计是**产生**结论的一方。
    players = load_pool(
        root,
        game,
        supports_compiled=_supports_compiled_players(root, game),
        apply_audit=False,
    )
    candidates = list(ranked_ladder(players)) if ranked_only else [p for p in players if p.runnable]
    if only is not None:
        wanted = {str(item) for item in only}
        candidates = [player for player in candidates if player.player_id in wanted]

    # 非对称游戏（rollman）：每份提交只实现一个角色，必须给它配一个**对位角色**的陪练。
    # 陪练取对位轨里 rank 最好的可运行选手，全程固定，这样"通过/不通过"的判据在
    # 整池范围内是同一个标准（换陪练会让通过率不可比）。
    sparring_by_track: dict[str, PoolPlayer] = {}
    for track in tracks_of(players):
        rival_track = opposing_track(track)
        if rival_track is None:
            continue
        rivals = players_in_track(players, rival_track)
        if rivals:
            sparring_by_track[track] = rivals[0]

    def sparring_for(player: PoolPlayer) -> PoolPlayer | None:
        if player.track is None:
            return None
        return sparring_by_track.get(player.track)

    def seat_for(player: PoolPlayer, attempt: int) -> str:
        """被审选手该坐哪个座位。

        对称游戏轮换座次（两侧都试，避免先后手偏差把边缘选手误杀）；
        非对称游戏**只能**坐自己实现的那个角色对应的座位。
        """

        if player.track is not None and player.track in roles:
            return player.track
        return roles[attempt % len(roles)]

    def check(player: PoolPlayer) -> AuditRow:
        started = time.time()
        last: AuditRow | None = None
        sparring = sparring_for(player)
        if player.track is not None and sparring is None:
            return AuditRow(
                player_id=player.player_id,
                rank=player.rank,
                elo=player.elo,
                verified=False,
                rounds=None,
                elapsed_s=0.0,
                diagnostic=(
                    f"角色 {player.track} 找不到对位陪练"
                    f"（{opposing_track(player.track)} 轨没有可运行选手）"
                ),
            )
        for attempt in range(max(1, attempts)):
            arena = _arena_for(
                game,
                root,
                work / player.player_id / f"attempt-{attempt}",
                player,
                roles,
                isolation_backend=isolation_backend,
                timeout_s=timeout_s,
                cpus_per_match=cpus_per_match,
                sparring=sparring,
            )
            try:
                result = arena.run_case(
                    MatchCase(
                        player.player_id,
                        # 对称游戏 self-play；非对称游戏打对位陪练。
                        sparring.player_id if sparring is not None else player.player_id,
                        seat_for(player, attempt),
                        seed + attempt,
                    ),
                    player.package_root,
                )
            except Exception as error:  # noqa: BLE001 - 审计要如实记录任何失败
                last = AuditRow(
                    player_id=player.player_id,
                    rank=player.rank,
                    elo=player.elo,
                    verified=False,
                    rounds=None,
                    elapsed_s=time.time() - started,
                    diagnostic=f"{type(error).__name__}: {error}",
                    track=player.track,
                    sparring_id=sparring.player_id if sparring else None,
                )
                continue
            evaluator_status = str(result.payload.get("evaluator_status") or result.status)
            rounds = result.rounds
            scores = result.payload.get("scores")
            # 判据：对战器说 complete，且有**打过的证据**。
            #
            # 这里不能写 `rounds > 0`：A 的部分游戏（rollman）压根不上报 rounds，
            # 于是整池 93 人全被判不可用（明明每局都跑完了）。反过来 `rounds == 0`
            # 是真实的失败信号（第一帧就判负），必须继续拦住。
            played_evidence = bool(rounds) or bool(scores) or result.result is not None
            verified = evaluator_status == "complete" and rounds != 0 and played_evidence
            row = AuditRow(
                player_id=player.player_id,
                rank=player.rank,
                elo=player.elo,
                verified=verified,
                rounds=rounds,
                elapsed_s=time.time() - started,
                diagnostic=(
                    None
                    if verified
                    # 编译诊断（g++/make 输出）动辄上千字符，截到 200 会把真正的
                    # 错误行切掉，只剩无用的开头几行；这里保留末尾 2000 字符
                    # （编译错误的信息量集中在末尾）。
                    else str(
                        result.payload.get("game_error") or result.error or evaluator_status
                    )[-2000:]
                ),
                track=player.track,
                sparring_id=sparring.player_id if sparring else None,
            )
            if verified:
                return row
            last = row
        return last or AuditRow(
            player_id=player.player_id,
            rank=player.rank,
            elo=player.elo,
            verified=False,
            rounds=None,
            elapsed_s=time.time() - started,
            diagnostic="no attempt produced a result",
            track=player.track,
            sparring_id=sparring.player_id if sparring else None,
        )

    rows: list[AuditRow] = []
    total = len(candidates)
    progress_lock = threading.Lock()

    def checked(player: PoolPlayer) -> AuditRow | None:
        if should_stop is not None and should_stop():
            return None
        row = check(player)
        with progress_lock:
            rows.append(row)
            if on_row is not None:
                on_row(row, len(rows), total)
        return row

    if parallel > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            list(pool.map(checked, candidates))
    else:
        for player in candidates:
            checked(player)

    verified = [row for row in rows if row.verified]
    stopped = bool(should_stop is not None and should_stop())
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "game": game,
        "generated_at": time.time(),
        "method": "self_play_smoke",
        "seed": seed,
        "attempts": attempts,
        "cpus_per_match": cpus_per_match,
        "scope": "ranked" if ranked_only else "all_runnable",
        # 被取消/中断时如实标记：下游（实测评分口径）需要知道这份结论不完整。
        "partial": stopped or len(rows) < total,
        "candidates": total,
        "checked": len(rows),
        "verified": len(verified),
        "verified_ids": sorted(row.player_id for row in verified),
        "rows": [row.as_dict() for row in rows],
    }
    if write:
        target = Path(root) / "games" / game / "players" / RUNNABLE_FILENAME
        existing: dict[str, object] = {}
        if target.is_file():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        # 增量合并：不同 scope 的审计结果互不覆盖对方已验证的条目。
        merged_rows = {
            str(row["player_id"]): row
            for row in (existing.get("rows") or [])
            if isinstance(row, dict) and row.get("player_id")
        }
        merged_rows.update({row.player_id: row.as_dict() for row in rows})
        document["rows"] = [merged_rows[key] for key in sorted(merged_rows)]
        document["verified_ids"] = sorted(
            key for key, row in merged_rows.items() if row.get("verified")
        )
        document["checked"] = len(merged_rows)
        document["verified"] = len(document["verified_ids"])  # type: ignore[arg-type]
        target.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        document["written_to"] = str(target)
    return document


def load_verified_ids(agentbench_root: str | Path, game: str) -> frozenset[str] | None:
    """读取审计结果；没有审计文件时返回 None（表示"未审计"，不是"全不可用"）。"""

    target = Path(agentbench_root) / "games" / game / "players" / RUNNABLE_FILENAME
    if not target.is_file():
        return None
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    ids = document.get("verified_ids")
    if not isinstance(ids, list):
        return None
    return frozenset(str(item) for item in ids)
