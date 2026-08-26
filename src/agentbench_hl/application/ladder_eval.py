"""全池实测评分（ladder eval）—— 把 A 的选手池真正打一遍，反解 Elo。

## 为什么需要

A 的 ``manifest.tsv`` 里的 Elo 来自历史比赛，它：

- 只覆盖参加过评测的选手（很多提交没有分）；
- 未必与当前后端版本一致（后端修过 bug，历史分不可直接复用）；
- 与"实测可运行"无关（审计发现带 Elo 的选手也可能根本跑不起来）。

所以要在本机后端上重新打一遍，得到**可复现的实测评分**，供 HL/提交的表现分锚定。

## 为什么不打全循环赛

全循环赛是 O(n²)：antwar2 有数百个选手，成本不可接受且没有必要。稀疏配对 +
整图 BT 拟合可以用 O(n·k) 局得到自洽评分。配对用**循环图**（circulant）：

```
对第 i 个选手（按 id 排序），与 (i+d) mod n 配对，d = 1..ceil(degree/2)
```

- ``d=1`` 本身就是一条哈密顿环 ⇒ **整张图一定连通**（BT 可解的前提）；
- 每个选手的度数 ≈ degree，成本线性；
- 每对都打**两种座次**（side-balanced），消掉先后手优势。

## 可复现与可续跑

- 每局结果 append 到 ``matches.jsonl``（含 case 键）。重启后自动跳过已完成的 case，
  因此这个 job 可以随时中断/继续，不会重复烧机时。
- 配对与 seed 完全由 ``(sorted player ids, degree, seeds)`` 决定 ⇒ 同样输入得到同样赛程。

产出：``games/<game>/players/measured_elo.json``（写回 A，因为这是 A 选手池的事实）。
"""

from __future__ import annotations

import csv
import json
import math
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.adapters.contract.arena import ContractArena
from agentbench_hl.adapters.contract.pool import (
    PoolPlayer,
    load_pool,
    players_in_track,
    ranked_ladder,
    tracks_of,
)
from agentbench_hl.adapters.isolation import select_candidate_isolation
from agentbench_hl.application.pool_audit import load_verified_ids
from agentbench_hl.domain.rating import PairwiseRecord, RatingRow, fit_ratings
from agentbench_hl.ports.arena import MatchCase
from agentbench_hl.ports.isolation import IsolationRequest

MEASURED_FILENAME = "measured_elo.json"
# 口径层（contract/pool.apply_ladder_scope）读的 TSV：measured_rank + measured_elo。
RANKING_FILENAME = "measured_ranking.tsv"
RESULTS_FILENAME = "matches.jsonl"
SCHEMA_VERSION = "1.0"
SCOPES = ("verified", "ranked", "all")


@dataclass(frozen=True)
class LadderPlan:
    """一次实测赛程（确定性）。"""

    game: str
    players: tuple[str, ...]
    cases: tuple[tuple[str, str, str, int], ...]  # (player, opponent, role, seed)

    @property
    def total(self) -> int:
        return len(self.cases)


def pairing_offsets(degree: int) -> tuple[int, ...]:
    """把"每个选手大约打 degree 个对手"翻译成循环图偏移量。"""

    count = max(1, math.ceil(max(1, degree) / 2))
    return tuple(range(1, count + 1))


def build_plan(
    game: str,
    player_ids: Sequence[str],
    roles: Sequence[str],
    *,
    degree: int,
    seeds: Sequence[int],
) -> LadderPlan:
    players = tuple(sorted(set(player_ids)))
    size = len(players)
    if size < 2:
        return LadderPlan(game=game, players=players, cases=())
    cases: list[tuple[str, str, str, int]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for offset in pairing_offsets(degree):
        if offset >= size:
            break
        for index, player in enumerate(players):
            other = players[(index + offset) % size]
            if other == player:
                continue
            for seed in seeds:
                # 两种座次各一局：谁执 roles[0] 就作为 case 的 candidate。
                for first, second in ((player, other), (other, player)):
                    key = (first, second, roles[0], int(seed))
                    if key in seen:
                        continue
                    seen.add(key)
                    cases.append(key)
    return LadderPlan(game=game, players=players, cases=tuple(cases))


def build_bipartite_plan(
    game: str,
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    left_role: str,
    right_role: str,
    *,
    degree: int,
    seeds: Sequence[int],
) -> LadderPlan:
    """非对称游戏的赛程：**跨角色**配对（rollman × ghost）。

    为什么不能用对称赛程：rollman 的每份提交只实现一个角色，同轨两人根本没法对局
    （两边都演 ghost ⇒ 0 回合双方 -1000）。所以比较关系天然是**二分图**：
    pacman 只和 ghost 打。

    这带来一个统计后果：同轨两人的强弱只能**通过共同对手间接比较**。因此这里让
    每个选手打对位轨里 ``degree`` 个"错位采样"的对手（``(index + offset) % size``），
    保证二分图连通，BT 拟合才有唯一解（差一个整体平移，由锚点固定）。

    座次是**固定的**（pacman 坐 pacman 位），没有"两侧座次各打一局"这回事——
    这也是非对称游戏不需要消先后手偏差的原因。
    """

    left = tuple(sorted(set(left_ids)))
    right = tuple(sorted(set(right_ids)))
    if not left or not right:
        return LadderPlan(game=game, players=left + right, cases=())
    cases: list[tuple[str, str, str, int]] = []
    seen: set[tuple[str, str, str, int]] = set()
    # 每个选手打 degree 个对位对手；两侧都扫一遍，保证冷门选手也有足够样本。
    for source, target, role in ((left, right, left_role), (right, left, right_role)):
        span = min(max(1, degree), len(target))
        for index, player in enumerate(source):
            for offset in range(span):
                other = target[(index + offset) % len(target)]
                for seed in seeds:
                    key = (player, other, role, int(seed))
                    if key in seen:
                        continue
                    seen.add(key)
                    cases.append(key)
    return LadderPlan(game=game, players=left + right, cases=tuple(cases))


def _select_players(
    agentbench_root: Path,
    game: str,
    pool: Sequence[PoolPlayer],
    scope: str,
) -> tuple[tuple[str, ...], str]:
    """按 scope 选出参赛选手，并返回一句可审计的口径说明。"""

    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    runnable = [item for item in pool if item.runnable]
    if scope == "ranked":
        ladder = list(ranked_ladder(pool))
        return tuple(item.player_id for item in ladder), "A 的 manifest 里有 rank 的可运行选手"
    if scope == "all":
        return tuple(item.player_id for item in runnable), "A 池中所有形式上可运行的选手"
    verified = load_verified_ids(agentbench_root, game)
    if verified is None:
        # 把未审计的池子直接拿去拟合，会让"跑不起来"的选手变成免费失分对手，
        # 整张榜都会被系统性拉高，因此这里拒绝而不是静默降级。
        raise ValueError(
            f"{game} 还没有审计结果（players/runnable.json）；先跑 `abhl pool audit {game} --all`，"
            "或显式使用 --scope ranked/all（会把跑不起来的选手也算进评分，口径变差）"
        )
    ids = tuple(item.player_id for item in runnable if item.player_id in verified)
    return ids, "self-play smoke 审计通过的选手"


def _arena(
    game: str,
    agentbench_root: Path,
    work_root: Path,
    players: dict[str, PoolPlayer],
    roles: Sequence[str],
    *,
    isolation_backend: str,
    timeout_s: float,
    cpus_per_match: int,
) -> ContractArena:
    def factory(request: IsolationRequest):
        return select_candidate_isolation(
            request,
            backend=isolation_backend,
            profile_path=work_root / "isolation" / "ladder.sb",
        )

    return ContractArena(
        game=game,
        agentbench_root=agentbench_root,
        roles=tuple(roles),
        artifact_root=work_root / "matches",
        build_root=work_root / "build",
        isolation_factory=factory,
        opponents=players,
        timeout_s=timeout_s,
        cpus_per_match=cpus_per_match,
    )


def _load_done(path: Path) -> dict[tuple[str, str, str, int], dict[str, object]]:
    if not path.is_file():
        return {}
    done: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            key = (
                str(row["player"]),
                str(row["opponent"]),
                str(row["role"]),
                int(row["seed"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        done[key] = row
    return done


def run_ladder(
    game: str,
    agentbench_root: str | Path,
    *,
    work_root: str | Path,
    scope: str = "verified",
    degree: int = 6,
    seeds: Sequence[int] = (7,),
    parallel: int = 4,
    cpus_per_match: int = 3,
    timeout_s: float = 900.0,
    isolation_backend: str = "auto",
    prior_matches: float = 1.0,
    write: bool = True,
    on_match: Callable[[dict[str, object], int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """跑（或续跑）一次全池实测评分，返回汇总文档。"""

    from agentbench_hl.adapters.contract.factory import (
        _supported_player_build_systems,
        game_roles,
    )

    root = Path(agentbench_root).resolve()
    work = Path(work_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    roles = game_roles(root, game)
    pool = load_pool(
        root,
        game,
        supported_build_systems=_supported_player_build_systems(root, game),
    )
    by_id = {item.player_id: item for item in pool}
    player_ids, scope_note = _select_players(root, game, pool, scope)
    # 非对称游戏（rollman）：比较关系是二分图，只能跨角色配对。
    pool_tracks = tracks_of(pool)
    selected = set(player_ids)
    if len(pool_tracks) == 2:
        left_role, right_role = pool_tracks[0], pool_tracks[1]
        left = [
            item.player_id
            for item in players_in_track(pool, left_role)
            if item.player_id in selected
        ]
        right = [
            item.player_id
            for item in players_in_track(pool, right_role)
            if item.player_id in selected
        ]
        plan = build_bipartite_plan(
            game, left, right, left_role, right_role, degree=degree, seeds=seeds
        )
        scope_note = (
            f"{scope_note}；非对称游戏按角色分轨跨轨配对"
            f"（{left_role} {len(left)} 人 × {right_role} {len(right)} 人）"
        )
    else:
        plan = build_plan(game, player_ids, roles, degree=degree, seeds=seeds)
    results_path = work / RESULTS_FILENAME
    done = _load_done(results_path)
    pending = [case for case in plan.cases if case not in done]

    arena = _arena(
        game,
        root,
        work,
        {pid: by_id[pid] for pid in plan.players if pid in by_id},
        roles,
        isolation_backend=isolation_backend,
        timeout_s=timeout_s,
        cpus_per_match=cpus_per_match,
    )
    write_lock = threading.Lock()
    started = time.time()
    finished = len(done)

    def play(case: tuple[str, str, str, int]) -> dict[str, object]:
        player, opponent, role, seed = case
        candidate = by_id[player]
        try:
            result = arena.run_case(MatchCase(player, opponent, role, seed), candidate.package_root)
            row: dict[str, object] = {
                "player": player,
                "opponent": opponent,
                "role": role,
                "seed": seed,
                "status": result.status,
                "result": result.result,
                "points": result.points,
                "rounds": result.rounds,
                "error": result.error,
            }
        except Exception as error:  # noqa: BLE001 - 任何失败都如实记为基础设施故障
            row = {
                "player": player,
                "opponent": opponent,
                "role": role,
                "seed": seed,
                "status": "incomplete",
                "result": None,
                "points": None,
                "rounds": None,
                "error": f"{type(error).__name__}: {error}",
            }
        return row

    if pending:
        # 先串行预热一局：A 的后端首次编译有 cache/quarantine 竞态。
        try:
            arena.warmup(by_id[plan.players[0]].package_root)
        except Exception:  # noqa: BLE001 - 预热失败交给正式对局报诊断
            pass

    def consume(row: dict[str, object]) -> None:
        nonlocal finished
        with write_lock:
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            done[
                (str(row["player"]), str(row["opponent"]), str(row["role"]), int(row["seed"]))  # type: ignore[arg-type]
            ] = row
            finished += 1
            if on_match is not None:
                on_match(row, finished, plan.total)

    if parallel > 1 and len(pending) > 1:
        # 用工作线程 + 游标（而不是一次性 submit 全部）：``should_stop`` 才能在
        # 每局边界及时生效，不会把几千局全部排进队列再逐个取消。
        cursor = 0
        cursor_lock = threading.Lock()

        def worker() -> None:
            nonlocal cursor
            while True:
                if should_stop is not None and should_stop():
                    return
                with cursor_lock:
                    if cursor >= len(pending):
                        return
                    case = pending[cursor]
                    cursor += 1
                consume(play(case))

        threads = [
            threading.Thread(target=worker, name=f"ladder-{index}", daemon=True)
            for index in range(min(parallel, len(pending)))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        for case in pending:
            if should_stop is not None and should_stop():
                break
            consume(play(case))

    document = summarize(
        game,
        done.values(),
        anchors={item.player_id: item.elo for item in pool if item.elo is not None},
        plan=plan,
        scope=scope,
        scope_note=scope_note,
        seeds=seeds,
        degree=degree,
        prior_matches=prior_matches,
        elapsed_s=time.time() - started,
        tracks={item.player_id: item.track for item in pool},
    )
    if write:
        target = root / "games" / game / "players" / MEASURED_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        document["written_to"] = str(target)
        # 同时落一份**排名口径**的 TSV：``contract/pool.apply_ladder_scope`` 只读
        # measured_rank / measured_elo 两列，用它把实测强度接进对手选择策略。
        # 两份文件同源同时写，避免"JSON 里有分数、口径层读不到"的契约错位。
        ranking = write_measured_ranking(root, game, document)
        document["ranking_written_to"] = str(ranking)
    return document


def write_measured_ranking(
    agentbench_root: str | Path, game: str, document: Mapping[str, object]
) -> Path:
    """把实测评分导出成 ``measured_ranking.tsv``（名次 = 按实测 Elo 降序）。

    只导出**拟合成功**的选手（`measured_elo` 非空）：饱和（全胜/全负）与不连通的
    选手没有可比强度，硬给一个名次会污染课程顺序。
    """

    rows = [
        row
        for row in (document.get("ratings") or [])
        if isinstance(row, Mapping)
        and isinstance(row.get("player_id"), str)
        and isinstance(row.get("measured_elo"), (int, float))
    ]
    rows.sort(key=lambda row: -float(row["measured_elo"]))  # type: ignore[arg-type]
    # 非对称游戏：名次**按轨各自从 1 开始**。把 pacman 和 ghost 混排出来的"第 3 名"
    # 没有意义（谁也不能同时打这两个角色），课程顺序也会错。
    per_track_rank: dict[str | None, int] = {}
    ranked: list[tuple[int, Mapping[str, object]]] = []
    for row in rows:
        track = row.get("track") if isinstance(row.get("track"), str) else None
        per_track_rank[track] = per_track_rank.get(track, 0) + 1
        ranked.append((per_track_rank[track], row))
    target = Path(agentbench_root).resolve() / "games" / game / "players" / RANKING_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "measured_rank",
                "player_id",
                "measured_elo",
                "matches",
                "winrate",
                "track",
                "note",
            ]
        )
        for index, row in ranked:
            writer.writerow(
                [
                    index,
                    row["player_id"],
                    round(float(row["measured_elo"]), 1),  # type: ignore[arg-type]
                    row.get("matches") or 0,
                    "" if row.get("winrate") is None else row.get("winrate"),
                    row.get("track") or "",
                    row.get("note") or "",
                ]
            )
    return target


def _failure_key(error: object) -> str:
    """把一条失败诊断归一化成"类别键"，用于聚类统计。

    诊断里普遍夹着一次性的路径、哈希、pid、player_id，直接按原文分组会得到
    几百个只出现一次的类别，等于没分类。这里把这些可变片段抹掉，
    让"同一个根因"的失败落到同一个桶里。
    """

    text = str(error or "unknown").strip()
    text = re.sub(r"/[\w./+@-]+", "<path>", text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<hash>", text)
    text = re.sub(r"\b\d+\b", "<n>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:160]


def summarize(
    game: str,
    rows: Sequence[dict[str, object]] | object,
    *,
    anchors: dict[str, float],
    plan: LadderPlan,
    scope: str,
    scope_note: str,
    seeds: Sequence[int],
    degree: int,
    prior_matches: float,
    elapsed_s: float,
    tracks: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    """把对局结果拟合成评分表并生成可审计文档。

    ``tracks`` 是"选手 → 角色天梯"的映射（非对称游戏才有）。传进来而不是在这里
    重新读池子，是为了让本函数保持纯函数、可单测。
    """

    records: list[PairwiseRecord] = []
    complete = 0
    infra = 0
    # 失败局必须能回答"为什么"。只报一个 infra_or_incomplete 计数时，
    # 曾经出现 miracle 344 局失败却无从下手的情况——根因全丢在日志里了。
    failures: Counter[str] = Counter()
    failure_samples: dict[str, dict[str, object]] = {}
    for row in rows:  # type: ignore[union-attr]
        if not isinstance(row, dict):
            continue
        if row.get("status") != "complete" or row.get("points") is None:
            infra += 1
            key = _failure_key(row.get("error"))
            failures[key] += 1
            failure_samples.setdefault(
                key,
                {
                    "player": row.get("player"),
                    "opponent": row.get("opponent"),
                    "role": row.get("role"),
                    "seed": row.get("seed"),
                    "status": row.get("status"),
                    "error": row.get("error"),
                },
            )
            continue
        complete += 1
        records.append(
            PairwiseRecord(
                player=str(row["player"]),
                opponent=str(row["opponent"]),
                points=float(row["points"]),  # type: ignore[arg-type]
            )
        )
    ratings: tuple[RatingRow, ...] = fit_ratings(
        records, anchors=anchors, prior_matches=prior_matches
    )
    rated = [row for row in ratings if row.elo is not None]
    residuals = [
        row.elo - row.anchor_elo  # type: ignore[operator]
        for row in rated
        if row.anchor_elo is not None
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "game": game,
        "generated_at": time.time(),
        "method": "sparse-circulant-pairing + bradley-terry MLE (Elo scale 400)",
        "scope": scope,
        "scope_note": scope_note,
        "degree": degree,
        "seeds": list(seeds),
        "prior_matches": prior_matches,
        "players": len(plan.players),
        "planned_matches": plan.total,
        "played_matches": complete,
        "infra_or_incomplete": infra,
        # 失败局按根因聚类：category -> {count, sample}。开跑前用它判断
        # "这批 Elo 能不能信"，而不是只看一个总数。
        "failure_digest": [
            {"category": key, "count": count, "sample": failure_samples.get(key)}
            for key, count in failures.most_common(12)
        ],
        "rated_players": len(rated),
        "elapsed_s": round(elapsed_s, 1),
        "anchor_alignment": {
            "anchored_players": sum(1 for row in rated if row.anchor_elo is not None),
            "mean_residual": (
                round(sum(residuals) / len(residuals), 3) if residuals else None
            ),
            "max_abs_residual": (round(max(abs(v) for v in residuals), 3) if residuals else None),
        },
        "ratings": [
            {
                "player_id": row.player_id,
                "measured_elo": row.elo,
                "anchor_elo": row.anchor_elo,
                "matches": row.matches,
                "points": row.points,
                "winrate": None if row.winrate is None else round(row.winrate, 4),
                "note": row.note,
                # 非对称游戏必须带轨：pacman 的 1800 与 ghost 的 1800 不是同一个东西，
                # 混在一张榜里排名是错的（二分图上只有跨轨胜负是可观测的）。
                "track": (tracks or {}).get(row.player_id),
            }
            for row in ratings
        ],
        "tracks": sorted({value for value in (tracks or {}).values() if value}),
    }


def load_measured(agentbench_root: str | Path, game: str) -> dict[str, object] | None:
    path = Path(agentbench_root) / "games" / game / "players" / MEASURED_FILENAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
