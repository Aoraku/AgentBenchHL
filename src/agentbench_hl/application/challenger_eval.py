"""挑战者 × **冻结人类池** 的实测评分：这一版策略在人类棋手里排第几。

与 ``pool_elo`` 的区别（两个不同的量，别混用）
--------------------------------------------
``IterationMetricsFinalized.pool_elo`` 用的是该 run **迄今全部** conquest 对局：
它不会遗忘第 1 轮的连败，也混合了同一轮里被证伪的探索候选，而且每个候选只有
``roles × seeds`` 局（实测 2 局）。它衡量的是"这条研究轨迹整体的平均位置"，
适合画学习曲线；用它回答"我们现在这版有多强"是错的口径——2 局样本下
全胜/全败会直接被正则先验顶到固定值，完全没有分辨力。

本模块做的是另一件事：**取定一版策略，与冻结人类池里每个有锚点的选手各打两个
座次，只用这一批干净样本反解 Elo**，并给出"插进人类榜会排第几"。

"冻结"是什么意思（这是本模块最重要的约束）
------------------------------------------
1. 锚点只**读** ``players/measured_elo.json``，**永不回写**。挑战者的胜负不参与
   人类选手之间的评分拟合——否则我们自己的策略会反过来改变尺子，
   "在人类池里排第几"这句话就失去意义（尺子随被测对象漂移）。
2. 每份结果都带 ``pool_fingerprint``（锚点集合的哈希）。池子一旦重测/扩容，
   指纹就变，旧结果会被显式标成不可比，而不是静默混在一起。
3. 参赛对手清单由 ``scope`` + 锚点可得性确定性地导出，同样输入同样赛程。

为什么必须两个座次
------------------
对称游戏里先后手有系统性优势（antwar 的地形与出兵顺序都不对称）。只打一个座次
会把座次优势算进策略强度里。所以每个对手固定打 P0/P1 各一局。

为什么打全池而不是采样
----------------------
成本可接受（antwar 94 人 × 2 座次 ≈ 188 局），而采样会引入"抽到的对手偏强/偏弱"
的方差——那正是我们要消掉的东西。

可续跑
------
每局 append 到 ``matches.jsonl``，重启时按 ``(player, opponent, role, seed)``
跳过已完成的 case。这个 job 本来就该在后台断断续续地跑：
**不占迭代的机时预算，也不参与迭代决策**。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.adapters.contract.pool import load_pool
from agentbench_hl.application.ladder_eval import _arena, _load_done, _select_players
from agentbench_hl.domain.pool_elo import estimate_pool_elo
from agentbench_hl.ports.arena import MatchCase

__all__ = [
    "FrozenPool",
    "evaluate_challenger",
    "evaluate_many",
    "load_challenger_elo",
    "load_frozen_pool",
]

RESULTS_FILENAME = "matches.jsonl"
SUMMARY_FILENAME = "challenger-elo.json"
SCHEMA = "challenger-elo/v1"

#: result → BT 记分。draw 记 0.5，与 ``estimate_pool_elo`` 的口径一致。
POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}


class FrozenPool:
    """冻结的人类池：一组 ``player_id → Elo`` 锚点，附带指纹与名次查询。"""

    def __init__(self, game: str, anchors: Mapping[str, float], source: Path) -> None:
        if not anchors:
            raise ValueError(f"{game} 的人类池没有任何 Elo 锚点，无法衡量排名")
        self.game = game
        self.anchors = dict(anchors)
        self.source = source
        self._sorted = sorted(self.anchors.values(), reverse=True)

    @property
    def size(self) -> int:
        return len(self.anchors)

    @property
    def fingerprint(self) -> str:
        """锚点集合的指纹：池子变了就不可比，必须能被检测出来。"""

        digest = hashlib.sha256()
        for player_id in sorted(self.anchors):
            digest.update(f"{player_id}:{self.anchors[player_id]:.4f}\n".encode())
        return digest.hexdigest()[:16]

    @property
    def top_elo(self) -> float:
        return self._sorted[0]

    def rank_of(self, elo: float) -> int:
        """这个 Elo 插进人类池后排第几（1 = 榜首）。"""

        return sum(1 for value in self._sorted if value > elo) + 1

    def summary(self) -> dict[str, object]:
        return {
            "game": self.game,
            "size": self.size,
            "fingerprint": self.fingerprint,
            "elo_min": round(self._sorted[-1], 2),
            "elo_max": round(self._sorted[0], 2),
            "source": str(self.source),
            "frozen": True,
        }


def load_frozen_pool(agentbench_root: str | Path, game: str) -> FrozenPool:
    """读取（只读）人类池实测 Elo 作为冻结锚点。"""

    root = Path(agentbench_root).resolve()
    path = root / "games" / game / "players" / "measured_elo.json"
    if not path.is_file():
        raise ValueError(
            f"{game} 还没有实测人类池评分（{path}）；先跑 `abhl ladder eval {game}`"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    anchors: dict[str, float] = {}
    for item in document.get("ratings") or []:
        if not isinstance(item, Mapping):
            continue
        value = item.get("measured_elo")
        player_id = item.get("player_id")
        if value is None or player_id is None:
            continue
        anchors[str(player_id)] = float(value)
    return FrozenPool(game, anchors, path)


def evaluate_challenger(
    game: str,
    agentbench_root: str | Path,
    challenger_id: str,
    challenger_root: str | Path,
    *,
    work_root: str | Path,
    pool: FrozenPool | None = None,
    scope: str = "verified",
    seeds: Sequence[int] = (7,),
    parallel: int = 4,
    cpus_per_match: int = 3,
    timeout_s: float = 1800.0,
    isolation_backend: str = "auto",
    challenger_track: str | None = None,
    on_match: Callable[[dict[str, object], int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    before_match: Callable[[], None] | None = None,
) -> dict[str, object]:
    """让 ``challenger_root`` 这一版策略打完冻结池，返回带 Elo 与名次的汇总。

    ``before_match`` 是给后台队列用的节流钩子：在每局**派发之前**调用，
    可以在里面等到 CPU 空闲再返回，从而不与主迭代抢核。
    """

    from agentbench_hl.adapters.contract.factory import _supports_compiled_players, game_roles

    root = Path(agentbench_root).resolve()
    work = Path(work_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    candidate_root = Path(challenger_root).resolve()
    if not (candidate_root / "main.py").is_file():
        raise ValueError(f"挑战者包缺少 main.py: {candidate_root}")

    frozen = pool if pool is not None else load_frozen_pool(root, game)
    roles = tuple(game_roles(root, game))
    players = load_pool(root, game, supports_compiled=_supports_compiled_players(root, game))
    by_id = {item.player_id: item for item in players}
    tracks = {item.track for item in players if item.track}
    if len(tracks) == 2:
        # 非对称游戏的比较关系是二分图，挑战者只能打对位轨；见 evaluate_many 里的
        # 同名处理。拿不到角色轨时拒绝，而不是算出一个错的 Elo。
        if challenger_track is None:
            raise ValueError(
                f"{game} 是非对称（分轨）游戏，挑战者评测需要显式指定角色轨"
                f"（challenger_track，可选 {sorted(tracks)}）"
            )
        if challenger_track not in tracks:
            raise ValueError(
                f"{game} 的角色轨只有 {sorted(tracks)}，收到 {challenger_track!r}"
            )
        opponent_track = next(item for item in tracks if item != challenger_track)
        players = [item for item in players if item.track == opponent_track]
        by_id = {item.player_id: item for item in players}
        if not players:
            raise ValueError(f"{game} 的 {opponent_track} 轨没有可用选手")

    eligible, scope_note = _select_players(root, game, players, scope)
    # 没有锚点的对手会被 estimate_pool_elo 跳过，这里就不要浪费机时去打。
    opponents = tuple(pid for pid in eligible if pid in frozen.anchors and pid in by_id)
    if not opponents:
        raise ValueError(f"{game} 的池选手没有可用锚点，无法反解挑战者 Elo")

    cases = [
        (challenger_id, opponent, role, int(seed))
        for opponent in opponents
        for role in roles
        for seed in seeds
    ]
    results_path = work / RESULTS_FILENAME
    done = _load_done(results_path)
    pending = [case for case in cases if case not in done]

    arena = _arena(
        game,
        root,
        work,
        {pid: by_id[pid] for pid in opponents},
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
        try:
            outcome = arena.run_case(MatchCase(player, opponent, role, seed), candidate_root)
            return {
                "player": player,
                "opponent": opponent,
                "role": role,
                "seed": seed,
                "status": outcome.status,
                "result": outcome.result,
                "points": outcome.points,
                "rounds": outcome.rounds,
                "error": outcome.error,
            }
        except Exception as error:  # noqa: BLE001 - 单局失败不该终止整轮评测
            return {
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

    def consume(row: dict[str, object]) -> None:
        nonlocal finished
        with write_lock:
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            done[
                (str(row["player"]), str(row["opponent"]), str(row["role"]), int(row["seed"]))
            ] = row
            finished += 1
            if on_match is not None:
                on_match(row, finished, len(cases))

    if pending:
        # 先串行预热一局：A 的后端首次编译有 cache/quarantine 竞态，
        # 不预热的话前几局会因为编译撞车白白失败。
        try:
            arena.warmup(candidate_root)
        except Exception:  # noqa: BLE001 - 预热失败交给正式对局报诊断
            pass

    # 用工作线程 + 共享游标（而不是 executor.map）：``should_stop`` 与
    # ``before_match`` 才能在**每局边界**生效。executor.map 会预取任务，
    # 让"CPU 忙就先别派发"这件事失效。
    cursor = 0
    cursor_lock = threading.Lock()
    stopped = False

    def worker() -> None:
        nonlocal cursor, stopped
        while True:
            if should_stop is not None and should_stop():
                stopped = True
                return
            if before_match is not None:
                before_match()
            with cursor_lock:
                if cursor >= len(pending):
                    return
                case = pending[cursor]
                cursor += 1
            consume(play(case))

    if pending:
        threads = [
            threading.Thread(target=worker, name=f"challenger-{index}", daemon=True)
            for index in range(max(1, min(parallel, len(pending))))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    rows = [
        {"opponent_id": str(row["opponent"]), "points": POINTS[str(row["result"])]}
        for row in done.values()
        if row.get("status") == "complete" and str(row.get("result")) in POINTS
    ]
    estimate = estimate_pool_elo(rows, frozen.anchors)
    complete = len(rows)
    wins = sum(1 for row in rows if row["points"] == 1.0)
    draws = sum(1 for row in rows if row["points"] == 0.5)
    beaten = sorted(
        {
            str(row["opponent"])
            for row in done.values()
            if row.get("status") == "complete" and row.get("result") == "win"
        }
    )

    document: dict[str, object] = {
        "schema": SCHEMA,
        "game": game,
        "challenger_id": challenger_id,
        "challenger_root": str(candidate_root),
        "pool": frozen.summary(),
        "pool_fingerprint": frozen.fingerprint,
        "challenger_track": challenger_track,
        "scope": scope,
        "scope_note": scope_note,
        "seeds": [int(item) for item in seeds],
        "roles": list(roles),
        "opponents": len(opponents),
        "planned_matches": len(cases),
        "complete_matches": complete,
        "failed_matches": len(done) - complete,
        "pending_matches": len(cases) - len(done),
        "partial": bool(stopped or len(done) < len(cases)),
        "wins": wins,
        "draws": draws,
        "losses": complete - wins - draws,
        "beaten_opponents": beaten,
        "elo": None if estimate is None else round(estimate.elo, 2),
        "pool_rank": None if estimate is None else frozen.rank_of(estimate.elo),
        "elo_gap_to_top": (
            None if estimate is None else round(frozen.top_elo - estimate.elo, 2)
        ),
        "score_rate": None if estimate is None else round(estimate.score_rate, 4),
        "anchored_matches": 0 if estimate is None else estimate.anchored_matches,
        "elapsed_s": round(time.time() - started, 1),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # 口径声明：这份 Elo 只用本次全池对局，人类池是冻结的、没有被回写。
        "provenance": (
            "challenger vs frozen human pool, both seats; anchors read-only from "
            "players/measured_elo.json (never rewritten); independent of the run's "
            "conquest matches and of other candidates"
        ),
    }
    (work / SUMMARY_FILENAME).write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def evaluate_many(
    game: str,
    agentbench_root: str | Path,
    challengers: Sequence[tuple[str, Path]],
    *,
    queue_root: str | Path,
    pool: FrozenPool | None = None,
    scope: str = "verified",
    seeds: Sequence[int] = (7,),
    parallel: int = 24,
    cpus_per_match: int = 3,
    timeout_s: float = 1800.0,
    isolation_backend: str = "auto",
    challenger_track: str | None = None,
    on_match: Callable[[str, int, int], None] | None = None,
    on_version_done: Callable[[dict[str, object]], None] | None = None,
    before_match: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[dict[str, object]]:
    """一次评测**多个**版本，用一个全局扁平队列把所有核吃满。

    为什么必须批量而不是逐版本串行
    ------------------------------
    ``arena.run_case(case, candidate_root)`` 的候选包是**逐次传入**的，
    所以同一个 arena 天然可以服务所有版本。逐版本串行会造成两个浪费：

    1. **队尾饥饿**：一个版本剩最后 3 局时只有 3 个线程在干活，其余核空转。
       版本越多，这种"尾巴"累积得越多。
    2. **并发上限被版本内的局数卡住**：188 局的版本最多也就并行 188 路，
       但真正的上限应该是 ``总核数 / 每局核数``。

    摊平成 (版本 × 对手 × 座次 × seed) 的全局队列后，任何时刻都有足够多的
    待办项填满线程池，直到全部跑完。32 核 / 每局 3 核 ⇒ 约 10 路并发，
    实测比逐版本串行快一个数量级。

    结果按版本分别落盘（每版一个 ``matches.jsonl`` + ``challenger-elo.json``），
    所以断点续跑、指纹校验这些语义都不变。
    """

    from agentbench_hl.adapters.contract.factory import _supports_compiled_players, game_roles

    root = Path(agentbench_root).resolve()
    queue = Path(queue_root).resolve()
    queue.mkdir(parents=True, exist_ok=True)
    frozen = pool if pool is not None else load_frozen_pool(root, game)
    roles = tuple(game_roles(root, game))
    players = load_pool(root, game, supports_compiled=_supports_compiled_players(root, game))
    by_id = {item.player_id: item for item in players}
    tracks = {item.track for item in players if item.track}
    if len(tracks) == 2:
        # 非对称游戏的比较关系是二分图：rollman 只和 ghost 交手，同轨之间没有对局，
        # 所以"挑战者打全池"这句话必须先说清它坐哪一边。
        #
        # ``challenger_track`` 说的是**挑战者自己扮演的角色**，它要打的是
        # **对位轨**的选手。传错方向会让挑战者和同轨选手互殴——那种对局在
        # 人类池里从未发生过，锚点 Elo 不适用，算出来的分数是无意义的。
        if challenger_track is None:
            raise ValueError(
                f"{game} 是非对称（分轨）游戏，挑战者评测需要显式指定角色轨"
                f"（--challenger-track，可选 {sorted(tracks)}）"
            )
        if challenger_track not in tracks:
            raise ValueError(
                f"{game} 的角色轨只有 {sorted(tracks)}，收到 {challenger_track!r}"
            )
        opponent_track = next(item for item in tracks if item != challenger_track)
        players = [item for item in players if item.track == opponent_track]
        by_id = {item.player_id: item for item in players}
        if not players:
            raise ValueError(f"{game} 的 {opponent_track} 轨没有可用选手")

    eligible, scope_note = _select_players(root, game, players, scope)
    opponents = tuple(pid for pid in eligible if pid in frozen.anchors and pid in by_id)
    if not opponents:
        raise ValueError(f"{game} 的池选手没有可用锚点，无法反解挑战者 Elo")

    # 一个 arena 服务所有版本：候选包按 run_case 的参数逐次传入。
    # work_root 用队列根目录，编译产物在所有版本之间共享（对手池的后端是同一份）。
    arena = _arena(
        game,
        root,
        queue,
        {pid: by_id[pid] for pid in opponents},
        roles,
        isolation_backend=isolation_backend,
        timeout_s=timeout_s,
        cpus_per_match=cpus_per_match,
    )

    @dataclass
    class _Slot:
        challenger_id: str
        candidate_root: Path
        work_root: Path
        results_path: Path
        done: dict[tuple[str, str, str, int], dict[str, object]]
        planned: int
        lock: threading.Lock

    slots: dict[str, _Slot] = {}
    queue_items: list[tuple[str, str, str, int]] = []
    for challenger_id, candidate_root in challengers:
        candidate_root = Path(candidate_root).resolve()
        if not (candidate_root / "main.py").is_file():
            continue
        work_root = queue / challenger_id
        work_root.mkdir(parents=True, exist_ok=True)
        results_path = work_root / RESULTS_FILENAME
        done = _load_done(results_path)
        cases = [
            (challenger_id, opponent, role, int(seed))
            for opponent in opponents
            for role in roles
            for seed in seeds
        ]
        slots[challenger_id] = _Slot(
            challenger_id=challenger_id,
            candidate_root=candidate_root,
            work_root=work_root,
            results_path=results_path,
            done=done,
            planned=len(cases),
            lock=threading.Lock(),
        )
        queue_items.extend(case for case in cases if case not in done)

    if not slots:
        return []

    # 预热：A 的后端首次编译有 cache/quarantine 竞态，不预热的话开局几十局会
    # 因为编译撞车集体失败。每个版本各预热一次（候选包不同、编译产物不同）。
    for slot in slots.values():
        if any(case[0] == slot.challenger_id for case in queue_items):
            try:
                arena.warmup(slot.candidate_root)
            except Exception:  # noqa: BLE001 - 预热失败交给正式对局报诊断
                pass

    started = time.time()
    cursor = 0
    cursor_lock = threading.Lock()
    finished = 0
    stopped = False

    # 每个版本还剩多少局没跑。归零时立刻落盘它的 challenger-elo.json，
    # 不等整批结束。
    #
    # 为什么必须这样：一批 172 个版本 × 188 局 = 32k 局要跑二十几小时，
    # 若只在整批结束时落盘，中途看不到**任何**完成版本，也无法出图；
    # worker 被重启还得从头再来。逐版本落盘让"跑完的就能用"。
    remaining: dict[str, int] = {}
    for case in queue_items:
        remaining[case[0]] = remaining.get(case[0], 0) + 1

    def flush(slot: _Slot) -> None:
        document = _summarise(
            game=game,
            challenger_id=slot.challenger_id,
            candidate_root=slot.candidate_root,
            frozen=frozen,
            scope=scope,
            scope_note=scope_note,
            seeds=seeds,
            roles=roles,
            opponents=opponents,
            done=slot.done,
            planned=slot.planned,
            stopped=False,
            elapsed=time.time() - started,
            challenger_track=challenger_track,
        )
        (slot.work_root / SUMMARY_FILENAME).write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if on_version_done is not None:
            on_version_done(document)

    def record(slot: _Slot, row: dict[str, object]) -> None:
        nonlocal finished
        with slot.lock:
            with slot.results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            slot.done[
                (str(row["player"]), str(row["opponent"]), str(row["role"]), int(row["seed"]))
            ] = row
        with cursor_lock:
            finished += 1
            current = finished
            remaining[slot.challenger_id] -= 1
            completed_now = remaining[slot.challenger_id] == 0
        if completed_now:
            flush(slot)
        if on_match is not None:
            on_match(slot.challenger_id, current, len(queue_items))

    def worker() -> None:
        nonlocal cursor, stopped
        while True:
            if should_stop is not None and should_stop():
                stopped = True
                return
            if before_match is not None:
                before_match()
            with cursor_lock:
                if cursor >= len(queue_items):
                    return
                case = queue_items[cursor]
                cursor += 1
            slot = slots[case[0]]
            record(slot, _play_case(arena, slot.candidate_root, case))

    if queue_items:
        threads = [
            threading.Thread(target=worker, name=f"pool-elo-{index}", daemon=True)
            for index in range(max(1, min(parallel, len(queue_items))))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    documents: list[dict[str, object]] = []
    for slot in slots.values():
        # ``stopped`` 是**整批**被中断的标志，但不能让它连坐已经跑满的版本：
        # 一个版本的 188 局全部完成，它的结论就是完整的，与"队列里别的版本
        # 还没跑"无关。只有该版本自己还缺局时才标 partial。
        slot_incomplete = len(slot.done) < slot.planned
        document = _summarise(
            game=game,
            challenger_id=slot.challenger_id,
            candidate_root=slot.candidate_root,
            frozen=frozen,
            scope=scope,
            scope_note=scope_note,
            seeds=seeds,
            roles=roles,
            opponents=opponents,
            done=slot.done,
            planned=slot.planned,
            stopped=stopped and slot_incomplete,
            elapsed=time.time() - started,
            challenger_track=challenger_track,
        )
        (slot.work_root / SUMMARY_FILENAME).write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        documents.append(document)
    return documents


def _play_case(
    arena: object, candidate_root: Path, case: tuple[str, str, str, int]
) -> dict[str, object]:
    player, opponent, role, seed = case
    try:
        outcome = arena.run_case(MatchCase(player, opponent, role, seed), candidate_root)  # type: ignore[attr-defined]
        return {
            "player": player,
            "opponent": opponent,
            "role": role,
            "seed": seed,
            "status": outcome.status,
            "result": outcome.result,
            "points": outcome.points,
            "rounds": outcome.rounds,
            "error": outcome.error,
        }
    except Exception as error:  # noqa: BLE001 - 单局失败不该终止整批评测
        return {
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


def _summarise(
    *,
    game: str,
    challenger_id: str,
    candidate_root: Path,
    frozen: FrozenPool,
    scope: str,
    scope_note: str,
    seeds: Sequence[int],
    roles: Sequence[str],
    opponents: Sequence[str],
    done: Mapping[tuple[str, str, str, int], dict[str, object]],
    planned: int,
    stopped: bool,
    elapsed: float,
    challenger_track: str | None = None,
) -> dict[str, object]:
    rows = [
        {"opponent_id": str(row["opponent"]), "points": POINTS[str(row["result"])]}
        for row in done.values()
        if row.get("status") == "complete" and str(row.get("result")) in POINTS
    ]
    estimate = estimate_pool_elo(rows, frozen.anchors)
    complete = len(rows)
    wins = sum(1 for row in rows if row["points"] == 1.0)
    draws = sum(1 for row in rows if row["points"] == 0.5)
    beaten = sorted(
        {
            str(row["opponent"])
            for row in done.values()
            if row.get("status") == "complete" and row.get("result") == "win"
        }
    )
    return {
        "schema": SCHEMA,
        "game": game,
        "challenger_id": challenger_id,
        "challenger_root": str(candidate_root),
        "pool": frozen.summary(),
        "pool_fingerprint": frozen.fingerprint,
        "challenger_track": challenger_track,
        "scope": scope,
        "scope_note": scope_note,
        "seeds": [int(item) for item in seeds],
        "roles": list(roles),
        "opponents": len(opponents),
        "planned_matches": planned,
        "complete_matches": complete,
        "failed_matches": len(done) - complete,
        "pending_matches": planned - len(done),
        "partial": bool(stopped or len(done) < planned),
        "wins": wins,
        "draws": draws,
        "losses": complete - wins - draws,
        "win_rate": None if complete == 0 else round((wins + 0.5 * draws) / complete, 4),
        "beaten_opponents": beaten,
        "elo": None if estimate is None else round(estimate.elo, 2),
        "pool_rank": None if estimate is None else frozen.rank_of(estimate.elo),
        "elo_gap_to_top": (
            None if estimate is None else round(frozen.top_elo - estimate.elo, 2)
        ),
        "score_rate": None if estimate is None else round(estimate.score_rate, 4),
        "anchored_matches": 0 if estimate is None else estimate.anchored_matches,
        "elapsed_s": round(elapsed, 1),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": (
            "challenger vs frozen human pool, both seats; anchors read-only from "
            "players/measured_elo.json (never rewritten); independent of the run's "
            "conquest matches and of other candidates"
        ),
    }


def load_challenger_elo(work_root: str | Path) -> dict[str, object] | None:
    path = Path(work_root) / SUMMARY_FILENAME
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None
