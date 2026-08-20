"""批量固定池评测：扁平队列必须吃满所有核。

为什么要有这组测试
------------------
第一版实现是"一版一版串行跑"，有两个硬伤：

1. **队尾饥饿**：一个版本剩最后 3 局时只有 3 个线程在干活，其余核空转。
   版本越多，这种尾巴累积得越多。
2. **并发上限被版本内局数卡住**：并发能力应该由 ``总核数 / 每局核数`` 决定，
   而不是由"当前这个版本还剩几局"决定。

摊平成 (版本 × 对手 × 座次 × seed) 的全局队列之后，任何时刻都有足够的待办项
填满线程池。这里锁住这个性质，以及批量模式下每个版本仍然独立落盘、
独立算 Elo、独立续跑。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from agentbench_hl.application import challenger_eval
from agentbench_hl.application.challenger_eval import FrozenPool, evaluate_many
from agentbench_hl.ports.arena import MatchResult


class _RecordingArena:
    """记录并发峰值的假 arena。"""

    def __init__(self, *, beats: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, str, str, int, Path]] = []
        self.beats = beats or {}
        self._lock = threading.Lock()
        self._active = 0
        self.peak_concurrency = 0
        self.warmups: list[Path] = []

    def warmup(self, candidate_root: Path) -> None:
        self.warmups.append(candidate_root)

    def run_case(self, case, candidate_root: Path) -> MatchResult:  # noqa: ANN001
        with self._lock:
            self._active += 1
            self.peak_concurrency = max(self.peak_concurrency, self._active)
            self.calls.append(
                (case.candidate_id, case.opponent_id, case.role, case.seed, candidate_root)
            )
        try:
            # 让线程有机会真正重叠，从而测到并发峰值。
            threading.Event().wait(0.01)
            won = self.beats.get(case.candidate_id) == case.opponent_id
            return MatchResult(
                case=case,
                status="complete",
                result="win" if won else "loss",
                points=1.0 if won else 0.0,
                score_margin=10.0 if won else -10.0,
                rounds=100,
            )
        finally:
            with self._lock:
                self._active -= 1


def _pool() -> FrozenPool:
    return FrozenPool(
        "antwar",
        {"strong": 1200.0, "middle": 900.0, "weak": 600.0},
        Path("/tmp/measured_elo.json"),
    )


@pytest.fixture
def wired(monkeypatch, tmp_path: Path):
    """把 arena / 选手池 / 角色 都替换成可控的假件。"""

    arena = _RecordingArena()

    class _Player:
        def __init__(self, player_id: str) -> None:
            self.player_id = player_id
            self.track = None
            self.runnable = True

    players = [_Player(name) for name in ("strong", "middle", "weak")]

    monkeypatch.setattr(challenger_eval, "_arena", lambda *a, **k: arena)
    monkeypatch.setattr(challenger_eval, "load_pool", lambda *a, **k: players)
    monkeypatch.setattr(
        challenger_eval,
        "_select_players",
        lambda *a, **k: (("strong", "middle", "weak"), "测试口径"),
    )

    # game_roles / _supports_compiled_players 是在 evaluate_many **函数体内**
    # from ... import 进来的，所以必须替换源模块上的属性（而不是 challenger_eval
    # 上的），否则每次调用都会重新拿到真实实现。
    import agentbench_hl.adapters.contract.factory as factory

    monkeypatch.setattr(factory, "game_roles", lambda *a, **k: ("P0", "P1"), raising=True)
    monkeypatch.setattr(
        factory, "_supports_compiled_players", lambda *a, **k: False, raising=True
    )
    return arena


def _challengers(tmp_path: Path, names: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for name in names:
        directory = tmp_path / "snapshots" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "main.py").write_text("print(1)\n", encoding="utf-8")
        out.append((name, directory))
    return out


def test_flat_queue_covers_every_version_and_seat(wired, tmp_path: Path) -> None:
    arena = wired
    documents = evaluate_many(
        "antwar",
        tmp_path,
        _challengers(tmp_path, ["v1", "v2", "v3"]),
        queue_root=tmp_path / "pool-elo",
        pool=_pool(),
        parallel=8,
    )
    # 3 版 × 3 对手 × 2 座次 = 18 局，一局不少。
    assert len(arena.calls) == 18
    assert len(documents) == 3
    for document in documents:
        assert document["complete_matches"] == 6
        assert document["planned_matches"] == 6
        assert document["partial"] is False
        assert sorted(document["roles"]) == ["P0", "P1"]


def test_queue_is_not_starved_by_per_version_serialisation(wired, tmp_path: Path) -> None:
    """并发峰值必须超过"单个版本的局数"，否则就是退化成串行了。

    每个版本只有 6 局；如果实现是一版一版跑，峰值最多 6。扁平队列应当能
    同时跑到 parallel 指定的 8 路。
    """

    arena = wired
    evaluate_many(
        "antwar",
        tmp_path,
        _challengers(tmp_path, ["v1", "v2", "v3"]),
        queue_root=tmp_path / "pool-elo",
        pool=_pool(),
        parallel=8,
    )
    assert arena.peak_concurrency > 6, (
        f"并发峰值只有 {arena.peak_concurrency}，说明队列被按版本切断了"
    )


def test_each_version_gets_its_own_elo_and_files(wired, tmp_path: Path) -> None:
    """版本之间必须完全独立：赢的那个版本 Elo 更高，且各自落盘。"""

    arena = wired
    # v_good 打赢最强的池选手；v_bad 全败。
    arena.beats = {"v_good": "strong"}
    documents = evaluate_many(
        "antwar",
        tmp_path,
        _challengers(tmp_path, ["v_good", "v_bad"]),
        queue_root=tmp_path / "pool-elo",
        pool=_pool(),
        parallel=4,
    )
    by_id = {document["challenger_id"]: document for document in documents}
    # v_good 在两个座次上都打赢 strong ⇒ 6 局里赢 2 局。
    assert by_id["v_good"]["wins"] == 2
    assert by_id["v_bad"]["wins"] == 0
    assert by_id["v_good"]["elo"] > by_id["v_bad"]["elo"]
    assert by_id["v_good"]["win_rate"] == pytest.approx(2 / 6, abs=1e-4)
    assert by_id["v_good"]["beaten_opponents"] == ["strong"]

    for name in ("v_good", "v_bad"):
        directory = tmp_path / "pool-elo" / name
        assert (directory / "matches.jsonl").is_file()
        summary = json.loads((directory / "challenger-elo.json").read_text(encoding="utf-8"))
        assert summary["challenger_id"] == name
        assert summary["pool_fingerprint"] == _pool().fingerprint


def test_resume_skips_completed_matches(wired, tmp_path: Path) -> None:
    """续跑：已完成的局不能重打（后台评测经常被 CPU 闸门打断）。"""

    arena = wired
    challengers = _challengers(tmp_path, ["v1"])
    evaluate_many(
        "antwar",
        tmp_path,
        challengers,
        queue_root=tmp_path / "pool-elo",
        pool=_pool(),
        parallel=4,
    )
    assert len(arena.calls) == 6

    arena.calls.clear()
    documents = evaluate_many(
        "antwar",
        tmp_path,
        challengers,
        queue_root=tmp_path / "pool-elo",
        pool=_pool(),
        parallel=4,
    )
    assert arena.calls == [], "第二次调用不应重打任何一局"
    assert documents[0]["complete_matches"] == 6


def test_stop_signal_marks_results_partial(wired, tmp_path: Path) -> None:
    """被中断的批次必须标 partial，否则半份数据会被当成定论。"""

    evaluate_many(
        "antwar",
        tmp_path,
        _challengers(tmp_path, ["v1", "v2"]),
        queue_root=tmp_path / "pool-elo",
        pool=_pool(),
        parallel=2,
        should_stop=lambda: True,
    )
    for name in ("v1", "v2"):
        summary = json.loads(
            (tmp_path / "pool-elo" / name / "challenger-elo.json").read_text(encoding="utf-8")
        )
        assert summary["partial"] is True
        assert summary["complete_matches"] == 0


def test_every_version_is_warmed_up(wired, tmp_path: Path) -> None:
    """每个版本都要预热：候选包不同、编译产物不同，漏预热会集体撞车失败。"""

    arena = wired
    challengers = _challengers(tmp_path, ["v1", "v2"])
    evaluate_many(
        "antwar",
        tmp_path,
        challengers,
        queue_root=tmp_path / "pool-elo",
        pool=_pool(),
        parallel=4,
    )
    assert sorted(path.name for path in arena.warmups) == ["v1", "v2"]


def test_versions_without_main_are_ignored(wired, tmp_path: Path) -> None:
    arena = wired
    broken = tmp_path / "snapshots" / "broken"
    broken.mkdir(parents=True)
    documents = evaluate_many(
        "antwar",
        tmp_path,
        [*_challengers(tmp_path, ["v1"]), ("broken", broken)],
        queue_root=tmp_path / "pool-elo",
        pool=_pool(),
        parallel=4,
    )
    assert [document["challenger_id"] for document in documents] == ["v1"]
    assert len(arena.calls) == 6
