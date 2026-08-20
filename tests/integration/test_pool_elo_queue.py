"""固定人类池评测的口径测试。

这层的价值全在"口径不能漂"：
* 人类池必须**冻结**——挑战者的胜负绝不能反过来改变尺子，否则"在人类池里排第几"
  这句话失去意义；
* 池子一旦重测，旧结果必须被判为**不可比**（靠指纹），而不是静默混用；
* 后台队列必须优先测**主线最佳候选**，因为机时可能永远追不上迭代速度；
* CPU 水位闸门必须真的会挡住派发，否则"不影响主迭代"只是句空话。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_hl.application.challenger_eval import FrozenPool, load_frozen_pool

WORKER = Path(__file__).resolve().parents[2] / "scripts" / "pool_elo_worker.py"


def _load_worker():
    """按路径加载 worker 脚本（它不在包里，是个运维脚本）。

    必须先注册进 ``sys.modules`` 再执行：模块里的 ``@dataclass`` 在解析注解时会
    ``sys.modules[cls.__module__]`` 反查命名空间，没注册就会 AttributeError。
    """

    import importlib.util
    import sys

    cached = sys.modules.get("pool_elo_worker")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("pool_elo_worker", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pool_elo_worker"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ FrozenPool


def _pool() -> FrozenPool:
    return FrozenPool(
        "antwar",
        {"top": 1183.2, "mid": 900.0, "low": 100.0, "floor": -286.9},
        Path("/tmp/measured_elo.json"),
    )


def test_rank_of_places_challenger_against_frozen_anchors() -> None:
    pool = _pool()
    assert pool.rank_of(2000.0) == 1  # 强过所有人
    assert pool.rank_of(1000.0) == 2  # 只弱于 top
    assert pool.rank_of(50.0) == 4
    assert pool.rank_of(-1000.0) == 5  # 垫底


def test_fingerprint_changes_when_pool_changes() -> None:
    """指纹是"可比性"的载体：池子重测过就必须变。"""

    baseline = _pool().fingerprint
    same = _pool().fingerprint
    assert baseline == same, "同样的池子必须给出同样的指纹"

    grown = FrozenPool(
        "antwar",
        {"top": 1183.2, "mid": 900.0, "low": 100.0, "floor": -286.9, "newcomer": 500.0},
        Path("/tmp/measured_elo.json"),
    )
    assert grown.fingerprint != baseline, "池子扩容后旧结果不可比"

    remeasured = FrozenPool(
        "antwar",
        {"top": 1200.0, "mid": 900.0, "low": 100.0, "floor": -286.9},
        Path("/tmp/measured_elo.json"),
    )
    assert remeasured.fingerprint != baseline, "锚点重测后旧结果不可比"


def test_empty_pool_is_rejected() -> None:
    """没有锚点就没有"第几名"这个概念，必须拒绝而不是返回一个数。"""

    with pytest.raises(ValueError, match="没有任何 Elo 锚点"):
        FrozenPool("antwar", {}, Path("/tmp/x.json"))


def test_load_frozen_pool_reads_measured_elo_only(tmp_path: Path) -> None:
    players = tmp_path / "games" / "antwar" / "players"
    players.mkdir(parents=True)
    (players / "measured_elo.json").write_text(
        json.dumps(
            {
                "ratings": [
                    {"player_id": "a", "measured_elo": 1000.0},
                    {"player_id": "b", "measured_elo": 800.0},
                    # 没有实测分的选手不能进锚点：把"未知强度"当平均水平会系统性偏移。
                    {"player_id": "c", "measured_elo": None},
                ]
            }
        ),
        encoding="utf-8",
    )
    pool = load_frozen_pool(tmp_path, "antwar")
    assert pool.anchors == {"a": 1000.0, "b": 800.0}
    assert pool.summary()["frozen"] is True


def test_missing_measured_elo_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="还没有实测人类池评分"):
        load_frozen_pool(tmp_path, "antwar")


# ------------------------------------------------------------------ 队列发现


def _event(event_type: str, payload: dict) -> str:
    return json.dumps({"event_type": event_type, "payload": payload}, ensure_ascii=False)


def _run_with_events(tmp_path: Path, snapshots: dict[str, int], best: set[str]) -> Path:
    run_root = tmp_path / "run"
    run_root.mkdir(exist_ok=True)
    lines: list[str] = []
    for candidate, iteration in snapshots.items():
        package = run_root / "snapshots" / candidate
        package.mkdir(parents=True, exist_ok=True)
        (package / "main.py").write_text("print(1)\n", encoding="utf-8")
        request = f"round-{iteration:03d}"
        lines.append(
            _event("GoalVersionSnapshot", {"candidate_id": candidate, "path": str(package)})
        )
        lines.append(
            _event("GoalMatchCompleted", {"candidate_id": candidate, "request_id": request})
        )
        lines.append(
            _event(
                "IterationMetricsFinalized",
                {
                    "request_id": request,
                    "research_iteration": iteration,
                    "best_candidate_id": next(
                        (item for item in best if item in snapshots and item == candidate),
                        None,
                    ),
                },
            )
        )
    (run_root / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_root


def test_queue_prioritises_mainline_then_newest(tmp_path: Path) -> None:
    """先测主线最佳候选，再按迭代从新到旧。

    机时可能永远追不上迭代速度；这个顺序保证"演进主线的曲线"总是完整的，
    而不是随机测到一堆被证伪的探索分支。
    """

    worker = _load_worker()
    run_root = _run_with_events(
        tmp_path,
        {"v1_a": 1, "v1_b": 1, "v2_a": 2, "v2_b": 2},
        best={"v1_a", "v2_a"},
    )
    jobs = worker.discover_jobs(run_root)
    assert [job.candidate_id for job in jobs] == ["v2_a", "v1_a", "v2_b", "v1_b"]
    assert [job.is_best for job in jobs] == [True, True, False, False]


def test_queue_skips_snapshots_without_main(tmp_path: Path) -> None:
    worker = _load_worker()
    run_root = _run_with_events(tmp_path, {"good": 1}, best=set())
    broken = run_root / "snapshots" / "broken"
    broken.mkdir(parents=True)
    with (run_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            _event("GoalVersionSnapshot", {"candidate_id": "broken", "path": str(broken)}) + "\n"
        )
    assert [job.candidate_id for job in worker.discover_jobs(run_root)] == ["good"]


def test_truncated_last_line_does_not_break_discovery(tmp_path: Path) -> None:
    """主进程正在追加时最后一行可能只写了一半，扫描必须容忍。"""

    worker = _load_worker()
    run_root = _run_with_events(tmp_path, {"v1": 1}, best=set())
    with (run_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"event_type": "GoalVersionSnapsho')
    assert [job.candidate_id for job in worker.discover_jobs(run_root)] == ["v1"]


def test_results_from_a_different_pool_are_requeued(tmp_path: Path) -> None:
    """人类池重测后旧 Elo 与新尺子不可比，必须重新排队。"""

    worker = _load_worker()
    pool = _pool()
    work_root = tmp_path / "pool-elo" / "v1"
    work_root.mkdir(parents=True)

    (work_root / "challenger-elo.json").write_text(
        json.dumps({"pool_fingerprint": pool.fingerprint, "partial": False}), encoding="utf-8"
    )
    assert worker.is_done(work_root, pool) is True

    (work_root / "challenger-elo.json").write_text(
        json.dumps({"pool_fingerprint": "stale-fingerprint", "partial": False}), encoding="utf-8"
    )
    assert worker.is_done(work_root, pool) is False


def test_partial_results_are_requeued(tmp_path: Path) -> None:
    """被 CPU 闸门中断的半份结果不算完成，否则会记下一个偏低的 Elo。"""

    worker = _load_worker()
    pool = _pool()
    work_root = tmp_path / "pool-elo" / "v1"
    work_root.mkdir(parents=True)
    (work_root / "challenger-elo.json").write_text(
        json.dumps({"pool_fingerprint": pool.fingerprint, "partial": True}), encoding="utf-8"
    )
    assert worker.is_done(work_root, pool) is False


# ------------------------------------------------------------------ CPU 闸门


def test_idle_gate_blocks_until_load_drops(monkeypatch) -> None:
    """CPU 忙就必须等——这是"不影响主迭代"承诺的全部实现。"""

    worker = _load_worker()
    monkeypatch.setattr(worker, "cpu_count", lambda: 32)
    readings = iter([28.0, 26.0, 19.0])
    monkeypatch.setattr(worker, "load_average", lambda: next(readings))
    slept: list[float] = []
    monkeypatch.setattr(worker.time, "sleep", slept.append)

    worker.wait_for_idle_cpu(headroom=12, poll_s=5.0)
    # 32 - 12 = 20：前两次读数（28、26）都超过水位，必须等；第三次 19 才放行。
    assert slept == [5.0, 5.0]


def test_idle_gate_returns_immediately_when_idle(monkeypatch) -> None:
    worker = _load_worker()
    monkeypatch.setattr(worker, "cpu_count", lambda: 32)
    monkeypatch.setattr(worker, "load_average", lambda: 2.0)
    slept: list[float] = []
    monkeypatch.setattr(worker.time, "sleep", slept.append)
    worker.wait_for_idle_cpu(headroom=12, poll_s=5.0)
    assert slept == []


def test_idle_gate_honours_stop_signal(monkeypatch) -> None:
    """收到停止信号要立刻返回，不能卡在等待里让关停超时。"""

    worker = _load_worker()
    monkeypatch.setattr(worker, "cpu_count", lambda: 32)
    monkeypatch.setattr(worker, "load_average", lambda: 31.0)
    monkeypatch.setattr(worker.time, "sleep", lambda _s: None)
    worker.wait_for_idle_cpu(headroom=12, poll_s=1.0, should_stop=lambda: True)


# ------------------------------------------------------------------ 索引


def test_index_marks_incomparable_versions(tmp_path: Path) -> None:
    """指纹不同的历史结果保留但标 comparable=false，绝不静默混进峰值。"""

    worker = _load_worker()
    pool = _pool()
    queue_root = tmp_path / "pool-elo"
    for candidate, fingerprint, elo, iteration in (
        ("v1", pool.fingerprint, 900.0, 1),
        ("v2", pool.fingerprint, 1250.0, 2),
        ("v0_old", "stale", 9999.0, 0),
    ):
        directory = queue_root / candidate
        directory.mkdir(parents=True)
        (directory / "challenger-elo.json").write_text(
            json.dumps(
                {
                    "challenger_id": candidate,
                    "iteration": iteration,
                    "elo": elo,
                    "pool_rank": pool.rank_of(elo),
                    "pool_fingerprint": fingerprint,
                    "partial": False,
                }
            ),
            encoding="utf-8",
        )

    document = worker.write_index(queue_root, pool)
    by_id = {row["candidate_id"]: row for row in document["versions"]}
    assert by_id["v0_old"]["comparable"] is False
    assert by_id["v2"]["comparable"] is True
    # 峰值只能从可比版本里取，否则那个 9999 会污染结论。
    assert document["peak"]["candidate_id"] == "v2"
    assert document["pool"]["frozen"] is True
