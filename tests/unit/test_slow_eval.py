"""慢评测自动启动：命令拼装与失败语义。

为什么这组测试值得存在
--------------------
``evaluation.background_pool`` 长期是个**假开关**：只被解析、没有任何消费点。
配置里写 ``true`` 什么也不会发生，慢评测一直靠人手动敲
``scripts/pool_elo_worker.py``。而漏做的表现是 **Elo 面板里少一条实测曲线**
——不报错、不崩溃，只是图上静静地少了东西，很容易一路跑完 32 轮才发现。

所以这里锁三件事：开关真的会起进程、抽样口径正确、分轨游戏不会漏掉轨道。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentbench_hl.application.slow_eval import DEFAULT_STRIDE, build_plan


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "pool_elo_worker.py").write_text("# worker\n", encoding="utf-8")
    return repo


def _plan(tmp_path: Path, **overrides: object):
    return build_plan(
        run_root=tmp_path / "runs" / "ab32-antwar2-fix",
        agentbench_root=tmp_path / "AgentBench",
        game="antwar2",
        repository_root=_repo(tmp_path),
        **overrides,  # type: ignore[arg-type]
    )


def test_plan_samples_the_evolution_mainline_every_three_iterations(tmp_path: Path) -> None:
    """默认口径：只测各轮最佳候选，每 3 轮一版。

    ``--best-only`` 的理由：落选候选是探索分支，不构成后续迭代的基础，
    测它们不会让主线曲线更完整，只会消耗机时。

    ``--iteration-stride 3`` 的理由：32 轮 → 11 版 × 约 458 局 ≈ 5k 局，
    与迭代并行跑得完。stride=1 是 32 版 ≈ 15k 局，慢评测会永远追不上迭代，
    图上只有前几个点——那正是"看起来配了慢评测，曲线却是断的"的来源。
    """

    command = _plan(tmp_path).command

    assert "--best-only" in command
    index = command.index("--iteration-stride")
    assert command[index + 1] == str(DEFAULT_STRIDE) == "3"


def test_plan_passes_the_game_and_frozen_pool_location(tmp_path: Path) -> None:
    command = _plan(tmp_path).command

    assert command[command.index("--game") + 1] == "antwar2"
    assert command[command.index("--agentbench-root") + 1].endswith("AgentBench")
    assert command[command.index("--run-root") + 1].endswith("ab32-antwar2-fix")


def test_asymmetric_games_must_carry_the_challenger_track(tmp_path: Path) -> None:
    """分轨游戏必须把轨道传下去，否则会同轨互殴。

    实测教训：rollman 少了这个参数时，挑战者会去打**同轨**选手
    （ghost 打 ghost），那种对局在协议层就没有意义——回放只有 2 行，
    IG 恒为常数。当时排查了很久才定位到轨道没分。
    """

    command = _plan(tmp_path, challenger_track="rollman").command

    assert command[command.index("--challenger-track") + 1] == "rollman"


def test_symmetric_games_omit_the_track_flag(tmp_path: Path) -> None:
    """对称游戏不该出现空的 --challenger-track（worker 会当成非法轨道名）。"""

    assert "--challenger-track" not in _plan(tmp_path).command


def test_parallelism_leaves_headroom_for_the_main_iteration(tmp_path: Path) -> None:
    """慢评测只能用主迭代剩下的机时。

    抢机时会让 agent 思考变慢，而思考占全程约 84% —— 那等于用观测通道
    拖慢被观测的对象。
    """

    plan = _plan(tmp_path, headroom=8)

    parallel = int(plan.command[plan.command.index("--parallel") + 1])
    headroom = int(plan.command[plan.command.index("--headroom") + 1])
    assert parallel >= 1
    assert headroom == 8


def test_seeds_are_forwarded_as_a_comma_list(tmp_path: Path) -> None:
    command = _plan(tmp_path, seeds=(7, 11)).command

    assert command[command.index("--seeds") + 1] == "7,11"


def test_missing_worker_script_is_loud(tmp_path: Path) -> None:
    """worker 脚本不在时必须报错，不能静默退化成"没有慢评测"。

    静默失败的代价是跑完整个 run 才发现图上缺一条线，而那时机时已经花掉了。
    """

    bare = tmp_path / "bare"
    (bare / "scripts").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="pool_elo_worker.py"):
        build_plan(
            run_root=tmp_path / "runs" / "x",
            agentbench_root=tmp_path / "AgentBench",
            game="antwar2",
            repository_root=bare,
        )


def test_log_path_sits_next_to_the_run_directory(tmp_path: Path) -> None:
    """日志要能被找到：慢评测的失败原因只在它自己的日志里。"""

    plan = _plan(tmp_path)

    assert plan.log_path.name == "ab32-antwar2-fix.pool-elo.log"
    assert plan.log_path.parent == (tmp_path / "runs")


def test_spawn_refuses_to_start_a_second_worker(tmp_path: Path, monkeypatch) -> None:
    """同一个 run 只能有一个 worker。

    慢评测有两个入口：新 run 由 ``background_pool: true`` 自动挂，
    已在跑的 run 用 ``attach_slow_eval.sh`` 手动挂。两者撞车时会有两个 worker
    **并发写同一个 pool-elo/**：同一版本重复调度、``matches.jsonl`` 交错追加、
    ``challenger-elo.json`` 互相覆盖。这些都不报错，只是白烧机时、数据可疑。
    """

    from agentbench_hl.application import slow_eval

    plan = _plan(tmp_path)
    monkeypatch.setattr(slow_eval, "already_running", lambda _root: 4242)

    assert slow_eval.spawn(plan) is None, "已有 worker 时必须返回 None 而不是再起一个"


def test_already_running_ignores_unrelated_workers(tmp_path: Path) -> None:
    """别的 run 的 worker 不该被误认成自己的。

    四个 ablation run 的 worker 会同时存在，run_root 只差最后一段路径；
    用子串匹配时若不带完整路径就会互相误判，结果是**三个 run 都不会挂上
    慢评测**（各自以为已经有人在跑了），而 Elo 面板静静地少三条线。
    """

    from agentbench_hl.application.slow_eval import already_running

    # 真实进程表里不会有这个路径的 worker。
    assert already_running(tmp_path / "runs" / "no-such-run-xyz") is None
