"""会话轮转：在 codex 的 remote compaction（对本模型必死）之前换 thread。

背景（这组测试存在的唯一理由）
------------------------------
codex 0.147 的自动压缩走 ``POST /responses/compact``，要求响应里恰好一个
compaction output item；glm-5.2 对任何请求都返回 ``[reasoning, message]`` 两个 item，
于是压缩**一触发就必死**，并把整个 turn 打成 ``status=failed``：

    expected exactly one compaction output item, got 0 from 2 output items

两次线上 run（antwar 在 ~135k、antwar2 在 ~90k 记账点）都死在这条错误上。
"调压缩阈值"救不了它——主动压缩与被动兜底是同一条通道。唯一的解是让上下文
永远走不到压缩线：框架在阈值前主动换新 thread，工作区历史一份不丢。

所以这里锁住三件事：轮转会发生、不该轮转时不发生、以及**轮转后 agent 被告知
"你的记忆在文件里"**（漏掉最后一条，它会把接力当成第一次见到这个游戏，从零重写）。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agentbench_hl.application.goal_led_service import GoalLedService
from agentbench_hl.ports.agent_runtime import AgentSession, RunContext
from agentbench_hl.ports.arena import MatchCase, MatchResult

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments"


class CountingRuntime:
    """每次 start() 发一个新 thread id，便于断言"确实换了线程"。"""

    def __init__(self) -> None:
        self.starts = 0
        self.resumes: list[str] = []

    def start(self, run_context: RunContext) -> AgentSession:
        self.starts += 1
        return AgentSession(f"thread-{self.starts}", "paused", False)

    def resume(self, session_id: str, run_context: RunContext) -> AgentSession:
        self.resumes.append(session_id)
        return AgentSession(session_id, "paused", False)

    def run_until_checkpoint(
        self, session: AgentSession, run_context: RunContext, _predicate: object
    ) -> AgentSession:
        return session

    def pause(self, session: AgentSession) -> AgentSession:
        return session


class IdleArena:
    def run_case(self, case: MatchCase, candidate_root: Path) -> MatchResult:  # pragma: no cover
        raise AssertionError("rotation tests never run matches")


def _service(
    tmp_path: Path,
    runtime: CountingRuntime,
    threshold: int | None,
    each_iteration: bool = False,
) -> GoalLedService:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    (bootstrap / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    gamepack = tmp_path / "gamepack"
    gamepack.mkdir(exist_ok=True)
    return GoalLedService(
        run_root=tmp_path / "run",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=runtime,
        arena=IdleArena(),
        model="glm-5.3",
        model_provider="OpenAI",
        game="antwar",
        runnable_opponent_ids=("rank01",),
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 2000.0},),
        thread_rotate_context_tokens=threshold,
        thread_rotate_each_iteration=each_iteration,
    )


def _usage(service: GoalLedService, input_tokens: int, tag: str) -> None:
    service._append(
        "AgentTokenUsage",
        {"harness": "codex", "input_tokens": input_tokens, "total_tokens": input_tokens},
        f"usage:{tag}",
    )


def test_context_tokens_reset_after_rotation(tmp_path: Path) -> None:
    """input_tokens 是"当前上下文大小"，换 thread 后必须重新从零计。"""

    runtime = CountingRuntime()
    service = _service(tmp_path, runtime, 100_000)
    _usage(service, 40_000, "a")
    _usage(service, 95_000, "b")
    assert service._thread_context_tokens() == 95_000

    service._append("GoalSessionRotated", {"iteration": 2}, "rotated:2")
    assert service._thread_context_tokens() is None
    _usage(service, 12_000, "c")
    assert service._thread_context_tokens() == 12_000


def test_below_threshold_keeps_the_same_thread(tmp_path: Path) -> None:
    runtime = CountingRuntime()
    service = _service(tmp_path, runtime, 110_000)
    _usage(service, 109_999, "a")
    session, handoff = service._rotate_if_needed(
        AgentSession("thread-keep", "paused", False), iteration=3
    )
    assert session.thread_id == "thread-keep"
    assert handoff == ""
    assert runtime.starts == 0


def test_rotation_starts_new_thread_and_hands_off_file_memory(tmp_path: Path) -> None:
    runtime = CountingRuntime()
    service = _service(tmp_path, runtime, 110_000)
    _usage(service, 118_400, "a")

    session, handoff = service._rotate_if_needed(
        AgentSession("thread-old", "paused", False), iteration=4
    )

    assert runtime.starts == 1
    assert session.thread_id == "thread-1"
    # 接力说明必须指出"历史都在文件里"，否则 agent 会从零重写策略。
    assert "会话已接力" in handoff
    assert "EXPERIENCE.md" in handoff
    assert "feedback/" in handoff
    assert "不要从零重写策略" in handoff

    events = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
    assert "GoalSessionRotated" in events
    assert "thread-old" in events
    # 状态文件必须跟着换，否则下一轮 resume 会回到那个即将撞上压缩线的旧 thread。
    state = (tmp_path / "run" / "goal-led-state.json").read_text(encoding="utf-8")
    assert "thread-1" in state


def test_each_iteration_rotates_even_when_context_is_small(tmp_path: Path) -> None:
    """每轮无条件换线程：阈值挡不住"单个 turn 内部"的增长。

    antwar2 就是这么死的——死在 97k，而当时阈值是 110k，检查根本没轮到。
    一个 turn 里 agent 会连发几十个请求（读 8 份 replay.md、写 k 个候选），
    所以只有"每轮从零开始"才能让单轮增量成为上下文的上界。
    """

    runtime = CountingRuntime()
    service = _service(tmp_path, runtime, 60_000, each_iteration=True)
    _usage(service, 3_000, "tiny")  # 远低于阈值

    session, handoff = service._rotate_if_needed(
        AgentSession("thread-old", "paused", False), iteration=5, force=True
    )
    assert runtime.starts == 1
    assert session.thread_id == "thread-1"
    assert "会话已接力" in handoff

    rotated = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["event_type"] == "GoalSessionRotated"
    ]
    assert [event["payload"]["trigger"] for event in rotated] == ["each_iteration"]


def test_handoff_survives_missing_token_accounting(tmp_path: Path) -> None:
    """强制轮转可能发生在还没有任何 token 记账时，接力说明不能因此崩。"""

    runtime = CountingRuntime()
    service = _service(tmp_path, runtime, None, each_iteration=True)
    _, handoff = service._rotate_if_needed(
        AgentSession("thread-old", "paused", False), iteration=1, force=True
    )
    assert "会话已接力" in handoff
    assert "上下文到了" not in handoff  # 没有数字就不要编一个


def test_disabled_by_default(tmp_path: Path) -> None:
    """不配阈值就完全不改变原行为（轮转是规避手段，不是默认策略）。"""

    runtime = CountingRuntime()
    service = _service(tmp_path, runtime, None)
    _usage(service, 900_000, "a")
    session, handoff = service._rotate_if_needed(
        AgentSession("thread-keep", "paused", False), iteration=9
    )
    assert (session.thread_id, handoff, runtime.starts) == ("thread-keep", "", 0)


def test_sota_configs_rotate_before_codex_would_compact() -> None:
    """两份主线配置的压缩防线必须成立，且两个游戏保持一致（否则主表不可比）。

    这道防线由三件事共同构成，缺一件都会重现"run 跑几轮就死"：
      1. 每轮无条件换线程（阈值挡不住 turn 内增长）；
      2. 保底阈值存在，且远低于压缩线；
      3. 模型能力走厂商官方 catalog，否则 codex 用 fallback 元数据，
         压缩线变成一个我们没设过也看不见的数（实测 97k）。
    """

    documents = {
        name: yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
        for name in ("exp2-antwar-conquest.yaml", "exp2-antwar2-conquest.yaml")
    }
    for name, document in documents.items():
        runtime = document["runtime"]
        provider = document["provider"]
        assert runtime["thread_rotate_each_iteration"] is True, f"{name} 未开启每轮换线程"
        rotate = runtime["thread_rotate_context_tokens"]
        compact = provider["auto_compact_token_limit"]
        assert isinstance(rotate, int) and rotate < compact, f"{name} 阈值顺序不成立"
        assert provider["model_catalog"] == "zhipu", f"{name} 未使用厂商官方 model catalog"
        # 用了 catalog 就不该再手写窗口，否则"压缩线是多少"又变成两处声明打架。
        assert "context_window" not in provider, f"{name} 同时写了 catalog 与 context_window"

    left, right = documents.values()
    for key in ("model", "reasoning_effort", "model_catalog", "auto_compact_token_limit"):
        assert left["provider"][key] == right["provider"][key], f"两份配置的 provider.{key} 不一致"
