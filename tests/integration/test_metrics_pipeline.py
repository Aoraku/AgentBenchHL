"""实验 1/2/4/5 的产出链路端到端护栏。

这里验证的是"**指标 → 曲线 → 主表**"这条链路，用真实的 ``GoalLedService`` 驱动
（只有 LLM 侧是假的 runtime，因为它不属于本链路）：

1. 逐轮指标里 ``pool_elo`` 真的算出来了（原来 ``fixed_pool_elo`` 恒为 null，
   导致实验 2 的核心纵坐标画不出来）；
2. ``pool_elo`` 跨轮可比 —— 换对手不会让它跳（有序课程必须依赖这个性质）；
3. ``export_curves.py`` 能吃真事件，且 **null 不会被画成 0**；
4. ``build_leaderboard.py`` 能把多个 run 聚成主表，缺 ``pool_elo`` 时如实报缺。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentbench_hl.application.goal_led_service import GoalLedService
from agentbench_hl.ports.agent_runtime import AgentSession, RunContext
from agentbench_hl.ports.arena import MatchCase, MatchResult

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

# 两个已知强度的人类对手：pool_elo 的锚点就是这两个 score。
LEADERBOARD = (
    {"opponent_id": "rank02", "rank": 2, "score": 1600.0},
    {"opponent_id": "rank01", "rank": 1, "score": 2000.0},
)


class ScriptedRuntime:
    """假 harness：按预设脚本逐轮写 ``action.json``，不调任何模型。

    服务每一轮会唤醒 harness **两次**：先"出招"（期待写 ``action.json``），
    跑完对局后再唤醒一次让它"读反馈"（这次不该再写 action）。
    所以只在奇数次唤醒时写——写第二份的话，反馈轮那份会被下一轮当成新请求消费，
    于是"下一轮"实际没打新局，``trajectories_seen`` 停在原地。
    """

    def __init__(self, opponents: list[str]) -> None:
        self.opponents = opponents
        self.turns = 0
        self.threads: list[str] = []

    def start(self, run_context: RunContext) -> AgentSession:
        self.threads.append("thread-A")
        return AgentSession("thread-A", "paused", False)

    def resume(self, session_id: str, run_context: RunContext) -> AgentSession:
        self.threads.append(session_id)
        return AgentSession(session_id, "paused", False)

    def run_until_checkpoint(
        self,
        session: AgentSession,
        run_context: RunContext,
        _checkpoint_predicate: object,
    ) -> AgentSession:
        self.turns += 1
        index = (self.turns - 1) // 2
        if self.turns % 2 == 0 or index >= len(self.opponents):
            return session
        # 候选 id 必须逐轮新增：重复用同一个 id 会被判 GoalProtocolViolation，
        # 那一轮不打对局（服务认为 agent 没有产出新版本）。
        candidate_id = f"v{index:03d}"
        action = run_context.cwd / ".agentbench" / "action.json"
        action.parent.mkdir(exist_ok=True)
        action.write_text(
            json.dumps(
                {
                    "request_id": f"iter-{index}",
                    "candidate_ids": [candidate_id],
                    "opponent_id": self.opponents[index],
                    "roles": ["P0"],
                    "seeds": [1, 2],
                    "rationale": "scripted",
                }
            ),
            encoding="utf-8",
        )
        return session

    def pause(self, session: AgentSession) -> AgentSession:
        return session


class HalfWinArena:
    """一胜一负的对战器：得分率恒为 0.5。

    这是本测试的关键设计——**得分率不变时，pool_elo 应当等于对手锚点**。
    于是"换对手"这件事会让 pool_elo 从 2000 变成两者之间，
    而不是像 ``elo_vs_opponent`` 那样只反映当轮。
    """

    def __init__(self) -> None:
        self.calls = 0

    def run_case(self, case: MatchCase, candidate_root: Path) -> MatchResult:
        self.calls += 1
        win = self.calls % 2 == 1
        replay = candidate_root / f"replay-{self.calls}.json"
        replay.write_text('[{"round_state":{"camps":[5,0],"winner":0}}]', encoding="utf-8")
        return MatchResult(
            case=case,
            status="complete",
            result="win" if win else "loss",
            points=1.0 if win else 0.0,
            score_margin=5.0 if win else -5.0,
            rounds=10,
            payload={},
            replay_path=replay,
            trace_path=None,
        )


def _service(
    root: Path, opponents: list[str], *, policy: str = "self_decide"
) -> tuple[GoalLedService, ScriptedRuntime]:
    """装配一个离线 goal-led 服务。

    ``policy`` 默认 ``self_decide``（生产默认值）。注意该策略有个硬约束：
    **第一次官方请求必须打榜首**，否则直接 ValueError。要脚本化地换对手就得
    显式换成框架决定对手的策略（这里用 ``fixed_rank``），否则测的是那条约束
    而不是指标链路。
    """

    bootstrap = root / "bootstrap"
    bootstrap.mkdir(parents=True)
    (bootstrap / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    gamepack = root / "gamepack"
    gamepack.mkdir()
    runtime = ScriptedRuntime(opponents)
    service = GoalLedService(
        run_root=root / "run",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=runtime,
        arena=HalfWinArena(),
        model="gpt-5.6",
        model_provider="OpenAI",
        runnable_opponent_ids=tuple(str(row["opponent_id"]) for row in LEADERBOARD),
        public_leaderboard=LEADERBOARD,
        opponent_policy=policy,
        opponent_rank=1 if policy == "fixed_rank" else None,
    )
    return service, runtime


def _metrics(run_root: Path) -> list[dict]:
    rows = []
    for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") == "IterationMetricsFinalized":
            rows.append(event["payload"])
    return rows


def test_pool_elo_is_recorded_and_anchored_to_the_opponent(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, ["rank01"])
    service.start()

    rows = _metrics(tmp_path / "run")
    assert rows, "没有产出 IterationMetricsFinalized"
    metric = rows[-1]
    # 一胜一负 → 得分率 0.5 → 强度应当≈对手锚点 2000。
    assert metric["pool_elo"] == pytest.approx(2000.0, abs=60.0)
    detail = metric["pool_elo_detail"]
    assert detail["method"] == "anchored_mle"
    assert detail["anchored_matches"] == 2
    # fixed_pool_elo 是"真跑一遍全池"的慢通道口径，仍然诚实留 null，不冒充。
    assert metric["fixed_pool_elo"] is None


def test_pool_elo_accumulates_across_iterations(tmp_path: Path) -> None:
    """第 2 轮的估计要把第 1 轮的局也算进来——这是"跨轮可比"的来源。

    只用当轮数据的话，每轮都是一次小样本重估，曲线会因为方差而上下抖；
    累积之后样本量随轮次增长（``anchored_matches`` 单调增），曲线才是能力趋势。
    """

    service, _ = _service(tmp_path, ["rank01", "rank01"])
    service.start()
    service.advance()

    rows = _metrics(tmp_path / "run")
    assert len(rows) == 2
    counts = [row["pool_elo_detail"]["anchored_matches"] for row in rows]
    assert counts == [2, 4], f"样本量应随轮次累积，实际 {counts}"


def test_export_curves_reads_real_events_and_reports_null_series(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, ["rank01", "rank02"])
    service.start()
    service.advance()

    out = tmp_path / "curves"
    completed = subprocess.run(  # noqa: S603 - 固定为本仓脚本
        (
            sys.executable,
            str(SCRIPTS / "export_curves.py"),
            "--run-root",
            str(tmp_path / "run"),
            "--out",
            str(out),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    series = json.loads((out / "series.json").read_text(encoding="utf-8"))
    entry = series["runs"][0]["series"]
    # pool_elo 与 win_rate 对两种横坐标都要有点（实验 2 的主纵坐标）。
    # tokens 不在断言里：假 harness 不产生 token 事件，这个 run 的 total_tokens
    # 本来就该是 null——把它也要求上就等于要求测试环境伪造 token 计数。
    for key in ("pool_elo", "win_rate"):
        for x_key in ("iteration", "trajectories"):
            assert entry.get(f"{key}_vs_{x_key}"), f"缺曲线 {key}_vs_{x_key}"
    # 横坐标"看过的完整轨迹数"必须随轮次累积（每轮 2 局）。
    trajectories = [x for x, _ in entry["pool_elo_vs_trajectories"]]
    assert trajectories == sorted(trajectories)
    assert trajectories[-1] >= 4, f"轨迹数没有累积：{trajectories}"

    rows = (out / "curves.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("run,label,x_key")
    # outcome_ig 从第 2 轮起才有值（要有上一版候选才能做配对比较），第 1 轮是 null。
    # 关键是**没有被补成 0**：序列长度应当少于总轮数。
    ig_points = entry.get("outcome_ig_vs_iteration") or []
    assert len(ig_points) < len(entry["pool_elo_vs_iteration"])
    # 假 harness 不产生 token 事件 → tokens 全程 null，脚本必须明说而不是画成 0。
    assert "tokens" not in json.dumps(entry)
    assert "全程为 null" in completed.stdout
    assert "tokens" in completed.stdout


def test_build_leaderboard_aggregates_runs(tmp_path: Path) -> None:
    for name in ("grid-a-s1", "grid-a-s2"):
        service, _ = _service(tmp_path / name, ["rank01"])
        service.start()

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for name in ("grid-a-s1", "grid-a-s2"):
        (runs_root / name).symlink_to(tmp_path / name / "run")

    out = tmp_path / "board"
    completed = subprocess.run(  # noqa: S603 - 固定为本仓脚本
        (
            sys.executable,
            str(SCRIPTS / "build_leaderboard.py"),
            "--runs-root",
            str(runs_root),
            "--out",
            str(out),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    document = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert len(document["runs"]) == 2
    # 两个 seed 落在同一个 (模型, harness, 游戏) 单元格里。
    assert len(document["cells"]) == 1
    cell = document["cells"][0]
    assert cell["seeds"] == 2
    assert cell["pool_elo_median"] is not None
    assert len(document["totals"]) == 1
    board = (out / "leaderboard.md").read_text(encoding="utf-8")
    assert "实验 1 主表" in board
