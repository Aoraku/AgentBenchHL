"""B → A 回放翻译桥的契约测试。

守的是"两仓分工"这条线：B 只负责调用与落盘，游戏语义全部在 A。
所以这里测的是**桥本身**（找得到 A、调得通、失败会降级），而不是叙述内容 ——
叙述内容的正确性由 A 仓的 `tests/test_replay_narration.py` 负责。

失败降级是重点：反馈通道断掉比反馈不完整严重得多。一次长跑几十小时，
如果某局回放恰好损坏就让整轮反馈丢掉，等于白烧一轮。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentbench_hl.application.replay_narration import NARRATION_FILENAME, narrate_case

REPO_ROOT = Path(__file__).resolve().parents[2]


def _agentbench_root() -> Path | None:
    value = os.environ.get("AGENTBENCH_ROOT")
    if value and (Path(value) / "src" / "agentbench" / "replay").is_dir():
        return Path(value)
    sibling = REPO_ROOT.parent / "AgentBench"
    if (sibling / "src" / "agentbench" / "replay").is_dir():
        return sibling
    return None


def test_missing_agentbench_root_is_reported_not_crashed(tmp_path: Path) -> None:
    path, note = narrate_case(
        "antwar2",
        None,
        tmp_path / NARRATION_FILENAME,
        match_id="m1",
        agentbench_root=None if "AGENTBENCH_ROOT" not in os.environ else tmp_path / "nope",
    )
    if path is None:
        assert "AGENTBENCH_ROOT" in note or "失败" in note
    else:
        # 环境里配了可用的 A 仓，那就应该真的翻出东西来。
        assert path.is_file()


def test_bridge_translates_a_real_replay(tmp_path: Path) -> None:
    root = _agentbench_root()
    if root is None:
        pytest.skip("找不到可用的 AgentBench(A) 仓")
    sample = next(
        iter(sorted((root / "data").glob("antwar2-matches/*/*/*/*/replay.json"))), None
    )
    if sample is None:
        pytest.skip("A 仓里没有 antwar2 回放样本")
    destination = tmp_path / NARRATION_FILENAME
    path, note = narrate_case(
        "antwar2",
        sample,
        destination,
        match_id="v001/P0-seed-1",
        perspective="P0",
        opponent_id="rank10",
        official_rounds=353,
        agentbench_root=root,
    )
    assert path == destination and path.is_file(), note
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert "## 判决" in text
    # 视角必须落到叙述里：agent 读的是"我"的复盘，不是中立解说。
    assert "你(P0)" in text
    # 关键收益：翻译后必须远小于裸回放，否则塞不进上下文。
    assert len(text) < sample.stat().st_size / 20


def test_no_replay_still_yields_actionable_text(tmp_path: Path) -> None:
    """0 回合判负是最常见的失败模式，桥必须把诊断带出来。"""

    root = _agentbench_root()
    if root is None:
        pytest.skip("找不到可用的 AgentBench(A) 仓")
    destination = tmp_path / NARRATION_FILENAME
    path, note = narrate_case(
        "antwar2",
        None,
        destination,
        match_id="v000/P0-seed-1",
        perspective="P0",
        diagnostic="候选启动失败：AttributeError: 'AI' object has no attribute 'choose_bundle'",
        agentbench_root=root,
    )
    assert path is not None and path.is_file(), note
    text = path.read_text(encoding="utf-8")
    assert "没有回放可读" in text
    assert "choose_bundle" in text, "对战器诊断必须出现在叙述里，否则 agent 会在同一个坑里反复迭代"


def test_corrupt_replay_degrades(tmp_path: Path) -> None:
    root = _agentbench_root()
    if root is None:
        pytest.skip("找不到可用的 AgentBench(A) 仓")
    broken = tmp_path / "replay.json"
    broken.write_text("[[[not-a-replay", encoding="utf-8")
    path, note = narrate_case(
        "antwar2",
        broken,
        tmp_path / NARRATION_FILENAME,
        match_id="m1",
        official_winner="P1",
        agentbench_root=root,
    )
    # 降级而不是抛异常：整轮反馈不能因为一局回放坏了就丢掉。
    assert path is not None, note
    assert "叙述降级" in path.read_text(encoding="utf-8")


def test_bridge_holds_no_game_semantics() -> None:
    """B 侧这层桥**不许**出现任何游戏字段名。

    一旦 B 也开始解析回放字段，两仓就会各说各话：A 改了解码、B 不知道，
    反馈里的数字会悄悄错掉，而且不报错。这条测试把分工钉死在代码里。
    """

    source = (
        REPO_ROOT / "src" / "agentbench_hl" / "application" / "replay_narration.py"
    ).read_text(encoding="utf-8")
    # 取自 8 个游戏回放的真实字段名，任何一个出现在 B 侧都说明语义泄漏了。
    forbidden = (
        "round_state",
        "pheromone",
        "fight_fish",
        "pacman_coord",
        "ghosts_coord",
        "item_list",
        "dead_snake",
        "score_dic",
        "EVENT_NAMES",
        "ACTION_NAMES",
    )
    leaked = [name for name in forbidden if name in source]
    assert not leaked, f"B 侧的桥里出现了游戏语义字段：{leaked}（这些只能在 A 仓）"


def test_feedback_row_carries_narration_path() -> None:
    """反馈行必须带 narration_path，否则下游不知道自然语言回放在哪。"""

    from agentbench_hl.application.goal_led_service import GoalLedService
    from agentbench_hl.ports.arena import MatchCase, MatchResult

    result = MatchResult(
        case=MatchCase("v001", "rank10", "P0", 1),
        status="complete",
        result="loss",
        points=0.0,
        score_margin=-1.0,
        rounds=0,
        replay_path=None,
        trace_path=None,
        error=None,
        payload={"game_error": "候选第一帧输出非法"},
    )
    row = GoalLedService._result_row(result, None, None, Path("/tmp/replay.md"), "note")
    assert row["narration_path"] == "/tmp/replay.md"
    assert row["narration_note"] == "note"
    assert row["diagnostic"] == "候选第一帧输出非法"
    assert json.dumps(row, ensure_ascii=False)
