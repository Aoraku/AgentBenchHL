"""一轮复盘并成一份：省的是"往返次数"，不是"上下文大小"。

为什么这不是可有可无的整理
--------------------------
实测（``scripts/diag_turn_breakdown.py`` 对 7 轮 run 的归因）：

* 一轮 30~68 次模型往返，工具执行只占总墙钟的 **1%**；
* 单次往返平均 13.3s，而其中约 10s 与上下文大小**无关**——同一个小问题
  （输入 58 tokens）在 reasoning_effort=high 下就要 9.8s，中转站还把
  reasoning token 报成 0，所以那段思考只体现为墙钟。

也就是说 agent 每多读一个文件就多付一次约 13 秒的"思考税"。antwar2 一轮 8 局、
8 份 replay.md 分 8 次读 ≈ 100s 纯税。并成一份后只付一次：多出的 prefill 几秒
且有 95% 缓存命中。

这组测试锁住三件事：合并文件真的生成、缺复盘的局要**如实写出原因**（那本身是
学习信号，不能静默留空）、以及提示词把 agent 指向合并版而不是逐局去读。
"""

from __future__ import annotations

from pathlib import Path

from agentbench_hl.application import goal_led_service as service_module
from agentbench_hl.application.goal_led_service import GoalLedService


def _rows(tmp_path: Path) -> list[dict[str, object]]:
    first = tmp_path / "a.md"
    first.write_text("第 1 局复盘正文\n判决细节若干\n", encoding="utf-8")
    second = tmp_path / "b.md"
    second.write_text("第 2 局复盘正文\n", encoding="utf-8")
    return [
        {
            "candidate_id": "v001_rush",
            "opponent_id": "rank10",
            "role": "P0",
            "seed": 1,
            "status": "complete",
            "result": "loss",
            "narration_path": str(first),
        },
        {
            "candidate_id": "v001_rush",
            "opponent_id": "rank10",
            "role": "P1",
            "seed": 1,
            "status": "complete",
            "result": "win",
            "narration_path": str(second),
        },
        {
            "candidate_id": "v002_eco",
            "opponent_id": "rank10",
            "role": "P0",
            "seed": 1,
            "status": "complete",
            "result": "loss",
            "narration_path": None,
            "narration_note": "评测器未产出回放",
            "diagnostic": "first frame invalid",
        },
    ]


def test_combined_document_has_index_and_every_match(tmp_path: Path) -> None:
    service = object.__new__(GoalLedService)  # 只用到纯函数部分，不需要完整装配
    root = tmp_path / "feedback"
    root.mkdir()

    combined = GoalLedService._write_combined_narration(service, root, _rows(tmp_path))

    assert combined is not None and combined.name == "all-replays.md"
    text = combined.read_text(encoding="utf-8")
    # 目录让 agent 能一眼定位到"哪一局值得细看"，而不必先通读全文。
    assert "## 目录" in text
    assert text.count("v001_rush") >= 2
    assert "第 1 局复盘正文" in text
    assert "第 2 局复盘正文" in text
    # 判决要跟在标题下，否则 agent 得回去翻 feedback.json 才知道这局赢没赢。
    assert "complete/win" in text
    assert "complete/loss" in text


def test_missing_narration_is_reported_not_silently_skipped(tmp_path: Path) -> None:
    """没有复盘本身是信息（评测器失败/首帧非法），必须写出来。"""

    service = object.__new__(GoalLedService)
    root = tmp_path / "feedback"
    root.mkdir()

    combined = GoalLedService._write_combined_narration(service, root, _rows(tmp_path))
    assert combined is not None
    text = combined.read_text(encoding="utf-8")
    assert "v002_eco" in text
    assert "评测器未产出回放" in text


def test_no_matches_writes_nothing(tmp_path: Path) -> None:
    service = object.__new__(GoalLedService)
    root = tmp_path / "feedback"
    root.mkdir()
    assert GoalLedService._write_combined_narration(service, root, []) is None
    assert not (root / "all-replays.md").exists()


def test_prompt_points_at_the_combined_file() -> None:
    source = Path(service_module.__file__).read_text(encoding="utf-8")
    assert "先读同目录下的 all-replays.md" in source
    # 必须明确劝止逐局读，否则 agent 仍会一份一份地拉。
    assert "不要逐局去读单局的 replay.md" in source
    # 单局文件仍要说明用途，别让它以为那些文件没了。
    assert "只在你要核对某个具体数字时才用" in source
