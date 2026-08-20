"""探索多样性信号的契约：k 个候选必须是 k 个尝试。

为什么要有这组测试
------------------
"一个尝试做 k 遍"曾经完整地通过了所有既有校验：k 个候选各自有 main.py、
``code_fingerprint`` 四个互异、preflight 全过、对局全部正常跑完——然后一轮的
信息量退化成 1 个假设。既有的去重只能挡住"逐字节相同"，挡不住"同一骨架换阈值"。

所以这里锁三件事：
1. 度量本身能把"只差一个常量"和"两条不同取胜路径"分开；
2. 判定为伪多样性时会产出一句**可执行**的反馈（要进下一轮提示词）；
3. 下发给 agent 的指令里显式写着"k 个不同尝试，不是一个尝试做 k 遍"——
   这条断言防的是有人日后"精简提示词"时把它顺手删掉。
"""

from __future__ import annotations

from pathlib import Path

from agentbench_hl.application.candidate_diversity import (
    NEAR_DUPLICATE_LINES,
    feedback_note,
    spread,
)

_SKELETON = """
class AI:
    THRESHOLD = {threshold}

    def choose(self, state):
        if state.coins > self.THRESHOLD:
            return "build"
        return "wait"
"""


def _write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "main.py").write_text(body, encoding="utf-8")
    return path


def test_threshold_only_variants_are_flagged_as_near_duplicate(tmp_path: Path) -> None:
    """同一骨架换阈值 —— 指纹互异，但必须被判成伪多样性。"""

    roots = {
        f"v000_t{value}": _write(tmp_path, f"v000_t{value}", _SKELETON.format(threshold=value))
        for value in (10, 20, 30, 40)
    }
    report = spread(roots)
    assert report is not None
    assert report["candidates"] == 4
    assert report["pairs"] == 6
    assert report["min_diff_lines"] < NEAR_DUPLICATE_LINES
    assert report["verdict"] == "near_duplicate"
    assert len(report["near_duplicate_pairs"]) == 6

    note = feedback_note(report)
    assert note is not None
    assert "同一个尝试做了 k 遍" in note
    assert "一个不同的优化假设" in note


def test_distinct_win_paths_pass(tmp_path: Path) -> None:
    """两条真正不同的取胜路径不该被误判（否则会逼 agent 灌水行数）。"""

    roots = {
        "v000_rush": _write(
            tmp_path,
            "v000_rush",
            "\n".join(
                [
                    "class AI:",
                    "    def choose(self, state):",
                    *[f"        if state.round == {i}: return 'attack'" for i in range(20)],
                    "        return 'wait'",
                ]
            ),
        ),
        "v000_economy": _write(
            tmp_path,
            "v000_economy",
            "\n".join(
                [
                    "class AI:",
                    "    def plan(self, state):",
                    *[f"        if state.coins > {i * 7}: return 'save'" for i in range(20)],
                    "        return 'spend'",
                ]
            ),
        ),
    }
    report = spread(roots)
    assert report is not None
    assert report["verdict"] == "distinct"
    assert report["min_diff_lines"] >= NEAR_DUPLICATE_LINES
    assert feedback_note(report) is None


def test_spread_needs_two_measurable_candidates(tmp_path: Path) -> None:
    """单候选（或只有非代码文件）无法度量：诚实返回 None，不编数字。"""

    only = {"v000_solo": _write(tmp_path, "v000_solo", "class AI: pass\n")}
    assert spread(only) is None

    empty = tmp_path / "v000_readme_only"
    empty.mkdir()
    (empty / "README.md").write_text("no code here", encoding="utf-8")
    assert spread({"v000_solo": only["v000_solo"], "v000_readme_only": empty}) is None


def test_developer_instructions_state_k_distinct_attempts() -> None:
    """指令里必须显式区分"k 个尝试"与"一个尝试 k 遍"。"""

    from agentbench_hl.application import goal_led_service as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "k 个候选 = k 个不同的优化尝试" in source
    assert "不是同一个尝试做" in source
    # 第 0 轮同样要交 k 个候选，那一轮没有回放，多样性是唯一的信息来源。
    assert "不同的开局哲学" in source
