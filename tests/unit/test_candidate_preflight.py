"""候选前置校验的契约测试。

最重要的一条：**复现线上那次"5 轮全 0 回合判负"的故障**——候选包里有 main.py
却没有 ai.py，导致 `from ai import AI` 直接 ImportError。这类失败必须在跑对局之前
被判掉，并给出可执行的原因。
"""

from __future__ import annotations

from pathlib import Path

from agentbench_hl.application.candidate_preflight import check_candidate

SCAFFOLD_MAIN = "from ai import AI\n\nif __name__ == '__main__':\n    AI()\n"


def test_missing_entry_is_rejected(tmp_path: Path) -> None:
    issues = check_candidate("v000", tmp_path)

    assert [issue.kind for issue in issues] == ["missing_entry"]
    assert "main.py" in issues[0].detail


def test_missing_interface_module_is_rejected_before_any_match(tmp_path: Path) -> None:
    # 线上真实形态：脚手架齐全、策略写在 strategy_core.py，但没有 ai.py
    (tmp_path / "main.py").write_text(SCAFFOLD_MAIN, encoding="utf-8")
    (tmp_path / "strategy_core.py").write_text("class Strategy:\n    pass\n", encoding="utf-8")

    issues = check_candidate("v002", tmp_path, candidate_interface="AI.choose_operations")

    kinds = [issue.kind for issue in issues]
    assert "missing_interface" in kinds
    detail = next(issue.detail for issue in issues if issue.kind == "missing_interface")
    assert "ai.py" in detail
    # 反馈必须可执行：要指出当前有哪些文件，而不是只说"失败"
    assert "strategy_core.py" in detail


def test_interface_module_without_the_class_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(SCAFFOLD_MAIN, encoding="utf-8")
    (tmp_path / "ai.py").write_text("class Agent:\n    pass\n", encoding="utf-8")

    issues = check_candidate("v003", tmp_path, candidate_interface="AI.choose_operations")

    assert [issue.kind for issue in issues] == ["missing_interface"]
    assert "AI" in issues[0].detail


def test_syntax_error_is_reported_with_line_number(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    issues = check_candidate("v004", tmp_path)

    assert [issue.kind for issue in issues] == ["syntax_error"]
    assert "main.py:1" in issues[0].detail


def test_startup_crash_is_reported_with_traceback_tail(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("import definitely_not_installed\n", encoding="utf-8")

    issues = check_candidate("v005", tmp_path)

    assert [issue.kind for issue in issues] == ["startup_crash"]
    assert "definitely_not_installed" in issues[0].detail


def test_valid_candidate_that_waits_for_input_passes(tmp_path: Path) -> None:
    # 正常选手会阻塞等配置帧：不能把"没有输出"当成失败
    (tmp_path / "main.py").write_text(
        "import sys\nwhile True:\n    data = sys.stdin.buffer.read(4)\n    if not data:\n"
        "        break\n",
        encoding="utf-8",
    )
    (tmp_path / "ai.py").write_text(
        "class AI:\n    def choose_operations(self):\n        return []\n", encoding="utf-8"
    )

    assert check_candidate("v006", tmp_path, candidate_interface="AI.choose_operations") == []


def test_declared_interface_is_ignored_when_absent(tmp_path: Path) -> None:
    """没有声明 candidate_interface 的游戏（目前 7 个）只做通用检查。"""

    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")

    assert check_candidate("v007", tmp_path, candidate_interface=None) == []
