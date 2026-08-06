from __future__ import annotations

from pathlib import Path

from agentbench_hl.cli.main import _parser


def test_goal_led_start_command_accepts_config_and_run_id() -> None:
    args = _parser().parse_args(
        [
            "goal-led",
            "start",
            "--config",
            "configs/experiments/antwar2-goal-I.yaml",
            "--run-id",
            "pilot-rank01",
        ]
    )

    assert args.group == "goal-led"
    assert args.command == "start"
    assert args.config == Path("configs/experiments/antwar2-goal-I.yaml")
    assert args.run_id == "pilot-rank01"
