"""加载 A 仓声明的行为信息增益测量口径。

**A 是唯一事实源**：|A|、动作 token 口径、occupancy state id 都写在
``AgentBench/games/<game>/decision_space.yaml`` 的 ``information_gain:`` 段里，
schema 校验也由 A 的 ``agentbench.core.decision_space`` 负责。B 只负责把文件找出来、
把解析结果交给测量流程——**不在 B 里复制一份 schema**，否则两仓迟早各说各话。

A 是零依赖包、也不一定装进 B 的环境，所以这里按需把 ``<AGENTBENCH_ROOT>/src`` 加进
``sys.path`` 再导入。加载失败一律返回 ``(None, 原因)``，由上层把 behavioral_ig 记 null。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SPEC_FILENAME = "decision_space.yaml"


def _agentbench_root(explicit: str | Path | None) -> Path | None:
    if explicit:
        return Path(explicit).resolve()
    value = os.environ.get("AGENTBENCH_ROOT")
    return Path(value).resolve() if value else None


def _ensure_importable(root: Path) -> None:
    source = root / "src"
    if source.is_dir():
        text = str(source)
        if text not in sys.path:
            sys.path.insert(0, text)


def decision_space_path(
    game: str, *, agentbench_root: str | Path | None = None
) -> Path | None:
    """``games/<game>/decision_space.yaml`` 的绝对路径（找不到返回 None）。"""

    root = _agentbench_root(agentbench_root)
    if root is None:
        return None
    path = root / "games" / game / SPEC_FILENAME
    return path if path.is_file() else None


def load_information_gain_spec(
    game: str,
    *,
    agentbench_root: str | Path | None = None,
    spec_path: str | Path | None = None,
) -> tuple[object | None, str]:
    """返回 ``(InformationGainSpec | None, 说明)``。"""

    root = _agentbench_root(agentbench_root)
    path = Path(spec_path) if spec_path else decision_space_path(game, agentbench_root=root)
    if path is None or not path.is_file():
        return None, f"{game} has no {SPEC_FILENAME} under AGENTBENCH_ROOT"
    if root is not None:
        _ensure_importable(root)
    try:
        import yaml  # noqa: PLC0415 - 仅在测量时需要
        from agentbench.core.decision_space import (  # noqa: PLC0415
            DecisionSpaceError,
            parse_information_gain,
        )
    except ImportError as error:
        return None, f"cannot import AgentBench decision space schema: {error}"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        spec = parse_information_gain(document, game=game)
    except (OSError, DecisionSpaceError, ValueError) as error:
        return None, f"{game} {SPEC_FILENAME} is unusable: {error}"
    if spec is None:
        return None, (
            f"{game} declares no information_gain contract in {SPEC_FILENAME} "
            "(behavioral IG intentionally not measured for this game)"
        )
    return spec, f"loaded information_gain contract {spec.schema} from {path}"
