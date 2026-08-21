#!/bin/bash
# 检查 game_roles() 对各游戏返回什么座次名。
#
# 座次名必须与 A 仓 games/<game>/game.yaml 的 roles 一致。不一致会让每一局都以
# "role X is not one of (...)" 失败，而指标上只显示 incomplete，极难排查。
set -uo pipefail

HL=/home/qingle/agentbench/AgentBenchHL
AB=/home/qingle/agentbench/AgentBench
PY=/home/qingle/agentbench/.venv/bin/python

cd "$HL"
"$PY" - "$AB" "$@" <<'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, "src")

from agentbench_hl.adapters.contract.factory import game_roles

root = Path(sys.argv[1])
for game in sys.argv[2:]:
    yaml_roles = "?"
    game_yaml = root / "games" / game / "game.yaml"
    if game_yaml.is_file():
        import yaml as yaml_module

        document = yaml_module.safe_load(game_yaml.read_text(encoding="utf-8")) or {}
        yaml_roles = document.get("roles")
    try:
        resolved = game_roles(root, game)
    except Exception as error:
        resolved = f"ERROR {type(error).__name__}: {error}"
    match = "OK" if tuple(yaml_roles or ()) == tuple(resolved or ()) else "MISMATCH"
    print(f"{game:<12} game.yaml={yaml_roles!s:<24} game_roles()={resolved!s:<24} {match}")
PYEOF
