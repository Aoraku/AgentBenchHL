"""generals 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

协议层**完全由官方 SDK 负责**（本目录下的 ``generals_impact_game/`` 包）。
官方把通讯封装成 ``run_ai(ai_func)``：读入是判题器原样写来的 JSON 地图/文本命令，
写出是 ``[4 字节大端无符号长度][多行命令]``、**以 ``8`` 结尾**（8 = 回合结束）。
这些细节你都不用碰。

你要写的（``ai.py``）::

    from generals_impact_game.gamestate import GameState

    class AI:
        def decide(self, round: int, my_seat: int, state: GameState) -> list[list[int]]:
            # 每个元素是一条命令，形如 [命令码, 参数...]
            # 1 军队移动 / 2 将领移动 / 3 将领升级 / 4 战法 / 5 科技 /
            # 6 超武 / 7 召唤副将 / 8 回合结束（**不要自己加 8，SDK 会加**）
            return []

``generals_impact_game/controller.py`` 里有一批 ``*_op()`` 辅助函数
（``move_army_op`` / ``release_general_skill_op`` / ``use_superweapon_op`` …），
用它们构造命令比手写数字列表安全。

**诊断只写 stderr**。
"""

from __future__ import annotations

import _bootstrap

CONTRACT = "def decide(self, round: int, my_seat: int, state: GameState) -> list[list[int]]"


def main() -> int:
    _bootstrap.install_path()
    ai_class = _bootstrap.load_ai_class(expected=CONTRACT)
    agent = _bootstrap.construct(ai_class)

    decide = getattr(agent, "decide", None)
    if not callable(decide):
        _bootstrap.log(f"[candidate] AI 缺少 decide 方法。该游戏要求：{CONTRACT}")
        return 12

    from generals_impact_game.controller import run_ai

    try:
        run_ai(decide)
    except (EOFError, BrokenPipeError):
        _bootstrap.log("[candidate] 对局结束（stdin 关闭）")
        return 0
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        import traceback

        _bootstrap.log(
            f"[candidate] 对局中抛异常：{type(error).__name__}: {error}\n" + traceback.format_exc()
        )
        return 20
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
