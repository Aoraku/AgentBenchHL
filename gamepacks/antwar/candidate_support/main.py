"""antwar 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

协议层**完全由官方 SDK 负责**（本目录下的 ``antwar/`` 包，与人类选手当年拿到的
逐字节相同）。官方已经把通讯封装成 ``run_antwar_ai(ai_func)``，所以这里只做一件事：
把你写的 ``ai.AI.decide`` 交给它。

你要写的（``ai.py``）::

    from antwar.gamedata import Operation      # 见 antwar/ 官方 SDK
    from antwar.gamestate import GameState

    class AI:
        def decide(self, my_seat: int, game_state: GameState) -> list[Operation]:
            # 返回本回合要执行的操作列表；非法操作会被 SDK 自己过滤掉
            return []

要点：
* ``my_seat``（0/1）由判题器下发，先后手的收发顺序差异官方 SDK 已处理；
* ``game_state`` 是官方 SDK 维护的完整局面（含 ``pheromone`` 信息素——蚂蚁寻路的原因）；
* **诊断只写 stderr**。往 stdout 打印任何东西都会破坏协议帧。
"""

from __future__ import annotations

import _bootstrap

CONTRACT = "def decide(self, my_seat: int, game_state: GameState) -> list[Operation]"


def main() -> int:
    _bootstrap.install_path()
    ai_class = _bootstrap.load_ai_class(expected=CONTRACT)
    agent = _bootstrap.construct(ai_class)

    decide = getattr(agent, "decide", None)
    if not callable(decide):
        _bootstrap.log(f"[candidate] AI 缺少 decide 方法。该游戏要求：{CONTRACT}")
        return 12

    from antwar.controller import run_antwar_ai

    # run_antwar_ai 是官方的无限回合循环；对局结束时判题器关闭 stdin，
    # 官方 SDK 会抛异常/EOF 退出，这属于正常终止，不能当成候选崩溃。
    try:
        run_antwar_ai(decide)
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
