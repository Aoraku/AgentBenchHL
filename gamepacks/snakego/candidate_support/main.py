"""snakego 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

协议层**完全由官方 SDK 负责**（本目录下的 ``adk.py``：读 stdin 裸二进制记录流，
写 ``[len:4 大端有符号][op:1]``，``op ∈ 1..6``）。官方的回合循环在 ``sampleAI.run()``
里，它内部会 ``ai = AI()`` 并逐蛇调用 ``ai.judge(snake, ctx)``。

这里用**依赖注入**复用官方循环：把 ``sampleAI`` 模块里的 ``AI`` 名字替换成你写的类，
再调官方 ``run()``。这样一行官方协议代码都不用重写——重写是"该游戏所有候选静默
0 回合判负"的主要来源。

你要写的（``ai.py``）::

    class AI:
        def judge(self, snake, ctx):
            # 返回本条蛇的操作码：1..4 移动（dx=[1,0,-1,0], dy=[0,1,0,-1]，
            # direction = op-1），5 融化射线，6 分裂
            return 1

要点：
* 每回合可能要对**多条蛇**分别决策，官方循环会反复调用 ``judge``；
* 返回非法操作会被官方 ``controller.apply`` 判为 ``Illegal Action`` 并抛错，
  整局作废，所以拿不准时返回一个一定合法的移动；
* ``ctx`` 是官方维护的完整上下文（``ctx.game_map`` / ``ctx.snake_list`` / ``ctx.turn``）。
  注意回放里**没有逐帧地图**，地图是官方 SDK 自己重放出来的；
* **诊断只写 stderr**。
"""

from __future__ import annotations

import _bootstrap

CONTRACT = "def judge(self, snake, ctx) -> int  # 1..6"


def main() -> int:
    _bootstrap.install_path()
    ai_class = _bootstrap.load_ai_class(expected=CONTRACT)
    if not callable(getattr(ai_class, "judge", None)):
        _bootstrap.log(f"[candidate] AI 缺少 judge 方法。该游戏要求：{CONTRACT}")
        return 12

    import sampleAI

    # 依赖注入：官方 run() 内部执行 `ai = AI()`，查的是 sampleAI 模块的全局名字。
    sampleAI.AI = ai_class

    try:
        sampleAI.run()
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
