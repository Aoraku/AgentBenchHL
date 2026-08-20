"""aquawar 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

**注意 aquawar 与其余 7 个游戏不同**：它的官方 Python SDK 是一个 pybind11 C++ 扩展
（需要 cmake + pybind11 现场编译，池里 194 个审计通过的选手全是 C++、
一个 Python 选手都没有）。在禁网沙箱里为每个候选编译一次既脆弱又浪费机时，
所以协议层是**纯 Python 移植版** ``aquawar_sdk.py``——它逐条对齐官方
``ai_client.hpp`` 的线协议与消息装配，来源与移植范围写在该文件开头。

你要写的（``ai.py``）::

    from aquawar_sdk import Action, AIClient

    class AI(AIClient):
        def pick(self, game):
            # 选 4 条鱼上场，返回它们的编号；编号 > 12 表示拟态（12 + 被模仿者编号）
            return [1, 2, 3, 4]

        def assert_fish(self, game):
            # 断言敌方某个位置是哪条鱼；(-1, -1) = 不断言
            return (-1, -1)

        def act(self, game):
            # Type 0 = 普通攻击（用 enemy_target），1 = 主动技能
            # enemy_target = -2 表示 AOE（SDK 会展开成所有存活敌方）
            targets = game.living_enemies()
            return Action(type=0, my_pos=game.living_allies()[0], enemy_target=targets[0])

要点：
* ``AI`` **必须继承 ``AIClient``**（协议循环在基类的 ``run()`` 里）；
* ``game.raw`` 是判题器这一帧的**原始 JSON**，官方字段一个不少
  （``GameInfo`` 里是 ``EnemyFish`` / ``EnemyHP`` / ``MyFish`` / ``MyHP`` / ``MyATK``，
  敌方鱼 ``id == -1`` 表示身份还没被断言出来）；
* 断言位置越界、或该位置身份已知时，SDK 会自动改发 ``Null``，不用你判断；
* 回放里 ``operation[].Action`` 是**字符串**（``"Pick"`` 等），且帧数组
  **末元素是 ``null`` 哨兵**，读回放时必须过滤；
* **诊断只写 stderr**。
"""

from __future__ import annotations

import _bootstrap

CONTRACT = "class AI(AIClient): def pick / assert_fish / act"


def main() -> int:
    _bootstrap.install_path()
    ai_class = _bootstrap.load_ai_class(expected=CONTRACT)

    from aquawar_sdk import AIClient

    if not issubclass(ai_class, AIClient):
        _bootstrap.log(
            "[candidate] ai.py 里的 AI 没有继承 AIClient（来自 aquawar_sdk），\n"
            "            这样不会有任何协议循环，对战器只会看到 0 回合判负。\n"
            f"            该游戏要求：{CONTRACT}"
        )
        return 12

    agent = _bootstrap.construct(ai_class)

    try:
        agent.run()
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
