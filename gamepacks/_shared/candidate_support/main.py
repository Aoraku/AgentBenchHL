"""候选选手入口 —— **8 个游戏逐字节相同**，不要改，也不要删。

对战器以进程方式启动选手：``python main.py``。本文件只做三件事：

1. 把候选目录加入 ``sys.path``（对战器的工作目录不一定是候选目录）；
2. 从 ``ai.py`` 取 ``AI``——**这是你唯一需要写的文件**；
3. 交给统一的会话驱动 ``session.run()``。

如果你把策略写在别的文件名里（``strategy.py`` / ``strategy_core.py`` / ``agent.py`` …），
这里会 ``ImportError``，进程启动即死，对战器如约判你负、且**回合数为 0**。
线上真实发生过：连续 5 轮迭代全是 0 回合判负，就是因为少了一个 ``ai.py``。
框架现在会在跑对局之前做前置校验并把原因回给你，但**最省事的做法是别改这个约定**。

``ai.py`` 的最小合法形态::

    from common import Observation      # 该游戏的观测结构（见 common.py）

    class AI:
        def decide(self, observation):
            # 返回该游戏的合法动作（见 common.py 与 decision_space.yaml）
            ...
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main() -> int:
    import session
    from saiblo import log

    try:
        from ai import AI
    except ImportError as error:
        log(
            "[main] 无法从 ai.py 导入 AI：" + str(error) + "\n"
            "       候选包必须包含 ai.py 且在其中定义 class AI（入口写死为 `from ai import AI`）。\n"
            "       当前目录下的 .py 文件：" + ", ".join(sorted(
                name for name in os.listdir(HERE) if name.endswith(".py")
            ))
        )
        return 10

    try:
        import protocol
    except ImportError as error:
        log("[main] 无法导入 protocol.py（脚手架文件，不应删除）：" + str(error))
        return 11

    try:
        ai = AI()
    except Exception as error:  # noqa: BLE001 - 构造失败要给出可执行原因
        log(f"[main] AI() 构造失败：{type(error).__name__}: {error}")
        return 12

    return session.run(ai, protocol)


if __name__ == "__main__":
    sys.exit(main())
