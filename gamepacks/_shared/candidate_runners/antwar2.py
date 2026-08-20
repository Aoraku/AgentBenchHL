"""antwar2 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

协议层**完全由官方 SDK 负责**（``protocol.py`` 的 ``ProtocolSession`` +
``SDK/`` 后端仿真）。官方本来的约定就是"``ai.py`` 里写 ``class AI(BaseAgent)``"，
主循环在官方 ``main.py`` 里（生成时改名为 ``official_main.py``，内容未改）。

你要写的（``ai.py``）::

    from common import BaseAgent

    class AI(BaseAgent):
        def choose_bundle(self, state, player, bundles=None):
            # bundles 是官方 ActionCatalog 枚举出的**合法**动作包，选一个返回
            bundles = bundles or self.list_bundles(state, player)
            return bundles[0]

要点：
* ``BaseAgent`` 是抽象基类，**唯一的抽象方法是 ``choose_bundle``**。
  只覆盖 ``choose_operations`` 是不够的——Python 会因"抽象方法未实现"直接拒绝
  实例化，表现为进程启动即死、0 回合判负；
* 想完全接管出招也可以再覆盖 ``choose_operations``（默认实现＝取
  ``choose_bundle`` 的 operations），或实现 ``create_session()`` 返回自定义
  ``MatchSession``；
* ``bundles`` 已经是合法动作，直接选比自己构造 ``Operation`` 安全；
* 操作码：``11`` BUILD / ``12`` UPGRADE / ``13`` DOWNGRADE / ``21`` STORM /
  ``22`` EMP / ``23`` DEFLECT / ``24`` EVASION / ``31`` UP_SPEED / ``32`` UP_ANTHP；
* 塔型：``0`` BASIC / ``1`` HEAVY / ``2`` QUICK / ``3`` MORTAR / ``11`` HEAVY+ /
  ``12`` ICE / ``13`` BW / ``21`` QUICK+ / ``22`` DOUBLE / ``23`` SNIPER / ``31`` MORTAR+；
* ``SDK/`` 里是官方的**状态与合法动作模块**（协议层需要它来枚举 ``bundles``）。
  它**不是**用来自己打比赛的：容器里没有对手、没有对战工具，本地跑一局也得不到
  任何关于人类选手的信息。唯一的评测通道是 ``.agentbench/action.json``；
* **诊断只写 stderr**。
"""

from __future__ import annotations

import _bootstrap

CONTRACT = "class AI(BaseAgent): def choose_bundle(self, state, player, bundles=None)"


def main() -> int:
    _bootstrap.install_path()
    ai_class = _bootstrap.load_ai_class(expected=CONTRACT)

    from common import BaseAgent

    if not issubclass(ai_class, BaseAgent) and not callable(
        getattr(ai_class, "create_session", None)
    ):
        _bootstrap.log(
            "[candidate] ai.py 里的 AI 既没有继承 BaseAgent，也没有 create_session()。\n"
            f"            该游戏要求：{CONTRACT}"
        )
        return 12

    import official_main

    try:
        official_main.main(ai_class)
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
