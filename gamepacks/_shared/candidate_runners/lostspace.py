"""lostspace 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

协议层**完全由官方 SDK 负责**（``lostspace_sdk.py``，生成时从官方单文件 ``main.py``
改名而来，内容未改）。它的收发是**不对称**的，这点很反直觉但必须尊重：

* 读：``int(str(stdin.read(4), "utf-8"))`` —— 长度前缀是 **4 字节 ASCII 十进制文本**；
* 写：``len.to_bytes(4, "big", signed=True)`` —— 长度前缀是 **4 字节二进制**。

官方用法是"继承 ``AIClient``，覆盖 ``play``"，主循环 ``run()`` 按消息 ``type``
分派（``id`` / ``roundbegin`` / ``offround`` / 其它事件），你不用碰。

你要写的（``ai.py``）::

    from lostspace_sdk import AIClient

    class AI(AIClient):
        def play(self):
            # 本玩家回合内的操作；不需要自己调 end_turn()，官方 start_turn() 会调
            self.test_move()

要点：
* **四人局**，名次分 4→1；``self.others`` 是另外三人（已按 player_id 排序）；
* 玩家状态：``ALIVE=0`` / ``DEAD=1`` / ``ESCAPED=2`` / ``SKIPPED=3``；
* 地图坐标是 **zyx 顺序**（``self.map.node[z][x][y]``），``edges`` 的方向是
  从上开始顺时针 1–8，``able=0`` 表示已被缩圈；
* 区域类型：0 普通 / 1 物资点 / 2 密钥机 / 3 电梯 / 4 逃生舱；
* ``lostspace_sdk.py`` 里 ``AIClient.play`` 自带一份可用的示例实现，可以照着改；
* **诊断只写 stderr**。
"""

from __future__ import annotations

import _bootstrap

CONTRACT = "class AI(AIClient): def play(self)"


def main() -> int:
    _bootstrap.install_path()
    ai_class = _bootstrap.load_ai_class(expected=CONTRACT)

    from lostspace_sdk import AIClient

    if not issubclass(ai_class, AIClient):
        _bootstrap.log(
            "[candidate] ai.py 里的 AI 没有继承 AIClient（来自 lostspace_sdk），\n"
            "            这样既拿不到动作函数，也不会有协议循环。\n"
            f"            该游戏要求：{CONTRACT}"
        )
        return 12

    agent = _bootstrap.construct(ai_class)

    try:
        agent.run()
    except (EOFError, BrokenPipeError, ValueError):
        # ValueError：对局结束时判题器关闭 stdin，官方 receive_data 会拿到空串再 int()。
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
