# 候选契约：lostspace

> 由 `scripts/gen_candidate_support.py` 从 `_shared/candidate_runners/lostspace.py`
> 的模块文档生成，**与实际执行的入口代码同源**。
> `candidate_interface` = `AI.play`

lostspace 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

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
