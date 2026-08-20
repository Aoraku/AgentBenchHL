"""generals 脚手架验收探针：最小合法 ai.py。

空操作列表 = 本回合什么都不做直接结束（官方 ``finish_and_send_our_ops`` 会补上
回合结束标记 ``8``，探针不需要也不应该自己加）。
"""


class AI:
    def decide(self, round, my_seat, state):
        return []
