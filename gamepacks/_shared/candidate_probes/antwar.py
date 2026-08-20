"""antwar 脚手架验收探针：最小合法 ai.py。

只用来证明"脚手架 + 官方 SDK 能在我们的对战器里跑完整局"，不含任何策略。
空操作列表在 antwar 是合法的（官方 ``try_apply_our_ops`` 接受空列表）。
"""


class AI:
    def decide(self, my_seat, game_state):
        return []
