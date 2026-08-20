"""antwar2 脚手架验收探针：最小合法 ai.py。

``BaseAgent`` 是 ABC，唯一抽象方法是 ``choose_bundle``，所以必须实现它
（这正是探针要验证的约束之一）。``list_bundles`` 由官方 ``ActionCatalog`` 枚举出
**合法**动作包，取第一个即可，不含任何策略。
"""

from common import BaseAgent


class AI(BaseAgent):
    def choose_bundle(self, state, player, bundles=None):
        bundles = bundles or self.list_bundles(state, player)
        return bundles[0]
