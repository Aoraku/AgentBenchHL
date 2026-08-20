"""aquawar 脚手架验收探针：最小合法 ai.py。

aquawar 的协议层是纯 Python 移植（见 ``aquawar_sdk.py``），没有官方 Python 示例
可以继承，所以这里写一个"永远合法"的最小策略：选前 4 条鱼、从不断言、
每回合用第一条存活的鱼普通攻击第一个存活敌人。
"""

from aquawar_sdk import Action, AIClient


class AI(AIClient):
    def pick(self, game):
        return [1, 2, 3, 4]

    def assert_fish(self, game):
        return (-1, -1)

    def act(self, game):
        allies = game.living_allies()
        enemies = game.living_enemies()
        return Action(
            type=0,
            my_pos=allies[0] if allies else 0,
            enemy_target=enemies[0] if enemies else 0,
        )
