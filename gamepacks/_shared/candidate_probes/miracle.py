"""miracle 脚手架验收探针：最小合法 ai.py。

复用官方 SDK 自带的完整示例（``official_main.AI``，即官方 ``main.py`` 里那份），
它已经实现了 ``choose_cards`` 与 ``play``，并且每回合正确调用 ``end_round()``。
"""

from official_main import AI as OfficialSampleAI


class AI(OfficialSampleAI):
    """原样沿用官方示例策略。"""
