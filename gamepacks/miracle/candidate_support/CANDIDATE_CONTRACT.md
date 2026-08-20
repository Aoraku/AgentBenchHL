# 候选契约：miracle

> 由 `scripts/gen_candidate_support.py` 从 `_shared/candidate_runners/miracle.py`
> 的模块文档生成，**与实际执行的入口代码同源**。
> `candidate_interface` = `AI.play`

miracle 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

协议层**完全由官方 SDK 负责**（``ai_client.py`` 的 ``send_opt`` / ``read_opt``：
长度前缀 + JSON）。官方的用法是"继承 ``AiClient``，覆盖 ``choose_cards`` 与 ``play``"，
回合循环是 ``while True: update_game_info(); play()``。本入口就照这个跑。

你要写的（``ai.py``）::

    from ai_client import AiClient

    class AI(AiClient):
        def choose_cards(self):
            # 必须先定好卡组与神器，然后调用 self.init()
            self.artifacts = ["HolyLight"]
            self.creatures = ["Archer", "Swordsman", "VolcanoDragon"]
            self.init()

        def play(self):
            # 本回合的全部操作，最后必须调用 self.end_round()
            self.end_round()

要点：
* ``AI`` **必须继承 ``AiClient``**，否则拿不到 ``summon`` / ``move`` / ``attack`` /
  ``use`` / ``end_round`` 这些动作函数和 ``self.players`` 等局面数据；
* ``play()`` 结尾**必须** ``self.end_round()``，否则判题器会一直等你；
* 协议层在 ``ai_client.py`` / ``card.py`` / ``gameunit.py`` / ``calculator.py``（官方原版，
  只负责通信与数据结构，不含任何策略）；
* **诊断只写 stderr**。
