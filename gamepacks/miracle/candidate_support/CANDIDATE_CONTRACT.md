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
* ``play()`` 结尾应当 ``self.end_round()``；**忘了也不会卡死**——本入口会替你补一条
  并在 stderr 里点名（理由见下），但它只是安全网，不是可以依赖的行为；
* 协议层在 ``ai_client.py`` / ``card.py`` / ``gameunit.py`` / ``calculator.py``（官方原版，
  只负责通信与数据结构，不含任何策略）；
* **诊断只写 stderr**。

为什么要替它补 end_round
------------------------
miracle 的一个回合是"若干操作 + 一条 ``endround``"，后端在 ``endround`` 到达前会
一直阻塞读 stdin。以前这份入口只是 ``while True: update_game_info(); play()``，
把"一定要收尾"整个压在 LLM 写的 ``play()`` 上 —— 而 ``play()`` 里任何一条提前
``return``（没好棋、条件不满足、异常分支）都会让对局**永久挂住**。

后果不是"这一局输了"，而是**这一局什么信息都没有**：卡到超时 → 记 0 回合 →
``result=loss`` / ``score_margin=0`` / ``evaluator_status=game_error``，
既没有回放可读，也没有分差梯度，agent 下一轮完全不知道自己错在哪。
实测 ``s8k4-miracle`` 的 ``v001_holylight_press`` 就是这样，**两个座次都**
``match timed out after 180.000s``。

8 个游戏里只有 miracle 把这件事交给候选（lostspace 的官方 SDK 自己会收尾）。
补上之后，忘记收尾的代价回归到它本来该有的样子：那一回合什么也没做（下棋很差），
而不是整局作废。
