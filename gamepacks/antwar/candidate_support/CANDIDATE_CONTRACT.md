# 候选契约：antwar

> 由 `scripts/gen_candidate_support.py` 从 `_shared/candidate_runners/antwar.py`
> 的模块文档生成，**与实际执行的入口代码同源**。
> `candidate_interface` = `AI.decide`

antwar 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

协议层**完全由官方 SDK 负责**（本目录下的 ``antwar/`` 包，与人类选手当年拿到的
逐字节相同）。官方已经把通讯封装成 ``run_antwar_ai(ai_func)``，所以这里只做一件事：
把你写的 ``ai.AI.decide`` 交给它。

你要写的（``ai.py``）::

    from antwar.gamedata import Operation      # 见 antwar/ 官方 SDK
    from antwar.gamestate import GameState

    class AI:
        def decide(self, my_seat: int, game_state: GameState) -> list[Operation]:
            # 返回本回合要执行的操作列表；非法操作会被 SDK 自己过滤掉
            return []

要点：
* ``my_seat``（0/1）由判题器下发，先后手的收发顺序差异官方 SDK 已处理；
* ``game_state`` 是官方 SDK 维护的完整局面（含 ``pheromone`` 信息素——蚂蚁寻路的原因）；
* **诊断只写 stderr**。往 stdout 打印任何东西都会破坏协议帧。
