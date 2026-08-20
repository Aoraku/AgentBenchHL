# 候选契约：rollman（非对称，角色轨：rollman、ghost）

> 由 `scripts/gen_candidate_support.py` 从 `_shared/candidate_runners/rollman.py`
> 的模块文档生成，**与实际执行的入口代码同源**。
> `candidate_interface` = `AI.decide`

rollman 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

**这是本仓唯一的非对称游戏**：两个座位玩的不是同一个游戏，官方也发了两份 SDK，
本候选包里**两份都在**：

* ``rollman_sdk/``（吃豆人）：每回合交 **1** 个方向 → ``pacman_to_judger(op)``
* ``ghost_sdk/``（幽灵）：每回合交 **3** 个方向 → ``ghost_to_judger(op1, op2, op3)``

本入口会读环境变量 ``AGENTBENCH_ROLE``（``rollman`` 或 ``ghost``，由对战器下发）
决定这一局用哪一套，并把角色写到你的实例上（``self.role``）。

为什么必须靠环境变量：判题器开局只给一个座位号，而官方 SDK 里 ``id == 0`` 和
``else`` 两个分支调用的是**同一个**提交函数——座位号只决定"先发还是后发"，
不决定角色。角色是选手**自己在协议里声明的**（``role: 0`` 吃豆人 / ``role: 1`` 幽灵）。
人类选手当年知道自己交的是哪一轨；你在一次 run 里要用同一份代码打两个角色，
所以由对战器显式告诉你。

你要写的（``ai.py``）::

    from core.gamedata import GameState

    class AI:
        def decide(self, game_state: GameState) -> list[int]:
            # 方向取值 0..4（0 = 不动）
            if self.role == "ghost":
                return [0, 0, 0]      # 三只幽灵各一个方向
            return [0]                # 吃豆人一个方向

要点：
* ``self.role`` 由入口注入，取值 ``"rollman"``（吃豆人）或 ``"ghost"``；
  也可以自己读 ``os.environ["AGENTBENCH_ROLE"]``；
* **返回长度必须与角色匹配**：吃豆人 1 个、幽灵 3 个。长度不对会被官方
  ``ai_to_judger`` 的断言拦下（``assert 0 <= op <= 4`` / 索引越界），整局作废；
* ``core/`` 是官方本地仿真（``PacmanEnv``），``game_state`` 是它的快照；
  两套 SDK 的 ``core`` 略有差异（幽灵轨的 ``core/ghost.py`` / ``core/utils.py``
  是幽灵专用版），入口只把**你这一局**那套放进 ``sys.path``；
* 协议层完全由官方 SDK 负责（``utils/utils.py`` 的 ``write_to_judger`` 写
  ``[4 字节大端无符号长度][JSON]``；关卡切换、``env.step`` 同步都在官方入口里）；
* **诊断只写 stderr**。
