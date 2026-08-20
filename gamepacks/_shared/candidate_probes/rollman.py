"""rollman 脚手架验收探针：最小合法 ai.py（两个角色轨共用一份）。

方向 0 = 不动，任何局面下都合法（``ai_to_judger`` 的断言是 ``0 <= op <= 4``）。
但**返回长度必须与角色匹配**：吃豆人 1 个方向、幽灵 3 个（三只幽灵各一个）。

这份探针故意**只依赖 ``self.role``**，不去读环境变量、也不猜。这样它同时验证了
两件事：脚手架能跑，以及对战器真的把角色下发到了候选进程
（A 侧 rollman evaluator 通过 ``ProcessSpec.env`` 注入 ``AGENTBENCH_ROLE``，
入口再写到 ``self.role``）。角色缺失时这里会抛异常而不是蒙一个长度——
蒙对了会掩盖注入链路的断裂。
"""


class AI:
    def decide(self, game_state):
        role = getattr(self, "role", None)
        if role == "ghost":
            return [0, 0, 0]
        if role == "rollman":
            return [0]
        raise RuntimeError(
            f"角色未注入（self.role={role!r}）：对战器应当通过 AGENTBENCH_ROLE 下发"
        )
