"""rollman 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

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
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import _bootstrap

CONTRACT = "def decide(self, game_state: GameState) -> list[int]  # 吃豆人 1 个 / 幽灵 3 个方向"

# 角色 → 该轨官方 SDK 所在子目录（与 gen_candidate_support.py 的 sdk_prefix 一致）
SDK_DIRS = {"rollman": "rollman_sdk", "ghost": "ghost_sdk"}
ROLE_ENV = "AGENTBENCH_ROLE"


def resolve_role(here: Path) -> str | None:
    """确定这一局的角色。拿不到就返回 None（由调用方给出可执行诊断）。"""

    role = (os.environ.get(ROLE_ENV) or "").strip()
    if role in SDK_DIRS:
        return role
    # 只装了一套 SDK 时无歧义（例如手工裁剪过的包），直接用那一套。
    present = [name for name, directory in SDK_DIRS.items() if (here / directory).is_dir()]
    if len(present) == 1:
        return present[0]
    return None


def main() -> int:
    here = Path(_bootstrap.install_path())

    role = resolve_role(here)
    if role is None:
        _bootstrap.log(
            f"[candidate] 无法确定角色：环境变量 {ROLE_ENV} 缺失或非法"
            f"（当前值 {os.environ.get(ROLE_ENV)!r}，应为 rollman 或 ghost），"
            f"且候选包里同时存在多套 SDK。对战器应当下发该变量；"
            f"本地自测时可以自己 export。"
        )
        return 14

    sdk_dir = here / SDK_DIRS[role]
    if not sdk_dir.is_dir():
        _bootstrap.log(f"[candidate] 角色 {role} 对应的官方 SDK 目录不存在：{sdk_dir}")
        return 14
    # 只把**本局这一轨**的 SDK 放进 sys.path：两套都有 core/、utils/、
    # ai_to_judger.py，同时可见会让 import 取到错误的那一套（幽灵轨的
    # core/ghost.py 与吃豆人轨不同），表现为行为诡异而不是报错。
    sys.path.insert(0, str(sdk_dir))

    ai_class = _bootstrap.load_ai_class(expected=CONTRACT)
    agent = _bootstrap.construct(ai_class)
    # 角色注入：让 ai.py 用 self.role 分支，而不是去猜自己是谁。
    try:
        agent.role = role
    except AttributeError:  # 极少数用 __slots__ 的实现
        _bootstrap.log(f"[candidate] 无法设置 self.role（{ROLE_ENV}={role}），请自行读该环境变量")

    decide = getattr(agent, "decide", None)
    if not callable(decide):
        _bootstrap.log(f"[candidate] AI 缺少 decide 方法。该游戏要求：{CONTRACT}")
        return 12

    # official_main 是官方 SDK 自带的入口（生成时从 main.py 改名而来，内容未改）。
    # 它顶层有 `from ai import *`，此时 ai 模块已由 _bootstrap 导入并缓存。
    import official_main

    # 官方 Controller.__init__ 第一行是 `int(input())` 读座位号。这里**不能**提前
    # 把它读掉（否则官方就读不到了），所以原样交给官方处理——角色信息我们已经
    # 从环境变量拿到了，不需要偷看 stdin。
    controller = _bootstrap.guard(
        official_main.Controller, what="读取座位号（官方 Controller 初始化）"
    )

    try:
        controller.run(decide)
    except (EOFError, BrokenPipeError):
        _bootstrap.log("[candidate] 对局结束（stdin 关闭）")
        return 0
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        import traceback

        _bootstrap.log(
            f"[candidate] 对局中抛异常：{type(error).__name__}: {error}\n" + traceback.format_exc()
        )
        return 20
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
