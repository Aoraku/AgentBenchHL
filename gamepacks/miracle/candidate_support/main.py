"""miracle 候选入口 —— 由 gen_candidate_support.py 生成，请勿改动。

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
"""

from __future__ import annotations

import _bootstrap

CONTRACT = "class AI(AiClient): def choose_cards(self) / def play(self)"


def main() -> int:
    _bootstrap.install_path()
    ai_class = _bootstrap.load_ai_class(expected=CONTRACT)

    import ai_client
    from ai_client import AiClient

    # 官方 read_opt 读到 EOF 时不是抛 EOFError，而是拿空串去 int()::
    #
    #     data_length = read_buffer.read(6)
    #     data = read_buffer.read(int(data_length.decode()))
    #     → ValueError: invalid literal for int() with base 10: ''
    #
    # 对局正常结束、或**对手崩了导致后端关管道**时都会走到这里，于是本来该是
    # "对局结束"的情形被记成"候选抛异常"（退出码 20）。实测 fix3-miracle：
    # 对手抛 std::invalid_argument 崩掉 → 我们候选跟着报 ValueError → 那一局
    # 的诊断变成 returncodes=[None, 20, None]，看起来像候选的错，其实是被牵连的。
    #
    # 这里只把"读到空 = EOF"这一种情况翻译成 EOFError（下面 except 会干净收尾），
    # 其余 ValueError 照旧冒出去——不能把真的解析错误也吞掉。
    official_read_opt = ai_client.read_opt

    def read_opt_eof_aware() -> object:
        buffer = ai_client.sys.stdin.buffer
        header = buffer.read(6)
        if not header:
            raise EOFError("backend closed stdin")
        body = buffer.read(int(header.decode()))
        return ai_client.json.loads(body)

    ai_client.read_opt = read_opt_eof_aware

    if not issubclass(ai_class, AiClient):
        _bootstrap.log(
            "[candidate] ai.py 里的 AI 没有继承 AiClient，拿不到任何动作函数。\n"
            f"            该游戏要求：{CONTRACT}"
        )
        return 12

    # 注意 AiClient.__init__ 自己就会 read_opt()（读阵营），所以上面的 EOF 补丁
    # 必须在 construct 之前装好；而 construct/choose_cards 也可能撞上 EOF
    # （对手开局就崩、后端提前关管道），那同样是"对局结束"而不是候选的错。
    try:
        agent = _bootstrap.construct(ai_class)
        _bootstrap.guard(agent.choose_cards, what="choose_cards()（选卡组并调用 self.init()）")
    except (EOFError, BrokenPipeError):
        _bootstrap.log("[candidate] 开局阶段 stdin 就关闭了（通常是对手崩了）")
        return 0

    # 安全网：包一层记录"这一回合有没有收尾"。包的是**实例属性**，
    # 所以 ai.py 里 self.end_round() 走的就是这个包装，官方 AiClient 不用改。
    ended = False
    official_end_round = agent.end_round

    def end_round_once(*args: object, **kwargs: object) -> object:
        nonlocal ended
        if ended:
            # 一个回合只允许一条 endround。多发的那条会被后端当成**下一回合**的
            # 操作读掉，于是那一回合凭空被结束——静默且极难查，所以这里挡住。
            _bootstrap.log("[candidate] play() 重复调用了 end_round()，已忽略多余的那次")
            return None
        ended = True
        return official_end_round(*args, **kwargs)

    agent.end_round = end_round_once  # type: ignore[method-assign]

    try:
        while True:
            agent.update_game_info()
            ended = False
            agent.play()
            if not ended:
                # play() 没收尾。不补的话后端会一直等，整局作废（见模块头详注）。
                _bootstrap.log(
                    "[candidate] play() 返回时没有调用 end_round()，框架已代为结束本回合。"
                    "这一回合等于什么都没做——请在 play() 的**每条**返回路径上收尾。"
                )
                official_end_round()
                ended = True
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


if __name__ == "__main__":
    raise SystemExit(main())
