"""候选包的公共引导层 —— **所有游戏逐字节相同**，不要改。

职责只有两件事，都是被线上事故逼出来的：

1. **让 ``import`` 可靠**：对战器以 ``python main.py`` 启动候选，但工作目录不保证是
   候选目录，官方 SDK 内部又普遍用平铺 import（``from adk import ...`` /
   ``from core.gamedata import ...``）。这里把候选目录插到 ``sys.path`` 首位。
2. **让失败可诊断**：历史上有一次 antwar2 的 run 连烧 5 轮、每轮都"0 回合判负"，
   根因是候选包没有 ``ai.py``，而入口第一行是 ``from ai import AI``——进程启动即死，
   反馈里却只有"你输了"。所以这里把 ImportError 变成一条**可执行的**诊断：
   列出当前目录有哪些 ``.py``，并明确说出约定。

纪律：**诊断只能写 stderr**。stdout 是协议通道，往里打印一个字节就会被判题器当成
非法输出——这是"0 回合判负"最常见的自伤方式。
"""

from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def log(message: str) -> None:
    """诊断输出。**只能**写 stderr，且立即 flush（崩溃时才看得到）。"""

    print(message, file=sys.stderr, flush=True)


def install_path() -> str:
    """把候选目录放到 ``sys.path`` 首位并切换工作目录，返回该目录。

    为什么还要 ``chdir``：官方 SDK 里有用**相对路径**读数据文件的写法
    （miracle 的 ``card.py``：``json.load(open("Data.json"))``）。对战器启动候选时
    工作目录并不保证是候选目录，于是 import 阶段就 ``FileNotFoundError``，
    而对战器那边只看到"0 回合判负"。切到候选目录后这类相对路径才成立。
    """

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        if os.getcwd() != HERE:
            os.chdir(HERE)
    except OSError as error:  # 只在极端沙箱下发生，不该让整局作废
        log(f"[candidate] 无法切换工作目录到 {HERE}：{error}")
    return HERE


def load_ai_class(*, expected: str):
    """从候选自己写的 ``ai.py`` 里取 ``class AI``。

    这是 8 个游戏**唯一统一的约定**：文件名 ``ai.py``、类名 ``AI``。
    类里该实现哪些方法由游戏决定（见该 GamePack 的 ``sdk_interface.md``），
    因为观测/动作是官方 SDK 的原生对象，重新包装一层只会引入翻译错误。
    """

    install_path()
    try:
        import ai as ai_module
    except ImportError as error:
        log(
            f"[candidate] 无法导入 ai.py：{error}\n"
            f"            候选包必须包含 ai.py，且在其中定义 class AI（{expected}）。\n"
            f"            当前目录下的 .py 文件："
            + ", ".join(sorted(name for name in os.listdir(HERE) if name.endswith(".py")))
        )
        raise SystemExit(10) from error
    except Exception as error:  # noqa: BLE001 - ai.py 顶层代码崩了也要留全栈
        log(f"[candidate] 导入 ai.py 时抛异常：{type(error).__name__}: {error}\n" + traceback.format_exc())
        raise SystemExit(11) from error

    ai_class = getattr(ai_module, "AI", None)
    if ai_class is None:
        log(
            "[candidate] ai.py 里没有 class AI。\n"
            f"            该游戏要求：{expected}\n"
            "            ai.py 里现有的名字：" + ", ".join(sorted(vars(ai_module)))
        )
        raise SystemExit(12)
    return ai_class


def construct(ai_class, *args, **kwargs):
    """构造 AI 实例，构造失败时给出可执行原因。"""

    try:
        return ai_class(*args, **kwargs)
    except Exception as error:  # noqa: BLE001
        log(
            f"[candidate] AI() 构造失败：{type(error).__name__}: {error}\n"
            + traceback.format_exc()
        )
        raise SystemExit(13) from error


def guard(action, *, what: str):
    """跑一段"必须成功"的逻辑，失败时把全栈写到 stderr 再退出。

    退出码的区分很重要（``candidate_preflight`` 靠它判断"是包坏了还是只是没输入"）：

    * ``14``：候选包自身的问题（前置步骤真的执行失败）；
    * ``20``：已经开始跟判题器交互但没拿到输入（``EOFError`` / 管道断开）。
      有些游戏的前置步骤本身就要读输入——rollman 的官方 ``Controller.__init__``
      第一行就是 ``int(input())`` 读座位号，miracle 的 ``choose_cards()`` 要跟
      判题器握手。空跑一次候选时它们必然 EOF，那不是缺陷。
    """

    try:
        return action()
    except (EOFError, BrokenPipeError):
        log(f"[candidate] {what} 时 stdin 已结束（判题器未下发输入）")
        raise SystemExit(20) from None
    except Exception as error:  # noqa: BLE001
        log(
            f"[candidate] {what} 失败：{type(error).__name__}: {error}\n"
            + traceback.format_exc()
        )
        raise SystemExit(14) from error
