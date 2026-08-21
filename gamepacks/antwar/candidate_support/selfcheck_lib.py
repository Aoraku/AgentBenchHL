"""候选包前置校验：在烧掉一整轮对局之前，先确认它**能启动**。

为什么必须有这一步
------------------
线上真实教训：一次 antwar2 的 HL run 连跑 5 轮，全部对局是"0 回合判负"。
根因是候选包里有 ``main.py`` / ``common.py`` / ``protocol.py`` / ``strategy_core.py``，
但**没有 ``ai.py``**——而脚手架 ``main.py`` 第一行就是 ``from ai import AI``。
选手进程一启动就 ``ImportError``，对战器如约判负，于是：

* 每轮白烧 k×座次×seed 局对局（antwar2 一局十几秒到两分钟）；
* 反馈里只有"你输了"，agent 无从知道是导入错误（这个诊断链路已另行修好）；
* 5 轮迭代全部报废，曲线上是一条毫无信息的水平线。

这类失败**在跑对局之前就可判定**，成本约 1 秒。本模块就干这件事：

1. **静态检查**：入口 ``main.py`` 必须存在；若 GamePack 声明了
   ``candidate_interface: AI.xxx``，则 ``ai.py`` 必须存在且导出 ``AI``；
   所有 ``.py`` 必须能通过 ``compile()``（语法错误当场抓出）。
2. **启动检查**：真的把 ``main.py`` 拉起来，stdin 给一个**永不结束的空管道**，
   看它是否在**接触输入之前**就带着 traceback 退出。

   这里两个细节都是被误杀教训逼出来的：

   * stdin **不能给 EOF**。很多官方 SDK 在构造函数里就读判题器
     （miracle 的 ``AiClient.__init__`` 第一行是 ``read_opt()['camp']``、
     rollman 的 ``Controller.__init__`` 第一行是 ``int(input())`` 读座位号）。
     给 EOF 会让 8 个健康脚手架全部"启动即崩"。阻塞在读输入上才是健康信号。
   * 判据**不能只看"崩了"**。只认两种信号：引导层的专用退出码 ``10–14``
     （见 ``candidate_support/_bootstrap.py``：缺 ``ai.py``、``ai.py`` 顶层异常、
     接口不符、构造失败——都发生在碰输入之前），以及 stderr 里的**结构性**
     异常名（导入/语法/名字/抽象类无法实例化）。
     退出码 ``20`` 表示"已经在跟判题器交互后失败"，不算问题。

设计约束：不引入超时长的等待、不做任何网络与写盘、失败信息必须**可执行**
（直接告诉 agent 缺哪个文件、哪一行语法错、什么异常）。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ENTRY_NAME = "main.py"
# 崩溃探测窗口：ImportError / SyntaxError 都在毫秒级发生，1.5s 足够且不拖慢迭代。
STARTUP_PROBE_S = 1.5
# 与 gamepacks/_shared/candidate_support/_bootstrap.py 约定的退出码：
# 10 缺 ai.py / 11 ai.py 导入期抛异常 / 12 接口不符 / 13 AI() 构造失败 / 14 前置步骤失败。
# 这些都发生在"读取任何判题器输入之前"，因此可以确定是候选包自身的问题。
BOOTSTRAP_REJECT_CODES = frozenset({10, 11, 12, 13, 14})
# 不依赖引导层时的兜底信号：只认结构性异常，不认泛泛的 "Traceback"。
STRUCTURAL_MARKERS = (
    "ModuleNotFoundError",
    "ImportError",
    "SyntaxError",
    "IndentationError",
    "NameError",
    "AttributeError",
    "Can't instantiate abstract class",
)


@dataclass(frozen=True)
class PreflightIssue:
    candidate_id: str
    kind: str          # missing_entry / missing_interface / syntax_error / startup_crash
    detail: str

    def as_note(self) -> str:
        return f"{self.candidate_id}: [{self.kind}] {self.detail}"


def _probe_environment(root: Path) -> dict[str, str]:
    """启动探针用的环境变量。

    非对称游戏（rollman）的入口必须知道这一局的角色才能选对官方 SDK，
    真实对局里由对战器通过 ``ProcessSpec.env`` 注入 ``AGENTBENCH_ROLE``。
    但**预检不是对局**，这里没有对战器；不给的话入口会按约定退出码 14 拒绝启动，
    于是每个 rollman 候选都会被判"启动即崩"而被丢弃——候选明明是好的。

    所以从候选包结构里推断一个**占位**角色（``<track>_sdk/`` 子目录，与
    ``gen_candidate_support.py`` 的约定一致）。预检只关心"包能不能起来"，
    用哪一轨不影响结论。
    """

    environment = dict(os.environ)
    tracks = sorted(
        item.name.removesuffix("_sdk")
        for item in root.iterdir()
        if item.is_dir() and item.name.endswith("_sdk")
    )
    if tracks:
        environment.setdefault("AGENTBENCH_ROLE", tracks[0])
    return environment


def _required_interface_module(candidate_interface: str | None) -> str | None:
    """``AI.choose_operations`` ⇒ 需要 ``ai.py`` 且导出 ``AI``。

    只处理"类名.方法名"这种声明；声明为空或形态不符时返回 ``None``（不做假设）。
    """

    if not candidate_interface or "." not in candidate_interface:
        return None
    class_name = candidate_interface.split(".", 1)[0].strip()
    if not class_name.isidentifier():
        return None
    return class_name


def check_candidate(
    candidate_id: str,
    root: Path,
    *,
    candidate_interface: str | None = None,
    python_executable: str | None = None,
    probe_seconds: float = STARTUP_PROBE_S,
) -> list[PreflightIssue]:
    """返回该候选的全部前置问题；空列表 = 可以进对局。"""

    issues: list[PreflightIssue] = []
    entry = root / ENTRY_NAME
    if not entry.is_file():
        return [
            PreflightIssue(
                candidate_id,
                "missing_entry",
                f"候选目录里没有 {ENTRY_NAME}；对战器以进程方式启动选手，入口必须是 {ENTRY_NAME}",
            )
        ]

    # --- 静态：语法
    for path in sorted(root.rglob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8", errors="replace"), str(path), "exec")
        except SyntaxError as error:
            issues.append(
                PreflightIssue(
                    candidate_id,
                    "syntax_error",
                    f"{path.relative_to(root).as_posix()}:{error.lineno}: {error.msg}",
                )
            )
        except (OSError, ValueError) as error:
            issues.append(
                PreflightIssue(
                    candidate_id, "syntax_error", f"{path.relative_to(root).as_posix()}: {error}"
                )
            )

    # --- 静态：声明的接口模块
    class_name = _required_interface_module(candidate_interface)
    if class_name is not None:
        module = root / f"{class_name.lower()}.py"
        if not module.is_file():
            issues.append(
                PreflightIssue(
                    candidate_id,
                    "missing_interface",
                    f"GamePack 声明 candidate_interface={candidate_interface}，"
                    f"因此必须提供 {module.name} 并在其中导出 {class_name}；"
                    f"当前目录只有：{sorted(p.name for p in root.glob('*.py'))}",
                )
            )
        elif not re.search(
            rf"^\s*(class|{class_name}\s*=)\s*{class_name}\b",
            module.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        ):
            issues.append(
                PreflightIssue(
                    candidate_id,
                    "missing_interface",
                    f"{module.name} 里找不到 {class_name} 的定义或赋值；"
                    f"入口会执行 `from {module.stem} import {class_name}`",
                )
            )

    if issues:
        # 语法/接口都不对时不必再启动，报错已经足够可执行
        return issues

    # --- 启动：真的拉起来看会不会当场崩
    #
    # stdin 给的是一个**永不结束的空管道**（不写、不关），而不是 EOF。
    # 这一点是必须的：很多官方 SDK 在**构造函数里**就读判题器（miracle 的
    # AiClient.__init__ 第一行是 read_opt()['camp']，rollman 的 Controller
    # 第一行是 int(input()) 读座位号）。给 EOF 会让它们全部当场炸掉，
    # 于是"能不能启动"的探针把 8 个健康脚手架全判成坏的。
    # 阻塞在读输入上 = 健康：真正的 import/语法/接口错误在毫秒级就崩了。
    try:
        process = subprocess.Popen(  # noqa: S603 - 命令由本地路径构成
            (python_executable or sys.executable, ENTRY_NAME),
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_probe_environment(root),
        )
    except OSError as error:
        return [PreflightIssue(candidate_id, "startup_crash", f"无法启动 {ENTRY_NAME}：{error}")]

    try:
        returncode = process.wait(timeout=probe_seconds)
    except subprocess.TimeoutExpired:
        # 还在跑（通常是阻塞在等判题器下发第一帧）= 没有"启动即崩"，正常。
        process.kill()
        process.wait(timeout=probe_seconds)
        return issues
    finally:
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    stderr = ""
    if process.stderr is not None:
        try:
            stderr = (process.stderr.read() or "").strip()
        except OSError:
            stderr = ""
        finally:
            process.stderr.close()

    # 只认"接触输入之前就失败"的信号：引导层专用退出码，或结构性异常。
    rejected = returncode in BOOTSTRAP_REJECT_CODES
    structural = returncode != 0 and any(marker in stderr for marker in STRUCTURAL_MARKERS)
    if rejected or structural:
        issues.append(
            PreflightIssue(
                candidate_id,
                "startup_crash",
                f"{ENTRY_NAME} 启动即退出（returncode={returncode}）：{stderr[-800:]}",
            )
        )
    return issues
