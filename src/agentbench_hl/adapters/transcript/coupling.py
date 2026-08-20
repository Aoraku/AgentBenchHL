"""公共随机流耦合：让"用了 `random` 的选手"也能被诚实测量。

问题
----

行为信息增益的前提是"策略是观测流 → 动作的确定性函数"。可是**大量真实选手会调
`random`**（本仓 miracle 的官方示例策略就有一处未播种的 `random.randint`）。对这类选手，
参考版本连自己都复现不了，于是每一轮都只能记 null——指标形同废掉。

做法
----

用**公共随机数**（common random numbers）耦合两次运行：在选手进程真正开始跑之前，
把 ``random``（以及 ``numpy.random``，如果它 import 得到）播种成同一个值。
录制那一局和之后的两次重放用**同一个种子**，于是：

* 参考版本能复现自己 ⇒ 确定性自校验通过；
* 参考与候选在第 i 个决策上共享同一条随机流 ⇒ 两者的动作差异来自**策略变化**，
  而不是来自两次抽样的运气。

这会把估计对象从"两个随机策略之间的 KL"改成"在一条公共随机流下的策略偏离"。
这是标准的方差缩减手段，但它是一个**测量约定**，所以必须显式声明：
种子写进流水文件头，口径 (``coupling``) 一路上报到事件与曲线，绝不偷偷做。

覆盖不到的非确定性
------------------

墙上时钟、进程号、线程调度、外部 I/O 顺序都不受种子影响。这类选手仍然会在确定性
自校验那一步被抓出来并记 null——这是对的：那种情况下"同一决策上下文"根本无法复现。
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

#: 公共随机流耦合：进程启动时把 random / numpy.random 播成同一个种子。
COUPLING_COMMON_RANDOM = "common_random_seed"
#: 不做任何干预（只固定 PYTHONHASHSEED）。用了 random 的选手会因此测不出来。
COUPLING_NONE = "none"

COUPLING_MODES = (COUPLING_COMMON_RANDOM, COUPLING_NONE)

#: 播种引导代码。写成 ``python -c`` 而不是往候选目录里塞文件，因为候选快照必须保持
#: 只读且逐字节可比——重放用的就是那份原始快照。
#:
#: ``argv[1]`` = 种子，``argv[2]`` = 真入口的文件名，其余参数原样传给入口。
BOOTSTRAP = """import runpy, sys, random
_seed = int(sys.argv[1])
_entry = sys.argv[2]
random.seed(_seed)
try:
    import numpy.random as _nr
except Exception:
    pass
else:
    _nr.seed(_seed % (2 ** 32))
sys.argv = [_entry] + sys.argv[3:]
runpy.run_path(_entry, run_name="__main__")
"""


def coupled_argv(
    entry: str,
    *,
    coupling: str,
    seed: int | None,
    python: str | None = None,
    args: Sequence[str] = (),
) -> list[str]:
    """构造启动选手进程的命令行。

    ``coupling`` 不是公共随机流、或没有种子时，退化为直接跑入口（行为与不耦合时一致）。
    """

    executable = python or sys.executable
    if coupling == COUPLING_COMMON_RANDOM and seed is not None:
        return [executable, "-u", "-c", BOOTSTRAP, str(int(seed)), entry, *args]
    return [executable, "-u", entry, *args]


def normalize_coupling(value: object) -> str:
    """把配置值收敛到合法口径；非法值直接报错而不是静默降级。"""

    text = str(value or COUPLING_COMMON_RANDOM).strip() or COUPLING_COMMON_RANDOM
    if text not in COUPLING_MODES:
        raise ValueError(
            f"behavioral_ig_coupling must be one of {COUPLING_MODES}, got {value!r}"
        )
    return text
