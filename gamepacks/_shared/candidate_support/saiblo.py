"""Saiblo 长度前缀 stdio 传输层 —— **8 个游戏逐字节相同**，不要改。

为什么这一层能统一
------------------
本仓 8 个游戏都沿用 Saiblo 的判题协议。选手侧只有两件事：

* **读**：从 stdin 读原始字节流。判题器发给选手的是**裸载荷**（不带长度前缀），
  各游戏载荷格式不同（snakego 是定长二进制记录，antwar2 是 JSON 文本），
  由同目录 ``protocol.py`` 负责解释。
* **写**：往 stdout 写 ``[len: 4 bytes big-endian signed][body]``。

因此"帧的搬运"可以完全统一，"帧的含义"必须逐游戏实现。这也是候选脚手架的分层：

    main.py     统一   加载 ai.AI，跑会话循环
    session.py  统一   读 → 解码 → 问 AI → 编码 → 写
    saiblo.py   统一   本文件：字节搬运 + 优雅退出 + 诊断落 stderr
    protocol.py 逐游戏 载荷编解码（唯一需要按游戏写的东西）
    common.py   逐游戏 观测/动作的数据结构

纪律：**所有诊断只能写 stderr**。stdout 是协议通道，往里打印一个字节就会让判题器
把你当成非法输出——这是"0 回合判负"最常见的自伤方式。
"""

from __future__ import annotations

import sys

BYTEORDER = "big"
LENGTH_BYTES = 4
MAX_FRAME = 1 << 24  # 16 MB：超过必然是协议错位，不要盲目分配内存


def log(message: str) -> None:
    """诊断输出。**只能**写 stderr，且立即 flush（崩溃时才看得到）。"""

    print(message, file=sys.stderr, flush=True)


class Channel:
    """选手侧的字节通道。"""

    def __init__(self) -> None:
        self._stdin = sys.stdin.buffer
        self._stdout = sys.stdout.buffer

    # ---------------------------------------------------------------- 读
    def read_exact(self, count: int) -> bytes | None:
        """读满 ``count`` 字节；流结束返回 ``None``（正常终止，不是错误）。"""

        if count <= 0:
            return b""
        chunks: list[bytes] = []
        remaining = count
        while remaining > 0:
            chunk = self._stdin.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read_length_prefixed(self) -> bytes | None:
        """读一个 ``[len][body]`` 帧（部分游戏的判题器这样发给选手）。"""

        header = self.read_exact(LENGTH_BYTES)
        if header is None:
            return None
        length = int.from_bytes(header, BYTEORDER, signed=True)
        if length < 0 or length > MAX_FRAME:
            log(f"[saiblo] 帧长非法：{length}，协议已错位")
            return None
        return self.read_exact(length) if length else b""

    def read_line(self) -> bytes | None:
        """读一行（JSON 文本型载荷的游戏用）。"""

        line = self._stdin.readline()
        return line.rstrip(b"\r\n") if line else None

    # ---------------------------------------------------------------- 写
    def write_frame(self, body: bytes) -> None:
        """写 ``[len: 4 BE signed][body]``，这是**所有游戏**统一的输出格式。"""

        self._stdout.write(len(body).to_bytes(LENGTH_BYTES, BYTEORDER, signed=True))
        self._stdout.write(body)
        self._stdout.flush()
