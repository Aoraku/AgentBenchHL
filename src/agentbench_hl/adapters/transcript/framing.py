"""Saiblo 选手帧的解析 —— ``[len:4 big-endian signed][body]``。

本仓 8 个游戏的判题器都用同一套传输层读选手输出（``struct.unpack(">i", head)`` 后
再读 ``len`` 个字节），所以这一段解析对所有游戏通用，并且由
``games/<game>/decision_space.yaml`` 的 ``information_gain.wire.player_frame``
声明为 ``length_prefixed_i32_be``。

这里刻意**不做容错猜测**：一旦长度头不合法就返回错误让上层记 null，而不是跳过几个
字节继续对齐——错位对齐会产出一串看起来正常、其实毫无意义的动作 token。
"""

from __future__ import annotations

import struct

HEADER_SIZE = 4
PLAYER_FRAME = "length_prefixed_i32_be"

#: 单帧上限。判题器侧同样有量级限制；超过它只能说明流已经错位。
MAX_FRAME_BYTES = 64 * 1024 * 1024


def split_frames(buffer: bytes) -> tuple[tuple[bytes, ...], bytes, str | None]:
    """把字节缓冲切成完整帧体。

    返回 ``(帧体元组, 尚不完整的尾巴, 错误说明或 None)``。
    """

    bodies: list[bytes] = []
    offset = 0
    total = len(buffer)
    while total - offset >= HEADER_SIZE:
        (length,) = struct.unpack(">i", buffer[offset : offset + HEADER_SIZE])
        if length < 0 or length > MAX_FRAME_BYTES:
            return (
                tuple(bodies),
                buffer[offset:],
                f"illegal player frame length {length} at byte offset {offset}",
            )
        end = offset + HEADER_SIZE + length
        if end > total:
            break
        bodies.append(buffer[offset + HEADER_SIZE : end])
        offset = end
    return tuple(bodies), buffer[offset:], None


def encode_frame(body: bytes) -> bytes:
    return struct.pack(">i", len(body)) + body
