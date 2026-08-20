"""候选会话驱动 —— **8 个游戏逐字节相同**，不要改。

职责：把"字节帧"变成"回合循环"，并把所有可预见的失败变成**stderr 上的可执行诊断**，
而不是让进程静默崩掉（静默崩掉在对战器里表现为"0 回合判负"，反馈里只有"你输了"，
迭代会卡在同一个坑里）。

与 ``protocol.py`` 的契约（每个游戏必须实现）::

    def handshake(channel) -> object | None
        # 读开局配置，返回"局面上下文"（没有配置阶段的游戏返回 None）

    def next_observation(channel, context) -> Observation | None
        # 读下一帧；返回 None 表示对局结束

    def encode(action, context) -> bytes | None
        # 把 AI 的动作编成载荷；返回 None 表示这一帧不需要回复

与 ``ai.py`` 的契约（**你要写的就是这个**）::

    class AI:
        def decide(self, observation) -> action

``decide`` 抛异常时：本会话把异常写到 stderr 并回退到 ``protocol.fallback_action``
（如果该游戏提供了），保证不会因为一个边界条件让整局作废。
"""

from __future__ import annotations

import time
import traceback

from saiblo import Channel, log


def run(ai, protocol) -> int:
    """跑完一局。返回进程退出码（0 = 正常结束）。"""

    channel = Channel()
    try:
        context = protocol.handshake(channel)
    except Exception:  # noqa: BLE001 - 握手失败要留全栈，否则无从排查
        log("[session] 握手失败：\n" + traceback.format_exc())
        return 2

    fallback = getattr(protocol, "fallback_action", None)
    turn = 0
    started = time.time()
    while True:
        try:
            observation = protocol.next_observation(channel, context)
        except Exception:  # noqa: BLE001
            log("[session] 读取局面失败：\n" + traceback.format_exc())
            return 3
        if observation is None:
            log(f"[session] 对局结束：共 {turn} 帧，用时 {time.time() - started:.1f}s")
            return 0

        turn += 1
        try:
            action = ai.decide(observation)
        except Exception:  # noqa: BLE001
            log(f"[session] 第 {turn} 帧 AI.decide 抛异常：\n" + traceback.format_exc())
            if fallback is None:
                return 4
            action = fallback(observation, context)

        try:
            payload = protocol.encode(action, context)
        except Exception:  # noqa: BLE001
            log(f"[session] 第 {turn} 帧动作编码失败（动作={action!r}）：\n" + traceback.format_exc())
            if fallback is None:
                return 5
            payload = protocol.encode(fallback(observation, context), context)

        if payload is not None:
            channel.write_frame(payload)
