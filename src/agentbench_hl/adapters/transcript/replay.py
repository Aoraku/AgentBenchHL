"""在冻结的观测流上重放一个候选进程，取出它的动作序列。

这是"行为信息增益"里唯一需要跑代码的一步：把参考版本真实对局的入站字节流原样喂给
另一个版本的 ``main.py``，收集它写回的帧。因为喂的是同一条流，被重放的策略的内部
记忆也沿着同一条参考轨迹演化，所以两边的第 i 个决策处在**同一个决策上下文**上。

为什么不能简单用 ``communicate(timeout=...)``
--------------------------------------------

很多官方 SDK 在终局后**不退出**，而是 ``while True: pass`` 自旋（snakego 就是这样，
A 仓对战器的正常收尾就是直接 SIGTERM 选手）。``communicate`` 要等到进程退出或超时，
于是每次重放都会白白烧满整个超时（一个 case 两次重放 = 半小时纯浪费，还占着核）。

所以这里自己管收流：拿到**预期数量的帧**且短暂静默后就收工，或者足够长时间没有任何
新字节就收工，两者都不满足才认超时。三种收尾方式都会如实记进 ``stop_reason``。

刻意不做的事
------------

* 不按回合交替喂：一次性喂完 + 并发收流，既避免死锁，也避免我们替游戏猜"这一帧该
  回复了没有"。若某个选手依赖读写时序，确定性校验会在参考版本自我重放那一步就失败，
  于是这一轮的 behavioral_ig 记 null（而不是给出一个错的数）。
* 不解释帧体语义：动作 token 就是帧体的 sha256。
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.adapters.transcript.coupling import coupled_argv
from agentbench_hl.adapters.transcript.framing import split_frames
from agentbench_hl.adapters.transcript.reader import WireTranscript

#: 拿够预期帧数之后再等这么久，确认它不会再吐新帧。
SETTLE_S = 1.0
#: 一直没拿够预期帧数时，允许的最长"完全没有新字节"的时间。
QUIET_S = 30.0
_POLL_S = 0.05


@dataclass(frozen=True)
class ReplayOutcome:
    action_bodies: tuple[bytes, ...]
    returncode: int | None
    timed_out: bool = False
    error: str | None = None
    #: ``exited`` / ``expected_frames`` / ``quiet`` / ``timeout`` —— 收流是怎么结束的。
    stop_reason: str = "exited"
    #: 重放实际花了多久（秒）。用于核算"测一次行为 IG 有多贵"。
    elapsed_s: float = 0.0

    @property
    def action_tokens(self) -> tuple[str, ...]:
        import hashlib  # noqa: PLC0415 - 仅在取 token 时需要

        return tuple(hashlib.sha256(body).hexdigest() for body in self.action_bodies)


def _spawn(
    root: Path, argv: Sequence[str], env: Mapping[str, str]
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 - 参数由本进程构造，非外部输入
        list(argv),
        cwd=str(root),
        env=dict(env),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def _reap(process: subprocess.Popen[bytes]) -> None:
    """自旋型选手不会自己退出，收流结束后必须主动收掉，别留孤儿占核。"""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def replay_actions(
    candidate_root: str | Path,
    transcript: WireTranscript,
    *,
    timeout_s: float,
    command_prefix: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
    expected_frames: int | None = None,
    settle_s: float = SETTLE_S,
    quiet_s: float = QUIET_S,
) -> ReplayOutcome:
    """用录制到的入站流驱动 ``candidate_root/main.py``，返回它写出的动作帧。

    ``expected_frames`` 一般传参考版本的决策数：拿够之后只再等 ``settle_s``，
    这样正常情况下重放几乎不浪费时间（否则自旋型 SDK 会把 ``timeout_s`` 烧干）。
    """

    root = Path(candidate_root)
    if not (root / "main.py").is_file():
        return ReplayOutcome((), None, error=f"candidate has no main.py: {root}")

    # 复现录制时子进程看到的环境：rollman 的 AGENTBENCH_ROLE、以及固定的
    # PYTHONHASHSEED 都必须一致，否则"同一策略"两次可能给出不同动作。
    env = dict(transcript.env)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if extra_env:
        env.update(extra_env)

    # 随机流耦合口径也必须沿用录制时那一份（种子记在流水文件头里）。
    argv = [
        *command_prefix,
        *coupled_argv(
            "main.py",
            coupling=transcript.coupling,
            seed=transcript.random_seed,
            python=sys.executable,
        ),
    ]
    try:
        process = _spawn(root, argv, env)
    except OSError as error:
        return ReplayOutcome((), None, error=f"replay could not start: {error}")

    assert process.stdin is not None
    assert process.stdout is not None
    collected = bytearray()
    lock = threading.Lock()
    finished = threading.Event()

    def drain() -> None:
        stream = process.stdout
        assert stream is not None
        while True:
            chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
            if not chunk:
                break
            with lock:
                collected.extend(chunk)
        finished.set()

    def feed() -> None:
        stream = process.stdin
        assert stream is not None
        try:
            stream.write(transcript.inbound)
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            stream.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    reader = threading.Thread(target=drain, daemon=True)
    writer = threading.Thread(target=feed, daemon=True)
    started = time.monotonic()
    reader.start()
    writer.start()

    stop_reason = "exited"
    last_size = 0
    last_change = started
    enough_since: float | None = None
    while True:
        if finished.is_set():
            stop_reason = "exited"
            break
        now = time.monotonic()
        with lock:
            size = len(collected)
            frames, _, _ = split_frames(bytes(collected))
        if size != last_size:
            last_size = size
            last_change = now
        if expected_frames is not None and len(frames) >= expected_frames:
            # 拿够了：再给一小段静默期，确认它不会再吐新帧，然后收工。
            enough_since = enough_since or now
            if now - max(enough_since, last_change) >= settle_s:
                stop_reason = "expected_frames"
                break
        else:
            enough_since = None
        if now - last_change >= quiet_s and size:
            stop_reason = "quiet"
            break
        if now - started >= timeout_s:
            stop_reason = "timeout"
            break
        time.sleep(_POLL_S)

    _reap(process)
    finished.wait(timeout=2.0)
    with lock:
        payload = bytes(collected)
    elapsed = time.monotonic() - started

    bodies, remainder, framing_error = split_frames(payload)
    error = framing_error
    if error is None and remainder and stop_reason in ("exited", "expected_frames"):
        # 尾巴意味着最后一帧没写完（通常是崩溃）。已完整的帧仍然可用，如实记一句。
        error = f"replay ended mid-frame with {len(remainder)} trailing bytes"
    return ReplayOutcome(
        action_bodies=bodies,
        returncode=process.returncode,
        timed_out=stop_reason == "timeout",
        error=error,
        stop_reason=stop_reason,
        elapsed_s=round(elapsed, 3),
    )
