"""录制垫片 —— 生成一个"透明中继 + 记账"的 ``main.py``。

为什么需要垫片
--------------

要在冻结的决策上下文上比较两个版本的策略，必须先拿到**参考版本真实对局的完整入站
字节流**。选手进程由 A 仓的对战器启动，环境变量被白名单过滤，我们没法从外面告诉
选手"请把观测记下来"。

但候选包的入口固定是 ``python main.py``，而候选快照是我们自己复制出来的。于是最干净
的做法是：复制一份快照，把原 ``main.py`` 改名，放一个自动生成的 ``main.py`` 垫片进去。
垫片启动真正的入口作为子进程，双向**原样**转发字节，同时把两个方向记进 jsonl。

关键性质
--------

* **透明**：只做字节转发，不解析、不改写、不重排，因此对局结果与不录制时一致；
* **通用**：不含任何游戏语义，8 个游戏共用同一个垫片；
* **可复现**：垫片给子进程固定 ``PYTHONHASHSEED=0``，并把子进程实际看到的环境记进
  文件头，重放时原样复现——否则集合/字典的哈希序会让"同一策略"两次给出不同动作，
  确定性校验必然失败；
* **崩溃可见**：边写边 flush，超时被杀也能保住已录到的部分；有 header/footer 标记，
  读取端能判断录制是否完整。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agentbench_hl.adapters.transcript.coupling import (
    BOOTSTRAP,
    COUPLING_COMMON_RANDOM,
    normalize_coupling,
)
from agentbench_hl.adapters.transcript.framing import PLAYER_FRAME

#: 原 ``main.py`` 在录制克隆里的新名字。
RECORDED_ENTRY = "_abhl_recorded_main.py"

#: 允许写进文件头的环境变量键（重放时据此精确复现选手看到的环境）。
#: 只保留对战器白名单那几个 + ``AGENTBENCH_*``（rollman 靠 AGENTBENCH_ROLE 选 SDK）。
ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SYSTEMROOT",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
)
ENV_PREFIX = "AGENTBENCH_"

#: 键名命中这些词一律不入档，避免把凭据写进实验产物。
ENV_FORBIDDEN = ("KEY", "TOKEN", "SECRET", "PASSWORD", "COOKIE")

#: 录制产物的目录布局。**只有 transcripts 需要对沙箱内的选手进程可写**；
#: 录制克隆的代码目录（snapshots）保持只读，这样"能不能写盘"这个前提在
#: 测量局与正式局之间是一致的。
TRANSCRIPT_DIRNAME = "transcripts"
SNAPSHOT_DIRNAME = "snapshots"


def transcript_root(work_root: str | Path) -> Path:
    """需要声明为沙箱可写的目录（Arena 装配处与测量流程共用同一个约定）。"""

    return Path(work_root) / TRANSCRIPT_DIRNAME


def snapshot_root(work_root: str | Path) -> Path:
    """录制克隆所在目录（只读即可）。"""

    return Path(work_root) / SNAPSHOT_DIRNAME

_SHIM_TEMPLATE = '''#!/usr/bin/env python3
# 线协议录制垫片：由 agentbench_hl.adapters.transcript.shim 自动生成，请勿手改。
# 作用：透明转发 stdin/stdout，并把双向字节按序记进 TRANSCRIPT_PATH。
from __future__ import annotations

import base64
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_PATH = {transcript_path!r}
RECORDED_ENTRY = {recorded_entry!r}
PLAYER_FRAME = {player_frame!r}
COUPLING = {coupling!r}
RANDOM_SEED = {random_seed!r}
BOOTSTRAP = {bootstrap!r}
ENV_ALLOWLIST = {env_allowlist!r}
ENV_PREFIX = {env_prefix!r}
ENV_FORBIDDEN = {env_forbidden!r}
CHUNK = 65536
MAX_FRAME = 64 * 1024 * 1024
# 每条记录都 flush 会把对局拖慢一个数量级：流水文件在沙箱里是绑定挂载的，
# 一次 fsync 级别的 flush 可达百毫秒量级，而 snakego 这种"每回合十几个小包"的
# 协议一局有上千条记录（实测 512 回合的对局被拖成 38 倍、直接撞超时）。
# 所以改成限频 flush：最多每 FLUSH_INTERVAL 秒落一次，header/footer/异常强制落。
# 代价是进程被杀时可能丢掉尾部若干条——读取端本来就按"不完整"处理，可接受。
FLUSH_INTERVAL = 0.5

_lock = threading.Lock()
_seq = [0]
_t0 = time.monotonic()
_last_flush = [_t0]

# 录制文件开不了（目录没被声明为沙箱可写、磁盘满……）时**不能把选手带崩**：
# 那会让一次测量故障伪装成候选的一场败局。这里退化成"纯透明转发"，
# 由上层因为"transcript missing"如实把 behavioral_ig 记 null。
try:
    _sink = open(TRANSCRIPT_PATH, "w", encoding="utf-8")
except OSError as _error:
    _sink = None
    sys.stderr.write("[abhl-transcript] cannot open %s: %s\\n" % (TRANSCRIPT_PATH, _error))
    sys.stderr.flush()


def _emit(record, force=False):
    if _sink is None:
        return
    with _lock:
        now = time.monotonic()
        record["seq"] = _seq[0]
        record["t"] = round(now - _t0, 6)
        _seq[0] += 1
        try:
            _sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\\n")
            if force or now - _last_flush[0] >= FLUSH_INTERVAL:
                _sink.flush()
                _last_flush[0] = now
        except (OSError, ValueError):
            pass


def _child_env():
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    kept = dict()
    for key, value in env.items():
        if any(word in key.upper() for word in ENV_FORBIDDEN):
            continue
        if key in ENV_ALLOWLIST or key.startswith(ENV_PREFIX):
            kept[key] = value
    return env, kept


def _read_some(stream):
    if hasattr(stream, "read1"):
        return stream.read1(CHUNK)
    return stream.read(CHUNK)


def _pump_inbound(destination):
    source = getattr(sys.stdin, "buffer", sys.stdin)
    while True:
        try:
            chunk = _read_some(source)
        except Exception:
            break
        if not chunk:
            break
        _emit({{"dir": "in", "b64": base64.b64encode(chunk).decode("ascii")}})
        try:
            destination.write(chunk)
            destination.flush()
        except (BrokenPipeError, ValueError, OSError):
            break
    try:
        destination.close()
    except Exception:
        pass


def _pump_outbound(source):
    target = getattr(sys.stdout, "buffer", sys.stdout)
    pending = b""
    broken = False
    while True:
        try:
            chunk = _read_some(source)
        except Exception:
            break
        if not chunk:
            break
        try:
            target.write(chunk)
            target.flush()
        except (BrokenPipeError, ValueError, OSError):
            broken = True
        pending += chunk
        while len(pending) >= 4:
            (length,) = struct.unpack(">i", pending[:4])
            if length < 0 or length > MAX_FRAME:
                _emit(
                    {{"dir": "framing_error", "detail": "illegal frame length %d" % length}},
                    force=True,
                )
                pending = b""
                break
            if len(pending) < 4 + length:
                break
            body = pending[4 : 4 + length]
            pending = pending[4 + length :]
            _emit({{"dir": "out", "b64": base64.b64encode(body).decode("ascii")}})
    if pending:
        _emit({{"dir": "trailing", "bytes": len(pending)}}, force=True)
    if broken:
        _emit({{"dir": "downstream_closed"}}, force=True)


def _child_argv():
    entry = os.path.join(HERE, RECORDED_ENTRY)
    if COUPLING == "common_random_seed" and RANDOM_SEED is not None:
        # 与 adapters/transcript/coupling.py 的 coupled_argv 必须完全一致：
        # 录制局与之后两次重放共享同一条随机流，否则确定性自校验必然失败。
        return [sys.executable, "-u", "-c", BOOTSTRAP, str(RANDOM_SEED), entry]
    return [sys.executable, "-u", entry]


def _reap(process, grace=2.0):
    """把真选手收干净。

    很多官方 SDK 在终局后是 ``while True: pass`` **自旋不退出**（snakego 就是），
    对战器的正常收尾就是直接 SIGTERM 选手进程。加了垫片之后被 SIGTERM 的是垫片，
    如果不主动收，里面那个自旋的真选手会变成**孤儿并占满一个核**，把同机其它对局
    一起拖慢。所以垫片必须保证：自己一退场，子进程跟着走。
    """

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except Exception:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.02)
    try:
        process.kill()
    except Exception:
        pass


def main():
    child_env, recorded = _child_env()
    _emit(
        {{
            "dir": "header",
            "player_frame": PLAYER_FRAME,
            "recorded_entry": RECORDED_ENTRY,
            "coupling": COUPLING,
            "random_seed": RANDOM_SEED,
            "cwd": os.getcwd(),
            "env": recorded,
        }},
        force=True,
    )
    process = subprocess.Popen(
        _child_argv(),
        cwd=os.getcwd(),
        env=child_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    writer = threading.Thread(target=_pump_inbound, args=(process.stdin,), daemon=True)
    reader = threading.Thread(target=_pump_outbound, args=(process.stdout,), daemon=True)
    writer.start()
    reader.start()

    # 对战器的正常收尾是 SIGTERM 选手进程。被 SIGTERM 的是垫片，所以这里必须接住
    # 信号并把真选手一起收掉，否则自旋型 SDK 会留下一个吃满 CPU 的孤儿进程。
    def _handle_signal(signum, _frame):
        _emit({{"dir": "signal", "signum": int(signum)}}, force=True)
        _reap(process)
        try:
            if _sink is not None:
                _sink.close()
        except Exception:
            pass
        os._exit(128 + int(signum))

    for _name in ("SIGTERM", "SIGINT", "SIGHUP"):
        _signum = getattr(signal, _name, None)
        if _signum is not None:
            try:
                signal.signal(_signum, _handle_signal)
            except (ValueError, OSError):
                pass

    code = process.wait()
    reader.join(timeout=10.0)
    _emit({{"dir": "footer", "returncode": code}}, force=True)
    try:
        if _sink is not None:
            _sink.close()
    except Exception:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
'''


def render_shim(
    *,
    transcript_path: Path,
    recorded_entry: str = RECORDED_ENTRY,
    coupling: str = COUPLING_COMMON_RANDOM,
    random_seed: int | None = None,
) -> str:
    """渲染垫片源码。"""

    return _SHIM_TEMPLATE.format(
        transcript_path=str(transcript_path),
        recorded_entry=recorded_entry,
        player_frame=PLAYER_FRAME,
        coupling=normalize_coupling(coupling),
        random_seed=None if random_seed is None else int(random_seed),
        bootstrap=BOOTSTRAP,
        env_allowlist=ENV_ALLOWLIST,
        env_prefix=ENV_PREFIX,
        env_forbidden=ENV_FORBIDDEN,
    )


def build_recording_snapshot(
    source: Path,
    destination: Path,
    transcript_path: Path,
    *,
    coupling: str = COUPLING_COMMON_RANDOM,
    random_seed: int | None = None,
) -> Path:
    """把候选快照复制成"会录音"的版本。

    原快照**不被改动**（它还要用于重放与后续对局）；克隆里 ``main.py`` 换成垫片，
    真入口改名为 :data:`RECORDED_ENTRY`。
    """

    source = Path(source)
    destination = Path(destination)
    if not (source / "main.py").is_file():
        raise FileNotFoundError(f"candidate snapshot has no main.py: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=True)
    (destination / "main.py").replace(destination / RECORDED_ENTRY)
    Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
    (destination / "main.py").write_text(
        render_shim(
            transcript_path=Path(transcript_path),
            coupling=coupling,
            random_seed=random_seed,
        ),
        encoding="utf-8",
    )
    return destination
