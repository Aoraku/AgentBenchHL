"""通用线协议探针 —— 让 8 个游戏都能算出决策级行为信息增益。

模块分工
--------

* :mod:`.framing` —— Saiblo 选手帧 ``[len:4 BE][body]`` 的解析（8 个游戏统一）；
* :mod:`.coupling` —— 公共随机流耦合（让调 ``random`` 的选手也可测，口径显式上报）；
* :mod:`.shim` —— 生成"透明中继 + 记账"的 ``main.py``，用于录下真实对局的字节流；
* :mod:`.reader` —— 把录制文件还原成决策序列（动作 token + 观测 id）；
* :mod:`.replay` —— 把参考观测流喂给另一个版本的进程，取它在同一批上下文上的动作。

口径（|A|、动作 token、occupancy state id）来自 A 仓
``games/<game>/decision_space.yaml`` 的 ``information_gain:`` 段，本包不含任何
游戏语义。
"""

from __future__ import annotations

from agentbench_hl.adapters.transcript.coupling import (
    COUPLING_COMMON_RANDOM,
    COUPLING_MODES,
    COUPLING_NONE,
    coupled_argv,
    normalize_coupling,
)
from agentbench_hl.adapters.transcript.framing import (
    PLAYER_FRAME,
    encode_frame,
    split_frames,
)
from agentbench_hl.adapters.transcript.reader import WireTranscript, read_transcript
from agentbench_hl.adapters.transcript.replay import ReplayOutcome, replay_actions
from agentbench_hl.adapters.transcript.shim import (
    RECORDED_ENTRY,
    SNAPSHOT_DIRNAME,
    TRANSCRIPT_DIRNAME,
    build_recording_snapshot,
    render_shim,
    snapshot_root,
    transcript_root,
)

__all__ = [
    "COUPLING_COMMON_RANDOM",
    "COUPLING_MODES",
    "COUPLING_NONE",
    "PLAYER_FRAME",
    "RECORDED_ENTRY",
    "SNAPSHOT_DIRNAME",
    "TRANSCRIPT_DIRNAME",
    "ReplayOutcome",
    "WireTranscript",
    "build_recording_snapshot",
    "coupled_argv",
    "encode_frame",
    "normalize_coupling",
    "read_transcript",
    "render_shim",
    "replay_actions",
    "snapshot_root",
    "split_frames",
    "transcript_root",
]
