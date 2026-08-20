"""读取录制垫片产出的线协议流水，还原成可比较的决策序列。"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from agentbench_hl.adapters.transcript.coupling import COUPLING_NONE
from agentbench_hl.domain.wire_policy import WireDecision, WireEpisode


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class WireTranscript:
    """一次录制的全部内容。"""

    #: 判题器→选手的**全部**入站字节（按序拼接）。重放时原样喂给另一个版本。
    inbound: bytes
    #: 选手→判题器的每一帧帧体（顺序即决策顺序）。
    outbound: tuple[bytes, ...]
    #: 每个决策之前新读入的观测字节的 sha256（occupancy 的 state id）。
    observation_ids: tuple[str, ...]
    #: 录制时子进程实际看到的环境变量（重放据此复现）。
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    returncode: int | None = None
    complete: bool = False
    error: str | None = None
    #: 上一次决策之后**没有新观测**就又出了一次动作的决策数。
    #:
    #: 回合制协议下判题器会等选手回话再发下一份观测，所以正常情况下这个数是 0。
    #: 不为 0 说明入站字节在管道里被合并了（一次 read 拿到了多回合观测），此时
    #: 这些决策的观测 id 只能由"前一个观测 + 序号"派生——KL 不受影响（它只看动作），
    #: 但 occupancy 的粒度变粗了，必须报出来让读者知道。
    coalesced_decisions: int = 0
    #: 录制时使用的随机流耦合口径（见 ``adapters/transcript/coupling.py``）。
    #: 重放必须沿用同一口径与同一种子，否则"同一策略"两次可能给出不同动作。
    coupling: str = COUPLING_NONE
    random_seed: int | None = None
    #: 录制持续了多久（秒，取最后一条记录的相对时刻）。用于成本核算与"是不是被录制
    #: 拖慢了"的排查——垫片曾因每条记录都 flush 把一局拖成 38 倍。
    duration_s: float | None = None

    @property
    def decision_count(self) -> int:
        return len(self.outbound)

    def episode(self, *, match_id: str, role: str) -> WireEpisode:
        return WireEpisode(
            match_id=match_id,
            role=role,
            decisions=tuple(
                WireDecision(
                    index=index,
                    observation_id=self.observation_ids[index],
                    action_token=_digest(body),
                )
                for index, body in enumerate(self.outbound)
            ),
            truncated=not self.complete,
        )


def read_transcript(path: str | Path) -> WireTranscript:
    """解析录制文件。

    对"录到一半被杀"是宽容的（``complete=False``），但对**顺序错乱或格式非法**是严格的
    （``error`` 非空）——那意味着还原出来的决策序列不可信，上层必须记 null。
    """

    file = Path(path)
    if not file.is_file():
        return WireTranscript(b"", (), (), error=f"transcript missing: {file}")

    inbound = bytearray()
    pending = bytearray()
    outbound: list[bytes] = []
    observation_ids: list[str] = []
    coalesced = 0
    env: dict[str, str] = {}
    cwd: str | None = None
    coupling = COUPLING_NONE
    random_seed: int | None = None
    duration_s: float | None = None
    returncode: int | None = None
    complete = False
    error: str | None = None
    expected_seq = 0

    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            # 最后一行可能因为进程被杀而截断：这是可容忍的不完整，不是格式错误。
            error = error or "truncated transcript record"
            break
        if not isinstance(record, dict):
            error = error or "transcript record is not an object"
            break
        seq = record.get("seq")
        if seq != expected_seq:
            error = error or f"transcript out of order at seq {seq!r}"
            break
        expected_seq += 1
        stamp = record.get("t")
        if isinstance(stamp, (int, float)):
            duration_s = float(stamp)
        direction = record.get("dir")
        if direction == "header":
            raw_env = record.get("env")
            env = (
                {str(key): str(value) for key, value in raw_env.items()}
                if isinstance(raw_env, dict)
                else {}
            )
            cwd = record.get("cwd") if isinstance(record.get("cwd"), str) else None
            if isinstance(record.get("coupling"), str):
                coupling = str(record["coupling"])
            seed_value = record.get("random_seed")
            random_seed = int(seed_value) if isinstance(seed_value, int) else None
        elif direction == "in":
            chunk = base64.b64decode(str(record.get("b64", "")))
            inbound.extend(chunk)
            pending.extend(chunk)
        elif direction == "out":
            outbound.append(base64.b64decode(str(record.get("b64", ""))))
            if pending:
                observation_ids.append(_digest(bytes(pending)))
                pending.clear()
            else:
                # 这一步之前没有新观测：入站块被合并了。派生一个稳定但明确"二手"的
                # state id，并计数——绝不给它一个空串哈希，那会让不同回合看起来同态。
                coalesced += 1
                previous = observation_ids[-1] if observation_ids else "genesis"
                observation_ids.append(
                    _digest(f"{previous}:coalesced{len(observation_ids)}".encode())
                )
        elif direction == "framing_error":
            error = error or f"player frame stream corrupted: {record.get('detail')}"
            break
        elif direction == "footer":
            value = record.get("returncode")
            returncode = int(value) if isinstance(value, int) else None
            complete = True
        # "trailing" / "downstream_closed" 只是诊断信息，不影响已还原的决策序列。

    return WireTranscript(
        inbound=bytes(inbound),
        outbound=tuple(outbound),
        observation_ids=tuple(observation_ids),
        env=env,
        cwd=cwd,
        returncode=returncode,
        complete=complete,
        error=error,
        coalesced_decisions=coalesced,
        coupling=coupling,
        random_seed=random_seed,
        duration_s=duration_s,
    )
