"""通用线协议探针的端到端集成测试。

用一个**假游戏**（判题器只发几条 JSON 观测、选手只回一个整数动作）把整条链路真跑一遍：
录制垫片 → 流水解析 → 冻结重放 → ε 正则 KL。假游戏不引入任何真游戏依赖，但走的是
和真游戏完全相同的帧格式与代码路径，所以能守住这条链路的行为。

同时钉住三条诚实性纪律：
* 参考策略若不是"观测流 → 动作"的确定性函数（比如用了随机），整轮记 null + 原因；
* 候选提前崩溃 ⇒ 只比较已产出的决策，并把截断量报出来；
* 游戏没声明 information_gain 契约 ⇒ 记 null，不猜一个 |A| 出来。
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentbench_hl.adapters.transcript import (
    RECORDED_ENTRY,
    build_recording_snapshot,
    read_transcript,
    replay_actions,
)
from agentbench_hl.application.behavioral_ig import (
    BehavioralIgCase,
    measure_behavioral_ig,
)

ROUNDS = 6

#: 假选手：读 [len:4][json]，回 [len:4][json]。``SHIFT`` 决定它的策略。
PLAYER = """
import json
import struct
import sys

SHIFT = {shift}
CRASH_AFTER = {crash_after}
RANDOMIZE = {randomize}
USE_SYSTEM_ENTROPY = {use_system_entropy}


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    served = 0
    while True:
        head = stdin.read(4)
        if len(head) < 4:
            return 0
        (length,) = struct.unpack(">i", head)
        body = stdin.read(length)
        if len(body) < length:
            return 0
        observation = json.loads(body.decode("utf-8"))
        if CRASH_AFTER is not None and served >= CRASH_AFTER:
            return 3
        move = (observation["round"] + SHIFT) % 4
        if RANDOMIZE:
            import random

            move = random.randrange(4)
        if USE_SYSTEM_ENTROPY:
            import os

            move = os.urandom(1)[0] % 4
        payload = json.dumps({{"move": move}}, sort_keys=True).encode("utf-8")
        stdout.write(struct.pack(">i", len(payload)) + payload)
        stdout.flush()
        served += 1


if __name__ == "__main__":
    sys.exit(main())
"""


def _snapshot(
    root: Path,
    *,
    shift: int,
    crash_after: int | None = None,
    randomize: bool = False,
    use_system_entropy: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(
        PLAYER.format(
            shift=shift,
            crash_after=crash_after,
            randomize=randomize,
            use_system_entropy=use_system_entropy,
        ),
        encoding="utf-8",
    )
    return root


def _judge_stream(rounds: int = ROUNDS) -> bytes:
    stream = b""
    for index in range(rounds):
        body = json.dumps({"round": index}, sort_keys=True).encode("utf-8")
        stream += struct.pack(">i", len(body)) + body
    return stream


def _play(root: Path) -> str:
    """扮演判题器：回合制地发观测、等动作（真游戏就是这个节奏）。

    刻意**不**一次性灌完整条流：回合制协议下判题器必须等选手回话才发下一份观测，
    这个节奏决定了入站块与决策一一对齐，也是观测 id 有意义的前提。
    """

    process = subprocess.Popen(  # noqa: S603 - 参数由测试构造
        [sys.executable, "-u", "main.py"],
        cwd=str(root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        for index in range(ROUNDS):
            body = json.dumps({"round": index}, sort_keys=True).encode("utf-8")
            process.stdin.write(struct.pack(">i", len(body)) + body)
            process.stdin.flush()
            head = process.stdout.read(4)
            if len(head) < 4:
                break
            (length,) = struct.unpack(">i", head)
            if len(process.stdout.read(length)) < length:
                break
        process.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    code = process.wait(timeout=120)
    return "complete" if code == 0 else f"crashed:{code}"


# --------------------------------------------------------------- 垫片与解析


def test_recording_snapshot_is_transparent(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "src", shift=0)
    plain = subprocess.run(
        [sys.executable, "-u", "main.py"],
        cwd=str(source),
        input=_judge_stream(),
        capture_output=True,
        check=True,
    )
    clone = build_recording_snapshot(
        source, tmp_path / "clone", tmp_path / "tx" / "reference.jsonl"
    )
    recorded = subprocess.run(
        [sys.executable, "-u", "main.py"],
        cwd=str(clone),
        input=_judge_stream(),
        capture_output=True,
        check=True,
    )

    # 垫片只做字节转发：判题器看到的输出必须逐字节相同。
    assert recorded.stdout == plain.stdout
    assert (clone / RECORDED_ENTRY).is_file()
    assert (source / "main.py").read_text(encoding="utf-8").startswith("\nimport json")


def test_transcript_recovers_every_decision(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "src", shift=0)
    transcript_path = tmp_path / "tx" / "reference.jsonl"
    clone = build_recording_snapshot(source, tmp_path / "clone", transcript_path)

    assert _play(clone) == "complete"
    transcript = read_transcript(transcript_path)

    assert transcript.error is None
    assert transcript.complete is True
    assert transcript.decision_count == ROUNDS
    assert transcript.inbound == _judge_stream()
    # 回合制节奏下每个决策都有自己的新观测：occupancy 的粒度就是"一决策一状态"。
    assert transcript.coalesced_decisions == 0
    assert len(set(transcript.observation_ids)) == ROUNDS


def test_coalesced_input_is_reported_not_faked(tmp_path: Path) -> None:
    """入站块被合并时，观测 id 不能退化成"同一个空哈希"。"""

    source = _snapshot(tmp_path / "src", shift=0)
    transcript_path = tmp_path / "tx" / "reference.jsonl"
    clone = build_recording_snapshot(source, tmp_path / "clone", transcript_path)
    # 一次性灌完 = 人为制造合并（真判题器不会这么干，但管道有权这么合并）。
    subprocess.run(
        [sys.executable, "-u", "main.py"],
        cwd=str(clone),
        input=_judge_stream(),
        capture_output=True,
        check=True,
    )
    transcript = read_transcript(transcript_path)

    assert transcript.decision_count == ROUNDS
    assert transcript.coalesced_decisions == ROUNDS - 1
    # 派生 id 仍然互不相同：不会把不同回合伪装成同一个状态。
    assert len(set(transcript.observation_ids)) == ROUNDS


def test_replay_reproduces_the_reference_actions(tmp_path: Path) -> None:
    source = _snapshot(tmp_path / "src", shift=0)
    transcript_path = tmp_path / "tx" / "reference.jsonl"
    clone = build_recording_snapshot(source, tmp_path / "clone", transcript_path)
    _play(clone)
    transcript = read_transcript(transcript_path)

    outcome = replay_actions(source, transcript, timeout_s=120)

    assert outcome.error is None
    assert outcome.action_tokens == transcript.episode(match_id="m", role="P0").action_tokens


def test_missing_transcript_is_reported_not_silently_empty(tmp_path: Path) -> None:
    transcript = read_transcript(tmp_path / "nope.jsonl")

    assert transcript.error is not None
    assert transcript.decision_count == 0


def test_unwritable_transcript_does_not_break_the_match(tmp_path: Path) -> None:
    """录制文件写不下去时，垫片必须退化成纯转发，而不是把选手带崩。

    否则一次**测量**故障会伪装成候选的一场败局——那比测不出来严重得多。
    模拟手法（与 root/非 root 无关）：把目标文件名占成一个目录，``open(..., "w")`` 必失败。
    """

    source = _snapshot(tmp_path / "src", shift=0)
    blocked = tmp_path / "tx" / "reference.jsonl"
    blocked.mkdir(parents=True)
    clone = build_recording_snapshot(source, tmp_path / "clone", blocked)

    assert _play(clone) == "complete"
    assert read_transcript(blocked).error is not None


def test_unwritable_transcript_yields_null_with_an_actionable_reason(tmp_path: Path) -> None:
    baseline = _snapshot(tmp_path / "baseline", shift=0)
    case = BehavioralIgCase(opponent_id="rank10__bot", role="P0", seed=7)
    work = tmp_path / "work"
    # 占掉录制文件名 ⇒ 等价于"忘了把 transcripts 声明成沙箱可写目录"。
    (work / "transcripts" / case.tag / "reference.jsonl").mkdir(parents=True)

    measurement = measure_behavioral_ig(
        spec=_spec(),
        epsilon=0.02,
        work_root=work,
        baseline_id="v0",
        baseline_root=baseline,
        candidate_id="v1",
        candidate_root=_snapshot(tmp_path / "cand", shift=1),
        cases=[case],
        run_match=_run_match_factory(),
        replay_timeout_s=120.0,
        max_cases=1,
    )

    assert measurement.value is None
    assert "extra_writable_roots" in measurement.reason


# ------------------------------------------------------------ 完整测量流程


@dataclass(frozen=True)
class _Support:
    cardinality: int
    cardinality_by_role: dict[str, int]

    def size_for(self, role: str | None = None) -> int:
        return self.cardinality_by_role.get(role or "", self.cardinality)


@dataclass(frozen=True)
class _Spec:
    """替身：形状与 agentbench.core.decision_space.InformationGainSpec 一致。"""

    support: _Support

    def describe(self, role: str | None = None) -> dict[str, object]:
        return {
            "support_mode": "enumerated",
            "support_cardinality": self.support.size_for(role),
            "support_provenance": "fake game action table",
        }


def _spec(cardinality: int = 4) -> _Spec:
    return _Spec(support=_Support(cardinality=cardinality, cardinality_by_role={}))


def _run_match_factory() -> object:
    def run_match(_player_id: str, root: Path, _case: BehavioralIgCase) -> dict[str, object]:
        return {"status": _play(root)}

    return run_match


def _measure(tmp_path: Path, *, candidate: Path, spec: object = None, cardinality: int = 4):
    baseline = _snapshot(tmp_path / "baseline", shift=0)
    return measure_behavioral_ig(
        spec=_spec(cardinality) if spec is None else spec,
        epsilon=0.02,
        work_root=tmp_path / "work",
        baseline_id="v0",
        baseline_root=baseline,
        candidate_id="v1",
        candidate_root=candidate,
        cases=[BehavioralIgCase(opponent_id="rank10__bot", role="P0", seed=7)],
        run_match=_run_match_factory(),
        replay_timeout_s=120.0,
        max_cases=1,
    )


def test_identical_policy_measures_zero_information_gain(tmp_path: Path) -> None:
    measurement = _measure(tmp_path, candidate=_snapshot(tmp_path / "cand", shift=0))

    assert measurement.value == 0.0
    assert measurement.disagreement_rate == 0.0
    assert measurement.compared_decisions == ROUNDS
    assert measurement.support["support_cardinality"] == 4
    assert "transcript replay" in measurement.reason


def test_changed_policy_measures_positive_information_gain(tmp_path: Path) -> None:
    measurement = _measure(tmp_path, candidate=_snapshot(tmp_path / "cand", shift=1))

    assert measurement.value is not None
    assert measurement.value > 0.0
    assert measurement.disagreement_rate == pytest.approx(1.0)
    assert measurement.cases[0].first_divergence == 0
    # 两个版本的观测流相同（判题器是脚本化的），所以 occupancy 位移应当是 0 而非 null。
    assert measurement.occupancy_shift == pytest.approx(0.0)


def test_crashing_candidate_is_truncated_and_reported(tmp_path: Path) -> None:
    candidate = _snapshot(tmp_path / "cand", shift=1, crash_after=2)

    measurement = _measure(tmp_path, candidate=candidate)

    assert measurement.compared_decisions == 2
    assert measurement.cases[0].candidate_decisions == 2
    assert measurement.cases[0].reference_decisions == ROUNDS
    assert "stopped after 2/6" in (measurement.cases[0].reason or "")


def test_nondeterministic_reference_yields_null_with_a_reason(tmp_path: Path) -> None:
    """关掉耦合时，用了 ``random`` 的参考策略必须记 null，而不是凑一个数。"""

    baseline = _snapshot(tmp_path / "baseline", shift=0, randomize=True)
    measurement = measure_behavioral_ig(
        spec=_spec(),
        epsilon=0.02,
        work_root=tmp_path / "work",
        baseline_id="v0",
        baseline_root=baseline,
        candidate_id="v1",
        candidate_root=_snapshot(tmp_path / "cand", shift=1),
        cases=[BehavioralIgCase(opponent_id="rank10__bot", role="P0", seed=7)],
        run_match=_run_match_factory(),
        replay_timeout_s=120.0,
        max_cases=1,
        coupling="none",
    )

    assert measurement.value is None
    assert "deterministic" in measurement.reason
    assert measurement.coupling == "none"
    # 原因里要给出下一步该怎么做，而不是只说"测不了"。
    assert "common_random_seed" in measurement.reason


def test_random_using_reference_is_measurable_under_common_random_stream(
    tmp_path: Path,
) -> None:
    """默认耦合下，调 ``random`` 的策略也能测——真实选手里这类占相当比例。"""

    measurement = measure_behavioral_ig(
        spec=_spec(),
        epsilon=0.02,
        work_root=tmp_path / "work",
        baseline_id="v0",
        baseline_root=_snapshot(tmp_path / "baseline", shift=0, randomize=True),
        candidate_id="v1",
        candidate_root=_snapshot(tmp_path / "cand", shift=1),
        cases=[BehavioralIgCase(opponent_id="rank10__bot", role="P0", seed=7)],
        run_match=_run_match_factory(),
        replay_timeout_s=120.0,
        max_cases=1,
    )

    assert measurement.value is not None
    assert measurement.coupling == "common_random_seed"
    # 口径必须随数一起出现，否则读者无法知道这个数是在公共随机流下比的。
    assert "coupling=common_random_seed" in measurement.reason
    assert measurement.payload()["behavioral_ig_coupling"] == "common_random_seed"


def test_coupling_does_not_whitewash_other_nondeterminism(tmp_path: Path) -> None:
    """耦合只播种 ``random``/``numpy.random``，**不是**"什么都能测"的漂白剂。

    直接取系统熵源（``os.urandom``，不受 ``random.seed`` 影响）的策略仍然复现不了自己，
    必须照样记 null——否则我们会把一堆噪声当成"策略变化"画到曲线上。
    """

    measurement = measure_behavioral_ig(
        spec=_spec(),
        epsilon=0.02,
        work_root=tmp_path / "work",
        baseline_id="v0",
        baseline_root=_snapshot(tmp_path / "baseline", shift=0, use_system_entropy=True),
        candidate_id="v1",
        candidate_root=_snapshot(tmp_path / "cand", shift=1),
        cases=[BehavioralIgCase(opponent_id="rank10__bot", role="P0", seed=7)],
        run_match=_run_match_factory(),
        replay_timeout_s=120.0,
        max_cases=1,
    )

    assert measurement.value is None
    assert "deterministic" in measurement.reason


def test_game_without_a_contract_is_null_not_guessed(tmp_path: Path) -> None:
    measurement = measure_behavioral_ig(
        spec=None,
        epsilon=0.02,
        work_root=tmp_path / "work",
        baseline_id="v0",
        baseline_root=tmp_path,
        candidate_id="v1",
        candidate_root=tmp_path,
        cases=[BehavioralIgCase(opponent_id="x", role="P0", seed=1)],
        run_match=_run_match_factory(),
    )

    assert measurement.value is None
    assert "declares no information_gain contract" in measurement.reason


def test_disabling_the_probe_is_distinguishable_from_failing(tmp_path: Path) -> None:
    measurement = measure_behavioral_ig(
        spec=_spec(),
        epsilon=0.02,
        work_root=tmp_path / "work",
        baseline_id="v0",
        baseline_root=tmp_path,
        candidate_id="v1",
        candidate_root=tmp_path,
        cases=[BehavioralIgCase(opponent_id="x", role="P0", seed=1)],
        run_match=_run_match_factory(),
        max_cases=0,
    )

    assert measurement.value is None
    assert "disabled" in measurement.reason


def test_declared_support_drives_the_scale(tmp_path: Path) -> None:
    small = _measure(tmp_path / "a", candidate=_snapshot(tmp_path / "a" / "cand", shift=1))
    large = _measure(
        tmp_path / "b",
        candidate=_snapshot(tmp_path / "b" / "cand", shift=1),
        cardinality=125,
    )

    assert small.value is not None
    assert large.value is not None
    assert large.value > small.value
    assert large.support["support_cardinality"] == 125


def test_trace_document_carries_the_per_decision_kl(tmp_path: Path) -> None:
    measurement = _measure(tmp_path, candidate=_snapshot(tmp_path / "cand", shift=1))
    document = measurement.trace_document()

    assert document["compared_decisions"] == ROUNDS
    assert len(document["cases"][0]["kl_trace"]) == ROUNDS
    assert all(item > 0 for item in document["cases"][0]["kl_trace"])


def test_recordings_are_reclaimed_by_default(tmp_path: Path) -> None:
    """不限轮数的 run 每轮产两份完整快照 + 两条整局字节流，必须测完即删。"""

    measurement = _measure(tmp_path, candidate=_snapshot(tmp_path / "cand", shift=1))
    work = tmp_path / "work"

    assert measurement.value is not None  # 先确认真的测出来了
    assert not list((work / "snapshots").glob("*/*")), "录制克隆没被回收"
    assert not list((work / "transcripts").glob("*/*.jsonl")), "字节流没被回收"


def test_recordings_can_be_kept_for_investigation(tmp_path: Path) -> None:
    measurement = measure_behavioral_ig(
        spec=_spec(),
        epsilon=0.02,
        work_root=tmp_path / "work",
        baseline_id="v0",
        baseline_root=_snapshot(tmp_path / "baseline", shift=0),
        candidate_id="v1",
        candidate_root=_snapshot(tmp_path / "cand", shift=1),
        cases=[BehavioralIgCase(opponent_id="rank10__bot", role="P0", seed=7)],
        run_match=_run_match_factory(),
        replay_timeout_s=120.0,
        max_cases=1,
        keep_recordings=True,
    )

    assert measurement.value is not None
    assert list((tmp_path / "work" / "transcripts").glob("*/reference.jsonl"))
