#!/usr/bin/env python3
"""决策级行为信息增益的**真跑验收**。

为什么必须真跑
--------------

``behavioral_ig`` 这条曲线的失效方式是**静默的**：录制垫片一坏、或者某个游戏的选手
不是"观测流 → 动作"的确定性函数，指标就会静静地变成 null（甚至更糟：变成一个恒为 0
的假水平线）。单测里的假游戏能守住算法，但守不住"真选手 + 真判题器 + 真沙箱"这条链路。

所以这里对每个（游戏, 角色轨）真跑，并检查两件互补的事：

* **零点校准**：拿同一份候选当基线和候选 ⇒ 行为信息增益必须**恰好等于 0**，
  且比较到的决策数 > 0。这一条同时证明了：录制到了决策、确定性自校验通过、
  重放逐帧复现。任何一环坏掉，这里不会是 0，而是 null。
* **灵敏度**：给候选套一层**线协议层面的扰动**（把第 1 个决策之后每一帧回复的
  最后一个字节 +1）⇒ 行为信息增益必须 > 0、动作分歧率 > 0、首次分歧点 = 1。
  这一条证明指标真的对"动作变了"有反应，而不是恒 0。

扰动是**验收用的合成扰动**，不代表任何策略学：它只用来证明测量链路有响应。

一种结果是 ``NULL`` 而不是 FAIL
-------------------------------

若参考策略本身就不是"观测流 → 动作"的确定性函数（例如直接取 ``os.urandom``，
公共随机流耦合也救不回来），那么 null **就是正确答案**。把这种情况判成失败，
只会逼着人去把确定性守卫拆掉——那才是真正的灾难。这里如实标成 ``NULL`` 并打印原因。

用法
----
    python scripts/verify_behavioral_ig.py --game antwar2
    python scripts/verify_behavioral_ig.py --parallel 4 --report out.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentbench_hl.adapters.contract.arena import ContractArena  # noqa: E402
from agentbench_hl.adapters.contract.factory import game_roles  # noqa: E402
from agentbench_hl.adapters.isolation import select_candidate_isolation  # noqa: E402
from agentbench_hl.adapters.transcript import transcript_root  # noqa: E402
from agentbench_hl.application.behavioral_ig import (  # noqa: E402
    BehavioralIgCase,
    measure_behavioral_ig,
)
from agentbench_hl.application.decision_space import (  # noqa: E402
    load_information_gain_spec,
)
from agentbench_hl.ports.arena import MatchCase  # noqa: E402
from agentbench_hl.ports.isolation import IsolationRequest  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from verify_candidate_support import (  # noqa: E402
    AGENTBENCH_ROOT,
    Target,
    discover_targets,
    pick_opponents,
    stage_candidate,
)

BASELINE_ID = "behavioral-ig-baseline"
CANDIDATE_ID = "behavioral-ig-candidate"
INNER_ENTRY = "_abhl_inner_main.py"

#: 线协议层面的合成扰动：转发一切，但把第 ``FROM`` 个决策起的回复帧体最后一字节 +1。
#: 帧长不变，所以判题器仍能正常解析（动作可能变非法，那只影响它自己那局的结果，
#: 不影响"在冻结观测流上重放"这件事）。
PERTURB = '''#!/usr/bin/env python3
# 验收用的线协议扰动垫片（scripts/verify_behavioral_ig.py 生成）。
from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
INNER = {inner!r}
FROM = {start!r}
CHUNK = 65536


def _read_some(stream):
    return stream.read1(CHUNK) if hasattr(stream, "read1") else stream.read(CHUNK)


def _pump_in(destination):
    source = getattr(sys.stdin, "buffer", sys.stdin)
    while True:
        chunk = _read_some(source)
        if not chunk:
            break
        try:
            destination.write(chunk)
            destination.flush()
        except (BrokenPipeError, ValueError, OSError):
            break
    try:
        destination.close()
    except Exception:
        pass


def _pump_out(source):
    target = getattr(sys.stdout, "buffer", sys.stdout)
    pending = b""
    index = 0
    while True:
        chunk = _read_some(source)
        if not chunk:
            break
        pending += chunk
        while len(pending) >= 4:
            (length,) = struct.unpack(">i", pending[:4])
            if length < 0 or len(pending) < 4 + length:
                break
            body = pending[4 : 4 + length]
            pending = pending[4 + length :]
            if index >= FROM and body:
                body = body[:-1] + bytes([(body[-1] + 1) % 256])
            index += 1
            try:
                target.write(struct.pack(">i", len(body)) + body)
                target.flush()
            except (BrokenPipeError, ValueError, OSError):
                return
    if pending:
        try:
            target.write(pending)
            target.flush()
        except Exception:
            pass


def main():
    process = subprocess.Popen(
        [sys.executable, "-u", os.path.join(HERE, INNER)],
        cwd=os.getcwd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    writer = threading.Thread(target=_pump_in, args=(process.stdin,), daemon=True)
    reader = threading.Thread(target=_pump_out, args=(process.stdout,), daemon=True)
    writer.start()
    reader.start()
    code = process.wait()
    reader.join(timeout=10.0)
    return code


if __name__ == "__main__":
    sys.exit(main())
'''


def stage_perturbed(baseline: Path, destination: Path, *, start: int) -> Path:
    """复制基线候选，并在它外面套一层线协议扰动垫片。"""

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(baseline, destination, symlinks=True)
    (destination / "main.py").replace(destination / INNER_ENTRY)
    (destination / "main.py").write_text(
        PERTURB.format(inner=INNER_ENTRY, start=start), encoding="utf-8"
    )
    return destination


MEASUREMENT_LABELS = ("zero", "perturbed")


def _arena(
    target: Target, roles: tuple[str, ...], opponent, work_root: Path, *, timeout_s: float,
    cpus_per_match: int,
) -> ContractArena:
    def factory(request: IsolationRequest):
        return select_candidate_isolation(
            request, backend="auto", profile_path=work_root / "verify.sb"
        )

    return ContractArena(
        game=target.game,
        agentbench_root=AGENTBENCH_ROOT,
        roles=roles,
        artifact_root=work_root / "matches",
        build_root=work_root / "build",
        isolation_factory=factory,
        # 录制垫片要往这里写线协议流水（沙箱内唯一新增的可写点）。
        # 每次测量各有一个 transcripts 目录，必须**逐个**声明：少声明一个，
        # 垫片就打不开文件，指标会静默变 null。
        extra_writable_roots=tuple(
            transcript_root(work_root / "behavioral-ig" / label)
            for label in MEASUREMENT_LABELS
        ),
        opponents={opponent.player_id: opponent},
        timeout_s=timeout_s,
        cpus_per_match=cpus_per_match,
    )


def _measure(
    *,
    target: Target,
    spec: object,
    arena: ContractArena,
    work_root: Path,
    baseline: Path,
    candidate: Path,
    case: BehavioralIgCase,
    label: str,
    timeout_s: float,
):
    def run_match(player_id: str, root: Path, item: BehavioralIgCase) -> dict[str, object]:
        result = arena.run_case(
            MatchCase(player_id, item.opponent_id, item.role, item.seed), root
        )
        return {"status": result.status, "rounds": result.rounds, "error": result.error}

    return measure_behavioral_ig(
        spec=spec,
        epsilon=0.01,
        work_root=work_root / "behavioral-ig" / label,
        baseline_id=BASELINE_ID,
        baseline_root=baseline,
        candidate_id=CANDIDATE_ID,
        candidate_root=candidate,
        cases=[case],
        run_match=run_match,
        replay_timeout_s=timeout_s,
        max_cases=1,
        # 验收要能事后翻录制素材（比如核对帧数、看首次分歧在哪一帧）。
        keep_recordings=True,
    )


def verify(
    target: Target, *, work_base: Path, timeout_s: float, cpus_per_match: int
) -> dict[str, object]:
    started = time.time()
    record: dict[str, object] = {
        "game": target.game,
        "track": target.track,
        "verified": False,
    }
    work_root = work_base / target.label.replace("[", "-").replace("]", "")
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        spec, note = load_information_gain_spec(target.game, agentbench_root=AGENTBENCH_ROOT)
        record["support_note"] = note
        if spec is None:
            # 没声明契约的游戏（deepclue）不是失败，是"有意不测"。
            record["verified"] = True
            record["skipped"] = True
            record["elapsed_s"] = round(time.time() - started, 1)
            return record
        roles = game_roles(AGENTBENCH_ROOT, target.game)
        baseline = stage_candidate(target, work_root)
        candidate = stage_perturbed(baseline, work_root / "perturbed", start=1)
        opponents, seat = pick_opponents(target, roles, limit=1)
        opponent = opponents[0]
        record["support_mode"] = spec.describe(seat)["support_mode"]
        record["support_cardinality"] = spec.describe(seat)["support_cardinality"]
    except Exception as error:  # noqa: BLE001 - 验收要如实记录任何失败
        record["diagnostic"] = f"{type(error).__name__}: {error}"
        record["elapsed_s"] = round(time.time() - started, 1)
        return record

    record["seat"] = seat
    record["opponent_id"] = opponent.player_id
    case = BehavioralIgCase(opponent_id=opponent.player_id, role=seat, seed=7)
    arena = _arena(
        target, roles, opponent, work_root, timeout_s=timeout_s, cpus_per_match=cpus_per_match
    )

    try:
        zero = _measure(
            target=target,
            spec=spec,
            arena=arena,
            work_root=work_root,
            baseline=baseline,
            candidate=baseline,
            case=case,
            label="zero",
            timeout_s=timeout_s,
        )
        sensitive = _measure(
            target=target,
            spec=spec,
            arena=arena,
            work_root=work_root,
            baseline=baseline,
            candidate=candidate,
            case=case,
            label="perturbed",
            timeout_s=timeout_s,
        )
    except Exception as error:  # noqa: BLE001
        record["diagnostic"] = f"{type(error).__name__}: {error}"
        record["elapsed_s"] = round(time.time() - started, 1)
        return record

    record["zero"] = {
        "behavioral_ig": zero.value,
        "decisions": zero.compared_decisions,
        "reason": zero.reason,
    }
    record["perturbed"] = {
        "behavioral_ig": sensitive.value,
        "disagreement": sensitive.disagreement_rate,
        "decisions": sensitive.compared_decisions,
        "first_divergence": sensitive.cases[0].first_divergence if sensitive.cases else None,
        "reason": sensitive.reason,
    }
    checks = {
        "zero_point": zero.value == 0.0 and zero.compared_decisions > 0,
        "sensitive": (sensitive.value or 0.0) > 0.0,
        "disagreement": (sensitive.disagreement_rate or 0.0) > 0.0,
    }
    record["checks"] = checks
    record["verified"] = all(checks.values())
    if not record["verified"] and _is_honest_null(zero):
        # 参考策略自己就不是"观测流 → 动作"的确定性函数（比如直接取系统熵源）。
        # 这种 null 是**正确答案**，不是链路故障：判成 FAIL 会逼着人把守卫拆掉。
        record["verified"] = True
        record["expected_null"] = True
    record["elapsed_s"] = round(time.time() - started, 1)
    return record


def _is_honest_null(measurement) -> bool:
    """这个 null 是不是"本该如此"（而不是探针坏了）。"""

    reason = measurement.reason or ""
    return measurement.value is None and (
        "not a deterministic function" in reason
        or "declares no information_gain contract" in reason
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", help="只验收指定游戏（可重复）")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--cpus-per-match", type=int, default=4)
    parser.add_argument(
        "--work-root", default=str(REPO_ROOT / "artifacts" / "behavioral-ig-verify")
    )
    parser.add_argument(
        "--report", default=str(REPO_ROOT / "artifacts" / "behavioral_ig_verification.json")
    )
    args = parser.parse_args(argv)

    if not (AGENTBENCH_ROOT / "games").is_dir():
        print(f"AGENTBENCH_ROOT 无效：{AGENTBENCH_ROOT}")
        return 1
    targets = discover_targets(args.game)
    if not targets:
        print("没有找到任何 candidate_support 目录，先跑 scripts/gen_candidate_support.py")
        return 1

    work_base = Path(args.work_root).resolve()
    work_base.mkdir(parents=True, exist_ok=True)
    print(f"验收 {len(targets)} 个目标：" + ", ".join(item.label for item in targets))

    def run(target: Target) -> dict[str, object]:
        record = verify(
            target,
            work_base=work_base,
            timeout_s=args.timeout_s,
            cpus_per_match=args.cpus_per_match,
        )
        if record.get("skipped"):
            print(f"  [SKIP] {target.label}: {record.get('support_note')}")
            return record
        mark = "PASS" if record.get("verified") else "FAIL"
        if record.get("expected_null"):
            mark = "NULL"
        zero = record.get("zero") or {}
        perturbed = record.get("perturbed") or {}
        print(
            f"  [{mark}] {target.label}: zero_ig={zero.get('behavioral_ig')} "
            f"decisions={zero.get('decisions')} | perturbed_ig={perturbed.get('behavioral_ig')} "
            f"disagreement={perturbed.get('disagreement')} "
            f"|A|={record.get('support_cardinality')} "
            f"({record.get('support_mode')}) {record.get('elapsed_s')}s"
        )
        for key in ("diagnostic",):
            if record.get(key):
                print(f"         {key}：{record[key]}")
        if record.get("expected_null"):
            print(f"         诚实的 null：{zero.get('reason')}")
        elif not record.get("verified"):
            print(f"         zero.reason：{zero.get('reason')}")
            print(f"         perturbed.reason：{perturbed.get('reason')}")
        return record

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            records = list(pool.map(run, targets))
    else:
        records = [run(target) for target in targets]

    passed = sum(1 for record in records if record.get("verified"))
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at": time.time(),
                "agentbench_root": str(AGENTBENCH_ROOT),
                "passed": passed,
                "total": len(records),
                "rows": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n{passed}/{len(records)} 通过；报告：{report_path}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    sys.exit(main())
