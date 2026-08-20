"""游戏无关的 Arena —— 通过 A 的对战器契约跑一局，并施加候选隔离。

设计要点（对应三仓铁律与你的两条补充要求）：

1. **零重复**：本模块不含任何游戏语义，只调 A 的 `evaluate(game, players, roles, seed)`
   （`docs/evaluator-contract.md`）。新接一个游戏，B 侧**不需要**新 adapter 代码。
2. **强隔离**：每一局都在 :mod:`match_worker` 子进程里跑，外面套 bubblewrap/Seatbelt/
   容器：禁网、只读、遮蔽整个人类选手池（只放开本局对手包）。
3. **并行安全**：每一局有独立工件目录；构建目录按 run 隔离并支持 :meth:`warmup`
   预热，避免 32 路并发同时首编译。
4. **三态忠实**：A 的 ``complete`` / ``game_error`` / ``infra_error`` 分别映射为
   "有效胜负 / 判负（候选自身故障）/ 不计入（基础设施故障）"。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from agentbench_hl.adapters.contract.backend_preseed import preseed_backend
from agentbench_hl.adapters.contract.cpu_leases import lease_cpus, taskset_prefix
from agentbench_hl.adapters.contract.pool import PoolPlayer
from agentbench_hl.ports.arena import MatchCase, MatchResult
from agentbench_hl.ports.isolation import CandidateIsolation, IsolationRequest

WORKER_MODULE = "agentbench_hl.adapters.contract.match_worker"


class ArenaError(RuntimeError):
    """对局无法发起（配置/资源问题，属基础设施故障）。"""


def _other_roles(roles: Sequence[str], candidate_role: str) -> tuple[str, ...]:
    return tuple(role for role in roles if role != candidate_role)


@dataclass
class ContractArena:
    """按 A 的契约执行一局，并把结果翻译成 B 的 ``MatchResult``。"""

    game: str
    agentbench_root: Path
    roles: tuple[str, ...]
    artifact_root: Path
    build_root: Path
    isolation_factory: object  # Callable[[IsolationRequest], CandidateIsolation]
    hidden_read_roots: tuple[Path, ...] = ()
    # 除工件目录与构建目录之外，还需要对**选手进程**可写的路径。
    # 目前唯一用途：行为信息增益的线协议录制文件（录制垫片在沙箱内往这里写）。
    # 刻意只放开 transcripts 目录本身，不放开录制克隆的代码目录——候选在录制局里
    # 仍然是只读的，否则"能不能写盘"这个前提在测量局与正式局之间就不一致了。
    extra_writable_roots: tuple[Path, ...] = ()
    opponents: Mapping[str, PoolPlayer] = field(default_factory=dict)
    timeout_s: float = 420.0
    python_executable: str = sys.executable
    # 每局独占的核数：后端 + 两个选手进程。少于 2 会让重计算选手被邻居拖成"超时"。
    cpus_per_match: int = 3
    _build_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _warm: bool = field(default=False, repr=False)
    _pool_dirs: tuple[Path, ...] | None = field(default=None, repr=False)
    # 后端依赖预置的失败原因（如有）；会随 payload 一起如实上报。
    _preseed_note: str | None = field(default=None, repr=False)

    # -- 公共 API -----------------------------------------------------------

    def warmup(self, candidate_root: Path, *, seed: int = 7) -> MatchResult | None:
        """串行跑一局，完成后端编译与选手准备，之后即可安全并行。"""

        with self._build_lock:
            if self._warm:
                return None
            opponent = next(iter(self.opponents.values()), None)
            if opponent is None:
                raise ArenaError(f"{self.game} has no runnable opponent to warm up with")
            case = MatchCase(
                candidate_id="warmup",
                opponent_id=opponent.player_id,
                role=self.roles[0],
                seed=seed,
            )
            result = self._run(case, Path(candidate_root), artifact_suffix="warmup")
            self._warm = True
            return result

    def run_case(self, case: MatchCase, candidate_root: Path) -> MatchResult:
        return self._run(case, Path(candidate_root))

    # -- 内部实现 -----------------------------------------------------------

    def _case_artifact_root(self, case: MatchCase, suffix: str | None) -> Path:
        parts = [case.candidate_id, case.opponent_id, f"{case.role}-seed-{case.seed}"]
        if suffix:
            parts.append(suffix)
        root = self.artifact_root.joinpath(*parts)
        if root.exists():
            root = root.with_name(f"{root.name}-{uuid.uuid4().hex[:6]}")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _pool_package_dirs(self, pool_root: Path) -> tuple[Path, ...]:
        """池内每个选手包目录（缓存）。

        我们**逐个**遮蔽选手包（而不是遮蔽整个 pool），因为 A 的池审计需要按
        ``rank*__*`` 这类前缀在 pool 下 glob 出目录结构；整体遮蔽会让审计报
        "rankNN maps to 0 pool packages"。逐个遮蔽后：目录仍在、内容为空，
        结构可见而**源码不可读**，两个需求同时满足。
        """

        if self._pool_dirs is None:
            self._pool_dirs = (
                tuple(sorted(item for item in pool_root.iterdir() if item.is_dir()))
                if pool_root.is_dir()
                else ()
            )
        return self._pool_dirs

    def _public_pool_assets(self, pool_root: Path) -> tuple[Path, ...]:
        """选手池里属于**公开资产**的路径（遮蔽池子后要挂回来）。

        A 的约定：官方公开 SDK 随某个提交一起冻结在 ``pool/<player>/SDK``
        （见 A 的 `evaluator/runtime.py`：``pool.glob("rank01__*/SDK")``）。
        它是公开材料而不是任何人的策略，必须对候选可读，否则连后端都起不来。
        """

        if not pool_root.is_dir():
            return ()
        assets: list[Path] = []
        for pattern in ("*/SDK", "*/sdk", "*/sdk_python", "*/SDK_python"):
            assets.extend(item for item in pool_root.glob(pattern) if item.is_dir())
        return tuple(sorted(set(assets)))

    def _isolation_for(
        self,
        opponent: PoolPlayer,
        artifact_root: Path,
        candidate_root: Path | None = None,
        *,
        scratch: Path | None = None,
    ) -> CandidateIsolation:
        players_root = (self.agentbench_root / "games" / self.game / "players").resolve()
        pool_root = players_root / "pool"
        exempt = {opponent.package_root.resolve()}
        if candidate_root is not None:
            # 候选也可能就在池里（例如慢通道用池内选手做基线对照）。
            exempt.add(Path(candidate_root).resolve())
        # 逐包遮蔽人类源码；保留 manifest.tsv（只有 rank/Elo 等公开元数据）。
        denied = tuple(
            item for item in self._pool_package_dirs(pool_root) if item not in exempt
        )
        if not denied and players_root.is_dir():
            denied = (players_root,)
        request = IsolationRequest(
            denied_read_roots=(*denied, *self.hidden_read_roots),
            # 遮蔽后挂回来的：本局对手包（必须能起进程）+ 公开 SDK。
            readable_roots=(
                opponent.package_root,
                *(() if candidate_root is None else (Path(candidate_root).resolve(),)),
                *self._public_pool_assets(pool_root),
            ),
            writable_roots=(artifact_root, self.build_root, *self.extra_writable_roots),
            scratch_dir=scratch,
        )
        factory = self.isolation_factory
        return factory(request)  # type: ignore[operator]

    def _run(
        self,
        case: MatchCase,
        candidate_root: Path,
        *,
        artifact_suffix: str | None = None,
    ) -> MatchResult:
        # 需要联网的后端依赖引导必须在**沙箱外**先做一次（见 backend_preseed）；
        # 否则 lostspace 这类游戏在禁网沙箱里永远卡在 pip 安装上。
        self._preseed_note = preseed_backend(self.agentbench_root, self.game, self.build_root)
        opponent = self.opponents.get(case.opponent_id)
        if opponent is None:
            return MatchResult(
                case=case,
                status="incomplete",
                result=None,
                points=None,
                score_margin=None,
                rounds=None,
                error=f"unknown or unrunnable opponent: {case.opponent_id}",
            )
        if case.role not in self.roles:
            return MatchResult(
                case=case,
                status="incomplete",
                result=None,
                points=None,
                score_margin=None,
                rounds=None,
                error=f"role {case.role} is not one of {self.roles}",
            )

        artifact_root = self._case_artifact_root(case, artifact_suffix)
        others = _other_roles(self.roles, case.role)
        request = {
            "agentbench_root": str(self.agentbench_root),
            "game": self.game,
            "seed": case.seed,
            "roles": [case.role, *others],
            # 规范序也一并给出：部分游戏（lostspace）要求 roles 必须是规范序，
            # worker 在对战器抱怨顺序时按座次对齐重试（见 match_worker）。
            "canonical_roles": list(self.roles),
            # 多人游戏：其余座次由同一对手占据（诚实记录在 payload 里）。
            "players": [
                {"player_id": case.candidate_id, "code_path": str(candidate_root)},
                *(
                    {"player_id": opponent.player_id, "code_path": str(opponent.package_root)}
                    for _ in others
                ),
            ],
            "build_root": str(self.build_root),
            "artifact_root": str(artifact_root),
        }
        request_path = artifact_root / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

        environment = dict(os.environ)
        environment["AGENTBENCH_ROOT"] = str(self.agentbench_root)
        environment.pop("ABHL_API_KEY", None)  # 对局侧永远不需要凭据
        with lease_cpus(self.cpus_per_match) as cpus, tempfile.TemporaryDirectory(
            prefix="abhl-match-"
        ) as scratch:
            # scratch 必须在**构建隔离之前**就存在：沙箱会把 /tmp 换成私有 tmpfs，
            # 宿主 /tmp 下的临时目录在沙箱里根本不存在。之前只设了 TMPDIR 环境变量，
            # 结果 make/g++ 拿到一个不存在的 TMPDIR → 所有需要编译的 C++ 选手
            # 一律 infra_error（antwar 审计 0/27 就是这个原因）。
            # 正确做法：把 scratch 声明给隔离层，由它 bind 进沙箱同名路径。
            isolation = self._isolation_for(
                opponent, artifact_root, candidate_root, scratch=Path(scratch)
            )
            # 独占核 + 隔离：墙钟超时判定不受邻居对局干扰（详见 cpu_leases 模块说明）。
            command = (
                *taskset_prefix(cpus),
                *isolation.wrap(
                    (self.python_executable, "-m", WORKER_MODULE, str(request_path))
                ),
            )
            environment["TMPDIR"] = scratch
            environment["ABHL_MATCH_CPUS"] = ",".join(str(cpu) for cpu in cpus)
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    env=environment,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return self._incomplete(case, f"match exceeded {self.timeout_s:.0f}s wall limit")
            pinned = bool(cpus)
        (artifact_root / "worker.log").write_text(
            (completed.stderr or "") + "\n" + (completed.stdout or ""), encoding="utf-8"
        )
        payload = self._parse(completed.stdout)
        if payload is None:
            tail = (completed.stderr or completed.stdout or "")[-800:]
            return self._incomplete(case, f"match worker produced no result: {tail}")
        return self._to_match_result(case, payload, artifact_root, isolation, pinned=pinned)

    @staticmethod
    def _parse(stdout: str) -> dict[str, object] | None:
        for line in reversed((stdout or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "status" in value:
                return value
        return None

    @staticmethod
    def _incomplete(case: MatchCase, error: str) -> MatchResult:
        return MatchResult(
            case=case,
            status="incomplete",
            result=None,
            points=None,
            score_margin=None,
            rounds=None,
            error=error,
        )

    def _to_match_result(
        self,
        case: MatchCase,
        payload: Mapping[str, object],
        artifact_root: Path,
        isolation: CandidateIsolation,
        *,
        pinned: bool = False,
    ) -> MatchResult:
        status = str(payload.get("status") or "")
        diagnostic = payload.get("diagnostic")
        replay = payload.get("replay_path")
        replay_path = Path(str(replay)) if isinstance(replay, str) and replay else None
        scores = payload.get("scores") if isinstance(payload.get("scores"), Mapping) else {}
        rounds_value = payload.get("rounds")
        rounds = int(rounds_value) if isinstance(rounds_value, (int, float)) else None
        base_payload: dict[str, object] = {
            "evaluator_status": status,
            "scores": dict(scores),  # type: ignore[arg-type]
            "isolation": dict(isolation.describe()),
            "artifact_root": str(artifact_root),
            # 未绑核的样本在墙钟超时判定上不可比，必须留痕以便事后剔除。
            "cpu_pinned": pinned,
        }
        if self._preseed_note:
            base_payload["backend_preseed_error"] = self._preseed_note
        if len(self.roles) > 2:
            base_payload["multiplayer_note"] = "non-candidate seats filled by the same opponent"

        if status == "infra_error":
            note = f" (backend preseed: {self._preseed_note})" if self._preseed_note else ""
            return self._incomplete(case, f"infra_error: {diagnostic}{note}")

        if status == "game_error":
            # 契约语义：候选侧超时/非法/崩溃 = 有效负局，必须让 Goal 看见并学习。
            base_payload["game_error"] = str(diagnostic or "")
            return MatchResult(
                case=case,
                status="complete",
                result="loss",
                points=0.0,
                score_margin=0.0,
                rounds=rounds if rounds is not None else 0,
                payload=base_payload,
                replay_path=replay_path,
            )

        if status != "complete":
            return self._incomplete(case, f"unknown evaluator status: {status!r}")

        winner = payload.get("winner")
        candidate_score = float(scores.get(case.role, 0.0)) if scores else 0.0  # type: ignore[union-attr]
        rival_scores = [
            float(value)
            for role, value in (scores or {}).items()  # type: ignore[union-attr]
            if role != case.role
        ]
        margin = candidate_score - (max(rival_scores) if rival_scores else 0.0)
        if winner is None:
            outcome, points = "draw", 0.5
        elif str(winner) == case.role:
            outcome, points = "win", 1.0
        else:
            outcome, points = "loss", 0.0
        return MatchResult(
            case=case,
            status="complete",
            result=outcome,  # type: ignore[arg-type]
            points=points,
            score_margin=margin,
            rounds=rounds if rounds is not None else 0,
            payload=base_payload,
            replay_path=replay_path,
        )
