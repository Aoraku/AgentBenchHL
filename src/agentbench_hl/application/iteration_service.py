"""Pure lifecycle decisions and compact prompts for Goal-led iterations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.application.evaluation_service import EvaluationResult
from agentbench_hl.domain.lineage import LineageState


def _text(value: str, field: str) -> str:
    if not value.strip():
        raise ValueError(f"iteration proposal {field} cannot be empty")
    return value.strip()


@dataclass(frozen=True)
class IterationProposal:
    condition: str
    mechanism: str
    intervention: str
    expected_observation: str
    continuation_rationale: str

    def __post_init__(self) -> None:
        for field in (
            "condition",
            "mechanism",
            "intervention",
            "expected_observation",
            "continuation_rationale",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field))


@dataclass(frozen=True)
class IterationPlan:
    iteration: int
    version_id: str
    parent_id: str
    target_id: str
    locked_regression_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("scientific iteration must be positive")
        if not (
            self.version_id.startswith("v")
            and len(self.version_id) >= 4
            and self.version_id[1:].isdigit()
        ):
            raise ValueError("iteration plan version must have vNNN form")
        for field in ("parent_id", "target_id"):
            object.__setattr__(self, field, _text(getattr(self, field), field))

    def to_payload(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "target_id": self.target_id,
            "locked_regression_ids": list(self.locked_regression_ids),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> IterationPlan:
        locked = payload.get("locked_regression_ids")
        if not isinstance(locked, list):
            raise ValueError("iteration plan locked regressions must be a list")
        return cls(
            int(payload["iteration"]),
            str(payload["version_id"]),
            str(payload["parent_id"]),
            str(payload["target_id"]),
            tuple(str(item) for item in locked),
        )


def choose_iteration_parent(lineage: LineageState) -> str:
    parent = lineage.frontier_id or lineage.champion_id
    if parent is None:
        raise ValueError("iteration requires a Champion or Frontier parent")
    return parent


def select_iteration_candidate(
    lineage: LineageState,
    candidate_id: str,
    evaluation: EvaluationResult,
    proposal: IterationProposal,
) -> LineageState:
    if evaluation.promotable:
        return lineage.promote(candidate_id)
    if evaluation.frontier_eligible:
        return lineage.choose_frontier(
            candidate_id,
            rationale=proposal.continuation_rationale,
        )
    return lineage


def build_iteration_prompt(
    *,
    iteration: int,
    parent_id: str,
    target_id: str,
    gamepack_root: str | Path,
    replay_root: str | Path,
    research_root: str | Path,
    evidence_entries: tuple[str, ...] = (),
) -> str:
    if iteration < 1:
        raise ValueError("improvement iteration must be positive")
    gamepack = Path(gamepack_root).resolve()
    replays = Path(replay_root).resolve()
    research = Path(research_root).resolve()
    evidence = "\n".join(f"- {entry}" for entry in evidence_entries)
    evidence_block = (
        f"\n目标正式对局证据索引：\n{evidence}\n"
        "若索引中存在败局，必须优先从败局中至少引用一个真实 state_id，再修改代码。\n"
        if evidence_entries
        else ""
    )
    return f"""执行科研迭代 {iteration}，父版本 {parent_id}，默认目标 {target_id}。

按需读取冻结规则与决策空间：{gamepack}
按需读取公开语义回放：{replays}
每轮必须读取并继续补充正负经验：{research}
{evidence_block}

先用回放 state_id 定位一个具体条件，解释失败机制，再提出代码干预和可证伪的预期
观察。禁止没有因果区别的参数枚举或网格搜索；允许规则树、状态机、规划、启发式
评分和确定性搜索，也允许代码增长及单点修补。

上下文预算纪律：不要把整个 ai.py、SDK 或 replay timeline 原样打印到对话中；
用 rg/grep 定位符号，用 sed/head/tail 读取必要的小片段，并在本地脚本中汇总统计。
单次命令输出控制在约 12,000 个字符以内；已经读过的文件不要重复输出。
最多先执行 4 次只读检查；随后必须立即修改 ai.py（即使是最小、可证伪的改动）并
写入 proposal.json，然后结束 checkpoint。不要为了“彻底阅读”继续追加检查命令。

只修改当前候选工作区。保留公开入口，并写入 `.agentbench/proposal.json`，字段为
condition、mechanism、intervention、expected_observation、continuation_rationale。
写完代码与 proposal 后结束本 checkpoint；后端会独立执行导入和公开 SDK 合法性检查。
不要执行删除或清理命令（尤其是 rm、rm -rf、find -delete）；不要删除任何已有文件。
如需检查文件，直接使用 ls、find（不带 -delete）或 Python 读取；完成后立即写入
proposal.json 并结束本 checkpoint。
"""
