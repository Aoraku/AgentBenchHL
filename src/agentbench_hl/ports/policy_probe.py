"""决策级策略探针契约（行为信息增益的唯一合法来源）。

行为信息增益（``behavioral_ig``，nats/决策）要求在**冻结的公开状态**上重放策略，
并在该状态的**合法动作支撑**上比较两份策略的分布。这两件事都需要游戏语义：

- 如何从回放里还原公开状态（回放 schema 是每个游戏自己的）；
- 如何枚举该状态下的合法动作（合法性判定在游戏 SDK 里）。

所以框架 core 只定义本 port，具体实现放在 ``adapters/<game>/policy_probe.py``，
并在这里注册。**没有注册探针的游戏，行为信息增益诚实记 null**（连同原因），
绝不用结果分布 KL 之类的替代量冒充它。

另外要注意一个真实存在的契约差异：

- Plan II（``abhl run``）的候选契约是 **``ai.py`` + 冻结 SDK**（in-process 调用
  ``AI.choose_operations``），现有 antwar2 探针就是按这个契约写的；
- Plan I（Goal-led 服务化）的候选契约是 A 的官方选手包 **``main.py`` 进程**
  （stdio 协议），无法被上面那种 in-process 探针驱动。

因此 :func:`probe_binding` 在返回探针时会一并声明它要求的候选契约，调用方必须先用
:func:`probe_availability` 判定当前候选是否满足；不满足时返回不可用原因，
而不是给出一个"看起来像但其实测的不是同一个东西"的数。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agentbench_hl.domain.policy import PolicyEpisodeTrace


class PolicyProbe(Protocol):
    """在一份回放上重放候选策略，得到逐决策的动作与合法支撑。"""

    def __call__(
        self,
        candidate_root: str | Path,
        replay_path: str | Path,
        *,
        match_id: str,
        role: str,
    ) -> PolicyEpisodeTrace: ...


@dataclass(frozen=True)
class ProbeBinding:
    """一个游戏的探针绑定：实现 + 它要求的候选契约 + schema 版本。"""

    game: str
    probe: PolicyProbe
    candidate_contract: str
    contract_marker: str
    schema: str

    def satisfied_by(self, candidate_root: str | Path) -> bool:
        return (Path(candidate_root) / self.contract_marker).exists()


_FACTORIES: dict[str, Callable[[], ProbeBinding]] = {}


def register_policy_probe(game: str, factory: Callable[[], ProbeBinding]) -> None:
    _FACTORIES[game] = factory


def probe_binding(game: str) -> ProbeBinding | None:
    factory = _FACTORIES.get(game)
    if factory is None:
        return None
    return factory()


def probe_availability(game: str, candidate_root: str | Path) -> tuple[ProbeBinding | None, str]:
    """返回（可用的探针绑定, 原因）。不可用时绑定为 None，原因写进事件与指标。"""

    binding = probe_binding(game)
    if binding is None:
        return None, f"no policy probe registered for game {game!r}"
    if not binding.satisfied_by(candidate_root):
        return None, (
            f"candidate does not satisfy probe contract {binding.candidate_contract!r} "
            f"(missing {binding.contract_marker})"
        )
    return binding, "available"


def _antwar2_binding() -> ProbeBinding:
    from agentbench_hl.adapters.antwar2.policy_probe import probe_policy_episode  # noqa: PLC0415

    return ProbeBinding(
        game="antwar2",
        probe=probe_policy_episode,
        candidate_contract="ai.py + frozen SDK (Plan II in-process contract)",
        contract_marker="ai.py",
        schema="antwar2-atomic-v1",
    )


register_policy_probe("antwar2", _antwar2_binding)
