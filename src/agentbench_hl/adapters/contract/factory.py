"""通用游戏 Factory —— 任何有 GamePack 的游戏都能装配出一次 Goal-led 迭代。

与 `adapters/antwar2/factory.py`（Plan II 全套：认证矩阵/策略探针/IG 度量）相比，
本 factory 面向**服务化的 Plan I（Goal-led）**：只需要

    GamePack（引用 A 的规则/决策空间/回放说明） + A 的对战器 + A 的选手池

因此**接入新游戏零代码**：放一个 `gamepacks/<game>/`（可全部引用 A）即可。

产出的 bundle 与 `RunService` 在 goal-led 路径上鸭子兼容
（root / bootstrap_root / gamepack_root / runtime / arena / model / opponents）。
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import yaml

from agentbench_hl.adapters.contract.arena import ContractArena
from agentbench_hl.adapters.contract.pool import (
    PoolPlayer,
    load_pool,
    public_leaderboard,
    ranked_ladder,
    runnable_players,
)
from agentbench_hl.adapters.isolation import select_candidate_isolation
from agentbench_hl.adapters.transcript import transcript_root
from agentbench_hl.application.opponent_policy import effective_batch_for
from agentbench_hl.config import ExperimentConfig, repository_root_for
from agentbench_hl.gamepack import GamePack
from agentbench_hl.ports.isolation import CandidateIsolation, IsolationRequest


class ContractFactoryError(RuntimeError):
    """无法为该游戏装配一次 Goal-led run。"""


@dataclass
class GoalRunBundle:
    """一次 Goal-led run 的装配结果（供 `build_goal_led_service` 消费）。"""

    root: Path
    bootstrap_root: Path
    gamepack_root: Path
    runtime: object
    arena: ContractArena
    model: str
    model_provider: str
    opponents: tuple[PoolPlayer, ...]
    config: ExperimentConfig
    pool: tuple[PoolPlayer, ...]


def game_roles(agentbench_root: Path, game: str) -> tuple[str, ...]:
    """从 A 的 ``game.yaml`` 读角色（唯一权威源；玩家数 = len(roles)）。"""

    path = agentbench_root / "games" / game / "game.yaml"
    if not path.is_file():
        raise ContractFactoryError(f"AgentBench game metadata not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    roles = document.get("roles") if isinstance(document, Mapping) else None
    if not isinstance(roles, list) or not roles:
        raise ContractFactoryError(f"game.yaml has no roles list: {path}")
    return tuple(str(item) for item in roles)


def _supported_player_build_systems(agentbench_root: Path, game: str) -> frozenset[str]:
    """读取 A 评测机声明的精确构建系统集合。"""

    module = agentbench_root / "games" / game / "evaluator" / "runtime.py"
    if not module.is_file():
        return frozenset()
    text = module.read_text(encoding="utf-8", errors="ignore")
    declaration = re.search(r"^SUPPORTED_PLAYER_BUILD_SYSTEMS\s*=\s*(.+)$", text, re.MULTILINE)
    if declaration is not None:
        try:
            value = ast.literal_eval(declaration.group(1))
        except (SyntaxError, ValueError):
            return frozenset()
        if isinstance(value, (tuple, list, set, frozenset)):
            return frozenset(str(item) for item in value if str(item) in {"make", "cmake"})
        return frozenset()
    # 旧版 A 只有函数名信号。它至多证明传统 Make 可用，绝不能据此放行 CMake。
    if "SUPPORTS_COMPILED_PLAYERS = False" in text:
        return frozenset()
    if "SUPPORTS_COMPILED_PLAYERS = True" in text:
        return frozenset({"make"})
    if "def prepare_player" in text or "def build_player" in text:
        return frozenset({"make"})
    return frozenset()


def _supports_compiled_players(agentbench_root: Path, game: str) -> bool:
    """Backward-compatible boolean view for callers that only need yes/no."""

    return bool(_supported_player_build_systems(agentbench_root, game))


def _isolation_provider(
    config: ExperimentConfig,
    run_root: Path,
) -> object:
    """返回 ``IsolationRequest -> CandidateIsolation`` 工厂（带 run 级 profile 路径）。"""

    def build(request: IsolationRequest, *, index: list[int] = [0]) -> CandidateIsolation:  # noqa: B006
        index[0] += 1
        return select_candidate_isolation(
            request,
            backend=config.isolation.backend,
            profile_path=run_root / "isolation" / f"candidate-{index[0]:05d}.sb",
            docker_image=config.isolation.docker_image,
        )

    return build


def _materialize_bootstrap(
    gamepack: GamePack,
    frozen_gamepack: Path,
    bootstrap: Path,
    *,
    seed_policy: Path | None,
) -> None:
    """搭出容器的初始内容：**图上那 6 样输入 + 格式示例**，别的一样都不放。

    容器里允许存在的东西（架构图 §Isolated Container）：

    ==================  ===============================================
    Game Rule           ``gamepack/rules.md``
    Decision Space      ``gamepack/decision_space.yaml``
    Replay Tutorial     ``gamepack/replay_skill.md`` + ``replay_format.md``
    Human Ranking       ``leaderboard.json``（由 GoalLedService 写，只有 id/rank/score）
    Iterated Code       agent 自己历次写的候选（工作区根 + ``.agentbench/rollouts/``）
    Experience          ``research/EXPERIENCE.md``
    格式示例 + 接口契约   候选脚手架（``main.py`` / ``ai_example.py`` /
                        ``CANDIDATE_CONTRACT.md`` / 官方协议层）
    ==================  ===============================================

    ⚠️ **``gamepack/candidate_support/`` 必须排除**。``GamePack.materialize()`` 会把
    ``candidate_support`` 也快照进 ``frozen-gamepack``（那是为了留指纹、可复现），
    而脚手架**已经**平铺在工作区根上了。整份 frozen_gamepack 直接 copytree 进来，
    等于容器里出现**两份**脚手架：一份能跑（根上那份），一份是死副本
    （``gamepack/candidate_support/``）。死副本没有任何用途，却是第二条泄漏路径 ——
    历史上被剔除的 ``tools/`` / 训练脚本如果哪天回归，会先从这里漏进来。
    """

    bootstrap.mkdir(parents=True, exist_ok=True)
    support = gamepack.path_for("candidate_support")
    if support is not None and support.is_dir():
        for item in sorted(support.iterdir()):
            target = bootstrap / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    # 公开研究材料（已在 frozen_gamepack 冻结）以只读副本形式进入工作区，
    # 这样 agent 在容器里"看得到规则/决策空间/回放阅读指南"。
    docs = bootstrap / "gamepack"
    if docs.exists():
        shutil.rmtree(docs)
    shutil.copytree(
        frozen_gamepack,
        docs,
        ignore=shutil.ignore_patterns("candidate_support"),
    )
    if seed_policy is not None:
        if seed_policy.is_dir():
            shutil.copytree(seed_policy, bootstrap, dirs_exist_ok=True)
        else:
            shutil.copy2(seed_policy, bootstrap / seed_policy.name)


#: 续跑时**允许**变化的配置键（顶层段名 → 该段内允许变化的字段）。
#:
#: 为什么需要区分
#: --------------
#: "续跑时配置必须逐字节一致"这道校验本身是对的：中途改模型、改 k、改对手
#: 策略会让前后轮次不可比，而事故只会在事后画图时才显现。
#:
#: 但它原来把**观测通道**和**实验变量**一视同仁，于是"打开慢评测"这种
#: 完全不碰迭代的改动也会让续跑直接失败：
#:
#:     ContractFactoryError: current experiment config differs from frozen run config
#:
#: 报错还不说是哪个字段变了，只能靠人 diff 两份 JSON。实测 random 组因此
#: 无法续跑最后 6 轮——而它死于第 26 轮的 checkpoint 超时，本该一条命令救回来。
#:
#: 判据：**这个字段会不会改变 agent 看到的东西或它的对局？** 不会的就放行。
#: * ``evaluation``：慢评测完全在另一个进程、另建对局，只写 pool-elo/。
#: * ``budget``：预算是停机条件而不是实验变量；续跑本来就是"再给一些预算"。
#: * ``runtime.max_iterations``：同理，"再跑几轮"是续跑的定义本身。
#: * ``runtime.match_parallelism`` / ``match_timeout_s``：并发度与超时是
#:   机时调度参数。超时放宽会改变"能不能跑完"，但不改变对局的规则与对手。
_RESUME_MUTABLE: dict[str, frozenset[str] | None] = {
    "evaluation": None,  # None = 整段都可变
    "budget": None,
    "runtime": frozenset({"max_iterations", "match_parallelism", "match_timeout_s"}),
}


def _significant_config_changes(previous: dict, current: dict) -> list[str]:
    """列出续跑时**不允许**变化的配置差异（空列表 = 可以续跑）。

    返回的是"段.字段"清单而不是布尔值：报错必须说出到底哪里变了，
    否则调用方只能自己 diff 两份 JSON 去猜。
    """

    changed: list[str] = []
    for key in sorted(set(previous) | set(current)):
        before, after = previous.get(key), current.get(key)
        if before == after:
            continue
        if key not in _RESUME_MUTABLE:
            changed.append(key)
            continue
        allowed = _RESUME_MUTABLE[key]
        if allowed is None:
            continue
        if not isinstance(before, dict) or not isinstance(after, dict):
            changed.append(key)
            continue
        for field in sorted(set(before) | set(after)):
            if before.get(field) != after.get(field) and field not in allowed:
                changed.append(f"{key}.{field}")
    return changed


def build_goal_run(
    config_path: str | Path,
    *,
    run_id: str,
    resume: bool = False,
) -> GoalRunBundle:
    """为 ``config.game`` 装配（或恢复）一次 Goal-led run。"""

    source = Path(config_path).resolve()
    config = ExperimentConfig.load(source)
    repository_root = repository_root_for(source)
    gamepack_root = repository_root / "gamepacks" / config.game
    agentbench_root = config.paths.agentbench_root
    run_root = config.paths.runs_root / run_id
    if not resume and (run_root / "run-manifest.json").exists():
        raise ContractFactoryError(f"run already exists: {run_id}")
    run_root.mkdir(parents=True, exist_ok=True)

    gamepack = GamePack.load(gamepack_root, agentbench_root=agentbench_root)
    frozen_gamepack = run_root / "frozen-gamepack"
    digests = gamepack.materialize(frozen_gamepack)

    roles = game_roles(agentbench_root, config.game)
    pool = load_pool(
        agentbench_root,
        config.game,
        supported_build_systems=_supported_player_build_systems(
            agentbench_root, config.game
        ),
        # 榜单口径决定对手选择策略能挑到谁（official 只有 11–32 人，
        # reference/measured 可达数百人）。见 contract/pool.py 的模块文档。
        ladder_scope=config.curriculum.ladder_scope,
    )
    runnable = runnable_players(pool)
    if not runnable:
        raise ContractFactoryError(
            f"{config.game} has no runnable player in A's pool; run `abhl pool audit {config.game}`"
        )
    ladder = ranked_ladder(pool)

    bootstrap = run_root / "bootstrap"
    if not resume:
        _materialize_bootstrap(
            gamepack,
            frozen_gamepack,
            bootstrap,
            seed_policy=config.goal.seed_policy_path,
        )

    hidden_read_roots: tuple[Path, ...] = (
        run_root / "codex-home",
        run_root / "agent-home",
        repository_root / ".env",
        gamepack_root / "evaluator",
    )
    arena = ContractArena(
        game=config.game,
        agentbench_root=agentbench_root,
        roles=roles,
        artifact_root=run_root / "official-matches",
        build_root=run_root / "frozen-build",
        isolation_factory=_isolation_provider(config, run_root),
        hidden_read_roots=hidden_read_roots,
        # 行为信息增益的录制文件：录制垫片在沙箱内往这里写线协议流水。
        # 只放开 transcripts 目录，录制克隆的代码目录仍然只读。
        extra_writable_roots=(transcript_root(run_root / "behavioral-ig"),),
        opponents={item.player_id: item for item in runnable},
        # 单局墙钟上限来自配置（默认 1800s）。曾经硬编码 420s，而 snakego 单局实测
        # 246s、评分时还有 19 局连 900s 都超——那会把长局判成候选的失败。
        timeout_s=config.runtime.match_timeout_s,
        # 每步上限：saiblo 按步计时，我们以前只有整局墙钟（详注见配置项本身）。
        step_timeout_s=config.runtime.step_timeout_s,
    )

    runtime = _build_runtime(config, run_root)

    manifest = {
        "schema_version": "1.1",
        "run_id": run_id,
        "game": config.game,
        "roles": list(roles),
        "harness": config.provider.harness,
        "model": config.provider.model,
        "isolation_backend": config.isolation.backend,
        "gamepack_digests": digests,
        "pool": {
            "total": len(pool),
            "runnable": len(runnable),
            "ranked_runnable": len(ladder),
        },
        "opponent_policy": config.curriculum.opponent_policy,
        "rollout_k": config.runtime.rollout_k,
        # b 是一等实验变量（"一轮对局数 = k × b × 座次"），清单里以前**根本没记**,
        # 于是读账本的人无从校验对局数。
        #
        # 记的是**实际值**而不是配置里的 batch：单目标策略（ladder_up /
        # ladder_down / fixed_rank）无论 batch 写多少都只打 1 个对手。
        # exp2 主线就是 ladder_up 配着默认 batch: 4，照抄配置会让清单里写 4、
        # 实际只打 1 个，对局数因此差 4 倍。
        "batch": effective_batch_for(
            config.curriculum.opponent_policy, config.curriculum.batch
        ),
        "match_parallelism": config.runtime.match_parallelism,
        "budget": {
            "tokens": config.budget.tokens,
            "wall_seconds": config.budget.wall_seconds,
        },
    }
    (run_root / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    frozen = config.frozen_dict()
    config_path_out = run_root / "run-config.json"
    if resume and config_path_out.is_file():
        previous = json.loads(config_path_out.read_text(encoding="utf-8"))
        changed = _significant_config_changes(previous, frozen)
        if changed:
            raise ContractFactoryError(
                "current experiment config differs from frozen run config: "
                + ", ".join(changed)
            )
        # 只有观测通道变了（例如打开慢评测）：更新冻结副本并继续。
        # 见 _significant_config_changes 的详注。
        if previous != frozen:
            config_path_out.write_text(
                json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
    else:
        config_path_out.write_text(
            json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    (run_root / "public-leaderboard.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "game": config.game,
                "opponents": list(public_leaderboard(pool)),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return GoalRunBundle(
        root=run_root,
        bootstrap_root=bootstrap,
        gamepack_root=frozen_gamepack,
        runtime=runtime,
        arena=arena,
        model=config.provider.model,
        model_provider="OpenAI",
        opponents=ladder or runnable,
        config=config,
        pool=pool,
    )


def _build_runtime(config: ExperimentConfig, run_root: Path) -> object:
    """按 harness 选择 Agent 运行时（codex / cc）。"""

    if config.provider.harness == "codex":
        from agentbench_hl.application.live_run import codex_goal_runtime  # noqa: PLC0415

        secret = config.secret_environment()[config.provider.api_key_env]
        return codex_goal_runtime(config, secret, codex_home=run_root / "codex-home")
    if config.provider.harness == "cc":
        from agentbench_hl.adapters.cc_goal.runtime import claude_code_runtime  # noqa: PLC0415

        secret = config.secret_environment()[config.provider.api_key_env]
        return claude_code_runtime(config, secret, agent_home=run_root / "agent-home")
    raise ContractFactoryError(f"unsupported harness: {config.provider.harness}")


class ContractAdapterFactory:
    """注册表用的工厂：任何游戏都用同一份实现。"""

    def __init__(self, game: str) -> None:
        self.game = game

    def build_run(self, config_path: str | Path, *, run_id: str) -> GoalRunBundle:
        return build_goal_run(config_path, run_id=run_id, resume=False)

    def resume_run(self, config_path: str | Path, *, run_id: str) -> GoalRunBundle:
        return build_goal_run(config_path, run_id=run_id, resume=True)


build_contract_factory = partial(ContractAdapterFactory)
