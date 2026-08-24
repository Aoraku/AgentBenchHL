"""Game-agnostic composition root and shared Codex runtime preflight.

The framework core assembles a run by looking up the game's adapter factory in
the registry (keyed by ``config.game``) and delegating the game-specific wiring
to it.  This module therefore contains no game import; it only owns the shared,
game-agnostic Codex App Server preflight and runtime helpers that any game's
factory can reuse.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from agentbench_hl.adapters.codex_goal.app_server import CodexGoalRuntime
from agentbench_hl.config import ExperimentConfig
from agentbench_hl.registry import build_game_adapters


@dataclass(frozen=True)
class CodexInstallationProbe:
    version: str
    goals_enabled: bool
    schema_tree_sha256: str
    schema_file_count: int


def goal_app_server_command(codex_binary: str) -> tuple[str, ...]:
    return (
        codex_binary,
        "app-server",
        "--listen",
        "stdio://",
        "--strict-config",
    )


def use_responses_compat_proxy(base_url: str) -> bool:
    return urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}


def _canonical_schema_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        value = json.loads(path.read_text(encoding="utf-8"))
        content = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def verify_codex_identity(
    resources: Mapping[str, object], probe: CodexInstallationProbe
) -> bool:
    stable_fields_match = all(
        (
            resources.get("codex_version") == probe.version,
            resources.get("codex_goals_enabled") == probe.goals_enabled,
            resources.get("codex_schema_file_count") == probe.schema_file_count,
        )
    )
    if not stable_fields_match:
        return False
    hash_kind = resources.get("codex_schema_hash_kind")
    if hash_kind is None:
        return True
    return all(
        (
            hash_kind == "canonical-json-v1",
            resources.get("codex_schema_tree_sha256") == probe.schema_tree_sha256,
        )
    )


def probe_codex_installation(
    codex_binary: str,
    probe_root: str | Path,
) -> CodexInstallationProbe:
    root = Path(probe_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            (codex_binary, *arguments),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"Codex preflight failed: {diagnostic}")
        return completed.stdout.strip()

    version = run("--version").splitlines()[0]
    features = run("features", "list")
    goals_enabled = any(
        line.split()[:1] == ["goals"] and line.split()[-1:] == ["true"]
        for line in features.splitlines()
    )
    if not goals_enabled:
        raise RuntimeError("Codex goals feature is not enabled")
    with tempfile.TemporaryDirectory(prefix="schema-", dir=root) as temporary:
        schema_root = Path(temporary)
        run(
            "app-server",
            "generate-json-schema",
            "--experimental",
            "--out",
            str(schema_root),
        )
        files = tuple(path for path in schema_root.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError("Codex generated no App Server schema files")
        schema_hash = _canonical_schema_tree_sha256(schema_root)
    return CodexInstallationProbe(version, True, schema_hash, len(files))


def codex_goal_runtime(
    config: ExperimentConfig,
    api_key: str,
    *,
    codex_home: Path,
) -> CodexGoalRuntime:
    """Build the shared isolated Codex Goal runtime from a frozen config."""

    return CodexGoalRuntime(
        command=goal_app_server_command(config.runtime.codex_binary),
        codex_home=codex_home,
        base_url=config.provider.base_url,
        model=config.provider.model,
        reasoning_effort=config.provider.reasoning_effort,
        api_key=api_key,
        use_responses_proxy=use_responses_compat_proxy(config.provider.base_url),
        sandbox_mode=config.isolation.agent_sandbox,
        context_window=config.provider.context_window,
        auto_compact_token_limit=config.provider.auto_compact_token_limit,
        model_catalog=config.provider.model_catalog,
        # None 时让运行时用它自己的默认值，不要在这里替它决定。
        **(
            {"client_name": config.provider.client_name}
            if config.provider.client_name
            else {}
        ),
    )


def read_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


# Backward-compatible private aliases retained for existing callers and tests.
_goal_app_server_command = goal_app_server_command
_use_responses_compat_proxy = use_responses_compat_proxy
_codex_identity_matches = verify_codex_identity
_read_json_object = read_json_object


def _game_of(config_path: str | Path) -> str:
    return ExperimentConfig.load(Path(config_path).resolve()).game


def challenger_seats(roles: Sequence[str], challenger_track: str | None) -> tuple[str, ...]:
    """候选这一轮要坐哪些座次。

    对称游戏（``challenger_track is None``）返回**全部**座次：先后手优势差很多，
    只打一边的胜率没有意义，所以每个对手都要正反各打一局。

    分轨游戏返回**唯一**那个座次。分轨游戏的两个角色不是"座次"而是两种不同的
    游戏 —— rollman 轨的选手只实现吃豆人一侧的协议。对手已经被过滤成对位轨了，
    这时再让候选去坐对位座次就是安排**同轨互殴**。

    实测 s8k4-rollman 第 1 轮 8 局里有 4 局就是这么来的::

        role=rollman → 352~500 回合，score_margin 37~140   （真实对局）
        role=ghost   → 0 回合，却记成 result=win / margin=0 （无效局）

    于是那一轮胜率被抬到 1.0。最坏的是它**归错了因**：监控只报得出
    "0 回合对局占 4/8 —— 候选大概率协议格式错"，而候选其实完全正常。
    """

    if challenger_track is None:
        return tuple(roles)
    if challenger_track not in roles:
        raise ValueError(f"challenger_track {challenger_track!r} 不在座次 {list(roles)} 里")
    return (challenger_track,)


def _load_player_tracks(agentbench_root: str | Path, game: str) -> dict[str, str]:
    """读 ``players/tracks.tsv``：非对称游戏里每名选手扮演哪个角色。

    只有分轨游戏才有这个文件（目前是 rollman：rollman / ghost 两轨）。
    对称游戏返回空字典，调用方据此跳过过滤——不要把"没有 tracks.tsv"
    当成错误，那会把 7 个对称游戏全挡掉。
    """

    path = Path(agentbench_root) / "games" / game / "players" / "tracks.tsv"
    if not path.is_file():
        return {}
    tracks: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            player = (row.get("player_id") or "").strip()
            track = (row.get("track") or "").strip()
            # unknown / unrunnable 不是角色轨，是分类失败的标记；
            # 把它们当轨道会让对手池里混进不可用选手。
            if player and track and track not in ("unknown", "unrunnable"):
                tracks[player] = track
    return tracks


def build_live_run(config_path: str | Path, *, run_id: str):
    """Assemble a fresh run for ``config.game`` through the game registry."""

    factory = build_game_adapters(_game_of(config_path))
    return factory.build_run(config_path, run_id=run_id)


def resume_live_run(config_path: str | Path, *, run_id: str):
    """Resume an existing run for ``config.game`` through the game registry."""

    factory = build_game_adapters(_game_of(config_path))
    return factory.resume_run(config_path, run_id=run_id)


def build_goal_led_service(
    config_path: str | Path,
    *,
    run_id: str,
    resume: bool = False,
):
    """Assemble the minimal Goal-led bridge without the Plan II controller.

    实验配置里的 harness / 对手策略 / K / seed / 并行度 / 预算 / 消融开关全部在这里
    落到 :class:`GoalLedService`，因此"网页表单填什么，后台就真的按什么跑"。
    """

    from agentbench_hl.adapters.contract.factory import build_goal_run
    from agentbench_hl.application.goal_led_service import GoalLedService

    config = ExperimentConfig.load(Path(config_path).resolve())
    # Plan I（服务化 Goal-led）统一走**游戏无关**的契约适配器：只依赖 A 的
    # `evaluate()` + GamePack，因此任何游戏零代码接入。Plan II（`abhl run …`）
    # 仍走 registry 里各游戏的原生 factory（含认证矩阵/策略探针等深度装配）。
    run = build_goal_run(config_path, run_id=run_id, resume=resume)
    if isinstance(run.runtime, CodexGoalRuntime):
        # Plan I is a true long-running Goal.  Its turn duration is controlled
        # by Codex itself, not by the five-minute Plan II checkpoint budget.
        #
        # 为什么是 2 小时而不是 1 小时
        # --------------------------
        # 这个预算是**扣掉上游退避之后**的净思考时间（app_server 会把退避等待从
        # deadline 里减掉，见 responses_proxy 的详注）。1 小时曾经够用，但那是
        # k=1 时代的量级；k=4 之后一轮要在同一个 turn 里读完 b×2 份回放**再写出
        # 4 份完整策略**，净思考时间本身就到几十分钟，中转一旦变慢（token/s 掉一半，
        # 但不报错、因此不计入退避）就会越过 1 小时。
        #
        # 实测 fix3-aquawar 第 2 轮：
        #     TimeoutError: Codex Goal did not reach a checkpoint in time
        #     （已扣除上游退避等待 721s，仍超过 3600s 预算）
        # 那一轮的 8 局对局本身是干净的，整个 run 却以 iteration_failed 报废。
        #
        # 注意这不是实验变量：它只决定"等多久算放弃"，不影响任何一轮的内容，
        # 所以放宽它对跨模型可比性没有影响（而调 reasoning_effort / rollout_k 有）。
        run.runtime.checkpoint_timeout_s = 7200.0
    seeds = (
        config.curriculum.development_seeds
        if config.curriculum.seed_mode == "generalize"
        else config.curriculum.development_seeds[:1]
    )
    # 角色（座次）名必须以 A 的 game.yaml 为唯一权威源。
    #
    # 这里曾经写 `getattr(run.arena, "roles", ("P0", "P1"))`——arena 没有暴露
    # roles 时就静默退回 P0/P1。对 antwar 这类对称游戏恰好是对的，但对
    # **非对称分轨**游戏是错的：rollman 的角色叫 rollman/ghost，
    # 拿 P0/P1 去跑会让每一局都以
    # `role P0 is not one of ('rollman', 'ghost')` 失败——而这在指标上表现为
    # "12/12 局 incomplete"，看起来像对局跑不起来，而不是座次名传错。
    #
    # 所以：优先问 game.yaml（唯一权威），arena 自己声明的 roles 只作校验；
    # 两者都拿不到就直接报错，绝不猜一个默认值。
    from agentbench_hl.adapters.contract.factory import game_roles

    roles = game_roles(config.paths.agentbench_root, config.game)
    arena_roles = getattr(run.arena, "roles", None)
    if arena_roles and tuple(arena_roles) != roles:
        raise ValueError(
            f"{config.game} 的座次定义不一致：game.yaml 说 {roles}，"
            f"arena 说 {tuple(arena_roles)}。请先对齐 A 侧定义再跑实验"
        )

    # 分轨（非对称）游戏必须按轨道过滤对手，否则会安排**同轨互殴**。
    #
    # 这类失败极其隐蔽：rollman 轨的选手只实现吃豆人一侧的协议，让两个
    # rollman 对打，双方都不响应对手协议，对局 0 回合结束、得分双 -1000，
    # 但对战器仍然记 ``status: complete`` 并判出一个 winner。于是
    #   * 胜率虚高（0 回合局也算赢），
    #   * IG 只录到 1 个决策点（KL 退化成 0 / ln(|A|/ε) 二值常数），
    #   * Elo 把无效局当有效局喂进 BT 拟合。
    # 实测 r4 的 ``rollman-seed-1`` replay 只有 2 行，而对位的
    # ``ghost-seed-1`` 有 1204 行——同一场对局差 600 倍。
    opponents = [item for item in run.opponents if item.runnable]
    tracks = _load_player_tracks(config.paths.agentbench_root, config.game)
    # 候选实际会坐的座次。对称游戏是全部座次（先后手都要打，见下）；
    # 分轨游戏**只有一个**，理由见 challenger_roles 的赋值处。
    challenger_roles = roles
    if tracks:
        # 候选扮演 challenger_track，对手只能来自对位轨。
        challenger_track = config.evaluation.challenger_track or roles[0]
        if challenger_track not in roles:
            raise ValueError(
                f"{config.game} 的座次只有 {list(roles)}，"
                f"evaluation.challenger_track 写的是 {challenger_track!r}"
            )
        opponent_track = next((item for item in roles if item != challenger_track), None)
        if opponent_track is None:
            raise ValueError(f"{config.game} 声明了分轨却只有一个角色：{roles}")
        eligible = [
            item
            for item in opponents
            if tracks.get(item.opponent_id) == opponent_track
        ]
        if not eligible:
            raise ValueError(
                f"{config.game} 的 {opponent_track} 轨没有可运行对手："
                f"候选扮演 {challenger_track}，对手池里 "
                f"{sorted({tracks.get(item.opponent_id) for item in opponents})} 都不对位"
            )
        opponents = eligible
        challenger_roles = challenger_seats(roles, challenger_track)

    return GoalLedService(
        run_root=run.root,
        bootstrap_root=run.bootstrap_root,
        gamepack_root=run.gamepack_root,
        runtime=run.runtime,
        arena=run.arena,
        model=run.model,
        model_provider=run.model_provider,
        runnable_opponent_ids=tuple(item.opponent_id for item in opponents),
        public_leaderboard=tuple(
            {
                "opponent_id": item.opponent_id,
                "rank": item.rank,
                # 分数可以为空：AquaWar 的外部参考排名只有加权分名次、没有 Elo。
                # 课程顺序由 rank 决定，硬要求分数会把 500+ 个可用对手全丢掉。
                "score": (None if item.score is None else float(item.score)),
                "score_source": getattr(item, "rank_source", "crawled"),
            }
            # 与 runnable_opponent_ids 用同一份已过滤集合：课程策略
            # （fixed_rank / ladder_up / …）是按榜单选对手的，只过滤前者
            # 会让它们继续选到同轨对手。
            for item in opponents
            if item.rank is not None
        ),
        game=config.game,
        roles=challenger_roles,
        seeds=seeds,
        rollout_k=config.runtime.rollout_k,
        opponent_policy=config.curriculum.opponent_policy,
        batch=config.curriculum.batch,
        opponent_rank=config.curriculum.opponent_rank,
        opponent_start_rank=config.curriculum.opponent_start_rank,
        advance_min_matches=config.curriculum.advance_min_matches,
        advance_win_rate=config.curriculum.advance_win_rate,
        advance_streak=config.curriculum.advance_streak,
        match_parallelism=config.runtime.match_parallelism,
        prompt_override=config.goal.prompt_override,
        experience_skills=config.goal.experience_skills,
        code_constraint=config.goal.code_constraint,
        history_mode=config.goal.history_mode,
        rival_code_visible=config.isolation.rival_code_visible,
        token_budget=config.budget.tokens,
        wall_budget_s=config.budget.wall_seconds,
        epsilon=config.measurement.epsilon,
        measure_information_gain=config.measurement.information_gain,
        behavioral_ig_cases=config.measurement.behavioral_ig_cases,
        behavioral_ig_timeout_s=config.measurement.behavioral_ig_timeout_s,
        behavioral_ig_coupling=config.measurement.behavioral_ig_coupling,
        behavioral_ig_probe=config.measurement.behavioral_ig_probe,
        # 行为 IG 的 |A| 与动作口径来自 A 仓 games/<game>/decision_space.yaml。
        agentbench_root=config.paths.agentbench_root,
        iteration_mode=config.runtime.iteration_mode,
        # 在 codex 的 remote compaction（对本模型必死）之前主动换 thread。
        thread_rotate_context_tokens=config.runtime.thread_rotate_context_tokens,
        thread_rotate_each_iteration=config.runtime.thread_rotate_each_iteration,
    )
