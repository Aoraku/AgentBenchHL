"""人类选手池 / 公开排行榜 —— 唯一数据源是 A 的 ``players/`` 目录。

B 不再维护第二份选手清单或第二套 Elo 锚点。本模块只读 A：

```
AgentBench/games/<game>/players/
    manifest.tsv            选手清单（player_id/dir/username/rank/elo/submission_id/…）
    reference_ranking.tsv   外部交付的参考排名（ref_rank/ref_elo/ref_metric），可选
    measured_ranking.tsv    我们自己全池实测的评分（measured_rank/measured_elo），可选
    runnable.json           self-play smoke 审计结论（只做降级）
```

**先把术语说清**：这个池子**没有"官方榜"**。它是分两批爬下来的，``manifest.tsv``
里带 ``rank``/``elo`` 的那少数几十个只是**第一批爬取时顺带记下的名次**，覆盖窄、
也不构成任何权威口径。真正可用的名次与分数是我们自己全池实测的
``measured_ranking.tsv``。

**榜单口径（ladder scope）**——直接决定对手选择策略能挑到谁，因此必须显式：

* ``crawled``   : ``manifest.tsv`` 自带的 ``rank``/``elo``（第一批爬取的残留）。
                  覆盖窄（每个游戏只有 11–32 人，antwar2 是 20），**只是爬取副产品，
                  不是权威**。留着它只为可追溯，正常实验不要用。
* ``reference`` : 外部交付的参考排名。覆盖宽（202–534 人），**不是我们自己跑的**，
                  可靠性未知；用它必须在实验记录里标注。
* ``measured``  : 我们自己在本机全池循环赛拟合出来的 Elo。**这是默认应当使用的口径**。
* ``auto``      : measured → reference → crawled 依次回落（默认）。

**跑不起来的选手等于不存在**：``runnable.json`` 的 self-play smoke 审计结论是硬门槛。
以 antwar2 为例，631 人里 525 有可执行代码、只有 **229 通过冒烟**，
下游（``runnable_players`` / ``ranked_ladder`` / 公开排行榜）一律只用这 229 个。
剩下的连自对弈都跑不完，不可能当对手，也就没有 Elo 可言。

对 Goal 公开的只有 **rank / score / player_id**（图上的 "Human Ranking"）；
``dir``（源码位置）与 ``username`` 永不进入 Goal 的可读区。
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

#: 榜单口径。``crawled`` 是正名（那批 rank 只是第一批爬取的副产品）；
#: ``official`` 保留为弃用别名，免得炸掉既有配置。
LADDER_SCOPES = ("auto", "crawled", "reference", "measured")
DEPRECATED_SCOPE_ALIASES = {"official": "crawled"}


def normalize_ladder_scope(scope: str) -> str:
    """把口径名收敛到正名；非法值直接报错，不静默降级。"""

    text = str(scope or "auto").strip() or "auto"
    text = DEPRECATED_SCOPE_ALIASES.get(text, text)
    if text not in LADDER_SCOPES:
        raise PoolError(f"unknown ladder scope: {scope!r} (expected {LADDER_SCOPES})")
    return text
REFERENCE_FILENAME = "reference_ranking.tsv"
MEASURED_FILENAME = "measured_ranking.tsv"


class PoolError(RuntimeError):
    """选手池缺失或不可用。"""


@dataclass(frozen=True)
class PoolPlayer:
    player_id: str
    rank: int | None
    elo: float | None
    username: str | None
    version: str | None
    lang: str | None
    split: str | None
    package_root: Path
    runnable: bool
    exclusion_diagnostic: str | None = None
    # 非对称游戏的角色天梯（rollman: "rollman" | "ghost"）；对称游戏为 None。
    # 语义：该提交**只实现这一个角色**，因此不能做对称 self-play，评分也要按角色分开。
    track: str | None = None
    # 排名来源：official（manifest）/ reference（外部交付）/ measured（我们实测）。
    # 上层必须能回答"这个 Elo 是谁测的"，否则跨口径混用会得出错误结论。
    rank_source: str = "crawled"
    # 结构探测只做分类，不把评测机尚未支持的提交形状冒充成坏策略。
    availability_status: str = "runnable"

    @property
    def public_row(self) -> dict[str, object]:
        """Goal 可见的公开排行榜行（**不含**源码路径与真实用户名）。"""

        return {
            "opponent_id": self.player_id,
            "rank": self.rank,
            "score": self.elo,
            "score_source": self.rank_source,
        }

    # -- 与 B 现有 ``Opponent`` 结构的鸭子兼容别名（避免上层再做映射） --------

    @property
    def opponent_id(self) -> str:
        return self.player_id

    @property
    def score(self) -> float | None:
        return self.elo


def _to_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _nested_entry(package_root: Path) -> Path | None:
    """递归找 Python 入口，取最浅、再按路径字典序最小。

    很多游戏的提交是"把整个 SDK 目录一起打包上传"，入口在下一层：
    rollman 309 个包里 **188 个**是这种结构，只有 93 个把 main.py 放在包根。
    只看根目录会把三分之二的选手误判为"不可运行"。
    """

    if not package_root.is_dir():
        return None
    candidates = [
        item
        for item in package_root.rglob("main.py")
        if "__pycache__" not in item.relative_to(package_root).parts
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda path: (len(path.relative_to(package_root).parts), path.as_posix()),
    ).parent


def _native_build_systems(package_root: Path) -> frozenset[str]:
    systems: set[str] = set()
    for name in ("Makefile", "makefile", "GNUmakefile", "CMakeLists.txt"):
        for path in package_root.rglob(name):
            if "__pycache__" in path.relative_to(package_root).parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lowered = text.lower()
            if name == "CMakeLists.txt":
                recognized = "add_executable" in lowered
            else:
                recognized = "sphinx-build" not in lowered and any(
                    marker in text
                    for marker in (
                        "g++",
                        "clang++",
                        "CXX",
                        "CXXFLAGS",
                        ".cpp",
                        ".cc",
                        ".cxx",
                    )
                )
            if recognized:
                systems.add("cmake" if name == "CMakeLists.txt" else "make")
    return frozenset(systems)


def _has_native_source(package_root: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}
        for path in package_root.rglob("*")
    )


def classify_availability(
    package_root: Path,
    *,
    supports_compiled: bool,
    supported_build_systems: frozenset[str] | None = None,
) -> tuple[str, str | None]:
    """Classify package structure without treating evaluator gaps as player failures."""

    if not package_root.is_dir():
        return "missing_package", "package directory is missing"
    if (package_root / "main.py").is_file() or _nested_entry(package_root) is not None:
        return "runnable", None
    build_systems = _native_build_systems(package_root)
    supported = (
        supported_build_systems
        if supported_build_systems is not None
        else frozenset({"make", "cmake"})
        if supports_compiled
        else frozenset()
    )
    if build_systems:
        if build_systems & supported:
            return "runnable", None
        found = ", ".join(sorted(build_systems))
        accepted = ", ".join(sorted(supported)) or "none"
        return (
            "evaluator_unsupported",
            f"native build system unsupported by evaluator: found={found}; accepted={accepted}",
        )
    if _has_native_source(package_root):
        return "build_metadata_required", "native source found but no supported build entry"
    return "unsupported_submission_shape", "no recognized Python or native build entry"


def _runnable(
    package_root: Path,
    *,
    supports_compiled: bool,
    supported_build_systems: frozenset[str] | None = None,
) -> tuple[bool, str | None]:
    status, diagnostic = classify_availability(
        package_root,
        supports_compiled=supports_compiled,
        supported_build_systems=supported_build_systems,
    )
    return status == "runnable", diagnostic


def _tracks(game_dir: Path) -> dict[str, str]:
    """读 A 生成的 ``players/tracks.tsv``：非对称游戏的"每个选手实现哪个角色"。

    这张表由 A 的 ``scripts/classify_rollman_tracks.py`` 用**对战器自己的原语**
    （``evaluator/runtime.py::player_track``）生成，B 只消费不判定——角色识别属于
    游戏语义，必须只有一处定义。表不存在时返回空表（对称游戏就没有这个文件）。
    """

    target = game_dir / "players" / "tracks.tsv"
    if not target.is_file():
        return {}
    tracks: dict[str, str] = {}
    try:
        with target.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                player_id = (row.get("player_id") or "").strip()
                track = (row.get("track") or "").strip()
                if player_id and track in {"rollman", "ghost"}:
                    tracks[player_id] = track
    except OSError:
        return {}
    return tracks


def evaluator_content_sha256(agentbench_root: str | Path, game: str) -> str:
    """Hash the exact evaluator/core source content that determines audit verdicts."""

    root = Path(agentbench_root).resolve()
    sources = (
        root / "src" / "agentbench" / "core",
        root / "games" / game / "evaluator",
    )
    digest = hashlib.sha256()
    for source in sources:
        label = source.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        if not source.is_dir():
            digest.update(b"<missing>")
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(root).as_posix().encode("utf-8")
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def load_valid_audit_document(game_dir: Path) -> dict[str, object] | None:
    """Load an audit only when its content provenance matches the current evaluator."""

    target = game_dir / "players" / "runnable.json"
    if not target.is_file():
        return None
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or not isinstance(document.get("audit_fingerprint"), str):
        return None
    expected = document.get("evaluator_content_sha256")
    if not isinstance(expected, str) or not expected:
        return None
    current = evaluator_content_sha256(game_dir.parents[1], game_dir.name)
    return document if expected == current else None


def _audit_verdicts(game_dir: Path) -> dict[str, tuple[bool, str | None]]:
    """读取 self-play smoke 审计结论（见 ``application/pool_audit.py``）。

    语义只做**降级**：被审计过且失败的选手判为不可用；没被审计过的保持文件探测结果
    （"未审计"不等于"不可用"）。
    """

    document = load_valid_audit_document(game_dir)
    if document is None:
        return {}
    verdicts: dict[str, tuple[bool, str | None]] = {}
    for row in document.get("rows") or []:
        if not isinstance(row, dict):
            continue
        player_id = row.get("player_id")
        if not isinstance(player_id, str):
            continue
        verdicts[player_id] = (
            bool(row.get("verified")),
            (str(row["diagnostic"]) if row.get("diagnostic") else None),
        )
    return verdicts


def _ranking_overlay(
    game_dir: Path, filename: str, prefix: str
) -> dict[str, tuple[int | None, float | None]]:
    """读取 ``reference_ranking.tsv`` / ``measured_ranking.tsv`` 的名次+分数覆盖层。

    只读两列：``<prefix>_rank`` 与 ``<prefix>_elo``。缺文件、缺列、坏行都安静跳过——
    这两份文件是**可选增强**，不该让整个池子加载失败。
    """

    target = game_dir / "players" / filename
    if not target.is_file():
        return {}
    overlay: dict[str, tuple[int | None, float | None]] = {}
    try:
        with target.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                player_id = (row.get("player_id") or "").strip()
                if not player_id:
                    continue
                overlay[player_id] = (
                    _to_int(row.get(f"{prefix}_rank")),
                    _to_float(row.get(f"{prefix}_elo")),
                )
    except OSError:
        return {}
    return overlay


def apply_ladder_scope(
    players: tuple[PoolPlayer, ...],
    agentbench_root: str | Path,
    game: str,
    scope: str = "auto",
) -> tuple[PoolPlayer, ...]:
    """按口径重写每个选手的 ``rank``/``elo``，并标注 ``rank_source``。

    ``auto`` 的回落顺序是 measured → reference → crawled：我们自己实测的最可信，
    外部交付次之，爬取副产品垫底。**逐选手**回落（不是整池二选一），这样刚跑完一部分
    实测评分也能立刻用上，而没覆盖到的人仍保留原有名次。

    ``official`` 是 ``crawled`` 的**弃用别名**：这个池子没有"官方榜"，那些 rank 只是
    第一批爬取的副产品。保留别名只为不炸掉既有配置。
    """

    scope = normalize_ladder_scope(scope)
    if scope == "crawled":
        return players
    game_dir = Path(agentbench_root).resolve() / "games" / game
    measured = _ranking_overlay(game_dir, MEASURED_FILENAME, "measured")
    reference = _ranking_overlay(game_dir, REFERENCE_FILENAME, "ref")

    def pick(player: PoolPlayer) -> PoolPlayer:
        chain: list[tuple[str, dict[str, tuple[int | None, float | None]]]] = []
        if scope in ("auto", "measured"):
            chain.append(("measured", measured))
        if scope in ("auto", "reference"):
            chain.append(("reference", reference))
        for source, overlay in chain:
            entry = overlay.get(player.player_id)
            if entry is None:
                continue
            rank, elo = entry
            if rank is None and elo is None:
                continue
            return replace(player, rank=rank, elo=elo, rank_source=source)
        if scope == "auto":
            return player  # 回落到爬取时带的名次
        # 显式指定了 measured/reference 却没覆盖这个人：清空名次，避免
        # 把官方 Elo 冒充成实测/参考值（宁缺毋滥）。
        return replace(player, rank=None, elo=None, rank_source=scope)

    return tuple(pick(player) for player in players)


def load_pool(
    agentbench_root: str | Path,
    game: str,
    *,
    supports_compiled: bool = False,
    supported_build_systems: frozenset[str] | None = None,
    apply_audit: bool = True,
    ladder_scope: str = "auto",
) -> tuple[PoolPlayer, ...]:
    """读取某游戏的完整选手池（含不可运行者，用于审计与诚实统计）。

    ``apply_audit=False`` 时**忽略** ``players/runnable.json``，只按文件探测判定
    可运行性。审计自身必须用这个模式（否则重审只会重审上一次的幸存者，
    失败者永无翻案机会，池子每审一次就单调缩小）。

    ``ladder_scope`` 见模块文档；默认 ``official`` 保持历史行为不变。
    """

    root = Path(agentbench_root).resolve()
    game_dir = root / "games" / game
    manifest = game_dir / "players" / "manifest.tsv"
    if not manifest.is_file():
        raise PoolError(f"player manifest not found: {manifest}")
    verdicts = _audit_verdicts(game_dir) if apply_audit else {}
    tracks = _tracks(game_dir)
    players: list[PoolPlayer] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            player_id = (row.get("player_id") or "").strip()
            if not player_id:
                continue
            relative = (row.get("dir") or "").strip()
            package_root = (game_dir / "players" / relative).resolve() if relative else game_dir
            availability, diagnostic = classify_availability(
                package_root,
                supports_compiled=supports_compiled,
                supported_build_systems=supported_build_systems,
            )
            runnable = availability == "runnable"
            verdict = verdicts.get(player_id)
            if verdict is not None:
                verified, audit_diagnostic = verdict
                if not verified:
                    runnable = False
                    availability = "runtime_failed"
                    diagnostic = audit_diagnostic or "failed self-play smoke audit"
                elif runnable:
                    diagnostic = None
            players.append(
                PoolPlayer(
                    player_id=player_id,
                    rank=_to_int(row.get("rank")),
                    elo=_to_float(row.get("elo")),
                    username=(row.get("username") or "").strip() or None,
                    version=(row.get("version") or "").strip() or None,
                    lang=(row.get("lang") or "").strip() or None,
                    split=(row.get("split") or "").strip() or None,
                    package_root=package_root,
                    runnable=runnable,
                    exclusion_diagnostic=diagnostic,
                    track=tracks.get(player_id),
                    availability_status=availability,
                )
            )
    if not players:
        raise PoolError(f"player manifest is empty: {manifest}")
    return apply_ladder_scope(tuple(players), root, game, ladder_scope)


def tracks_of(players: tuple[PoolPlayer, ...]) -> tuple[str, ...]:
    """池子里出现过的角色天梯（对称游戏返回空元组）。"""

    return tuple(sorted({item.track for item in players if item.track}))


def players_in_track(
    players: tuple[PoolPlayer, ...], track: str
) -> tuple[PoolPlayer, ...]:
    """某条角色天梯里的可运行选手（按 rank 升序）。"""

    return tuple(item for item in runnable_players(players) if item.track == track)


def opposing_track(track: str) -> str | None:
    """rollman ↔ ghost 的对位角色。"""

    return {"rollman": "ghost", "ghost": "rollman"}.get(track)


def runnable_players(players: tuple[PoolPlayer, ...]) -> tuple[PoolPlayer, ...]:
    """按 rank 升序（rank 缺失排最后）返回可运行选手。"""

    runnable = [item for item in players if item.runnable]
    runnable.sort(key=lambda item: (item.rank is None, item.rank if item.rank is not None else 0))
    return tuple(runnable)


def ranked_ladder(
    players: tuple[PoolPlayer, ...], *, require_score: bool = False
) -> tuple[PoolPlayer, ...]:
    """可运行且**有名次**的选手（公开排行榜与对手选择策略的作用域）。

    ``require_score=False``（默认）：只要求 rank。定义课程顺序靠的是名次，分数只是
    强度锚点——例如 AquaWar 的外部参考排名只有加权分名次、没有 Elo，若强求分数会
    把 534 个可用对手全部丢掉。需要 Elo 锚点的地方（单对手 Elo 反推）自己检查
    ``score is None``。
    """

    ladder = [item for item in runnable_players(players) if item.rank is not None]
    if require_score:
        ladder = [item for item in ladder if item.elo is not None]
    return tuple(ladder)


def public_leaderboard(players: tuple[PoolPlayer, ...]) -> tuple[dict[str, object], ...]:
    """图上 "Human Ranking"：只含 opponent_id / rank / score。"""

    return tuple(item.public_row for item in ranked_ladder(players))


def find_player(players: tuple[PoolPlayer, ...], player_id: str) -> PoolPlayer:
    for item in players:
        if item.player_id == player_id:
            return item
    raise PoolError(f"unknown player_id: {player_id}")
