"""隔离沙箱内的单局执行器 —— 只通过 A 的公开契约跑一局。

为什么要独立进程：候选策略与官方后端必须在**禁网 / 只读 / 遮蔽隐藏材料**的沙箱里
运行，而 A 的 `evaluate(game, players, roles, seed)` 是一个 in-process 调用，无法从外面
给它注入沙箱。于是我们把"调 A 跑一局"这件事本身放进沙箱：

```
bwrap …隔离参数… python -m agentbench_hl.adapters.contract.match_worker <request.json>
```

沙箱内：
- 可读：系统 + A 的后端资产 + 本局对手包（其它选手包被 tmpfs 遮蔽）；
- 可写：仅本局工件目录与后端构建目录；
- 无网络。

输出：一行 JSON 到 stdout（B 的 arena 解析成 ``MatchResult``）。
本模块**不 import 任何具体游戏**：游戏语义全部在 A 的 `games/<game>/evaluator/`。
"""

from __future__ import annotations

import contextlib
import fcntl
import inspect
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PREPARED_MARKER = ".prepared"
BUILD_LOCK = ".build.lock"


@contextlib.contextmanager
def _build_guard(build_root: Path) -> Iterator[None]:
    """跨进程串行化"首次后端准备"。

    A 的对战器在 ``build_root`` 下用 cache + quarantine 目录做原子构建。多个 match
    进程同时首次构建同一个 build_root 会撞成
    ``backend cache and quarantine both exist`` → 整批对局 infra_error（实测复现）。

    这里用文件锁保证只有一个进程做首次准备：构建成功后写 ``.prepared`` 标记，
    之后的进程直接走无锁快路径（真正的并行阶段没有任何额外开销）。
    """

    build_root.mkdir(parents=True, exist_ok=True)
    marker = build_root / PREPARED_MARKER
    if marker.exists():
        yield
        return
    lock_path = build_root / BUILD_LOCK
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
            marker.touch()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_evaluator(agentbench_root: Path, game: str, build_root: Path, artifact_root: Path) -> Any:
    """按 A 的注册表装配该游戏的对战器，并尽量使用 per-case 的构建/工件目录。

    per-case 目录是并行安全的前提：默认目录是 ``A/data/<game>-build`` 共享路径，
    32 路并发会互相踩；而且在沙箱里 A 仓是**只读**的，写默认目录会直接
    ``[Errno 30] Read-only file system``（rollman/snakego 全池审计 0 通过就是这个原因）。

    各游戏构造函数支持的关键字并不一致（例如 rollman/snakego 只有 ``artifact_root``，
    没有 ``build_root``）。因此**逐个探测签名**按需传参——原先"两个一起传，
    TypeError 就整体回落"的写法会让这些游戏静默退回只读默认目录。
    """

    source_root = agentbench_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from agentbench.core.registry import get_plugin  # noqa: PLC0415 - 沙箱内延迟导入

    games_root = agentbench_root / "games"
    plugin = get_plugin(game, games_root)
    game_dir = games_root / game
    evaluator = plugin.evaluator_factory(game_dir)
    factory = type(evaluator)
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # pragma: no cover - 极少数 C 扩展
        return evaluator
    kwargs: dict[str, Any] = {}
    if "build_root" in parameters:
        kwargs["build_root"] = build_root
    if "artifact_root" in parameters:
        kwargs["artifact_root"] = artifact_root
    if not kwargs:
        return evaluator
    try:
        return factory(game_dir, **kwargs)
    except TypeError:
        return evaluator


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    agentbench_root = Path(request["agentbench_root"]).resolve()
    game = str(request["game"])
    build_root = Path(request["build_root"]).resolve()
    artifact_root = Path(request["artifact_root"]).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    source_root = agentbench_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from agentbench.core.contract import PlayerRef  # noqa: PLC0415

    players = [
        PlayerRef(player_id=str(item["player_id"]), code_path=item.get("code_path"))
        for item in request["players"]
    ]
    roles = [str(item) for item in request["roles"]]
    canonical_roles = [str(item) for item in (request.get("canonical_roles") or roles)]
    seed = int(request["seed"])

    evaluator = _load_evaluator(agentbench_root, game, build_root, artifact_root)
    with _build_guard(build_root):
        result = evaluator.evaluate(players, roles, seed)
        result = _retry_with_canonical_roles(
            evaluator, result, players, roles, canonical_roles, seed
        )
    status = getattr(result.status, "value", str(result.status))
    return {
        "status": status,
        "winner": result.winner,
        "scores": dict(result.scores or {}),
        "rounds": result.rounds,
        "replay_path": result.replay_path,
        "diagnostic": result.diagnostic,
    }


def _retry_with_canonical_roles(
    evaluator: Any,
    result: Any,
    players: list[Any],
    roles: list[str],
    canonical_roles: list[str],
    seed: int,
) -> Any:
    """A 的各游戏对 ``roles`` 顺序有**两套约定**，这里自适应，不写死游戏名表。

    * 约定一（antwar2 / miracle …）：``roles[0]`` 就是候选的座次，允许任意排列，
      我们默认按"候选优先"传参。
    * 约定二（lostspace …）：``roles`` 必须是规范序 ``[P0, P1, P2, P3]``，
      ``players[i]`` 坐 ``roles[i]``；传排列会直接 INFRA_ERROR
      （lostspace 全池 192 人审计 0 通过就是这个原因）。

    做法：先按约定一试；只有当对战器**明确抱怨 roles 顺序**时，才按座次对齐重排
    一次。结果归属不受影响——我们自己按 ``scores[候选座次]`` 判定胜负，不依赖
    对战器对"谁是候选"的理解。
    """

    status = getattr(result.status, "value", str(getattr(result, "status", "")))
    diagnostic = str(getattr(result, "diagnostic", "") or "")
    if status != "infra_error" or "roles must be" not in diagnostic:
        return result
    if list(roles) == list(canonical_roles):
        return result  # 已经是规范序，抱怨的是别的事
    seat_of = dict(zip(roles, players, strict=False))
    if set(seat_of) != set(canonical_roles):
        return result
    ordered = [seat_of[role] for role in canonical_roles]
    return evaluator.evaluate(ordered, list(canonical_roles), seed)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            json.dumps(
                {"status": "infra_error", "diagnostic": "usage: match_worker <request.json>"}
            )
        )
        return 2
    request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    try:
        payload = run_request(request)
    except BaseException as exc:  # noqa: BLE001 - 必须把任何故障翻译成契约三态
        payload = {
            "status": "infra_error",
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - 子进程入口
    raise SystemExit(main(sys.argv))
