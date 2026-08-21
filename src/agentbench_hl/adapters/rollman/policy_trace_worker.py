"""Rollman（吃豆人）的候选侧决策探针。

它做什么
--------
在**候选自己的包**里重放一局公开回放：逐回合把状态喂给候选的策略函数，
记录它实际选了什么动作、以及当时的合法动作集（支持集）。上层
``application/support_probe.py`` 用这些数据算精确的 behavioral IG。

与 antwar / antwar2 的同构关系
------------------------------
三者都遵循同一个契约：
``--candidate/--replay/--match-id/--role`` 进，stdout 上一行
``AGENTBENCH_POLICY_TRACE=<json>`` 出，json 含 ``decisions[]``，
每项有 ``state_id`` / ``actions`` / ``legal_supports``。

Rollman 的特殊之处（决定了本文件的写法）
----------------------------------------
1. **动作空间静态**：吃豆人 5 个方向（STAY/UP/DOWN/LEFT/RIGHT），
   幽灵是 3 只各 5 个方向 = 125 个联合动作。撞墙**不算非法**（后端
   ``step()`` 会把撞墙走成原地不动并在路径块上打 ``HIT_OFFSET`` 标记），
   所以支持集就是全空间，不需要向 SDK 逐个询问合法性。
   这与 antwar 的"逐格 dry-run"完全不同——那边的合法性依赖金币/冷却/占位。

   ⚠️ 正因如此，本探针给出的 |A(s)| 是**常量**（5 或 125）。这不是近似：
   规则层面确实每个状态都有这么多合法动作。``decision_space.yaml`` 里
   ``support.mode: enumerated`` 的声明是准确的。

2. **回放是 JSONL 且不含完整棋盘快照**：逐步帧只有 ``pacman_step_block``
   这类增量，完整棋盘只出现在首帧与关卡切换帧。所以状态必须靠
   ``env.ai_reset(帧)`` 建立、再用 ``env.step(真实动作)`` 推进——
   不能像 antwar2 那样每回合 sync 一次公开状态。

3. **动作要从路径块反推**：回放不直接记"选了哪个方向"，只记走过的格子。
   相邻两格的差分就是方向；撞墙时路径块只有起点，此时按 STAY 处理
   （见 ``_direction_from_block``）。

4. **双角色不对称**：P0 控 3 只幽灵，P1 控吃豆人。``role`` 决定探测哪一侧。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

#: 后端在路径块上标记"撞墙"的偏移量（core/gamedata.py）。
PACMAN_HIT_OFFSET = -100
GHOST_HIT_OFFSET = -200

#: 五个方向的枚举值（core/gamedata.py::Direction）。
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
DIRECTIONS = (STAY, UP, DOWN, LEFT, RIGHT)

#: (dx, dy) → Direction。dx 是行增量、dy 是列增量，与后端 step() 一致。
_DELTA_TO_DIRECTION = {
    (0, 0): STAY,
    (1, 0): UP,
    (-1, 0): DOWN,
    (0, -1): LEFT,
    (0, 1): RIGHT,
}


def _load_frames(path: Path) -> list[dict]:
    """读 JSONL 回放。

    后端异常时末尾可能追加非 JSON 的 traceback 文本，按 replay_format.md
    的约定跳过——这不是容错兜底，是回放格式的一部分。
    """

    frames: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            frames.append(value)
    if not frames:
        raise ValueError("rollman replay has no usable frames")
    return frames


def _is_reset_frame(frame: Mapping[str, object]) -> bool:
    """是否是初始/关卡切换帧（带完整棋盘）。"""

    return "board" in frame and "beannumber" in frame


def _direction_from_block(block: object, *, hit_offset: int) -> int:
    """从路径块反推这一步选了哪个方向。

    路径块是本步走过的格子序列。取首尾两格的差分即方向；只有一格
    （撞墙或原地）时按 STAY 处理。撞墙格带 ``hit_offset`` 偏移，
    先还原再比较。
    """

    if not isinstance(block, Sequence) or len(block) < 2:
        return STAY
    first, last = block[0], block[-1]
    if not (isinstance(first, Sequence) and isinstance(last, Sequence)):
        return STAY
    if len(first) < 2 or len(last) < 2:
        return STAY

    def _restore(value: object) -> int:
        number = int(value)  # type: ignore[arg-type]
        # 撞墙标记是在坐标上加偏移，还原它才能算出真实位移。
        if number <= hit_offset // 2:
            number -= hit_offset
        return number

    delta = (
        _restore(last[0]) - _restore(first[0]),
        _restore(last[1]) - _restore(first[1]),
    )
    # 速度加成时一步会走两格；取符号即可还原方向。
    normalised = (
        (delta[0] > 0) - (delta[0] < 0),
        (delta[1] > 0) - (delta[1] < 0),
    )
    return _DELTA_TO_DIRECTION.get(normalised, STAY)


def _state_fingerprint(state: object) -> str:
    """状态指纹，用于 occupancy 去重。

    只取公开可见的字段（棋盘/坐标/分数/技能），与 antwar2 的
    ``_occupancy_id`` 同一思路：同一个局面无论怎么到达，指纹相同。
    """

    payload = {
        "level": int(getattr(state, "level", 0)),
        "round": int(getattr(state, "round", 0)),
        "board": _to_plain(getattr(state, "board", [])),
        "pacman": _to_plain(getattr(state, "pacman_pos", [])),
        "ghosts": _to_plain(getattr(state, "ghosts_pos", [])),
        "skills": _to_plain(getattr(state, "pacman_skill_status", [])),
        "score": [
            int(getattr(state, "pacman_score", 0)),
            int(getattr(state, "ghosts_score", 0)),
        ],
        "portal": bool(getattr(state, "portal_available", False)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _to_plain(value: object) -> object:
    """numpy → 原生类型，让 json 能序列化。"""

    if hasattr(value, "tolist"):
        return value.tolist()  # type: ignore[union-attr]
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _pacman_support() -> tuple[str, ...]:
    """吃豆人的支持集：5 个方向，恒定。

    撞墙不是非法动作——后端会把它执行成原地不动。所以合法集不随棋盘变化，
    这与 ``decision_space.yaml`` 的 ``by_role.pacman: 5`` 一致。
    """

    return tuple(f"P{direction}" for direction in DIRECTIONS)


def _ghost_support() -> tuple[str, ...]:
    """幽灵的支持集：3 只 × 5 方向的联合动作 = 125 个，恒定。"""

    return tuple(
        f"G{first}{second}{third}"
        for first in DIRECTIONS
        for second in DIRECTIONS
        for third in DIRECTIONS
    )


def _normalise_pacman_action(raw: object) -> int:
    """把候选返回值规整成一个方向枚举值。

    候选的 ``ai(game_state)`` 约定返回含 1 个元素的数组，但池子里的实现
    有直接返回 int 的。两种都接受——探针的职责是**观测**候选行为，
    不是校验它的返回格式（那是 arena 的事）。
    """

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if not raw:
            return STAY
        return int(raw[0])
    return int(raw)  # type: ignore[arg-type]


def _normalise_ghost_actions(raw: object) -> list[int]:
    """把候选返回值规整成 3 个方向。"""

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        actions = [int(item) for item in list(raw)[:3]]
    else:
        actions = [int(raw)] * 3  # type: ignore[arg-type]
    while len(actions) < 3:
        actions.append(STAY)
    return actions


def _find_entry_root(candidate: Path) -> Path:
    """定位候选包里真正可运行的目录（含 ``main.py`` 的最浅一层）。

    为什么不能直接用 ``candidate``：rollman 的候选包大多把官方 SDK 整个塞进
    一个子目录（``PacmanSDK-python-main/`` / ``GhostsSDK-python/`` 等），
    ``main.py`` 和 ``core/`` 都在那一层里，包根目录下什么都没有。

    这里复刻 A 侧 ``evaluator/runtime.py::_find_python_entry`` 的规则
    （深度最浅、同深度取字典序最小），保证探针和 arena 看到的是**同一个包**。
    口径不一致会让 IG 测的是另一份代码，比测不出来更糟。
    """

    candidates = sorted(
        candidate.rglob("main.py"),
        key=lambda path: (len(path.relative_to(candidate).parts), path.as_posix()),
    )
    if not candidates:
        raise RuntimeError(f"rollman candidate has no main.py: {candidate}")
    return candidates[0].parent


def _load_strategy(entry_root: Path, role: str):
    """从候选包里取出策略可调用对象。

    两类候选：

    **一、我们自己生成的候选（HL 迭代产物）** —— IG 真正要测的对象。
    它遵守框架注入的契约：``ai.py`` 里有 ``class AI``，方法是
    ``decide(self, game_state)``。先按这条契约找。

    **二、人类池里的历史候选** —— 接口五花八门：``ai(game_state)`` 顶层函数、
    ``PacmanAI`` / ``GhostAI`` 类等。按名字与常见方法去找。

    找不到就明确报错——绝不静默返回假策略，那会让 IG 变成纯噪声。
    """

    import importlib.util

    # ---------------------------------------------------------------- 路径一
    ai_py = entry_root / "ai.py"
    if ai_py.is_file():
        spec = importlib.util.spec_from_file_location("candidate_ai_module", ai_py)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception:  # noqa: BLE001 - 退到路径二
                module = None
            if module is not None:
                for class_name in ("AI", "Agent", "MyAI"):
                    ai_class = getattr(module, class_name, None)
                    if not isinstance(ai_class, type):
                        continue
                    try:
                        instance = ai_class()
                    except Exception:  # noqa: BLE001 - 构造失败就绕过 __init__
                        instance = ai_class.__new__(ai_class)  # type: ignore[misc]
                    for method_name in ("decide", "choose_move", "choose_moves", "act"):
                        handler = getattr(instance, method_name, None)
                        if callable(handler):
                            return handler
                # 类里找不到就退回模块级函数（历史候选常见写法）。
                names = (
                    ("ai", "pacman_ai") if role == "pacman" else ("ghost_ai", "ai")
                )
                for name in names:
                    handler = getattr(module, name, None)
                    if callable(handler) and not isinstance(handler, type):
                        return handler

    # ---------------------------------------------------------------- 路径二
    import ai as ai_module

    if role == "pacman":
        names = ("ai", "pacman_ai", "PacmanAI")
        methods = ("__call__", "decide", "choose_move")
    else:
        names = ("ghost_ai", "ai", "GhostAI")
        methods = ("__call__", "decide", "choose_moves")

    for name in names:
        attribute = getattr(ai_module, name, None)
        if attribute is None:
            continue
        if isinstance(attribute, type):
            try:
                instance = attribute()
            except Exception:  # noqa: BLE001
                instance = attribute.__new__(attribute)  # type: ignore[misc]
            for method in methods:
                handler = getattr(instance, method, None)
                if callable(handler):
                    return handler
            continue
        if callable(attribute):
            return attribute
    raise RuntimeError(f"candidate exposes no {role} strategy (tried {names})")


def _role_track(entry_root: Path, role: str) -> str:
    """判断本次要探测的是 rollman（吃豆人）轨还是 ghost（幽灵）轨。

    rollman 是**非对称分轨**游戏（``game.yaml``: ``roles: [rollman, ghost]``、
    ``roles_symmetric: false``），所以角色名就是 ``rollman`` / ``ghost``，
    **不是** antwar 那样的 ``P0`` / ``P1``。一个候选只实现一侧
    （包名就叫 PacmanSDK / GhostsSDK）。

    调用方给的 ``role`` 是权威来源；只有它是历史遗留的 P0/P1 时才回退到
    包名与 ``main.py`` 调用图去推断。搞错会把 ghost 候选按 pacman 去问，
    拿到的动作全是错的。
    """

    normalised = role.strip().lower()
    if normalised in ("rollman", "pacman"):
        return "pacman"
    if normalised == "ghost":
        return "ghost"

    lowered = entry_root.as_posix().lower()
    if "pacman" in lowered:
        return "pacman"
    if "ghost" in lowered:
        return "ghost"
    blob = ""
    main_py = entry_root / "main.py"
    if main_py.is_file():
        blob = main_py.read_text(encoding="utf-8", errors="ignore")
    if "pacman_to_judger" in blob or "pacman_op" in blob:
        return "pacman"
    if "ghost_to_judger" in blob or "ghosts_op" in blob:
        return "ghost"
    # 兜底：P1 是吃豆人（与 game.yaml 的 roles 顺序一致）。
    return "pacman" if normalised == "p1" else "ghost"


def _find_backend_root(entry_root: Path, explicit: Path | None) -> Path:
    """定位提供 ``core/`` 包的后端目录。

    为什么需要它：rollman 的候选包**不自带** ``core/``（棋盘/环境/枚举都在里面），
    只有 ``main.py`` + ``ai.py`` + ``utils/``。真实对局时是后端进程 import
    ``core.GymEnvironment``，候选通过 stdio 与它通信。探针要在候选侧重放，
    就必须自己把 ``core/`` 拿到。

    这与 antwar / antwar2 不同：那两个游戏的候选包里就有完整 SDK。

    按四级查找（前面的优先）：
      1. ``--backend-root`` 显式指定；
      2. 候选自带 ``core/``（少数候选会打包进来，那才是它真正运行的代码）；
      3. ``AGENTBENCH_ROOT`` 环境变量 —— **HL 迭代产生的候选必须靠这条**，
         因为它们在 ``runs/<run>/snapshots/`` 下，向上找不到 AgentBench；
      4. 从 entry_root 向上找 AgentBench 仓（适用于池子里的选手包）。
    """

    relative = Path("backend_sources/corpus/29_rollman/logic/gamecode_logic/PacmanLogic")

    if explicit is not None:
        if not (explicit / "core").is_dir():
            raise RuntimeError(f"--backend-root has no core/ package: {explicit}")
        return explicit

    if (entry_root / "core").is_dir():
        return entry_root

    import os

    agentbench_root = os.environ.get("AGENTBENCH_ROOT")
    if agentbench_root:
        candidate = Path(agentbench_root) / relative
        if (candidate / "core").is_dir():
            return candidate

    for parent in entry_root.resolve().parents:
        candidate = parent / relative
        if (candidate / "core").is_dir():
            return candidate
    raise RuntimeError(
        "cannot locate rollman backend core/ package; "
        "set AGENTBENCH_ROOT or pass --backend-root explicitly"
    )


def run(
    candidate: Path,
    replay_path: Path,
    match_id: str,
    role: str,
    backend_root: Path | None = None,
) -> dict[str, object]:
    import sys

    entry_root = _find_entry_root(candidate)
    backend = _find_backend_root(entry_root, backend_root)
    # 顺序要紧：entry_root 在最前，保证 ai.py 取的是**这个候选**的策略；
    # backend 提供 core/ 包。两者同名模块不冲突（候选侧没有 core/）。
    for path in (str(backend), str(entry_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from core.GymEnvironment import PacmanEnv

    frames = _load_frames(replay_path)
    track = _role_track(entry_root, role)
    strategy = _load_strategy(entry_root, track)
    is_pacman = track == "pacman"
    support = _pacman_support() if is_pacman else _ghost_support()

    environment = PacmanEnv()
    decisions: list[dict[str, object]] = []
    initialised = False

    for index, frame in enumerate(frames):
        if _is_reset_frame(frame):
            # 关卡开始（含首帧）：用完整棋盘重建状态，本帧没有动作可探测。
            environment.ai_reset(frame)
            initialised = True
            continue
        if not initialised:
            continue

        state = environment.game_state()
        level = int(frame.get("level", 0))
        round_index = int(frame.get("round", index))
        state_id = f"{match_id}:l{level:d}:r{round_index:04d}"

        # 先让候选面对这个状态做决策——这是我们要观测的量。
        # 策略抛异常时记 STAY 并继续：一个状态上的崩溃不该让整局探针作废
        # （与 antwar 的"判定崩溃按非法处理"同一原则）。
        try:
            raw = strategy(state)
            if is_pacman:
                chosen = [f"P{_normalise_pacman_action(raw)}"]
            else:
                actions = _normalise_ghost_actions(raw)
                chosen = [f"G{actions[0]}{actions[1]}{actions[2]}"]
        except Exception:  # noqa: BLE001 - 候选策略崩溃，记为 STAY 并继续
            chosen = ["P0"] if is_pacman else ["G000"]

        decisions.append(
            {
                "state_id": state_id,
                "actions": chosen,
                "legal_supports": [list(support)],
                "occupancy_id": _state_fingerprint(state),
            }
        )

        # 再用回放里**真实发生**的动作推进状态，保证后续状态与实际对局一致。
        pacman_action = _direction_from_block(
            frame.get("pacman_step_block"), hit_offset=PACMAN_HIT_OFFSET
        )
        ghost_blocks = frame.get("ghosts_step_block") or []
        ghost_actions = [
            _direction_from_block(
                ghost_blocks[slot] if slot < len(ghost_blocks) else None,
                hit_offset=GHOST_HIT_OFFSET,
            )
            for slot in range(3)
        ]
        try:
            environment.step(pacman_action, ghost_actions)
        except Exception as error:  # noqa: BLE001
            # 状态推进失败说明回放与后端版本不匹配，继续下去只会产生错误数据。
            raise RuntimeError(
                f"failed to replay rollman transition at frame {index}: {error}"
            ) from error

    return {"match_id": match_id, "role": role, "decisions": decisions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    # 角色名由 game.yaml 定义。rollman 是非对称分轨游戏，用的是
    # rollman/ghost 而不是 P0/P1；这里放宽取值，由 _role_track 归一化。
    parser.add_argument("--role", required=True)
    parser.add_argument(
        "--backend-root",
        type=Path,
        default=None,
        help="提供 core/ 包的后端目录；省略时自动从 backend_sources 定位",
    )
    arguments = parser.parse_args()
    result = run(
        arguments.candidate.resolve(),
        arguments.replay.resolve(),
        arguments.match_id,
        arguments.role,
        arguments.backend_root.resolve() if arguments.backend_root else None,
    )
    print("AGENTBENCH_POLICY_TRACE=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
