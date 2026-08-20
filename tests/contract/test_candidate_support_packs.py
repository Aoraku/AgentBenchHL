"""候选脚手架的护栏测试：每个 GamePack 的 ``candidate_support`` 必须真的能起进程。

这些测试的作用是**防回退**。脚手架的失效方式是静默的——候选起不来时对战器只报
"0 回合判负"，主表上会读成"这个模型在这个游戏完全不行"。历史上踩过的坑：

* 候选包少一个 ``ai.py``，连烧 5 轮迭代；
* 官方 SDK 需要数据文件（miracle 的 ``Data.json``），只 vendor ``.py`` 就 import 失败；
* 官方 SDK 用相对路径读数据，工作目录不对同样 import 失败；
* ``BaseAgent`` 是 ABC，只覆盖非抽象方法会导致**无法实例化**。

上面每一条都会被本文件里的 ``check_candidate``（含真实启动探针）当场抓住，
不需要等到跑完一局对局才发现。

真跑一局的验收是 ``scripts/verify_candidate_support.py``（需要 A 仓与 Linux 沙箱），
这里只做能在任何机器上快速重复的部分。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMEPACKS = REPO_ROOT / "gamepacks"
PROBES = GAMEPACKS / "_shared" / "candidate_probes"

sys.path.insert(0, str(REPO_ROOT / "src"))

from agentbench_hl.application.candidate_preflight import check_candidate  # noqa: E402


def _discover() -> list[tuple[str, str | None, Path]]:
    """所有（游戏, 角色轨, 脚手架目录）。

    每个游戏**只有一个**候选包（非对称游戏也一样，一次 run 里同一份候选快照
    要打完所有角色）。多套官方 SDK 放在包内 ``<track>_sdk/`` 子目录，
    角色轨由此推断。
    """

    found: list[tuple[str, str | None, Path]] = []
    for pack in sorted(GAMEPACKS.iterdir()):
        if not pack.is_dir() or pack.name.startswith("_"):
            continue
        support = pack / "candidate_support"
        if not support.is_dir():
            continue
        tracks = sorted(
            item.name.removesuffix("_sdk")
            for item in support.iterdir()
            if item.is_dir() and item.name.endswith("_sdk")
        )
        if tracks:
            found.extend((pack.name, track, support) for track in tracks)
        else:
            found.append((pack.name, None, support))
    return found


SUPPORTS = _discover()
IDS = [game if track is None else f"{game}-{track}" for game, track, _ in SUPPORTS]


def _probe_for(game: str, track: str | None) -> Path:
    """最小合法 ai.py：优先按轨，否则用该游戏通用那份（rollman 用通用的）。"""

    if track is not None:
        per_track = PROBES / f"{game}-{track}.py"
        if per_track.is_file():
            return per_track
    return PROBES / f"{game}.py"


def _interface_for(game: str) -> str | None:
    manifest = GAMEPACKS / game / "manifest.yaml"
    if not manifest.is_file():
        return None
    document = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    value = document.get("candidate_interface")
    return str(value) if value else None


def test_discovered_all_eight_games() -> None:
    """8 个游戏都要有脚手架；rollman 非对称，单包内含两条角色轨的 SDK。"""

    games = {game for game, _, _ in SUPPORTS}
    assert games == {
        "antwar",
        "antwar2",
        "aquawar",
        "generals",
        "lostspace",
        "miracle",
        "rollman",
        "snakego",
    }
    rollman_tracks = {track for game, track, _ in SUPPORTS if game == "rollman"}
    assert rollman_tracks == {"rollman", "ghost"}


@pytest.mark.parametrize("game", sorted({game for game, _, _ in SUPPORTS}))
def test_manifest_declares_support_as_plain_string(game: str) -> None:
    """``candidate_support`` 必须是**字符串**。

    ``gamepack.path_for()`` 只接受字符串；写成 ``{track: dir}`` 会让该游戏在搭
    Goal 工作区时直接抛 ``GamePackError``——rollman 曾经就是这样，一挂实验就崩，
    而且崩在真正开跑之前，很容易被当成"配置写错了"。
    """

    document = yaml.safe_load((GAMEPACKS / game / "manifest.yaml").read_text(encoding="utf-8"))
    value = document.get("candidate_support")
    assert isinstance(value, str) and value, f"{game}: candidate_support={value!r} 必须是非空字符串"
    assert (GAMEPACKS / game / value).is_dir()


@pytest.mark.parametrize(("game", "track", "support"), SUPPORTS, ids=IDS)
def test_support_pack_shape(game: str, track: str | None, support: Path) -> None:
    """脚手架必须自带入口、引导层与契约说明，并且不能夹带候选该自己写的 ai.py。"""

    assert (support / "main.py").is_file(), "对战器以 python main.py 启动候选"
    assert (support / "_bootstrap.py").is_file()
    assert (support / "CANDIDATE_CONTRACT.md").is_file(), "agent 需要知道 AI 要实现什么"
    assert (support / "SUPPORT_PROVENANCE.json").is_file(), "官方 SDK 来源必须可追溯"
    assert not (support / "ai.py").exists(), (
        "ai.py 是候选自己要写的；脚手架里放了就等于替 agent 写好了策略"
    )
    assert _probe_for(game, track).is_file(), "缺少验收探针，无法证明脚手架可用"


@pytest.mark.parametrize(("game", "track", "support"), SUPPORTS, ids=IDS)
def test_candidate_starts_with_minimal_ai(
    game: str, track: str | None, support: Path, tmp_path: Path
) -> None:
    """脚手架 + 最小合法 ``ai.py`` 必须通过 preflight（含真实启动探针）。

    ``check_candidate`` 会真的 ``python main.py`` 起一次进程：import 失败、
    ABC 无法实例化、缺数据文件这些问题都会在这里暴露，而不是等到对局里。
    """

    candidate = tmp_path / "candidate"
    shutil.copytree(support, candidate)
    shutil.copy2(_probe_for(game, track), candidate / "ai.py")

    issues = check_candidate(
        f"probe-{game}",
        candidate,
        candidate_interface=_interface_for(game),
    )
    assert not issues, "\n".join(issue.as_note() for issue in issues)


@pytest.mark.parametrize(("game", "track", "support"), SUPPORTS, ids=IDS)
def test_missing_ai_is_reported(
    game: str, track: str | None, support: Path, tmp_path: Path
) -> None:
    """**没有** ``ai.py`` 时必须报错——这正是当年连烧 5 轮迭代的那个失败模式。"""

    candidate = tmp_path / "candidate"
    shutil.copytree(support, candidate)

    issues = check_candidate(
        f"probe-{game}-no-ai",
        candidate,
        candidate_interface=_interface_for(game),
    )
    assert issues, "缺少 ai.py 居然没被 preflight 拦下"
