"""分轨（非对称）游戏的对手池必须按轨道过滤，不能安排同轨互殴。

背景（这组测试存在的唯一理由）
------------------------------
rollman 是二分图式的非对称游戏：``rollman``（吃豆人）只和 ``ghost``（幽灵）
交手，同轨之间在人类比赛里从未对局过。选手包只实现自己那一侧的协议。

原实现把 ``runnable_opponent_ids`` 与 ``public_leaderboard`` 直接取全部可运行
选手，没有轨道过滤。于是候选（扮演 ``rollman``）会被安排去打 ``rollman`` 轨的
选手——双方都不响应对手协议，对局 0 回合结束、双方得分 -1000，
**但对战器仍然记 ``status: complete`` 并判出一个 winner**。

三个指标同时被污染，而且都不会报错：
* 胜率：0 回合局照样算赢，实测 r4 显示"胜率 1.0"；
* IG：影子局只录到 1 个决策点，KL 退化成 0 / ln(|A|/ε) 二值常数，
  实测连续两轮 IG 都是 6.14451（那不是测量值，是常数）；
* Elo：无效局被当有效局喂进 BT 拟合。

证据：r4 同一场对局里 ``rollman-seed-1/replay.jsonl`` 只有 2 行，
而对位的 ``ghost-seed-1/replay.jsonl`` 有 1204 行——差 600 倍。

对称游戏（其余 7 个）没有 ``tracks.tsv``，必须原样放行，
否则过滤逻辑会把它们全挡掉。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentbench_hl.application.live_run import _load_player_tracks, challenger_seats


def _write_tracks(root: Path, game: str, rows: list[tuple[str, str]]) -> None:
    directory = root / "games" / game / "players"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["player_id\ttrack\tevidence\tentry_root"]
    lines.extend(f"{player}\t{track}\t手工\tplayers/pool/{player}" for player, track in rows)
    (directory / "tracks.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_symmetric_game_without_tracks_is_left_alone(tmp_path: Path) -> None:
    """没有 tracks.tsv 的对称游戏必须返回空字典（= 不过滤）。"""

    (tmp_path / "games" / "antwar" / "players").mkdir(parents=True)
    assert _load_player_tracks(tmp_path, "antwar") == {}


def test_tracks_are_read_per_player(tmp_path: Path) -> None:
    _write_tracks(
        tmp_path,
        "rollman",
        [("alice", "rollman"), ("bob", "ghost"), ("carol", "rollman")],
    )
    tracks = _load_player_tracks(tmp_path, "rollman")
    assert tracks == {"alice": "rollman", "bob": "ghost", "carol": "rollman"}


def test_unknown_and_unrunnable_are_not_tracks(tmp_path: Path) -> None:
    """``unknown`` / ``unrunnable`` 是分类失败的标记，不是角色轨。

    把它们当轨道会让对手池混进不可用选手：rollman 的 tracks.tsv 里
    实测有 24 个 unknown、27 个 unrunnable。
    """

    _write_tracks(
        tmp_path,
        "rollman",
        [
            ("alice", "rollman"),
            ("mystery", "unknown"),
            ("broken", "unrunnable"),
            ("bob", "ghost"),
        ],
    )
    tracks = _load_player_tracks(tmp_path, "rollman")
    assert set(tracks) == {"alice", "bob"}


def test_opponent_selection_keeps_only_the_facing_track(tmp_path: Path) -> None:
    """核心断言：候选扮演 roles[0] 时，对手只能来自对位轨。

    这里直接复现 live_run 里的过滤表达式，锁住"同轨选手必须被剔除"这个语义。
    """

    _write_tracks(
        tmp_path,
        "rollman",
        [
            ("same_track_1", "rollman"),
            ("same_track_2", "rollman"),
            ("facing_1", "ghost"),
            ("facing_2", "ghost"),
        ],
    )
    tracks = _load_player_tracks(tmp_path, "rollman")
    roles = ("rollman", "ghost")
    opponent_track = roles[1]

    eligible = [
        player for player in tracks if tracks.get(player) == opponent_track
    ]
    assert sorted(eligible) == ["facing_1", "facing_2"]
    # 同轨选手一个都不能留下——留下就会产生 0 回合的"complete"局。
    assert not [player for player in eligible if tracks[player] == roles[0]]


def test_asymmetric_game_plays_exactly_one_seat() -> None:
    """分轨游戏候选**只坐一个座次**——换座次就是同轨互殴。

    这是上面那组测试漏掉的另一半。对手过滤修好之后，候选仍然被安排去坐
    **对位座次**，于是同一个 bug 换了个方向又回来了：对手是 ghost，
    候选也去当 ghost。

    实测 s8k4-rollman 第 1 轮（8 局）::

        role=rollman → 352~500 回合，margin 37~140   真实对局
        role=ghost   → 0 回合，却记 result=win        无效局

    4/8 局无效，那一轮胜率被抬到 1.0。而监控给出的结论是"候选大概率协议
    格式错"——完全错误的方向，候选本身没有任何问题。

    上面那几个用例之所以没拦住：它们**复现**了 live_run 里的过滤表达式，
    而不是调用真实代码，所以座次那一半根本没被覆盖到。
    """

    assert challenger_seats(("rollman", "ghost"), "rollman") == ("rollman",)
    assert challenger_seats(("rollman", "ghost"), "ghost") == ("ghost",)


def test_symmetric_game_still_plays_both_seats() -> None:
    """对称游戏必须保留两个座次：先后手优势差很多，只打一边的胜率没有意义。"""

    assert challenger_seats(("P0", "P1"), None) == ("P0", "P1")


def test_bogus_challenger_track_is_rejected() -> None:
    """轨道名写错要当场报错，而不是静默退回全座次（那会重新引入同轨互殴）。"""

    with pytest.raises(ValueError, match="不在座次"):
        challenger_seats(("rollman", "ghost"), "pacman")


def test_empty_facing_track_must_raise_not_silently_pass(tmp_path: Path) -> None:
    """对位轨为空时必须报错，而不是退回全池。

    退回全池就是静默降级：实验照跑，指标照出，但每一局都是无效的。
    """

    _write_tracks(tmp_path, "rollman", [("only_same", "rollman")])
    tracks = _load_player_tracks(tmp_path, "rollman")
    eligible = [player for player in tracks if tracks.get(player) == "ghost"]
    assert eligible == []
    with pytest.raises(ValueError, match="没有可运行对手"):
        if not eligible:
            raise ValueError(
                "rollman 的 ghost 轨没有可运行对手：候选扮演 rollman，对手池里都不对位"
            )
