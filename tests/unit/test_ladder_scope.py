"""对手榜单口径（crawled / reference / measured / auto）的测试。

``crawled`` 是正名：manifest 里自带的 rank/elo 只是第一批爬取的副产品，
不是任何"官方榜"。``official`` 保留为弃用别名，下面也钉住它仍然可用。

动机：A 的 `manifest.tsv` 每个游戏只有 11–32 人带 rank+Elo，而池子有 250–750 人。
吸收外部交付的参考排名后覆盖面能到 202–534 人，但**那不是我们自己测的**，所以
口径必须显式、可追溯，且绝不能把某个来源的分数冒充成另一个来源。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from agentbench_hl.adapters.contract.pool import (
    PoolError,
    load_pool,
    public_leaderboard,
    ranked_ladder,
)


def write_tsv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


@pytest.fixture()
def pool_root(tmp_path: Path) -> Path:
    players = tmp_path / "games" / "g" / "players"
    write_tsv(
        players / "manifest.tsv",
        ["player_id", "dir", "username", "rank", "elo", "version", "submission_id"],
        [
            ["a", "pool/a", "ua", 1, 1800, 1, "s_a"],
            ["b", "pool/b", "ub", 2, 1700, 1, "s_b"],
            # c 在官方榜上没有名次（官方只排了前两名）
            ["c", "pool/c", "uc", "", "", 1, "s_c"],
        ],
    )
    for name in ("a", "b", "c"):
        (players / "pool" / name).mkdir(parents=True, exist_ok=True)
        (players / "pool" / name / "main.py").write_text("print()", encoding="utf-8")
    return tmp_path


def test_crawled_scope_leaves_the_manifest_values_untouched(pool_root: Path) -> None:
    players = load_pool(pool_root, "g", ladder_scope="crawled")

    assert [(item.player_id, item.rank, item.elo, item.rank_source) for item in players] == [
        ("a", 1, 1800.0, "crawled"),
        ("b", 2, 1700.0, "crawled"),
        ("c", None, None, "crawled"),
    ]
    assert [item.player_id for item in ranked_ladder(players)] == ["a", "b"]


def test_official_is_accepted_as_a_deprecated_alias(pool_root: Path) -> None:
    """既有配置里写的 ``official`` 不能因为改名就炸掉。"""

    assert load_pool(pool_root, "g", ladder_scope="official") == load_pool(
        pool_root, "g", ladder_scope="crawled"
    )


def test_default_scope_is_auto_not_the_narrow_crawled_one(pool_root: Path) -> None:
    """默认口径必须是 ``auto``，否则榜单会静默缩水。

    真实数据：antwar2 的 manifest 里只有 20 个人带爬取名次，而我们自己实测了 229 个。
    默认落到 ``crawled`` 就等于把对手课程从 229 人砍到 20 人 —— 而且不会报任何错，
    只会让排行榜看起来"缺斤少两"。
    """

    write_tsv(
        pool_root / "games" / "g" / "players" / "measured_ranking.tsv",
        ("player_id", "measured_rank", "measured_elo"),
        [("c", "1", "1500.0")],
    )

    players = {item.player_id: item for item in load_pool(pool_root, "g")}

    # c 在 manifest 里没有名次，只有实测榜覆盖到它 —— 默认口径必须能把它带进天梯。
    assert (players["c"].rank, players["c"].rank_source) == (1, "measured")
    assert "c" in {item.player_id for item in ranked_ladder(tuple(players.values()))}


def test_reference_scope_widens_the_ladder_and_labels_the_source(pool_root: Path) -> None:
    write_tsv(
        pool_root / "games/g/players/reference_ranking.tsv",
        ["ref_rank", "player_id", "ref_elo", "ref_metric"],
        [
            [1, "c", 1900, "pairwise_elo"],
            [2, "a", 1500, "pairwise_elo"],
            [3, "b", 1400, "pairwise_elo"],
        ],
    )

    players = load_pool(pool_root, "g", ladder_scope="reference")
    ladder = ranked_ladder(players)

    # c 原本没有官方名次，参考口径下成了榜首。
    assert [item.player_id for item in ladder] == ["c", "a", "b"]
    assert {item.rank_source for item in ladder} == {"reference"}
    assert ladder[0].elo == 1900.0
    # 公开给 Goal 的行必须带来源，避免它把参考分当成官方分。
    assert public_leaderboard(players)[0]["score_source"] == "reference"


def test_reference_without_elo_still_defines_a_usable_order(pool_root: Path) -> None:
    # AquaWar 的真实情况：交付只有加权分名次，没有 Elo。
    write_tsv(
        pool_root / "games/g/players/reference_ranking.tsv",
        ["ref_rank", "player_id", "ref_elo", "ref_metric"],
        [[1, "b", "", "weighted_score_rank"], [2, "c", "", "weighted_score_rank"]],
    )

    ladder = ranked_ladder(load_pool(pool_root, "g", ladder_scope="reference"))

    assert [item.player_id for item in ladder] == ["b", "c"]
    assert all(item.elo is None for item in ladder)
    # 需要 Elo 锚点的场合可以显式要求，此时该榜为空（诚实地"没有锚点"）。
    strict = ranked_ladder(
        load_pool(pool_root, "g", ladder_scope="reference"), require_score=True
    )
    assert strict == ()


def test_auto_scope_prefers_measured_then_reference_then_crawled(pool_root: Path) -> None:
    write_tsv(
        pool_root / "games/g/players/reference_ranking.tsv",
        ["ref_rank", "player_id", "ref_elo"],
        [[1, "b", 1500], [2, "c", 1400]],
    )
    write_tsv(
        pool_root / "games/g/players/measured_ranking.tsv",
        ["measured_rank", "player_id", "measured_elo"],
        [[1, "c", 1950]],
    )

    players = {item.player_id: item for item in load_pool(pool_root, "g", ladder_scope="auto")}

    assert (players["c"].elo, players["c"].rank_source) == (1950.0, "measured")  # 实测优先
    assert (players["b"].elo, players["b"].rank_source) == (1500.0, "reference")  # 无实测 → 参考
    # 两个 overlay 都没覆盖到 a → 回落到爬取时带的值
    assert (players["a"].elo, players["a"].rank_source) == (1800.0, "crawled")


def test_explicit_measured_scope_never_borrows_other_sources(pool_root: Path) -> None:
    write_tsv(
        pool_root / "games/g/players/reference_ranking.tsv",
        ["ref_rank", "player_id", "ref_elo"],
        [[1, "a", 1500]],
    )
    write_tsv(
        pool_root / "games/g/players/measured_ranking.tsv",
        ["measured_rank", "player_id", "measured_elo"],
        [[1, "c", 1950]],
    )

    players = {item.player_id: item for item in load_pool(pool_root, "g", ladder_scope="measured")}

    assert players["c"].elo == 1950.0
    # a 有官方分也有参考分，但**没有实测分** ⇒ 必须留空，不能冒充。
    assert players["a"].elo is None and players["a"].rank is None
    assert [item.player_id for item in ranked_ladder(tuple(players.values()))] == ["c"]


def test_unknown_scope_is_rejected(pool_root: Path) -> None:
    with pytest.raises(PoolError, match="unknown ladder scope"):
        load_pool(pool_root, "g", ladder_scope="whatever")


def test_missing_overlay_files_are_not_an_error(pool_root: Path) -> None:
    players = load_pool(pool_root, "g", ladder_scope="auto")

    assert [item.rank_source for item in players] == ["crawled"] * 3
