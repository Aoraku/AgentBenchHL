"""实测评分导出成对手选择口径（``measured_ranking.tsv``）的测试。

这条链路是"我们自己跑出来的 Elo 能不能真的当对手榜用"的唯一接缝：
`ladder_eval` 写 TSV → `contract/pool.apply_ladder_scope` 读 TSV → 对手策略排序。
两边字段名一旦漂移，实验会静默退回官方口径（只有十几个对手），而且不报错。
"""

from __future__ import annotations

import csv
from pathlib import Path

from agentbench_hl.adapters.contract.pool import MEASURED_FILENAME
from agentbench_hl.application.ladder_eval import RANKING_FILENAME, write_measured_ranking


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def test_filename_and_columns_match_the_scope_layer_contract(tmp_path: Path) -> None:
    # 两个模块必须指向同一个文件名，否则实测口径永远读不到数据。
    assert RANKING_FILENAME == MEASURED_FILENAME == "measured_ranking.tsv"

    document = {
        "ratings": [
            {"player_id": "a", "measured_elo": 1500.0, "matches": 12, "winrate": 0.5, "note": None},
            {"player_id": "b", "measured_elo": 1720.4, "matches": 12, "winrate": 0.8, "note": None},
        ]
    }

    target = write_measured_ranking(tmp_path, "g", document)
    rows = read_tsv(target)

    assert target == tmp_path / "games" / "g" / "players" / "measured_ranking.tsv"
    assert [row["player_id"] for row in rows] == ["b", "a"]  # 按实测 Elo 降序
    assert [row["measured_rank"] for row in rows] == ["1", "2"]
    assert rows[0]["measured_elo"] == "1720.4"
    assert set(rows[0]) >= {"measured_rank", "player_id", "measured_elo", "matches", "winrate"}


def test_unrated_players_are_excluded_rather_than_given_a_fake_rank(tmp_path: Path) -> None:
    document = {
        "ratings": [
            {"player_id": "a", "measured_elo": 1500.0, "matches": 8, "winrate": 0.5},
            # 饱和（全胜）与不连通的选手没有可比强度：必须缺席，不能硬排名次。
            {"player_id": "saturated", "measured_elo": None, "note": "saturated"},
            {"player_id": "isolated", "measured_elo": None, "note": "disconnected"},
        ]
    }

    rows = read_tsv(write_measured_ranking(tmp_path, "g", document))

    assert [row["player_id"] for row in rows] == ["a"]


def test_empty_ratings_still_produce_a_header_only_file(tmp_path: Path) -> None:
    rows = read_tsv(write_measured_ranking(tmp_path, "g", {"ratings": []}))

    assert rows == []  # 只有表头：口径层会安静地当作"没有实测数据"
