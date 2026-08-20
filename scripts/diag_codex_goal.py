#!/usr/bin/env python3
"""诊断：codex goal 线程到底说了什么、为什么 blocked。

``goal-led`` 只能看到"没交 action.json / 状态 blocked"这种结果性错误，而真正的原因
（模型回了什么、有没有调工具、被什么挡住）在 codex 自己的 sqlite 里。这个脚本把这些
读出来，避免每次都靠猜。

用法::

    python scripts/diag_codex_goal.py <run_root>
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def _tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _dump(path: Path, *, limit: int) -> None:
    if not path.is_file():
        print(f"-- 缺少 {path.name}")
        return
    print(f"\n===== {path.name}")
    # 只读打开：run 可能还在跑，不能干扰它。
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = _tables(connection)
        print("   表：", names)
        for table in names:
            try:
                count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error as error:
                print(f"   [{table}] 读取失败：{error}")
                continue
            print(f"   [{table}] {count} 行")
            if not count:
                continue
            try:
                rows = connection.execute(
                    f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT {limit}"
                ).fetchall()
            except sqlite3.Error as error:
                print(f"      取样失败：{error}")
                continue
            probe = connection.execute(f"SELECT * FROM {table} LIMIT 0")
            columns = [item[0] for item in probe.description]
            for row in reversed(rows):
                record = {}
                for key, value in zip(columns, row, strict=False):
                    text = value if isinstance(value, (str, int, float, type(None))) else "<blob>"
                    if isinstance(text, str) and len(text) > 600:
                        text = text[:600] + f"…(+{len(value) - 600})"
                    record[key] = text
                print("      ", json.dumps(record, ensure_ascii=False)[:1400])
    finally:
        connection.close()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    home = Path(argv[0]).resolve() / "codex-home"
    if not home.is_dir():
        print(f"没有 {home}")
        return 1
    limit = int(argv[1]) if len(argv) > 1 else 3
    for name in sorted(item.name for item in home.glob("*.sqlite")):
        if name.startswith(("goals", "thread_history", "queue", "logs")):
            _dump(home / name, limit=limit)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
