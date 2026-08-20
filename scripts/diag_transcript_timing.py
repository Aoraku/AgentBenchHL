#!/usr/bin/env python3
"""诊断：把录制流水按时间戳拆开，看时间花在哪一步。

用法::

    python scripts/diag_transcript_timing.py <transcript.jsonl> [...]

输出每条流水的记录构成、总时长、最慢的几个间隔，以及"时间按转移类型的归属"。
``in -> out`` 的耗时是**选手思考**，``out -> in`` 是**判题器推进**，
两者都不大却总时长很长，说明卡在别处（比如录制本身）。
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def rows_of(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def report(path: Path) -> None:
    rows = rows_of(path)
    if not rows:
        print(f"== {path}: 空文件")
        return
    span = float(rows[-1].get("t") or 0.0)
    dirs = collections.Counter(str(row.get("dir")) for row in rows)
    print(f"== {path}")
    print(f"   {len(rows)} 条记录，跨度 {span:.1f}s，构成 {dict(dirs)}")
    gaps = [
        (
            float(later.get("t") or 0.0) - float(earlier.get("t") or 0.0),
            str(earlier.get("dir")),
            str(later.get("dir")),
        )
        for earlier, later in zip(rows, rows[1:], strict=False)
    ]
    if not gaps:
        return
    by_transition: collections.Counter[str] = collections.Counter()
    for gap, before, after in gaps:
        by_transition[f"{before}->{after}"] += gap
    slowest = [(round(g, 3), f"{a}->{b}") for g, a, b in sorted(gaps, reverse=True)[:5]]
    print("   最慢间隔：", slowest)
    print(
        "   时间归属：",
        [(key, round(value, 1)) for key, value in by_transition.most_common(5)],
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for item in argv:
        report(Path(item))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
