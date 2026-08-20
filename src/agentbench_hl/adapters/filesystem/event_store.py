"""Durable append-only JSONL event storage."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentbench_hl.domain.events import FinalizedEvent


class JsonlEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # 幂等索引常驻内存：每次 append 都重读整个 JSONL 会让指标收集变成 O(n²)。
        # 但**只加载一次**会造成跨进程陈读（另一个进程/另一个实例 append 之后，
        # 本实例的 read_all 仍返回旧快照）。因此这里改为**增量加载**：
        # 记住已消费的字节偏移，每次 read_all 只解析新增部分——既不重复解析，
        # 也不会漏掉别人写入的事件。
        self._offset = 0
        self._line = 0
        self._events: list[FinalizedEvent] = []
        self._by_key: dict[str, FinalizedEvent] = {}

    def _load(self) -> None:
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        if size == self._offset:
            return
        if size < self._offset:
            # 文件被截断/重建：索引失效，整体重建（不应发生，但不能静默错乱）。
            self._offset = 0
            self._line = 0
            self._events = []
            self._by_key = {}
        with self.path.open("rb") as stream:
            stream.seek(self._offset)
            chunk = stream.read()
        if not chunk:
            return
        # 逐行推进偏移：解析失败（文件损坏/写坏）时按既有约定直接报错，
        # 不静默跳过——事件流是所有指标的唯一事实源，宁可炸也不能悄悄丢。
        start_line = self._line
        consumed = 0
        lines = 0

        def commit() -> None:
            self._offset += consumed
            self._line += lines

        for offset_in_chunk, line in enumerate(chunk.decode("utf-8").splitlines(keepends=True), 1):
            line_number = start_line + offset_in_chunk
            payload_bytes = len(line.encode("utf-8"))
            if not line.strip():
                consumed += payload_bytes
                lines += 1
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("event line is not an object")
                event = FinalizedEvent.from_dict(value)
            except (json.JSONDecodeError, ValueError) as exc:
                commit()
                raise ValueError(f"invalid finalized event at line {line_number}: {exc}") from exc
            previous = self._by_key.get(event.idempotency_key)
            if previous is not None:
                if previous.event_type == event.event_type and dict(previous.payload) == dict(
                    event.payload
                ):
                    consumed += payload_bytes
                    lines += 1
                    continue
                commit()
                raise ValueError(
                    f"duplicate idempotency key at line {line_number}: {event.idempotency_key}"
                )
            self._by_key[event.idempotency_key] = event
            self._events.append(event)
            consumed += payload_bytes
            lines += 1
        commit()

    def read_all(self) -> tuple[FinalizedEvent, ...]:
        self._load()
        return tuple(self._events)

    def append(self, event: FinalizedEvent) -> bool:
        return self.append_many((event,))[0]

    def append_many(self, events: tuple[FinalizedEvent, ...]) -> tuple[bool, ...]:
        if not events:
            return ()
        self._load()
        existing = self._by_key
        pending: list[FinalizedEvent] = []
        outcomes: list[bool] = []
        staged: dict[str, FinalizedEvent] = {}
        for event in events:
            previous = existing.get(event.idempotency_key) or staged.get(event.idempotency_key)
            if previous is not None:
                if previous.event_type != event.event_type or dict(previous.payload) != dict(
                    event.payload
                ):
                    raise ValueError(f"conflicting idempotency key: {event.idempotency_key}")
                outcomes.append(False)
                continue
            staged[event.idempotency_key] = event
            pending.append(event)
            outcomes.append(True)
        if not pending:
            return tuple(outcomes)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = "".join(
            (
                json.dumps(
                    event.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            for event in pending
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Commit the in-memory index only after the complete batch is durable;
        # a validation error above therefore leaves the store unchanged.
        existing.update(staged)
        self._events.extend(pending)
        return tuple(outcomes)
