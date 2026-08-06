"""Process-scoped lease preventing concurrent mutation of one scientific run."""

from __future__ import annotations

import fcntl
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType


class RunLeaseBusy(RuntimeError):
    pass


class RunLease:
    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root).resolve()
        self.path = self.run_root / "run.lock"
        self._descriptor: int | None = None

    def acquire(self) -> RunLease:
        if self._descriptor is not None:
            return self
        self.run_root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RunLeaseBusy(f"scientific run is already active: {self.run_root}") from exc
        metadata = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquired_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, metadata + b"\n")
        os.fsync(descriptor)
        self._descriptor = descriptor
        return self

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> RunLease:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
