"""Length-prefixed AntWar2 match transport and role-correct results."""

from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

from agentbench_hl.ports.arena import MatchCase, MatchResult, ProcessSpec


class AntWarMatchError(RuntimeError):
    """The native run did not produce a scientifically complete match."""


def _load_replay(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AntWarMatchError(f"cannot read replay {path}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise AntWarMatchError("replay must be a non-empty JSON array")
    if not all(isinstance(record, dict) for record in value):
        raise AntWarMatchError("replay contains a non-object round")
    return value


def match_result_from_replay(
    replay_path: str | Path,
    *,
    candidate_id: str,
    opponent_id: str,
    role: str,
    seed: int,
    trace_path: str | Path | None = None,
    events_path: str | Path | None = None,
    process_returncodes: tuple[int, int, int] | None = None,
) -> MatchResult:
    """Map replay-format ``camps`` to canonical role-aware base HP."""

    case = MatchCase(candidate_id, opponent_id, role, seed)
    source = Path(replay_path).resolve()
    replay = _load_replay(source)
    terminal = replay[-1].get("round_state")
    if not isinstance(terminal, dict):
        raise AntWarMatchError("replay has no terminal round_state")
    winner = terminal.get("winner")
    if winner not in {0, 1}:
        raise AntWarMatchError("replay has no valid terminal winner")
    replay_camps = terminal.get("camps")
    if (
        not isinstance(replay_camps, list)
        or len(replay_camps) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in replay_camps
        )
    ):
        raise AntWarMatchError("replay has invalid terminal base-HP mapping")
    base_hp = (float(replay_camps[0]), float(replay_camps[1]))
    candidate_player = 0 if role == "P0" else 1
    opponent_player = 1 - candidate_player
    won = winner == candidate_player
    return MatchResult(
        case=case,
        status="complete",
        result="win" if won else "loss",
        points=1.0 if won else 0.0,
        score_margin=base_hp[candidate_player] - base_hp[opponent_player],
        terminal_base_hp=base_hp,
        rounds=len(replay),
        replay_path=source,
        trace_path=None if trace_path is None else Path(trace_path).resolve(),
        events_path=None if events_path is None else Path(events_path).resolve(),
        process_returncodes=process_returncodes,
    )


def write_public_trace(replay_path: Path, trace_path: Path) -> None:
    """Keep accepted operations and public states; omit all private process data."""

    replay = _load_replay(replay_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as stream:
        sequence = 0
        for round_index, record in enumerate(replay):
            for player in (0, 1):
                operations = record.get(f"op{player}", [])
                if not isinstance(operations, list):
                    raise AntWarMatchError(f"round {round_index} P{player} operations are invalid")
                stream.write(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "round": round_index,
                            "kind": "accepted_operations",
                            "player": player,
                            "operations": operations,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sequence += 1
            public_state = record.get("round_state")
            if not isinstance(public_state, dict):
                raise AntWarMatchError(f"round {round_index} has no public state")
            stream.write(
                json.dumps(
                    {
                        "sequence": sequence,
                        "round": round_index,
                        "kind": "public_state",
                        "public_state": public_state,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sequence += 1


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected EOF after {count - remaining}/{count} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _packet(value: object) -> bytes:
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def _write(stream: BinaryIO, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(stream.fileno(), view)
        if written < 1:
            raise BrokenPipeError("process pipe accepted zero bytes")
        view = view[written:]


def _reader(stream: BinaryIO, output: queue.Queue, *, game: bool) -> None:
    try:
        while True:
            size = struct.unpack(">I", _read_exact(stream, 4))[0]
            if size > 64 * 1024 * 1024:
                raise ValueError(f"frame exceeds 64 MiB: {size}")
            object_id = struct.unpack(">i", _read_exact(stream, 4))[0] if game else None
            output.put(("frame", (object_id, _read_exact(stream, size))))
    except EOFError:
        output.put(("eof", None))
    except BaseException as exc:
        output.put(("error", exc))


def _drain_stderr(stream: BinaryIO, destination: Path, tail: bytearray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            try:
                chunk = stream.read(65536)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            handle.write(chunk)
            handle.flush()
            tail.extend(chunk)
            if len(tail) > 8192:
                del tail[:-8192]


def _safe_environment(extra: Mapping[str, str]) -> dict[str, str]:
    allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TMPDIR")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    forbidden_fragments = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")
    for key, value in extra.items():
        if any(fragment in key.upper() for fragment in forbidden_fragments):
            raise AntWarMatchError(f"credential-like environment variable rejected: {key}")
        environment[str(key)] = str(value)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _start(spec: ProcessSpec) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        spec.argv,
        cwd=spec.cwd,
        env=_safe_environment(spec.env),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.terminate()
        raise AntWarMatchError("failed to open process pipes")
    return process


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            # A sandboxed child can remain unreapable after SIGKILL.  Never
            # let cleanup block the evaluation worker indefinitely.
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _next_frame(
    frames: queue.Queue,
    *,
    deadline: float,
    label: str,
    processes: tuple[subprocess.Popen[bytes], ...],
) -> tuple[int | None, bytes]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{label} timed out")
    try:
        kind, payload = frames.get(timeout=remaining)
    except queue.Empty as exc:
        codes = [process.poll() for process in processes]
        raise TimeoutError(f"{label} timed out; returncodes={codes}") from exc
    if kind == "error":
        raise AntWarMatchError(f"{label} reader failed: {payload}") from payload
    if kind == "eof":
        codes = [process.poll() for process in processes]
        raise AntWarMatchError(f"{label} closed early; returncodes={codes}")
    return payload


def run_native_match(
    *,
    game: ProcessSpec,
    candidate_process: ProcessSpec,
    opponent_process: ProcessSpec,
    case: MatchCase,
    replay_path: Path,
    trace_path: Path,
    events_path: Path,
    timeout_s: float,
) -> MatchResult:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_player = 0 if case.role == "P0" else 1
    specs = (
        (candidate_process, opponent_process)
        if candidate_player == 0
        else (opponent_process, candidate_process)
    )
    game_process = _start(game)
    players = (_start(specs[0]), _start(specs[1]))
    processes = (game_process, *players)
    game_frames: queue.Queue = queue.Queue()
    player_frames = (queue.Queue(), queue.Queue())
    stderr_tails = (bytearray(), bytearray(), bytearray())
    readers = [
        threading.Thread(
            target=_reader,
            args=(game_process.stdout, game_frames),
            kwargs={"game": True},
            daemon=True,
        ),
        *(
            threading.Thread(
                target=_reader,
                args=(players[index].stdout, player_frames[index]),
                kwargs={"game": False},
                daemon=True,
            )
            for index in (0, 1)
        ),
    ]
    stderr_readers = [
        threading.Thread(
            target=_drain_stderr,
            args=(
                processes[index].stderr,
                events_path.with_suffix(f".{('game', 'p0', 'p1')[index]}.stderr.log"),
                stderr_tails[index],
            ),
            daemon=True,
        )
        for index in range(3)
    ]
    for thread in (*readers, *stderr_readers):
        thread.start()
    deadline = time.monotonic() + timeout_s
    ended = False
    try:
        with events_path.open("w", encoding="utf-8") as event_log:

            def record_event(kind: str, **details: object) -> None:
                event_log.write(json.dumps({"kind": kind, **details}) + "\n")
                event_log.flush()

            _write(
                game_process.stdin,
                _packet(
                    {
                        "player_list": [1, 1],
                        "player_num": 2,
                        "config": {"random_seed": case.seed},
                        "replay": str(replay_path.resolve()),
                    }
                ),
            )
            record_event("send_init", seed=case.seed)
            while not ended:
                # ``_next_frame`` uses the deadline as a queue wait timeout,
                # but a chatty game can keep returning frames immediately.
                # Check the absolute match deadline on every iteration so a
                # non-terminating native game cannot keep an evaluation alive
                # indefinitely merely by producing transport traffic.
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"match timed out after {timeout_s:.3f}s; returncodes="
                        f"{[process.poll() for process in processes]}"
                    )
                object_id, payload = _next_frame(
                    game_frames,
                    deadline=deadline,
                    label="game",
                    processes=processes,
                )
                record_event("game_packet", object=object_id, size=len(payload))
                if object_id in {0, 1}:
                    _write(players[object_id].stdin, payload)
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise AntWarMatchError("game emitted invalid JSON") from exc
                if not isinstance(message, dict):
                    raise AntWarMatchError("game message is not an object")
                public_players = message.get("player", [])
                public_content = message.get("content", [])
                if public_players or public_content:
                    if (
                        not isinstance(public_players, list)
                        or not isinstance(public_content, list)
                        or len(public_players) != len(public_content)
                    ):
                        raise AntWarMatchError("game broadcast has invalid shape")
                    for raw_player, content in zip(public_players, public_content, strict=True):
                        player = int(raw_player)
                        _write(players[player].stdin, str(content).encode("utf-8"))
                for raw_player in message.get("listen", []):
                    player = int(raw_player)
                    _unused, body = _next_frame(
                        player_frames[player],
                        deadline=deadline,
                        label=f"player {player}",
                        processes=processes,
                    )
                    framed = struct.pack(">I", len(body)) + body
                    _write(
                        game_process.stdin,
                        _packet({"player": player, "content": framed.decode("latin1"), "time": 0}),
                    )
                if "end_state" in message:
                    record_event("end", end_state=message["end_state"])
                    ended = True
        game_process.wait(timeout=max(deadline - time.monotonic(), 0.1))
        for process in players:
            process.stdin.close()
            process.wait(timeout=5)
        returncodes = (
            int(game_process.returncode),
            int(players[0].returncode),
            int(players[1].returncode),
        )
        if any(code != 0 for code in returncodes):
            tails = [tail.decode("utf-8", errors="replace") for tail in stderr_tails]
            raise AntWarMatchError(f"non-zero process return code: {returncodes}; stderr={tails}")
        if not ended:
            raise AntWarMatchError("game exited without end_state")
        write_public_trace(replay_path, trace_path)
        return match_result_from_replay(
            replay_path,
            candidate_id=case.candidate_id,
            opponent_id=case.opponent_id,
            role=case.role,
            seed=case.seed,
            trace_path=trace_path,
            events_path=events_path,
            process_returncodes=returncodes,
        )
    finally:
        for process in processes:
            _stop(process)


Runner = Callable[..., MatchResult]


class AntWarArena:
    def __init__(
        self,
        *,
        game: ProcessSpec,
        opponents: Mapping[str, ProcessSpec],
        artifact_root: Path,
        timeout_s: float = 120.0,
        candidate_command_prefix: tuple[str, ...] = (),
        runner: Runner = run_native_match,
    ) -> None:
        self.game = game
        self.opponents = dict(opponents)
        self.artifact_root = Path(artifact_root)
        self.timeout_s = float(timeout_s)
        self.candidate_command_prefix = tuple(candidate_command_prefix)
        self.runner = runner

    def run_case(self, case: MatchCase, candidate_root: Path) -> MatchResult:
        try:
            opponent = self.opponents[case.opponent_id]
        except KeyError as exc:
            raise ValueError(f"unknown or unrunnable opponent: {case.opponent_id}") from exc
        root = (
            self.artifact_root
            / case.candidate_id
            / case.opponent_id
            / case.role.lower()
            / f"seed-{case.seed}"
        )
        try:
            return self.runner(
                game=self.game,
                candidate_process=ProcessSpec(
                    (*self.candidate_command_prefix, sys.executable, "main.py"),
                    candidate_root,
                ),
                opponent_process=opponent,
                case=case,
                replay_path=root / "replay.json",
                trace_path=root / "public-trace.jsonl",
                events_path=root / "transport-events.jsonl",
                timeout_s=self.timeout_s,
            )
        except (AntWarMatchError, TimeoutError, OSError, subprocess.TimeoutExpired) as exc:
            return MatchResult(
                case=case,
                status="incomplete",
                result=None,
                points=None,
                score_margin=None,
                terminal_base_hp=None,
                rounds=None,
                error=str(exc),
            )
