from __future__ import annotations

import struct
import sys
from collections.abc import Iterable

from common import BaseAgent, MatchSession
from SDK.backend.engine import PublicRoundState
from SDK.backend.model import Operation
from SDK.backend.runtime import MatchRuntime
from SDK.utils.constants import OperationType


class ProtocolIO:
    def __init__(self, stdin=None, stdout=None) -> None:
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer

    def recv_line(self) -> str | None:
        raw = self.stdin.readline()
        return None if not raw else raw.decode("utf-8", errors="replace").rstrip("\n")

    def send_packet(self, payload: str) -> None:
        data = (payload if payload.endswith("\n") else payload + "\n").encode()
        self.stdout.write(struct.pack(">I", len(data)))
        self.stdout.write(data)
        self.stdout.flush()

    def recv_init(self) -> tuple[int, int]:
        line = self.recv_line()
        if line is None:
            raise RuntimeError("missing init line")
        player, seed = map(int, line.split())
        return player, seed

    def recv_operations(self) -> list[Operation]:
        line = self.recv_line()
        if line is None:
            raise RuntimeError("missing operation count")
        operations: list[Operation] = []
        for _ in range(int(line)):
            parts = [int(item) for item in (self.recv_line() or "").split()]
            if not parts:
                raise RuntimeError("missing operation")
            operation_type = OperationType(parts[0])
            operations.append(Operation(operation_type, *parts[1:]))
        return operations

    def recv_round_state(self) -> PublicRoundState | None:
        line = self.recv_line()
        if line is None:
            return None
        round_index = int(line)
        towers = [
            tuple(map(int, (self.recv_line() or "").split()))
            for _ in range(int(self.recv_line() or "0"))
        ]
        ants = [
            tuple(map(int, (self.recv_line() or "").split()))
            for _ in range(int(self.recv_line() or "0"))
        ]
        coins = tuple(map(int, (self.recv_line() or "0 0").split()[:2]))
        camp_fields = tuple(map(int, (self.recv_line() or "0 0").split()))
        cooldowns = [
            tuple(map(int, (self.recv_line() or "").split()))
            for _ in range(int(self.recv_line() or "0"))
        ]
        effects = [
            tuple(map(int, (self.recv_line() or "").split()))
            for _ in range(int(self.recv_line() or "0"))
        ]
        return PublicRoundState(
            round_index=round_index,
            towers=towers,
            ants=ants,
            coins=coins,
            camps_hp=camp_fields[:2],
            speed_lv=camp_fields[2:4] if len(camp_fields) >= 4 else None,
            anthp_lv=camp_fields[4:6] if len(camp_fields) >= 6 else None,
            weapon_cooldowns=tuple(cooldowns),
            active_effects=effects,
        )

    def send_operations(self, operations: Iterable[Operation]) -> None:
        items = list(operations)
        lines = [str(len(items))]
        lines.extend(
            " ".join(str(token) for token in operation.to_protocol_tokens())
            for operation in items
        )
        self.send_packet("\n".join(lines) + "\n")


class ProtocolSession(MatchSession):
    def __init__(self, agent: BaseAgent, io: ProtocolIO | None = None) -> None:
        self.io = io or ProtocolIO()
        player, seed = self.io.recv_init()
        self.runtime = MatchRuntime.create(player=player, seed=seed, prefer_native=False)
        self.agent = agent
        self.agent.on_match_start(player, seed)

    @property
    def player(self) -> int:
        return self.runtime.player

    def perform_self_turn(self) -> None:
        proposed = self.agent.choose_operations(self.runtime.state, self.player)
        accepted: list[Operation] = []
        for operation in proposed:
            if self.runtime.state.can_apply_operation(self.player, operation, accepted):
                accepted.append(operation)
        self.runtime.apply_self_operations(accepted)
        self.agent.on_self_operations(accepted)
        self.io.send_operations(accepted)

    def receive_opponent_turn(self) -> bool:
        try:
            operations = self.io.recv_operations()
        except (RuntimeError, ValueError):
            return False
        self.runtime.apply_opponent_operations(operations)
        self.agent.on_opponent_operations(operations)
        return True

    def sync_round(self) -> bool:
        state = self.io.recv_round_state()
        if state is None:
            return False
        self.runtime.finish_round(state)
        self.agent.on_round_state(state)
        return True
