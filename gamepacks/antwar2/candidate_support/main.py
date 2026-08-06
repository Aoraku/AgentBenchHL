from __future__ import annotations

from ai import AI
from common import BaseAgent, MatchSession
from protocol import ProtocolSession


def build_session(agent: BaseAgent) -> MatchSession:
    return ProtocolSession(agent)


def run_session(session: MatchSession) -> None:
    while True:
        if session.player == 0:
            session.perform_self_turn()
            if not session.receive_opponent_turn() or not session.sync_round():
                break
        else:
            if not session.receive_opponent_turn():
                break
            session.perform_self_turn()
            if not session.sync_round():
                break


def main() -> None:
    run_session(build_session(AI()))


if __name__ == "__main__":
    main()
