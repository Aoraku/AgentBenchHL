import random

from generals_impact_game.controller import run_ai
from generals_impact_game.gamedata import Direction
from generals_impact_game.gamestate import GameState


def example_ai(round: int, my_seat: int, state: GameState) -> list[list[int]]:
    """一个AI示例"""
    operations = []
    operations.append(
        [
            1,
            state.generals[my_seat].position[0],
            state.generals[my_seat].position[1],
            random.randint(1, 4),
            1,
        ]
    )
    return operations


run_ai(example_ai)
