from __future__ import annotations

from typing import Iterator

from SDK.utils.constants import MAP_PROPERTY, MAP_SIZE, OFFSET, Terrain


def is_valid_pos(x: int, y: int) -> bool:
    return 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and MAP_PROPERTY[x][y] != Terrain.VOID


def is_path(x: int, y: int) -> bool:
    return 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and MAP_PROPERTY[x][y] == Terrain.PATH


def is_highland(player: int, x: int, y: int) -> bool:
    target = Terrain.PLAYER0_HIGHLAND if player == 0 else Terrain.PLAYER1_HIGHLAND
    return 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and MAP_PROPERTY[x][y] == target


def hex_distance(x0: int, y0: int, x1: int, y1: int) -> int:
    dy = abs(y0 - y1)
    if dy % 2:
        if x0 > x1:
            dx = max(0, abs(x0 - x1) - dy // 2 - (y0 % 2))
        else:
            dx = max(0, abs(x0 - x1) - dy // 2 - (1 - (y0 % 2)))
    else:
        dx = max(0, abs(x0 - x1) - dy // 2)
    return dx + dy


def neighbors(x: int, y: int) -> Iterator[tuple[int, int, int]]:
    for direction, (dx, dy) in enumerate(OFFSET[y % 2]):
        nx = x + dx
        ny = y + dy
        yield direction, nx, ny


def direction_between(x0: int, y0: int, x1: int, y1: int) -> int:
    for index, (dx, dy) in enumerate(OFFSET[y0 % 2]):
        if x0 + dx == x1 and y0 + dy == y1:
            return index
    return -1
