import math

from generals_impact_game.computation import compute_attack, compute_defence
from generals_impact_game.constant import *
from generals_impact_game.gamedata import CellType, Direction, Oilwell, WeaponType

# 本文件用于实现将军和军队的移动逻辑


def outrange(location: list[int, int]) -> bool:  # 判断越界
    return (
        location[0] < 0 or location[0] >= row or location[1] < 0 or location[1] >= col
    )


def calculate_new_pos(
    location: list[int, int], direction: Direction
) -> tuple[int, int]:  # 计算目标位置
    if direction == Direction.UP:
        newX = location[0] - 1
        newY = location[1]
    if direction == Direction.DOWN:
        newX = location[0] + 1
        newY = location[1]
    if direction == Direction.LEFT:
        newX = location[0]
        newY = location[1] - 1
    if direction == Direction.RIGHT:
        newX = location[0]
        newY = location[1] + 1
    if outrange([newX, newY]):
        return (-1, -1)
    return (newX, newY)


def army_move(
    location: list[int, int],
    gamestate,
    player: int,
    direction: Direction,
    num: int,
) -> bool:  # 军队移动
    x, y = location[0], location[1]
    if outrange(location):  # 越界
        return False
    if player != 0 and player != 1:  # 玩家参数非法
        return False
    if gamestate.board[x][y].player != player:  # 操作格子非法
        return False
    if gamestate.rest_move_step[player] == 0:
        return False
    if gamestate.board[x][y].army <= 1:
        return False

    if num <= 0:  # 移动数目非法
        return False
    if num >= gamestate.board[x][y].army - 1:  # 超过最多移动兵力
        num = gamestate.board[x][y].army - 1

    for sw in gamestate.active_super_weapon:  # 超级武器效果
        if (
            sw.position == [x, y]
            and sw.rest
            and sw.type == WeaponType.TRANSMISSION
            and sw.player == player
        ):  # 超时空传送眩晕
            return False
        if (
            abs(sw.position[0] - x) <= 1
            and abs(sw.position[1] - y) <= 1
            and sw.rest
            and sw.type == WeaponType.TIME_STOP
        ):  # 时间暂停效果
            return False

    newX, newY = calculate_new_pos(location, direction)
    if newX < 0:  # 越界
        return False
    if (
        gamestate.board[newX][newY].type == CellType.MOUNTAIN
        and gamestate.tech_level[player][1] == 0
    ):  # 不能爬山
        return False

    if gamestate.board[newX][newY].player == player:  # 目的地格子己方所有
        gamestate.board[newX][newY].army += num
        gamestate.board[x][y].army -= num

    elif (
        gamestate.board[newX][newY].player == 1 - player
        or gamestate.board[newX][newY].player == -1
    ):  # 攻击敌方或无主格子
        attack = compute_attack(gamestate.board[x][y], gamestate)
        defence = compute_defence(gamestate.board[newX][newY], gamestate)
        vs = num * attack - gamestate.board[newX][newY].army * defence
        if vs > 0:  # 攻下
            gamestate.board[newX][newY].player = player
            gamestate.board[newX][newY].army = math.ceil(vs / attack)
            gamestate.board[x][y].army -= num
            if gamestate.board[newX][newY].generals != None:  # 将军易主
                gamestate.board[newX][newY].generals.player = player
        elif vs < 0:  # 防住
            gamestate.board[newX][newY].army = math.ceil((-vs) / defence)
            gamestate.board[x][y].army -= num

        elif vs == 0:  # 中立
            if gamestate.board[newX][newY].generals == None:  # 将军不变中立
                gamestate.board[newX][newY].player = -1
            gamestate.board[newX][newY].army = 0
            gamestate.board[x][y].army -= num
    gamestate.rest_move_step[player] -= 1
    return True


def general_move(
    location: list[int, int],
    gamestate,
    player: int,
    destination: list[int, int],
) -> bool:
    able, steps = check_general_movement(location, gamestate, player, destination)
    if not able:
        return False
    x, y = location[0], location[1]
    newX, newY = destination[0], destination[1]
    gen = gamestate.board[x][y].generals
    gen.rest_move -= steps
    gamestate.board[newX][newY].generals = gen
    gamestate.board[newX][newY].generals.position = [newX, newY]
    gamestate.board[x][y].generals = None

    return True


def check_general_movement(
    location: list[int, int],
    gamestate,
    player: int,
    destination: list[int, int],
) -> (bool, int):  # 检查将军移动合法性
    x, y = location[0], location[1]
    if outrange(location):  # 越界
        return False, -1
    if player != 0 and player != 1:  # 玩家非法
        return False, -1
    if gamestate.board[destination[0]][destination[1]].generals != None:
        return False, -1
    if (
        gamestate.board[x][y].player != player or gamestate.board[x][y].generals == None
    ):  # 起始格子非法
        return False, -1
    for sw in gamestate.active_super_weapon:  # 超级武器效果
        if (
            sw.position == [x, y]
            and sw.rest
            and sw.type == WeaponType.TRANSMISSION
            and sw.player == player
        ):  # 超时空传送眩晕
            return False, -1
        if (
            abs(sw.position[0] - x) <= 1
            and abs(sw.position[1] - y) <= 1
            and sw.rest
            and sw.type == WeaponType.TIME_STOP
        ):  # 时间暂停效果
            return False, -1

    newX, newY = destination[0], destination[1]

    # bfs检查可移动性
    op, cl = -1, 0
    queue = []  # bfs队列
    steps = []  # 移动步数
    check = [[False for i in range(row)] for j in range(col)]  # 是否入队
    queue.append((x, y))
    steps.append(0)
    check[x][y] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while op < cl:
        op += 1
        if steps[op] > gamestate.board[x][y].generals.rest_move:  # 步数超限
            break
        if queue[op] == (newX, newY):  # 到达目的地
            return True, steps[op]
        p, q = queue[op][0], queue[op][1]
        for direction in directions:
            newP, newQ = p + direction[0], q + direction[1]
            if outrange([newP, newQ]):  # 越界
                continue
            if check[newP][newQ]:  # 已入队
                continue
            if (
                gamestate.board[newP][newQ].type == CellType.MOUNTAIN
                and gamestate.tech_level[player][1] == 0
            ):  # 无法爬山
                continue
            if (
                gamestate.board[newP][newQ].player != player
                or gamestate.board[newP][newQ].generals != None
            ):  # 目的地格子非法
                continue
            queue.append((newP, newQ))  # 入队
            cl += 1
            steps.append(steps[op] + 1)
            check[newP][newQ] = True
    # bfs结束，没到达目的地
    return False, -1
