import random

from generals_impact_game.constant import *
from generals_impact_game.gamedata import CellType, MainGenerals, Oilwell, SubGenerals
from generals_impact_game.gamestate import GameState


def count_surroundings(map, i, j, landtype):
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            ni = i + dx
            nj = j + dy
            if ni >= 0 and ni < row and nj >= 0 and nj < col:
                if map[ni][nj].type in landtype:
                    count += 1
    return count


def update_map(gamestate: GameState):
    for row in gamestate.board:
        for cell in row:
            if cell.type == 2:
                sur_num = count_surroundings(
                    gamestate.board, cell.position[0], cell.position[1], [2]
                )
                if sur_num < 3:
                    cell.type = 0

    for row in gamestate.board:
        for cell in row:
            if cell.type in [0, 1]:
                sur_num = count_surroundings(
                    gamestate.board, cell.position[0], cell.position[1], [0, 1]
                )
                if sur_num < 4:
                    cell.type = 2


dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]


def is_valid(board, x, y):
    m = len(board)
    n = len(board[0])
    return 0 <= x < m and 0 <= y < n and board[x][y].type == CellType(0)


# 定义一个函数，使用 DFS 来遍历棋盘上的连通区域
def dfs(board, x, y, visited):
    # 将当前位置标记为已访问
    visited[x][y] = True
    # 遍历四个方向
    for i in range(4):
        # 计算下一个位置的坐标
        nx = x + dx[i]
        ny = y + dy[i]
        # 如果下一个位置是有效的，且没有被访问过，继续 DFS
        if is_valid(board, nx, ny) and not visited[nx][ny]:
            dfs(board, nx, ny, visited)


# 定义一个函数，判断两个位置是否联通
def is_connected(board, p1, p2):
    # 获取棋盘的行数和列数
    m = len(board)
    n = len(board[0])
    # 获取两个位置的坐标
    x1, y1 = p1
    x2, y2 = p2
    # 判断两个位置是否有效
    if not is_valid(board, x1, y1) or not is_valid(board, x2, y2):
        return False
    # 创建一个二维数组，记录每个位置是否被访问过
    visited = [[False] * n for _ in range(m)]
    # 从第一个位置开始 DFS
    dfs(board, x1, y1, visited)
    # 返回第二个位置是否被访问过
    return visited[x2][y2]


def generate_general_points():
    x1 = random.randint(0, 14)
    y1 = random.randint(0, 14)
    x2 = random.randint(0, 14)
    y2 = random.randint(0, 14)
    return (x1, y1), (x2, y2)


def manhattan_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


def connect_points(board, start, end):
    x1, y1 = start
    x2, y2 = end

    # 随机选择横向或纵向移动
    while x1 != x2 or y1 != y2:
        if x1 != x2 and y1 != y2:
            # 随机选择横向或纵向移动
            direction = random.choice(["x", "y"])
        elif x1 != x2:
            direction = "x"
        else:
            direction = "y"

        if direction == "x":
            x_step = 1 if x1 < x2 else -1
            x1 += x_step
            board[x1][y1].type = CellType(0)
        else:
            y_step = 1 if y1 < y2 else -1
            y1 += y_step
            board[x1][y1].type = CellType(0)

    # 将目标位置设为 0
    board[x2][y2].type = CellType(0)


def init_generals(gamestate: GameState):
    # init random position
    while True:
        point1, point2 = generate_general_points()
        distance = manhattan_distance(point1, point2)
        if distance > 18:
            break
    mainpos = [point1, point2]
    # generate main generals
    for player in range(2):
        gen = MainGenerals(player=player, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        x = mainpos[player][0]
        y = mainpos[player][1]
        gen.position[0] = x
        gen.position[1] = y
        gamestate.generals.append(gen)
        gamestate.board[x][y].generals = gen
        gamestate.board[x][y].type = CellType(0)
        gamestate.board[x][y].player = player
    if not is_connected(gamestate.board, point1, point2):
        connect_points(gamestate.board, point1, point2)
    positions = []
    positions_mountain = []
    for i in range(row):
        for j in range(col):
            if (
                gamestate.board[i][j].type == CellType(0)
                and not gamestate.board[i][j].generals
            ):
                positions.append([i, j])
            if gamestate.board[i][j].type == CellType(2):
                positions_mountain.append([i, j])
    random.shuffle(positions)
    random.shuffle(positions_mountain)
    # generate sub generals
    for player in range(subgen_num):
        gen = SubGenerals(player=-1, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].army = random.randint(10, 20)

    # generate farmer
    for i in range(oilwell_num - 3):
        gen = Oilwell(player=-1, produce_level=1, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].army = random.randint(3, 5)

    for i in range(3):
        gen = Oilwell(player=-1, produce_level=1, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions_mountain.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].army = random.randint(3, 5)


def generate_map():
    gamestate = GameState()
    update_map(gamestate)
    init_generals(gamestate)
    gamestate.coin = [40, 40]
    return gamestate
