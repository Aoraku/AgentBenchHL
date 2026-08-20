import math

from generals_impact_game.constant import *
from generals_impact_game.gamedata import SkillType
from generals_impact_game.movement import *

# 本文件定义了将军战法


# 用于处理军队突袭
def army_rush(
    location: list[int, int],
    gamestate,
    player: int,
    destination: list[int, int],
) -> bool:
    x, y = location[0], location[1]
    new_x, new_y = destination[0], destination[1]
    num = gamestate.board[x][y].army - 1
    if gamestate.board[new_x][new_y].player == -1:
        gamestate.board[new_x][new_y].army += num
        gamestate.board[x][y].army -= num
        gamestate.board[new_x][new_y].player = player
    elif gamestate.board[new_x][new_y].player == player:
        gamestate.board[x][y].army -= num
        gamestate.board[new_x][new_y].army += num
    elif gamestate.board[new_x][new_y].player == 1 - player:
        attack = compute_attack(gamestate.board[x][y], gamestate)
        defence = compute_defence(gamestate.board[new_x][new_y], gamestate)
        vs = num * attack - gamestate.board[new_x][new_y].army * defence
        assert vs > 0
        gamestate.board[new_x][new_y].player = player
        gamestate.board[new_x][new_y].army = math.ceil(vs / attack)
        gamestate.board[x][y].army -= num

    return True


def check_rush_param(
    player: int,
    destination: list[int, int],
    location: list[int, int],
    gamestate,
) -> bool:
    x, y = location[0], location[1]
    x_new, y_new = destination[0], destination[1]
    # 检查参数合理性
    if gamestate.board[x][y].generals == None:
        return False
    if gamestate.board[x_new][y_new].generals != None:
        return False
    if gamestate.board[x][y].army < 2:
        return False
    if gamestate.board[x_new][y_new].type==2 and not gamestate.tech_level[player][1]:
        return False
    if gamestate.board[x_new][y_new].player == 1 - player:
        num = gamestate.board[x][y].army - 1
        attack = compute_attack(gamestate.board[x][y], gamestate)
        defence = compute_defence(gamestate.board[x_new][y_new], gamestate)
        vs = num * attack - gamestate.board[x_new][y_new].army * defence
        if vs <= 0:
            return False
    return True


def handle_breakthrough(destination: list[int, int], gamestate) -> bool:
    x, y = destination[0], destination[1]
    if gamestate.board[x][y].army > 20:
        gamestate.board[x][y].army -= 20
    else:
        gamestate.board[x][y].army = 0
        if gamestate.board[x][y].generals == None:
            gamestate.board[x][y].player = -1
    return True


def skill_activate(
    player: int,
    location: list[int, int],
    destination: list[int, int],
    gamestate,
    skillType: SkillType,
) -> bool:
    # 首先检查参数范围
    if player != 0 and player != 1:
        return False
    x, y = location[0], location[1]
    if x < 0 or x > row or y < 0 or y > col:
        return False
    if destination == [-1, -1]:
        destination = None
    if destination != None:
        x_new, y_new = destination[0], destination[1]
        if x_new < 0 or x_new > row or y_new < 0 or y_new > col:
            return False
        d1 = abs(x_new - x)
        d2 = abs(y_new - y)
        if d1 > 2 or d2 > 2:
            return False
    # 检查参数合理性
    if gamestate.board[x][y].player != player:
        return False
    coin = gamestate.coin[player]
    general = gamestate.board[location[0]][location[1]].generals
    if general == None or type(general).__name__ == "Oilwell":
        return False
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
    if skillType == SkillType.SURPRISE_ATTACK:
        # 检查参数是否合法
        if not check_rush_param(player, destination, location, gamestate):
            return False

        if coin >= tactical_strike and general.skills_cd[0] == 0:
            army_rush(location, gamestate, player, destination)

            gamestate.board[location[0]][location[1]].generals = None
            gamestate.board[destination[0]][destination[1]].generals = general
            general.position = [destination[0], destination[1]]
            general.skills_cd[0] = 5

            gamestate.coin[player] -= tactical_strike
            return True
        else:
            return False
    elif skillType == SkillType.ROUT:
        if coin >= breakthrough and general.skills_cd[1] == 0:
            handle_breakthrough(destination, gamestate)
            general.skills_cd[1] = 10
            gamestate.board[location[0]][location[1]].generals = general
            gamestate.coin[player] -= breakthrough
            return True
        else:
            return False
    elif skillType == SkillType.COMMAND:
        if coin >= leadership and general.skills_cd[2] == 0:
            general.skills_cd[2] = 10
            general.skill_duration[0] = 10
            gamestate.board[location[0]][location[1]].generals = general
            gamestate.coin[player] -= leadership
            return True
        else:
            return False
    elif skillType == SkillType.DEFENCE:
        if coin >= fortification and general.skills_cd[3] == 0:
            general.skills_cd[3] = 10
            general.skill_duration[1] = 10
            gamestate.board[location[0]][location[1]].generals = general
            gamestate.coin[player] -= fortification
            return True
        else:
            return False
    else:
        if coin >= weakening and general.skills_cd[4] == 0:
            general.skills_cd[4] = 10
            general.skill_duration[2] = 10
            gamestate.board[location[0]][location[1]].generals = general
            gamestate.coin[player] -= weakening
            return True
        else:
            return False
