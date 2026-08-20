from generals_impact_game.constant import *
from generals_impact_game.gamedata import MainGenerals, SuperWeapon, WeaponType


def handle_bomb(gamestate, location: list[int, int]) -> bool:
    combinations = set()
    x, y = location[0], location[1]
    x1 = x - 1 if x - 1 >= 0 else 0
    x2 = x
    x3 = x + 1 if x + 1 <= col - 1 else col - 1
    y1 = y - 1 if y - 1 >= 0 else 0
    y2 = y
    y3 = y + 1 if y + 1 <= col - 1 else col - 1
    combinations.add((x1, y1))
    combinations.add((x1, y2))
    combinations.add((x1, y3))
    combinations.add((x2, y1))
    combinations.add((x2, y2))
    combinations.add((x2, y3))
    combinations.add((x3, y1))
    combinations.add((x3, y2))
    combinations.add((x3, y3))
    for combination in combinations:
        x, y = combination
        handle_bomb_cell(gamestate, x, y)
    return True


def handle_bomb_cell(gamestate, x: int, y: int) -> bool:
    cell = gamestate.board[x][y]
    if isinstance(cell.generals, MainGenerals):
        cell.army = cell.army // 2
    else:
        cell.army = 0
        cell.player = -1
        cell.generals = None
        for general in gamestate.generals:
            if general.position == [x, y]:
                gamestate.generals.remove(general)
    gamestate.board[x][y] = cell

    return True


def bomb(gamestate, location: list[int, int], player: int) -> bool:
    # 先检查参数
    if player != 0 and player != 1:
        return False
    if location[0] < 0 or location[0] > col:
        return False
    if location[1] < 0 or location[1] > row:
        return False
    is_super_weapon_unlocked = gamestate.super_weapon_unlocked[player]
    cd = gamestate.super_weapon_cd[player]
    if is_super_weapon_unlocked == True and cd == 0:
        gamestate.active_super_weapon.append(
            SuperWeapon(WeaponType(0), player, 0, 5, location)
        )
        gamestate.super_weapon_cd[player] = use_cd

        handle_bomb(gamestate, location)

        return True
    else:
        return False


def strengthen(gamestate, location: list[int, int], player: int) -> bool:
    # 先检查参数
    if player != 0 and player != 1:
        return False
    if location[0] < 0 or location[0] > col:
        return False
    if location[1] < 0 or location[1] > row:
        return False
    is_super_weapon_unlocked = gamestate.super_weapon_unlocked[player]
    cd = gamestate.super_weapon_cd[player]
    if is_super_weapon_unlocked == True and cd == 0:
        weapon = SuperWeapon(WeaponType(1), player, 5, 5, location)
        gamestate.active_super_weapon.append(weapon)
        gamestate.super_weapon_cd[player] = use_cd

        return True
    else:
        return False


def tp(gamestate, start: list[int, int], to: list[int, int], player: int) -> bool:
    # 先检查参数范围
    if player != 0 and player != 1:
        return False
    if start[0] < 0 or start[0] > col:
        return False
    if start[1] < 0 or start[1] > row:
        return False
    if to[0] < 0 or to[0] > col:
        return False
    if to[1] < 0 or to[1] > row:
        return False
    # 还需判断目标是否合法
    is_super_weapon_unlocked = gamestate.super_weapon_unlocked[player]
    cd = gamestate.super_weapon_cd[player]
    if is_super_weapon_unlocked == True and cd == 0:
        x_st, y_st = start[0], start[1]
        x_to, y_to = to[0], to[1]
        cell_st, cell_to = gamestate.board[x_st][y_st], gamestate.board[x_to][y_to]
        if cell_st.player != player:
            return False
        if cell_to.generals != None:
            return False
        num = 0
        if cell_st.army == 0 or cell_st.army == 1:
            return False
        if cell_to.type==2 and not gamestate.tech_level[player][1]:
            return False
        else:
            num = cell_st.army - 1
            cell_st.army = 1
        cell_to.army = num
        cell_to.player = player
        gamestate.board[x_st][y_st] = cell_st
        gamestate.board[x_to][y_to] = cell_to
        gamestate.super_weapon_cd[player] = use_cd
        weapon = SuperWeapon(WeaponType(2), player, 2, 2, to)
        gamestate.active_super_weapon.append(weapon)

        return True
    return False


def timestop(gamestate, location: list[int, int], player: int) -> bool:
    # 首先检查参数范围
    if player != 0 and player != 1:
        return False
    x, y = location[0], location[1]
    if x < 0 or x > col or y < 0 or y > row:
        return False
    # 检查参数合理性
    is_super_weapon_unlocked = gamestate.super_weapon_unlocked[player]
    cd = gamestate.super_weapon_cd[player]
    if is_super_weapon_unlocked == True and cd == 0:
        weapon = SuperWeapon(
            type=WeaponType(3), player=player, cd=10, rest=10, position=location
        )
        gamestate.active_super_weapon.append(weapon)
        gamestate.super_weapon_cd[player] = use_cd

        return True
    else:
        return False
