import generals_impact_game.constant as constant
from generals_impact_game.gamedata import WeaponType


# 本文件中的函数负责升级将军，解锁技能等，需要的金币从demo里的constant.py中调取，方便修改
def production_up(location: list[int, int], gamestate, player: int) -> bool:
    if not gamestate.board[location[0]][location[1]].generals:
        return False  # 没有将军
    if gamestate.board[location[0]][location[1]].player != player:
        return False  # 操作了别人的格子
    for sw in gamestate.active_super_weapon:  # 超级武器效果
        if (
            sw.position == location
            and sw.rest
            and sw.type == WeaponType.TRANSMISSION
            and sw.player == player
        ):  # 超时空传送眩晕
            return False
        if (
            abs(sw.position[0] - location[0]) <= 1
            and abs(sw.position[1] - location[1]) <= 1
            and sw.rest
            and sw.type == WeaponType.TIME_STOP
        ):  # 时间暂停效果
            return False
    if type(gamestate.board[location[0]][location[1]].generals).__name__ == "Oilwell":
        if gamestate.board[location[0]][location[1]].generals.produce_level == 1:
            if gamestate.coin[player] < constant.oilwell_production_T1:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.produce_level = 2
                gamestate.coin[player] -= constant.oilwell_production_T1
        elif gamestate.board[location[0]][location[1]].generals.produce_level == 2:
            if gamestate.coin[player] < constant.oilwell_production_T2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.produce_level = 4
                gamestate.coin[player] -= constant.oilwell_production_T2
        elif gamestate.board[location[0]][location[1]].generals.produce_level == 4:
            if gamestate.coin[player] < constant.oilwell_production_T3:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.produce_level = 6
                gamestate.coin[player] -= constant.oilwell_production_T3
        else:
            return False
    if (
        type(gamestate.board[location[0]][location[1]].generals).__name__
        == "MainGenerals"
    ):
        if gamestate.board[location[0]][location[1]].generals.produce_level == 1:
            if gamestate.coin[player] < constant.lieutenant_production_T1 // 2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.produce_level = 2
                gamestate.coin[player] -= constant.lieutenant_production_T1 // 2
        elif gamestate.board[location[0]][location[1]].generals.produce_level == 2:
            if gamestate.coin[player] < constant.lieutenant_production_T2 // 2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.produce_level = 4
                gamestate.coin[player] -= constant.lieutenant_production_T2 // 2
        else:
            return False
    if (
        type(gamestate.board[location[0]][location[1]].generals).__name__
        == "SubGenerals"
    ):
        if gamestate.board[location[0]][location[1]].generals.produce_level == 1:
            if gamestate.coin[player] < constant.lieutenant_production_T1:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.produce_level = 2
                gamestate.coin[player] -= constant.lieutenant_production_T1
        elif gamestate.board[location[0]][location[1]].generals.produce_level == 2:
            if gamestate.coin[player] < constant.lieutenant_production_T2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.produce_level = 4
                gamestate.coin[player] -= constant.lieutenant_production_T2
        else:
            return False
    return True


def defence_up(location: list[int, int], gamestate, player: int) -> bool:
    if not gamestate.board[location[0]][location[1]].generals:
        return False  # 没有将军
    if gamestate.board[location[0]][location[1]].player != player:
        return False  # 操作了别人的格子
    for sw in gamestate.active_super_weapon:  # 超级武器效果
        if (
            sw.position == location
            and sw.rest
            and sw.type == WeaponType.TRANSMISSION
            and sw.player == player
        ):  # 超时空传送眩晕
            return False
        if (
            abs(sw.position[0] - location[0]) <= 1
            and abs(sw.position[1] - location[1]) <= 1
            and sw.rest
            and sw.type == WeaponType.TIME_STOP
        ):  # 时间暂停效果
            return False
    if type(gamestate.board[location[0]][location[1]].generals).__name__ == "Oilwell":
        if gamestate.board[location[0]][location[1]].generals.defense_level == 1:
            if gamestate.coin[player] < constant.oilwell_defense_T1:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.defense_level = 1.5
                gamestate.coin[player] -= constant.oilwell_defense_T1
        elif gamestate.board[location[0]][location[1]].generals.defense_level == 1.5:
            if gamestate.coin[player] < constant.oilwell_defense_T2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.defense_level = 2
                gamestate.coin[player] -= constant.oilwell_defense_T2
        elif gamestate.board[location[0]][location[1]].generals.defense_level == 2:
            if gamestate.coin[player] < constant.oilwell_defense_T3:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.defense_level = 3
                gamestate.coin[player] -= constant.oilwell_defense_T3
        else:
            return False
    if (
        type(gamestate.board[location[0]][location[1]].generals).__name__
        == "MainGenerals"
    ):
        if gamestate.board[location[0]][location[1]].generals.defense_level == 1:
            if gamestate.coin[player] < constant.lieutenant_defense_T1 // 2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.defense_level = 2
                gamestate.coin[player] -= constant.lieutenant_defense_T1 // 2
        elif gamestate.board[location[0]][location[1]].generals.defense_level == 2:
            if gamestate.coin[player] < constant.lieutenant_defense_T2 // 2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.defense_level = 3
                gamestate.coin[player] -= constant.lieutenant_defense_T2 // 2
        else:
            return False
    if (
        type(gamestate.board[location[0]][location[1]].generals).__name__
        == "SubGenerals"
    ):
        if gamestate.board[location[0]][location[1]].generals.defense_level == 1:
            if gamestate.coin[player] < constant.lieutenant_defense_T1:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.defense_level = 2
                gamestate.coin[player] -= constant.lieutenant_defense_T1
        elif gamestate.board[location[0]][location[1]].generals.defense_level == 2:
            if gamestate.coin[player] < constant.lieutenant_defense_T2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.defense_level = 3
                gamestate.coin[player] -= constant.lieutenant_defense_T2
        else:
            return False
    return True


def movement_up(location: list[int, int], gamestate, player: int) -> bool:
    if not gamestate.board[location[0]][location[1]].generals:
        return False  # 没有将军
    if gamestate.board[location[0]][location[1]].player != player:
        return False  # 操作了别人的格子
    if type(gamestate.board[location[0]][location[1]].generals).__name__ == "Oilwell":
        return False  # 农民不可以升级
    for sw in gamestate.active_super_weapon:  # 超级武器效果
        if (
            sw.position == location
            and sw.rest
            and sw.type == WeaponType.TRANSMISSION
            and sw.player == player
        ):  # 超时空传送眩晕
            return False
        if (
            abs(sw.position[0] - location[0]) <= 1
            and abs(sw.position[1] - location[1]) <= 1
            and sw.rest
            and sw.type == WeaponType.TIME_STOP
        ):  # 时间暂停效果
            return False
    if (
        type(gamestate.board[location[0]][location[1]].generals).__name__
        == "MainGenerals"
    ):
        if gamestate.board[location[0]][location[1]].generals.mobility_level == 1:
            if gamestate.coin[player] < constant.general_movement_T1 // 2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.mobility_level = 2
                gamestate.board[location[0]][location[1]].generals.rest_move += 1
                gamestate.coin[player] -= constant.general_movement_T1 // 2
        elif gamestate.board[location[0]][location[1]].generals.mobility_level == 2:
            if gamestate.coin[player] < constant.general_movement_T2 // 2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.mobility_level = 4
                gamestate.board[location[0]][location[1]].generals.rest_move += 2
                gamestate.coin[player] -= constant.general_movement_T2 // 2
        else:
            return False
    if (
        type(gamestate.board[location[0]][location[1]].generals).__name__
        == "SubGenerals"
    ):
        if gamestate.board[location[0]][location[1]].generals.mobility_level == 1:
            if gamestate.coin[player] < constant.general_movement_T1:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.mobility_level = 2
                gamestate.board[location[0]][location[1]].generals.rest_move += 1
                gamestate.coin[player] -= constant.general_movement_T1
        elif gamestate.board[location[0]][location[1]].generals.mobility_level == 2:
            if gamestate.coin[player] < constant.general_movement_T2:
                return False
            else:
                gamestate.board[location[0]][location[1]].generals.mobility_level = 4
                gamestate.board[location[0]][location[1]].generals.rest_move += 2
                gamestate.coin[player] -= constant.general_movement_T2
        else:
            return False
    return True


"""
以上部分是将军升级的函数，接受location，gamestate，player参数，其中。location是一个list，location[0]为x坐标，location[1]为y坐标
gamestate为全局唯一的对象，而player传入player0或player1，这是为了防止恶意操纵他人的格子
在这三个函数里，首先需要通过location找到具体操作的格子，接下来判断该操作是否合法(是否操作了他人格子，钱是否够，格子上是否有将军)，
如果不合法则不扣除金币，不进行操作，返回False
如果合法则通过操作gamestate.board[location[0]][location[1]].generals对属性进行修改，并扣除相应的金币

"""


def tech_update(tech_type: int, gamestate, player: int) -> bool:  # 规定0123分别代表文档中的对应序号科技
    if tech_type == 0:
        if gamestate.tech_level[player][0] == 2:
            if gamestate.coin[player] < constant.army_movement_T1:
                return False
            else:
                gamestate.tech_level[player][0] = 3
                gamestate.coin[player] -= constant.army_movement_T1
                gamestate.rest_move_step[player] += 1
                return True
        elif gamestate.tech_level[player][0] == 3:
            if gamestate.coin[player] < constant.army_movement_T2:
                return False
            else:
                gamestate.tech_level[player][0] = 5
                gamestate.coin[player] -= constant.army_movement_T2
                gamestate.rest_move_step[player] += 2
                return True
        else:
            return False
    elif tech_type == 1:
        if gamestate.tech_level[player][1] == 0:
            if gamestate.coin[player] < constant.mountaineering:
                return False
            else:
                gamestate.tech_level[player][1] = 1
                gamestate.coin[player] -= constant.mountaineering
                return True
        else:
            return False
    elif tech_type == 2:
        if gamestate.tech_level[player][2] == 0:
            if gamestate.coin[player] < constant.swamp_immunity:
                return False
            else:
                gamestate.tech_level[player][2] = 1
                gamestate.coin[player] -= constant.swamp_immunity
                return True
        else:
            return False
    elif tech_type == 3:
        if gamestate.tech_level[player][3] == 0:
            if gamestate.coin[player] < constant.unlock_super_weapon:
                return False
            else:
                gamestate.tech_level[player][3] = 1
                gamestate.super_weapon_cd[player] = constant.start_cd
                gamestate.super_weapon_unlocked[player] = True
                gamestate.coin[player] -= constant.unlock_super_weapon
                return True
        else:
            return False
