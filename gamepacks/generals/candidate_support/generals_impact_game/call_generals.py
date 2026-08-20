from generals_impact_game.gamedata import SubGenerals, WeaponType


def call_generals(state, player: int, position: list[int, int]) -> bool:
    if state.coin[player] < 50:
        return False
    elif state.board[position[0]][position[1]].player != player:
        return False
    elif state.board[position[0]][position[1]].generals != None:
        return False
    for sw in state.active_super_weapon:  # 超级武器效果
        if (
            sw.position == position
            and sw.rest
            and sw.type == WeaponType.TRANSMISSION
            and sw.player == player
        ):  # 超时空传送眩晕
            return False
        if (
            abs(sw.position[0] - position[0]) <= 1
            and abs(sw.position[1] - position[1]) <= 1
            and sw.rest
            and sw.type == WeaponType.TIME_STOP
        ):  # 时间暂停效果
            return False
    gen = SubGenerals(state.next_generals_id, player, position)
    state.board[position[0]][position[1]].generals = gen
    state.generals.append(gen)
    state.next_generals_id += 1
    state.coin[player] -= 50
    return True
