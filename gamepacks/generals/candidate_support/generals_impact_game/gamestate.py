# 本文件定义了游戏状态类，以及负责初始化将军，更新回合的函数
from dataclasses import dataclass, field

from generals_impact_game.constant import *
from generals_impact_game.gamedata import *


@dataclass
class GameState:
    round: int = 1  # 当前游戏回合数
    generals: list[Generals] = field(default_factory=list)  # 游戏中的将军列表，用于通信
    coin: list[int] = field(
        default_factory=lambda: [init_coin() for p in range(2)]
    )  # 每个玩家的金币数量列表，分别对应玩家1，玩家2
    active_super_weapon: list[SuperWeapon] = field(default_factory=list)
    super_weapon_unlocked: list[bool] = field(
        default_factory=lambda: [False, False]
    )  # 超级武器是否解锁的列表，解锁了是true，分别对应玩家1，玩家2

    super_weapon_cd: list[int] = field(
        default_factory=lambda: [-1, -1]
    )  # 超级武器的冷却回合数列表，分别对应玩家1，玩家2

    tech_level: list[list[int]] = field(
        default_factory=lambda: [[2, 0, 0, 0], [2, 0, 0, 0]]
    )
    # 科技等级列表，第一层对应玩家一，玩家二，第二层分别对应行动力，攀岩，免疫沼泽，超级武器

    rest_move_step: list[int, int] = field(default_factory=lambda: [2, 2])
    board: list[list[Cell]] = field(
        default_factory=lambda: [
            [Cell(position=[i, j]) for j in range(col)] for i in range(row)
        ]
    )  # 游戏棋盘的二维列表，每个元素是一个Cell对象

    next_generals_id: int = 0
    winner: int = -1

    def find_general_position_by_id(self, general_id: int):
        for gen in self.generals:
            if gen.id == general_id:
                return gen.position
        return None


def update_round(gamestate: GameState):
    for i in range(row):
        for j in range(col):
            # 将军
            if gamestate.board[i][j].generals != None:
                gamestate.board[i][j].generals.rest_move = gamestate.board[i][
                    j
                ].generals.mobility_level
            if isinstance(gamestate.board[i][j].generals, MainGenerals):
                gamestate.board[i][j].army += gamestate.board[i][
                    j
                ].generals.produce_level
            elif isinstance(gamestate.board[i][j].generals, SubGenerals):
                if gamestate.board[i][j].generals.player != -1:
                    gamestate.board[i][j].army += gamestate.board[i][
                        j
                    ].generals.produce_level
            elif isinstance(gamestate.board[i][j].generals, Oilwell):
                if gamestate.board[i][j].generals.player != -1:
                    gamestate.coin[
                        gamestate.board[i][j].generals.player
                    ] += gamestate.board[i][j].generals.produce_level
            # 每25回合增兵
            if gamestate.round % 10 == 0:
                if gamestate.board[i][j].player != -1:
                    gamestate.board[i][j].army += 1
            # 沼泽减兵
            if (
                gamestate.board[i][j].type == CellType(1)
                and gamestate.board[i][j].player != -1
                and gamestate.board[i][j].army > 0
            ):
                if gamestate.tech_level[gamestate.board[i][j].player][2] == 0:
                    gamestate.board[i][j].army -= 1
                    if (
                        gamestate.board[i][j].army == 0
                        and gamestate.board[i][j].generals == None
                    ):
                        gamestate.board[i][j].player = -1

    # 超级武器判定
    for weapon in gamestate.active_super_weapon:
        if weapon.type == WeaponType(0):
            for _i in range(
                max(0, weapon.position[0] - 1), min(row, weapon.position[0] + 2)
            ):
                for _j in range(
                    max(0, weapon.position[1] - 1), min(col, weapon.position[1] + 2)
                ):
                    if gamestate.board[_i][_j].army > 0:
                        gamestate.board[_i][_j].army = max(
                            0, gamestate.board[_i][_j].army - 3
                        )
                        gamestate.board[_i][_j].player = (
                            -1
                            if (
                                gamestate.board[_i][_j].army == 0
                                and gamestate.board[_i][_j].generals == None
                            )
                            else gamestate.board[_i][_j].player
                        )

    # 更新超级武器信息
    gamestate.super_weapon_cd = [
        i - 1 if i > 0 else i for i in gamestate.super_weapon_cd
    ]
    for weapon in gamestate.active_super_weapon:
        weapon.rest -= 1
    # cd和duration 减少
    for gen in gamestate.generals:
        gen.skills_cd = [i - 1 if i > 0 else i for i in gen.skills_cd]
        gen.skill_duration = [i - 1 if i > 0 else i for i in gen.skill_duration]
    # 移动步数恢复
    gamestate.rest_move_step = [gamestate.tech_level[0][0], gamestate.tech_level[1][0]]

    # 更新超武时间
    gamestate.active_super_weapon = list(
        filter(lambda x: (x.rest > 0), gamestate.active_super_weapon)
    )

    gamestate.round += 1
