'''游戏单位
'''

UNIT_TYPE = ["Archer", "Swordsman", "BlackBat", "Priest", "VolcanoDragon", "Inferno"]
ARTIFACT_NAME = ["HolyLight", "SalamanderShield", "InfernoFlame"]
ARTIFACT_STATE = ["Ready", "In Use", "Cooling Down"]
ARTIFACT_TARGET = ["Pos", "Unit"]


class Unit:
    '''生物
    '''

    def __init__(self, unit_list=None):
        if unit_list is None:
            unit_list = [-1, -1, 0, 0, 0, 0, 0,
                         (0, 0), 0, 0, (0, 0, 0), 0, False, False, False, False, False, False]
        self.id = unit_list[0]  # id
        self.camp = unit_list[1]  # 阵营
        self.type = UNIT_TYPE[unit_list[2]]  # 种类
        self.cost = unit_list[3]  # 法力消耗
        self.atk = unit_list[4]  # 攻击
        self.max_hp = unit_list[5]  # 生命上限
        self.hp = unit_list[6]  # 当前生命
        self.atk_range = unit_list[7]  # 最小攻击范围 最大攻击范围
        self.max_move = unit_list[8]  # 行动力
        self.cool_down = unit_list[9]  # 冷却时间
        self.pos = tuple(unit_list[10])  # 位置
        self.level = unit_list[11]  # 等级
        self.flying = bool(unit_list[12])  # 是否飞行
        self.atk_flying = bool(unit_list[13])  # 是否对空
        self.agility = bool(unit_list[14])  # 是否迅捷
        self.holy_shield = bool(unit_list[15])  # 有无圣盾
        self.can_atk = bool(unit_list[16])  # 能否攻击
        self.can_move = bool(unit_list[17])  # 能否移动


class Barrack:
    '''驻扎点
    '''

    def __init__(self, pos=(0,0,0), summon_pos_list=None, camp=-1):
        if summon_pos_list is None:
            summon_pos_list=[]
        self.pos = tuple(pos)  # 位置
        self.camp = camp  # 阵营
        self.summon_pos_list = summon_pos_list  # 出兵点位置


class Miracle:
    '''神迹
    '''

    def __init__(self, camp, max_hp, hp, pos, summon_pos_list, _id):
        self.camp = camp  # 阵营
        self.max_hp = max_hp  # 最大生命值
        self.hp = hp  # 当前生命值
        self.pos = tuple(pos)  # 位置
        self.summon_pos_list = [tuple(x) for x in summon_pos_list]  # 初始出兵点位置
        self.id = _id  # id


class Obstacle:
    def __init__(self, _type, pos, allow_flying, allow_ground):
        self.type = _type  # 种类
        self.pos = tuple(pos)  # 位置
        self.allow_flying = allow_flying  # 是否允许飞行生物通过
        self.allow_ground = allow_ground  # 是否允许地面生物通过


class Artifact:
    '''神器
    '''

    def __init__(self, artifact_list=None):
        if artifact_list is None:
            artifact_list = [-1, 0,  0, 0, 0, 0, 0]
        self.camp = artifact_list[0]
        self.name = ARTIFACT_NAME[artifact_list[1]]
        self.id = self.camp
        self.cost = artifact_list[2]
        self.max_cool_down = artifact_list[3]
        self.cool_down_time = artifact_list[4]
        self.state = ARTIFACT_STATE[artifact_list[5]]
        self.target_type = ARTIFACT_TARGET[artifact_list[6]]


class CreatureCapacity:
    def __init__(self, cc_list=None):
        if cc_list is None:
            cc_list = [0, 0, []]
        self.type = UNIT_TYPE[cc_list[0]]         # 种类
        self.available_count = cc_list[1]  # 生物槽容量
        self.cool_down_list = cc_list[2]  # 冷却时间


class Map:
    '''地图
    '''

    def __init__(self):
        self.units = [Unit()]
        self.barracks = [Barrack((-6, -6, 12), [(-7, -5, 12), (-5, -7, 12), (-5, -6, 11)]),
                         Barrack((6, 6, -12), [(7, 5, -12), (5, 7, -12), (5, 6, -11)]),
                         Barrack((0, -5, 5), [(0, -4, 4), (-1, -4, 5), (-1, -5, 6)]),
                         Barrack((0, 5, -5), [(0, 4, -4), (1, 4, -5), (1, 5, -6)])]
        self.miracles = [Miracle(0, 30, 30, (-7, 7, 0), [(-8, 6, 2), (-7, 6, 1), (-6, 6, 0), (-6, 7, -1), (-6, 8, -2)], 0),
                         Miracle(1, 30, 30, (7, -7, 0), [(8, -6, -2), (7, -6, -1), (6, -6, 0), (6, -7, 1), (6, -8, 2)], 1)]
        ABYSS_POS_LIST = [(0, 0, 0), (-1, 0, 1), (0, -1, 1), (1, -1, 0), (1, 0, -1),
                          (0, 1, -1), (-1, 1, 0), (-2, -1, 3), (-1, -2, 3), (-2, -2, 4),
                          (-3, -2, 5), (-4, -4, 8), (-5, -4, 9), (-4, -5, 9), (-5, -5, 10),
                          (-6, -5, 11), (1, 2, -3), (2, 1, -3), (2, 2, -4), (3, 2, -5),
                          (4, 4, -8), (5, 4, -9), (4, 5, -9), (5, 5, -10), (6, 5, -11)]
        self.obstacles = [Obstacle("Abyss", pos, True, False) for pos in ABYSS_POS_LIST]\
            + [Obstacle("Miracle", (-7, 7, 0), False, False)]\
            + [Obstacle("Miracle", (7, -7, 0), False, False)]

        self.flying_obstacles = []
        self.ground_obstacles = self.obstacles[:]

    def update(self, map_dict=None):
        if map_dict is None:
            return
        self.units = [Unit(x) for x in map_dict.get('units', [])]
        for barrack_index, barrack_camp in enumerate(map_dict.get('barracks', [])):
            self.barracks[barrack_index].camp = barrack_camp
        miracle_hp_list = map_dict.get('miracles', [])
        self.miracles[0].hp, self.miracles[1].hp = miracle_hp_list[0], miracle_hp_list[1]



class Player:
    '''玩家
    '''

    def __init__(self, camp, player_list=None):
        if player_list is None:
            player_list = [[], 0, 0, [], []]
        self.camp = camp
        self.artifact = [Artifact(x) for x in player_list[0]]  # 神器
        self.mana = player_list[1]  # 当前法力值
        self.max_mana = player_list[2]  # 最大法力值
        self.creature_capacity = [CreatureCapacity()]
        self.creature_capacity = [CreatureCapacity(x) for x in player_list[3]]
        self.new_summoned_id_list = player_list[4]  # 最新召唤的生物id
