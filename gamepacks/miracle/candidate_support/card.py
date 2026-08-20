'''标准卡牌
'''
import json
import os


class Creature:
    '''生物
    '''

    def __init__(self, _type="", _count=0, _level=0, _cost=0, _atk=0, _maxhp=0, _minatk=0,
                 _maxatk=0, _maxmove=0, _cool=0, _fly=False, _atkfly=False, _agility=False,
                 _holyshield=False):
        self.type = _type  # 种类
        self.available_count = _count  # 生物槽容量
        self.level = _level  # 等级
        self.cost = _cost  # 法力消耗
        self.atk = _atk  # 攻击
        self.max_hp = _maxhp  # 最大生命值
        self.min_atk_range = _minatk  # 最小攻击范围
        self.max_atk_range = _maxatk  # 最大攻击范围
        self.max_move = _maxmove  # 行动力
        self.cool_down = _cool  # 冷却时间
        self.flying = _fly  # 是否飞行
        self.atk_flying = _atkfly  # 能否对空
        self.agility = _agility  # 是否迅捷
        self.holy_shield = _holyshield  # 有无圣盾

    def __str__(self):
        return 'type: {}, available_count: {}, level: {}, cost: {}, atk: {}, max_hp: {},\
 atk_range: [{}, {}], max_move: {}, cool_down: {}, flying: {},\
 atk_flying: {}, agility: {}, holy_shield: {}'.format(
            self.type, self.available_count, self.level, self.cost, self.atk, self.max_hp,
            self.min_atk_range, self.max_atk_range, self.max_move, self.cool_down,
            self.flying, self.atk_flying, self.agility, self.holy_shield)


class Artifact:
    '''神器
    '''

    def __init__(self, _name, _cost, _cool, _targettype):
        self.name = _name  # 名字
        self.cost = _cost  # 法力消耗
        self.cool_down = _cool  # 冷却时间
        self.target_type = _targettype  # 目标类型

    def __str__(self):
        return '''name: {}, camp: {}, cool_down: {}, target_type: {}'''.format(
            self.name, self.cost, self.cool_down, self.target_type)


DATA = json.load(open("Data.json", "r"))
CREATURE_DATA = DATA["UnitData"]
ARTIFACT_DATA = DATA["Artifacts"]


def get_creature_data(name):
    '''
    Returns:
        a list contains a creature's data of all levels
    '''
    data = CREATURE_DATA[name]
    creature_list = []
    for level in range(len(data["hp"])):
        creature_list.append(Creature(name, data["duplicate"][level], level + 1, data["cost"][level],
                                      data["atk"][level], data["hp"][level], data["atk_range"][level][0],
                                      data["atk_range"][level][1], data["max_move"][level], data["cool_down"][level],
                                      data["flying"], data["atk_flying"], data["agility"], data["holy_shield"]))
    return creature_list


def get_artifact_data(name):
    '''
    Returns:
        an Artifact object
    '''
    data = ARTIFACT_DATA[name]
    return Artifact(name, data["cost"], data["cool_down"], data["target_type"])


SWORDSMAN = [Creature("Swordsman")] + get_creature_data("Swordsman") # 剑士
ARCHER = [Creature("Archer")] + get_creature_data("Archer") # 弓箭手
BLACKBAT = [Creature("BlackBat")] + get_creature_data("BlackBat") # 黑蝙蝠
PRIEST = [Creature("Priest")] + get_creature_data("Priest") # 牧师
VOLCANOGRAGON = [Creature("VolcanoDragon")] + get_creature_data("VolcanoDragon") # 火山之龙
FROSTDRAGON = [Creature("FrostDragon")] + get_creature_data("FrostDragon") # 冰霜之龙
INFERNO = get_creature_data("Inferno")[0] # 地狱火

HOLYLIGHT = get_artifact_data("HolyLight") # 圣光之耀
SALAMANDERSHIELD = get_artifact_data("SalamanderShield") # 阳炎之盾
INFERNOFLAME = get_artifact_data("InfernoFlame") # 地狱之火
WINDBLESSING = get_artifact_data("WindBlessing") # 风神之佑

CARD_DICT = {
    'Swordsman': SWORDSMAN,
    'Archer': ARCHER,
    'BlackBat': BLACKBAT,
    'Priest': PRIEST,
    'VolcanoDragon': VOLCANOGRAGON,
    'FrostDragon': FROSTDRAGON
}

if __name__ == "__main__":
    for creatures in CARD_DICT.values():
        for creature in creatures:
            print(creature)
    print(INFERNO)
    print(HOLYLIGHT, SALAMANDERSHIELD, INFERNOFLAME, sep="\n")
    print("3星弓箭手的攻击力:", ARCHER[3].atk)
    print("圣光之耀的法力消耗:", HOLYLIGHT.cost)
