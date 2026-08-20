'''AI基类
'''

import sys
import json
import calculator
import gameunit


def convert_byte(data_str):
    '''加数据长度作为数据头,并转化为bytes

    Args:
        data_str: str

    Returns:
        头部加上了长度的bytes串
    '''
    message_len = len(data_str)
    message = message_len.to_bytes(4, byteorder='big', signed=True)
    message += bytes(data_str, encoding="utf8")
    return message


def send_opt(data_str):
    '''发送操作

    Args:
        data_str: str
    '''
    sys.stdout.buffer.write(convert_byte(data_str))
    sys.stdout.flush()


def read_opt():
    '''读取信息

    Return:
        dict
    '''
    read_buffer = sys.stdin.buffer
    data_length = read_buffer.read(6)
    data = read_buffer.read(int(data_length.decode()))
    return json.loads(data)


class AiClient:
    '''AI基类,包含各种可用函数
    '''

    def __init__(self):
        self.map = gameunit.Map()                                   # 地图信息
        self.players = [gameunit.Player(0), gameunit.Player(1)]     # 两名玩家的信息
        self.round = 0                                              # 当前回合
        self.my_camp = read_opt()['camp']                           # 己方阵营
        self.artifacts = ["HolyLight"]                              # 己方神器
        self.creatures = ["Archer", "Swordsman", "VolcanoDragon"]   # 己方生物

    def update_game_info(self):
        '''更新游戏信息
        '''
        game_info = read_opt()
        self.round = game_info['round']
        self.my_camp = game_info['camp']
        self.map.update(game_info['map'])
        self.players = [gameunit.Player(0, game_info['players'][0]),
                        gameunit.Player(1, game_info['players'][1])]

    def choose_cards(self):
        '''(获取阵营后)选择初始卡组
        '''
        self.init()

    def play(self):
        '''处理游戏信息并做出操作
        '''
        if self.round < 5:
            self.end_round()
        else:
            exit(0)

    def init(self):
        '''选择初始神器artifacts和生物creatures

        Args:
            artifacts: 神器名字的字符串数组
            creatures: 生物名字的字符串数组
        '''
        message = {
            "player": self.my_camp,
            "round": 0,
            "operation_type": "init",
            "operation_parameters":
            {
                "artifacts": self.artifacts,
                "creatures": self.creatures
            }
        }
        send_opt(json.dumps(message))

    def summon(self, _type: str, level: int, position: tuple):
        '''在位置position处召唤一个本方类型为_type,星级为level的生物

        Args:
            _type: 描述生物类型的字符串
            level: 生物星级
            position: 地图上的(x,y,z)位置
        '''
        message = {
            "player": self.my_camp,
            "round": self.round,
            "operation_type": "summon",
            "operation_parameters":
            {
                "position": position,
                "type": _type,
                "level": level
            }
        }
        send_opt(json.dumps(message))
        self.update_game_info()

    def move(self, mover: int, position: tuple):
        '''将id为mover的生物移动到位置position处

        Args:
            mover: 需要移动的生物的id
            position: 目标在地图上的(x,y,z)位置
        '''
        message = {
            "player": self.my_camp,
            "round": self.round,
            "operation_type": "move",
            "operation_parameters":
            {
                "mover": mover,
                "position": position
            }
        }
        send_opt(json.dumps(message))
        self.update_game_info()

    def attack(self, attacker: int, target: int):
        '''令id为attacker的生物攻击id为target的生物或神迹

        Args:
            attacker: 发起攻击生物的id
            target: 被攻击生物的id
        '''
        message = {
            "player": self.my_camp,
            "round": self.round,
            "operation_type": "attack",
            "operation_parameters":
            {
                "attacker": attacker,
                "target": target
            }
        }
        send_opt(json.dumps(message))
        self.update_game_info()

    def use(self, artifact: int, target):
        '''对目标target使用artifact神器

        Args:
            artifact: 使用的神器的id
            target: 目标生物的id / 地图的[x,y,z]位置
        '''
        message = {
            "player": self.my_camp,
            "round": self.round,
            "operation_type": "use",
            "operation_parameters":
            {
                "card": artifact,
                "target": target
            }
        }
        send_opt(json.dumps(message))
        self.update_game_info()

    def end_round(self):
        '''结束当前回合
        '''
        message = {
            "player": self.my_camp,
            "round": self.round,
            "operation_type": "endround",
            "operation_parameters": {}
        }
        send_opt(json.dumps(message))

    def get_distance_on_ground(self, pos_a: tuple, pos_b: tuple, camp: int) -> int:
        '''获取camp阵营生物从位置pos_a到位置pos_b的地面距离(不经过地面障碍或敌方地面生物)

        Args:
            pos_a: 起点坐标
            pos_b: 终点坐标
            camp: 生物阵营

        Returns:
            int 距离
        '''
        # 地图边界
        obstacles_pos = calculator.MAPBORDER()
        # 地面障碍
        obstacles_pos += self.map.ground_obstacles
        # 敌方地面生物
        for unit in self.map.units:
            if unit.camp != camp and (not unit.flying):
                obstacles_pos.append(unit.pos)
        return len(calculator.search_path(pos_a, pos_b, obstacles_pos, []))

    def get_distance_in_sky(self, pos_a: tuple, pos_b: tuple, camp: int) -> int:
        '''获取camp阵营生物从位置pos_a到位置pos_b的飞行距离(不经过飞行障碍或敌方飞行生物)

        Args:
            pos_a: 起点坐标
            pos_b: 终点坐标
            camp: 生物阵营

        Returns:
            int 距离
        '''
        # 地图边界
        obstacles_pos = calculator.MAPBORDER()
        # 地面障碍
        obstacles_pos += self.map.flying_obstacles
        # 敌方地面生物
        for unit in self.map.units:
            if unit.camp != camp and unit.flying:
                obstacles_pos.append(unit.pos)
        return len(calculator.search_path(pos_a, pos_b, obstacles_pos, []))

    def check_barrack(self, pos: tuple) -> int:
        '''判定位置pos的驻扎情况

        Args:
            pos: 地图上的(x,y,z)位置

        Returns:
            int 不是驻扎点返回-2,中立返回-1,否则返回占领该驻扎点的阵营(0或1)
        '''
        for barrack in self.map.barracks:
            if barrack.pos == pos:
                return barrack.camp
        return -2

    def can_attack(self, attacker: gameunit.Unit, target: gameunit.Unit) -> bool:
        '''判断生物attacker能否攻击到生物target(只考虑攻击力、攻击范围)

        Args:
            attacker: 发起攻击的生物
            target: 受攻击的生物

        Returns:
            bool 能攻击到返回True,不能攻击到返回False
        '''
        # 攻击力小于等于0的生物无法攻击
        if attacker.atk <= 0:
            return False
        # 攻击范围
        dist = calculator.cube_distance(attacker.pos, target.pos)
        if dist < attacker.atk_range[0] or dist > attacker.atk_range[1]:
            return False
        # 对空攻击
        if target.flying and (not attacker.atk_flying):
            return False
        return True

    def can_use_artifact(self, artifact: gameunit.Artifact, target, camp: int) -> bool:
        '''判断阵营camp能否对目标target使用神器artifact(不考虑消耗、冷却)

        Args:
            artifact: 神器
            target: 目标(Unit或者Pos)
            camp: 使用神器的阵营

        Returns:
            bool 能攻击到返回True,不能攻击到返回False
        '''
        if isinstance(target, gameunit.Unit):
            if artifact.name == "SalamanderShield":
                return artifact.camp == target.camp
        elif isinstance(target, tuple) and len(target) == 3:
            if artifact.name == "HolyLight" or artifact.name == "WindBlessing":
                return calculator.in_map(target)
            if artifact.name == "InfernoFlame":
                # 不处于障碍物上
                for obstacle in self.map.obstacles:
                    if obstacle.pos == target:
                        return False
                # 无地面生物
                for unit in self.map.units:
                    if (unit.pos == target) and (not unit.flying):
                        return False
                # 神迹范围<=7
                if calculator.cube_distance(target, self.map.miracles[camp].pos) <= 7:
                    return True
                # 占领驻扎点范围<=5
                for barrack in self.map.barracks:
                    if (barrack.camp == camp and
                            calculator.cube_distance(target, barrack.pos) <= 5):
                        return True
        return False

    def get_unit_by_pos(self, pos: tuple, flying: bool) -> gameunit.Unit:
        '''获取位置pos上的生物

        Args:
            pos: 地图上的(x,y,z)位置
            flying: 飞行生物还是地面生物

        Returns:
            Unit 如果有,返回对应的unit,否则返回None
        '''
        for unit in self.map.units:
            if unit.pos == pos and unit.flying == flying:
                return unit
        return None

    def get_unit_by_id(self, unit_id: int) -> gameunit.Unit:
        '''获取id为unit_id的生物

        Args:
            unit_id: 要找的生物的id

        Returns:
            Unit 如果有,返回对应的unit,否则返回None
        '''
        for unit in self.map.units:
            if unit.id == unit_id:
                return unit
        return None

    def get_units_by_camp(self, unit_camp: int) -> list:
        '''获取地图上所有阵营为unit_camp的生物

        Args:
            unit_camp: 要找的生物的阵营

        Returns:
            list (Unit) 返回camp等于unit_camp的Unit列表(没有时返回空列表)
        '''
        camp_units = []
        for unit in self.map.units:
            if unit.camp == unit_camp:
                camp_units.append(unit)
        return camp_units

    def get_summon_pos_by_camp(self, camp: int) -> list:
        '''获取地图上所有属于阵营camp的出兵点(初始出兵点+额外出兵点)

        Args:
            camp: 阵营

        Returns:
            list (Pos)
        '''
        summon_pos = []
        for miracle in self.map.miracles:
            if miracle.camp == camp:
                summon_pos += miracle.summon_pos_list
        for barrack in self.map.barracks:
            if barrack.camp == camp:
                summon_pos += barrack.summon_pos_list
        return summon_pos
