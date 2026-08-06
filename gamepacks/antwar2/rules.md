# AntWar2 冻结规则

AntWar2 是 19×19 offset hex 网格上的双人塔防游戏。P0 和 P1 各有 50 HP
基地，位置分别为 `(2,9)` 与 `(16,9)`。每轮双方提交有序操作列表；蚂蚁移动、
攻击、生成、状态效果和伤害结算由环境完成，策略不能直接指定蚂蚁移动。

## 公开状态

公开 `BackendState` 包含 round、tower delta、ants、双方 coins、bases、武器冷却和
active effects。塔 ID 为全局 ID，必须检查 owner。`state.towers` 的输入是增量流：
`type == -1` 删除该塔，其余记录创建或更新；策略 SDK 会维护完整状态。

实时策略读取：

- 基地 HP：`state.bases[player].hp`
- 出兵等级：`state.bases[player].generation_level`
- 蚂蚁生命等级：`state.bases[player].ant_level`
- 金币：`state.coins[player]`

回放 JSON 的 `camps/speedLv/anthpLv` 映射到上述 `bases` 字段。实时策略没有
`state.camps`。

## 胜负

基地降至 0 HP 时失败；同一结算中双方均归零则 P0 获胜。第 512 轮仍未结束时，
冻结后端依次比较基地 HP、击杀敌蚁数、较少超级武器使用量和较低 AI 总耗时；
完全相同则 P0 获胜。只有无协议或基础设施故障的终局 `winner` 才是有效结果。

## 操作与经济

- 11 `BUILD_TOWER(x,y)`：建造 Basic 塔，无塔型参数。
- 12 `UPGRADE_TOWER(tower_id,target_type)`。
- 13 `DOWNGRADE_TOWER(tower_id)`：降级或拆除 Basic 并按规则退款。
- 21–24：Lightning Storm、EMP、Deflector、Emergency Evasion，参数为 `(x,y)`。
- 31 `UPGRADE_GENERATION_SPEED`。
- 32 `UPGRADE_GENERATED_ANT`。
- 空列表在测量中记为 `HOLD`。

双方初始 50 金币，基础收入每两轮 3。下一座塔造价为
`15 * 3**(n//2)`，现有塔数 n 为奇数时再乘 2。两级塔升级花费 60/200，基地两类
升级花费 200/250；21–24 武器花费分别为 90/135/60/60。击杀、突破和降级退款
以冻结后端为准。

每个操作按列表顺序在“公开状态 + 同轮已接受前缀”上验证；地图边界、所属地块、
占用、塔所有权和升级树、同轮重复使用、EMP、金币、冷却和等级上限都参与合法性。

规则权威是内容哈希固定的官方 C++ 后端；公开 Python SDK 是候选接口权威。
