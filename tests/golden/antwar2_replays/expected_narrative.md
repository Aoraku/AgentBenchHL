# AntWar2 公开回放：fixture

有效终局胜者：P0。

## 原子时间线

- state_id=fixture:r0001:p0 | P0 在 (8,11) 建造基础塔；动作前金币=50，双方基地HP=(50, 50)。
- state_id=fixture:r0001:p1 | P1 未提交操作（HOLD）；动作前金币=50，双方基地HP=(50, 50)。
- state_id=fixture:r0028:p0 | P0 在 (16,9) 使用闪电风暴；动作前金币=35，双方基地HP=(50, 50)。
- state_id=fixture:r0028:p1 | P1 升级出兵速度；动作前金币=50，双方基地HP=(50, 50)。
- state_id=fixture:r0029:p0 | P0 未提交操作（HOLD）；动作前金币=10，双方基地HP=(50, 42)。
- state_id=fixture:r0029:p1 | P1 未提交操作（HOLD）；动作前金币=5，双方基地HP=(50, 42)。

## 有证据的汇总

- state_id=fixture:r0001:p0 | P0 的首次公开操作发生在第 1 轮：BUILD_TOWER。
- state_id=fixture:r0028:p0 | P0 的首次超级武器发生在第 28 轮：USE_LIGHTNING_STORM。
- state_id=fixture:r0028:p1 | P1 的首次公开操作发生在第 28 轮：UPGRADE_GENERATION_SPEED。
- state_id=fixture:r0028:p0 | 第 28 轮结算后 P1 基地损失 8 HP，剩余 42 HP。
- state_id=fixture:r0029:p0 | 第 29 轮结算后 P1 基地损失 42 HP，剩余 0 HP。
- state_id=fixture:r0029:p0 | 终局胜者为 P0，基地HP=(50.0, 0.0)。
