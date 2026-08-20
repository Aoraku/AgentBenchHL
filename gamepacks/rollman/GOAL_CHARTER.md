# rollman · Goal 章程

## 唯一目标

在 rollman 上刷出 SOTA：相对人类选手池取得尽可能高的 Elo 与胜率。

## 你能看到

- `gamepack/rules.md`：规则（A 的权威版本）
- `gamepack/decision_space.yaml`：决策空间与合法动作语义
- `gamepack/replay_format.md` + `gamepack/replay_skill.md`：回放字段与阅读方法
- `leaderboard.json`：人类排行榜（对手 id / rank / Elo）
- `feedback/<request_id>/`：你自己每轮对局的回放与结果
- `research/`：你自己写的迭代经验
- 你自己历次候选代码（`.agentbench/rollouts/` 与 run 的 `snapshots/`）

## 你看不到

- 任何对手的源码（除非本轮实验显式开启消融）
- 认证/评测矩阵、参考策略、其它 run 的记忆
- 互联网

## 工作循环

0. 第 0 轮：只读规则与协议，写出**格式绝对正确**的裸策略 v000。
1. 每轮产出 k 个有机制差异的候选，写 `.agentbench/action.json` 请求官方对局。
2. 读回放 → 更新经验 → 改策略 → 下一轮。
3. 每次改动都要有回放证据支撑，避免无根据的大改写。
