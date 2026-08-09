# Reproducibility

## 1. 固定依赖

- macOS 与 `/usr/bin/sandbox-exec`；
- Python 3.11 或更高版本；
- 支持 App Server JSON-RPC 且启用 `goals` 的 Codex CLI；
- C++17 工具链与 GNU Make；
- 含 AntWar2 官方后端和人类榜单的冻结 `Aoraku/AgentBench`；
- OpenAI-compatible Responses API key。

实验配置固定模型、reasoning effort、provider base URL、开发 seeds、决策 KL 的
epsilon 和无迭代上限。App Server 版本、生成的 JSON schema、公开 SDK、官方后端、
人类包和隐藏认证配置均写入 SHA-256 资源清单。

## 2. 凭据

```bash
cp .env.example .env
chmod 600 .env
```

`.env` 只包含：

```dotenv
ABHL_API_KEY=填入本次实验使用的key
AGENTBENCH_ROOT=/absolute/path/to/Aoraku/AgentBench
```

Framework 只把 key 传给 Codex App Server 的 provider 连接。Codex shell 环境策略显式
排除 key；候选、游戏和人类比赛进程使用白名单环境；事件构造器和 CLI 输出对凭据执行
拒绝或脱敏。

## 3. 离线与合同门槛

```bash
MPLCONFIGDIR=/tmp/abhl-mpl \
AGENTBENCH_ROOT=/absolute/path/to/Aoraku/AgentBench \
python -m pytest -q

python -m ruff check src tests
git diff --check
```

本机 Codex 协议验证不发起模型 turn：

```bash
ABHL_RUN_CODEX_PROTOCOL_TEST=1 \
python -m pytest tests/integration/test_codex_goal_runtime.py -m live -q
```

macOS 隔离验证覆盖两类进程：

```bash
ABHL_RUN_SEATBELT_TEST=1 \
python -m pytest tests/integration/test_live_preflight.py -m live -q
```

验证条件包括：Goal 可读活动候选但不能读人类源码；候选可读自身与公开 SDK，但不能
读取隐藏人类目录、写文件或建立网络连接。

## 4. From-scratch 初始化

```bash
abhl run init \
  --config configs/experiments/antwar2-goal-k1.yaml \
  --run-id antwar2-goal-k1
```

初始化执行以下原子阶段：

1. 审计并哈希冻结资源；
2. 创建只含公开 SDK、协议入口和 process support 的空策略工作区；
3. 创建独立 `HOME`、`CODEX_HOME` 和持久非 ephemeral thread；
4. 禁用跨 run Memory，并设置长期 Goal；
5. Goal 根据 GamePack 从零编写 `ai.py`；
6. 候选通过结构、双角色合法动作与隔离 smoke；
7. 候选封存为不可变 `v000`；
8. 与课程中最弱的可运行人类进行第一场完整官方比赛；
9. 生成公开 trace、自然语言 replay、双角色 Experience 和 checkpoint。

Smoke 是运行门槛，不作为正式迭代曲线的第零点。第一次 improvement act 会用固定
人类 measurement panel、P0/P1 和开发 seeds 对 `v000` 建立 iteration 0 指标。

## 5. 长程研究

单步监督模式：

```bash
abhl run resume \
  --config configs/experiments/antwar2-goal-k1.yaml \
  --run-id antwar2-goal-k1 \
  --acts 1
```

持续 Goal 模式：

```bash
abhl run pursue \
  --config configs/experiments/antwar2-goal-k1.yaml \
  --run-id antwar2-goal-k1
```

每个科研 act 包含：

1. 从 Champion、Frontier 或 Archive 选择一个明确 parent；
2. 向同一持久 Goal 提供冻结规则索引、相关 Experience、当前代码和目标公开回放；
3. Goal 从同一 parent 生成默认 `k=4` 个机制不同的策略包；每个策略包包含回放依据、
   协同代码修改、预期观察、风险和证伪标准；
4. Framework 验证并封存四个候选；
5. 四个候选在同一诊断对局条件下各执行一局正式比赛并返回可比较回放；
6. Goal 根据四份回放选择胜出或有显著改善的候选，再只对它追加确认局；
7. 对 rank01 的稳定晋升使用四个固定 seed 的角色均衡检查，而不是把这项成本施加给
   每一个探索候选；
8. 对目标回放形成正、负、混合或未触发 Experience；
9. 将满足全部门槛的候选晋升为 Champion；完整但失败的候选可成为 Frontier；
10. 在固定 measurement panel 上计算正式指标并刷新曲线；
11. 由最弱未击破人类向更强人类推进。

失败候选不会覆盖 Champion，也不会丢失。未完成基础设施结果不计科研胜负，并沿用相同
cache key 重试。中断后的恢复复用已封存候选、已完成比赛和未完成工作区，不重复付费生成。

## 6. 指标定义

- Behavioral IG：同一候选固定回放状态上，parent 与 candidate 的确定性原子行为使用
  epsilon-smoothed one-hot KL；只比较共享合法前缀，首次分歧后停止该决策的后续原子比较。
- Occupancy shift：两个策略在固定 measurement panel 上实际到达公共状态分布的 total
  variation distance。
- Elo：候选对固定人类池的完整胜负，以冻结榜单 score 作为人类锚点计算。
- Win rate：候选在固定 panel 的完整比赛平均得分。
- Score margin：候选终局基地生命值减去对手基地生命值。
- Token/time：App Server usage 通知中的 input、cached input、output、reasoning、total
  tokens 与每个 turn、评测阶段的 wall time。

所有正式曲线横轴均为整数 `research_iteration`。原始逐状态 KL trace、state ID、合法
原子支持和 occupancy hash 保存在事件账本中，可独立复算。

## 7. 认证与停止条件

公开开发课程全部击破后，Framework 对当前 Champion 执行隐藏冻结矩阵：所有可运行
人类 × P0/P1 × 认证 seeds。只有同一个 Champion 在每个完整 case 中获胜，运行才写入
`RunCompleted` 并由 `run pursue` 正常退出。

认证失败只向 Goal 暴露计数，不暴露隐藏 seed、case 或 replay。Framework 将公开课程
重新定位到最强人类继续研究。API 额度、provider 传输、进程中断或不完整比赛不会改变
已确认的科学状态；补充额度或修复外部依赖后执行同一命令即可恢复。

## 8. 审计

```bash
abhl run status --run-root ../runs/antwar2-goal-k1
abhl run audit --run-root ../runs/antwar2-goal-k1
```

`status` 报告整数 iteration、smoke/formal 比赛数、候选数、Experience 数、Champion、
Frontier 和可恢复状态。`audit` 检查 from-scratch origin、凭据泄漏、参考策略泄漏、完整
比赛、语义回放、Experience 与 checkpoint。
