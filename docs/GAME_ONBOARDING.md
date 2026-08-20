# 接入一个新游戏：完整清单

> 目标读者：要把 HL 迭代框架 broadcast 到新游戏的人（含 AI agent）。
> 前置阅读：[`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) —— 里面的 B、C、I 三条
> 直接决定新游戏的指标是否可信。

## 当前状态（2026-08-21）

| 游戏 | 人数 | 决策空间声明 | 精确探针 | measured_elo |
|---|---|---|---|---|
| antwar | 94 | opcode_alphabet(10) | ✅ | degree=24 已重测 |
| antwar2 | 229 | opcode_alphabet(10) | ✅ | 待重测 |
| generals | 81 | opcode_alphabet(8) | ❌ | degree=24 已重测 |
| rollman | 116 | enumerated(null) | ❌ | 重测中 |
| snakego | 123 | enumerated(6) | ❌ | 待重测 |
| lostspace | 133 | opcode_alphabet(7) | ❌ | 待重测 |
| aquawar | 194 | opcode_alphabet(6) | ❌ | 待重测 |
| miracle | 305 | opcode_alphabet(7) | ❌ | 待重测 |
| deepclue | 0 | 无声明 | 不可枚举 | 排除 |

**只有 antwar / antwar2 是真正跑通的**。其余游戏有池子和 evaluator，
但决策空间仍是近似口径、没有精确探针。

**deepclue 为什么排除**：动作是自由文本，决策空间不可枚举，
`measured_elo` 有 0 个有分选手。它需要另一套 IG 定义。

---

## 三仓分工

| 仓 | 角色 | 关键路径 |
|---|---|---|
| `AgentBench`（A） | 游戏资产**唯一权威源** | `games/<game>/` |
| `AgentBenchHL`（B） | harness / 迭代循环 / 指标 | `gamepacks/<game>/`、`src/agentbench_hl/adapters/<game>/` |
| `AgentBenchWeb` | 实验发起与可视化 | `app/main.py`、`app/jobs.py` |

**原则**：规则 / 决策空间 / 回放格式**只存在于 A**。B 的 gamepack 用
`@agentbench:` 引用，不留副本（`manifest.yaml` 记 sha256，`--check` 检测漂移）。

---

## 规则与精确定义的三层来源

按权威性排序：

1. **后端源码（最终权威）**：`backend_sources/corpus/<NN_game>/logic/`
   —— Saiblo 判题逻辑包解包。`game.yaml` 的 `backend_source` 指向它。
   **有歧义时一律以此为准。**
2. **官方文档存档**：`docs/saiblo-official/` —— 目前只有 antwar、antwar2 两份。
3. **提炼文档**：`games/<game>/rules.md` —— 人工从 1+2 提炼，给 agent 读。

### 关于 saiblo 抓取（重要）

- `www.saiblo.net` 是 Nuxt SSR 单页应用，游戏详情页**需要登录**；
- 唯一记录在案的可用抓取：`curl https://docs.saiblo.net/search/search_index.json`
  （866 段 / 42 页）；
- `experiments-data/saiblo_metadata.json`（1.1 MB）是平台 Game 表导出，
  含 `judger_config` 与 `introduction` 全文 —— **这是唯一留下的平台元数据快照**；
- **抓取链路已断**：`merge_population_pool.py`、`build_player_submissions.py`
  在 README 里被引用但文件已不存在；原始包 `archives/` 与 `SHA256SUMS`
  只存在于本地采集环境。

**结论**：接新游戏时不要指望复用抓取脚本。直接从 judge-dev 逻辑包拿，
或重建抓取。

---

## 决策空间：这是主要工作量

`games/<game>/decision_space.yaml` 分两段：

- **上半段**给人/agent 读：`policy_interface`、`atomic_actions`、`legal_actions`；
- **下半段 `information_gain:`** 给机器读，写错直接抛 `DecisionSpaceError`。

`support` 有两种口径：

| mode | 含义 | 问题 |
|---|---|---|
| `opcode_alphabet` | 把 \|A(s)\| 当**常量**（如 10） | 真实合法动作数是**状态相关**的。antwar 实测中位 **246**，用 10 算 IG 系统性偏低 25% |
| `enumerated` | 逐状态枚举真实 \|A(s)\| | 正确口径 |

**新游戏直接上 `enumerated`，不要走 `opcode_alphabet` 的弯路。**

### 写精确探针

位置：`src/agentbench_hl/adapters/<game>/policy_trace_worker.py`

核心是用**官方合法性判定**枚举支持集，并且——见
[LESSONS_LEARNED B 条](LESSONS_LEARNED.md#b-探针被候选包-sdk-的边界-bug-掀翻)——
把"判定自身抛异常"按非法处理：

```python
def _is_legal(state, player, operation) -> bool:
    """官方合法性判定，且判定自身崩溃时算非法。

    候选包自带的 SDK 在边界状态上会抛异常而不是返回 False
    （antwar: 满级时 [200,250][2] 越界）。枚举支持集要把每个候选动作都问一遍，
    所以一个格子的崩溃会掀翻整局探针，让这一轮静默退回近似口径。
    """
    try:
        return bool(state.is_operation_valid(player, operation))
    except Exception:
        return False
```

还要注意：

- **`HOLD` 必须在支持集里**。一个回合提交 0 个操作永远合法，
  漏掉它会让 \|A(s)\| 系统性偏小 1；
- 合法性判定要用**纯检查**接口（`is_operation_valid` / `can_apply_operation`，
  内部是 `apply_operation(dry_run=True)`），不能改状态。

### 验证探针

```bash
python scripts/recompute_behavioral_ig.py <run_root>
```

**精确支撑集覆盖率低于 90% 就说明有 SDK 边界问题**，
不要接受"反正有近似兜底"。

---

## 完整步骤

### 1. A 仓：游戏资产

```
games/<game>/
├── game.yaml              # backend_source 指向 backend_sources/corpus/
├── rules.md               # 从后端逻辑提炼
├── decision_space.yaml    # support.mode: enumerated
├── replay_format.md       # 回放字段说明
├── replay_skill.md        # 怎么读回放（给 agent）
├── plugin.py
├── evaluator/
│   ├── runtime.py         # 编译候选
│   ├── arena.py           # 跑协议
│   └── narrate.py         # 回放 → 自然语言
└── players/               # 选手池
```

**`MatchResult` 必须提供 `score_margin`** —— 见
[LESSONS_LEARNED G 条](LESSONS_LEARNED.md#g-二值胜率丢掉全部梯度)。
没有连续量的游戏会退化成真正的二值反馈，全败轮完全没有梯度。

### 2. 选手池审计

```bash
abhl pool audit <game> --agentbench-root $AGENTBENCH_ROOT --all
```

产出 `players/runnable.json`。池子里大部分选手跑不起来是正常的
（antwar 池 644 → 可运行 122）。

### 3. 人类池实测 Elo

```bash
abhl ladder eval <game> --agentbench-root $AGENTBENCH_ROOT \
    --degree 28 --parallel 20 --cpus-per-match 3
```

**`--degree` 不要用默认的 6。** 见
[LESSONS_LEARNED I 条](LESSONS_LEARNED.md#i-人类池样本量撑不起排第几)：
degree=6 → 标准误 ±106，而前十相邻分差只有 5~21，**名次顺序不可辨识**。

反推公式：`degree_new = degree_old × (se_old / se_target)²`。

批量重测所有游戏：

```bash
python scripts/remeasure_all_pools.py \
    --abhl $VENV/bin/abhl --agentbench-root $AGENTBENCH_ROOT \
    --degree 28 --parallel 20 --log-dir runs/ladder-remeasure
```

### 4. 验证池子可信

```bash
python scripts/audit_pool_elo.py \
    --measured $AGENTBENCH_ROOT/games/<game>/players/measured_elo.json \
    --target-se 50
```

**通过判据**：

- 标准误中位 < 前十相邻分差中位（否则名次不可辨识，脚本会打 ⚠）；
- 胜率 = 1.000 的选手数尽量为 0（饱和 = BT 只能给下界）。

实测效果：重测后 antwar 饱和 3 人 → **0 人**。

### 5. B 仓：GamePack

```bash
AGENTBENCH_ROOT=<path> python scripts/gen_gamepacks.py --game <game>
```

产出：

```
gamepacks/<game>/
├── GOAL_CHARTER.md        # 给 agent 的任务章程
├── manifest.yaml          # @agentbench: 引用 + source_digests
├── sdk_interface.md       # 候选包要实现什么接口
└── candidate_support/     # 注入候选的辅助库（含 selfcheck）
```

检测 A 侧漂移：

```bash
python scripts/gen_gamepacks.py --check
```

### 6. B 仓：精确探针

见上文「写精确探针」。写完后跑
`scripts/verify_behavioral_ig.py` 与 `scripts/recompute_behavioral_ig.py` 验证覆盖率。

### 7. 实验配置

复制 `configs/experiments/exp2-antwar-conquest.yaml` 改 `game`。
**四个必须保留的设置**（每个都对应一条事故）：

```yaml
runtime:
  thread_rotate_each_iteration: true    # A 条：躲开 codex×glm 的压缩不兼容
provider:
  model_catalog: zhipu                  # A 条：让 codex 认识非自家模型
  auto_compact_token_limit: 900000       # A 条：压缩线设到够不着
budget:
  tokens: null                          # 爬梯没有"跑够就够了"的意思
```

### 8. 起跑

```bash
abhl goal-led run --config configs/experiments/<game>.yaml --run-id <run-id>
```

### 9. 后台全池评测

```bash
python scripts/pool_elo_worker.py \
    --run-root runs/<run-id> --agentbench-root $AGENTBENCH_ROOT \
    --game <game> --parallel 10 --cpus-per-match 3 --headroom 6
```

CPU 空闲时自动给每个中间版本打全池 Elo，不与主迭代抢核
（load 超过 `总核数 - headroom` 就暂停派发）。

查看进度：

```bash
python scripts/pool_elo_status.py runs/<run-id>
```

### 10. 出图

```bash
python scripts/plot_learning_curves.py \
    --run-dir runs/<run-id> --out-dir analysis/<name> \
    --require-evaluated 10
```

`--require-evaluated` 是护栏：全池评测覆盖不足就拒绝出图。见
[LESSONS_LEARNED K 条](LESSONS_LEARNED.md#k-用半份数据出图)。

---

## 验收清单

接入完成的判据（缺一条都不算通过）：

- [ ] `rules.md` 与后端逻辑一致（有歧义处以后端为准）
- [ ] `decision_space.yaml` 的 `support.mode` 是 `enumerated`
- [ ] 精确支撑集覆盖率 ≥ 90%
- [ ] `MatchResult` 提供 `score_margin`
- [ ] `measured_elo.json` 的 degree ≥ 28
- [ ] `audit_pool_elo.py` 无 ⚠ 警告（标准误 < 前十相邻分差）
- [ ] 胜率饱和选手数 ≈ 0
- [ ] `gen_gamepacks.py --check` 无漂移
- [ ] 配置含 A 条的四个设置
- [ ] 跑通 3 轮迭代且 `behavioral_ig` 非 null
- [ ] 后台队列能产出至少 1 个版本的全池 Elo

---

## 常见故障速查

| 症状 | 可能原因 | 见 |
|---|---|---|
| run 死于 `remote compaction fatal error` | 模型返回多个 output item | [A](LESSONS_LEARNED.md#a-codex-remote-compaction-与-glm-不兼容) |
| `behavioral_ig` 一直是近似值 | 探针被 SDK 边界 bug 掀翻 | [B](LESSONS_LEARNED.md#b-探针被候选包-sdk-的边界-bug-掀翻) |
| token 曲线呈阶梯状 | 跨 thread 取了全局 max | [C](LESSONS_LEARNED.md#c-token-计量在会话轮转下低估-29-倍) |
| `conflicting idempotency key` | 幂等键用了内存下标 | [D](LESSONS_LEARNED.md#d-幂等键冲突打死-run) |
| run 死于 `IncompleteRead` | 读响应体没进重试 | [E](LESSONS_LEARNED.md#e-上游重试覆盖不全) |
| 整局 IG 数据被丢弃 | 探针与线协议计数差 1 | [F](LESSONS_LEARNED.md#f-探针计数差-1-导致整局丢弃) |
| 全败轮 agent 拿不到方向 | 只报胜率不报分差 | [G](LESSONS_LEARNED.md#g-二值胜率丢掉全部梯度) |
| 全池评测慢得离谱 | 按版本串行 | [H](LESSONS_LEARNED.md#h-池评测被按版本串行) |
| "排第几"结论不稳 | degree 太小 | [I](LESSONS_LEARNED.md#i-人类池样本量撑不起排第几) |
| Elo 忽高忽低 | 混用了两种 Elo 口径 | [J](LESSONS_LEARNED.md#j-两个-elo-口径不可混用) |
