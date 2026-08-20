# HL 迭代页面：接口文档

> 给维护 `AgentBenchWeb` 的同学。本文档描述 HL（Human-Level 迭代）实验的
> 发起接口、参数语义、以及**当前缺口**。
>
> 相关：[`LESSONS_LEARNED.md`](LESSONS_LEARNED.md)（为什么某些参数必须固定）、
> [`GAME_ONBOARDING.md`](GAME_ONBOARDING.md)（游戏接入状态）。

## 1. 系统架构

```
浏览器
  │  POST /api/hl/start   (multipart form)
  ▼
AgentBenchWeb (FastAPI)
  │  ① 校验参数
  │  ② hl_config_document()  表单 → B 的 v1.1 YAML
  │  ③ write_hl_config()     落到 B 仓 configs/experiments/web-<run_id>.yaml
  │  ④ 起子进程：abhl goal-led run --config <那份 yaml> --run-id <run_id>
  ▼
AgentBenchHL (abhl)
  │  写 runs/<run_id>/events.jsonl（唯一真相源）
  ▼
AgentBenchWeb scheduler
     ingest_hl_events() 增量解析 → SQLite hl_iters 表 → 前端轮询/SSE
```

**关键约束**：

- 配置**必须**落在 B 仓的 `configs/experiments/` 下（B 由此定位 gamepacks）；
- `events.jsonl` 是唯一真相源，Web 侧的 SQLite 只是缓存，可随时重建；
- Web 侧**不复制**三仓逻辑：对局走 B 的 arena，隔离走 B 的
  `agentbench_hl.ports.isolation`。

## 2. 发起实验

### `POST /api/hl/start`

`Content-Type: multipart/form-data`，需要 submit 权限（`Depends(require_submit)`）。

代码位置：`AgentBenchWeb/app/main.py:425`，
表单 → YAML 的翻译在 `AgentBenchWeb/app/jobs.py:227` (`hl_config_document`)。

#### 2.1 游戏

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `game` | str | 必填 | 游戏 slug |

**服务端校验**：`load_game(game)` 存在；`game_info.hl_ready`（有 GamePack）；
`game_info.players_runnable > 0`。

**当前可选值**：`antwar`、`antwar2` 真正跑通；`generals`、`rollman`、`snakego`、
`lostspace`、`aquawar`、`miracle` 有池子但**决策空间仍是近似口径**（IG 会偏低约 25%）；
`deepclue` 不可用（动作是自由文本，0 个有分选手）。

建议前端按 `hl_ready` + 探针状态分两组展示，把"IG 口径为近似"标出来。

#### 2.2 模型

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `model` | str | 必填 | 必须在 `MODELS` 白名单内 |
| `reasoning_effort` | str | `xhigh` | `low` / `medium` / `high` / `xhigh` |

**白名单定义**：`AgentBenchWeb/app/config.py:108` 的 `MODEL_GROUPS`。

⚠️ **当前白名单与你要的 9 个模型不一致**，需要更新：

| 你要的 | 白名单现状 |
|---|---|
| fable 5 | ❌ 缺 |
| opus 5 | ❌ 缺 |
| gpt-5.6-sol | ✅ 有 |
| gpt-5.6-luna | ✅ 有 |
| glm-5.3 | ❌ 只有 glm-5.2 / 5.1 / 5 |
| kimi k3 | ✅ 有（`k3`） |
| deepseek v4 | ❌ 缺 |
| qwen3.8 | ❌ 缺 |
| LongCat-2.0 | ✅ 有 |

**更新方法**：改 `MODEL_GROUPS`，但**必须先用中转站 `/v1/models` 核实名称**
——白名单的注释写明它是"中转站实测可用的模型"，写一个不存在的名字会在
run 起来后才失败。

**还有一个隐藏依赖**（见 [LESSONS_LEARNED A 条](LESSONS_LEARNED.md#a-codex-remote-compaction-与-glm-不兼容)）：
非 OpenAI 模型需要配 `provider.model_catalog`，否则 codex 用 fallback 元数据、
压缩会在一个没设过的点触发并**打死整个 run**。目前 `hl_config_document()`
**没有生成这个字段**，需要补一张 `model → catalog` 映射表：

```python
MODEL_CATALOG = {
    "glm-5.3": "zhipu",
    "glm-5.2": "zhipu",
    # gpt-* 系列不需要（codex 自家模型）
}
```

#### 2.3 Harness

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `harness` | str | `codex` | `codex` / `cc` |

- `codex` —— Codex App Server（Goal 模式）
- `cc` —— Claude Code（headless，经 anthropic→openai 桥接）

选 `cc` 时 `hl_config_document()` 会把 `runtime.agent_binary` 设成
`SETTINGS.claude_binary`。

**注意**：`cc` 路径的实测覆盖远少于 codex，两个 SOTA run 都是 codex 跑的。

#### 2.4 Rollout 数 k

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `rollout_k` | int | 4 | 每轮候选数 |

语义：每轮 agent 写 k 个**不同**的候选策略（框架强制多样性检查：
pairwise 行差 + method_surface + base_classes），每个候选跑
`roles × development_seeds` 局。

**成本**：每轮对局数 = `k × 对手数 × 座次数 × seed 数`。
默认配置（k=4、1 对手、2 座次、2 seed）= 16 局/轮。

#### 2.5 历史可见性

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `history_mode` | str | `full` | `full` / `no_notes` / `memoryless` |

这就是你说的"是否能看到自己的历史"，是实验 4 的消融维度：

| 值 | 语义 |
|---|---|
| `full` | 能看到全部历史（迭代记录 + 自己写的经验笔记 EXPERIENCE.md） |
| `no_notes` | 能看到迭代记录，但看不到自己的经验笔记 |
| `memoryless` | 每轮从零开始（只断 harness 的对话记忆，磁盘上的历史文件不动） |

#### 2.6 对手策略

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `opponent_policy` | str | `fixed_top` | 见下表 |
| `opponent_rank` | int | 0 | `fixed_rank` 时必填 |
| `opponent_start_rank` | int | 0 | `ladder_*` 的起点 |
| `ladder_scope` | str | `auto` | `auto` / `official` / `reference` / `measured` |
| `advance_min_matches` | int | 2 | 晋级判据：本轮最少局数 |
| `advance_win_rate` | float | 0.6 | 晋级判据：得分率下限 |
| `advance_streak` | int | 1 | 晋级判据：连续达标轮数 |

**已实现的 7 种**（`AgentBenchHL/src/agentbench_hl/application/opponent_policy.py`）：

| 值 | 语义 | 你的需求对应 |
|---|---|---|
| `fixed_top` | 固定打榜首 | 「固定打第 i 名」的 i=1 特例 |
| `fixed_rank` | 固定打第 N 名（`opponent_rank`） | ✅ 固定打第 i 名 |
| `ladder_up` | 从第 `start_rank` 名**往上**逐个征服 | ✅ 从第 j 名开始逐个向上打 |
| `ladder_down` | 从第 `start_rank` 名**往下**逐个征服 | ✅ 从第 i 名开始逐个往后打 |
| `random` | 每轮随机一个对手 | ✅ 每次迭代随机抽一个人打 |
| `k_diverse` | k 个候选**分层**打不同对手（覆盖强中弱） | ≈ k 个 rollout 打 k 个不同人（但是分层而非随机） |
| `self_decide` | 模型读排行榜自主选 | 部分对应「让 LLM 决定」 |

⚠️ **缺 3 种**，需要在 B 仓 `opponent_policy.py` 里新增
（`build_policy()` 加分支 + 实现 `OpponentPolicy` 协议）：

| 需要新增 | 语义 | 实现要点 |
|---|---|---|
| `k_random` | k 个 rollout **随机**抽 k 个不同人 | `select()` 返回长度 k 的随机不重复元组。与 `k_diverse` 的区别是不分层 |
| `k_window_promote` | k 个 rollout 打 `(j+k, j]` 名，**打过了就把那个往前提** | 需要维护一个滑动窗口 + 已征服集合；晋级判据复用 `conquest.py` |
| `llm_decide` | 单独调模型，输入规则+迭代历史+排行榜+状况，返回 k 个 int | 与 `self_decide` **不同**：`self_decide` 是主 agent 在 action.json 里顺便写，这个是**独立一次 LLM 调用**、只做选敌决策、返回结构化 int 数组。需要新的 provider 调用路径 |

`OpponentPolicy` 协议（必须实现的三个方法）：

```python
class OpponentPolicy(Protocol):
    name: str

    def select(self, *, iteration: int, k: int, cleared: int) -> tuple[str, ...]:
        """返回本轮对手；长度 1（全部候选打同一对手）或 k（每候选一个）。
        返回空元组 = 不干预，由 Goal 在 action.json 里决定。"""

    def instruction(self, *, iteration: int, k: int, cleared: int) -> str:
        """注入给 Goal 的自然语言说明。"""

    def target_sequence(self) -> tuple[str, ...]:
        """有序课程的目标序列；非顺序策略返回空元组。"""
```

**`ladder_scope` 为什么默认 `auto`**：官方榜覆盖太窄（antwar 只有 20 人有官方 Elo，
而可运行的有 174 人、我们自己实测覆盖 94 人）。`official` 口径下对手梯度会莫名塌掉。
`auto` = 逐选手回落 `measured` → `reference` → `official`。

#### 2.7 其余参数

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `max_iters` | int | 5 | 迭代轮数上限 |
| `token_budget` | int | 0 | 0 = 不限 |
| `wall_budget` | int | 0 | 墙钟秒数，0 = 不限 |
| `match_parallelism` | int | 0 | 0 = 用默认 |
| `seed_mode` | str | `fixed` | `fixed`（单 seed）/ `generalize`（4 seed） |
| `code_constraint` | str | `any` | `any` / `if_else` |
| `experience_skills` | int | 1 | 是否允许写经验笔记 |
| `iteration_mode` | str | `lockstep` | `lockstep` / `goal_autonomous` |
| `information_gain` | int | 1 | 是否开 IG 测量 |
| `rival_code_visible` | int | 0 | 对手代码是否可见 |
| `goal_prompt` | str | "" | prompt 覆盖，空 = 用默认 |
| `note` | str | "" | 备注 |

## 3. 查询接口

| 端点 | 方法 | 权限 | 说明 |
|---|---|---|---|
| `/api/hl/{run_id}/iters` | GET | 公开 | 逐轮指标（从 `hl_iters` 表读） |
| `/api/hl/{run_id}/cancel` | POST | admin | 取消 run |
| `/api/admin/hl-grid` | POST | admin | 批量起一组实验（网格搜索） |
| `/api/admin/overview` | GET | admin | 全局概览 |
| `/api/stream` | GET | 公开 | SSE 事件流 |
| `/healthz` | GET | 公开 | 健康检查 |

页面路由：`/hl`（列表）、`/hl/{run_id}`（详情）、`/ladder`（人类池榜单）。

## 4. 逐轮指标字段

`ingest_hl_events()` 解析 `events.jsonl` 里的 `IterationMetricsFinalized`。
主要字段：

| 字段 | 说明 |
|---|---|
| `research_iteration` | 轮次 |
| `trajectories_seen` | agent 看过的完整对局轨迹数（第二个横坐标） |
| `win_rate` | 本轮 k 个候选对**当轮对手**的总胜率 |
| `pool_elo` | 该 run **迄今全部**对局的累计 BT-MLE |
| `behavioral_ig` | 行为信息增益（nats） |
| `total_tokens` | 累计 token |
| `best_candidate_id` | 本轮被选中的候选 |
| `margin_mean` / `margin_best` / `margin_by_candidate` | 分差（连续奖励） |
| `opponent_ids` | 本轮对手 |

### ⚠️ 展示时必须区分两种 Elo

见 [LESSONS_LEARNED J 条](LESSONS_LEARNED.md#j-两个-elo-口径不可混用)。

| 口径 | 来源 | 含义 | 陷阱 |
|---|---|---|---|
| `pool_elo` | 逐轮指标 | 整条轨迹的平均位置 | **不是当前实力**。含早期连败与被证伪的探索候选；第 42 轮时分母 300+ 局，天然平滑 |
| 静态池 Elo | `runs/<id>/pool-elo/index.json` | 单个版本独立打全池 | 唯一能回答"这一版有多强" |

实测差距（同批版本，2 局估计 vs 188 局实测）：

| 版本 | 2 局估计 | 188 局实测 |
|---|---|---|
| v039 | 1183 / #1 | 1077.95 / #5 |
| v040 | 992 / #7 | **750.00 / #22** |

**前端建议**：主曲线用静态池 Elo（`pool-elo/index.json`），
`pool_elo` 单独标注为"累计口径"。没有静态池数据时**宁可不画**，
不要用 `pool_elo` 冒充能力曲线。

`pool-elo/index.json` 结构：

```json
{
  "schema": "pool-elo-index/v1",
  "pool": {"size": 94, "fingerprint": "a1b2c3...", "frozen": true},
  "versions": [
    {"candidate_id": "v039...", "iteration": 40, "elo": 1077.95,
     "pool_rank": 5, "win_rate": 0.899, "complete_matches": 188,
     "comparable": true}
  ],
  "peak": {...}
}
```

`comparable: false` 表示该结果是在**旧池指纹**下测的，与当前尺子不可比
（人类池重测后会出现）。前端必须过滤掉或明确标注。

## 5. 后台全池评测

不由 Web 触发，是独立的运维进程：

```bash
python scripts/pool_elo_worker.py \
    --run-root runs/<run-id> --agentbench-root $AGENTBENCH_ROOT \
    --game <game> --parallel 10 --cpus-per-match 3 --headroom 6
```

CPU 空闲时自动给每个中间版本打全池 Elo（`--headroom` 指定给主迭代预留的核数，
load 超过 `总核数 - headroom` 就暂停派发）。

**Web 侧可以做的**：读 `runs/<id>/pool-elo/worker-status.json` 展示队列进度：

```json
{"discovered": 42, "pending": 39, "load_average": 13.1, "cpu_count": 32}
```

## 6. 不要动的参数

这四个是踩坑换来的，改了会直接打死 run
（见 [LESSONS_LEARNED A 条](LESSONS_LEARNED.md#a-codex-remote-compaction-与-glm-不兼容)、
[C 条](LESSONS_LEARNED.md#c-token-计量在会话轮转下低估-29-倍)）：

```yaml
runtime:
  thread_rotate_each_iteration: true    # 躲开 codex×glm 压缩不兼容
provider:
  model_catalog: zhipu                  # 让 codex 认识非自家模型
  auto_compact_token_limit: 900000       # 压缩线设到够不着
budget:
  tokens: null                          # 爬梯没有"跑够就够了"
```

⚠️ 当前 `hl_config_document()` **缺前三个**，需要补。

## 7. Web 侧待办清单

按优先级：

1. **补 `model_catalog` 字段**（P0）—— 不补则非 OpenAI 模型的 run 会在压缩点死掉
2. **补 `thread_rotate_each_iteration: true`**（P0）—— 同上
3. **更新 `MODEL_GROUPS`**（P1）—— 加 fable 5 / opus 5 / glm-5.3 / deepseek v4 /
   qwen3.8，先用中转站 `/v1/models` 核实名称
4. **新增 3 种对手策略**（P1）—— `k_random` / `k_window_promote` / `llm_decide`，
   需要先在 B 仓实现再加到 `OPPONENT_POLICIES`
5. **静态池 Elo 曲线**（P1）—— 读 `pool-elo/index.json`，替换现在用
   `pool_elo` 画的能力曲线
6. **展示 IG 口径**（P2）—— 近似口径的游戏要标出来
7. **队列进度展示**（P2）—— 读 `worker-status.json`
