# AgentBenchWeb 待办清单

> 给维护 HL 迭代页面的同学。
>
> **本文档描述的是"HL 框架已经具备、Web 侧需要接上"的能力。**
> 不着急做——B 仓（AgentBenchHL）先把接口稳定下来，Web 侧照着接即可。
>
> 接口细节见 [`HL_WEB_API.md`](HL_WEB_API.md)；踩过的坑见
> [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md)。

## 页面要提供的选项

用户在 `/hl` 发起一次实验时，需要能选：

| 选项 | 字段 | 取值 |
|---|---|---|
| 游戏 | `game` | 见下「游戏可用状态」 |
| 模型 | `model` | 见下「模型白名单」 |
| Harness | `harness` | `codex` / `cc` |
| 每轮 rollout 数 k | `rollout_k` | 整数，默认 4 |
| 能否看到自己的历史 | `history_mode` | `full` / `no_notes` / `memoryless` |
| 对手选择策略 | `opponent_policy` | 见下「对手策略」 |

---

## P0：不补就会打死 run

这两项**必须**在 `app/jobs.py::hl_config_document()` 里补上，否则非 OpenAI 模型
的 run 会在上下文压缩点直接死掉（实测两个 40+ 轮的 run 都是这样没的，
详见 [LESSONS_LEARNED A 条](LESSONS_LEARNED.md#a-codex-remote-compaction-与-glm-不兼容)）。

### 1. `runtime.thread_rotate_each_iteration: true`

codex 0.147 的 remote compaction 要求响应里恰好一个 compaction output item，
而 glm 系列返回 `[reasoning, message]` 两个。只要上下文涨到压缩线，压缩必死。
每轮换新 thread 可以让上下文永远到不了压缩线。

### 2. `provider.model_catalog`

不给这个字段时 codex 会用 fallback 元数据，压缩在一个我们没设过的点触发。
需要一张 `model → catalog` 映射表：

```python
MODEL_CATALOG = {
    "glm-5.3": "zhipu",
    # gpt-* 系列不需要（codex 自家模型，自带元数据）
    # 其余非 OpenAI 模型接入前必须确认厂商是否提供 models.json
}
```

同时建议一起补 `provider.auto_compact_token_limit: 900000`（压缩线设到够不着）。

---

## P1：功能缺口

### 3. 更新模型白名单

`app/config.py` 的 `MODEL_GROUPS` 需要与下面这 9 个对齐。**只保留这 9 个**，
多余的删掉（消融维度已确定，额外模型只会让实验矩阵失控）：

| 模型 | 白名单现状 |
|---|---|
| fable 5 | ❌ 缺 |
| opus 5 | ❌ 缺 |
| gpt-5.6-sol | ✅ 有 |
| gpt-5.6-luna | ✅ 有（真实 id `gpt-5.6-luna-2026-07-09`） |
| glm-5.3 | ❌ 只有 glm-5.2 / 5.1 / 5 |
| kimi k3 | ✅ 有（`k3`） |
| deepseek v4 | ❌ 缺 |
| qwen3.8 | ❌ 缺 |
| LongCat-2.0 | ✅ 有 |

**必须先用中转站 `/v1/models` 核实名称再写进白名单** —— 写一个不存在的名字
会在 run 起来之后才失败，浪费一轮机时。

### 4. 补 2 种对手策略

B 仓 `application/opponent_policy.py` 已实现 7 种，还缺 2 种。
**只做这 2 个，不要新增别的策略**：

| 需要新增 | 语义 | 实现要点 |
|---|---|---|
| `k_random` | k 个 rollout 随机抽 k 个**不同**人打 | `select()` 返回长度 k 的随机不重复元组。与已有的 `k_diverse` 区别是不分层（`k_diverse` 会刻意覆盖强/中/弱） |
| `k_window_promote` | k 个 rollout 打 `(j+k, j]` 名，**哪个打过了就把那个往前提** | 需要维护滑动窗口 + 已征服集合；晋级判据复用 `conquest.py` |

已实现的 7 种（Web 侧直接可用）：

| 值 | 语义 |
|---|---|
| `fixed_top` | 固定打榜首 |
| `fixed_rank` | 固定打第 N 名（配 `opponent_rank`） |
| `ladder_up` | 从第 j 名**往上**逐个征服（配 `opponent_start_rank`） |
| `ladder_down` | 从第 i 名**往下**逐个征服 |
| `random` | 每轮随机一个对手 |
| `k_diverse` | k 个候选分层打不同对手 |
| `self_decide` | 主 agent 读排行榜自主决定打谁 |

**关于"让 LLM 决定打谁"**：这就是 `self_decide`，不需要单独的策略。
决定者是主 agent —— 它在 `action.json` 里写明本轮要打谁，
框架把规则、迭代历史、排行榜、当前战况都已经放进它的上下文了。
不要再实现一个"独立调一次模型只做选敌决策"的分支。

### 5. 能力曲线改用静态池 Elo

现在页面用逐轮指标里的 `pool_elo` 画能力曲线，这是**错的口径**。

| 口径 | 来源 | 含义 |
|---|---|---|
| `pool_elo` | `IterationMetricsFinalized` | 该 run **迄今全部**对局的累计 BT-MLE。含早期连败与被证伪的探索候选，天然平滑，**不是当前实力** |
| 静态池 Elo | `runs/<id>/pool-elo/index.json` | 单个版本独立打完冻结人类池（antwar 188 局）。唯一能回答"这一版有多强" |

实测差距（同一批版本，2 局估计 vs 188 局实测）：

| 版本 | 2 局估计 | 188 局实测 |
|---|---|---|
| v039 | 1183 / #1 | 1077.95 / #5 |
| v040 | 992 / #7 | **750.00 / #22** |

`pool-elo/index.json` 结构：

```json
{
  "schema": "pool-elo-index/v1",
  "pool": {"size": 94, "fingerprint": "088bf7c4...", "frozen": true},
  "versions": [
    {"candidate_id": "v039...", "iteration": 40, "elo": 1077.95,
     "pool_rank": 5, "win_rate": 0.899, "complete_matches": 188,
     "comparable": true}
  ],
  "peak": {"...": "..."}
}
```

⚠️ `comparable: false` 表示该结果是在**旧池指纹**下测的，与当前尺子不可比
（人类池重测后会出现）。前端必须过滤掉或明确标注，不能混进峰值计算。

没有静态池数据时**宁可不画**，不要用 `pool_elo` 冒充能力曲线。

---

## P2：体验改进

### 6. 标注 IG 口径

不是所有游戏的 behavioral IG 都是精确的。前端应该把口径显示出来：

| 游戏 | 决策空间口径 | 说明 |
|---|---|---|
| antwar / antwar2 | ✅ 精确枚举 | 逐状态 dry-run |
| rollman | ✅ 精确 | 动作空间静态（5 / 125），撞墙不算非法 |
| generals | ✅ 精确 | 深拷贝试执行，支持集 8~23 |
| miracle | ⚠️ 近似偏大 | 法力值约束无法从回放推出，召唤/神器按"手牌里有就算合法"计入 |
| aquawar / lostspace / snakego | ❌ 字母表近似 | 尚未写精确探针，IG 系统性偏低约 25% |
| deepclue | ❌ 不可用 | 动作是自由文本，决策空间不可枚举 |

判据在逐轮指标的 `behavioral_ig_support_mode` 字段：
`exact_enumeration` 是精确，`opcode_alphabet` 是近似。

### 7. 展示后台评测队列进度

全池 Elo 评测是独立的运维进程（`scripts/pool_elo_worker.py`），
不由 Web 触发。但可以读 `runs/<id>/pool-elo/worker-status.json` 展示进度：

```json
{"discovered": 172, "pending": 169, "load_average": 13.1, "cpu_count": 32}
```

### 8. 游戏可用状态分组

建议按接入完成度分两组展示，把未完成的标出来：

**已跑通**：antwar、antwar2、rollman、generals、miracle
**池子可用但 IG 是近似口径**：aquawar、lostspace、snakego
**不可用**：deepclue

判据可以读 A 仓的 `games/<game>/decision_space.yaml` 里
`information_gain.support.mode`，以及 B 仓
`src/agentbench_hl/adapters/<game>/policy_trace_worker.py` 是否存在。

---

## 不要动的参数

这四个是踩坑换来的，改了会直接打死 run 或让指标失去意义：

```yaml
runtime:
  thread_rotate_each_iteration: true    # 躲开 codex×glm 压缩不兼容
provider:
  model_catalog: <厂商>                  # 让 codex 认识非自家模型
  auto_compact_token_limit: 900000       # 压缩线设到够不着
budget:
  tokens: null                          # 爬梯没有"跑够就够了"的意思
```
