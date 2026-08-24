# AgentBenchHL 使用手册

给要做**多模型 × 多游戏**测试的同事。读完你应该能独立起一批实验、看懂进度、
出图，并且知道哪些地方会静默出错。

---

## 0. 三十秒上手

```bash
cd ~/agentbench/AgentBenchHL
V=~/agentbench/.venv/bin/python

# 1) 生成配置（不要手写 yaml，理由见 §3）。默认形态 k=4/b=1，一轮 8 局
$V scripts/make_verify_configs.py --out-dir configs/experiments/mine \
  --games antwar2 --model glm-5.2 --iterations 32 \
  --rollout-k 4 --batch 1 --start-rank 20 --prefix mine

# 2) 起 run（第三个参数是"本次再跑多少轮"）
bash scripts/run_hl.sh configs/experiments/mine/mine-glm-5.2-antwar2.yaml my-run 32

# 3) 看进度
$V scripts/watch_runs.py --runs-root runs --log-dir ~/agentbench/runs \
  --target 32 --run-id my-run

# 4) 核对链路真的通了（每轮 k 个候选、代码互不相同、对手名次对）
$V scripts/audit_iteration_shape.py --runs-root runs --expect-k 4 --run-id my-run

# 5) ★ 迭代跑完之后测终局 Elo。慢评测默认是关的，不做这一步就没有绝对标尺
bash scripts/final_elo.sh antwar2 my-run

# 6) 出图（等 5) 跑完再出，否则只有快通道那条线）
$V scripts/plot_learning_curves.py --run-dir runs/my-run \
  --out-dir ~/agentbench/analysis/mine
```

第 5 步不能省：迭代过程里的胜率衡量的是**对手难度**，不是绝对强度
（`progress` 课程下它长期贴在 0.5 附近才正常）。只有全池实测的 Elo 与池内名次
才能回答"这一版到底多强"。想强行先看图就加 `--require-evaluated 0`。

---

## 1. 迭代流程：一轮里到底发生了什么

```
      ┌─────────────────────────────────────────────────────┐
      │  第 N 轮                                             │
      │                                                     │
      │  ① agent 读材料                                      │
      │     · gamepack/rules.md（规则）                       │
      │     · feedback/（上一轮的回放，自然语言）               │
      │     · leaderboard.json（对手榜单）                    │
      │                                                     │
      │  ② agent 写一版策略 → .agentbench/rollouts/<id>/      │
      │     并写 .agentbench/action.json 说明改了什么、打谁     │
      │                                                     │
      │  ③ 框架校验 + 快照                                    │
      │     工作区 ⊕ overlay → runs/<id>/snapshots/<候选>/     │
      │     预检：能不能 import、能不能走完一帧                  │
      │                                                     │
      │  ④ 打 b 个对手 × 2 座次 = 8 局（b=4 时）                │
      │                                                     │
      │  ⑤ 回放 → narrate → feedback/（下一轮的输入）          │
      │     指标 → events.jsonl                              │
      └─────────────────────────────────────────────────────┘
                            ↓ 每 3 轮抽一版
      ┌─────────────────────────────────────────────────────┐
      │  慢评测（另一个进程，不影响迭代）                        │
      │  这一版 vs 全部 229 个人类选手 → 458 局 → 真实 Elo + 名次 │
      └─────────────────────────────────────────────────────┘
```

### 几个容易误解的点

**"一轮 8 局"是怎么来的**：`1 个候选 × 4 个对手 × 2 个座次 × 1 个 seed`。
两个座次是必须的 —— 很多游戏先手后手优势差很多，只打一个座次的胜率没有意义。

**回放是自然语言的**。A 仓每个游戏都有 `evaluator/narrate.py`，把官方 JSON
回放逐帧翻译成 `replay.md`（"第 34 回合 P0 降级了兵营，第 35 回合放了风暴"）。
agent 读的是这个，不是裸 JSON。8 个游戏全覆盖。

**每轮换一个 codex 会话**（`thread_rotate_each_iteration`）。这是上下文控制的
主要手段：轮末清零，所以单轮内不会膨胀。注意它**不影响** `history_mode: full`
的语义 —— 工作区里的历史代码、经验文档一份不动，断掉的只是 harness 的对话记忆。

---

## 2. 全部参数

只有 5 组旋钮需要你决定，其余在生成的配置里已经钉死（改了很可能让数据不可比）。

### ① 游戏 + 模型

```yaml
game: antwar2                    # 8 个，见下表
provider:
  model_profile: glm-5.2         # configs/models/ 下的档案名
```

#### 8 个游戏

| 游戏 | 人类池 | 座次 | 分差可用 | 每局要注意的 |
|---|---|---|---|---|
| `antwar2` | 229 | P0/P1 | ✅ 55 档 | 主线用它：规则密集、池子最大 |
| `antwar` | 94 | P0/P1 | ✅ 54 档 | — |
| `miracle` | 305 | P0/P1 | ✅ 4 档（值域 ±3 万）| 分差里 30000 是胜利加成 |
| `snakego` | 123 | P0/P1 | ✅ 13 档 | 单局最慢（实测 246s）|
| `rollman` | 111 | **rollman/ghost** | ✅ 9 档 | **唯一分轨游戏**，见 §2⑤ |
| `aquawar` | 194 | P0/P1 | ❌ `{−2,0}` | `rounds` 是**小局数（最多 3）**，不是回合数 |
| `lostspace` | 133 | **P0~P3（四人）** | ❌ `{−3,0}` | 四人同场，其余 3 席由同一对手占 |
| `generals` | 81 | P0/P1 | ❌ `{−1,+1}` | 分差就是胜负本身 |

两个容易踩的：**`aquawar` 的 `rounds` 量纲是 0–3**，所以 `watch_runs` 那条
"0 回合占比高 → 候选协议错"的判据对它天生失效；**`lostspace` 是四人局**，
一轮的对局数仍是 `k × b × 座次`，但"座次"是 4 个里挑 1 个给候选。

#### 8 个模型

模型**独立成档案**（`configs/models/<name>.yaml`），实验里只写一个名字。
为什么这么设计：8 个模型的中转站、api key、上下文窗口、model catalog 全不同，
散在每个实验配置里必然漏字段，而**漏一个 `context_window` 会让 run 在几十分钟后
被压缩打死**。

| 档案 | 中转 | harness | 状态 |
|---|---|---|---|
| `gpt-5.6-sol` | sbtunnel | codex | ✅ 跑通 32 轮 |
| `glm-5.2` | 清华 | codex | ✅ 8 游戏都验过 |
| `glm-5.3` | 清华 | codex | ✅ 验过（高峰会限流）|
| `kimi-k3` | 清华 | codex | ✅ 验过 |
| `deepseek-v4-pro` | 官方 API | codex | ✅ 验过（k=4 两轮）|
| `qwen3.8`（slug `qwen3.8-max`）| 官方 API | codex | ✅ 验过（k=4 两轮）|
| `opus-5` | teamorouter**.cn** | **cc** | ✅ 接通（形状与其余 7 个不同，见下）|
| `longcat-2.0` | 清华 | codex | ❌ 中转拒绝，见 §7 |

模型**独立成档案**（`configs/models/<name>.yaml`），实验里只写一个名字。
为什么这么设计：7 个模型的中转站、api key、上下文窗口、model catalog 全不同，
散在每个实验配置里必然漏字段，而**漏一个 `context_window` 会让 run 在几十分钟后
被压缩打死**。

| 档案 | 中转 | harness | 状态 |
|---|---|---|---|
| `gpt-5.6-sol` | sbtunnel | codex | ✅ 跑通 32 轮 |
| `glm-5.2` | 清华 | codex | ✅ 8 游戏都验过 |
| `glm-5.3` | 清华 | codex | ✅ 验过（高峰会限流）|
| `kimi-k3` | 清华 | codex | ✅ 验过 |
| `deepseek-v4-pro` | 官方 API | codex | ✅ 验过（k=4 两轮）|
| `qwen3.8`（slug `qwen3.8-max`）| 官方 API | codex | ✅ 验过（k=4 两轮）|
| `opus-5` | teamorouter**.cn** | **cc** | ✅ 接通（形状与其余 7 个不同，见下）|
| `longcat-2.0` | 清华 | codex | ❌ 中转拒绝，见 §7 |

**两个官方 API 没有 catalog，所以必须手写 `context_window`。** 只有 `zhipu`
一个内置 catalog，deepseek / qwen 都不在里面。不写这个键 codex 就报
`Unknown model … will use fallback model metadata`，压缩线又变成看不见的数。
两个值都是实测来的，不是抄文档：deepseek 的上限写在它自己的报错里
（`maximum context length is 1048576 tokens`），qwen 不报上限（超长只静默截断），
所以取"实际发进去过并返回 200"的 800000 作为下界。

**`opus-5` 是唯一走 `cc`（Claude Code）harness 的档案**，因为上游只给
Anthropic Messages API：打 `/v1/responses` 会被明确拒绝（"模型和协议不匹配，
正确的请求协议为 Anthropic Messages API"），codex harness 从协议上就走不通。
它因此有三处和别人不一样，都不是风格问题：

* `base_url` **不带** `/v1`（`AnthropicBridge` 自己拼 `/v1/chat/completions`，
  带了就成 `/v1/v1/...`）—— 注意这和同一个中转上 codex harness 的规则**相反**；
* 实验配置要写 `runtime.agent_binary: claude`（生成脚本用 `--agent-binary`）；
* **没有 `reasoning_effort`**。cc 侧没这个旋钮（thinking 块在桥接里还原不了），
  所以横向比较时 opus-5 的"推理深度"与其余 7 个的 `high` 不是同一个量，
  必须在结论里注明。

域名也要注意：`api.teamorouter.com` 在本机被**按 SNI 阻断**（同一个 IP 换个
SNI 就能握手），必须用 `api.teamorouter.cn`。

**加新模型时照抄 `glm-5.3.yaml`，不要自己发明字段。** 那份配置来自跑通 40+ 轮的
`sota-antwar2`，每一行都有理由（写在注释里）。三个已经踩过的坑：

* `model` 用了 `model_catalog` 时必须与 catalog 里的 slug **大小写一致**
  （codex 精确匹配，不符就退回兜底元数据 → 压缩提前触发 → glm 系必死）。
  中转本身不区分大小写，所以用 curl 测**测不出来**。
* `auto_compact_token_limit: 900000` —— 这是"引雷开关"不是"保险"，设低了
  会主动触发压缩。曾经写成 90000（少一个 0），两个 run 直接死。
* `client_name` **只有** sbtunnel 这类按客户端白名单放行的中转需要。

这些都有测试守着（`tests/unit/test_model_profiles.py`），配错跑测试就会报。

#### 每轮实测耗时（排期用）

同一个 antwar2 / k=4 / b=1 / 2 轮的批次里量的，**思考时间**（不含对局）：

| 模型 | 思考/轮 | 输出 token（2 轮）| 请求数 |
|---|---|---|---|
| `glm-5.3` | 11.1m | 51,804 | 69 |
| `gpt-5.6-sol` | 12.3m | 23,734 | 34 |
| `glm-5.2` | 12.5m | 106,071 | 79 |
| `opus-5` | 19.0m | 255,104 | **461** |
| `deepseek-v4-pro` | 22.8m | 183,467 | 137 |
| `kimi-k3` | 25.1m | 79,371 | 76 |
| `qwen3.8` | **39.4m** | 194,342 | 135 |

三件事值得记住：

* **快的那一档就是 11~12.5 分钟/轮**，慢的是上游生成速度（qwen 是 glm-5.3 的 3.5 倍），
  不是我们的配置问题。排期按最慢的那个模型算。
* `opus-5` 的请求数是 gpt 的 13 倍 —— Claude Code 每轮的工具调用远多于 codex，
  而桥接是非流式的（先取完整响应再回放成 SSE），所以每次往返都串行。
* **k=4 放大的是"写"，不是"想"**：一轮要产出 4 份完整策略，输出 token 约是 k=1 的
  4 倍。中转对**单请求时长/体量**有限制，长响应会打出 `HTTP 504` 与
  `IncompleteRead`（响应体读到一半断流）—— 实测清华中转上小请求 3/3 秒级返回、
  同一时刻我们的长 turn 退避 17 次累计等 822s。这不是中转挂了，是单请求太大。

#### 全部旋钮一览（含默认值）

只有标 ★ 的 5 组需要你决定，其余在生成的配置里已经钉死——改了很可能让数据不可比。

| 键 | 默认 | 含义 / 为什么是这个值 |
|---|---|---|
| `game` | — | ★ 8 个之一 |
| `provider.model_profile` | — | ★ `configs/models/` 下的档案名 |
| `runtime.max_iterations` | 32 | ★ 整数 / 不写(=32) / `unbounded`(=128) |
| `runtime.rollout_k` | **4** | ★ 一轮交几个候选。k>1 才有并行假设检验 |
| `curriculum.batch` | **1** | ★ 一轮打几个对手。默认形态是 k=4 / b=1 |
| `curriculum.opponent_policy` | `progress` | ★ `random` / `self` / `progress` / `fix` |
| `curriculum.opponent_start_rank` | **20** | progress 的起点名次（各游戏池都远大于 20）|
| `goal.history_mode` | `full` | ★ `full` / `last_only` |
| `evaluation.background_pool` | **false** | 中间版本慢评测。默认关，见 §4 |
| `evaluation.pool_stride` | 4 | 开了才有意义：每 4 轮取一版 |
| `evaluation.challenger_track` | null | **rollman 必填** |
| `runtime.match_timeout_s` | 1800 | 单局墙钟。会传进对战器内层 |
| `runtime.step_timeout_s` | 30 | 每步思考上限。saiblo 官方是 3s，见下 |
| `runtime.match_parallelism` | `min(16, k·b·2)` | 一轮能一次打完 |
| `runtime.thread_rotate_each_iteration` | true | 每轮换 thread：上下文控制的主要手段 |
| `runtime.thread_rotate_context_tokens` | 60000 | 轮内的保底阈值 |
| `runtime.iteration_mode` | `lockstep` | — |
| `curriculum.ladder_scope` | `auto` | **绝不能写 `crawled`**（榜单会从 229 缩到 20 人）|
| `curriculum.order` | `lowest_rank_first` | 唯一合法值 |
| `curriculum.seed_mode` / `development_seeds` | `fixed` / `[1]` | — |
| `goal.experience_skills` | true | 维护经验文档 |
| `goal.code_constraint` | `any` | `last_only` 下框架会强制单文件 |
| `isolation.backend` | `auto` | bwrap；`rival_code_visible: false` |
| `budget.tokens` / `wall_seconds` | null / null | 停止条件是 `max_iterations` |

**`step_timeout_s` 是个已知的口径偏差，用之前先读这条。** saiblo 判题器对 AI
是**按步**计时的：lostspace/miracle 后端每步都重发 `send_init(AI_TIME, length)`，
miracle 写得最直白 `AI_TIME = 3` / `PLAYER_TIME = 300`（真人 300 秒、AI 3 秒）。
我们的 arena 一直把这一帧里的 `time` 丢掉，只保留整局墙钟。两层后果：

1. 一名选手卡在某一步，saiblo 上只是那一步判超时、对局继续；我们这边要等整局
   墙钟耗尽，**整局作废**；
2. 更要紧的是**有效性**：人类池是在"每步 3 秒"下写出来的（所以清一色 C++），
   而我们的候选此前享受每步无限时间 —— 那样算出来的池内 Elo 不是同一个游戏的。

默认给 30s（宽松但有限）。要做同条件对照就显式写 3。

### ② 轮数

```yaml
runtime:
  max_iterations: 32       # 整数 / 不写(=32) / unbounded(=128)
```

`unbounded` 固定等于 128，不是"真的不限"—— 那样不同 run 的轮数不可比。

### ③ 历史可见性

```yaml
goal:
  history_mode: full       # full | last_only
```

* `full`（默认）：常驻会话 + 经验文档 + 全部历史候选代码都能看到。
* `last_only`：每轮全新会话，工作区只留**上一版**策略 + 最近一份反馈。
  这一档下框架会**强制单文件** —— 不许建 `v2.py/v3.py`、不许 import 历史版本。
  否则"只能看到上一版"会被 import 链绕过（读一下 `v2.py` 就等于看到了历史），
  消融失效而且从曲线上完全看不出来。

### ④ 对手选择方式 + batch

```yaml
runtime:
  rollout_k: 4                # k：一轮提交几个候选策略
curriculum:
  opponent_policy: progress   # random | self | progress | fix
  batch: 1                    # b：一轮打几个对手
  opponent_start_rank: 20     # progress 的起点名次
```

一轮的对局数 = `k × b × 座次`。**默认形态 k=4 / b=1**，对称游戏 4×1×2 = **8 局**。
分轨游戏（rollman）只坐一个座次，是 4×1×1 = 4 局。

**`batch` 有一个坑：单目标策略会忽略它。** `ladder_up` / `ladder_down` /
`fixed_rank` 的 `select()` 恒返回 1 个对手，无论 `batch` 写多少。以前账本、
`run-manifest.json`、提示词里都照抄配置值，于是写着 4、实际打 1 个，
"一轮 = k × b × 座次"算出来差 4 倍。现在这三处一律记**策略实际会打的个数**
（`effective_batch`，有测试守着），所以看到的 b 就是真值。

#### k：一轮提交几个候选（**这是影响最大的一个参数**）

`k>1` 的价值不是"一轮多试几个"，而是**并行假设检验**：一轮把 k 条不同的
取胜路径同时下水，下一轮把胜出那条变成所有候选的共同底盘，再从那里分叉测
增量。一轮拿到 k bit 而不是 1 bit。

实测对照（antwar2，同一个 229 人池，同样是 progress 课程）：

| 设置 | 进池内 #100 | 进 #24 | 进前 10 |
|---|---|---|---|
| **k=4** + 单对手 | **第 3 轮** | 第 9 轮 | 第 21 轮 |
| k=1 + b=4 对手 | 第 30 轮 | 未达到 | 未达到 |

**差了一个数量级。** 而且 k=1 那组的对手**更弱**（平均 Elo 1691 vs 1968），
所以真实差距比这张表更大。

> 曾经把默认值改成 1，理由是"k=4 的多样性会退化成同一份代码改几个阈值"。
> 那个判断被数据推翻了：旧 run 前 14 轮的 pairwise 行差异是 **48~251 行**
> （阈值 15），全部判定 `distinct` —— 多样性是真的。
>
> 当时还假设"b 个对手可以提供探索广度"，这也是错的：**b 个对手只让同一个
> 策略的评估更精确（降方差），不产生新的候选假设。广度必须来自策略侧。**

`k` 是消融维度，显式写 `rollout_k: 1` 会生效。但要注意 k=1 测的是"没有并行
假设检验时能走多远"，**不该用它代表框架能达到的水平**。

框架为 k>1 做了两件事：

* **多样性度量**（`candidate_diversity.py`）：逐轮算 pairwise 行差异、
  方法级覆盖面、继承链。判定为伪多样性（阈值 15 行）时，下一轮提示词开头
  会点名说"上一轮 k 个候选实质相同，等于花 k 倍开销探了 1 个点"。
  这是**事后反馈**而不是拦截 —— 拦下来会让那一轮完全没有数据。
* **逐候选战绩**（`feedback.json` 的 `margin_by_candidate`）：带
  `win_rate` / `played` / `is_best`，按 (胜率, 分差) 降序排，
  **第一项就是本轮的底盘**。全败的一轮按分差排（"哪条路线离赢最近"）。

#### 四种对手选择方式

| 方式 | 含义 |
|---|---|
| `random` | 每轮从全池随机抽 b 个。抗过拟合，但没有难度递进 |
| `self` | agent 自己读榜 + 读战绩决定 b 个。框架只校验对手存在且可运行 |
| `progress` | 从第 20 名起铺 b 个槽位往上爬。**稳定打赢的槽位前进一名，且跳过已打过的名次** |
| `fix` | 固定打榜单前 b 名。最难 |

**`progress` 的晋级判据**：对同一个对手，**最近 2 轮共 4 局里赢下至少 3 局**。

> 为什么不是"累计胜率 > 75%"：累计值会被早期噪声永久拖住。第 1 轮运气不好
> 0/2、之后连赢 6 局，累计恰好 6/8 = 0.75，不严格大于门槛，明明连赢三轮
> 游标还卡在原地。而且它衡量的是"历史平均"而非"当前实力"，可策略是在迭代的。

判据严格意味着**窗口可能整个 run 都不推进**，那不是故障。实测 32 轮的
`ab32-antwar2-progress`：三个对手全程没换过，因为最近 2 轮战绩分别是
`0/4`、`2/8`、`2/4` —— 都没到 3 分门槛。这时该读的是分差
（那一组从 −36 收窄到 −8）而不是"窗口为什么不动"。

`b=1` 时四种方式都退化成单目标形态（`progress` 就是逐个往上升）。

**`b` 决定胜率曲线的分辨率**：b=1 时胜率只有 {0, 0.5, 1} 三档，几乎必然画成
直线；b=4 才有 0/0.25/…/1 九档。

### ⑤ 慢评测

```yaml
evaluation:
  background_pool: false     # 是否自动挂**中间版本**的慢评测（默认关）
  pool_stride: 4             # 开了的话每几轮抽一版
  challenger_track: null     # 分轨游戏必填！见下
```

**默认关掉，只测终局。** 绝大多数实验要的就是一个数字（"迭代 N 轮之后排第几"），
那在迭代跑完之后用 `scripts/final_elo.sh` 单独测最省事也最省机时。
开着它的代价是实测过的：它每个 run 起一个 worker，而水位控制是各自判断的，
N 个 worker 会一起下水（16 个 run 时 load 冲到 40/32，迭代慢 10~20 倍，
而且 45 个排队版本一个都没测完）。详见 §4。

**当前口径（谁开谁不开）**：

| 实验 | 中间版本慢评测 | 怎么拿 Elo |
|---|---|---|
| 一（主表：多模型 × 多游戏）| ❌ | 迭代跑完 `final_elo.sh` 测终局 |
| **二 / 三（主线）** | ✅ `background_pool: true`, `pool_stride: 4` | 学习曲线 |
| 四 / 五（消融）| ❌ | 同实验一 |
| 链路验证、烟测 | ❌ | 一般不需要；要的话同上 |

只有主线要回答"随轮次怎么进步"，所以只有它需要中间点（32 轮 → 8 个点）。
其余实验（含主表）要回答的是"同样 N 轮之后排第几"，那是**一个数字**，
跑完再测终局就够，而且便宜一个数量级。完整命令见 §2.5。

**`rollman` 是唯一的分轨（非对称）游戏**，必须写 `challenger_track: rollman`，
指明挑战者扮演哪一轨。漏了会同轨互殴（ghost 打 ghost），那种对局在协议层就
没有意义 —— 实测回放只有 2 行。生成脚本会自动处理。

★ 这个字段虽然写在 `evaluation:` 下，但它**同时管迭代和慢评测**。它以前只被
慢评测消费，迭代那边照旧按对称游戏换座次，于是分轨游戏一半的对局是无效的：
对手已经被过滤成 ghost 了，候选却还要去坐 ghost 那个座次。实测
`s8k4-rollman` 第 1 轮 8 局::

    role=rollman → 352~500 回合，margin 37~140    真实对局
    role=ghost   → **0 回合**，却记成 result=win  无效局

4/8 无效**且都算赢**，那一轮胜率被抬到 1.0；而监控只报得出"0 回合对局占
4/8 —— 候选大概率协议格式错"，指向的是完全无辜的候选。

所以：**分轨游戏一轮的对局数是 `k × b × 1`，不是 `k × b × 2`。**
分轨游戏的两个角色不是"座次"而是两种不同的游戏（rollman 轨的选手只实现
吃豆人一侧的协议），不存在"换先后手"这回事。对称游戏仍然必须打两个座次。

短 run（≤ 5 轮）要把 `pool_stride` 设成 1，否则一个数据点都拿不到。

---

## 2.5 实验清单：每个实验是什么、怎么挂、Elo 怎么测

| 实验 | 内容 | 配置 | 中间慢评测 | Elo 怎么拿 |
|---|---|---|---|---|
| **一** | 主表：多模型 × 多游戏 | 生成脚本 | ❌ | **终局** `final_elo.sh` |
| **二** | HL vs 人类主线（不限轮）| `exp2-*-conquest.yaml` | ✅ stride 4 | 学习曲线 |
| **三** | （还没建）| — | ✅ stride 4 | 学习曲线 |
| **四** | 消融：能否看到自己的历史 | `exp4-abl-history_*.yaml` | ❌ | 终局 |
| **五** | 消融：选敌人顺序 | `exp5-abl-*.yaml` | ❌ | 终局 |

判据很简单：**要"随轮次怎么进步"才开中间慢评测；只要"N 轮之后排第几"就测终局。**
后者便宜一个数量级，而且跑在迭代结束之后、不抢机时。

### 实验一：主表（多模型 × 多游戏）

```bash
V=~/agentbench/.venv/bin/python
cd ~/agentbench/AgentBenchHL

# 多模型（固定游戏）
$V scripts/make_verify_configs.py --out-dir configs/experiments/main \
  --models gpt-5.6-sol glm-5.2 glm-5.3 kimi-k3 deepseek-v4-pro qwen3.8 \
  --game antwar2 --iterations 32 \
  --rollout-k 4 --batch 1 --start-rank 20 --prefix main

# 多游戏（固定模型）
$V scripts/make_verify_configs.py --out-dir configs/experiments/main \
  --games antwar antwar2 generals miracle rollman snakego aquawar lostspace \
  --model glm-5.2 --iterations 32 \
  --rollout-k 4 --batch 1 --start-rank 20 --prefix main

# opus-5 走 cc harness，要额外指定 harness 可执行文件
$V scripts/make_verify_configs.py --out-dir configs/experiments/main \
  --models opus-5 --game antwar2 --iterations 32 \
  --rollout-k 4 --batch 1 --start-rank 20 --agent-binary claude --prefix main

# 起 run（分批，见 §3 的并发上限）
for m in glm-5.2 glm-5.3 kimi-k3; do
  bash scripts/run_hl.sh configs/experiments/main/main-$m-antwar2.yaml main-$m 32
  sleep 3
done
```

**迭代全部跑完之后**再测终局 Elo（这是实验一唯一需要的 Elo）：

```bash
# 一条命令测一批。幂等：同一个 run 已有 worker 就跳过；--once 跑完即退
bash scripts/final_elo.sh antwar2 main-glm-5.2 main-glm-5.3 main-kimi-k3

# 分轨游戏要指定挑战者轨
CHALLENGER_TRACK=rollman bash scripts/final_elo.sh rollman main-glm-5.2-rollman

# 并发可以开大（迭代已结束，机器是空的）；机器共用，仍留 6 核
PARALLEL=10 HEADROOM=6 bash scripts/final_elo.sh antwar2 main-gpt-5.6-sol
```

成本差别是决定性的：`--best-only --iteration-stride 3` 口径下 4 个 run 是
50 版 / 22900 局 / 约 17 小时，而终局口径是 4 版 / 1832 局 / 约 1.6 小时。

### 实验二：HL vs 人类主线

配置已经写好，直接起（**不限轮数**，靠 budget 收尾）：

```bash
bash scripts/run_hl.sh configs/experiments/exp2-antwar2-conquest.yaml exp2-antwar2
bash scripts/run_hl.sh configs/experiments/exp2-antwar-conquest.yaml  exp2-antwar
```

口径：`progress` / **起点第 20 名** / k=4 / b=1（一轮 8 局）/ `history_mode: full` /
中间慢评测开、stride 4（32 轮出 8 个点）。它是唯一需要学习曲线的实验，所以
`background_pool: true` 写在配置里，起 run 时会自动挂 worker，不用手动 attach。

### 实验四 / 五：消融

```bash
for f in configs/experiments/exp4-abl-history_*.yaml; do
  bash scripts/run_hl.sh "$f" "$(basename "$f" .yaml)" 32; sleep 3
done
# 跑完统一测终局
bash scripts/final_elo.sh antwar2 exp4-abl-history_full exp4-abl-history_no_notes \
  exp4-abl-history_memoryless
```

⚠️ **实验五当前有一处口径不齐**：5 组声称"只差 `opponent_policy`"，但
`ladder_up`/`ladder_down` 的 `select()` 恒返回 1 个对手，而 `fix`/`random`/`self`
尊重 `batch: 4`。于是前两组一轮 8 局、后三组一轮 32 局，**评估样本差 4 倍**，
胜率曲线的分辨率也不同（b=1 只有 {0,0.5,1} 三档）。要横向比就得先把 b 对齐。

---

## 3. 配置怎么生成（不要手写）

```bash
V=~/agentbench/.venv/bin/python

# 链路验证（2 轮、铺一批）：慢评测默认就是关的
$V scripts/make_verify_configs.py --out-dir configs/experiments/mine \
  --models glm-5.2 glm-5.3 kimi-k3 gpt-5.6-sol deepseek-v4-pro qwen3.8 \
  --game antwar2 --iterations 2 \
  --rollout-k 4 --batch 1 --start-rank 20 --prefix mine

# opus-5 走 cc harness，要额外指定 harness 可执行文件
$V scripts/make_verify_configs.py --out-dir configs/experiments/mine \
  --models opus-5 --game antwar2 --iterations 2 \
  --rollout-k 4 --batch 1 --start-rank 10 \
  --agent-binary claude --prefix mine

# 只有需要**学习曲线**的主线（实验二/三）才加 --background-pool
# 主表（实验一）、消融（四/五）都不加，跑完用 final_elo.sh 测终局
$V scripts/make_verify_configs.py --out-dir configs/experiments/mine \
  --game antwar2 --models glm-5.2 --iterations 32 \
  --rollout-k 4 --batch 1 --start-rank 20 \
  --background-pool --stride 4 --prefix mine
```

慢评测**默认关**，理由见 §4（每个 worker 各自判水位，N 个一起下水会把机器
打满）。默认流程是迭代跑完之后用 `scripts/final_elo.sh` 只测终局。

跑完之后用这个脚本核对"链路真的通了"——它逐轮打印 k、**k 个候选的代码指纹
是否互不相同**、对手名次、0 回合局数：

```bash
$V scripts/audit_iteration_shape.py --runs-root runs --expect-k 4 \
  --run-id mine-glm-5.2 mine-glm-5.3
```

为什么需要它而不是看 `watch_runs.py`：后者的"候选=N 个"数的是**轮数**
（每轮的最佳候选），k=4 的 run 看起来也是"候选=2 个"，**看不出**一轮到底
交了几个策略、更看不出那几个策略是不是同一份代码改了几个阈值。

**手写 8~12 份 yaml 必然漂移**，而漂移在图上看不出来：只要有人改了其中一份的
`match_timeout_s`，那组数据就再也不可比，但曲线看起来一切正常。

生成后建议 diff 复验一遍，确认只差被测维度：

```bash
cd configs/experiments/mine
for f in mine-glm-5.3-antwar2.yaml mine-kimi-k3-antwar2.yaml; do
  diff <(grep -vE '^ *#' mine-glm-5.2-antwar2.yaml) <(grep -vE '^ *#' $f) | grep '^[<>]'
done
```

### 多模型 × 多游戏：并发与磁盘怎么管

**一次最多铺 6~8 个 run。** 三个硬约束，都实测撞过。

#### ① CPU（32 核）

一个 run 占 `match_parallelism` 个对局并发（k=4/b=1 时是 8）。但**对局不是瓶颈**：
一波 8 局在空机器上约 56 秒，而 agent 思考占每轮 11~39 分钟。所以真正决定
并发上限的是 ②③。

铺 12 个时实测 load 冲到 75，所有东西一起变慢。

#### ② 中转账号并发（**按中转分开数，不是按 run 数**）

铺 12 个时报 `Concurrency limit exceeded for account`，直接让 run 失败。
关键是**同一个中转上的 run 数**：

| 中转 | 用它的档案 | 建议同时 ≤ |
|---|---|---|
| 清华 sub2api | `glm-5.2` `glm-5.3` `kimi-k3` `longcat-2.0` | 8 |
| sbtunnel | `gpt-5.6-sol` | — |
| 官方 API | `deepseek-v4-pro` `qwen3.8` | — |
| teamorouter.cn | `opus-5` | — |

所以"多游戏 × glm-5.2"这种批次全压在清华一个账号上，8 个游戏就是上限；
而"多模型 × antwar2"天然分散在 4 个中转上，压力小得多。**排批次时按中转分组，
不要按游戏或模型分组。**

另外 k=4 会放大**单请求**体量（一次写 4 份策略），中转对单请求时长/体量有限制，
长响应会打出 `504` / `IncompleteRead`。这不占并发额度，但会让那一轮墙钟从
几分钟涨到几十分钟（退避会吸收掉，不丢数据）。

#### ③ 磁盘（这是多模型 × 多游戏真正的墙）

实测单个 run 的构成：

| | 2 轮的 run | 32 轮的 run |
|---|---|---|
| 合计 | **52M** | **3.4G** |
| `official-matches/`（对局工件）| 800K | **2.0G**（58%）|
| `pool-elo/`（慢评测）| — | 576M |
| `codex-home/`（会话日志）| 43M | 459M |
| `workspace/`（回放/反馈）| 908K | 362M |
| `snapshots/` + `frozen-build/` | 8M | 61M |

按 3.4G/run 估算主表规模：**8 游戏 × 8 模型 × 32 轮 ≈ 64 个 run ≈ 190G**。
当前盘 504G、已用 283G、**可用 201G** —— 也就是"刚好装不下"。所以：

```bash
df -h /home                      # 起批之前先看
du -sh runs/*/ | sort -h | tail  # 谁在占
```

三条省空间的办法，按性价比排：

1. **分批跑完就归档**：`official-matches/` 占 58%，而它在出图之后只有排查价值。
   跑完一批、出完图、确认结论之后再压缩或删掉那一批的 `official-matches/`。
2. **别对不需要曲线的实验开 `background_pool`**：省掉 576M/run，还顺带省机时。
3. **`codex-home/` 可以事后删**：它是会话日志，续跑需要 `state`，但历史 sqlite
   在 run 结束后没有分析价值。

⚠️ **绝不要删正在跑的 run 的目录**（见 §8）：进程会继续往已删除的 inode 写，
`ls` 看起来什么都没产出，白跑几十分钟。

推荐做法：**先跑通 1 个确认配置无误，再按中转分组分批铺开，每批跑完先出图再起下一批。**

---

## 4. 什么是"慢评测"

### 它回答什么问题

迭代过程里的胜率是"对**本轮那 4 个对手**的胜率"，会被对手难度混淆：

* `fix` 组固定打榜单前 4 名 → 胜率长期是 0；
* `random` 组随机抽人 → 胜率能到 0.5；
* `progress` 组变强了就换更强的对手 → 胜率长期贴在 0.5 附近。

这三个数字**互相不可比**。所以要回答"哪种策略更好"、"哪个模型更强"，
唯一可比的标尺是：**同样迭代 N 轮之后，这一版插进 229 人的人类池排第几**。

慢评测就是算这个：每 3 轮抽一版，让它打完整个冻结人类池（**458 局**），
拟合出真实 Elo 与池内名次。

### 它为什么是独立进程

* **不共写账本**：worker 只写 `runs/<id>/pool-elo/`，主账本 `events.jsonl`
  由迭代进程独占。两个进程追加同一个账本会让对局记录交错，且看不出来。
* **不拖慢迭代**：它自带 CPU 水位控制（`--headroom`），忙时自己停下等。
* **能单独重启**：慢评测崩了不影响迭代，反之亦然。

### ★ 慢评测经常滞后，要手动挂

**这是最需要注意的一件事。** 慢评测的工作量比迭代大一个数量级：

| | 一轮的对局数 |
|---|---|
| 迭代 | 8 局 |
| 慢评测（每 3 轮一次） | 458 局 |

32 轮 → 11 版 × 458 = **约 5000 局**。实测吞吐约 **19 局/分钟**（antwar2，
6 并发），也就是 **单个 run 的慢评测要 4~5 小时**。四组 ablation 并行时约 17 小时。

**所以：迭代早就跑完了，慢评测还差得远，是完全正常的状态。**

#### ★★ 铺多个 run 时必须关掉它（`--no-background-pool`）

`background_pool: true` 是**每个 run 起一个 worker**。它的 CPU 水位控制是
**各自判断**的，不是全局配额：每个 worker 等的是
`load + headroom <= 总核数`，于是 N 个 worker 会在同一时刻各自看到
"load 18 < 24，可以派发"然后一起下水。**每个都"预留 8 核"等于谁也没预留。**

实测（16 个 run × 2 轮，32 核）：

| | 数字 |
|---|---|
| `worker-status.json` 记到的 load | **40.4**（超配 125%）|
| 慢评测打的局数 | 1484 |
| 迭代自己打的局数 | 232（慢评测是它的 **6.4 倍**）|
| 同一个 8 局波次：机器空时 / 铺满后 | **56s / 607~1230s**（慢 10~20 倍）|
| 45 个排队版本里完成的 | **0 个** |

最后一行是关键：每版要 458 局，而 worker 随 run 一起退出，2 轮的 run
根本凑不满一版（实测某版停在 234/458）。**那 1484 局机时换回来 0 个 Elo
数据点，同时把迭代拖慢一个数量级。**

所以短 run / 多 run 的链路验证一律加 `--no-background-pool`，
要 Elo 就等迭代跑完再用 `attach_slow_eval.sh` 单独挂（那时机器是空的，
还能把 `--parallel` 开大）。

#### 什么时候需要手动挂

1. run 启动时 `background_pool: false`（或者那时还没开这个开关）；
2. worker 因为机器过载/被误杀而退出；
3. 你想临时提高并发赶数据。

#### 怎么挂

```bash
cd ~/agentbench/AgentBenchHL
bash scripts/attach_slow_eval.sh <run-id> [stride] [game]

# 例：
bash scripts/attach_slow_eval.sh my-run 3 antwar2
```

这个脚本是**幂等**的：已经有 worker 在跑同一个 run 时会直接跳过，不会起第二个。
（两个 worker 并发写同一个 `pool-elo/` 会重复调度、结果文件互相覆盖，且不报错。）

#### 想跑快一点

`attach_slow_eval.sh` 默认 `--parallel 2 / --headroom 10`（为了避让迭代）。
迭代跑完之后可以手动提高：

```bash
# 先停掉现有 worker
ps -eo pid,args | grep "[p]ool_elo_worker" | grep <run-id> | awk '{print $1}' | xargs -r kill

# 用更高并发重挂
cd ~/agentbench/AgentBenchHL
setsid nohup ~/agentbench/.venv/bin/python -u scripts/pool_elo_worker.py \
  --run-root runs/<run-id> --agentbench-root ~/agentbench/AgentBench \
  --game antwar2 --seeds 7 --best-only --iteration-stride 3 \
  --parallel 6 --headroom 6 \
  >> ~/agentbench/runs/<run-id>.pool-elo.log 2>&1 < /dev/null &
```

#### ★ 只要终点、不要曲线（**这是默认口径**）

比较几种设置哪个更好时，需要的往往只是一个数字："同样迭代 32 轮之后，
各自在人类池排第几"。这时不要跑整条曲线。

生成的配置默认 `background_pool: false`，所以标准流程是**迭代跑完之后**
单独测终局：

```bash
cd ~/agentbench/AgentBenchHL
bash scripts/final_elo.sh antwar2 vk4-glm-5.2 vk4-glm-5.3 vk4-kimi-k3

# 分轨游戏要指定挑战者轨（rollman 是唯一一个）
CHALLENGER_TRACK=rollman bash scripts/final_elo.sh rollman s8k4-rollman
```

它是幂等的（同一个 run 已有 worker 就跳过），`--once` 跑完即退。
因为迭代已经结束、机器是空的，`--parallel` 默认给到 8（可用环境变量覆盖）。

底层就是下面这条命令，需要精确控制时直接用：

```bash
setsid nohup ~/agentbench/.venv/bin/python -u scripts/pool_elo_worker.py \
  --run-root runs/<run-id> --agentbench-root ~/agentbench/AgentBench \
  --game antwar2 --seeds 7 \
  --last-n-best 1 \
  --parallel 7 --headroom 4 --once \
  >> ~/agentbench/runs/<run-id>.final-elo.log 2>&1 < /dev/null &
```

`--last-n-best 1` = 只测末轮选中的那个版本。成本差别是决定性的：

| 口径 | 版本数（4 个 run） | 局数 | 耗时 |
|---|---|---|---|
| `--best-only --iteration-stride 3` | 50 | 22900 | **约 17 小时** |
| `--last-n-best 1` | 4 | 1832 | **约 1.6 小时** |

`--once` 让 worker 跑完就退出，不留守轮询。

也可以用 `--only-candidates v031_xxx v030_yyy` 精确指定版本（拼错的名字会
直接报错退出，不会静默少测）。

#### 怎么定 `--parallel`

32 核，留 4~6 给别人和 agent 思考，每局约 1 核。四个 run 并行时每个给 6~7。
别忘了机器是共用的：

```bash
ps -eo user,pcpu,args --sort=-pcpu | head    # 看谁在占机时
```

**动别人的进程前一定要先问。**

#### 怎么看慢评测进度

```bash
cd ~/agentbench/AgentBenchHL
for r in my-run; do
  echo "$r: 完成 $(ls runs/$r/pool-elo/*/challenger-elo.json 2>/dev/null | wc -l) 版 / \
排队 $(ls -d runs/$r/pool-elo/*/ 2>/dev/null | wc -l) / \
已打 $(cat runs/$r/pool-elo/*/matches.jsonl 2>/dev/null | wc -l) 局"
done

tail -5 ~/agentbench/runs/my-run.pool-elo.log     # worker 日志
```

worker 在队列空 + 迭代已结束时会**自动退出**释放机时，不用手动收。

---

## 5. 看进度

```bash
$V scripts/watch_runs.py --runs-root runs --log-dir ~/agentbench/runs \
  --target 32 --run-id my-run another-run
```

它不只报轮数，还会点出**"轮数在涨但其实在空转"**这类看不出来的故障：

| 告警 | 含义 |
|---|---|
| `候选 id 一直不变` | agent 在重交同一份代码，没有真的迭代 |
| `0 回合对局占比高` | 候选协议格式错（0 回合 = 直接判负，学不到东西）|
| `policy=random/self 却只打过 ≤b 个对手` | 对手策略没生效 —— 消融会退化成同一组 |
| `policy=fix 却打过 >b+2 个对手` | fix 应当固定打前 b 名 |
| `上游退避重试 N 次` | 限流。**不是错误**，但会让每轮墙钟从几分钟涨到几十分钟 |

关于限流：503 会被自动重试吸收（最多 24 次 / 累计 30 分钟），实测四组 ablation
累计 508 次 503、**最终放弃 0 次**。它只拖慢吞吐，不丢数据。表现是"进度看起来
卡住了"而日志里毫无报错 —— 所以监控会把它显示出来。

---

## 6. 出图

### 完整流程

```bash
cd ~/agentbench/AgentBenchHL
V=~/agentbench/.venv/bin/python

# ① 主图：学习曲线（3 组纵坐标 × 2 种横坐标 = 6 个子图）
$V scripts/plot_learning_curves.py \
  --run-dir runs/main-glm-5.2 runs/main-glm-5.3 runs/main-kimi-k3 \
  --out-dir ~/agentbench/analysis/main \
  --require-evaluated 0

# ② 导出原始数据（自己做表 / 复核用）。--run-root 可以给多次做对比
$V scripts/export_curves.py \
  --run-root runs/main-glm-5.2 --run-root runs/main-glm-5.3 \
  --label-by model --out ~/agentbench/analysis/main

# ③ 汇总表：一组 run 的末轮成绩 + 池内 Elo/名次，回答"哪个设置最强"
#    它按 <prefix><arm> 拼 run-id，所以起 run 时的命名要成体系
$V scripts/ablation_report.py --prefix main-glm-5.2- \
  --arms antwar antwar2 generals miracle rollman snakego aquawar lostspace
```

`export_curves.py` 的 `--label-by` 决定曲线用什么当图例：消融比 `opponent_policy`
或 `history_mode`，主表比 `model` 或 `game`。`ablation_report.py` 是**按 run-id 拼
名字**的（`--prefix` + `--arms`），所以起 run 时 run-id 要有规律，否则汇总不到。

参数只有一个需要你决定：

| 参数 | 含义 |
|---|---|
| `--run-dir` | 一个或多个 run 目录。给多个会**另出一张对比图** |
| `--out-dir` | 图和数据落在哪。惯例是 `~/agentbench/analysis/<批次名>` |
| `--require-evaluated N` | 全池评测覆盖的轮数少于 N 就**拒绝出图**（默认 1）|
| `--pool-elo-dir` | 慢评测产物不在 run 目录里时手动指定 |

`--require-evaluated` 默认是 1 而不是 0，是刻意的：**没有慢评测数据时画出来的
曲线只有快通道，而快通道衡量的是"对手难度"不是"绝对强度"**，很容易被读成
"没有进展"。要强制出图就显式写 `--require-evaluated 0`，那等于声明"我知道这张图
缺绝对标尺"。慢通道没评测完的版本**不画点**——数据没到就是没到。

### 产物落在哪（实测的文件名）

```
~/agentbench/analysis/<批次名>/
├── curves-<run-id>.png     # plot_learning_curves：每个 run 一张（6 个子图）
├── curves-<run-id>.json    #   那张图对应的数据点
├── curves-compare.png      #   给了多个 --run-dir 时的对比图
├── curves.csv              # export_curves：逐轮指标（自己做表用）
├── series.json             #   同上，结构化
└── summary.md              #   文字摘要
```

`export_curves.py` 会把"全程为 null 的纵坐标"显式列出来（例如 IG 相关的几项
已从主线移除），那是**提示不是错误**——免得有人对着空面板猜是不是跑坏了。

### 三组纵坐标

**① 胜率** —— 两条线，口径不同，不要直接比高低：

* 虚线（快通道）：对**本轮那 b 个对手**的胜率。零成本、每轮都有，
  但它衡量的是**难度**而不是绝对强度。`progress` 组会随 agent 变强而换更强的
  对手，所以这条线**长期贴在 0.5 附近才是正常的**。
* 实线（慢通道）：对**冻结人类池**的总胜率。这条才是绝对强度。

**② Elo** —— 同样两条：

* 橙点（零成本反解）：拿"本轮真打过的那几局 + 冻结池锚点"做锚定 BT/MLE。
  一分钱不多花，但样本只有 8 局，噪声大。点面积 ∝ 对局数。
* 绿线（慢通道）：全池实测，标注是池内插入名次。**结论要读这条。**

两条线用**同一个估计器**，所以同尺度可比。
（历史 bug：旧实现用钳位反解，全败时会恒等于一个常数 —— `fix` 组曾出现
14 轮 Elo 全是 1431.37 的假平线，而同期分差在稳步改善。）

**③ 分差** —— 胜率与 Elo 看不见的那段进展。

胜率有阈值效应（赢不下来就一直是 0），分差是连续量，所以它先动。实测 `fix` 组
14 轮胜率恒 0、Elo 恒定，而分差从 −36 收窄到 −28（最好的一局 −32 → −18）。

**但分差不是所有游戏都有意义**（脚本会自动判断并留白说明）：

| 有意义 | 取值数 | 没意义 | 取值 |
|---|---|---|---|
| antwar / antwar2 | 54 / 55 | **generals** | `{−1, +1}` ← 就是胜负本身 |
| snakego / rollman | 13 / 9 | lostspace | `{−3, 0}` |
| miracle | 4（值域 ±3 万）| aquawar | `{−2, 0}` |

**④ token** —— 柱状是每轮增量，折线是累计。

（历史 bug：`tokenUsage.last` 是"这一次请求"的用量，旧实现按"会话累计值"处理，
导致某些 run 连续 10 轮报同一个数、另一些逐轮精确翻倍，真实值低估 14 倍。）

---

## 7. 已知问题

### longcat-2.0：中转拒绝，不是我们能配的

报 `400 Model is not supported by composite groups`。我排除了四个假设，
每个都有实测：

| 假设 | 验证方式 | 结果 |
|---|---|---|
| slug 大小写 | 改成中转返回的 `LongCat-2.0` | 仍拒 |
| `auto_compact` 太低 | 改成 900000 | 仍拒 |
| `remote_compaction_v2` beta header | 加 flag 关掉，指纹确认已消失 | 仍拒 |
| originator | 加 `client_name`，指纹确认已是 `codex_exec` | 仍拒 |

关键事实：**`codex exec`（简单对话）能通，app-server（goals 协议）被拒。**
两者剩下的唯一差异是请求体 —— goals 协议会发 `tools` 列表。所以
`Model is not supported by composite groups` 很可能指**这个模型在该中转
不支持工具调用**。需要向中转方确认。

### opus-5：已解决（记录在此，因为三个坑各自都会让人白花很久）

结论是 `harness: cc` + `api.teamorouter.cn`，见 §2。过程里三件事**每一件都
指向错误的方向**：

1. **sbtunnel 上根本没有 claude**（`/v1/models` 只列 OpenAI 家），
   `/v1/messages` 返回 `403 This group does not allow /v1/messages dispatch`。
   在那个端点上换路径是白费时间。
2. **`api.teamorouter.com` 被按 SNI 阻断**，表现是 `Connection reset by peer`，
   极容易误判成"域名写错了"或"中转挂了"。判据是换 SNI 打同一个 IP：

   ```bash
   openssl s_client -connect <ip>:443 -servername api.teamorouter.com   # write:errno=104
   openssl s_client -connect <ip>:443 -servername www.example.com       # 正常拿到证书链
   ```

   TCP 443 本身是通的（`nc -vz` 成功），所以只测端口会得出"网络没问题"的
   错误结论。改用 `.cn` 即可。
3. **`/v1/responses` 被拒不是配置问题**：上游原话是"模型和协议不匹配，正确的
   请求协议为 Anthropic Messages API"。这是唯一一个 codex harness 从协议上
   就走不通的模型，只能换 harness。

另外修了一个**只在我们这种压缩方式下才会犯的 bug**：cc harness 原先按"这是
第几个 turn"决定发 `--session-id` 还是 `--resume`，而
`thread_rotate_each_iteration` 每轮都会 `start()` 一个**全新** thread，
于是第 2 轮对一个从未建立过的会话发 `--resume`：

```
claude exited 1: No conversation found with session ID: 97264be0-…
```

第 1 轮的 4 个候选和 8 局全部正常，run 死在第 2 轮开头 —— 从账本上看像是
"模型这一轮没产出候选"。现在按**每个 thread 自己**是否已建立来判断
（`tests/unit/test_cc_session_rotation.py` 守着）。

### 排查上游问题的正确方法

**不要用 curl 探连通性。** 有些中转按客户端指纹放行，裸 curl 一律 403，
会让人误判"端点坏了"。

**也不要只用 `codex exec` 探。** 它自己报 `originator: codex_exec`，
而框架走 app-server 报的是 `agentbench-hl` —— 那才是框架真正的路径。
用 exec 探通了不代表框架能用（longcat 就是这样）。

**看 `[llm-upstream]` 日志。** 上游返回非 2xx 时，`responses_proxy` 会把
originator / user-agent 打进 driver 日志（绝不打印 authorization）：

```bash
grep "llm-upstream" ~/agentbench/runs/<run-id>.out | tail -3
```

---

## 8. 续跑

```bash
bash scripts/run_hl.sh <config> <run-id> 32     # 第一次：1 → 32
bash scripts/run_hl.sh <config> <run-id> 32     # 再执行：33 → 64
```

**第三个参数是"本次再跑多少轮"，不是"总共跑到第几轮"。** driver 的计数每个
进程从 0 开始，"这个 run 是否已经开始过"由磁盘状态判断。

checkpoint 是自动的：`events.jsonl`（完整事件账本）+ `state`（thread id 与
已完成轮数）+ `snapshots/`（每轮候选代码）。进程被杀、机器重启之后用同样的
命令就能接着跑。

`run_hl.sh` 会拒绝对同一个 run-id 起第二个进程 —— 两个进程写同一份
`events.jsonl` 会让对局记录交错、thread 状态互相覆盖，而这从曲线上看不出来。

**续跑时改配置**：只有观测通道能改（`evaluation` / `budget` /
`runtime.max_iterations` / `match_parallelism` / `match_timeout_s`）。
改模型、改 k、改对手策略会被拒绝并列出具体字段 —— 那些会让前后轮次不可比。

### ⚠️ 不要删正在跑的 run 的目录

进程会继续往已删除的 inode 写，`ls` 看起来什么都没产出，白跑几十分钟。
（我为了看日志删过一次，longcat 白跑 23 分钟。）

---

## 9. 代码结构与产物位置

### 两个仓的分工（先搞清这条，否则会改错地方）

```
~/agentbench/
├── AgentBench/      ← A 仓：游戏资产 + 官方后端 + 对战器（“事实源”）
└── AgentBenchHL/    ← B 仓：迭代框架（本手册讲的东西）
```

* **规则裁决在 A 仓，而且是 saiblo 官方后端进程**（`backend_sources/corpus/<游戏>/`，
  arena 用 `subprocess` 真起 `gamecode_logic/main.py`）。规则不是我们实现的，
  所以**偏差不会出在规则上**。
* **判题转发层是我们自研的**（A 仓 `games/<游戏>/evaluator/arena.py`）：
  在后端与选手之间转发帧、判超时、处理选手崩溃。历史上的坑基本都在这一层。
* **迭代、课程、慢评测、出图在 B 仓。**

改东西之前先问：这是规则问题（去 A 仓后端，基本不该改）、转发问题（A 仓 arena）、
还是迭代问题（B 仓）。

### B 仓（AgentBenchHL）的分层

```
src/agentbench_hl/
├── config.py              # 全部配置项的唯一定义 + 校验（要查默认值看这里）
├── domain/                # 纯计算：pool_elo / 指标，无 IO
├── application/           # 业务流程（不碰具体 harness / 游戏）
│   ├── goal_led_service.py    #   一轮迭代的主流程（最大的一个文件）
│   ├── goal_led_driver.py     #   多轮循环、checkpoint、续跑
│   ├── live_run.py            #   装配 GoalLedService（座次/对手过滤在这）
│   ├── opponent_policy.py     #   progress / random / self / fix
│   ├── challenger_eval.py     #   慢评测：挑战者 vs 冻结人类池
│   ├── slow_eval.py           #   background_pool 的进程管理
│   └── candidate_diversity.py #   k 个候选是不是伪多样性
├── adapters/              # 与外部世界的接口
│   ├── codex_goal/            #   codex harness（app_server / responses_proxy）
│   ├── cc_goal/               #   Claude Code harness（opus-5 走这条）
│   ├── contract/              #   对局调度：arena / match_worker / factory
│   ├── isolation/             #   bwrap 沙箱 + uds 网关
│   └── <游戏>/                #   各游戏的探针/叙述适配
├── ports/                 # 接口定义（agent_runtime / isolation）
└── cli/main.py            # `abhl` 命令入口
```

```
gamepacks/<游戏>/           # 喂给 agent 的材料
├── rules.md                   #   规则
└── candidate_support/         #   候选写策略要用的 SDK（官方 SDK + 我们的入口）
    ├── CANDIDATE_CONTRACT.md  #     候选契约（agent 先读这个）
    ├── main.py                #     入口，由 _shared/candidate_runners/ 生成
    └── SUPPORT_PROVENANCE.json#     哪些文件是官方原版、哪些是我们加的

gamepacks/_shared/candidate_runners/<游戏>.py   # ★ 入口的**源头**，改这里
```

⚠️ `gamepacks/<游戏>/candidate_support/main.py` 是**生成物**。要改入口逻辑得改
`_shared/candidate_runners/<游戏>.py` 再同步过去，否则下次重新生成就被覆盖。
契约文档由 `scripts/gen_candidate_support.py` 从入口的 docstring 生成。

### 常用脚本

| 脚本 | 干什么 |
|---|---|
| `make_verify_configs.py` | **生成实验配置**（不要手写 yaml）|
| `run_hl.sh` | 起 run / 续跑 |
| `watch_runs.py` | 看进度，并报"轮数在涨但其实在空转"这类故障 |
| `audit_iteration_shape.py` | 核对每轮真的交了 k 个**代码不同**的候选 |
| `final_elo.sh` | **只测终局** Elo（默认口径）|
| `attach_slow_eval.sh` | 给已在跑的 run 补挂中间版本慢评测 |
| `plot_learning_curves.py` | 出图 |
| `export_curves.py` | 导出曲线原始数据 |
| `ablation_report.py` | 消融汇总表 |
| `probe_responses_models.py` | 探中转/模型是否真的可用（探 `/responses`）|

### 产物位置

```
runs/<run-id>/                        # 一个 run 的全部产物
├── events.jsonl                      # ★ 事件账本 —— 所有分析的唯一事实源
├── run-config.json                   # 冻结的实验配置（续跑校验用）
├── run-manifest.json                 # 实际生效的参数（含 k / b / 座次）
├── public-leaderboard.json           # 该 run 冻结的对手榜单（含 rank）
├── workspace/                        # agent 的工作区
│   ├── main.py                       #   当前策略
│   ├── research/                     #   经验文档（跨轮接力靠它）
│   ├── feedback/<轮>/<候选>/<座次>/   #   replay.md 是自然语言回放
│   └── .agentbench/                  #   action.json / rollouts/
├── snapshots/<候选>/                  # 每轮的候选代码（含运行时支撑文件）
├── official-matches/                 # 逐局工件（回放/线协议/各进程 stderr）★ 最占空间
├── frozen-gamepack/                  # 冻结的 gamepack（保证可复现）
├── frozen-build/                     # 编译产物（C++ 选手/后端）
├── codex-home/                       # codex 的 config.toml / models.json / 会话历史
└── pool-elo/                         # 慢评测产物
    └── <候选>/
        ├── matches.jsonl             #   逐局结果
        └── challenger-elo.json       #   该版本的池内 Elo 与名次

~/agentbench/runs/<run-id>.out            # driver 日志（★ 排查先看这个）
~/agentbench/runs/<run-id>.pool-elo.log   # 慢评测日志
~/agentbench/runs/<run-id>.final-elo.log  # 终局 Elo 日志
~/agentbench/analysis/<批次>/             # 图与导出的曲线数据
```

排查时的先后顺序：`<run-id>.out`（有没有上游退避/异常）→ `events.jsonl`
（每轮到底产出了什么）→ `official-matches/<...>/*.stderr.log`（某一局里
哪个进程说了什么）。**逐局 stderr 是定位选手崩溃的唯一线索来源。**

日志是**追加**写的（续跑时上一段是排查依据）。`watch_runs.py` 只读最后一次
启动之后的部分 —— 否则上一次失败的错误会被算到这一次头上。

---

## 10. 密钥

全部放在 `AgentBenchHL/.env`，模型档案里只写 `api_key_env`（环境变量名）。
**配置文件里绝不出现 key 本身**，有测试守着这一点。

```bash
# .env（键名 → 哪个模型档案在用）
AGENTBENCH_ROOT=/home/qingle/agentbench/AgentBench   # A 仓路径，配置里用 ${AGENTBENCH_ROOT}

ABHL_KEY_SBTUNNEL=sk-...       # gpt-5.6-sol
ABHL_KEY_GLM=sk-...            # glm-5.2 / glm-5.3
ABHL_KEY_KIMI=sk-...           # kimi-k3
ABHL_KEY_LONGCAT=sk-...        # longcat-2.0（当前中转拒绝，见 §7）
ABHL_KEY_TEAMOROUTER=sk-...    # opus-5（cc harness）
ABHL_KEY_QWEN=sk-...           # qwen3.8（阿里云百炼 DashScope）
ABHL_KEY_DEEPSEEK=sk-...       # deepseek-v4-pro（DeepSeek 官方）
```

**换机器/新环境时先自查这两件事**，否则会在起 run 几十分钟后才炸：

```bash
cd ~/agentbench/AgentBenchHL
V=~/agentbench/.venv/bin/python

# ① 档案自身的一致性（catalog 大小写、压缩阈值、base_url 的 /v1、key 不进配置）
$V -m pytest tests/unit/test_model_profiles.py -q

# ② 端点真的能用（探 /responses，不要用 curl，理由见 §7）
$V scripts/probe_responses_models.py \
  --base-url https://lab.cs.tsinghua.edu.cn/ai-platform/sub2api \
  --api-key-env ABHL_KEY_GLM --models glm-5.2 --rounds 3
```

`opus-5` 还额外需要**本机装了 Claude Code**（`which claude`），因为它走 cc harness。
