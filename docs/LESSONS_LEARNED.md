# HL 迭代框架：经验教训

> 这份文档记录 antwar / antwar2 两个游戏从零跑通 SOTA 迭代过程中踩到的**全部**坑。
> 每条都是实测事故，不是理论推演。broadcast 到新游戏前请通读——
> 其中 B、C、D、E 四条会静默污染科学结论或直接打死长跑。

## 目录

- [A. codex remote compaction 与 glm 不兼容](#a-codex-remote-compaction-与-glm-不兼容)
- [B. 探针被候选包 SDK 的边界 bug 掀翻](#b-探针被候选包-sdk-的边界-bug-掀翻)
- [C. token 计量在会话轮转下低估 29 倍](#c-token-计量在会话轮转下低估-29-倍)
- [D. 幂等键冲突打死 run](#d-幂等键冲突打死-run)
- [E. 上游重试覆盖不全](#e-上游重试覆盖不全)
- [F. 探针计数差 1 导致整局丢弃](#f-探针计数差-1-导致整局丢弃)
- [G. 二值胜率丢掉全部梯度](#g-二值胜率丢掉全部梯度)
- [H. 池评测被按版本串行](#h-池评测被按版本串行)
- [I. 人类池样本量撑不起"排第几"](#i-人类池样本量撑不起排第几)
- [J. 两个 Elo 口径不可混用](#j-两个-elo-口径不可混用)
- [K. 用半份数据出图](#k-用半份数据出图)
- [方法论教训](#方法论教训)

---

## A. codex remote compaction 与 glm 不兼容

**症状**：两个 run 同时死于 `remote compaction fatal error`。

**根因**：codex 0.147 的 `POST /responses/compact` 要求响应里**恰好一个** compaction
output item，而 glm 系列返回 `[reasoning, message]` 两个 item。只要上下文涨到压缩线，
压缩必死。这不是我们的 bug，是 harness 与模型的协议错配。

**修法**（两条同时上，缺一不可）：

```yaml
runtime:
  thread_rotate_each_iteration: true   # 每轮换新 thread，上下文永远到不了压缩线
provider:
  model_catalog: zhipu                 # 用厂商官方 models.json，让 codex 认识 glm-5.3
  auto_compact_token_limit: 900000     # 压缩线设得足够高
```

**为什么两条都要**：只设 `auto_compact_token_limit` 不够——不给 `model_catalog` 时
codex 会用 fallback 元数据，压缩在一个我们没设过的点触发（实测 antwar2 死在 97k）。
只靠轮转也不够——阈值挡不住"单个 turn 内部"的上下文增长。

**broadcast 注意**：换任何非 OpenAI 模型都要重新确认这一点。判据是
`codex debug models` 是否认识该模型名。

---

## B. 探针被候选包 SDK 的边界 bug 掀翻

**严重性：会静默污染 IG 指标，且极难发现。**

**症状**：`behavioral_ig` 一直是字母表近似（|A| ≡ 10），精确支撑集只覆盖 9/41 轮。
表面上看不出任何错误，日志里也没有 traceback。

**根因**：候选包自带的 SDK 在满级状态下抛异常而不是返回 False：

```python
# antwar/gamedata.py
def upgrade_cost(level):
    return [200, 250][level]     # level == 2（已满级）⇒ IndexError
```

枚举支持集必须把**每个**候选动作都问一遍 `is_operation_valid`，于是一个格子的崩溃
掀翻整局探针，这一轮静默退回近似口径。antwar 有 11 轮（it18-20、22-28）因此丢失。

**修法**：合法性判定自身抛异常时**按非法处理**。

```python
def _is_legal(state, player, operation) -> bool:
    try:
        return bool(state.is_operation_valid(player, operation))
    except Exception:      # SDK 边界 bug，视为非法
        return False
```

语义上这也是对的：一个连合法性都算不出来的操作，提交上去同样会被后端拒绝。

**效果**：antwar 精确 IG 覆盖 9 → 40/41 轮，antwar2 17 → 47/54 轮。
antwar 精确 IG 均值 0.2767 vs 近似 0.2067 —— **近似口径系统性偏低 25%**。

**broadcast 注意**：每个新游戏写完探针后，必须验证精确支撑集的覆盖率。
低于 90% 就说明有类似的 SDK 边界问题，不要接受"反正有近似兜底"。

---

## C. token 计量在会话轮转下低估 29 倍

**严重性：会让 token 预算守卫永久失效。**

**症状**：token 曲线呈阶梯状（只在"出现更贵的段"时才上升）。

**根因**：harness 报的 `total_tokens` 是**会话累计值**语义（codex 的
`tokenUsage/updated` 给的是当前 thread 的总量，单调增）。所以单个 thread 内要取
**峰值**——求和会把同一份累计量重复叠加。

但 A 条的修复（`thread_rotate_each_iteration`）把这件事变成了分段问题：每轮换新
thread，计数**归零重来**。此时对全 run 取全局 max，等于只记住"最贵的那一段"。

| run | 原报数 | 真实值 | 倍数 |
|---|---|---|---|
| sota-antwar | 169,515 | 4,797,274 | 28× |
| sota-antwar2 | 223,677 | 7,997,094 | 36× |

**修法**：**段内取峰值，跨段求和**。分段边界用会话事件识别
（`GoalLedStarted` / `GoalSessionReset` / `GoalSessionRotated`）。

```python
segments, current = [], None
for event in events:
    if event.event_type in ("GoalLedStarted", "GoalSessionReset", "GoalSessionRotated"):
        if current is not None:
            segments.append(current)
        current = None
    elif isinstance(value := payload.get("total_tokens"), int):
        current = value if current is None else max(current, value)
return sum(segments) + (current or 0)
```

**为什么没炸**：只是因为当时 `budget.tokens: null`。如果设了上限，守卫会拿着一个
低估 29 倍的数，永远不触发。

**broadcast 注意**：这是一条**通用**教训，不限于 antwar。任何"累计值语义 + 分段
重置"的指标都有同样的陷阱。

---

## D. 幂等键冲突打死 run

**症状**：`ValueError: conflicting idempotency key: agent-usage:64`，第 43 轮，
整个 run 终止。丢失的信息只是几条 token 计数。

**根因**：幂等键用了 harness **内存列表下标**（`agent-usage:{index}`）。那个下标只在
当前进程里单调，而事件账本是跨进程持久的：

1. `resume` 起新进程，harness 的 `events` 从 0 重新计数；
2. `agent-usage:0` 带着一份**不同的** payload 再写一次；
3. 事件存储对"同键不同 payload"抛 `ValueError`（这个严格性是对的）；
4. run 死于一个纯记账问题。

**修法**（两层）：

1. 序号改为按**账本里已有的同类事件条数**发号 —— 只依赖持久状态，重启自然接续；
2. 区分**科学事件**与**遥测事件**：

```python
def _append_telemetry(self, event_type, payload, key):
    """遥测撞键只记诊断，绝不打断迭代。"""
    try:
        self._append(event_type, payload, key)
    except ValueError as error:
        with suppress(ValueError):
            self._append("TelemetryAppendSkipped", {...}, f"telemetry-skipped:{key}")
```

**关键判断**：对局结果 / 指标 / 快照撞键**必须**抛错（说明有真实的重复记账）；
token 用量这类遥测撞键只该记诊断。为遥测终止一次几百万 token 的长跑是代价错配。

---

## E. 上游重试覆盖不全

**症状**：两个 run 各死于一种上游抖动。

**根因 1：读响应体的失败根本没进重试路径。** 退避只包住了 `urlopen`，而 glm 的长响应
经常在 `response.read()` 阶段断流（`http.client.IncompleteRead`）。那时 urlopen 早已
成功返回、退避层已经退场，异常直接冒到 HTTP 处理线程。sota-antwar 就是这样死的
（502 重试成功后，读 205 KB 响应体时断掉）。

**根因 2：退避预算太短。** 原来 `attempts=5, max_delay=30` 的总等待上限约 25 s，
而中转站的限流窗口是**分钟级**。sota-antwar2 连续 15 次 429/503 把额度用光后终止。

**修法**：

1. 新增 `request_bytes_with_backoff()`：把**读体也放进重试循环**，保证"一次完整的
   请求-响应"具备原子的重试语义；
2. 退避改为按**总预算**驱动（默认 10 分钟），次数只是上限而非瓶颈。限流是
   **等得起**的故障，等 10 分钟远好过丢掉一轮几十万 token 的迭代；
3. 代理层兜底：重试耗尽翻译成 502 交回 CLI，不让异常冒进 HTTP 线程。

```python
RETRYABLE_READ_ERRORS = (
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    ConnectionError,
    TimeoutError,
)
```

---

## F. 探针计数差 1 导致整局丢弃

**根因**：探针枚举出 246 个决策点，线协议记录 247 个，
`len(probed) >= len(decisions)` 为假 → 整局数据丢弃。

**修法**：引入缺口容差 `SUPPORT_ALIGN_ABS_TOLERANCE = 2`；同时让
`observed_support()` 盖住静态声明（避免 `support_mode` 被覆盖），
并新增 `support_exact_decisions` / `support_exact_fraction` / `support_notes`
三个字段，让"精确到什么程度"可观测而不是二值通过/失败。

---

## G. 二值胜率丢掉全部梯度

**症状**：antwar2 对 rank1 连续 15 轮胜率恒为 0，看起来"没有信号"。

**判断错误**：这不是二值 reward 的 RL。`score_margin`（终局分差）和 `rounds`
（撑住多少回合）都是逐局连续量，**早就落在事件里，只是从来没端到 agent 面前**。
那 15 轮里分差完全可能在收窄——那就是有梯度的。

**更严重的连带问题**：`best_candidate_id` 原来只按得分率排序。全败轮里所有候选
得分率都是 0，`best` 退化成**字典序**，下一轮的基线是随机挑的。**这才是真正的
0 信号。**

**修法**：

1. 汇总新增 `margin_mean` / `margin_best` / `margin_by_candidate`（逐候选的
   平均/最好/最差分差 + 平均回合数）；
2. 反馈正文加「逐候选分差」段落，按分差排序；
3. `best_candidate` 排序键改为 `(得分率, 平均分差)` —— 分差破平。

```python
key = (entry["points"] / played, entry["margin"] / played)
```

**broadcast 注意**：新游戏的 `MatchResult` 必须提供 `score_margin`。
没有连续量的游戏会退化成真正的二值反馈。

---

## H. 池评测被按版本串行

**症状**：32 核机器上，给每个版本打全池 Elo 慢得离谱。

**根因**：`arena.run_case(case, candidate_root)` 的候选包是**逐次传入**的，
所以一个 arena 天然可以服务所有版本。但第一版实现给每版建独立 arena、串行排队：

1. **队尾饥饿**：一个版本剩最后 3 局时只有 3 个线程在干活，其余核空转。
   版本越多，尾巴累积越多；
2. **并发上限被版本内局数卡住**：188 局的版本最多并行 188 路，但真正的上限
   应该是 `总核数 / 每局核数`。

**修法**：摊平成 `(版本 × 对手 × 座次 × seed)` 的**全局扁平队列**
（`evaluate_many()`）。任何时刻都有足够待办填满线程池。

**测试怎么锁**：每版本 6 局；若按版本切断，并发峰值最多 6。
断言 `peak_concurrency > 6`。

---

## I. 人类池样本量撑不起"排第几"

**严重性：会让所有排名结论失效。**

**症状**：审计发现 9 个池全部是 `degree=6`（每人只打 12 局）。

**量化后果**（`scripts/audit_pool_elo.py`）：

| 指标 | antwar | antwar2 |
|---|---|---|
| Elo 标准误中位 | ±115.8 | ±106.4 |
| 前十相邻分差中位 | 20.8 | **5.1** |
| 胜率 = 1.000 的选手 | 3（前十占 3） | 12（**前十占 8**） |
| Elo 不可辨识 | 8 | 18 |

**结论**：误差比信号大 5~20 倍，**前十的具体顺序完全不可辨识**。
"我们排第 5"的真实含义只是"我们在前十这一档"。

而且 antwar2 前十里 8 个从没输过任何一局 —— 对"零失败"选手，BT 只能给出下界，
真实强度不可辨识。

**degree 怎么定**：不要拍脑袋。BT 标准误 ∝ 1/√n，所以
`degree_new = degree_old × (se_old / se_target)²`。压到 ±50 需 `degree ≈ 28~33`。
`scripts/audit_pool_elo.py --target-se 50` 会直接给出这个数。

**验证修复有效**：重测后 antwar 胜率饱和从 3 人 → **0 人**，
generals 从 3 人 → 1 人。饱和消失说明强度重新变得可辨识。

**broadcast 注意**：新游戏跑 `abhl ladder eval` **不要用默认 degree=6**，
直接上 28。事后重测的代价是整条排名结论作废。

---

## J. 两个 Elo 口径不可混用

这是**概念**问题，不是 bug，但混用会得出完全错误的结论。

| 口径 | 定义 | 适用 | 陷阱 |
|---|---|---|---|
| `pool_elo` | 该 run **迄今全部**对局的累计 BT-MLE | 看整条轨迹的平均位置 | 含早期连败与被证伪的探索候选，**不是当前实力**；第 42 轮时分母 300+ 局，再加 8 局推不动 → 天然平滑 |
| 静态池 Elo | **单个版本**独立打全池，以池选手 measured_elo 为固定锚点 | 回答"这一版有多强" | 样本量小时被正则先验顶到固定值 |

**实测对比**（同一批版本，2 局估计 vs 188 局实测）：

| 版本 | 2 局估计 | 188 局实测 | 误差 |
|---|---|---|---|
| v039_p1_mortar_plus | 1183 / #1 | 1077.95 / #5 | 4 个名次 |
| v040_postseal_sprint | 992 / #7 | 750.00 / #22 | **15 个名次** |
| v041_aggressive_emp | 992 / #7 | 879.19 / #12 | 5 个名次 |

**两条硬规则**：

1. **画曲线只用静态池实测**。迭代过程里对单个对手的 2 局胜率不能当能力指标；
2. **人类池永远冻结**。挑战者的胜负绝不回写 `measured_elo.json`——否则我们自己的
   策略会改变尺子，"在人类池里排第几"失去意义。每份结果带 `pool_fingerprint`
   （锚点集合哈希），池子一变旧结果显式标为不可比。

**还暴露了一个更深的问题**：v039 → v040 一轮之间掉 328 分 / 17 个名次，
而主循环用「对当轮 1 个对手打 2 局的胜率」选 best candidate，**根本分辨不出**
这个差距。迭代不是单调上升，是在剧烈震荡，而主循环看不见。

---

## K. 用半份数据出图

**教训**：后台队列只跑完 3/42 个版本时就出了图。半份数据画出的曲线会被误读为
"能力在下降"，而实际只是"后面的版本还没测"。

**修法**：绘图脚本加 `--require-evaluated N` 护栏，覆盖不足直接拒绝出图并说明原因：

```
[antwar] 跳过出图：全池评测只覆盖 3 轮，低于要求的 5 轮。
```

**原则**：数据没到就是没到。宁可图上少几个点，也不要画一条会被误读的曲线。

---

## 方法论教训

这些比具体 bug 更重要。

### 1. 指标口径必须自带出处

每个数字都要能回答"这是用什么样本、什么锚点、什么公式算出来的"。
本项目的做法是让每份结果携带 `provenance` 字段与指纹：

```json
"provenance": "challenger vs frozen human pool, both seats; anchors read-only
               from players/measured_elo.json (never rewritten); independent of
               the run's conquest matches and of other candidates"
```

### 2. 静默降级是最危险的失败模式

B 条（探针崩溃退回近似）和 C 条（token 低估）都**没有任何报错**，
只是数字悄悄变得没意义。凡是有"精确路径 + 近似兜底"的地方，
都必须把"实际走了哪条路"记成可观测字段（如 `support_mode`、
`support_exact_fraction`），并定期检查覆盖率。

### 3. 区分"科学事件"与"遥测"

科学事件（对局结果、指标、快照）要严格幂等，撞键立即暴露。
遥测（token 用量、耗时）撞键只记诊断。混为一谈的代价见 D 条。

### 4. 发现口径有问题就重测全部，不要只审计

审计出人类池不可信之后，正确动作是**重测全部 8 个游戏**，
而不是写一份报告说"这个数不可信"。带着已知错误的尺子继续跑，
后面所有结论都要作废重来。

### 5. 并发设计要看机器而不是看数据结构

H 条的错误本质是"让数据的自然分组（按版本）决定了并发粒度"。
正确做法是先算出机器能承载多少并发（`总核数 / 每局核数`），
再把工作摊平到那个宽度。

### 6. 基建抖动不能表现为"模型能力不足"

A、D、E 三条都会让事件账本记成"这一轮 agent 没产出候选"。
如果不在最底层把这类混淆消掉，后面分析学习曲线时会把基建问题
误读成模型行为。
