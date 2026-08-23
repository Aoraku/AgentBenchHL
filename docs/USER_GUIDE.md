# AgentBenchHL 使用手册

给要做**多模型 × 多游戏**测试的同事。读完你应该能独立起一批实验、看懂进度、
出图，并且知道哪些地方会静默出错。

---

## 0. 三十秒上手

```bash
cd ~/agentbench/AgentBenchHL
V=~/agentbench/.venv/bin/python

# 1) 生成配置（不要手写 yaml，理由见 §3）
$V scripts/make_verify_configs.py --out-dir configs/experiments/mine \
  --games antwar2 --model glm-5.2 --iterations 32 --prefix mine

# 2) 起 run（第三个参数是"本次再跑多少轮"）
bash scripts/run_hl.sh configs/experiments/mine/mine-glm-5.2-antwar2.yaml my-run 32

# 3) 看进度
$V scripts/watch_runs.py --runs-root runs --log-dir ~/agentbench/runs \
  --target 32 --run-id my-run

# 4) 出图
$V scripts/plot_learning_curves.py --run-dir runs/my-run \
  --out-dir ~/agentbench/analysis/mine --require-evaluated 0
```

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
game: antwar2                    # 8 个：antwar antwar2 generals miracle
                                 #       rollman snakego aquawar lostspace
provider:
  model_profile: glm-5.2         # configs/models/ 下的档案名
```

模型**独立成档案**（`configs/models/<name>.yaml`），实验里只写一个名字。
为什么这么设计：7 个模型的中转站、api key、上下文窗口、model catalog 全不同，
散在每个实验配置里必然漏字段，而**漏一个 `context_window` 会让 run 在几十分钟后
被压缩打死**。

| 档案 | 中转 | 状态 |
|---|---|---|
| `gpt-5.6-sol` | sbtunnel | ✅ 跑通 32 轮 |
| `glm-5.2` | 清华 | ✅ 8 游戏都验过 |
| `glm-5.3` | 清华 | ✅ 验过（高峰会限流）|
| `kimi-k3` | 清华 | ✅ 验过 |
| `longcat-2.0` | 清华 | ❌ 中转拒绝，见 §7 |
| `opus-5` | ？ | ❌ 未接通，见 §7 |
| `deepseek-v4-pro` / `qwen3.8` | — | ⏸ 等官方 API |

**加新模型时照抄 `glm-5.3.yaml`，不要自己发明字段。** 那份配置来自跑通 40+ 轮的
`sota-antwar2`，每一行都有理由（写在注释里）。三个已经踩过的坑：

* `model` 用了 `model_catalog` 时必须与 catalog 里的 slug **大小写一致**
  （codex 精确匹配，不符就退回兜底元数据 → 压缩提前触发 → glm 系必死）。
  中转本身不区分大小写，所以用 curl 测**测不出来**。
* `auto_compact_token_limit: 900000` —— 这是"引雷开关"不是"保险"，设低了
  会主动触发压缩。曾经写成 90000（少一个 0），两个 run 直接死。
* `client_name` **只有** sbtunnel 这类按客户端白名单放行的中转需要。

这些都有测试守着（`tests/unit/test_model_profiles.py`），配错跑测试就会报。

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
curriculum:
  opponent_policy: progress   # random | self | progress | fix
  batch: 4                    # b：一轮打几个对手
  opponent_start_rank: 20     # progress 的起点名次
```

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
直线；b=4 才有 0/0.25/…/1 九档。这是 `b=4` 作为默认值的主要原因。

### ⑤ 慢评测

```yaml
evaluation:
  background_pool: true      # 是否自动挂慢评测
  pool_stride: 3             # 每几轮抽一版
  challenger_track: null     # 分轨游戏必填！见下
```

**`rollman` 是唯一的分轨（非对称）游戏**，必须写 `challenger_track: rollman`，
指明挑战者扮演哪一轨。漏了会同轨互殴（ghost 打 ghost），那种对局在协议层就
没有意义 —— 实测回放只有 2 行。生成脚本会自动处理。

短 run（≤ 5 轮）要把 `pool_stride` 设成 1，否则一个数据点都拿不到。

---

## 3. 配置怎么生成（不要手写）

```bash
V=~/agentbench/.venv/bin/python

# 多游戏（固定模型）
$V scripts/make_verify_configs.py --out-dir configs/experiments/mine \
  --games antwar antwar2 generals miracle rollman snakego aquawar lostspace \
  --model glm-5.2 --iterations 32 --prefix mine

# 多模型（固定游戏）
$V scripts/make_verify_configs.py --out-dir configs/experiments/mine \
  --models glm-5.2 glm-5.3 kimi-k3 gpt-5.6-sol --game antwar2 \
  --iterations 32 --prefix mine
```

**手写 8~12 份 yaml 必然漂移**，而漂移在图上看不出来：只要有人改了其中一份的
`match_timeout_s`，那组数据就再也不可比，但曲线看起来一切正常。

生成后建议 diff 复验一遍，确认只差被测维度：

```bash
cd configs/experiments/mine
for f in mine-glm-5.3-antwar2.yaml mine-kimi-k3-antwar2.yaml; do
  diff <(grep -vE '^ *#' mine-glm-5.2-antwar2.yaml) <(grep -vE '^ *#' $f) | grep '^[<>]'
done
```

### 多模型 × 多游戏时的并发上限

**一次最多铺 6~8 个 run。** 两个硬约束，都实测撞过：

1. **机器**：32 核。一个 run 占 8 个对局并发，加上 agent 思考与慢评测，
   铺 12 个会让 load 冲到 75，所有东西一起变慢。
2. **中转账号并发**：铺 12 个时报
   `Concurrency limit exceeded for account`，直接让 run 失败。

推荐做法：**先跑通 1 个确认配置无误，再分批铺开。**

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

**`--parallel` 怎么定**：32 核，留 6 给别人和 agent 思考，每局约 1 核。
四个 run 并行时每个给 6（共 24）比较合适。别忘了机器是共用的：

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

```bash
$V scripts/plot_learning_curves.py \
  --run-dir runs/run-a runs/run-b runs/run-c \
  --out-dir ~/agentbench/analysis/mine \
  --require-evaluated 0
```

`--require-evaluated 0` 表示"即使没有慢评测数据也出图"。不加的话脚本会拒绝
出图 —— 那是为了防止用半份数据画出会被误读的曲线。

每个 run 出一张 4×2 的图（4 个指标 × 2 种横坐标：迭代轮数 / 看过的轨迹数），
多个 run 另出一张对比图。

### 四个指标

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

### opus-5：端点无权

sbtunnel 的 `/v1/models` 只列 OpenAI 家模型，没有任何 claude/opus。
`/v1/messages` 返回 `403 This group does not allow /v1/messages dispatch`
—— 端点**存在**但当前 key 的分组无权。换路径是白费时间。

需要三者之一：有权限的 key / 确认走 `harness: cc`（Claude Code，配置形状
完全不同）/ 另一个中转端点。

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

## 9. 目录结构

```
runs/<run-id>/
├── events.jsonl          # 事件账本 —— 所有分析的唯一事实源
├── run-config.json       # 冻结的实验配置（续跑校验用）
├── run-manifest.json     # 实际生效的参数
├── workspace/            # agent 的工作区
│   ├── main.py           #   当前策略
│   ├── feedback/         #   回放（replay.md 是自然语言的）
│   └── .agentbench/      #   action.json / rollouts/
├── snapshots/<候选>/      # 每轮的候选代码（含运行时支撑文件）
├── codex-home/           # codex 的 config.toml / models.json / 会话历史
└── pool-elo/             # 慢评测产物
    └── <候选>/
        ├── matches.jsonl        # 逐局结果
        └── challenger-elo.json  # 该版本的池内 Elo 与名次

~/agentbench/runs/<run-id>.out            # driver 日志
~/agentbench/runs/<run-id>.pool-elo.log   # 慢评测日志
```

日志是**追加**写的（续跑时上一段是排查依据）。`watch_runs.py` 只读最后一次
启动之后的部分 —— 否则上一次失败的错误会被算到这一次头上。

---

## 10. 密钥

全部放在 `AgentBenchHL/.env`，模型档案里只写 `api_key_env`（环境变量名）。
**配置文件里绝不出现 key 本身**，有测试守着这一点。

```bash
# .env
ABHL_KEY_SBTUNNEL=sk-...
ABHL_KEY_GLM=sk-...
ABHL_KEY_KIMI=sk-...
ABHL_KEY_LONGCAT=sk-...
```
