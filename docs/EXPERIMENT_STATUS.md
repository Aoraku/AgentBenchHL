# 进行中的实验与交接状态

> 最后更新：2026-08-23。这份文档的用途是**上下文丢失后能立刻接手** ——
> 每个正在跑的东西是什么、验收标准是什么、怎么看进度、怎么救。

---

## A. 正在跑：对手选择方式 ablation（antwar2 × gpt-5.6-sol）

### 它要回答什么

四种对手选择方式，**在相同迭代轮数下，哪一种让 agent 在整个人类静态池里
达到更高水平**。

⚠️ 评价标准不是"它对自己挑的那批对手赢得多不多"。`fix` 组固定打榜单前 4 名，
胜率长期是 0；`random` 组随机抽人，胜率能到 0.5。这两个数字**不可比** ——
它们的对手难度差了一个量级。唯一可比的标尺是**全池实测 Elo**（慢评测产出）。

### 四个 run

| run-id | policy | 含义 |
|---|---|---|
| `ab32-antwar2-random` | random | 每轮从全池随机抽 4 个 |
| `ab32-antwar2-self` | self | agent 自己读榜 + 读战绩挑 4 个 |
| `ab32-antwar2-progress` | progress | 从第 20 名起铺 4 个槽位往上爬 |
| `ab32-antwar2-fix` | fix | 固定打榜单前 4 名 |

配置在 `configs/experiments/ablation/ab32-antwar2-<policy>.yaml`，
**四份只差 `curriculum.opponent_policy` 一个字段**（已用 diff 逐字节验证，
也用实际运行产出的 `run-manifest.json` 复验过）。

共同设置：`k=1` / `b=4` / 32 轮 / `history_mode: full` / reasoning `high` /
`background_pool: true`（stride=3）。

### 进度（2026-08-23 17:00）

| 组 | 轮数 | 状态 |
|---|---|---|
| self | 32/32 | ✅ 完成（`stop_reason: iterations_exhausted`）|
| progress | 32/32 | ✅ 完成 |
| fix | 30/32 | 🔄 在跑 |
| random | 26 + 续跑 6 | 🔄 在跑（第 26 轮曾被限流误杀，已修因并救回）|

### 怎么看进度

```bash
cd ~/agentbench/AgentBenchHL
~/agentbench/.venv/bin/python scripts/watch_runs.py \
  --runs-root runs --log-dir ~/agentbench/runs --target 32 \
  --run-id ab32-antwar2-{random,self,progress,fix}
```

它会点出"轮数在涨但其实在空转"这类看不出来的故障：候选 id 不变、
0 回合对局占比高、对手策略没生效。

### 怎么续跑（32 → 64）

```bash
bash scripts/run_hl.sh configs/experiments/ablation/ab32-antwar2-random.yaml \
  ab32-antwar2-random 32
```

⚠️ 第三个参数是"**本次再跑多少轮**"，不是"总共跑到第几轮"。
driver 的 `completed` 每进程从 0 数，"是否已开始过"由磁盘状态判断。
所以再执行一次 `32` 就是接着跑 33→64。

### 怎么出图

```bash
~/agentbench/.venv/bin/python scripts/plot_learning_curves.py \
  --run-dir runs/ab32-antwar2-{random,self,progress,fix} \
  --out-dir ~/agentbench/analysis/ablation-32 --require-evaluated 0
```

四组曲线：胜率 / Elo / 分差 / token。Elo 面板有两条线 ——
橙点是零成本反解（本轮那几局 + 冻结池锚点），绿线是全池实测（慢评测）。
**结论要读绿线。**

### 已知的坑（都已修，但值得记住）

1. **`fix` 组的 Elo 曾是一条假平线**（14 轮全是 1431.37）。根因是旧实现把
   胜率钳到 `[0.02, 0.98]`，全败时恒等于 `2107.5 − 676`。已改用
   `estimate_pool_elo`（锚定 BT/MLE + 正则）。
2. **`random` 组第 26 轮死于 checkpoint 超时**，但真凶是 503 退避吃掉了
   固定墙钟预算。已让 app-server 把退避时间加回 deadline。
3. **慢评测曾被饿死**（35 版排队只完成 1 版）：9 个早已跑完的 run 的 worker
   还在空转 `--parallel 6`。已加自动退出。

---

## B. 正在跑：多模型跑通（任务 1）

目标：每个模型用 antwar2 跑 **2 轮**（`b=4` / progress / 用历史 / high），
确认有迭代数据、可画图、慢评测可用。

run-id 是 `verify-<模型档案名>`，配置由脚本生成（见 §E）。

### 模型档案与实测状态

档案在 `configs/models/<name>.yaml`，实验里只写 `provider.model_profile`。

| 档案 | 中转 | 模型名 | 状态 |
|---|---|---|---|
| `gpt-5.6-sol` | sbtunnel | `gpt-5.6-sol` | ✅ 已跑完 32 轮 ablation |
| `glm-5.3` | 清华 | `GLM-5.3` | 🔄 2 轮验收中 |
| `glm-5.2` | 清华 | `GLM-5.2` | ✅ 8 游戏烟测主用模型 |
| `longcat-2.0` | 清华 | `LongCat-2.0` | 🔄 2 轮验收中 |
| `kimi-k3` | 清华 | `kimi-k3` | 🔄 2 轮验收中 |
| `opus-5` | ？ | ？ | ❌ **阻塞，见下** |
| `deepseek-v4-pro` | — | — | ⏸ 等官方 API |
| `qwen3.8` | — | — | ⏸ 等官方 API |

### ★ `client_name: codex_exec` 是所有中转的默认动作

**每个模型档案都要写它。** 不写的话上游看到的 originator 是框架默认的
`agentbench-hl`，而中转站可能按客户端白名单放行。两种被拒形态都遇到过，
**报错都指向错误的方向**：

* sbtunnel：`403 This account only allows Codex official clients`
* 清华 + LongCat-2.0：`400 Model is not supported by composite groups`
  —— 这句话听起来像"模型不支持"，实际是"你这个客户端不被允许调这个模型"。

同一中转上 kimi-k3 与 GLM-5.x 不写也能通（它们不在受限分组里），
但没有理由赌哪个模型受限。

**排查这类问题时的一个陷阱**（踩过，浪费很久）：用 `codex exec` 探测**永远
测不出来**，因为 codex CLI 自己就报 `originator: codex_exec`，而框架走
app-server 报的是 `agentbench-hl` —— 那才是唯一没被覆盖的组合。
反过来，用 curl 复现也会失败，但那是**另一个原因**（缺 `tools` /
`instructions` / `client_metadata` 等 codex 请求体字段），不能用来
否证 originator 假设。

真正有效的线索来自 `responses_proxy.py` 的 `[llm-upstream]` 日志：
上游返回非 2xx 时它会把 originator 与 user-agent 打进 stderr。

### opus-5 的阻塞点

sbtunnel 的 `/v1/models` **只列 OpenAI 家模型**，没有任何 claude/opus。
探测 `/v1/messages`（Anthropic 原生协议）得到：

```
403 {"error":{"message":"This group does not allow /v1/messages dispatch",
              "type":"permission_error"}}
```

端点**存在**，是当前 key 所属**分组**无权。所以问题不在路径也不在协议 ——
换路径试是白费时间（`/anthropic/v1/models` 返回站点 HTML 首页）。

需要人确认三者之一：
1. 给一个有 `/v1/messages` 权限的 key（或把现有 key 加进那个分组）；
2. 确认 opus5 走 **Claude Code harness**（`harness: cc`）而不是 codex ——
   那条路不经过 `/responses`，配置形状完全不同；
3. 给一个别的中转端点。

### opus-5 的阻塞点

sbtunnel 的 `/v1/models` **只列 OpenAI 模型**，没有任何 claude/opus。
探测 `/v1/messages` 得到：

```
403 {"error":{"message":"This group does not allow /v1/messages dispatch"}}
```

端点**存在**但当前 key 的分组无权。所以需要：另一个 key / 另一个分组 /
或者确认 opus5 走 Claude Code harness（`harness: cc`）而不是 codex。
**这一项需要人确认，不要瞎猜端点。**

### 两个中转的关键差异（配错就是 403/404，且报错极具误导性）

| | sbtunnel | 清华 sub2api |
|---|---|---|
| `base_url` | `.../v1`（带 v1）| `.../sub2api`（**不带** v1）|
| 客户端白名单 | 要 `client_name: codex_exec` | 不需要 |
| `requires_openai_auth` | 不需要 | **要 `true`** |

* sbtunnel 对默认 originator（`agentbench-hl`）返回
  `403 This account only allows Codex official clients`；
* 探连通性**必须用 `codex exec`，不能用 curl** —— 裸 curl 一律 403，
  会让人误判"端点坏了"。

---

## C. 正在跑：8 游戏烟测（任务 2）

用 `glm-5.2` 在新框架（`k=1` / `b=4` / progress / 用历史）下把 8 个游戏
各跑 2 轮，要有日志、有信号。run-id 是 `smoke8-<游戏>`。

8 个游戏：`antwar` `antwar2` `generals` `miracle` `rollman` `snakego`
`aquawar` `lostspace`。

### 分轨游戏要注意

`rollman` 是非对称的（rollman / ghost 两轨），配置里必须写
`evaluation.challenger_track: rollman`。漏了会同轨互殴（ghost 打 ghost），
那种对局在协议层就没意义 —— 实测回放只有 2 行、IG 恒为常数，
当时排查了很久才定位。生成脚本已按 `CHALLENGER_TRACKS` 自动处理。

### progress 起点统一 20（先查过池子大小）

实测各游戏人类池：miracle 305 / antwar2 229 / aquawar 194 / lostspace 133 /
snakego 123 / rollman 111 / antwar 94 / generals 81 —— 全部远大于 20。

为什么要先查：起点超过池子大小时窗口会被夹到榜首，"渐进课程"会悄悄退化成
"一上来就打第一名"（那是 fix 而不是 progress）。而从曲线上只看得到
"胜率一直是 0"，看不出课程根本没在渐进。

### 分差面板不适用于三个游戏

实测 `score_margin` 取值个数：generals 只有 `{−1, +1}`（**分差就是胜负**）、
lostspace `{−3, 0}`、aquawar `{−2, 0}`。绘图脚本按实测取值个数自动判断
（阈值 4 = 严格高于胜/平/负三档），不适用的留白并写明原因，不用手动配。

---

## E. 配置怎么生成（不要手写）

```bash
V=~/agentbench/.venv/bin/python

# 任务 1：换模型，固定 antwar2
$V scripts/make_verify_configs.py --out-dir configs/experiments/verify \
  --models gpt-5.6-sol glm-5.3 longcat-2.0 kimi-k3 --iterations 2 --stride 1

# 任务 2：换游戏，固定 glm-5.2
$V scripts/make_verify_configs.py --out-dir configs/experiments/smoke8 \
  --games antwar antwar2 generals miracle rollman snakego aquawar lostspace \
  --model glm-5.2 --iterations 2 --stride 1 --prefix smoke8
```

**手写 8~12 份 yaml 必然漂移**，而漂移在图上看不出来：有人改了其中一份的
`match_timeout_s`，那组数据就再也不可比。生成后用 diff 复验只差被测维度：

```bash
cd configs/experiments
for f in verify/verify-{glm-5.3,longcat-2.0,kimi-k3}-antwar2.yaml; do
  diff <(grep -vE '^ *#' verify/verify-gpt-5.6-sol-antwar2.yaml) \
       <(grep -vE '^ *#' $f) | grep '^[<>]'
done
```

`--stride 1` 对短 run 是必需的：验收只跑 2 轮，默认 stride=3 会让慢评测
一个数据点都拿不到（第 1 轮总保留，但第 2 轮会被跳过）。

---

## D. 通用运维

```bash
# 起 run（第三参数 = 本次再跑多少轮）
bash scripts/run_hl.sh <config> <run-id> <iterations>

# 给已在跑的 run 补挂慢评测（新 run 由 background_pool: true 自动挂）
bash scripts/attach_slow_eval.sh <run-id> [stride] [game]

# 看进度
python scripts/watch_runs.py --runs-root runs --log-dir ~/agentbench/runs \
  --target <N> --run-id <run-id>...
```

### 机器是共用的

`agentlab` 上有别人的进程。曾经有另一位用户的 VSCode `rg` 索引占了约 23 核
（`--follow` 全盘扫），把慢评测彻底饿死。**动别人的进程前要先问。**

检查：`ps -eo pcpu,args --sort=-pcpu | head`

### 密钥

全部放在 `AgentBenchHL/.env`，模型档案里只写 `api_key_env` 名字。
**配置文件里绝不出现 key 本身。**
