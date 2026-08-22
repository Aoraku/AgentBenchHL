# 对手选择方式 ablation —— 运维手册

一句话：**antwar2 + gpt-5.6-sol + k=1 + b=4**，四个 run 只差
`curriculum.opponent_policy` 一个字段，各跑 32 轮。

```
configs/experiments/ablation/ab32-antwar2-random.yaml     random   随机选 4 个
configs/experiments/ablation/ab32-antwar2-self.yaml       self     agent 自己挑 4 个
configs/experiments/ablation/ab32-antwar2-progress.yaml   progress 从 #20 起铺 4 个槽位往上爬
configs/experiments/ablation/ab32-antwar2-fix.yaml        fix      固定打榜单前 4 名
```

单变量性可随时复验（应当只输出 `opponent_policy` 那一行）：

```bash
cd ~/agentbench/AgentBenchHL/configs/experiments/ablation
for p in self progress fix; do
  diff <(grep -v '^#' ab32-antwar2-random.yaml) <(grep -v '^#' ab32-antwar2-$p.yaml)
done
```

## 启动 / 续跑

```bash
cd ~/agentbench/AgentBenchHL
bash scripts/run_hl.sh configs/experiments/ablation/ab32-antwar2-random.yaml   ab32-antwar2-random   32
bash scripts/run_hl.sh configs/experiments/ablation/ab32-antwar2-self.yaml     ab32-antwar2-self     32
bash scripts/run_hl.sh configs/experiments/ablation/ab32-antwar2-progress.yaml ab32-antwar2-progress 32
bash scripts/run_hl.sh configs/experiments/ablation/ab32-antwar2-fix.yaml      ab32-antwar2-fix      32
```

### ★ 续到 64 轮：**再执行一次同样的命令**

第三个参数是"**本次再跑多少轮**"，不是"总共跑到第几轮"。driver 的
`completed` 每个进程从 0 开始数，而"这个 run 是否已经开始过"由磁盘状态
（`runs/<id>/.../state`）判断。所以：

| 命令 | 实际跑的轮次 |
|---|---|
| 第一次 `… 32` | 1 → 32 |
| 第二次 `… 32` | 33 → 64 |
| ~~第二次 `… 64`~~ | ~~33 → 96~~（**不要这样**） |

checkpoint 是自动的：`events.jsonl`（完整事件账本）+ `state`（thread id 与
已完成轮数）+ `snapshots/`（每轮候选代码）。进程被杀、机器重启之后，
用同样的命令就能接着跑。

`run_hl.sh` 会拒绝对同一个 run-id 启动第二个进程：两个进程写同一份
`events.jsonl` 会让对局记录交错、thread 状态互相覆盖，而这从曲线上看不出来。

## 看进度

```bash
cd ~/agentbench/AgentBenchHL
~/agentbench/.venv/bin/python scripts/watch_runs.py \
  --runs-root ~/agentbench/runs --target 32 \
  --run-id ab32-antwar2-random ab32-antwar2-self ab32-antwar2-progress ab32-antwar2-fix
```

它不只报轮数，还会点出**"轮数在涨但其实在空转"**这类看不出来的故障：

* 候选 id 一直不变 → agent 在重交同一份代码；
* 0 回合对局占比过高 → 候选协议格式错（0 回合 = 直接判负，学不到东西）；
* `policy=random/progress/self` 却只打过 ≤4 个对手 → **对手策略没生效，
  四组消融会退化成同一组**（这条最要紧，它直接决定 ablation 有没有意义）；
* `policy=fix` 却打过 >6 个对手 → fix 应当固定打前 b 名。

## 出图

三组曲线（胜率 / Elo / token）：

```bash
~/agentbench/.venv/bin/python scripts/plot_learning_curves.py \
  --run-dir ~/agentbench/runs/ab32-antwar2-{random,self,progress,fix} \
  --out-dir ~/agentbench/analysis/ablation-32 \
  --require-evaluated 0
```

`--require-evaluated 0` 是必需的：这四个 run 的配置里
`evaluation.background_pool: false`，所以没有慢通道（全池实测）数据。
不加这个参数脚本会拒绝出图。

**为什么关掉慢评测**：每个版本打完 229 人池要 458 局。四个 run × 32 轮
= 128 个版本，那是 58k 局，会把机器占满并拖慢迭代本身。等 32 轮跑完、
挑出各组最好的几版再补评测：

```bash
~/agentbench/.venv/bin/python scripts/pool_elo_worker.py --help   # --best-only --iteration-stride 3
```

在那之前，图上能看的是：
* **胜率**（虚线，快通道）：对本轮那 4 个对手的胜率。注意 `progress` 组
  会随 agent 变强而换更强的对手，所以它的胜率**长期贴在 0.5 附近才是正常**
  —— 那条线衡量的是难度而不是绝对强度；
* **Elo**（橙点，零成本反解）：拿本轮那 4 局 + 冻结池锚点逐对手反解再平均；
* **token**：柱=每轮，线=累计。

## 这次踩过的坑（再遇到直接对照）

**1. `403 This account only allows Codex official clients`**

sbtunnel 按**客户端白名单**放行。默认 `originator: agentbench-hl` 会被拒，
必须在模型档案里写 `client_name: codex_exec`（已配好）。

这个 403 极具误导性：端点、key、模型名全都是对的。定位靠的是
`responses_proxy.py` 里的 `[llm-upstream]` 指纹日志——上游返回非 2xx 时
会把 `originator` / `user-agent` 打进 app-server stderr。以后遇到类似问题
先看那行。

**顺带**：探这个站的连通性**必须用 `codex exec`，不能用 curl**。裸 curl
一律 403，会让人误判"端点坏了"。

**2. `ModuleNotFoundError: No module named '_bootstrap'`**

k=1 改造时的回归：曾有一个"只有 1 个候选就直接拿 overlay 当快照"的捷径，
而 agent 的 overlay 只放它改动的文件（实测就 main.py + ai.py），
于是框架自己提供的 `_bootstrap.py` 被落在外面。已修（快照永远是
"工作区 + overlay 叠加"），并有回归测试
`test_k1_candidate_snapshot_keeps_the_runtime_support_files` 锁住。

**3. 配置放子目录导致的路径故障**

把配置放进 `configs/experiments/ablation/` 后，原先写死"往上数 2 层"的
根路径推导全部错位。已改成向上搜索。特别注意其中一处是 `.env` 定位——
它找不到文件时**静默跳过**，后果是 key 没加载、run 起来才报 401，
而那个错误完全指不回真正的原因。
