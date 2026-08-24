#!/usr/bin/env python3
"""生成"跑通验收"用的实验配置。

两种用途，共用一套模板以保证除了被测维度之外**完全一致**：

* ``--models``：固定 antwar2，换模型（任务 1，验证各中转/模型接得通）；
* ``--games``：固定 glm-5.2，换游戏（任务 2，验证 8 个游戏在新框架下跑得通）。

为什么要脚本生成而不是手写 8~11 份 yaml
--------------------------------------
手写必然漂移。之前 ablation 的四份配置就是逐字节对比过才敢下结论；
一旦有人手改其中一份的 ``match_timeout_s``，那组数据就再也不可比，
而这件事**在图上看不出来**。脚本生成 + 生成后 diff 校验是唯一可靠的做法。

分轨游戏的处理也在这里：``rollman`` 必须带 ``challenger_track``，
漏了会同轨互殴（ghost 打 ghost），协议层就没意义（实测回放只有 2 行）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: 非对称（分轨）游戏 → 挑战者扮演哪一轨。
#:
#: 只有这一个游戏是分轨的；其余 7 个对称，写了反而会被 worker 当成非法轨道名。
CHALLENGER_TRACKS = {"rollman": "rollman"}

#: progress 课程的起点名次。
#:
#: 统一 20，因为实测**每个游戏的人类池都远大于 20**：
#:
#:     miracle 305 / antwar2 229 / aquawar 194 / lostspace 133 /
#:     snakego 123 / rollman 111 / antwar 94 / generals 81
#:
#: 为什么要先查再定：起点超过池子大小时，窗口会被夹到榜首，于是"渐进课程"
#: 悄悄退化成"一上来就打第一名"——那是 fix 而不是 progress，消融变量不干净，
#: 而且从曲线上只看得到"胜率一直是 0"，看不出课程根本没在渐进。
#: （deepclue 池是 0 人，它还没接入，不在这 8 个里。）
DEFAULT_START_RANK = 20

TEMPLATE = """\
# 自动生成（scripts/make_verify_configs.py），不要手改。
#
# 用途：{purpose}
#
# 除了被测维度（{axis}）之外，所有字段与同批其它配置**逐字节一致**——
# 手写必然漂移，而漂移在图上看不出来。要改就改生成脚本再重新生成。
schema_version: '1.1'

game: {game}
origin: from_scratch

provider:
  model_profile: {model}

runtime:
  max_iterations: {iterations}
  codex_binary: codex
  # harness 可执行文件：agent_binary 优先，回落 codex_binary。
  # cc harness（opus-5）要写成 claude —— 不写的话会去执行 codex，
  # 而 codex 不认识 Claude Code 的参数，报错指向的方向完全是错的。
  agent_binary: {agent_binary}
  branch_width: 1
  # k = 一轮提交几个候选策略。
  #
  # k>1 的价值不是"一轮多试几个"，而是**并行假设检验**：一轮把 k 条不同的
  # 取胜路径同时下水，下一轮把胜出那条变成所有候选的共同底盘，再从那里分叉。
  # 一轮拿到 k bit 而不是 1 bit。
  #
  # 实测对照（antwar2，同一个人类池）：
  #   k=4 + 单对手 progress → 第 3 轮进池内 #84，第 9 轮 #24，第 21 轮 #10
  #   k=1 + b=4 对手        → 第 30 轮才 #107
  # b 个对手只是把同一个策略的评估变精确（降方差），**不产生新的候选假设**，
  # 所以它替代不了 k。
  rollout_k: {rollout_k}
  # 一轮的对局数 = k × b × 座次 × seed。
  match_parallelism: {parallelism}
  # 各游戏单局长度差一个数量级（miracle 约 7s，snakego 实测 246s）。
  # 420s 会误杀长局，而那在结果里看起来像"这个模型不会玩这个游戏"。
  match_timeout_s: 1800
  network_access: disabled
  # 每轮换 thread：这是上下文控制的主要手段（轮末清零），也顺带躲开 codex 的
  # remote compaction。工作区侧历史一份不动，所以不影响 history_mode: full。
  thread_rotate_each_iteration: true
  thread_rotate_context_tokens: 60000
  iteration_mode: lockstep

curriculum:
  opponent_policy: progress
  batch: {batch}
  opponent_start_rank: {start_rank}
  order: lowest_rank_first
  development_seeds: [1]
  seed_mode: fixed
  # auto = measured → reference → crawled 逐选手回落。
  # 绝不能写 crawled：那只有第一批爬取的选手有名次，榜单会从 229 人缩到 20 人。
  ladder_scope: auto

goal:
  # 用历史：常驻会话 + 经验文档 + 历次候选代码都保留。
  history_mode: full
  prompt_override: null
  experience_skills: true
  code_constraint: any
  seed_policy_path: null

evaluation:
  # 慢评测（中间版本）：每 pool_stride 轮把一版拉去打完整个冻结人类池，
  # 算它的真实池内 Elo 与名次。独立进程、只写 pool-elo/。
  #
  # ★ 默认 false：**绝大多数实验只需要终局名次**，那个在迭代跑完之后用
  #   pool_elo_worker 的 --last-n-best 1 单独测就行（几个版本 vs 几十个版本，
  #   成本差一个数量级，而且那时机器是空的，--parallel 可以开大）。
  #   只有需要**学习曲线**的实验才开它（--background-pool）。
  #
  # 为什么默认关：它是**每个 run 一个 worker**，而水位控制是**各自判断**的，
  # 不是全局配额 —— 每个 worker 等的是 `load + headroom <= 总核数`，于是 N 个
  # worker 会在同一时刻各自看到"load 18 < 24，可以派发"然后一起下水。
  # **每个都"预留 8 核"等于谁也没预留。**
  #
  # 实测（16 个 run × 2 轮，32 核）：
  #   worker-status.json 记到 load_average 40.4（超配 125%）
  #   慢评测打了 1484 局，而迭代本身只有 232 局 —— 6.4 倍
  #   同一个 8 局波次：机器空时 56s，铺满后 607~1230s（慢 10~20 倍）
  #   45 个排队版本**一个都没完成**（每版要 458 局，worker 随 run 退出，
  #   实测某版停在 234/458）—— 那 1484 局机时换回来 0 个 Elo 数据点。
  background_pool: {background_pool}
  # 只在 background_pool 为 true 时有意义。4 轮一版：32 轮出 8 个点。
  pool_stride: {stride}
  pool_sample: 16
  pool_seeds: [7]
  challenger_track: {track}

isolation:
  backend: auto
  rival_code_visible: false
  docker_image: null

budget:
  # 不设 token 预算：停止条件是 max_iterations。
  # 旧的 3000000 是在 token 记账低估 14 倍的口径下拍的，修好后会在第 1 轮掐死 run。
  tokens: null
  wall_seconds: null

paths:
  agentbench_root: ${{AGENTBENCH_ROOT}}
  runs_root: ../../../runs
"""


def render(
    *,
    game: str,
    model: str,
    iterations: int,
    stride: int,
    purpose: str,
    axis: str,
    rollout_k: int = 1,
    batch: int = 4,
    start_rank: int = DEFAULT_START_RANK,
    agent_binary: str | None = None,
    background_pool: bool = True,
) -> str:
    track = CHALLENGER_TRACKS.get(game)
    return TEMPLATE.format(
        agent_binary="null" if agent_binary is None else agent_binary,
        background_pool="true" if background_pool else "false",
        game=game,
        model=model,
        iterations=iterations,
        stride=stride,
        start_rank=start_rank,
        track="null" if track is None else track,
        purpose=purpose,
        axis=axis,
        rollout_k=rollout_k,
        batch=batch,
        # 一轮 k×b×2 局。并发给到"一轮能一次打完"，但不超过 16 ——
        # 32 核上再高就会和 agent 思考、慢评测互相抢机时（实测铺满时
        # load 冲到 75，所有东西一起变慢）。
        parallelism=min(16, max(4, rollout_k * batch * 2)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="*", default=[], help="换模型（固定 antwar2）")
    parser.add_argument("--games", nargs="*", default=[], help="换游戏（固定 glm-5.2）")
    parser.add_argument("--game", default="antwar2", help="--models 模式下用哪个游戏")
    parser.add_argument("--model", default="glm-5.2", help="--games 模式下用哪个模型")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help=(
            "慢评测每几轮取一版（只在 --background-pool 打开时有意义）。"
            "4 是需要曲线时的默认口径：32 轮 → 8 个点，够看趋势，"
            "而每个点要 458 局，再密就追不上迭代了"
        ),
    )
    parser.add_argument("--prefix", default="verify")
    parser.add_argument(
        "--rollout-k",
        type=int,
        default=1,
        help="一轮提交几个候选（k）。k>1 才有并行假设检验，见模板里的详注",
    )
    parser.add_argument("--batch", type=int, default=4, help="一轮打几个对手（b）")
    parser.add_argument(
        "--start-rank",
        type=int,
        default=DEFAULT_START_RANK,
        help=(
            "progress 课程的起点名次（榜单里第几名，数字越小越强）。"
            "注意它是**对手**的名次，不是 agent 自己的池内名次 —— "
            "后者由慢评测测出来，两者同名不同义"
        ),
    )
    parser.add_argument(
        "--agent-binary",
        default=None,
        help="harness 可执行文件（cc harness 用 claude）。不给则回落 codex_binary",
    )
    parser.add_argument(
        "--background-pool",
        action="store_true",
        help=(
            "开启**中间版本**的后台慢评测（每 --stride 轮取一版）。"
            "默认关闭：绝大多数实验只需要终局名次，那个用 pool_elo_worker "
            "的 --last-n-best 1 在迭代跑完之后单独测，又快又不抢机时。"
            "只有需要**学习曲线**的实验（多模型对比 / 多游戏对比）才开它"
        ),
    )
    arguments = parser.parse_args(argv)

    if not arguments.models and not arguments.games:
        parser.error("给 --models 或 --games 至少一个")

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    shape = f"k={arguments.rollout_k}/b={arguments.batch}"

    for model in arguments.models:
        path = arguments.out_dir / f"{arguments.prefix}-{model}-{arguments.game}.yaml"
        path.write_text(
            render(
                game=arguments.game,
                model=model,
                iterations=arguments.iterations,
                stride=arguments.stride,
                purpose=f"验证 {model} 这个模型/中转跑得通（{shape}）",
                axis="provider.model_profile",
                rollout_k=arguments.rollout_k,
                batch=arguments.batch,
                start_rank=arguments.start_rank,
                agent_binary=arguments.agent_binary,
                background_pool=arguments.background_pool,
            ),
            encoding="utf-8",
        )
        written.append(path)

    for game in arguments.games:
        path = arguments.out_dir / f"{arguments.prefix}-{arguments.model}-{game}.yaml"
        path.write_text(
            render(
                game=game,
                model=arguments.model,
                iterations=arguments.iterations,
                stride=arguments.stride,
                purpose=f"验证 {game} 在 {shape}/progress/用历史 下跑得通",
                axis="game（以及随之而来的 challenger_track）",
                rollout_k=arguments.rollout_k,
                batch=arguments.batch,
                start_rank=arguments.start_rank,
                agent_binary=arguments.agent_binary,
                background_pool=arguments.background_pool,
            ),
            encoding="utf-8",
        )
        written.append(path)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
