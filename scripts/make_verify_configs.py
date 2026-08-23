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
  agent_binary: null
  branch_width: 1
  # k=1：一轮只出一个策略，探索广度由 b 个对手提供。
  rollout_k: 1
  # 一轮的对局数 = 1 × b × 座次 × seed = 1 × 4 × 2 × 1 = 8 局。
  match_parallelism: 8
  # 各游戏单局长度差一个数量级（miracle 约 7s，snakego 实测 246s）。
  # 420s 会误杀长局，而那在结果里看起来像"这个模型不会玩这个游戏"。
  match_timeout_s: 1800
  network_access: disabled
  # 每轮换 thread：躲开 codex 的 remote compaction（对 glm 系必死——
  # 它返回 [reasoning, message] 两个 item，codex 只接受一个）。
  # 工作区侧历史一份不动，所以不影响 history_mode: full 的语义。
  thread_rotate_each_iteration: true
  thread_rotate_context_tokens: 60000
  iteration_mode: lockstep

curriculum:
  opponent_policy: progress
  batch: 4
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
  # 慢评测：每 3 轮把中间版本拉去打完整个冻结人类池，算真实池内 Elo 与名次。
  # 独立进程、只写 pool-elo/、自带 CPU 水位控制。Elo 面板的实测曲线只来自这里。
  background_pool: true
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
    *, game: str, model: str, iterations: int, stride: int, purpose: str, axis: str
) -> str:
    track = CHALLENGER_TRACKS.get(game)
    return TEMPLATE.format(
        game=game,
        model=model,
        iterations=iterations,
        stride=stride,
        start_rank=DEFAULT_START_RANK,
        track="null" if track is None else track,
        purpose=purpose,
        axis=axis,
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
        default=1,
        help="慢评测每几轮取一版。验收跑只有 2 轮，必须用 1 才能拿到数据点",
    )
    parser.add_argument("--prefix", default="verify")
    arguments = parser.parse_args(argv)

    if not arguments.models and not arguments.games:
        parser.error("给 --models 或 --games 至少一个")

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for model in arguments.models:
        path = arguments.out_dir / f"{arguments.prefix}-{model}-{arguments.game}.yaml"
        path.write_text(
            render(
                game=arguments.game,
                model=model,
                iterations=arguments.iterations,
                stride=arguments.stride,
                purpose=f"验证 {model} 这个模型/中转在新框架下跑得通",
                axis="provider.model_profile",
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
                purpose=f"验证 {game} 在新框架（k=1/b=4/progress/用历史）下跑得通",
                axis="game（以及随之而来的 challenger_track）",
            ),
            encoding="utf-8",
        )
        written.append(path)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
