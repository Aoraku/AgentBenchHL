"""后台固定池评测队列 —— CPU 空闲时给每个中间版本测"在人类池里排第几"。

它解决什么问题
--------------
主迭代循环里每个候选只打 ``roles × seeds`` 局（实测 2 局），那个样本量足够
给 agent 反馈"这次改动有没有打赢当前对手"，但**不足以**回答"这一版在人类棋手
里排第几"：2 局全胜/全败会被 BT 的正则先验直接顶到固定值，分辨力为零。

后台再打一遍是唯一诚实的办法。但它必须满足三个硬约束，否则会伤到主线：

1. **不抢机时**：主迭代的对局与 agent 思考都要 CPU。所以本模块只在系统负载
   低于水位时派发新对局，并且在每一局边界重新判断（而不是一次性把几百局
   排进队列）。CPU 一忙就自己停下等，主线永远优先。
2. **不改尺子**：人类池是冻结的，锚点只读（见 ``challenger_eval``）。
3. **不影响迭代决策**：结果只写到 ``pool-elo/`` 目录与自己的账本，
   主迭代的 ``events.jsonl`` 不被本进程写入——两个进程各写各的文件，
   避免并发追加同一个事件账本。

队列语义
--------
* **发现**：扫 run 的 ``events.jsonl`` 里的 ``GoalVersionSnapshot``，
  每个 ``candidate_id`` 就是一个待评测版本。新版本随迭代不断出现，
  worker 循环重新扫描即可，不需要主线通知。
* **优先级**：先测**每轮被选中的最佳候选**（``best_candidate_id``，
  它们构成实际的演进主线），再按迭代序号从新到旧补齐其余候选。
  这样即使机时永远追不上迭代速度，主线曲线也总是完整的。
* **续跑**：每个版本一个目录 ``pool-elo/<candidate_id>/``，里面是
  ``matches.jsonl`` + ``challenger-elo.json``。杀掉 worker 再起来会接着打。
* **池指纹校验**：人类池重测过（指纹变了）的旧结果会被重新排队，
  因为它们与新尺子不可比。

产出
----
``<run_root>/pool-elo/index.json``：所有已完成版本的 Elo 与名次，
带迭代序号，可以直接画"人类池排名 vs 迭代轮次"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from agentbench_hl.application.challenger_eval import (  # noqa: E402
    SUMMARY_FILENAME,
    FrozenPool,
    evaluate_many,
    load_challenger_elo,
    load_frozen_pool,
)

QUEUE_DIRNAME = "pool-elo"
INDEX_FILENAME = "index.json"
STATUS_FILENAME = "worker-status.json"


@dataclass(frozen=True)
class Job:
    """一个待评测版本。"""

    candidate_id: str
    snapshot: Path
    iteration: int | None
    is_best: bool

    @property
    def priority(self) -> tuple[int, int]:
        """排序键：先主线最佳候选，再按迭代从新到旧。

        为什么"从新到旧"：机时可能永远追不上迭代速度，那就优先测最近的版本——
        它们才是"我们现在有多强"的答案。老版本的历史曲线可以慢慢补。
        """

        iteration = -1 if self.iteration is None else self.iteration
        return (0 if self.is_best else 1, -iteration)


# ------------------------------------------------------------------ CPU 水位


def load_average() -> float:
    return os.getloadavg()[0]


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - 非 Linux
        return os.cpu_count() or 1


def wait_for_idle_cpu(
    *,
    headroom: int,
    poll_s: float = 30.0,
    log=None,
    should_stop=None,
) -> None:
    """等到"至少还有 ``headroom`` 个核没被占"再返回。

    用 1 分钟 load average 而不是瞬时 CPU 占用：瞬时值抖动太大，会让 worker
    在主迭代的对局间隙里反复插进去抢核。load average 天然平滑，
    正好符合"持续空闲才动手"的语义。

    这个函数是本模块"不抢机时"承诺的全部实现——``evaluate_challenger`` 会在
    **每一局派发之前**调它，所以主迭代一忙起来，后台评测就自己停在这里。
    """

    total = cpu_count()
    while True:
        if should_stop is not None and should_stop():
            return
        current = load_average()
        if current + headroom <= total:
            return
        if log is not None:
            log(
                f"[pool-elo] CPU 忙（load {current:.1f}/{total}，需留 {headroom} 核空闲），"
                f"{poll_s:.0f}s 后再看"
            )
        time.sleep(poll_s)


# ------------------------------------------------------------------ 队列发现


def _read_events(run_root: Path) -> list[dict[str, object]]:
    path = run_root / "events.jsonl"
    if not path.is_file():
        return []
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # 主进程正在追加时最后一行可能只写了一半；下一轮扫描就完整了。
            continue
    return events


def discover_jobs(run_root: Path) -> list[Job]:
    """从 run 的事件流找出所有可评测版本，按优先级排好序。"""

    events = _read_events(run_root)
    snapshots: dict[str, Path] = {}
    iteration_of_request: dict[str, int] = {}
    request_of_candidate: dict[str, str] = {}
    best: set[str] = set()

    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if event_type == "GoalVersionSnapshot":
            candidate = payload.get("candidate_id")
            path = payload.get("path")
            if isinstance(candidate, str) and isinstance(path, str):
                snapshots[candidate] = Path(path)
        elif event_type == "GoalMatchCompleted":
            candidate = payload.get("candidate_id")
            request = payload.get("request_id")
            if isinstance(candidate, str) and isinstance(request, str):
                request_of_candidate.setdefault(candidate, request)
        elif event_type == "IterationMetricsFinalized":
            request = payload.get("request_id")
            iteration = payload.get("research_iteration")
            if isinstance(request, str) and isinstance(iteration, int):
                iteration_of_request[request] = iteration
            chosen = payload.get("best_candidate_id")
            if isinstance(chosen, str):
                best.add(chosen)

    jobs = [
        Job(
            candidate_id=candidate,
            snapshot=path,
            iteration=iteration_of_request.get(request_of_candidate.get(candidate, "")),
            is_best=candidate in best,
        )
        for candidate, path in snapshots.items()
        if (path / "main.py").is_file()
    ]
    jobs.sort(key=lambda job: (job.priority, job.candidate_id))
    return jobs


def is_done(work_root: Path, pool: FrozenPool) -> bool:
    """这个版本是否已经在**当前**池指纹下测完。

    指纹不同就算没测——人类池重测过之后，旧 Elo 与新尺子不可比，
    静默复用会得出错误的排名。
    """

    document = load_challenger_elo(work_root)
    if document is None:
        return False
    if str(document.get("pool_fingerprint")) != pool.fingerprint:
        return False
    return not document.get("partial", False)


# ------------------------------------------------------------------ 索引汇总


def write_index(queue_root: Path, pool: FrozenPool) -> dict[str, object]:
    """把已完成的版本汇总成一张可直接画曲线的表。"""

    rows: list[dict[str, object]] = []
    for directory in sorted(queue_root.iterdir()) if queue_root.is_dir() else []:
        if not directory.is_dir():
            continue
        document = load_challenger_elo(directory)
        if document is None or document.get("elo") is None:
            continue
        rows.append(
            {
                "candidate_id": document.get("challenger_id"),
                "iteration": document.get("iteration"),
                "elo": document.get("elo"),
                "pool_rank": document.get("pool_rank"),
                "elo_gap_to_top": document.get("elo_gap_to_top"),
                "wins": document.get("wins"),
                "draws": document.get("draws"),
                "losses": document.get("losses"),
                "complete_matches": document.get("complete_matches"),
                "partial": document.get("partial"),
                "pool_fingerprint": document.get("pool_fingerprint"),
                "comparable": document.get("pool_fingerprint") == pool.fingerprint,
            }
        )
    rows.sort(
        key=lambda row: (
            row["iteration"] is None,
            row["iteration"] if row["iteration"] is not None else 0,
            str(row["candidate_id"]),
        )
    )
    comparable = [row for row in rows if row["comparable"] and row["elo"] is not None]
    peak = max(comparable, key=lambda row: float(row["elo"])) if comparable else None
    document = {
        "schema": "pool-elo-index/v1",
        "pool": pool.summary(),
        "versions": rows,
        "evaluated": len(rows),
        "peak": peak,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": (
            "elo = 该版本对冻结人类池的实测 BT 估计；pool_rank = 插进人类榜的名次。"
            "与 IterationMetricsFinalized.pool_elo 不是同一个量（后者是全 run 累计）。"
        ),
    }
    queue_root.mkdir(parents=True, exist_ok=True)
    (queue_root / INDEX_FILENAME).write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def _write_status(queue_root: Path, payload: dict[str, object]) -> None:
    queue_root.mkdir(parents=True, exist_ok=True)
    (queue_root / STATUS_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------ 主循环


def run_worker(
    run_root: Path,
    agentbench_root: Path,
    *,
    game: str,
    scope: str,
    seeds: tuple[int, ...],
    parallel: int,
    cpus_per_match: int,
    headroom: int,
    timeout_s: float,
    poll_s: float,
    once: bool,
    log=print,
) -> int:
    queue_root = run_root / QUEUE_DIRNAME
    queue_root.mkdir(parents=True, exist_ok=True)
    pool = load_frozen_pool(agentbench_root, game)
    log(
        f"[pool-elo] 冻结人类池：{game} {pool.size} 人，"
        f"Elo {pool.summary()['elo_min']}~{pool.summary()['elo_max']}，"
        f"指纹 {pool.fingerprint}"
    )
    log(f"[pool-elo] CPU {cpu_count()} 核，给主迭代预留 {headroom} 核")

    evaluated = 0
    while True:
        jobs = discover_jobs(run_root)
        pending = [job for job in jobs if not is_done(queue_root / job.candidate_id, pool)]
        _write_status(
            queue_root,
            {
                "run_root": str(run_root),
                "game": game,
                "pool_fingerprint": pool.fingerprint,
                "discovered": len(jobs),
                "pending": len(pending),
                "evaluated_this_session": evaluated,
                "load_average": round(load_average(), 2),
                "cpu_count": cpu_count(),
                "headroom": headroom,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        if not pending:
            write_index(queue_root, pool)
            if once:
                log("[pool-elo] 队列已空，退出（--once）")
                return 0
            log(f"[pool-elo] 队列已空，{poll_s:.0f}s 后重新扫描新版本")
            time.sleep(poll_s)
            continue

        # 一次把**所有**待测版本交给扁平队列。
        #
        # 为什么不是一版一版跑：逐版本串行有两个硬伤——队尾只剩几局时线程池
        # 大部分空转，而且并发上限被"单版本剩余局数"卡住，跟机器有多少核无关。
        # 摊平成 (版本 × 对手 × 座次) 的全局队列后，任何时刻都有足够待办填满
        # 线程池，直到全部跑完。
        log(
            f"[pool-elo] 本批 {len(pending)} 个版本进入扁平队列，"
            f"并发 {parallel} 路（每局 {cpus_per_match} 核）"
        )
        for job in pending[:5]:
            log(
                f"[pool-elo]   排队 {job.candidate_id} (iter {job.iteration}"
                f"{'，主线最佳' if job.is_best else ''})"
            )
        if len(pending) > 5:
            log(f"[pool-elo]   ……以及另外 {len(pending) - 5} 个")

        def progress(challenger_id: str, finished: int, total: int) -> None:
            if finished % 50 == 0 or finished == total:
                log(f"[pool-elo]   进度 {finished}/{total} 局（最近 {challenger_id}）")

        try:
            documents = evaluate_many(
                game,
                agentbench_root,
                [(job.candidate_id, job.snapshot) for job in pending],
                queue_root=queue_root,
                pool=pool,
                scope=scope,
                seeds=seeds,
                parallel=parallel,
                cpus_per_match=cpus_per_match,
                timeout_s=timeout_s,
                on_match=progress,
                before_match=lambda: wait_for_idle_cpu(
                    headroom=headroom, poll_s=poll_s, log=log
                ),
            )
        except Exception as error:  # noqa: BLE001 - 整批失败也不该让 worker 退出
            log(f"[pool-elo] 本批评测失败：{type(error).__name__}: {error}")
            (queue_root / "error.txt").write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
            if once:
                return 1
            time.sleep(poll_s)
            continue

        # 迭代序号是队列侧的知识（来自 run 的事件流），补写进结果里，
        # 这样 index.json 能直接按轮次画曲线。
        iteration_of = {job.candidate_id: job for job in pending}
        for document in documents:
            challenger_id = str(document.get("challenger_id"))
            job = iteration_of.get(challenger_id)
            if job is None:
                continue
            document["iteration"] = job.iteration
            document["is_best_of_iteration"] = job.is_best
            (queue_root / challenger_id / SUMMARY_FILENAME).write_text(
                json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not document.get("partial"):
                evaluated += 1
            log(
                f"[pool-elo] 完成 {challenger_id} (iter {job.iteration}): "
                f"elo={document.get('elo')} 名次=#{document.get('pool_rank')} "
                f"胜率={document.get('win_rate')} "
                f"W-D-L={document.get('wins')}-{document.get('draws')}-{document.get('losses')}"
            )
        write_index(queue_root, pool)
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--agentbench-root", required=True, type=Path)
    parser.add_argument("--game", required=True)
    parser.add_argument("--scope", default="verified")
    parser.add_argument("--seeds", default="7", help="逗号分隔，例如 7 或 7,11")
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="同时进行的对局数。上限应约等于 (总核数 - headroom) / cpus-per-match",
    )
    parser.add_argument("--cpus-per-match", type=int, default=3)
    parser.add_argument(
        "--headroom",
        type=int,
        default=6,
        help="必须留给主迭代的空闲核数；load average 超过 (总核数 - headroom) 就暂停派发",
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="只评测一个版本就退出（便于验证）")
    parser.add_argument("--index-only", action="store_true", help="只重建 index.json 后退出")
    arguments = parser.parse_args(argv)

    run_root = arguments.run_root.resolve()
    agentbench_root = arguments.agentbench_root.resolve()
    seeds = tuple(int(item) for item in str(arguments.seeds).split(",") if item.strip())

    if arguments.index_only:
        pool = load_frozen_pool(agentbench_root, arguments.game)
        document = write_index(run_root / QUEUE_DIRNAME, pool)
        print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    return run_worker(
        run_root,
        agentbench_root,
        game=arguments.game,
        scope=arguments.scope,
        seeds=seeds,
        parallel=arguments.parallel,
        cpus_per_match=arguments.cpus_per_match,
        headroom=arguments.headroom,
        timeout_s=arguments.timeout_s,
        poll_s=arguments.poll_s,
        once=arguments.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
