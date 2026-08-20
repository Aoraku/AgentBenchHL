#!/usr/bin/env python3
"""一屏看完多个 goal-led run 的迭代进度（给人在终端里反复敲的那种）。

为什么不用 ``export_curves.py``
------------------------------
那个是给论文出图的（写 csv/png，跑完再看）。这里要的是**运行中**的体检表：
现在跑到第几轮、这一轮的候选相对上一轮改了多少、IG 是不是真的测出来了、
pool_elo 有没有在长、进程还活着吗。所以它只读事件账本 + snapshots，不写任何东西。

读什么
------
``runs/<run_id>/events.jsonl``  —— 唯一真相来源：
  * ``IterationMetricsFinalized`` : 每轮定稿指标（pool_elo / win_rate / behavioral_ig / …）
  * ``GoalMatchCompleted``        : 逐局结果（用来显示"进行中那一轮打到第几局"）
  * ``InformationGainMeasured``   : IG 的测量口径（support_mode / |A| / decisions）
  * ``AgentTokenUsage``           : 累计 token
  * ``GoalLedDriveFinished``      : 收尾原因
``runs/<run_id>/snapshots/<cand>/ai.py`` —— 算"改了哪些"：每个候选相对**上一轮最佳**
  候选做 unified diff，报 +/- 行数；这是"代码可解释"这条理念唯一可自动核验的部分。

存活判断不靠 pid 文件（run.lock 里没有 pid），直接扫 ``/proc/*/cmdline`` 找
``--run-id <run_id>``，因为 driver 是 nohup setsid 起的，pid 会变但命令行不会。

用法::

    python3 scripts/watch_progress.py --runs-root ../runs sota-antwar sota-antwar2
    python3 scripts/watch_progress.py --runs-root ../runs sota-antwar --iters 3   # 只看最近 3 轮
"""
from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import os
import re
from pathlib import Path

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def _c(text: str, color: str, enable: bool) -> str:
    return f"{color}{text}{RESET}" if enable else text


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 尾行可能正被写入，跳过即可
    return out


def _alive(run_id: str) -> int | None:
    """扫 /proc 找还在跑这个 run 的 driver 进程；返回 pid 或 None。"""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    needle = f"--run-id\x00{run_id}\x00"
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "goal-led" not in raw:
            continue
        if needle in raw or raw.rstrip("\x00").endswith(f"--run-id\x00{run_id}"):
            return int(entry.name)
    return None


def _ts(value: str | None) -> _dt.datetime | None:
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_ago(when: _dt.datetime | None) -> str:
    if when is None:
        return "-"
    delta = _dt.datetime.now(_dt.timezone.utc) - when
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s 前"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s 前"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m 前"


def _fmt_num(value, digits: int = 3, dash: str = "-") -> str:
    if value is None:
        return dash
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _short_cand(cid: str) -> str:
    """v003_hp_lightning_guard -> hp_lightning_guard（前缀 v00x 由 it 列表达）。"""
    return re.sub(r"^v\d+_", "", cid or "")


def _short_opp(oid: str) -> str:
    if not oid:
        return "-"
    tail = oid.split("__")[-1]
    head = oid.split("__")[1][:6] if "__" in oid else oid[:6]
    return f"{head}..{tail}"


# 快照目录里这些是脚手架/官方 SDK，不是 agent 写的东西，diff 时必须排掉，
# 否则每个候选都会显示成"改了几百行"。
_SCAFFOLD = {
    "ai_example.py",
    "_bootstrap.py",
    "main.py",
    "official_main.py",
}


def _code_of(run_root: Path, candidate_id: str) -> list[str] | None:
    """agent 自写代码 = 快照顶层的 .py 减去脚手架。

    antwar 的候选主体在 ``ai.py``，antwar2 会额外拆出 ``vNNN_planner.py``，
    所以不能只盯 ``ai.py``，否则 antwar2 的改动量会被严重低估。
    """
    base = run_root / "snapshots" / candidate_id
    if not base.is_dir():
        return None
    parts: list[str] = []
    for path in sorted(base.glob("*.py")):
        if path.name in _SCAFFOLD:
            continue
        parts.append(f"### {path.name}\n")
        parts.extend(path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True))
    return parts or None


def _diff_stat(run_root: Path, base_id: str | None, cand_id: str) -> str:
    """候选相对上一轮最佳的 +新增/-删除 行数。基线缺失就只报行数。"""
    cur = _code_of(run_root, cand_id)
    if cur is None:
        return "?"
    if not base_id:
        return f"{len(cur)}L"
    base = _code_of(run_root, base_id)
    if base is None:
        return f"{len(cur)}L"
    plus = minus = 0
    for line in difflib.unified_diff(base, cur, n=0):
        if line.startswith("+") and not line.startswith("+++"):
            plus += 1
        elif line.startswith("-") and not line.startswith("---"):
            minus += 1
    if plus == 0 and minus == 0:
        return "同上轮"
    return f"+{plus}/-{minus}"


def _spread(
    run_root: Path,
    candidate_ids: list[str],
    fingerprints: dict[str, str],
    recorded: dict | None = None,
) -> str:
    """同一轮 k 个候选彼此差多少行 —— 探索是不是"只改了一个常量"。

    优先用框架自己记在 ``IterationMetricsFinalized.candidate_spread`` 里的度量
    （与反馈给 agent 的是同一个数，避免这里算出第二个口径）；老 run 没这个字段时
    退回本地重算。只看指纹唯一性会漏掉"4 份代码互不相同但只差 1 行"这种伪多样性。
    """
    uniq = len({fingerprints.get(cid, cid) for cid in candidate_ids})
    if recorded:
        verdict = recorded.get("verdict")
        tag = "伪多样性" if verdict == "near_duplicate" else "OK"
        return (
            f"指纹互异 {uniq}/{len(candidate_ids)} · 候选间差异 min {recorded.get('min_diff_lines')}"
            f" / 均 {recorded.get('mean_diff_lines')} / max {recorded.get('max_diff_lines')} 行"
            f" · 判定 {tag}（阈值 {recorded.get('threshold_lines')}）"
        )
    codes = {cid: _code_of(run_root, cid) for cid in candidate_ids}
    diffs: list[int] = []
    ids = [cid for cid in candidate_ids if codes.get(cid)]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            n = sum(
                1
                for line in difflib.unified_diff(codes[ids[i]], codes[ids[j]], n=0)
                if line[:1] in "+-" and not line.startswith(("+++", "---"))
            )
            diffs.append(n)
    if not diffs:
        return f"指纹互异 {uniq}/{len(candidate_ids)}"
    avg = sum(diffs) / len(diffs)
    return f"指纹互异 {uniq}/{len(candidate_ids)} · 候选间平均差 {avg:.0f} 行（min {min(diffs)}）"


def _experience_note(run_root: Path) -> str:
    """EXPERIENCE.md 是"从可解释经验中学习"的落点，看它有没有在长。"""
    hits = sorted(run_root.glob("workspace/**/EXPERIENCE.md")) + sorted(
        run_root.glob("workspace/EXPERIENCE.md")
    )
    if not hits:
        return "EXPERIENCE.md 尚未出现"
    path = hits[-1]
    lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    mtime = _dt.datetime.fromtimestamp(path.stat().st_mtime, _dt.timezone.utc)
    return f"EXPERIENCE.md {lines} 行，{_fmt_ago(mtime)}更新"


def _tail_log(runs_root: Path, run_id: str, run_root: Path) -> str:
    for cand in (runs_root / f"{run_id}.driver.log", run_root / "driver.log"):
        if not cand.exists():
            continue
        try:
            data = cand.read_bytes()[-4000:].decode("utf-8", "replace")
        except OSError:
            continue
        lines = [ln.rstrip() for ln in data.splitlines() if ln.strip()]
        if lines:
            return lines[-1][:160]
    return ""


def report(runs_root: Path, run_id: str, max_iters: int, color: bool) -> None:
    run_root = runs_root / run_id
    events = _read_events(run_root / "events.jsonl")
    if not events:
        print(_c(f"[{run_id}] 还没有 events.jsonl（未启动或刚起来）", YELLOW, color))
        tail = _tail_log(runs_root, run_id, run_root)
        if tail:
            print(f"    日志尾: {tail}")
        print()
        return

    by_type: dict[str, list[dict]] = {}
    for ev in events:
        by_type.setdefault(ev.get("event_type", "?"), []).append(ev)

    manifest = (by_type.get("RunReproducibilityManifest") or [{}])[0].get("payload", {})
    game = manifest.get("game", "?")
    exp = manifest.get("experiment_variables", {})
    per_iter_matches = (
        int(exp.get("rollout_k", 0) or 0)
        * max(1, len(exp.get("roles") or []))
        * max(1, len(exp.get("seeds") or []))
    )

    metrics = [e["payload"] for e in by_type.get("IterationMetricsFinalized", [])]
    metrics.sort(key=lambda p: p.get("research_iteration", 0))
    done = len(metrics)
    matches_all = by_type.get("GoalMatchCompleted", [])
    tokens = sum(
        int(e["payload"].get("total_tokens") or 0) for e in by_type.get("AgentTokenUsage", [])
    ) or (metrics[-1].get("token_events_sum") if metrics else 0)

    started = _ts((by_type.get("GoalLedStarted") or [{}])[0].get("occurred_at"))
    last_ev = events[-1]
    finished = by_type.get("GoalLedDriveFinished")
    pid = _alive(run_id)

    if pid:
        head_state = _c(f"运行中 pid={pid}", GREEN, color)
    elif finished:
        stop = finished[-1]["payload"].get("stop_reason")
        err = finished[-1]["payload"].get("error")
        head_state = _c(f"已收尾 ({stop}{'/' + str(err) if err else ''})", YELLOW, color)
    else:
        head_state = _c("进程不在（异常退出？看日志尾）", RED, color)

    elapsed = ""
    if started:
        secs = int((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds())
        elapsed = f" · 已跑 {secs // 3600}h{(secs % 3600) // 60:02d}m"
    # 会话轮转：codex 的压缩对本模型必死，框架靠换 thread 规避。轮转次数与当前
    # 上下文占用要看得见——上下文逼近配置阈值而没轮转，就是这道防线失效了。
    rotations = len(by_type.get("GoalSessionRotated", []))
    ctx = metrics[-1].get("thread_context_tokens") if metrics else None
    session_bit = f" · 会话轮转 {rotations} 次" if rotations else ""
    if isinstance(ctx, int):
        session_bit += f" · 上下文 {ctx / 1000:.0f}k"
    print(
        f"{_c('■ ' + run_id, BOLD, color)}  [{game}]  {head_state}"
        f" · 完成 {done} 轮 · tokens {tokens / 1000:.1f}k{elapsed}{session_bit}"
    )

    # ---- 逐轮表 ----
    shown = metrics[-max_iters:] if max_iters > 0 else metrics
    if shown:
        print(
            _c(
                "  it │ 对手(rank)        │ 候选 k │  W- L- D  │ pool_elo   Δ     │ "
                "beh_ig   |A|          │ out_ig │ wall",
                DIM,
                color,
            )
        )
    prev_elo = None
    if shown and metrics.index(shown[0]) > 0:
        prev_elo = metrics[metrics.index(shown[0]) - 1].get("pool_elo")
    for m in shown:
        it = m.get("matches", 0) or 0
        wr = m.get("win_rate")
        dr = m.get("draw_rate") or 0.0
        wins = int(round((wr or 0.0) * it))
        draws = int(round(dr * it))
        losses = it - wins - draws
        elo = m.get("pool_elo")
        delta = "-"
        if elo is not None and prev_elo is not None:
            d = float(elo) - float(prev_elo)
            delta = _c(f"{d:+.1f}", GREEN if d >= 0 else RED, color)
        prev_elo = elo if elo is not None else prev_elo
        rank = m.get("conquest", {}).get("target_index")
        opp = _short_opp((m.get("opponent_ids") or [""])[0])
        alpha = m.get("behavioral_ig_support_cardinality")
        mode = (m.get("behavioral_ig_support_mode") or "")[:12]
        # |A| 的口径要一眼看得出：exact 才是逐点真实合法集，opcode_alphabet 是常量近似。
        # 曾经因为汇总层用静态声明覆盖了实际口径，连续 14 轮把精确值报成了近似值。
        exact_frac = m.get("behavioral_ig_support_exact_fraction")
        if mode.startswith("exact"):
            support_cell = "exact"
        elif mode.startswith("mixed"):
            support_cell = f"mixed {float(exact_frac or 0) * 100:.0f}%"
        elif alpha:
            support_cell = f"|A|={alpha} 近似"
        else:
            support_cell = "-"
        ig = m.get("behavioral_ig")
        ig_cell = _fmt_num(ig, 3) if ig is not None else _c("null", YELLOW, color)
        print(
            f"  {m.get('research_iteration', '?'):>2} │ {opp:<17} │"
            f" {len(m.get('candidate_ids') or []):>5}  │"
            f" {wins:>2}-{losses:>2}-{draws:>2}  │"
            f" {_fmt_num(elo, 1):>8}  {delta:>13} │"
            f" {ig_cell:>14} {support_cell:<14} │"
            f" {_fmt_num(m.get('outcome_ig_nats'), 2):>6} │ {_fmt_num(m.get('total_wall_time_s'), 0)}s"
            + (f"  {DIM}#{rank + 1}{RESET}" if color and rank is not None else "")
        )
        if ig is None and m.get("behavioral_ig_reason"):
            print(_c(f"       ↳ ig=null: {m['behavioral_ig_reason'][:110]}", DIM, color))

    # ---- 最近一轮改了哪些 ----
    if metrics:
        last = metrics[-1]
        base = metrics[-2].get("best_candidate_id") if len(metrics) > 1 else None
        fingerprints = {
            e["payload"].get("candidate_id"): e["payload"].get("code_fingerprint")
            for e in by_type.get("GoalVersionSnapshot", [])
        }
        cand_ids = list(last.get("candidate_ids") or [])
        parts = []
        for cid in cand_ids:
            flag = "*" if cid == last.get("best_candidate_id") else " "
            parts.append(f"{flag}{_short_cand(cid)}({_diff_stat(run_root, base, cid)})")
        if parts:
            label = f"vs 上轮最佳 {_short_cand(base)}" if base else "首轮从零写"
            print(f"  改动({label}): " + "  ".join(parts))
            print(f"  {_spread(run_root, cand_ids, fingerprints, last.get('candidate_spread'))}")
        print(f"  {_experience_note(run_root)}")

    # ---- 进行中那一轮 ----
    if pid:
        cur_iter = done + 1
        requested = by_type.get("GoalMatchRequested", [])
        started_iters = len(requested)
        in_flight = [
            e
            for e in matches_all
            if requested
            and e.get("occurred_at", "") >= requested[-1].get("occurred_at", "")
        ]
        phase = "写候选/读回放（尚未开局）"
        if started_iters >= cur_iter and in_flight:
            ok = sum(1 for e in in_flight if e["payload"].get("status") == "complete")
            w = sum(1 for e in in_flight if e["payload"].get("result") == "win")
            phase = f"对局中 {ok}/{per_iter_matches or '?'} 局（已胜 {w}）"
        elif started_iters >= cur_iter:
            phase = f"对局中 0/{per_iter_matches or '?'} 局"
        print(
            f"  {_c('第 ' + str(cur_iter) + ' 轮进行中', CYAN, color)}: {phase}"
            f" · 最新事件 {last_ev.get('event_type')} {_fmt_ago(_ts(last_ev.get('occurred_at')))}"
        )
        stale = _ts(last_ev.get("occurred_at"))
        if stale and (_dt.datetime.now(_dt.timezone.utc) - stale).total_seconds() > 2400:
            print(_c("  ⚠ 事件已 40 分钟无更新，可能卡在模型侧，去看日志尾", RED, color))

    tail = _tail_log(runs_root, run_id, run_root)
    if tail:
        print(_c(f"  日志尾: {tail}", DIM, color))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="goal-led run 进度体检表")
    ap.add_argument("run_ids", nargs="+", help="要看的 run-id（可多个）")
    ap.add_argument(
        "--runs-root",
        default=os.environ.get("ABHL_RUNS_ROOT", "../runs"),
        help="runs 根目录（默认 ../runs 或 $ABHL_RUNS_ROOT）",
    )
    ap.add_argument("--iters", type=int, default=8, help="最多显示最近几轮（0=全部）")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    color = not args.no_color and os.environ.get("TERM") != "dumb"
    root = Path(args.runs_root).expanduser().resolve()
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(_c(f"===== {stamp}  runs_root={root} =====", BOLD, color))
    for run_id in args.run_ids:
        report(root, run_id, args.iters, color)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
