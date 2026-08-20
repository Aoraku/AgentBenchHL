"""候选多样性度量：把"一个尝试做了 k 遍"和"k 个不同尝试"区分开。

为什么需要它
------------
探索批次的全部价值在于**一轮里试 k 个互斥假设**。如果 k 个候选是同一骨架改了一个
常量，那么这一轮的信息量等于 1 个候选，却花掉了 k×座次×seed 局对局与整轮墙钟。

这不是假想的失效模式，是实测：某次 4 轮 run 里，antwar 的 4 个候选彼此只差 **2 行**、
antwar2 的 4 个候选只差 **6 行**（``code_fingerprint`` 全部互异，所以既有的
"完全相同就跳过"去重完全放行）。那 4 轮的 pool_elo 单调下滑，因为每轮实际只探了一个点。

因此需要一个比"指纹是否相同"更敏感的判据：**pairwise 代码行差异**。它有两个用途：

1. 记进 ``IterationMetricsFinalized``，让"探索多样性"变成可追踪的曲线而不是感觉；
2. 反馈给 agent —— 判定为伪多样性时，下一轮提示词里点名说明，形成闭环。

口径
----
只看代码文件（``.py`` / ``.pyi``），按相对路径排序后拼成一条行序列，两两做
``difflib.unified_diff(n=0)`` 数改动行（+ 与 - 各计一行）。**不做语义分析**：
行差异是一个保守的下界代理量——它可能把"重命名变量"高估成差异，但绝不会把
"重写了一条取胜路径"低估成 0。

``NEAR_DUPLICATE_LINES = 15`` 的依据：15 行大致是"新增一个带条件判断的分支或一个
小函数"的最小体量。低于它，两个候选不可能走在不同的机制上。
"""

from __future__ import annotations

import ast
import difflib
from collections.abc import Mapping
from pathlib import Path

CODE_SUFFIXES = (".py", ".pyi")
NEAR_DUPLICATE_LINES = 15


def code_lines(root: Path, suffixes: tuple[str, ...] = CODE_SUFFIXES) -> list[str]:
    """候选的代码行序列（带文件名分隔，避免跨文件行错位对齐）。"""

    lines: list[str] = []
    if not root.is_dir():
        return lines
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        lines.append(f"### {path.relative_to(root).as_posix()}\n")
        lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True))
    return lines


def method_surface(root: Path, suffixes: tuple[str, ...] = CODE_SUFFIXES) -> set[str]:
    """候选定义/覆盖了哪些方法，形如 ``{"V004Agent._pick_target", ...}``。

    为什么要看方法而不只看行数：已知能刷到 SOTA 的写法是**继承链**——
    ``class V004Agent(V003Agent)`` 只覆盖一两个方法，那么"这一轮改了什么"
    精确等于"覆盖了哪些方法"。行数差异只是代理量，方法集合才是那个语义单位；
    两个候选覆盖同一个方法 = 它们在争同一个决策点，哪怕行数差很多也未必是
    两个不同的假设。

    解析失败（语法错等）时返回空集合：这里只做度量，合法性由 preflight 负责报错。
    """

    surface: set[str] = set()
    if not root.is_dir():
        return surface
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    surface.add(f"{node.name}.{item.name}")
    return surface


def base_classes(root: Path, suffixes: tuple[str, ...] = CODE_SUFFIXES) -> set[str]:
    """候选里出现的父类名——用来看它是不是真的在继承链上演进。"""

    bases: set[str] = set()
    if not root.is_dir():
        return bases
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.add(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.add(base.attr)
    return bases


def diff_lines(left: list[str], right: list[str]) -> int:
    return sum(
        1
        for line in difflib.unified_diff(left, right, n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )


def spread(
    candidate_roots: Mapping[str, Path],
    *,
    threshold_lines: int = NEAR_DUPLICATE_LINES,
    suffixes: tuple[str, ...] = CODE_SUFFIXES,
) -> dict[str, object] | None:
    """一轮内 k 个候选的 pairwise 代码差异。

    返回 ``None`` 表示无法度量（候选不足 2 个）。``verdict``：

    * ``distinct``        : 任意两个候选都相差 ≥ ``threshold_lines`` 行；
    * ``near_duplicate``  : 存在一对候选差异过小 —— 本轮探索被折叠成了更少的尝试。
    """

    ids = [cid for cid in candidate_roots if code_lines(candidate_roots[cid], suffixes)]
    if len(ids) < 2:
        return None
    codes = {cid: code_lines(candidate_roots[cid], suffixes) for cid in ids}
    pairs: list[tuple[int, str, str]] = []
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            pairs.append((diff_lines(codes[left], codes[right]), left, right))
    pairs.sort()
    smallest, a, b = pairs[0]
    counts = [count for count, _, _ in pairs]
    # 只要有一对过近，本轮就没有真正试到 k 个方向：报最坏情况，不报平均值掩盖。
    near = [(count, x, y) for count, x, y in pairs if count < threshold_lines]
    # 方法级视图：继承链演进下，"改了什么"就等于"覆盖了哪些方法"。
    surfaces = {cid: method_surface(candidate_roots[cid], suffixes) for cid in ids}
    touched: dict[str, int] = {}
    for names in surfaces.values():
        for name in names:
            touched[name] = touched.get(name, 0) + 1
    # 所有候选都覆盖同一个方法 = 它们在争同一个决策点，即便行数差得多。
    shared_methods = sorted(name for name, count in touched.items() if count == len(ids))
    inherits = sorted(
        {
            base
            for cid in ids
            for base in base_classes(candidate_roots[cid], suffixes)
            if base not in {"object", "ABC"}
        }
    )
    return {
        "candidates": len(ids),
        "pairs": len(pairs),
        "min_diff_lines": smallest,
        "mean_diff_lines": round(sum(counts) / len(counts), 1),
        "max_diff_lines": max(counts),
        "closest_pair": [a, b],
        "near_duplicate_pairs": [[x, y, count] for count, x, y in near],
        "threshold_lines": threshold_lines,
        "verdict": "near_duplicate" if near else "distinct",
        # 方法级证据（不参与 verdict 判定，只让"改了什么"可读）：
        "methods_per_candidate": {cid: len(surfaces[cid]) for cid in ids},
        "methods_shared_by_all": shared_methods,
        "base_classes": inherits,
    }


def feedback_note(report: Mapping[str, object] | None) -> str | None:
    """把度量结果翻成一句可执行的反馈（判定合格时返回 ``None``，不啰嗦）。"""

    if not report or report.get("verdict") != "near_duplicate":
        return None
    pairs = report.get("near_duplicate_pairs") or []
    shown = "；".join(f"{item[0]} 与 {item[1]} 只差 {item[2]} 行" for item in pairs[:3])  # type: ignore[index]
    return (
        f"【上一轮探索被浪费了】k={report.get('candidates')} 个候选里有 {len(pairs)} 对"
        f"几乎是同一份代码（{shown}；判定阈值 {report.get('threshold_lines')} 行）。"
        "这等于**同一个尝试做了 k 遍**：k 倍的对局开销，只换回 1 个假设的证据。"
        "本轮请让每个候选承载**一个不同的优化假设**（不同的取胜路径 / 不同的机制），"
        "而不是同一骨架换阈值。"
    )
