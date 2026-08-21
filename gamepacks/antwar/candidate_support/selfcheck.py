#!/usr/bin/env python3
"""候选自检：在提交 action.json 之前，自己确认 k 个候选都是**能启动的合法代码**。

为什么容器里需要这个
--------------------
框架侧本来就有同一套前置校验，不合法的候选**不会**进对局。但那是在你提交之后
才发生的：你要等一整轮，才从反馈里读到"你的候选缺 ai.py"。一次 8 局的对局
是几十秒到几分钟，加上一轮思考，代价是一整轮迭代。

真实教训：曾有一次 antwar2 连跑 5 轮，32 局全是"0 回合判负"，根因只是候选包里
没有 ``ai.py``（入口第一行就是 ``from ai import AI``）。5 轮迭代报废，
曲线上是一条毫无信息的水平线。这类失败 1 秒就能判掉。

**这不是自评测。** 这里不打对局、不估胜率、不需要对手，只回答一个问题：
"这份代码能不能被判题器拉起来"。容器规则禁止的是自己组织比赛，不是禁止你
检查自己的代码合法性。

口径与框架侧完全一致
--------------------
判定逻辑来自 ``selfcheck_lib.py``，它是框架那份前置校验的**同一份代码**
（拷贝进来的，零改动），所以这里通过、框架侧就一定通过；不会出现两套标准。

用法::

    python3 selfcheck.py            # 检查 .agentbench/rollouts/ 下所有候选
    python3 selfcheck.py --workspace-only   # 只检查工作区当前版本
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from selfcheck_lib import check_candidate
except ImportError:  # pragma: no cover - 脚手架不完整时给出可执行的提示
    print("selfcheck_lib.py 不在当前目录，无法自检；请在候选工作区根目录运行本脚本。")
    raise SystemExit(2) from None

# 与框架侧 _snapshot_root 一致：这些目录不是候选代码，叠加时必须排除。
IGNORED = shutil.ignore_patterns(
    "feedback", "research", "snapshots", "processed-requests", "rollouts", "runtime-tmp"
)


def _overlay(workspace: Path, overlay: Path | None, destination: Path) -> Path:
    """把候选叠加层铺在工作区当前版本之上，得到判题器真正会看到的那份代码。"""

    shutil.copytree(workspace, destination, ignore=IGNORED)
    if overlay is not None and overlay.is_dir():
        shutil.copytree(overlay, destination, dirs_exist_ok=True)
    return destination


def _interface(workspace: Path) -> str | None:
    """从 CANDIDATE_CONTRACT.md 里取声明的接口（如 ``AI.choose_operations``）。"""

    contract = workspace / "CANDIDATE_CONTRACT.md"
    if not contract.is_file():
        return None
    for line in contract.read_text(encoding="utf-8", errors="replace").splitlines():
        if "candidate_interface" in line and "." in line:
            for token in line.replace("`", " ").replace("=", " ").split():
                if "." in token and token.split(".", 1)[0].isidentifier():
                    return token.strip(" ,;:")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="候选合法性自检（不打对局）")
    parser.add_argument("--workspace-only", action="store_true", help="只检查工作区当前版本")
    args = parser.parse_args()

    workspace = Path.cwd()
    interface = _interface(workspace)
    rollouts = workspace / ".agentbench" / "rollouts"
    targets: list[tuple[str, Path | None]] = []
    if not args.workspace_only and rollouts.is_dir():
        targets = [(item.name, item) for item in sorted(rollouts.iterdir()) if item.is_dir()]
    if not targets:
        targets = [("<workspace>", None)]

    print(f"自检 {len(targets)} 个目标（接口声明：{interface or '未声明'}）")
    failed = 0
    for candidate_id, overlay in targets:
        with tempfile.TemporaryDirectory(prefix="selfcheck-") as tmp:
            root = _overlay(workspace, overlay, Path(tmp) / "candidate")
            issues = check_candidate(candidate_id, root, candidate_interface=interface)
        if not issues:
            print(f"  ✅ {candidate_id}")
            continue
        failed += 1
        for issue in issues:
            print(f"  ❌ {issue.as_note()}")

    if failed:
        print(
            f"\n{failed}/{len(targets)} 个候选不合法。修好再写 action.json —— "
            "提交这样的候选只会换回一份「你崩了」的反馈，白烧一整轮。"
        )
        return 1
    print("\n全部通过，可以提交 action.json。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
