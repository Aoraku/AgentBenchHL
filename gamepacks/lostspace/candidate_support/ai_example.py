"""lostspace 候选**格式示例** —— 由 gen_candidate_support.py 生成。

这份文件只回答一个问题：**「怎么写才符合提交格式」**。
策略部分是刻意留白的占位（返回第一个合法动作 / 固定动作），强度为零。

把它另存为 `ai.py` 再动手写你自己的策略。入口只会加载 `ai.py`，
所以这份示例本身永远不会被当成候选提交。
"""

from __future__ import annotations

class AI:
    def play(self, *args, **kwargs):
        # 占位策略：什么都不做，直接结束本回合。
        return ("finish", None)
