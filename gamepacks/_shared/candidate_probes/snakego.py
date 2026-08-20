"""snakego 脚手架验收探针：最小合法 ai.py。

snakego 的动作**必须合法**——非法动作会被官方 ``controller.apply`` 判为
``Illegal Action`` 并抛错，整局作废。所以探针不能"随便返回 1"，而是直接复用
官方 SDK 自带的示例策略（``sampleAI.AI.judge``）。

探针的目的是验证"脚手架 + 官方 SDK 能跑完整局"，不是验证策略强度。
"""

from sampleAI import AI as OfficialSampleAI


class AI(OfficialSampleAI):
    """原样沿用官方示例策略。"""
