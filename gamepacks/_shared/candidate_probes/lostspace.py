"""lostspace 脚手架验收探针：最小合法 ai.py。

官方单文件 SDK 的 ``AIClient.play`` 本身就带一份可用的示例实现
（探物资箱 → 交互密钥机 → 试逃生舱 → 移动），直接继承即可。
"""

from lostspace_sdk import AIClient


class AI(AIClient):
    """原样沿用官方 AIClient 自带的示例 play()。"""
