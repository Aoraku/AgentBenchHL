"""出图用的中文字体注册（两个绘图脚本共用）。

为什么需要单独一层
------------------
matplotlib 的字体索引带缓存：新装的字体（例如刚 apt 装上的 fonts-noto-cjk）
不会自动进入缓存，于是 ``font.sans-serif`` 里写了名字也匹配不上，
所有中文渲染成方框。

这类失败很危险，因为它**不报错**：脚本正常退出、PNG 正常生成、数据也完全正确，
只有坐标轴和图例变成一排方框。如果不盯着看，很容易把这种图直接拿去用。

做法是扫已知路径的字体文件并 ``addfont`` 主动注册，绕开"缓存是否新鲜"这个问题。
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager

#: 常见的中文字体落点（Debian/Ubuntu 的 noto、文泉驿，macOS 的苹方/冬青）。
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
)

#: 注册失败时的兜底名单（系统里本来就装好并已进缓存的情况）。
FALLBACK_NAMES = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
)


def register_cjk_fonts() -> list[str]:
    """注册能找到的中文字体，返回可直接写进 ``font.sans-serif`` 的名字列表。"""

    registered: list[str] = []
    for path in FONT_CANDIDATES:
        file = Path(path)
        if not file.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(file))
            name = font_manager.FontProperties(fname=str(file)).get_name()
        except (RuntimeError, OSError):
            # 单个字体加载失败不该让整个出图挂掉，换下一个。
            continue
        if name not in registered:
            registered.append(name)
    return registered


def sans_serif_stack() -> list[str]:
    """完整的字体优先级列表：先用主动注册到的，再退回系统名单。"""

    return [*register_cjk_fonts(), *FALLBACK_NAMES]
