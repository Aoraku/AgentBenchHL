"""模型档案的健全性校验。

这组测试针对的是**配错了不会报错、只会在几十分钟后打死 run** 的字段。
它们全都来自真实事故，不是假想。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MODELS_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"

#: 远程压缩阈值的下限。
#:
#: 为什么要卡下限而不是上限：codex 的 remote compaction 对多数非 OpenAI 模型
#: **必定失败** —— 模型返回 ``[reasoning, message]`` 两个 item，而 codex v2
#: 只接受一个::
#:
#:     Error running remote compact task: Fatal error: remote compaction v2
#:     expected exactly one compaction output item, got 0 from 2 output items
#:
#: 一旦触发就把整个 turn 打成 failed，run 直接死。所以阈值**低 = 主动引雷**，
#: 与"留个保险"的直觉完全相反。
#:
#: 实测事故：档案里写成 90000（比 900000 少一个 0），于是 verify-glm-5.3
#: 在第 0 轮、smoke8-miracle 在第 2 轮都被压缩打死；而历史上跑通的 run
#: （snakego4 / aquawar4 / g4 / m4）用的都是 900000。
#:
#: 400000 这个下限取自已知能跑通的最小值（opus-5 档案），
#: 足以拦住"少一个零"这一类错误 —— 那正是人眼最容易漏掉的。
MIN_AUTO_COMPACT_LIMIT = 400_000


def _profiles() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for path in sorted(MODELS_DIR.glob("*.yaml")):
        out.append((path.name, yaml.safe_load(path.read_text(encoding="utf-8")) or {}))
    return out


@pytest.mark.parametrize(("name", "profile"), _profiles())
def test_auto_compact_limit_is_high_enough_to_never_fire(name: str, profile: dict) -> None:
    """压缩阈值必须高到让远程压缩永不触发。见 MIN_AUTO_COMPACT_LIMIT 的详注。"""

    limit = profile.get("auto_compact_token_limit")
    if limit is None:
        return  # 不设也可以（用 codex 默认），但设了就不能设低
    assert limit >= MIN_AUTO_COMPACT_LIMIT, (
        f"{name}: auto_compact_token_limit={limit} 太低，会主动触发对多数非 OpenAI "
        f"模型必定失败的远程压缩（少写一个 0 是最常见的原因）"
    )


@pytest.mark.parametrize(("name", "profile"), _profiles())
def test_client_whitelisted_relays_report_the_official_originator(
    name: str, profile: dict
) -> None:
    """**只有** sbtunnel 这类按客户端白名单放行的中转需要 ``client_name``。

    sbtunnel 对框架默认的 ``originator: agentbench-hl`` 返回
    ``403 This account only allows Codex official clients``。

    但**不要**把它当成所有中转的默认动作：``sota-antwar2`` 用框架默认
    originator 在清华 sub2api 上跑通了 40+ 轮。给不需要的中转加这个字段
    是凭空增加一个与验证过的配置的差异 —— 而每一处这样的差异都要重新验证。
    """

    if profile.get("harness") not in (None, "codex"):
        return
    url = profile.get("base_url") or ""
    if "sbtunnel" not in url and "teamorouter" not in url:
        return
    assert profile.get("client_name"), (
        f"{name}: 这个中转按客户端白名单放行，不写 client_name 会被拒为 "
        "'This account only allows Codex official clients'"
    )


@pytest.mark.parametrize(("name", "profile"), _profiles())
def test_model_slug_matches_the_builtin_catalog_casing(name: str, profile: dict) -> None:
    """用了内置 catalog 时，``model`` 必须与 catalog 里的 slug **大小写一致**。

    这是一个"看起来能用、实际必死"的坑：中转本身不区分大小写，所以 curl 和
    ``codex exec`` 用大写 ``GLM-5.3`` 都通；但 codex 是按 slug **精确匹配**
    catalog 的，大小写不符就退回兜底元数据（很小的 context_window），
    于是压缩在一个我们没设过的点触发 —— 而 glm 系的远程压缩必定失败，
    整个 run 死。

    实测：把档案从 ``glm-5.3`` 改成 ``GLM-5.3`` 后，verify-glm-5.3 在第 0 轮、
    smoke8-miracle 在第 2 轮都被压缩打死；而跑通 40+ 轮的 sota-antwar2
    用的是小写。
    """

    catalog = profile.get("model_catalog")
    model = profile.get("model")
    if not catalog or not model:
        return
    from agentbench_hl.adapters.codex_goal.app_server import MODEL_CATALOGS

    slugs = {str(entry["slug"]) for entry in MODEL_CATALOGS[str(catalog)]}
    assert model in slugs, (
        f"{name}: model={model!r} 不在 {catalog} catalog 的 slug 里 {sorted(slugs)}；"
        "大小写不符会让 codex 退回兜底元数据并触发必死的远程压缩"
    )


@pytest.mark.parametrize(("name", "profile"), _profiles())
def test_base_url_matches_the_relay_convention(name: str, profile: dict) -> None:
    """``/responses`` 是直接拼在 ``base_url`` 后面的，所以两个中转的写法不同。

    * teamorouter / sbtunnel 的端点是 ``/v1/responses`` → 必须带 ``/v1``
    * 清华 ``.../sub2api`` 自身就是根 → **不带** ``/v1``

    写错的表现是 404，而且返回的是站点 HTML 首页，报错极具误导性。
    """

    url = profile.get("base_url")
    if not url:
        return
    if "sub2api" in url:
        assert not url.rstrip("/").endswith("/v1"), (
            f"{name}: 清华 sub2api 自身就是根，带 /v1 会 404 并返回 HTML 首页"
        )
    if "sbtunnel" in url or "teamorouter" in url:
        assert url.rstrip("/").endswith("/v1"), (
            f"{name}: 这个中转的端点是 /v1/responses，base_url 必须带 /v1"
        )


@pytest.mark.parametrize(("name", "profile"), _profiles())
def test_secrets_never_live_in_profiles(name: str, profile: dict) -> None:
    """档案里只能有 ``api_key_env``（环境变量名），绝不能有 key 本身。"""

    blob = yaml.safe_dump(profile)
    assert "sk-" not in blob, f"{name}: 档案里出现了疑似 api key"
    assert profile.get("api_key_env"), f"{name}: 缺 api_key_env"
