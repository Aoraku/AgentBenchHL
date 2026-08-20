"""改动依据、旧版本可回退、座次不对称、容器内自检。

依据不是理论，是一份实际刷到 SOTA 的 if-else 策略（约 2300 行）的性质：

* 每一代改动都能追到一条机制或一段回放证据（注释里写着「v14 回放里的那个停顿」）；
* 旧版本行为始终以**可调用的形式**留着，所以能回退，也能按局面在两个历史版本
  之间切换——这类组合要求旧版本没被覆盖掉；
* 最终入口是**座次不对称**的：P0 走一套策略，P1 走另一套。

刻意**不**做两件事：不锁定实现形式（那份代码用继承链，但那只是"限定只能写
if-else"这个约束下的一种手法，组合/委托、可替换部件、参数表都能达到同样性质），
也**不引导改动幅度**。框架对"改一条规则"还是"换整条战略"没有偏好——该改多少
由机制与回放里的证据决定，我们只要求 rationale 能说清依据、假设和证伪信号。
早先版本的指令里写过"每个候选只改一件事"和"改动要局部化"，那是错的：
它把迭代压成了调参式的随机游走。这组测试同时防止那类幅度引导被写回去。
"""

from __future__ import annotations

from pathlib import Path

from agentbench_hl.application import goal_led_service as service_module
from agentbench_hl.application.candidate_diversity import base_classes, method_surface

SOURCE = Path(service_module.__file__).read_text(encoding="utf-8")


def test_change_size_is_driven_by_evidence_not_by_us() -> None:
    assert "改多少由证据决定" in SOURCE
    assert "该改一条规则就改一条，该换整条战略就换整条" in SOURCE
    assert "框架对改动幅度没有偏好" in SOURCE
    # rationale 要求：依据、假设、证伪信号——这才是我们真正约束的东西。
    assert "赌的核心假设" in SOURCE
    assert "算它被证伪" in SOURCE


def test_no_change_size_steering_leaks_back_in() -> None:
    """既不许引导少改，也不许引导多改。"""

    for banned in (
        "只改一件事",
        "改动要局部化",
        "只覆盖你要改的那一两个方法",
        "改动幅度不设上限",
        "大胆重构比小步修补",
        "不要做微调式的随机游走",
    ):
        assert banned not in SOURCE, f"指令里又出现了幅度引导：{banned}"


def test_instructions_keep_old_versions_callable() -> None:
    assert "旧版本留得住" in SOURCE
    assert "按局面组合" in SOURCE
    # 形式不限，否则又变成一种强制写法。
    assert "怎么实现随你" in SOURCE


def test_instructions_allow_role_asymmetric_strategies() -> None:
    assert "座次可以不对称" in SOURCE
    assert "按 player 分派" in SOURCE


def test_prompts_require_selfcheck_before_submitting() -> None:
    """自检必须在**提交前**跑：框架侧也会查，但那要等一整轮才反馈。"""

    assert SOURCE.count("selfcheck.py") >= 2  # 第 0 轮 + 每轮反馈都要提
    assert "不通过就不要提交" in SOURCE


def test_method_surface_reads_overridden_methods(tmp_path: Path) -> None:
    """方法级视图：改动局部化时，"改了什么"≈"动了哪些方法"。"""

    root = tmp_path / "v004"
    root.mkdir()
    (root / "ai.py").write_text(
        "\n".join(
            [
                "from strategy_v003 import V003Agent",
                "",
                "class V004GuardAgent(V003Agent):",
                "    def _pick_target(self, state):",
                "        return None",
                "",
                "    def _budget(self, state):",
                "        return 0",
                "",
                "class AI(V004GuardAgent):",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )

    assert method_surface(root) == {
        "V004GuardAgent._pick_target",
        "V004GuardAgent._budget",
    }
    assert base_classes(root) == {"V003Agent", "V004GuardAgent"}


def test_method_surface_tolerates_broken_code(tmp_path: Path) -> None:
    """度量不负责报语法错（那是 preflight 的活），不能因此抛异常。"""

    root = tmp_path / "broken"
    root.mkdir()
    (root / "ai.py").write_text("class Oops(:\n", encoding="utf-8")
    assert method_surface(root) == set()
    assert base_classes(root) == set()


def test_selfcheck_scaffold_is_shipped_into_containers() -> None:
    """自检脚本与它的判定库必须真的进候选包，否则指令里那句话是空的。"""

    generator = Path(__file__).resolve().parents[2] / "scripts" / "gen_candidate_support.py"
    source = generator.read_text(encoding="utf-8")
    assert 'staging / "selfcheck.py"' in source
    # 判定逻辑必须是框架侧那份代码本身，避免容器内过了、框架侧又打回。
    assert 'staging / "selfcheck_lib.py"' in source
    assert "candidate_preflight.py" in source

    shared = (
        Path(__file__).resolve().parents[2]
        / "gamepacks"
        / "_shared"
        / "candidate_support"
        / "selfcheck.py"
    )
    text = shared.read_text(encoding="utf-8")
    assert "这不是自评测" in text  # 必须和"禁止容器内自对弈"划清界限
    assert "check_candidate" in text
