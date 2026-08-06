# GamePack 接入指南

一个 GamePack 描述“模型为了从公开比赛反馈中学习该游戏，必须知道什么”。它不携带
密钥、对手源码、参考策略、隐藏认证 case 或其他 run 的产物。

## 必需内容

在 `gamepacks/<game>/` 创建以下文件：

| 文件 | 负责人 | 内容 |
| --- | --- | --- |
| `rules.md` | 游戏负责人 | 完整规则、角色关系、胜负条件、资源与回合语义。 |
| `decision_space.yaml` | 游戏负责人 | 最本源的原子决策、参数域与每种状态的合法性定义。用于确定性策略行为的 KL。 |
| `replay_skill.md` | 游戏负责人 | 将官方 replay JSON 字段和事件翻译成自然语言游戏事实的方法与例子。 |
| `sdk_interface.md` | 游戏负责人 | 候选策略的稳定公共入口、输入、输出、错误与确定性要求。 |
| `manifest.yaml` | 框架/游戏负责人 | 上述资源路径、公共 SDK 摘要和 Goal 的允许/禁止读取边界。 |
| `candidate_support/` | 游戏负责人 | 可供候选调用、但不包含人类实现的公共工具与协议代码。 |

还需要在 `src/agentbench_hl/adapters/<game>/` 实现：

- `arena.py`：以指定 candidate、对手 ID、角色与 seed 执行冻结官方比赛；
- `replay.py`：保存原始 replay，并生成可审计的公共 trace 与自然语言事实；
- `runtime.py`：将 GamePack、当前候选和 Experience 装入 Goal 的受控工作区；
- `smoke.py`：验证候选可以编译/导入、在双方角色中产生合法动作；
- 必要的 `policy_probe.py`：在固定公开状态集上导出确定性原子动作，供 IG 计算。

## 决策空间原则

决策空间由游戏负责人手工审核，不能为了方便分析而发明抽象“战术假设空间”。它应当
直接对应规则中的可观察、可执行行为，例如移动方向、攻击目标、建造位置、道具选择或
资源操作。

每个状态必须能给出：

1. 状态 ID 与公开状态摘要；
2. 可行动作类型；
3. 每个动作的合法参数域；
4. 策略实际选择的一个确定性原子动作或原子动作序列。

框架对相同状态、相同合法支持上的两个 one-hot 动作计算平滑 KL；不要求策略输出概率
分布。不要把对手身份、seed、replay ID 或隐藏信息放入状态或动作定义。

## 回放阅读 Skill

`replay_skill.md` 必须让模型能回答：发生了什么、何时发生、双方采取了什么行动、哪项
规则或资源变化造成了结果、下一版策略应在什么可观测条件下改变。建议包含：

- JSON 字段与人类可读含义的逐项映射；
- 一个完整的短回放片段及其自然语言时间线；
- 如何识别关键决策窗口、资源临界点、合法动作与终局原因；
- 明确禁止从未公开字段推断对手源码或隐藏状态。

## 配置与评测

新增 `configs/experiments/<game>-goal-I.yaml`，固定：模型/provider 名称、由环境变量给
出的密钥名称、运行根目录、网络策略、开发 seed、测量 epsilon 和课程初始设置。配置
中只能引用 `${ENVIRONMENT_VARIABLE}`，不能包含任何密钥。

Goal 只能通过 `action.json` 提交候选版本和下一位对手。框架执行比赛后回传公开结果与
回放。每个完整候选保留版本、代码、Experience、比赛与指标；失败候选也可被保留为
Frontier，供后续从中继续探索，而不会覆盖 Champion。

## 提交前检查

```bash
python -m pytest tests/golden tests/contract -q
python -m ruff check src tests
git diff --check
git status --ignored
```

确认 `.env`、`runs/`、原始回放、SQLite/App Server 状态和所有 provider 传输记录均处于
ignored 状态，且测试 fixture 不含真实凭据。
