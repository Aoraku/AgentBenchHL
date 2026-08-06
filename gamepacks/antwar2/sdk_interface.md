# AntWar2 候选接口

提交包包含 `ai.py`、`common.py`、`main.py`、`protocol.py` 和冻结 `SDK/`。
`ai.py` 必须导出 `AI`，公共入口为：

```python
AI.choose_operations(
    state: SDK.backend.state.BackendState,
    player: int,
    bundles: list[SDK.utils.actions.ActionBundle] | None = None,
) -> list[SDK.backend.model.Operation]
```

同一源码支持 P0/P1。返回列表保持执行顺序，协议层用
`state.can_apply_operation(player, operation, accepted_prefix)` 过滤合法前缀。

使用 `Operation(OperationType, arg0, arg1)` 构造原子操作。建塔和武器参数为坐标，
塔升级参数为全局塔 ID 与目标类型，降级只传塔 ID，基地升级无参数。

允许只由公开观察派生的有限内部记忆。禁止对手身份、seed、replay ID 和隐藏状态。
