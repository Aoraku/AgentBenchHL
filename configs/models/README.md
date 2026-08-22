# 模型档案库

主表要横向比 7 个模型，而它们的**中转站、api key、上下文窗口、model catalog
全都不一样**。把这些散在每个实验配置里的后果是实测过的：漏掉一个
`context_window`，codex 就用兜底模型元数据，远程压缩在一个我们没设过也看不见
的点触发，然后打死整个 run（antwar2 死在 97k，而当时声明的是 200000）。

所以模型独立成档案。实验配置里只写一行：

```yaml
provider:
  model_profile: glm-5.3
```

需要临时覆盖某个字段（例如试一档不同的 `reasoning_effort`）时，在实验配置的
`provider:` 里同名写一遍即可 —— 实验里的值优先于档案。

## 档案里必填三项

| 字段 | 含义 |
|---|---|
| `model` | 中转站认的模型名（不是我们内部叫法） |
| `base_url` | 中转站端点 |
| `api_key_env` | 从哪个环境变量读 key（**绝不把 key 写进文件**） |

## base_url 有个容易踩的坑

codex 的请求路径 `/responses` 是**直接拼接**在 `base_url` 后面的
（见 `adapters/codex_goal/responses_proxy.py` 的 `path = upstream.path + self.path`）：

- teamorouter 的端点是 `/v1/responses` → `base_url` 必须带 `/v1`
- 清华中转的 `.../sub2api` 自身就是根 → **不带** `/v1`

写错的表现是 404，而且返回的是 HTML 首页，报错信息极具误导性。

## 非 OpenAI 家的模型必须给 model_catalog

codex 自带的模型目录只认识自家模型（`codex debug models` 只返回 gpt-5.x）。
glm / kimi / qwen 这些不配 `model_catalog` 就会打印
`Unknown model … will use fallback model metadata`，上下文窗口与压缩阈值
全走兜底值。当前支持的 catalog：`zhipu`。

## 当前状态

| 档案 | 状态 |
|---|---|
| `glm-5.2` | ✅ 跑通（8 游戏烟测都用它） |
| `glm-5.3` | ✅ 跑通，但会 429 限流 |
| 其余 5 个 | ⚠️ 端点与 key 待填（占位档案，`api_key_env` 已就位） |

占位档案里的 `base_url` 是待确认值。拿到真实中转站后只改这一处，
所有引用该 profile 的实验自动跟上。
