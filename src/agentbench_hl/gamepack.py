"""GamePack 解析器 —— 支持引用 A 仓资产，杜绝跨仓资源重复。

**为什么需要**：规则 / 决策空间 / 回放字段说明的**唯一权威源是 A**
（`AgentBench/games/<game>/`）。早期 B 的 GamePack 把它们**复制**了一份，结果两边
已经开始漂移（antwar2 的 rules.md 在 A 是 3407 字节、在 B 是 2308 字节）。一旦 A 更新
规则，B 的 Goal 读到的就是过期材料，而且没人会发现。

**做法**：GamePack 的每个文档字段都可以写成引用：

```yaml
rules: "@agentbench:games/antwar2/rules.md"     # 指向 A，运行时解析
replay_skill: replay_skill.md                    # B 独有材料，仍在 GamePack 内
```

`@agentbench:` 前缀相对 `AGENTBENCH_ROOT` 解析。B 只保留**B 独有**的东西：
回放阅读技能（回放→自然语言）、SDK 接口说明、Goal 章程、候选脚手架、开发矩阵、
认证配置。

**可复现性**：每次 run 会把解析后的材料快照到 `run_root/frozen-gamepack/` 并记录
sha256（`materialize()` 返回值写进 run 清单）。所以"引用而非复制"不牺牲可复现性。
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

REFERENCE_PREFIX = "@agentbench:"

# GamePack 文档字段 -> 是否必需
DOCUMENT_FIELDS: dict[str, bool] = {
    "rules": True,
    "decision_space": True,
    "replay_skill": True,
    "sdk_interface": False,
    "replay_format": False,
    "goal_charter": False,
    "development_matrix": False,
}

DIRECTORY_FIELDS: dict[str, bool] = {
    "candidate_support": False,
}


class GamePackError(RuntimeError):
    """GamePack 缺失、字段非法或引用无法解析。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class GamePack:
    """一个游戏的"研究输入包"，字段可引用 A 仓。"""

    game: str
    root: Path
    agentbench_root: Path
    document: Mapping[str, object]

    # -- 装载 ---------------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path, *, agentbench_root: str | Path) -> GamePack:
        pack_root = Path(root).resolve()
        manifest_path = pack_root / "manifest.yaml"
        if not manifest_path.is_file():
            raise GamePackError(f"GamePack manifest not found: {manifest_path}")
        value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise GamePackError(f"GamePack manifest must be a mapping: {manifest_path}")
        document = cast(Mapping[str, object], value)
        game = document.get("game")
        if not isinstance(game, str) or not game:
            raise GamePackError(f"GamePack manifest has no game name: {manifest_path}")
        return cls(
            game=game,
            root=pack_root,
            agentbench_root=Path(agentbench_root).resolve(),
            document=document,
        )

    # -- 引用解析 -----------------------------------------------------------

    def resolve(self, value: str, *, field: str) -> Path:
        """把一个字段值解析为绝对路径（支持 ``@agentbench:`` 引用）。"""

        if value.startswith(REFERENCE_PREFIX):
            relative = value[len(REFERENCE_PREFIX) :].lstrip("/")
            target = (self.agentbench_root / relative).resolve()
            if not target.exists():
                raise GamePackError(
                    f"GamePack field {field!r} references missing AgentBench asset: {target}"
                )
            return target
        target = (self.root / value).resolve()
        if not target.exists():
            raise GamePackError(f"GamePack field {field!r} points to missing path: {target}")
        return target

    def path_for(self, field: str, *, required: bool | None = None) -> Path | None:
        """取某字段对应的路径；字段缺失时按 required 决定报错或返回 None。"""

        is_required = (
            required
            if required is not None
            else DOCUMENT_FIELDS.get(field, DIRECTORY_FIELDS.get(field, False))
        )
        raw = self.document.get(field)
        if raw is None:
            # 约定回落：同名文件存在即用（兼容老 GamePack）。
            for candidate in (self.root / f"{field}.md", self.root / f"{field}.yaml"):
                if candidate.is_file():
                    return candidate
            if is_required:
                raise GamePackError(f"GamePack {self.game} is missing required field {field!r}")
            return None
        if not isinstance(raw, str) or not raw.strip():
            raise GamePackError(f"GamePack field {field!r} must be a non-empty string")
        return self.resolve(raw.strip(), field=field)

    # -- 隔离清单 -----------------------------------------------------------

    @property
    def learning_isolation(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        raw = self.document.get("learning_isolation")
        if raw is None:
            return ((), ())
        if not isinstance(raw, Mapping):
            raise GamePackError("learning_isolation must be a mapping")
        allowed = raw.get("allowed") or ()
        forbidden = raw.get("forbidden") or ()
        if not isinstance(allowed, Sequence) or not isinstance(forbidden, Sequence):
            raise GamePackError("learning_isolation.allowed/forbidden must be lists")
        return (
            tuple(str(item) for item in allowed),
            tuple(str(item) for item in forbidden),
        )

    @property
    def candidate_interface(self) -> str | None:
        value = self.document.get("candidate_interface")
        return value if isinstance(value, str) and value else None

    # -- 物化（run 级冻结快照） --------------------------------------------

    def materialize(self, destination: str | Path) -> dict[str, str]:
        """把全部研究输入快照到 ``destination``，返回 name -> sha256。

        Goal 只能看到这个快照目录（隔离配置里唯一允许读的 GamePack 根），因此
        "引用 A"不会让 Goal 顺着引用读到 A 仓的其它东西（如人类源码）。
        """

        dest = Path(destination).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        digests: dict[str, str] = {}

        for field in DOCUMENT_FIELDS:
            source = self.path_for(field)
            if source is None:
                continue
            target = dest / f"{field}{source.suffix}"
            shutil.copy2(source, target)
            digests[field] = _sha256(target)

        for field in DIRECTORY_FIELDS:
            source = self.path_for(field)
            if source is None:
                continue
            target = dest / field
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            digests[field] = _tree_sha256(target)

        manifest = {
            "schema_version": "1.1",
            "game": self.game,
            "candidate_interface": self.candidate_interface,
            "sources": {
                field: str(self.path_for(field))
                for field in (*DOCUMENT_FIELDS, *DIRECTORY_FIELDS)
                if self.path_for(field) is not None
            },
            "digests": digests,
        }
        (dest / "gamepack-manifest.json").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        return digests


def discover_gamepacks(gamepacks_root: str | Path) -> tuple[str, ...]:
    """列出所有可用 GamePack（目录名 = 游戏名，须含 manifest.yaml）。"""

    root = Path(gamepacks_root)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            item.name
            for item in root.iterdir()
            if item.is_dir()
            and not item.name.startswith("_")
            and (item / "manifest.yaml").is_file()
        )
    )
