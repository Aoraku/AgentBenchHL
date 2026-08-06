"""Content-addressed, immutable-by-convention candidate artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

_CREDENTIAL = re.compile(rb"\bsk-[A-Za-z0-9_-]{8,}")
_EXCLUDED_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".agentbench"})
_REQUIRED_FILES = frozenset({"ai.py", "main.py", "common.py", "protocol.py"})


class FilesystemArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects_root = self.root / "objects"

    def materialize(self, source: Path) -> tuple[str, Path, dict[str, str]]:
        source = source.resolve()
        source_hashes = self._validate_and_hash(source)
        tree_hash = self._tree_hash(source_hashes)
        destination = self.objects_root / tree_hash
        if destination.exists():
            return tree_hash, destination, source_hashes

        self.objects_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{tree_hash}.", dir=self.objects_root))
        try:
            for relative in sorted(source_hashes):
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
            try:
                os.replace(temporary, destination)
            except FileExistsError:
                shutil.rmtree(temporary)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return tree_hash, destination, source_hashes

    def _validate_and_hash(self, source: Path) -> dict[str, str]:
        if not source.is_dir():
            raise ValueError(f"candidate workspace is not a directory: {source}")
        missing = sorted(name for name in _REQUIRED_FILES if not (source / name).is_file())
        if missing:
            raise ValueError(f"candidate workspace missing required files: {', '.join(missing)}")
        if not (source / "SDK").is_dir():
            raise ValueError("candidate workspace missing required SDK directory")

        hashes: dict[str, str] = {}
        for path in sorted(source.rglob("*")):
            relative_path = path.relative_to(source)
            if any(part in _EXCLUDED_DIRECTORIES for part in relative_path.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"candidate workspace contains symlink: {relative_path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"candidate workspace contains unsupported path: {relative_path}")
            resolved = path.resolve()
            if not resolved.is_relative_to(source):
                raise ValueError(f"candidate path escapes workspace: {relative_path}")
            content = path.read_bytes()
            if _CREDENTIAL.search(content):
                raise ValueError(f"candidate workspace contains credential: {relative_path}")
            hashes[relative_path.as_posix()] = hashlib.sha256(content).hexdigest()
        return hashes

    @staticmethod
    def _tree_hash(source_hashes: dict[str, str]) -> str:
        digest = hashlib.sha256()
        for relative, source_hash in sorted(source_hashes.items()):
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source_hash.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()
