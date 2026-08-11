"""Frozen AntWar2 resources, builds, and human-pool audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


class AntWar2RuntimeError(RuntimeError):
    """A frozen runtime resource failed an integrity requirement."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AntWar2RuntimeError(f"runtime tree contains a symlink: {path}")
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True)
class AntWar2Layout:
    agentbench_root: Path
    build_root: Path
    backend_archive: Path
    human_manifest: Path
    human_extracted_root: Path
    public_sdk_root: Path

    @classmethod
    def from_root(cls, agentbench_root: str | Path, build_root: str | Path) -> AntWar2Layout:
        agentbench = Path(agentbench_root).resolve()
        ladder = agentbench / "top_algorithms/corpus/30_antwar2_ladder"
        extracted = ladder / "extracted"
        return cls(
            agentbench_root=agentbench,
            build_root=Path(build_root).resolve(),
            backend_archive=(
                agentbench / "backend_sources/corpus/30_antwar2/archives/gamecode_logic__141.zip"
            ),
            human_manifest=ladder / "MANIFEST.tsv",
            human_extracted_root=extracted,
            public_sdk_root=(extracted / "rank01__yyzsanyi__ai_storm__v32" / "SDK"),
        )

    def validate(self, expected_sdk_sha256: str | None = None) -> None:
        for path in (self.backend_archive, self.human_manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in (self.human_extracted_root, self.public_sdk_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if expected_sdk_sha256 is not None:
            actual = tree_sha256(self.public_sdk_root)
            if actual != expected_sdk_sha256:
                raise AntWar2RuntimeError(
                    f"public SDK hash mismatch: expected {expected_sdk_sha256}, got {actual}"
                )


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            relative = Path(member.filename)
            mode = member.external_attr >> 16
            target = (destination / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or stat.S_ISLNK(mode)
                or not target.is_relative_to(resolved_destination)
            ):
                raise AntWar2RuntimeError(f"unsafe archive member: {member.filename}")
        package.extractall(destination)


def validate_cached_backend(
    executable: Path,
    manifest_path: Path,
    archive_sha256: str,
) -> dict[str, object] | None:
    if not executable.is_file() or not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("archive_sha256") != archive_sha256:
        return None
    if value.get("executable_sha256") != sha256_file(executable):
        return None
    return value


def _enable_windows_binary_transport(main_source: Path) -> bool:
    if sys.platform != "win32":
        return False
    text = main_source.read_text(encoding="utf-8")
    if "AGENTBENCH_BINARY_TRANSPORT" in text:
        return True
    include_anchor = "#include <vector>"
    main_anchor = "int main(/*int argc, char *argv[]*/) {"
    if include_anchor not in text or main_anchor not in text:
        raise AntWar2RuntimeError("cannot apply Windows binary transport patch")
    text = text.replace(
        include_anchor,
        include_anchor + "\n#ifdef _WIN32\n#include <fcntl.h>\n#include <io.h>\n#endif",
        1,
    ).replace(
        main_anchor,
        main_anchor
        + "\n#ifdef _WIN32 // AGENTBENCH_BINARY_TRANSPORT\n"
        + "    _setmode(_fileno(stdin), _O_BINARY);\n"
        + "    _setmode(_fileno(stdout), _O_BINARY);\n#endif",
        1,
    )
    main_source.write_text(text, encoding="utf-8")
    return True


@dataclass(frozen=True)
class FrozenBackend:
    executable: Path
    archive_sha256: str
    executable_sha256: str
    manifest_path: Path


def build_backend(layout: AntWar2Layout) -> FrozenBackend:
    layout.validate()
    archive_hash = sha256_file(layout.backend_archive)
    root = layout.build_root / f"backend-{archive_hash[:16]}"
    game_root = root / "game"
    executable = game_root / "output" / ("main.exe" if sys.platform == "win32" else "main")
    manifest_path = root / "build-manifest.json"
    cached = validate_cached_backend(executable, manifest_path, archive_hash)
    if cached is not None:
        return FrozenBackend(
            executable,
            archive_hash,
            str(cached["executable_sha256"]),
            manifest_path,
        )
    if root.exists():
        quarantine = root.with_name(f"{root.name}.invalid")
        if quarantine.exists():
            raise AntWar2RuntimeError(f"backend cache and quarantine both exist: {root}")
        root.rename(quarantine)
    safe_extract(layout.backend_archive, root)
    if not game_root.is_dir():
        raise AntWar2RuntimeError("backend archive has no game directory")
    patched = _enable_windows_binary_transport(game_root / "src/main.cpp")
    completed = subprocess.run(
        ("make", "-C", str(game_root), f"-j{max(1, min(os.cpu_count() or 1, 8))}"),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0 or not executable.is_file():
        diagnostic = (completed.stderr or completed.stdout)[-8000:]
        raise AntWar2RuntimeError(f"backend build failed: {diagnostic}")
    compiler = subprocess.run(
        ("g++", "--version"),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    executable_hash = sha256_file(executable)
    manifest = {
        "schema_version": "1.0",
        "archive": str(layout.backend_archive),
        "archive_sha256": archive_hash,
        "executable": str(executable),
        "executable_sha256": executable_hash,
        "platform": platform.platform(),
        "python": sys.version,
        "compiler": (compiler.stdout or compiler.stderr).splitlines()[0],
        "windows_binary_transport_patch": patched,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return FrozenBackend(executable, archive_hash, executable_hash, manifest_path)


@dataclass(frozen=True)
class Opponent:
    opponent_id: str
    rank: int
    username: str | None
    score: int | None
    archive: Path
    archive_sha256: str
    package_root: Path
    runnable: bool
    entry_command: tuple[str, ...] | None
    exclusion_diagnostic: str | None


def audit_human_pool(layout: AntWar2Layout) -> tuple[Opponent, ...]:
    layout.validate()
    pool: list[Opponent] = []
    with layout.human_manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rank = int(row["rank"])
            packages = tuple(sorted(layout.human_extracted_root.glob(f"rank{rank:02d}__*")))
            if len(packages) != 1:
                raise AntWar2RuntimeError(
                    f"rank{rank:02d} maps to {len(packages)} extracted packages"
                )
            package = packages[0]
            archive = (layout.human_manifest.parent / row["archive"]).resolve()
            if not archive.is_file():
                raise FileNotFoundError(archive)
            main = package / "main.py"
            runnable = main.is_file()
            pool.append(
                Opponent(
                    opponent_id=f"rank{rank:02d}",
                    rank=rank,
                    username=row.get("username") or None,
                    score=int(row["score"]) if row.get("score") else None,
                    archive=archive,
                    archive_sha256=sha256_file(archive),
                    package_root=package,
                    runnable=runnable,
                    entry_command=(sys.executable, "main.py") if runnable else None,
                    exclusion_diagnostic=None if runnable else "missing public main.py entry point",
                )
            )
    pool.sort(key=lambda item: item.rank)
    if [item.rank for item in pool] != list(range(1, len(pool) + 1)):
        raise AntWar2RuntimeError("human ranks must be contiguous from rank01")
    return tuple(pool)


def materialize_bootstrap(layout: AntWar2Layout, support_root: Path, destination: Path) -> None:
    """Copy only the public SDK and policy support; the Goal must create ai.py."""

    layout.validate()
    if destination.exists() and any(destination.iterdir()):
        raise AntWar2RuntimeError(f"bootstrap destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "common.py", "protocol.py"):
        source = support_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)
    shutil.copytree(
        layout.public_sdk_root,
        destination / "SDK",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
