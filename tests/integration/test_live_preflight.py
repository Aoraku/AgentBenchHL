from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import agentbench_hl.application.live_run as live_run
from agentbench_hl.adapters.antwar2.runtime import Opponent
from agentbench_hl.adapters.codex_goal.read_isolation import (
    isolated_app_server_command,
    write_candidate_isolation_profile,
    write_read_isolation_profile,
)
from agentbench_hl.adapters.antwar2.factory import official_human_ratings
from agentbench_hl.application.live_run import (
    build_live_run,
    probe_codex_installation,
    resume_live_run,
)


def test_new_run_requires_the_api_key_before_creating_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    config_root = repository / "configs/experiments"
    config_root.mkdir(parents=True)
    (repository / "gamepacks/antwar2").mkdir(parents=True)
    config = config_root / "antwar2.yaml"
    config.write_text(
        """schema_version: "1.0"
game: antwar2
origin: from_scratch
provider:
  model: gpt-5.5
  reasoning_effort: xhigh
  base_url: https://provider.invalid
  api_key_env: ABHL_API_KEY
  disable_response_storage: true
runtime:
  codex_binary: codex
  branch_width: 1
  max_iterations: null
  network_access: disabled
paths:
  agentbench_root: ${AGENTBENCH_ROOT}
  runs_root: ../../../runs
curriculum:
  order: lowest_rank_first
  development_seeds: [1]
measurement:
  epsilon: 0.01
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTBENCH_ROOT", str(tmp_path / "bench"))
    monkeypatch.delenv("ABHL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="missing API key"):
        build_live_run(config, run_id="missing-key")

    assert not (tmp_path / "runs/missing-key").exists()


def test_codex_preflight_records_version_goal_feature_and_schema_hash(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "codex-fixture"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("codex-cli fixture-1")
elif args == ["features", "list"]:
    print("goals stable true")
elif args[:3] == ["app-server", "generate-json-schema", "--experimental"]:
    if "--out-dir" in args or "--out" not in args:
        raise SystemExit(3)
    output = pathlib.Path(args[args.index("--out") + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / "schema.json").write_text(json.dumps({"fixture": True}))
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    result = probe_codex_installation(str(executable), tmp_path / "probe")

    assert result.version == "codex-cli fixture-1"
    assert result.goals_enabled is True
    assert len(result.schema_tree_sha256) == 64
    assert result.schema_file_count == 1


def test_codex_preflight_canonicalizes_schema_object_order(tmp_path: Path) -> None:
    executable = tmp_path / "codex-fixture"
    executable.write_text(
        f"""#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
counter = pathlib.Path({str(tmp_path / "counter")!r})
if args == ["--version"]:
    print("codex-cli fixture-1")
elif args == ["features", "list"]:
    print("goals stable true")
elif args[:3] == ["app-server", "generate-json-schema", "--experimental"]:
    output = pathlib.Path(args[args.index("--out") + 1])
    output.mkdir(parents=True, exist_ok=True)
    invocation = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(invocation + 1))
    aggregate = '{{"a":1,"b":2}}' if invocation == 0 else '{{"b":2,"a":1}}'
    (output / "codex_app_server_protocol.v2.schemas.json").write_text(aggregate)
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    first = probe_codex_installation(str(executable), tmp_path / "probe-a")
    second = probe_codex_installation(str(executable), tmp_path / "probe-b")

    assert first.schema_tree_sha256 == second.schema_tree_sha256


def test_legacy_codex_identity_uses_stable_fields_when_raw_schema_hash_varies() -> None:
    resources = {
        "codex_version": "codex-cli 0.146.0-alpha.9.2",
        "codex_goals_enabled": True,
        "codex_schema_tree_sha256": "a" * 64,
        "codex_schema_file_count": 349,
    }
    probe = live_run.CodexInstallationProbe("codex-cli 0.146.0-alpha.9.2", True, "b" * 64, 349)

    assert live_run._codex_identity_matches(resources, probe)


def test_canonical_codex_identity_requires_the_schema_hash() -> None:
    resources = {
        "codex_version": "codex-cli fixture-1",
        "codex_goals_enabled": True,
        "codex_schema_tree_sha256": "a" * 64,
        "codex_schema_file_count": 1,
        "codex_schema_hash_kind": "canonical-json-v1",
    }
    probe = live_run.CodexInstallationProbe("codex-cli fixture-1", True, "b" * 64, 1)

    assert not live_run._codex_identity_matches(resources, probe)


def test_goal_app_server_command_avoids_nested_macos_seatbelt() -> None:
    command = live_run._goal_app_server_command("codex")

    assert command == (
        "codex",
        "app-server",
        "--listen",
        "stdio://",
        "--strict-config",
    )
    assert "/usr/bin/sandbox-exec" not in command


def test_remote_provider_uses_local_responses_compat_proxy() -> None:
    assert live_run._use_responses_compat_proxy(
        "https://lab.cs.tsinghua.edu.cn/ai-platform/sub2api"
    )
    assert not live_run._use_responses_compat_proxy("http://127.0.0.1:8123/v1")


def test_resume_live_run_rejects_an_unknown_run_before_provider_startup(
    monkeypatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    config_root = repository / "configs/experiments"
    config_root.mkdir(parents=True)
    (repository / "gamepacks/antwar2").mkdir(parents=True)
    config = config_root / "antwar2.yaml"
    config.write_text(
        """schema_version: "1.0"
game: antwar2
origin: from_scratch
provider:
  model: gpt-5.5
  reasoning_effort: xhigh
  base_url: https://provider.invalid
  api_key_env: ABHL_API_KEY
  disable_response_storage: true
runtime:
  codex_binary: codex
  branch_width: 1
  max_iterations: null
  network_access: disabled
paths:
  agentbench_root: ${AGENTBENCH_ROOT}
  runs_root: ../../../runs
curriculum:
  order: lowest_rank_first
  development_seeds: [1]
measurement:
  epsilon: 0.01
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTBENCH_ROOT", str(tmp_path / "bench"))
    monkeypatch.setenv("ABHL_API_KEY", "fixture-secret")

    with pytest.raises(FileNotFoundError, match="run manifest not found"):
        resume_live_run(config, run_id="missing-run")


def test_official_human_ratings_use_the_frozen_ladder_scores(tmp_path: Path) -> None:
    opponents = (
        Opponent(
            "rank01",
            1,
            "a",
            1812,
            tmp_path / "a.zip",
            "a" * 64,
            tmp_path,
            True,
            ("python",),
            None,
        ),
        Opponent(
            "rank02",
            2,
            "b",
            1644,
            tmp_path / "b.zip",
            "b" * 64,
            tmp_path,
            True,
            ("python",),
            None,
        ),
    )

    ratings = official_human_ratings(opponents)

    assert ratings == {"rank01": 1812.0, "rank02": 1644.0}


@pytest.mark.live
def test_seatbelt_profile_denies_hidden_read_but_allows_candidate(
    tmp_path: Path,
) -> None:
    if os.environ.get("ABHL_RUN_SEATBELT_TEST") != "1":
        pytest.skip("set ABHL_RUN_SEATBELT_TEST=1 to test macOS Seatbelt")
    hidden = tmp_path / "hidden"
    candidate = tmp_path / "candidate"
    hidden.mkdir()
    candidate.mkdir()
    (hidden / "secret.txt").write_text("human-policy", encoding="utf-8")
    (candidate / "ai.py").write_text("policy", encoding="utf-8")
    profile = write_read_isolation_profile(
        tmp_path / "goal-runtime.sb",
        denied_read_roots=(hidden,),
    )

    allowed = subprocess.run(
        isolated_app_server_command(("/bin/cat", str(candidate / "ai.py")), profile),
        capture_output=True,
        text=True,
        check=False,
    )
    denied = subprocess.run(
        isolated_app_server_command(("/bin/cat", str(hidden / "secret.txt")), profile),
        capture_output=True,
        text=True,
        check=False,
    )

    assert allowed.returncode == 0
    assert allowed.stdout == "policy"
    assert denied.returncode != 0
    assert "human-policy" not in denied.stdout


def test_candidate_profile_declares_read_write_and_network_denials(tmp_path: Path) -> None:
    profile = write_candidate_isolation_profile(
        tmp_path / "candidate.sb",
        denied_read_roots=(tmp_path / "hidden",),
    )

    text = profile.read_text(encoding="utf-8")
    assert "(deny network*)" in text
    assert "(deny file-write*)" in text
    assert str((tmp_path / "hidden").resolve()) in text


@pytest.mark.live
def test_candidate_seatbelt_enforces_read_write_and_network_denials(
    tmp_path: Path,
) -> None:
    if os.environ.get("ABHL_RUN_SEATBELT_TEST") != "1":
        pytest.skip("set ABHL_RUN_SEATBELT_TEST=1 to test macOS Seatbelt")
    hidden = tmp_path / "hidden"
    candidate = tmp_path / "candidate"
    hidden.mkdir()
    candidate.mkdir()
    (hidden / "secret.txt").write_text("human-policy", encoding="utf-8")
    (candidate / "ai.py").write_text("policy", encoding="utf-8")
    profile = write_candidate_isolation_profile(
        tmp_path / "candidate.sb",
        denied_read_roots=(hidden,),
    )

    allowed_read = subprocess.run(
        isolated_app_server_command(("/bin/cat", str(candidate / "ai.py")), profile),
        capture_output=True,
        text=True,
        check=False,
    )
    denied_read = subprocess.run(
        isolated_app_server_command(("/bin/cat", str(hidden / "secret.txt")), profile),
        capture_output=True,
        text=True,
        check=False,
    )
    denied_write = subprocess.run(
        isolated_app_server_command(("/usr/bin/touch", str(candidate / "output")), profile),
        capture_output=True,
        text=True,
        check=False,
    )
    denied_network = subprocess.run(
        isolated_app_server_command(
            (
                "/usr/bin/python3",
                "-c",
                "import socket; socket.socket().bind(('127.0.0.1', 0))",
            ),
            profile,
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert allowed_read.returncode == 0
    assert allowed_read.stdout == "policy"
    assert denied_read.returncode != 0
    assert "human-policy" not in denied_read.stdout
    assert denied_write.returncode != 0
    assert not (candidate / "output").exists()
    assert denied_network.returncode != 0
