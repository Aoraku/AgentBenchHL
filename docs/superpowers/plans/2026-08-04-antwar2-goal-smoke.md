# AntWar2 Goal-Led Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline-tested Goal-led AntWar2 research kernel and complete one isolated paid smoke that creates `v000` from the frozen GamePack and runs one full-strength official match.

**Architecture:** Domain objects and finalized events form an append-only scientific core. AntWar2, filesystem storage, and Codex App Server are adapters behind narrow ports; the application layer exposes an idempotent JSON CLI used by a persistent Codex Goal. The first deliverable ends after a valid from-scratch `v000`, semantic replay, Experience artifact, metrics row, and recoverable checkpoint exist.

**Tech Stack:** Python 3.11+, standard-library dataclasses/JSON/subprocess, PyYAML 6, matplotlib 3.8+, pytest 8+, ruff, Codex App Server JSON-RPC over stdio, official AntWar2 C++ backend and Python SDK.

## Global Constraints

- AntWar2 only; no website, Rollman, plugin registry, or Framework-led multi-agent scheduler.
- `v000` may read only the frozen GamePack and public SDK before its first match.
- Do not expose human source code, `v22` through `v239`, their Skills, or their replays to Codex.
- Human packages and the official backend remain unmodified and content-hashed.
- API credentials may exist only in process environment or a run-local permission-restricted secret file excluded from events and manifests.
- Research state changes occur only through idempotent application services and finalized events.
- Missing provider token fields are `null`, never zero.
- The default curriculum orders runnable unsolved opponents from rank20 toward rank01.
- No hard auto-rollback and no configured research iteration ceiling.
- All file edits use UTF-8 and all serialized scientific JSON uses sorted keys.

---

## File Map

```text
pyproject.toml                         package, CLI, test and lint configuration
README.md                              installation and reproducible run entry points
src/agentbench_hl/domain/models.py     immutable experiment, candidate, match, usage models
src/agentbench_hl/domain/events.py     finalized event envelope and event validation
src/agentbench_hl/domain/lineage.py    Champion/Frontier/Archive state machine
src/agentbench_hl/domain/experience.py Experience schema and verdicts
src/agentbench_hl/domain/metrics.py    metric row and resource totals
src/agentbench_hl/ports/*.py           runtime, arena, and storage protocols
src/agentbench_hl/adapters/filesystem/ finalized event and artifact stores
src/agentbench_hl/adapters/antwar2/    runtime, arena, replay, legal-action probe
src/agentbench_hl/adapters/codex_goal/ App Server JSON-RPC client and Goal runtime
src/agentbench_hl/application/*.py     candidate, curriculum, research and run services
src/agentbench_hl/reporting/*.py       tables, Markdown reports and curves
src/agentbench_hl/cli/main.py          `abhl` JSON CLI
gamepacks/antwar2/*                    frozen Goal Charter and public game semantics
configs/experiments/*.yaml             reproducible experiment inputs
tests/unit/*                           pure domain and application tests
tests/contract/*                       filesystem, arena and JSON-RPC boundary tests
tests/golden/*                         replay-to-semantics fixtures
tests/e2e/*                            offline fake runtime and opt-in paid smoke tests
```

## Task 1: Package, Typed Configuration, and Frozen GamePack

**Files:**
- Create: `AgentBench-HL/pyproject.toml`
- Create: `AgentBench-HL/README.md`
- Create: `AgentBench-HL/src/agentbench_hl/__init__.py`
- Create: `AgentBench-HL/src/agentbench_hl/config.py`
- Create: `AgentBench-HL/gamepacks/antwar2/GOAL_CHARTER.md`
- Create: `AgentBench-HL/gamepacks/antwar2/rules.md`
- Create: `AgentBench-HL/gamepacks/antwar2/decision_space.yaml`
- Create: `AgentBench-HL/gamepacks/antwar2/replay_skill.md`
- Create: `AgentBench-HL/gamepacks/antwar2/development_matrix.yaml`
- Create: `AgentBench-HL/gamepacks/antwar2/candidate_support/common.py`
- Create: `AgentBench-HL/gamepacks/antwar2/candidate_support/main.py`
- Create: `AgentBench-HL/gamepacks/antwar2/candidate_support/protocol.py`
- Create: `AgentBench-HL/evaluator-config/antwar2-certification.yaml`
- Create: `AgentBench-HL/configs/experiments/antwar2-goal-k1.yaml`
- Test: `AgentBench-HL/tests/unit/test_config.py`

**Interfaces:**
- Consumes: external AgentBench root `/Users/qingle/Code/SAST/AgentBench` only through configuration.
- Produces: `ExperimentConfig.load(path: Path, env: Mapping[str, str]) -> ExperimentConfig`, `EvaluatorConfig.load(path: Path) -> EvaluatorConfig`, `ExperimentConfig.frozen_dict() -> dict[str, object]`, and `ExperimentConfig.secret_environment() -> dict[str, str]`.

- [ ] **Step 1: Write the failing configuration tests**

```python
from pathlib import Path

import pytest

from agentbench_hl.config import ExperimentConfig


def test_config_expands_public_paths_but_never_serializes_key(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
schema_version: '1.0'
game: antwar2
provider:
  model: gpt-5.5
  reasoning_effort: xhigh
  base_url: https://example.invalid/responses
  api_key_env: ABHL_API_KEY
runtime:
  codex_binary: codex
  branch_width: 1
  max_iterations: null
paths:
  agentbench_root: ${AB_ROOT}
  runs_root: ./runs
curriculum:
  order: lowest_rank_first
  development_seeds: [1, 2]
measurement:
  epsilon: 0.01
""",
        encoding="utf-8",
    )
    config = ExperimentConfig.load(
        path,
        env={"AB_ROOT": "/bench", "ABHL_API_KEY": "secret-value"},
    )
    assert config.paths.agentbench_root == Path("/bench")
    assert config.runtime.max_iterations is None
    assert config.secret_environment() == {"ABHL_API_KEY": "secret-value"}
    assert "secret-value" not in repr(config.frozen_dict())


def test_config_rejects_non_from_scratch_origin(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("schema_version: '1.0'\norigin: v239\n", encoding="utf-8")
    with pytest.raises(ValueError, match="from_scratch"):
        ExperimentConfig.load(path, env={})
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_config.py -q`

Expected: FAIL because `agentbench_hl.config` does not exist.

- [ ] **Step 3: Implement the package and immutable configuration types**

Implement frozen dataclasses with these exact public fields:

```python
@dataclass(frozen=True)
class ProviderConfig:
    model: str
    reasoning_effort: str
    base_url: str
    api_key_env: str
    disable_response_storage: bool


@dataclass(frozen=True)
class RuntimeConfig:
    codex_binary: str
    branch_width: int
    max_iterations: int | None
    network_access: Literal["disabled"]


@dataclass(frozen=True)
class PathConfig:
    agentbench_root: Path
    runs_root: Path


@dataclass(frozen=True)
class CurriculumConfig:
    order: Literal["lowest_rank_first"]
    development_seeds: tuple[int, ...]


@dataclass(frozen=True)
class EvaluatorConfig:
    certification_seeds: tuple[int, ...]
    roles: tuple[Literal["P0", "P1"], ...]


@dataclass(frozen=True)
class MeasurementConfig:
    epsilon: float


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    game: Literal["antwar2"]
    origin: Literal["from_scratch"]
    provider: ProviderConfig
    runtime: RuntimeConfig
    paths: PathConfig
    curriculum: CurriculumConfig
    measurement: MeasurementConfig
```

Use `${NAME}` substitution only for complete path scalar values. Reject an
unset API key when `secret_environment()` is called, not during read-only
configuration inspection. `frozen_dict()` stores the API key environment
variable name and never its value.

Populate `antwar2-goal-k1.yaml` with `origin: from_scratch`,
`branch_width: 1`, `max_iterations: null`, `epsilon: 0.01`, development seeds
`[1, 2]`,
`disable_response_storage: true`, `network_access: disabled`, and the
user-specified Responses base URL. Put certification seeds `[11, 12, 13]`
and roles `[P0, P1]` only in `evaluator-config/antwar2-certification.yaml`.
Neither key value nor certification cases may appear in Goal-visible context.

- [ ] **Step 4: Write the GamePack source audit test**

```python
def test_goal_charter_contains_learning_contract_without_reference_versions() -> None:
    root = Path("gamepacks/antwar2")
    combined = "\n".join(path.read_text(encoding="utf-8") for path in root.iterdir() if path.is_file())
    assert "from scratch" in combined.lower()
    assert "回放" in combined
    assert "失败" in combined
    assert "v239" not in combined
    assert "rank01__" not in combined


def test_certification_matrix_is_outside_goal_gamepack() -> None:
    assert not (Path("gamepacks/antwar2") / "certification_matrix.yaml").exists()
    assert Path("evaluator-config/antwar2-certification.yaml").is_file()
```

- [ ] **Step 5: Run tests and lint**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_config.py -q && python -m ruff check src tests`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add AgentBench-HL/pyproject.toml AgentBench-HL/README.md AgentBench-HL/src AgentBench-HL/gamepacks AgentBench-HL/configs AgentBench-HL/tests/unit/test_config.py
git commit -m "feat: define AntWar2 Goal experiment contract"
```

## Task 2: Finalized Event Store and Resource Accounting

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/domain/events.py`
- Create: `AgentBench-HL/src/agentbench_hl/domain/models.py`
- Create: `AgentBench-HL/src/agentbench_hl/ports/event_store.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/filesystem/event_store.py`
- Test: `AgentBench-HL/tests/unit/test_events.py`
- Test: `AgentBench-HL/tests/contract/test_filesystem_event_store.py`

**Interfaces:**
- Produces: `FinalizedEvent.create(event_type, payload, idempotency_key, occurred_at=None)`, `JsonlEventStore.append(event) -> bool`, `JsonlEventStore.read_all() -> tuple[FinalizedEvent, ...]`, `Usage.from_mapping(mapping) -> Usage`.

- [ ] **Step 1: Write failing event tests**

```python
from datetime import UTC, datetime
from pathlib import Path

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.models import Usage


def test_event_store_is_idempotent_and_append_only(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    event = FinalizedEvent.create(
        "CandidateSealed",
        {"version_id": "v000", "content_hash": "a" * 64},
        idempotency_key="seal:v000",
        occurred_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    assert store.append(event) is True
    assert store.append(event) is False
    assert store.read_all() == (event,)


def test_missing_usage_stays_unknown() -> None:
    usage = Usage.from_mapping({"input_tokens": 10})
    assert usage.input_tokens == 10
    assert usage.output_tokens is None
    assert usage.total_tokens is None
```

- [ ] **Step 2: Verify failures**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_events.py tests/contract/test_filesystem_event_store.py -q`

Expected: FAIL because event modules do not exist.

- [ ] **Step 3: Implement canonical finalized events**

Use this envelope:

```python
@dataclass(frozen=True)
class FinalizedEvent:
    schema_version: Literal["1.0"]
    event_id: str
    event_type: str
    idempotency_key: str
    occurred_at: str
    payload: Mapping[str, JSONValue]
```

Create `event_id` as SHA-256 of canonical JSON containing event type,
idempotency key, timestamp, and payload. Reject credential patterns matching
`\bsk-[A-Za-z0-9_-]{8,}` at event construction and append time. Append one
line only after `flush()` and `os.fsync()` succeed. An existing matching
idempotency key returns `False`; a conflicting event raises `ValueError`.

Define `Usage` with nullable `input_tokens`, `cached_input_tokens`,
`output_tokens`, `reasoning_tokens`, and `total_tokens`, plus `wall_time_s`.

- [ ] **Step 4: Add corruption and secret contract tests**

```python
def test_store_rejects_truncated_json(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        JsonlEventStore(path).read_all()


def test_event_rejects_api_key() -> None:
    with pytest.raises(ValueError, match="credential"):
        FinalizedEvent.create("GoalStarted", {"value": "sk-" + "abcdefghijk"}, "goal:1")
```

- [ ] **Step 5: Run tests and commit**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_events.py tests/contract/test_filesystem_event_store.py -q`

Expected: PASS.

```bash
git add AgentBench-HL/src/agentbench_hl/domain AgentBench-HL/src/agentbench_hl/ports AgentBench-HL/src/agentbench_hl/adapters/filesystem AgentBench-HL/tests
git commit -m "feat: add finalized scientific event store"
```

## Task 3: Immutable Candidate Snapshots and Lineage

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/domain/lineage.py`
- Create: `AgentBench-HL/src/agentbench_hl/ports/artifact_store.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/filesystem/artifact_store.py`
- Create: `AgentBench-HL/src/agentbench_hl/application/candidate_service.py`
- Test: `AgentBench-HL/tests/unit/test_lineage.py`
- Test: `AgentBench-HL/tests/integration/test_candidate_service.py`

**Interfaces:**
- Consumes: `JsonlEventStore`.
- Produces: `CandidateService.create(parent_id) -> CandidateWorkspace`, `CandidateService.seal(workspace_id) -> CandidateVersion`, `LineageState.replay(events) -> LineageState`, `LineageState.choose_frontier(version_id, rationale)`, `LineageState.promote(version_id)`.

- [ ] **Step 1: Write failing lineage tests**

```python
def test_weak_frontier_does_not_replace_champion() -> None:
    state = LineageState.empty().add_version(version("v000", parent=None))
    state = state.promote("v000")
    state = state.add_version(version("v001", parent="v000"))
    state = state.choose_frontier("v001", rationale="diagnostic improved")
    assert state.champion_id == "v000"
    assert state.frontier_id == "v001"
    assert state.archive_ids == frozenset({"v000", "v001"})


def test_soft_exploration_debt_never_auto_rolls_back() -> None:
    state = lineage_with_unpromoted_depth(4)
    assert state.frontier_id == "v004"
    assert state.requires_continuation_rationale is True
```

- [ ] **Step 2: Verify failures**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_lineage.py tests/integration/test_candidate_service.py -q`

Expected: FAIL because lineage modules do not exist.

- [ ] **Step 3: Implement content-addressed candidates**

Candidate sealing must:

1. reject symlinks, paths outside the workspace, credential patterns, and
   missing `ai.py`, `main.py`, `common.py`, `protocol.py`, or `SDK`;
2. compute a tree SHA-256 over relative paths and bytes while excluding
   `__pycache__`, `.pytest_cache`, and `.agentbench`;
3. copy into `runs/{run_id}/candidates/objects/{content_hash}/` using a temporary sibling
   directory and atomic rename;
4. assign the next integer `vNNN` from finalized `CandidateSealed` events;
5. persist parent ID, workspace ID, content hash, source file hashes, and
   creation reason.

Define lineage without score assumptions:

```python
@dataclass(frozen=True)
class LineageState:
    versions: Mapping[str, CandidateVersion]
    champion_id: str | None
    frontier_id: str | None
    exploration_debt: int
    soft_non_improving_depth: int = 3
```

Exact repeated content on the same parent is archived as a duplicate and may
not become Frontier.

- [ ] **Step 4: Add resume integration test**

```python
def test_replay_restores_same_lineage(tmp_path: Path) -> None:
    service = build_candidate_service(tmp_path)
    v000 = seal_minimal_candidate(service)
    service.promote(v000.version_id)
    expected = service.state
    resumed = build_candidate_service(tmp_path).state
    assert resumed == expected
```

- [ ] **Step 5: Run tests and commit**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_lineage.py tests/integration/test_candidate_service.py -q`

Expected: PASS.

```bash
git add AgentBench-HL/src/agentbench_hl/domain/lineage.py AgentBench-HL/src/agentbench_hl/ports/artifact_store.py AgentBench-HL/src/agentbench_hl/adapters/filesystem/artifact_store.py AgentBench-HL/src/agentbench_hl/application/candidate_service.py AgentBench-HL/tests
git commit -m "feat: add immutable candidate lineage"
```

## Task 4: Frozen AntWar2 Runtime, Pool Audit, and Official Arena

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/ports/arena.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/antwar2/runtime.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/antwar2/arena.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/antwar2/smoke.py`
- Test: `AgentBench-HL/tests/unit/test_antwar2_runtime.py`
- Test: `AgentBench-HL/tests/contract/test_antwar2_arena.py`

**Interfaces:**
- Produces: `AntWarLayout.from_root(agentbench_root, build_root)`, `build_backend(layout) -> FrozenBackend`, `audit_human_pool(layout) -> tuple[Opponent, ...]`, `AntWarArena.run_case(case, candidate_root) -> MatchResult`, `verify_smoke(candidate_root) -> ValidationResult`.

- [ ] **Step 1: Write failing runtime and match-record tests**

```python
def test_layout_resolves_only_frozen_public_and_hidden_roots(tmp_path: Path) -> None:
    layout = AntWarLayout.from_root(tmp_path / "AgentBench", tmp_path / "build")
    assert layout.backend_archive.name == "gamecode_logic__141.zip"
    assert layout.human_manifest.name == "MANIFEST.tsv"
    assert "30_antwar2_ladder" in layout.human_extracted_root.as_posix()


def test_terminal_replay_becomes_role_aware_match_result(tmp_path: Path) -> None:
    replay = write_terminal_replay(tmp_path, winner=1, bases=[0, 7])
    result = match_result_from_replay(
        replay, candidate_id="v000", opponent_id="rank20", role="P1", seed=1
    )
    assert result.result == "win"
    assert result.score_margin == 7.0
    assert result.terminal_base_hp == (0.0, 7.0)
```

- [ ] **Step 2: Verify failures**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_antwar2_runtime.py tests/contract/test_antwar2_arena.py -q`

Expected: FAIL because the AntWar2 adapter does not exist.

- [ ] **Step 3: Adapt the official runtime boundary**

Use the frozen resources at:

```text
backend_sources/corpus/30_antwar2/archives/gamecode_logic__141.zip
top_algorithms/corpus/30_antwar2_ladder/MANIFEST.tsv
top_algorithms/corpus/30_antwar2_ladder/extracted/
```

Implement safe ZIP extraction, content-addressed backend builds, compiler and
binary hashes, no source modifications on macOS/Linux, and the transport-only
binary-mode patch on Windows. Human pool audit reads manifest metadata and
entry-point presence but never reads `ai.py` content. The audit stores
`runnable`, `entry_command`, archive hash, and an exclusion diagnostic for
each rank.

Before candidate materialization, hash every available `SDK` tree and require
the selected public SDK tree to match the frozen GamePack manifest. Copy only
that SDK directory plus `gamepacks/antwar2/candidate_support`; do not copy a
human `ai.py`. The Goal creates the policy `ai.py` inside the candidate
workspace.

Adapt the length-prefixed match loop as a focused adapter. Use a whitelist
environment containing PATH, locale, TMPDIR, Python unbuffered mode, and no API
credential variables. A completed match requires a valid terminal winner,
base HP pair, both process statuses, replay, public trace, and event log.

- [ ] **Step 4: Add full-strength smoke fixtures**

```python
def test_smoke_rejects_illegal_operation(candidate_package: Path) -> None:
    write_illegal_policy(candidate_package)
    result = verify_smoke(candidate_package)
    assert result.status == "failed"
    assert "illegal operation" in result.error


def test_match_failure_is_incomplete_not_loss(fake_arena: AntWarArena) -> None:
    result = fake_arena.run_case(case(timeout=True), candidate_root=Path("candidate"))
    assert result.status == "incomplete"
    assert result.result is None
```

- [ ] **Step 5: Run offline and official build tests**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_antwar2_runtime.py tests/contract/test_antwar2_arena.py -q`

Expected: PASS, including a content-hash check of the existing official build or a fresh official build.

- [ ] **Step 6: Commit**

```bash
git add AgentBench-HL/src/agentbench_hl/ports/arena.py AgentBench-HL/src/agentbench_hl/adapters/antwar2 AgentBench-HL/tests
git commit -m "feat: add frozen AntWar2 arena"
```

## Task 5: Grounded Replay Decoder and Golden Narratives

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/adapters/antwar2/replay.py`
- Create: `AgentBench-HL/src/agentbench_hl/application/replay_service.py`
- Create: `AgentBench-HL/tests/golden/antwar2_replays/fixture.json`
- Create: `AgentBench-HL/tests/golden/antwar2_replays/expected_timeline.jsonl`
- Create: `AgentBench-HL/tests/golden/antwar2_replays/expected_narrative.md`
- Test: `AgentBench-HL/tests/golden/test_antwar2_replay.py`

**Interfaces:**
- Consumes: official replay JSON and optional public trace.
- Produces: `decode_replay(replay, trace=None) -> ReplayReport`, `ReplayService.materialize(match_id) -> ReplayArtifacts`, and `ReplayReport.window(start_state_id, end_state_id)`.

- [ ] **Step 1: Write the failing golden test**

```python
def test_replay_decodes_numbers_into_grounded_chinese() -> None:
    report = decode_replay(load_fixture("fixture.json"))
    assert report.frames[0].base_hp == (50, 50)
    assert report.timeline[0].state_id == "fixture:r0001:p0"
    assert "在 (8,11) 建造基础塔" in report.timeline[0].text
    assert "state_id=fixture:r0001:p0" in report.narrative
    assert report.metrics["first_weapon_round"]["P0"] == 28
```

- [ ] **Step 2: Verify failure**

Run: `cd AgentBench-HL && python -m pytest tests/golden/test_antwar2_replay.py -q`

Expected: FAIL because replay modules do not exist.

- [ ] **Step 3: Implement five-layer replay semantics**

Define immutable `CanonicalFrame`, `AtomicEvent`, `CriticalWindow`, and
`ReplayReport`. Reconstruct tower deltas; normalize official `bases`, replay
`camps`, coins, cooldowns, effects, ants, and towers into canonical fields.
Translate operation types 11, 12, 13, 21, 22, 23, 24, 31, and 32 plus HOLD.
Every Chinese sentence includes a state ID and factual values. Derived facts
cover first action, first weapon, downgrade-to-weapon timing, weapon coverage,
base breaches, idle resources, and build/downgrade churn. Derived labels are
not legal-action categories.

Materialize exactly:

```python
@dataclass(frozen=True)
class ReplayArtifacts:
    summary_json: Path
    timeline_jsonl: Path
    critical_windows_json: Path
    narrative_md: Path
```

- [ ] **Step 4: Add grounding validation**

```python
def test_every_narrative_claim_has_evidence_reference() -> None:
    report = decode_replay(load_fixture("fixture.json"))
    for paragraph in report.strategic_claims:
        assert paragraph.evidence_state_ids
        assert all(state_id in report.frame_by_id for state_id in paragraph.evidence_state_ids)
```

- [ ] **Step 5: Run tests and commit**

Run: `cd AgentBench-HL && python -m pytest tests/golden/test_antwar2_replay.py -q`

Expected: PASS with byte-identical golden artifacts.

```bash
git add AgentBench-HL/src/agentbench_hl/adapters/antwar2/replay.py AgentBench-HL/src/agentbench_hl/application/replay_service.py AgentBench-HL/tests/golden
git commit -m "feat: decode AntWar2 replays into grounded behavior"
```

## Task 6: Immutable Experience Ledger and Materialized Skill

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/domain/experience.py`
- Create: `AgentBench-HL/src/agentbench_hl/application/research_service.py`
- Create: `AgentBench-HL/src/agentbench_hl/reporting/research_report.py`
- Test: `AgentBench-HL/tests/unit/test_experience.py`
- Test: `AgentBench-HL/tests/integration/test_research_service.py`

**Interfaces:**
- Consumes: candidate, match, replay, hypothesis, and selection events.
- Produces: `ExperienceRecord`, `ResearchService.record(record) -> bool`, `ResearchService.materialize() -> ResearchArtifacts`, and `ResearchService.context(target, role, max_records) -> ResearchContext`.

- [ ] **Step 1: Write failing positive/negative retention tests**

```python
def test_bad_experience_survives_skill_materialization(tmp_path: Path) -> None:
    service = build_research_service(tmp_path)
    service.record(experience("exp-good", verdict="supported", outcome="win"))
    service.record(experience("exp-bad", verdict="refuted", outcome="loss"))
    artifacts = service.materialize()
    playbook = artifacts.playbook.read_text(encoding="utf-8")
    failures = artifacts.failed_hypotheses.read_text(encoding="utf-8")
    assert "exp-good" in playbook
    assert "exp-bad" in failures


def test_supersession_never_deletes_source_record(tmp_path: Path) -> None:
    service = build_research_service(tmp_path)
    service.record(experience("exp-1", verdict="mixed"))
    service.record(experience("exp-2", verdict="supported", supersedes=("exp-1",)))
    assert [record.experience_id for record in service.read_all()] == ["exp-1", "exp-2"]
```

- [ ] **Step 2: Verify failure**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_experience.py tests/integration/test_research_service.py -q`

Expected: FAIL because experience modules do not exist.

- [ ] **Step 3: Implement typed Experience and deterministic documents**

Define verdicts exactly as `supported`, `refuted`, `mixed`, `inconclusive`,
`integration_failure`, and `not_activated`. Require condition, mechanism,
proposed change, expected observation, parent/candidate IDs, selection,
match IDs, evidence windows, and measured outcome. `integration_failure`
records preserve the hypothesis without treating it as falsified.

Materialize `PLAYBOOK.md`, `FAILED_HYPOTHESES.md`, `OPEN_QUESTIONS.md`,
`ROLE_P0.md`, `ROLE_P1.md`, `OPPONENT_NOTES.md`, and one immutable iteration
report. Sort by scientific iteration and experience ID. Context retrieval
includes relevant failures as well as successes.

- [ ] **Step 4: Add secret and evidence validation tests**

```python
def test_experience_rejects_missing_replay_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        experience("exp-1", verdict="supported", evidence_windows=())


def test_experience_rejects_secret() -> None:
    with pytest.raises(ValueError, match="credential"):
        experience("exp-1", mechanism="sk-" + "abcdefghijk")
```

- [ ] **Step 5: Run tests and commit**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_experience.py tests/integration/test_research_service.py -q`

Expected: PASS.

```bash
git add AgentBench-HL/src/agentbench_hl/domain/experience.py AgentBench-HL/src/agentbench_hl/application/research_service.py AgentBench-HL/src/agentbench_hl/reporting/research_report.py AgentBench-HL/tests
git commit -m "feat: retain grounded positive and negative experience"
```

## Task 7: Behavioral IG, Elo, Performance, Resource Curves

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/domain/metrics.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/antwar2/policy_probe.py`
- Create: `AgentBench-HL/src/agentbench_hl/application/metrics_service.py`
- Create: `AgentBench-HL/src/agentbench_hl/reporting/curves.py`
- Test: `AgentBench-HL/tests/unit/test_metrics.py`
- Test: `AgentBench-HL/tests/integration/test_metrics_service.py`

**Interfaces:**
- Produces: `epsilon_regularized_kl(old_action, new_action, legal_actions, epsilon)`, `fit_anchored_elo(results, human_ratings)`, `MetricsService.finalize_iteration(version_id) -> IterationMetrics`, and `build_curves(rows, output_dir) -> CurveArtifacts`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_deterministic_action_change_has_finite_kl() -> None:
    value = epsilon_regularized_kl("HOLD", "BUILD:1,2", ("HOLD", "BUILD:1,2"), 0.01)
    assert value > 0
    assert math.isfinite(value)


def test_same_action_has_zero_kl() -> None:
    assert epsilon_regularized_kl("HOLD", "HOLD", ("HOLD", "BUILD:1,2"), 0.01) == pytest.approx(0.0)


def test_candidate_elo_is_version_local() -> None:
    first = fit_anchored_elo(results_for("v001"), {"rank20": 1200.0})
    second = fit_anchored_elo(results_for("v002"), {"rank20": 1200.0})
    assert first == second
```

- [ ] **Step 2: Verify failure**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_metrics.py tests/integration/test_metrics_service.py -q`

Expected: FAIL because metric modules do not exist.

- [ ] **Step 3: Implement exact metric semantics**

Enumerate legal protocol atoms from the public SDK predicate for each frozen
state. Apply the same epsilon measurement channel to both deterministic
actions. Store per-state KL trace, mean nats per decision, action disagreement,
and occupancy histogram shift separately.

Fit candidate Elo independently per version against frozen human ratings with
0.5 pseudo-wins and 0.5 pseudo-losses. Store P0, P1, and combined estimates.
Compute win rate as `(W + 0.5 * D) / completed`, mean terminal base-HP margin,
and missing aggregate when any required evaluation case is incomplete.

`IterationMetrics` contains integer iteration, candidate and Champion IDs,
IG, occupancy shift, Elo, win rate, margin, learning/evaluation/total usage,
and learning/evaluation/total wall time.

- [ ] **Step 4: Add the four-panel report test**

```python
def test_curve_rows_use_integer_iteration_and_four_primary_panels(tmp_path: Path) -> None:
    artifacts = build_curves(sample_metric_rows(), tmp_path)
    assert artifacts.primary_png.is_file()
    assert artifacts.csv.is_file()
    header = artifacts.csv.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("research_iteration,")
    assert artifacts.panel_names == ("Behavioral IG", "Fixed-pool Elo", "Win rate", "Score margin")
```

- [ ] **Step 5: Run tests and commit**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_metrics.py tests/integration/test_metrics_service.py -q`

Expected: PASS.

```bash
git add AgentBench-HL/src/agentbench_hl/domain/metrics.py AgentBench-HL/src/agentbench_hl/adapters/antwar2/policy_probe.py AgentBench-HL/src/agentbench_hl/application/metrics_service.py AgentBench-HL/src/agentbench_hl/reporting/curves.py AgentBench-HL/tests
git commit -m "feat: add AntWar2 scientific metrics"
```

## Task 8: Bottom-Up Curriculum and Evaluation Service

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/application/curriculum_service.py`
- Create: `AgentBench-HL/src/agentbench_hl/application/evaluation_service.py`
- Test: `AgentBench-HL/tests/unit/test_curriculum.py`
- Test: `AgentBench-HL/tests/integration/test_evaluation_service.py`

**Interfaces:**
- Consumes: audited `Opponent` values, development seeds, match events, and lineage.
- Produces: `CurriculumService.status() -> CurriculumStatus`, `CurriculumService.default_target() -> Opponent`, `EvaluationService.calibrate_human_pool(matrix) -> HumanCalibration`, and `EvaluationService.evaluate_version(version_id, matrix) -> EvaluationResult`.

- [ ] **Step 1: Write failing curriculum tests**

```python
def test_curriculum_selects_weakest_runnable_unsolved_rank() -> None:
    service = curriculum(ranks=(20, 19, 18), solved=(20,), unrunnable=(19,))
    assert service.default_target().opponent_id == "rank18"


def test_target_is_not_solved_when_one_role_is_incomplete() -> None:
    service = curriculum_with_results(rank=20, p0="win", p1="incomplete")
    assert service.status().by_opponent["rank20"].state == "incomplete"


def test_locked_regression_loss_preserves_old_champion() -> None:
    result = evaluate_candidate(target_win=True, locked_regression_win=False)
    assert result.promotable is False
    assert result.frontier_eligible is True


def test_human_calibration_is_hash_cached_and_order_independent() -> None:
    first = calibrate_humans(case_order=("rank20-rank19", "rank19-rank18"))
    second = calibrate_humans(case_order=("rank19-rank18", "rank20-rank19"))
    assert first.ratings == second.ratings
    assert first.matrix_hash == second.matrix_hash
```

- [ ] **Step 2: Verify failure**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_curriculum.py tests/integration/test_evaluation_service.py -q`

Expected: FAIL because application services do not exist.

- [ ] **Step 3: Implement target and regression semantics**

Evaluate `v000` over the complete development matrix. Thereafter select the
largest numeric rank among runnable unsolved opponents. A target is solved
only if all required development seeds and both roles are valid wins. Solved
opponents enter a locked regression set. A regression prevents Champion
promotion but may leave the candidate eligible as an exploratory Frontier.

Cache match cases by `(backend_hash, candidate_hash, opponent_hash, role,
seed)` and reuse only complete, hash-matching artifacts. Incomplete cases are
retryable and excluded from win/loss aggregates.

Before candidate Elo is reported, run and cache the frozen human-vs-human
calibration matrix through the same official arena. Fit anchored human ratings
from the complete matrix, independent of match execution order. Curriculum
selection continues to use official rank, not estimated Elo.

- [ ] **Step 4: Run tests and commit**

Run: `cd AgentBench-HL && python -m pytest tests/unit/test_curriculum.py tests/integration/test_evaluation_service.py -q`

Expected: PASS.

```bash
git add AgentBench-HL/src/agentbench_hl/application/curriculum_service.py AgentBench-HL/src/agentbench_hl/application/evaluation_service.py AgentBench-HL/tests
git commit -m "feat: add weakest-unsolved AntWar2 curriculum"
```

## Task 9: Codex App Server Goal Runtime and Isolation

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/ports/agent_runtime.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/codex_goal/protocol.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/codex_goal/app_server.py`
- Create: `AgentBench-HL/src/agentbench_hl/adapters/codex_goal/event_mapper.py`
- Test: `AgentBench-HL/tests/contract/test_codex_goal_protocol.py`
- Test: `AgentBench-HL/tests/integration/test_codex_goal_runtime.py`

**Interfaces:**
- Produces: `AgentRuntime.start(run_context) -> AgentSession`, `AgentRuntime.resume(session_id)`, `AgentRuntime.run_until_checkpoint(session, checkpoint_predicate)`, `AgentRuntime.pause(session)`, and `CodexGoalRuntime` implementation.

- [ ] **Step 1: Write failing JSON-RPC contract tests**

```python
def test_goal_runtime_starts_isolated_thread(fake_app_server: FakeAppServer, tmp_path: Path) -> None:
    runtime = CodexGoalRuntime(fake_app_server.command, codex_home=tmp_path / "codex-home")
    session = runtime.start(run_context(tmp_path))
    methods = fake_app_server.received_methods
    assert methods[:3] == ["initialize", "thread/start", "thread/goal/set"]
    assert session.thread_id == "thread-1"
    assert session.goal_status == "active"


def test_runtime_maps_usage_without_inventing_missing_tokens(fake_app_server: FakeAppServer) -> None:
    event = map_app_server_event({"type": "usage", "input_tokens": 10})
    assert event.payload["input_tokens"] == 10
    assert event.payload["output_tokens"] is None
```

- [ ] **Step 2: Verify failure**

Run: `cd AgentBench-HL && python -m pytest tests/contract/test_codex_goal_protocol.py tests/integration/test_codex_goal_runtime.py -q`

Expected: FAIL because the runtime adapter does not exist.

- [ ] **Step 3: Implement pinned stdio JSON-RPC client**

Launch `codex app-server --listen stdio://` with a dedicated `CODEX_HOME`, the
API key in environment, and a generated config pointing to the Responses base
URL. The run config file stores only the environment variable name. Use
newline-delimited JSON-RPC requests with monotonic request IDs, a reader
thread, bounded waits, and stderr capture with credential redaction.

Start a thread with explicit base instructions, developer instructions, cwd,
runtime workspace roots, model/provider, sandbox, permissions, ephemeral
false, and memory disabled. Set the Goal through `thread/goal/set`; never send
`/goal` as user text. Map thread, turn, tool, usage, compaction, Goal status,
and error notifications to finalized events. Persist only the thread ID and
resume through `thread/resume`.

The Goal workspace roots include the GamePack, run-local research artifacts,
and active candidate workspace. They exclude `evaluator-config`, human package
roots, certification case material, and every reference-policy path.

- [ ] **Step 4: Add isolation tests**

```python
def test_runtime_roots_exclude_human_and_reference_policy_paths(run_context: RunContext) -> None:
    roots = run_context.runtime_workspace_roots
    assert run_context.candidate_root in roots
    assert run_context.gamepack_root in roots
    assert run_context.human_pool_root not in roots
    assert all("handoff_next_agent" not in str(path) for path in roots)


def test_generated_codex_config_contains_no_literal_key(tmp_path: Path) -> None:
    path = write_codex_config(tmp_path, provider_config())
    assert "sk-" not in path.read_text(encoding="utf-8")
```

- [ ] **Step 5: Run tests and commit**

Run: `cd AgentBench-HL && python -m pytest tests/contract/test_codex_goal_protocol.py tests/integration/test_codex_goal_runtime.py -q`

Expected: PASS using the fake App Server.

```bash
git add AgentBench-HL/src/agentbench_hl/ports/agent_runtime.py AgentBench-HL/src/agentbench_hl/adapters/codex_goal AgentBench-HL/tests
git commit -m "feat: add isolated Codex Goal runtime"
```

## Task 10: Run Service, JSON CLI, Resume, and Offline End-to-End Test

**Files:**
- Create: `AgentBench-HL/src/agentbench_hl/application/run_service.py`
- Create: `AgentBench-HL/src/agentbench_hl/cli/main.py`
- Create: `AgentBench-HL/src/agentbench_hl/cli/__init__.py`
- Test: `AgentBench-HL/tests/e2e/test_offline_goal_run.py`
- Test: `AgentBench-HL/tests/contract/test_cli.py`

**Interfaces:**
- Consumes: all services and ports from Tasks 1–9.
- Produces: the `abhl` command and `RunService.initialize`, `RunService.resume`, `RunService.status`, and `RunService.checkpoint`.

- [ ] **Step 1: Write failing offline end-to-end test**

```python
def test_fake_goal_creates_v000_match_replay_experience_and_metrics(tmp_path: Path) -> None:
    run = build_offline_run(tmp_path, fake_goal=GoalThatWritesBaseline())
    result = run.execute_until("first_match_finalized")
    assert result.lineage.champion_id == "v000"
    assert result.events.count("CandidateSealed") == 1
    assert result.events.count("MatchFinalized") == 1
    assert (result.root / "replays" / result.match_id / "narrative.md").is_file()
    assert (result.root / "research" / "PLAYBOOK.md").is_file()
    assert result.metrics[0].research_iteration == 0


def test_resume_reuses_finalized_match(tmp_path: Path) -> None:
    run = build_offline_run(tmp_path, interrupt_after="MatchFinalized")
    run.execute()
    resumed = RunService.resume(run.root)
    resumed.execute_until("checkpoint")
    assert resumed.fake_arena.call_count == 0
    assert resumed.events.count("MatchFinalized") == 1
```

- [ ] **Step 2: Verify failure**

Run: `cd AgentBench-HL && python -m pytest tests/e2e/test_offline_goal_run.py tests/contract/test_cli.py -q`

Expected: FAIL because `RunService` and the CLI do not exist.

- [ ] **Step 3: Implement run lifecycle and JSON commands**

`RunService.initialize` freezes all config and resource hashes, audits the
pool, creates run directories, starts the runtime, and appends
`RunInitialized` and `GoalStarted`. It exposes CLI operations from the design
spec and returns a JSON object with `status`, `event_ids`, `artifact_paths`,
and `error`. Command failures write no finalized scientific event unless a
durable failure artifact is explicitly part of the domain protocol.

`RunService.resume` replays events, verifies hashes, reconnects App Server or
resumes the persisted thread, and continues from the first unfinished state.
It never repeats complete paid model turns or official matches with matching
cache identities.

- [ ] **Step 4: Add CLI secret redaction test**

```python
def test_cli_error_redacts_key(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ABHL_API_KEY", "sk-" + "abcdefghijk")
    result = cli_runner.invoke(["run", "status", "--config", "missing.yaml"])
    assert "sk-" not in result.stdout
    assert "sk-" not in result.stderr
```

- [ ] **Step 5: Run the complete offline suite**

Run: `cd AgentBench-HL && python -m pytest -q && python -m ruff check src tests`

Expected: PASS with no paid model call.

- [ ] **Step 6: Commit**

```bash
git add AgentBench-HL/src/agentbench_hl/application/run_service.py AgentBench-HL/src/agentbench_hl/cli AgentBench-HL/tests AgentBench-HL/pyproject.toml
git commit -m "feat: complete offline Goal-led research loop"
```

## Task 11: Live Compatibility Probe and Paid Goal Smoke

**Files:**
- Create: `AgentBench-HL/docs/reproducibility.md`
- Create: `AgentBench-HL/tests/e2e/test_live_goal_smoke.py`
- Modify: `AgentBench-HL/README.md`

**Interfaces:**
- Consumes: the complete offline system and environment variable `ABHL_API_KEY`.
- Produces: a run directory containing `v000`, one official match, semantic replay, Experience, iteration-zero metrics, token/time usage, and a resumable Goal checkpoint.

- [ ] **Step 1: Add an opt-in live smoke test**

```python
@pytest.mark.live
def test_live_goal_reaches_first_official_match(live_config: Path) -> None:
    result = RunService.from_config(live_config).execute_until("first_match_finalized")
    assert result.lineage.versions["v000"].origin == "from_scratch"
    assert result.matches[0].status == "complete"
    assert result.usage.learning.wall_time_s > 0
    assert result.checkpoint.thread_id
```

The marker skips unless `ABHL_LIVE=1` and `ABHL_API_KEY` are present.

- [ ] **Step 2: Run no-token provider preflight**

Run: `cd AgentBench-HL && codex --version && codex features list && codex app-server generate-json-schema --experimental --out-dir /tmp/abhl-codex-schema`

Expected: Codex is available, `goals` is enabled, and App Server schemas are generated. Store the binary version and schema tree hash in the run manifest, not the temporary schema path.

- [ ] **Step 3: Run the complete offline gate**

Run: `cd AgentBench-HL && python -m pytest -m 'not live' -q && python -m ruff check src tests`

Expected: PASS.

- [ ] **Step 4: Run exactly one paid Goal smoke**

Run: `cd AgentBench-HL && ABHL_LIVE=1 python -m pytest tests/e2e/test_live_goal_smoke.py -m live -vv -s`

Expected: PASS with one new run, a from-scratch `v000`, one complete official match against the weakest runnable rank selected by the pool audit, semantic replay artifacts, an Experience record, iteration-zero metrics, and a persisted resumable Goal thread.

- [ ] **Step 5: Audit artifacts before long-run continuation**

Run: `cd AgentBench-HL && python -m agentbench_hl.cli.main run audit --latest`

Expected JSON fields:

```json
{
  "status": "complete",
  "credential_leaks": 0,
  "reference_policy_leaks": 0,
  "candidate_origin": "from_scratch",
  "matches_complete": 1,
  "semantic_replays": 1,
  "experience_records": 1,
  "resumable": true
}
```

- [ ] **Step 6: Commit documentation and live-smoke contract**

```bash
git add AgentBench-HL/docs/reproducibility.md AgentBench-HL/tests/e2e/test_live_goal_smoke.py AgentBench-HL/README.md
git commit -m "test: verify from-scratch Codex Goal smoke"
```

## Task 12: Start the Unbounded k=1 Research Run

**Files:**
- Create at runtime: `AgentBench-HL/runs/{run_id}/run-config.json`
- Create at runtime: `AgentBench-HL/runs/{run_id}/events.jsonl`
- Create at runtime: `AgentBench-HL/runs/{run_id}/checkpoint.json`

**Interfaces:**
- Consumes: audited smoke run and `configs/experiments/antwar2-goal-k1.yaml`.
- Produces: an active Goal-led run whose only successful stop is complete certification.

- [ ] **Step 1: Start a fresh scientific run, not the smoke run**

Run: `cd AgentBench-HL && abhl run init --config configs/experiments/antwar2-goal-k1.yaml --run-id antwar2-goal-k1-20260804`

Expected: a new isolated run with no imported candidate, replay, Skill, or Codex memory.

- [ ] **Step 2: Confirm curriculum and stop policy**

Run: `cd AgentBench-HL && abhl run status --run-id antwar2-goal-k1-20260804`

Expected: `max_iterations=null`, `hard_auto_rollback=false`, `branch_width=1`, and the weakest runnable unsolved rank as target after v000 development evaluation.

- [ ] **Step 3: Continue by resumable checkpoints**

Run: `cd AgentBench-HL && abhl run resume --run-id antwar2-goal-k1-20260804`

Expected: the persistent Goal continues research, records every candidate and Experience, and pauses only for API exhaustion, process interruption, or a true external integrity failure. Certification success marks the run complete.
