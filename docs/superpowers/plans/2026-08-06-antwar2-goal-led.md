# AntWar2 Goal-Led Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a reproducible AntWar2 experiment in which one isolated Codex App Server Goal chooses and performs its own iterative research, beginning against rank01.

**Architecture:** A run-local App Server thread owns all research decisions. It writes declarative match requests and candidate snapshots into its workspace; a thin bridge executes only the requested official matches and returns sanitized replay feedback to the same thread. The bridge records immutable versions, replays, usage, and event rows but never selects opponents, branches, promotions, or rollback.

**Tech Stack:** Python 3.11, Codex App Server JSON-RPC, existing AntWar2 native runner, JSONL event ledger, pytest.

## Global Constraints

- The first requested official opponent is `rank01` when it is runnable.
- One run owns one non-rotating Codex App Server thread and one Goal.
- Goal workspace has no network access and cannot read human source, certification files, credentials, or another run's Codex state.
- The bridge accepts only public match requests and returns only public replay/result artifacts.
- External metrics are recorded but do not decide the next experiment; final all-human certification is explicit.

---

### Task 1: Define the Goal-to-bridge request and feedback contract

**Files:**
- Create: `src/agentbench_hl/application/goal_led_protocol.py`
- Test: `tests/unit/test_goal_led_protocol.py`

**Interfaces:**
- Produces `MatchRequest.from_path(path: Path) -> MatchRequest`.
- Produces `MatchFeedback.to_json() -> dict[str, object]`.
- A request contains one or more candidate snapshot directories, one opponent id, one or more roles and seeds, and a scientific rationale.

- [ ] Write tests rejecting a request with an unknown role, duplicate candidate id, or no candidate snapshots.
- [ ] Implement immutable dataclasses and JSON schema validation without external dependencies.
- [ ] Run `pytest -q tests/unit/test_goal_led_protocol.py`.

### Task 2: Implement the thin request bridge

**Files:**
- Create: `src/agentbench_hl/application/goal_led_service.py`
- Modify: `src/agentbench_hl/application/live_run.py`
- Test: `tests/integration/test_goal_led_service.py`

**Interfaces:**
- Produces `GoalLedService.start() -> GoalLedSession` and `GoalLedService.advance() -> GoalLedOutcome`.
- `advance()` either waits for the active Goal, executes exactly the request emitted by the Goal, or supplies one feedback turn to the same thread.
- It emits `GoalLedStarted`, `GoalMatchRequested`, `GoalMatchCompleted`, `GoalFeedbackDelivered`, `GoalVersionSnapshot` and `GoalCertificationRequested` events.

- [ ] Write a fake-runtime integration test proving the bridge preserves one thread id across request and feedback.
- [ ] Implement snapshotting and single-request execution using the existing `AntWarArena`.
- [ ] Make the bridge reject an initial request that is not `rank01`, and never choose later opponents itself.
- [ ] Run the integration test and existing AntWar2 arena contract tests.

### Task 3: Add a dedicated CLI and AntWar2 goal charter

**Files:**
- Modify: `src/agentbench_hl/cli/main.py`
- Create: `configs/experiments/antwar2-goal-I.yaml`
- Modify: `gamepacks/antwar2/GOAL_CHARTER.md`
- Test: `tests/e2e/test_goal_led_run.py`

**Interfaces:**
- `abhl goal-led start --config ... --run-id ...`
- `abhl goal-led continue --config ... --run-id ...`
- `abhl goal-led certify --config ... --run-id ...`

- [ ] Write CLI parser tests for the three commands.
- [ ] Make the initial prompt require `v000` from rules, a first rank01 request, replay-grounded Skill updates, and no grid search.
- [ ] Make `continue` resume the existing thread without any synthetic Goal rotation.
- [ ] Run the new e2e test with a fake Codex runtime.

### Task 4: Verify a live rank01 pilot and preserve evidence

**Files:**
- Create: `runs/antwar2-goal-I/` at execution time only
- Verify: `runs/antwar2-goal-I/events.jsonl`, `research/`, `snapshots/`, `official-matches/`

- [ ] Run static and targeted tests.
- [ ] Start one isolated live Goal-led run with `gpt-5.6` through the configured mirror.
- [ ] Confirm the first official request is rank01 and that its replay feedback returns to the same thread.
- [ ] Report token use, thread id continuity, strategy snapshot, and first rank01 result without claiming certification.
