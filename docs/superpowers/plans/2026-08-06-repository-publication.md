# AgentBench-HL Repository Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish AgentBench-HL as a self-contained collaboration repository without committing credentials or local experimental artifacts.

**Architecture:** The repository contains reusable framework source, tests, experiment templates and an AntWar2 GamePack. Local credentials and execution state remain outside Git through a repository-level ignore policy. The README is the entry point; a GamePack authoring guide is the contract for adding games.

**Tech Stack:** Python 3.11, pytest, Ruff, Git, Codex App Server JSON-RPC.

## Global Constraints

- Never commit `.env`, provider credentials, API requests, App Server state, replays, candidate artifacts, or run directories.
- Preserve the active `/Users/qingle/Documents/SAST/runs/antwar2-goal-I` run untouched.
- Publish on `Aoraku/AgentBenchHL` branch `main` only.

---

### Task 1: Establish public repository metadata

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `docs/gamepack-authoring.md`

**Interfaces:**
- Consumes: `configs/experiments/antwar2-goal-I.yaml`, `gamepacks/antwar2/manifest.yaml`, `docs/reproducibility.md`
- Produces: documented setup and GamePack contribution contract.

- [ ] **Step 1: Rewrite README around the Goal-led entry point**

Document the separation between the Codex Goal and deterministic framework, local credential setup, a Scheme I command, repository layout, and the exact files required for another game.

- [ ] **Step 2: Write the GamePack contribution guide**

Specify rules, decision space, replay guide, manifest, candidate support and arena adapter responsibilities; explicitly exclude opponent code and secrets.

- [ ] **Step 3: Harden ignore rules**

Ignore credentials, run roots, generated reports/caches, App Server SQLite state and platform metadata while retaining `.env.example` and source templates.

- [ ] **Step 4: Verify ignore behaviour**

Run `git check-ignore -v .env` and `git check-ignore -v /Users/qingle/Documents/SAST/runs/antwar2-goal-I/events.jsonl` after repository initialization. Confirm that source and templates remain visible to Git.

### Task 2: Publish a clean independent repository

**Files:**
- Create: `.git/` (Git metadata only)

**Interfaces:**
- Consumes: remote `https://github.com/Aoraku/AgentBenchHL.git` branch `main`
- Produces: a clean `main` branch with framework content.

- [ ] **Step 1: Initialize the nested repository and fetch remote main**

Create independent Git metadata in `AgentBench-HL`, configure `origin`, and fetch `origin/main`.

- [ ] **Step 2: Merge the remote README into the repository history**

Use the remote `main` as the first parent, replace its placeholder README with the framework README, and preserve the remote default branch.

- [ ] **Step 3: Validate public contents**

Run the focused tests, Ruff, `git diff --check`, `git status --ignored`, and scan staged files for credential-like `ABHL_API_KEY=` values that are not empty examples.

- [ ] **Step 4: Commit and push `main`**

Commit only the independent repository contents and push `main` to `origin` without force-pushing.
