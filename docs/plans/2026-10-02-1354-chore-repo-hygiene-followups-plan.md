---
title: Repo Hygiene Follow-ups - Plan
type: chore
date: 2026-10-02
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Repo Hygiene Follow-ups - Plan

## Goal Capsule

- **Objective:** Someone reading this repo or its git history sees real project decisions and evidence, not hourly bot noise, stray third-party tooling, or a notes-encoder setting that contradicts the locked design.
- **Means:** collapse the no-op loop entries and change the loop protocol (KTD1); untrack the vendored design skill (KTD2); align the notes-encoder config to the MEMORY.md decision and lock it with a test (KTD3).
- **Authority:** MEMORY.md wins on design decisions, then AGENTS.md, then this plan.
- **Execution profile:** three independent units; mostly docs and config, with one contract test.
- **Stop conditions:** stop if any code, config, test, website, or build file references the impeccable skill files (historical prose under `docs/plans/` does not count), or if any code path reads `notes.encoder` in a way the change would break.
- **Tail ownership:** the calling pipeline owns review, PR, and CI.

---

## Product Contract

### Summary

Three cleanups left over from the 2026-10-02 documentation audit. The overnight GEM log keeps its 28 entries that record work or checks, including 14 "Idle-verify + …" entries with guardrail, API, or close-out evidence. Its 131 "Idle-verify: all green, no new instructions" entries become one summary line. The log's protocol then tells the loop not to append or commit idle passes. The vendored `impeccable` design skill, its broken symlink, and its lockfile leave version control. `configs/data.yaml` names the same frozen notes encoder as MEMORY.md and `NotesEncoder`'s default, and a test keeps them in sync.

### Problem Frame

145 of the repo's 283 commits are hourly idle-verify commits. Most of them appended an eight-line "all green, no new instructions" entry to `docs/plans/gem-overnight-log.md`. That log is cited from README, MEMORY.md, and the website as the GEM evidence trail, and the real evidence is now buried under about 1,200 lines of repeated status. A third-party frontend design skill sits in `agent/skills/impeccable/` (59 files, about 1.6 MB). Nothing in the project references it, its `.claude/skills` symlink points at a gitignored directory and is broken in every clone, and the user already has the same skill installed as a global plugin. The notes modality has three different encoder names: MEMORY.md says frozen BioClinical ModernBERT-base, `NotesEncoder` defaults to BioClinical-ModernBERT-base, and `configs/data.yaml` says `abhinand/MedEmbed-small-v0.1`.

### Requirements

**Overnight log**

- R1. Collapse every pure idle-verify entry in `docs/plans/gem-overnight-log.md` into one dated summary line that gives the count and date range and points to git history. Every substantive entry stays verbatim.
- R2. The log's protocol header says that a loop pass which changes nothing must not append to the log or commit. Heartbeats go to an untracked path, or nowhere.

**Vendored skill**

- R3. Remove `agent/skills/impeccable/`, `.claude/skills/impeccable`, and `skills-lock.json` from version control, and add ignore rules so a re-install does not re-add them.
- R4. AGENTS.md's do-not-commit list names vendored agent skills, so the rule is visible to the next agent.

**Notes encoder**

- R5. `configs/data.yaml → notes.encoder` names the same model as MEMORY.md's multimodal decision and `NotesEncoder`'s default. The usage example in the `NotesEncoder` docstring agrees.
- R6. A data-free test fails if the configured notes encoder and the `NotesEncoder` default ever diverge again.

### Scope Boundaries

- Stopping the running overnight loop process is outside this repo. The plan changes the protocol the loop reads each hour (R2). The user stops or re-prompts the loop session itself.
- Wiring `NotesEncoder` to read its model name from config is out of scope, because notes are disabled (`notes.enabled: false`).
- Git history is not rewritten. The 145 idle commits stay in history.

### Deferred to Follow-Up Work

- Wiring the notes config into the multimodal branch when `notes.enabled` turns on.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Collapse idle entries in place and gate the loop by protocol text, without moving the log.** The loop "reconstructs progress from this log", so a protocol line at the top of the log is the one lever the repo controls. Moving the file would break the website's `blob/main` link and four in-repo citations. An entry is "pure idle" exactly when its heading reads `Idle-verify: all green, no new instructions` (131 entries on 2026-10-02). Every `Idle-verify + …` entry stays verbatim. These are the re-audit close-out, the repo finalization, the live API spot check, and the 11 guardrail checks, which are the log's only record of the periodic data/PHI hard-rule checks.
- KTD2. **Untrack all three impeccable artifacts rather than repairing the symlink.** The skill is generic frontend tooling: the website and sequence viewer used it once for audits, but no code, test, or doc depends on it. The user has it installed globally as the `impeccable` plugin, so a repo copy is redundant. A repaired symlink would still point at the gitignored `.agents/`. Ignore rules are anchored to the repo root: `/agent/`, `/.claude/skills/`, and `/skills-lock.json`. Add `/.claude/scheduled_tasks.lock` as well, a session-local lock that appeared in this worktree.
- KTD3. **Align the config to BioClinical-ModernBERT-base, not MedEmbed-small or -large.** MEMORY.md's design spec ("BioClinical ModernBERT-base frozen") and the `NotesEncoder` default already agree, so only the config is the outlier. Base also fits the ~30M one-node thesis better than large. TextCode's use of `-large` is a separate tokenization arm and stays unchanged.

### Assumptions

- The user wants the loop to keep running but stop committing no-op passes, rather than stop the loop outright. R2 serves either case.
- Deleting the tracked impeccable files is acceptable in every clone, because the global plugin replaces them.

### Risks & Dependencies

- **Merge conflict on the log.** The loop appends to `docs/plans/gem-overnight-log.md` on `main` every hour, so this branch will conflict on that file at merge time. Mitigation: rebase onto `origin/main` immediately before opening the PR. Re-collapse any newly appended idle entries, keep any new substantive entry, and update the summary count.

---

## Implementation Units

### U1. Collapse idle-verify entries and gate the loop protocol

- **Goal:** the log reads as a record of real work, and the loop stops appending no-op passes.
- **Requirements:** R1, R2 (KTD1)
- **Dependencies:** none
- **Files:** `docs/plans/gem-overnight-log.md`
- **Approach:**
  1. Under the existing protocol paragraph at the top, add a rule saying that a pass which finds nothing to do must not append or commit. Heartbeats may go to `output/` (gitignored) or be skipped.
  2. Replace each pure idle-verify entry with nothing, and add one summary entry at the position of the first removed entry. The summary gives the count, the first and last timestamps, and the note that the full record is in `git log`.
  3. Keep every entry with substantive content verbatim, in order.
- **Patterns to follow:** the existing `### YYYY-MM-DD HH:MM — title` entry shape.
- **Test scenarios:** Test expectation: none -- docs-only change. Verified by structure checks under Verification.
- **Verification:**
  - No `Idle-verify: all green, no new instructions` heading remains.
  - Every other entry from before the edit is still present and byte-identical.
  - Exactly one summary entry exists, giving the count, the first and last timestamps, and the git-history pointer.
  - The README, MEMORY.md and website links to the file still resolve.

### U2. Untrack the vendored impeccable skill

- **Goal:** the repo carries no third-party agent tooling and no broken symlink.
- **Requirements:** R3, R4 (KTD2)
- **Dependencies:** none
- **Files:** `agent/skills/impeccable/` (remove), `.claude/skills/impeccable` (remove), `skills-lock.json` (remove), `.gitignore`, `AGENTS.md`
- **Approach:**
  1. Remove the three paths from version control and from the working tree.
  2. Add the root-anchored ignore rules from KTD2 next to the existing `.agents/` rule in `.gitignore`.
  3. Add vendored agent skills to the AGENTS.md "Do NOT commit" bullet.
- **Patterns to follow:** the existing `.agents/` ignore entry and the AGENTS.md "Do NOT commit" bullet.
- **Test scenarios:** Test expectation: none -- removes untracked-in-spirit tooling with no code references. Verified by a reference scan and the suites.
- **Verification:**
  - `git ls-files` lists nothing under `agent/`, `.claude/skills/`, or `skills-lock.json`.
  - A search of code, configs, tests, `website/`, and build files finds no reference to them. Historical prose under `docs/plans/` is exempt, because R1 keeps it verbatim.
  - Both test suites still pass.

### U3. Align the notes-encoder config and lock it with a test

- **Goal:** one notes-encoder name across config, code, and MEMORY.md, enforced by a test.
- **Requirements:** R5, R6 (KTD3)
- **Dependencies:** none
- **Files:** `configs/data.yaml`, `src/model/notes_encoder.py`, `tests/test_notes_modality.py`
- **Approach:**
  1. Set `notes.encoder` to `thomas-sounack/BioClinical-ModernBERT-base` and update its comment.
  2. Change the `NotesEncoder` docstring usage example from `-large` to `-base`, matching the constructor default.
  3. Add a test that loads `configs/data.yaml` and checks that `notes.encoder` equals `NotesEncoder`'s default `model_name`. Read the default from the constructor signature, so no model download is triggered.
- **Patterns to follow:** existing YAML-reading config tests in `tests/test_data_config.py`, and the lazy-construction test in `tests/test_notes_modality.py`.
- **Test scenarios:**
  - Happy path: with the aligned config, the configured encoder equals the `NotesEncoder` default, and the test passes without network access.
  - Error path: if `notes.encoder` is changed to a different model id, the assertion fails and names both values.
- **Verification:** the new test passes, and so does the full data-free suite.

---

## Verification Contract

| Check | Command | Applies to |
|---|---|---|
| Repo suite | `uv run --with pytest python -m pytest tests/ -q` | U2, U3 |
| Site-package suite (vendor drift) | `cd clif-validate && uv run --with pytest python -m pytest tests/ -q` | U2 |
| Docs site builds without broken links | `cd website && npm run build` | U1 (linked log) |
| No stale idle entries | heading scan of `docs/plans/gem-overnight-log.md` | U1 |

---

## Definition of Done

- R1–R6 hold, and every Verification Contract check passes.
- The branch is rebased on `origin/main`, and any idle entries appended in the meantime are collapsed (see Risks).
- No abandoned-attempt edits remain in the diff.
