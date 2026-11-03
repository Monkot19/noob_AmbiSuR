# Selector position-check implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline; user has approved implementation, no extra confirmation round.

**Goal:** Implement only approved valid-pool sampling and current-normal relocation gate, retaining original ALR/losses/schedule.

**Architecture:** Two default-off optimization flags share the existing selector. Filter valid propagation support before sampling; reuse current cost and support data, compute fixed-current-normal proposal costs only for jointly accepted targets. Return a movement subset without changing ALR labels.

**Tech Stack:** Existing PyTorch, unittest, metadata and observer; no dependencies.

**Spec:** docs/superpowers/specs/2026-11-03-selector-position-check-design.md

## Global constraints
- D workspace and isolated branch; preserve all previous docs and outputs.
- Explicit budget16384/interval100/until15000/30K/native7K/r2/seed0; old defaults unchanged.
- No new loss/LR/renderer/propagation/GT/diagnostic framework or smoke/baseline reruns.

## Review focus
- Joint accepted but position worsening must retain ALR and return no move.
- Partial validity and unsorted IDs must not pair wrong targets.
- No new source renders or gate outside target requests; no global RNG consumption.
- Empty propagation preservesA; disabled flags reproduce frozenV3.
- Observer must validate actual subset targets and retain accepted's ALR meaning.

## Task 1: shared selector behavior and tests
Files: arguments/__init__.py, utils/candidate_selector.py, tests/test_candidate_selector.py.
Interfaces: old select labels 3-tuple and with_targets4-tuple unchanged; optional stats position_checked/position_passed only for enabled target requests.
- [ ] RED real CLI flags; prefilter budget fixture; real NCC joint-win/position-loss; partial move/empty/support cases.
- [ ] GREEN minimal flags/filter/gate, same common supports, no new position threshold; production executor subset order.
- [ ] Verify frozenV3 defaults and original tests, CPU real training helper and native test extension.

## Task 2: consumers, handoff and final verification
Files: train.py, scripts/evaluate_candidate_actions.py, same test entry; docs/research launch and project ledgers.
- [ ] RED observer subset-target invariant using real mixed model.
- [ ] GREEN observer target-mask adaptation, event log joint checked/pass counts, original record schema preserved.
- [ ] Run complete unittest discover; document native dependency skips, Python3.10 syntax/frozen source scope checks.
- [ ] Independent whole-change review through requesting-code-review; repair material issue with regression first.
- [ ] Commit preserved prior analysis separately, then feature/docs; push fixed branch after status check, prepare exact direct30K handoff. No server start.
