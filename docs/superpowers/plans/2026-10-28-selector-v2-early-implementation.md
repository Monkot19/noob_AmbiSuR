# v2_early Implementation Plan

**Goal:** Add one backward-compatible exclusive cutoff for relocation, then hand off the approved native7K→30K experiment.

**Spec:** `docs/superpowers/specs/2026-10-28-selector-v2-early-design.md` (approved user proposal, no new confirmation required).

**Architecture:** Existing relocation activity function gates both the pre-render action and its event logging. Add one option there; leave train.py and all selection/optimizer/ALR implementations unchanged.

## Global Constraints
- D main directory/newisolatedbranch; no newworktree, baseline/history/data/output untouched.
- Default0 and absentfield preserve originalV2;15000exclusive gives79events7100..14900.
- Aftercutoff fullV1labels/extra/ALR/normaloptimizer continue to30K.
- Only three codefiles; one existingtestentry; no newenvironment/framework/GPU smoke.

## Review Focus
- OldNamespace without cutoff and featureoff remain compatible; metadata uses existing writer.
- Exactly15000 event excluded but candidate selector remains active; final30000 behavior unchanged.
- Real traininghelper aftercutoff must still perform V1selection/source renders and mainforward/backward/step.
- Launch preserves original fullV2 and privateinputs and nativecheckpoint; no newbaseline/recompile.

### Task 1: Optional cutoff, same-path tests and fixed-commit handoff

**Files:** Modify `arguments/__init__.py`, `utils/candidate_selector.py`, `tests/test_candidate_selector.py`; create `docs/research/selector-v2-early-native7k-launch.md`; update handoff/progress docs.

**Interfaces:** Consumes existing candidate_relocation_active(iteration,opt), real traininghelper and OptimizationParams metadata; produces optional integer candidate_relocation_until_iter=0, CLI explicit15000. Existing return types/functions unchanged.

- [x] Baseline existing entry:30tests/24CPU pass/6native missingextension skips.
- [x] Add failing existing-entry tests: default0/explicitCLI/metadata, missingfieldcompat, exact79vs229schedule, invalidcutoff; extend realhelper event14900/cutoff15000/later/noevent/featureoff paths.
- [x] Run focused tests and observe RED due absentfield/cutoff ignored.
- [x] Implement fielddefault0 and validatedexclusivecutoff only in existing activityfunction.
- [x] Run `D:/research_Space/ambisur-eval-runtime/Scripts/python.exe -B -m unittest tests.test_candidate_selector -v`.
  Expected:31tests/25CPU pass/6native skips, zero failures; no GPU claim.
- [x] Check Python3.10 syntax/whitelistedcode diff, commit module `feat: add optional cutoff for candidate relocation` (a52ab904).
- [x] Create pinnedcommit direct30K launch doc with uniqueview/output, sameparentcheckpoint/extension, explicitcutoff15000; verify Bash syntax/actualCLI parse.
- [ ] Complete whole-task test via task-done; one fresh finalreview; fix only material issues; publish isolatedbranch for serverfetch and give short pinned-doc execution command.

No methods/loss changes outside the spec. No automatic cloud execution without connection; returnlogs required to establish launch/completion.
