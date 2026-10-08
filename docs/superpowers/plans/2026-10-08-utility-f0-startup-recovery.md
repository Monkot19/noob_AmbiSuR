# Utility F0 startup recovery

> **For agentic workers:** Use superpowers:executing-plans and test-driven-development. User approved this bounded recovery in the current task.

**Goal:** Retry seed0 after the reproduced MKL startup failure without changing science or overwriting evidence.

**Architecture:** Reuse the existing operational launcher, canonical confirmation schema/serializer and asset qualifier. A one-shot recovery adapter creates fresh v3 confirmation/targets and records failed-v2 lineage. Only process-local threading changes: GNU, OMP1, no FORCE_INTEL. Production checkout stays c701424.

**Tech Stack:** stdlib operational tests locally; installed ambisur Python on AutoDL for real import-chain verification and training.

**Spec:** Explicit user authorization following DEFAULT1/GNU0 import-only diagnostic; existing approved Utility transfer specification.

## Global constraints

- No production method/package changes, DA3, seed1/2 launch, GT content, geometry, routing or C1.
- Preserve v2 failure and confirmation. Fresh v3 run/view/state/launcher/qualification/probe targets precede any training; scientific fields exactly equal v2.
- Same execution commit, snapshot, training argv except output/source paths. Failed attempt is not resumed or counted as a scientific seed result.
- Existing qualified scientific tests/core are not redeveloped; only changed operational checks are required here per user reuse policy.

## Review focus

- Inherited INTEL/FORCE settings: sanitize and pass explicit GNU at both subprocess boundaries.
- Failed attempt: verify startup-only failure and preserve/record bytes before recovery; reject completed/active attempts.
- Confirmation: reject any scientific change or preexisting new target, retain old records.
- Worker: exact new confirmation/digest/argv, no generic seed override, no automatic retry.
- Import failure: stop before confirmation/run creation; CPU import smoke is not asset qualification.

## Single bounded task

- [x] Extend operational tests: child env must be GNU/OMP1, no FORCE or Python path overrides; fresh confirmation differs only in identity/time/paths; startup-only failure gate refuses completion/checkpoints; existing handoff receipt checks remain regression.
- [x] Observe RED using bundled stdlib Python (no local Conda/Torch).
- [x] Add minimal recovery adapter and launcher env/binding changes; preserve original server artifact through immutable commits. Reuse existing guarded I/O/canonical serializer/qualifier.
- [x] Run focused operational checks, AST/diff checks; obtain one read-only review. Two review findings fixed with RED-to-GREEN checks. Local15 checks:13 pass/2 Windows link privilege skips; AutoDL must exercise both. Real import-chain check runs before publication/launch and must stop on failure.
- [ ] Commit/push exact operational artifacts; provide one user-operated command. Qualify after exit with the original auditor and new confirmation digest. No success claim until server receipts.
