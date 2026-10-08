# Utility F1 operational handoff

> **For agentic workers:** Use superpowers:executing-plans and test-driven-development in the existing isolated codex branch. User explicitly authorized Gate F1 launch and qualification.

**Goal:** Start only preregistered v3 seed1, once, after qualified seed0; then reuse the existing asset auditor.

**Architecture:** Extend the existing operational F0 transport with an explicit seed1 entrypoint, sharing worker, environment, manifest-copy and audit logic. Keep production checkout c701424 and confirmation SHA8ae37c8d0d884936c9f8416b9fd833f05f2c4c03efc7e073c8f78dccaa7653a1 unchanged; no replacement confirmation.

**Tech Stack:** stdlib operational checks using bundled Python locally; real imports and experiment remain user-operated AutoDL.

**Spec:** Approved Utility transfer spec/Task10 F1 and current explicit user authorization.

## Global constraints
- Read qualified seed0 record SHAfbad915581157be0c89e24ae13e605ddc9f1fe03a1495dc94dc44e61ce59a1c9 and its recorded input fingerprints through existing protected readers; preserve its bytes.
- Never recreate seed0, create seed2/probe targets, change confirmation, rerun DA3, access GT, change scientific code or auto-retry.
- Same15GiB storage admission, private-view copying, exact argv, GNU/OMP1 environment, worker receipts and production auditor; no415-test scientific rerun for operational-only changes.

## Review focus
- Prior qualification and live asset mutation: pinned SHA plus protected full inventory/fingerprint check before dispatch and worker start.
- Already-existing target/dangling link: raw absence checks before schema path normalization.
- Seed override: only explicit seed1 path; no generic CLI seed2 selector.
- Import/environment failure: no target creation before import smoke and existing gates.
- Audit: reuse production qualifier --seed1 without dispatch, no writes to seed0 or GT.

## One bounded task
- [x] RED: seed1 worker selects exact row1, staysGNU, writes existing receipt schema; wrong row and monitor failure cannot pass. Prior binding/changed fingerprint fails. Seed1 admission allows completed seed0 but blocks any seed1/2/probe/qualification-SHA target; unknown seed rejected. Initial20 checks fail on five missing-feature boundaries.
- [x] GREEN: parameterize existing operational prepare/worker/audit internally only0/1; add protected predecessor checker and explicit launch/audit/worker-seed1 entrypoints. No production change.
- [x] Verify23 narrow checks including original15 regressions:21 pass/two Windows link skips, AST/diff pass. One fresh read-only review approves with no actionable findings. Actual AutoDL link/import/disk/dispatch/completion/qualification evidence remains pending.
- [ ] Commit/push pinned docs and give user-operated launch command. Qualification only after exit; actual server result pending.

Ruling: this is a Task10 operational adapter, not new scientific functionality; existing approved implementation and same scientific core are reused. Current Gate F1 authorization covers its necessary transport work. Server execution remains user-operated, and partial targets on any failure remain untouched.

Final review rulings: no deferred minors. Reviewer does not certify server bytes/hashes, POSIX links, CUDA/import, resource readiness, training, qualification or scientific validity; these require actual authorized AutoDL receipts. Concurrent adversarial filesystem writers are not supported by this controlled operational handoff; no concurrent writers are permitted, with existing guarded I/O/fingerprints providing fail-closed detection, not a filesystem transaction. Treating those deferred runtime claims as proven would risk incorrect admission, so they remain explicit server gates.
