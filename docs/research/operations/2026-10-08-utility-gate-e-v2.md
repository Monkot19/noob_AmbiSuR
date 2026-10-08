# Authorized Utility Gate E v2 handoff

Status: **server receipt accepted by the user**. Canonical confirmation SHA256 is `51376355bb19434ebfd118090ea2c1b38371156a9dbaf4da51bb96aaf385020e`. Seed0 Gate F0 is separately authorized; seeds1/2 and GT access are not authorized. Do not rerun this creation script.

Run `2026-10-08-create-utility-gate-e-v2.sh` on the AutoDL checkout at exact qualified `c701424c1b1f5a9006e6f19776769ee7bc8cb299`, with a clean worktree and no training/DA3. Fetching the documentation commit does not require switching the execution checkout or rerunning415 tests. The script must be transported from its pinned documentation commit, not from an unreviewed moving branch.

The script reuses `build_prior_transfer_confirmation`, `write_prior_transfer_confirmation` and `load_prior_transfer_confirmation`; it adds no solver, metric or production change. It creates only:

- `/root/autodl-tmp/ambisur_diagnostics/Utility_Room/prior-transfer/utility_prior_transfer_softv4_20261008_v2.confirmation.json`
- The same path plus `.sha256`.

It preserves v1 SHA `b991a1624f03060f0633b6b70eee16b763b4fd3fc1739daf1d956a29ea2f62c9`, reuses snapshot file SHA `522bad824116e9910636eb734efe2e222b2d5978abc3d384a0bf5f51e30cb143` and internal SHA `307b176e41111af403a565db94cfe8ada0a7d739361e1e380dcfaca08f49fc22`, and freezes fresh v2 seed/probe paths. All old and new run/view/state/launcher/qualification/probe targets must still be absent. GT metadata is copied from the frozen record; no GT byte read, parsing, visualization or evaluation is performed.

Expected receipt: `GATE_E_V2_CONFIRMATION=PASS`, canonical path/SHA, three run commands and probe targets, old-record preservation, target absence and all no-training/no-GT/no-C1 sentinels. Any error stops; do not delete records, retry with another scientific protocol or start training. Send the receipt for review before requesting seed0 launch authorization.
