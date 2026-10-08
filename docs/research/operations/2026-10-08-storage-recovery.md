# Storage recovery — inventory only, no deletion yet

User prefers inspection/verified offload before paid expansion. F0 remains stopped before target creation. Server-reported persistent available bytes:12308430848 (approximately11.46GiB); operational launch headroom is15GiB. No threshold change or automatic retry.

## Verified local result copies

Read-only check on2026-10-08 under `D:/research_Space/output/ambisur_diagnostics/Tool_Room/d0-g1`. Every manifest-listed artifact was checked for existence, bytes and SHA256; no local output was modified. All four inventories have zero mismatches. The subsequent user-operated server inventory reports exactly the same four manifest SHA256 values below. Server cleanup still requires fresh validation of the individual removal targets.

| Directory basename | Checked artifacts | Checked bytes | Local manifest SHA256 |
|---|---:|---:|---|
| d0_g1_exploratory500_toolroom_seed0_20260917_v3 |55|219071513|5e2b02903b0b3f44ea19f3a3dff6fab0f0486eba24f0e28e734e6f56a1bfc9aa|
| d0_g1_exploratory500_toolroom_seed0_20260917_v4 |55|219080953|30d292db1156823f6d14b5052fbc1dd063fe976e9d5f41733fb9e2e035600c62|
| d0_g1_formal7k_v2_toolroom_seed0_20260918_v1 |108|3873612476|2f3d79004afd5fc3755ef06e8e04d275aa761c6e7ed2f52efbdcbb6e8cab4743|
| d0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1 |108|4133679236|d96ecd76ee537de516435220de7ca2f81f7158c9e2d06f5bd4299037c7e3801e|

Total326 checked artifacts,8445444178 bytes (~7.87GiB), excluding manifest files. Four directories are locally present and validated; no additional download is automatically necessary for those same identities.

## Next admission before any cleanup

1. User-operated server read-only inventory: per-output allocated usage, archive file bytes, and manifest hashes. No GT mesh parser or experiment.
2. Compare server/local manifest identities and identify exact server targets. Check that chosen outputs are not live inputs/referenced evidence needed on the server. Preserve canonical records, snapshots, model weights and checkpoint assets while investigating output duplicates first.
3. Present an exact cleanup/offload proposal, expected recoverable bytes and recovery location; obtain explicit target approval before destructive removal. Hash-verified local backup is necessary but does not authorize deleting all server results indiscriminately.
4. After approved cleanup, recheck actual free space and absent F0 targets. Resume only the existing authorized seed0 protocol after the storage gate is satisfied and the failure is reviewed. No DA3, seed1/2, GT, geometry or C1 launch follows from storage recovery.

No server file has been removed, moved, overwritten or downloaded by this audit. No local result was edited. Removing installation caches on overlay would not by itself recover the persistent filesystem's free space.

## Exact removal scope — user approved, server execution pending

Server parent: `/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1`.

1. The four exact `.tar.gz` files whose basenames match the four table rows above:2182448417 bytes in total. Keep their detached SHA receipts. Local unpacked contents are verified; original compressed bytes are not locally backed up. Repacking can recover contents but is not guaranteed to reproduce the original archive SHA.
2. Only the16 manifest-listed `iteration_003000/fields/*.ply` and `iteration_007000/fields/*.ply` artifacts in each of the two formal directories (32 files,2727437757 bytes total). The eight exact field names at each iteration are `A`, `K`, `N`, `S`, `T_g`, `T_p`, `gt_distance`, `state`. These are colored visualization exports, not training point clouds/checkpoints. Restore from the same relative paths in the verified local root above.

Expected logical reclaim:4909886174 bytes (~4.57GiB); adding the reported free12308430848 bytes projects ~16.04GiB free, above the unchanged15GiB F0 launch gate. Actual allocated recovery must be checked after removal; no automatic training retry.

Keep all JSON/CSV/PNG/PDF/SVG, manifests and scientific receipts, all small component/complementarity/semantic audits, all training assets, models, canonical inputs and the Utility snapshot. Local manifest inspection shows the largest files are actually formal `report.json` (~2.12/2.34GB), not vector figures; those frozen records are explicitly excluded. Current Utility F0/probe code consumes training assets and confirmations, not these old visualization exports or archives.

User approved this exact scope on2026-10-08. If performed, the two server formal output folders become partial/offloaded copies. Preserve original manifests unchanged and publish a separate offload receipt; do not claim their full server inventory still passes. Verify all selected paths/bytes/SHA before any removal, reject links/escapes, and log exact removed targets outside the frozen result folders. Server execution remains pending; no deletion command has been run.

### Operational handoff plan

- Write a small standard-library-only checker against temporary fixtures: corrupt final target causes zero removals; manifest mutation, links and preexisting receipt fail closed; successful removal preserves unrelated reports and receipts.
- Prepare a fixed36-target operation under this directory, not a reusable recursive cleanup API. Validate every selected artifact before any deletion; journal the complete plan and each removal with flush/fsync. Stop on any error and do not auto-retry partial cleanup.
- Verify the operation with those bounded checks and AST/diff checks. Publish a docs-only handoff commit; server extracts it without changing the qualified training checkout. No full scientific suite or real-data experiment is needed for this operational step.
- User executes the pinned handoff; collect actual removal/free-space receipt before resuming F0. No automatic training dispatch.

### Safety review and bounded verification

One read-only review found a validation/unlink mutation window and missing receipt-directory fsync. A temporary regression reproduced the mutation before the fix. The operation now retains file descriptors and POSIX exclusive advisory locks, traverses and unlinks through no-follow retained parent directory descriptors, checks descriptor/entry identity immediately before removal, rejects existing writable target descriptors via Linux `/proc`, and fsyncs both journal parent and affected directories. Windows temporary checks close their handles before unlink; real execution requires Linux. As with any advisory lock workflow, this requires an explicitly quiescent operator boundary: do not start concurrent writers/copies while it runs. It is not a sandbox against malicious root processes ignoring locks/admission.

Tests cover corrupted final target (zero removal), report preservation, existing receipt, path escape, leaf/ancestor symlinks, hardlinks and late mutation. Real local pinned manifests yield exactly36 targets/4909886174 bytes without modifying local artifacts. Actual POSIX checks and target hashes run on AutoDL before removal; no local test result substitutes for the server receipt.
