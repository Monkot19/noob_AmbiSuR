# Observed correspondence audit implementation

**Goal:** Execute the user-approved four-reference/eight-pair offline check, without training changes.
**Spec:** docs/superpowers/specs/2026-11-10-observed-correspondence-probe-design.md (approved 2026-11-10).
**Architecture:** One existing CLI, one offline geometry helper, one configuration, unified tests. RoMaV2 precise produces fixed observed pixels; normalized COLMAP operators provide optional depths. Freeze non-GT evidence before reading existing GT grids and reuse paired_target_stats.

## Constraints
- Work in D main directory on codex/observed-correspondence-audit, based on saved docs commit 30b7dad.
- Preserve all original inference/scoring/training behavior, data and baseline outputs.
- Exactly original 65536 pixel IDs, eight fixed pairs, precise settings and pinned source/weight.
- No model download/local GPU run or automatic server execution. Separate server environment; all caches on data disk.
- No GT thresholds, new training mask/loss, support re-selection or silent fallback.

## Tasks
1. Write necessary coordinate/producer, common-support, triangulation/sensitivity and serialization regressions; run RED.
2. Implement helper and existing CLI match/correspondence-score extensions; config binds actual frozen probes. Run GREEN and real input preflight.
3. Prepare isolated AutoDL environment/eight-pair commands and small probe/code bundles, perform one independent branch review, verify unchanged production tree and unified tests, commit and hand off.

## Review focus
Pixel-center conventions and full bilinear footprint; reverse invalidity; saved row-transform algebra; normalized COLMAP operators; z-to-reference-ray adaptation; invalid perturbations; candidate common supports; GT-before evidence freeze; input identity and resource/environment isolation. CPU checks cannot establish actual donor GPU forward, target quality or surface benefit.

## Ledger
- Setup: user confirmed this exact batch. Prior analysis/spec preserved as 30b7dad; D isolated branch created.
- Interface preflight: match records use original native integer pixel IDs and captured K/transforms; scoring consumes frozen evidence, original depths and cached GT only. Original modes stay separate.
- Tasks1/2 complete: five new regressions RED→GREEN; actual donor geometry/confidence functions exercised, unified79/70pass9skip. Real four native-array inputs validated at65536IDs, using synthetic current projections only, no model/GT run. Config binds actual probe checksums.
- Task3 in progress: launch Bash syntax checked, original production tree unchanged. Dedicated Torch2.8/cu128+vision0.23+fused0.3.28 environment specified from official package metadata. Whole-branch review dispatched.
- Final review: independent read-only reviewer found no critical/important defects; flagged optional MVS depth_masked parity. Ruling: treat it as a reusable producer-interface contract and fix now — original score validity extracted verbatim into regrid_mvs and both modes reuse it, with no scientific change; risk if wrong is old MVS domain drift, covered by real four-file exact equality and old real serializer tests.
- Final fix RED→GREEN: original regrid regression extended to a masked hole; absent shared adapter failed, then passed. Full79/70pass9skip in5.856s. Four actual native regrids exactly equal original validity predicate (NaNs included), no GT read. Patch tool had one transient file-write error; disk/permissions normal, retry succeeded without permissions changes.
- Task3 complete pending commit/handoff: eight-pair commands, independent environment pins, 1.01MB original-probe archive,7.03MB full pinned RoMa source bundle. No deferred minors, model forward or actual quality claim.
