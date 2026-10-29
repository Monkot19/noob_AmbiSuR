# Static candidate action audit plan

User approved a static real-selector check, including assessment of additional snapshots/views, and routine autonomous fixes. Server subsequently confirmed OFF: do not start it. Local native rasterizer/compiler unavailable; finish executable CPU verification and cloud handoff, distinguish pending real capture.

## Contract

- One reusable `scripts/evaluate_candidate_actions.py` with `capture` (native renderer, no GT) and `score` (CPU GT, no training/native imports) commands. No training/selector/renderer/optimizer edits.
- Call the actual selector with targets; observe its Python return locals via a scoped profile callback. Do not copy NCC/sampling/validity rules. Restore profiler even on exceptions. Export actual accepted and rejected comparable-plane proposals; records are candidate-view observations, not independent unique Gaussians.
- Inputs read only. No `Scene` initialization (it writes source PLY/metadata), optimizer steps, relocation, gradient execution or checkpoint saves. Reuse production readColmapCameras/loadCam, GaussianModel load/restore, original render and loss/geometry utilities.
- Source camera global UID must come from real COLMAP reader (not cameras.json enumeration); first two supports from existing multi_view.json. Verify actual poses/intrinsics against recorded cameras.
- Scope: original five illustrative views + twelve sorted-name equally spaced views, separately reported. No GT error based extension. Four existing snapshots: clean native7K, C0/fullV2/early30K. No midpoint rerun; endpoints do not establish15K transition or historical decisions.
- Budget16384/seed0, saved SH/NCC settings. Snapshot7000 probed at7100, final30000 at30000 (counterfactual calls, no actual movement). Record both clocks and options.
- Score actual current/target nearest-triangle errors and first-hit reference-z errors; evaluate prior/current normal against same production GT-depth normal. Visibility/boundary/invalid domains reported separately; no GT-based training mask. Original acceptance remains fixed.
- Reuse formal mesh `_read_mesh`/`TriangleDistance`, metadata serializer. Preserve full capture and per-view/object aggregate scores; no per-pixel GT cache for training.

## Steps

- [x] TDD in existing selector test entry: real CPU selector capture equals direct output, profiler restoration, real BVH/normal scorer, input/schema/selection contracts.
- [x] Implement single capture/score script and small versioned config; real small producer→NPZ→score→JSON/CSV integration. No new test framework.
- [x] Run relevant existing tests, Python3.10 syntax, CLI/config/local snapshot checks; native test skips explicit. One fresh review plus narrow fix re-review:37 tests,30 pass/7 native skips. FP32 batch rounding issue reproduced RED then fixed; exact production targets exported.
- [x] Commit clean code c5e6243; provide pinned server capture and local score commands, update handoff. Server execution pending OFF state, no output benefits claimed.

## Expected deliverables

One production audit script, same test entry extensions, one config and one launch guide. Existing analysis docs preserved. Local CPU evidence and explicit unexecuted-native receipt. No new reconstruction run.

## 2026-10-29 COLMAP input compatibility repair

Server c5e6243 failed before camera loading/candidate execution: images.bin missing. Local actual source has images.txt/cameras.txt; original readColmapSceneInfo already falls back from binary to text. Audit omitted that input contract.
- [x] Existing test entry: real colmap_loader with tiny text/binary files, preserve insertion order/values, binary priority and original three-directory lookup. RED first.
- [x] Thin audit-only pose reader delegates to unchanged production readers with original binary→text fallback. No Scene/source writes, data conversion, UID sorting or selection changes.
- [x] Full existing two-entry suite46 tests:39 pass/7native skips; real local406-camera text poses/intrinsics match saved C0, Python3.10 parse pass; independent narrow review no blocker.
- [x] Commit fixed audit9603ef6 and pin new non-overwriting static_v2 launch directory, preserve static_v1 failure. Publish handoff next; no server code patch or new training.
