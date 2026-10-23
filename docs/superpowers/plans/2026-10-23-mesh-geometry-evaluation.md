# Mesh geometry evaluation implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation; one final fresh code review.

**Goal:** One reusable CPU mesh evaluation entry for the approved formal Tool Room comparison.
**Architecture:** Single public script owns sampling, triangle distance, six metrics and JSON/CSV; existing metadata writer, JSON protocol and explicit CLI paths.
**Tech Stack:** Python3.10+, NumPy, existing Open3D; unittest.
**Spec:** `docs/superpowers/specs/2026-10-23-mesh-geometry-evaluation-design.md`.

## Constraints and review focus

D main directory/existing second-run branch. No training, rendering, baseline/data writes or old reliability system.1M points/seed0/full surface;1/2/5cm, main5cm.

Review: explicit unit/frame/model identity; bounded triangle working sets; PRF directions/strict inequality/zero denominator; real PLY producer through existing serializer/CSV; input/output non-overwrite and completion identity. Native dependency skips are not native evidence.

## Task1: shared evaluator

- [x] Write `tests/test_mesh_geometry.py`: hand metrics, validation, area sampler, native planes/missing surfaces and real PLY/CLI/metadata/CSV. Observe RED from missing module.
- [x] Implement `scripts/evaluate_mesh_geometry.py`: `summarize_distances`, `surface_samples`, `closest_triangle_distances`, `evaluate_meshes`, `main`; add `experiments/mesh_geometry/scannetpp_surface_v1.json`.
- [x] Focused tests, actual native integration if dependency available; compare real old TnT PRF producer on synthetic arrays.
- [x] One independent review and material fixes with regression tests; compile/whitespace, update handoff/ledgers.
- [x] Commit/push implementation4010fd01e4b5bbe651ac4e221e00532bf99f4ff9; origin ref matched and D clean. Pinned AutoDL handoff in evaluation guide/final response; no remote execution or new training.

## Execution ledger

Task1 code complete: initial missing-module RED observed;8/8 real tests pass, zero skips in isolated Open3D0.19.0 runtime. Real TnT scalar producer agrees; PLY→CLI→existing serializer→JSON/CSV passes; inputs unchanged and output non-overwrite verified. Python3.10 grammar and whitespace pass.
Final review: one fresh reviewer; seed+1 finding reproduced [0,1] vs[0,0] then fixed to declared seed; entire8-test suite GREEN.
Ruling: actual large-mesh cost/GT alignment/outcomes cannot be judged using synthetic tests; do not claim them. Native CLI/geometry declined by reviewer environment were independently executed by implementer. Server tests and real metrics remain pending.
