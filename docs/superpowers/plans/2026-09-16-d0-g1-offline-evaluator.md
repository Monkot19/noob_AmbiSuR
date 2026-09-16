# D0/G1 Offline Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only, deterministic Tool Room offline evaluator that joins D0 evidence to same-iteration Core checkpoints, computes all-finite-Gaussian-to-full-GT-triangle error and frozen G1 metrics, produces downloadable scientific figures, and cannot leak GT into training.

**Architecture:** Pure NumPy/Torch modules own input validation, accelerated triangle-surface queries, metrics, and deterministic colors; a thin diagnostics CLI orchestrates checkpoint/snapshot loading, plotting, Gaussian `override_color` rendering, manifests, and archival. The evaluator reads immutable run/data/GT inputs and writes only to a new diagnostics directory outside the run. Formal 7k execution remains a separate approval gate after implementation and an exploratory 500-run smoke pass.

**Tech Stack:** Python 3.10, NumPy 1.26.3, PyTorch 2.7.1+cu128, Open3D 0.18.0 RaycastingScene, matplotlib, plyfile, existing AmbiSuR Gaussian renderer, standard-library `unittest`, JSON/CSV/tar/gzip/SHA256.

**Spec:** `docs/research/ambisur-reliability-routing-design.md` §13 G1（2026-09-16 approved clarification）

## Global Constraints

- GT is offline-only and may not be imported, opened, serialized, or referenced by `train.py`, `reliability/collector.py`, `reliability/shadow.py`, evidence snapshots, or checkpoints.
- Primary error is unsigned world-meter distance from every finite checkpoint Gaussian center to the full valid GT triangle surface; no opacity/scaling/visibility/frustum/AABB/crop filtering.
- Primary label is `distance > 0.05`; 0.02, 0.10, and scene top-20% are diagnostic only. Primary prevalence outside `[0.05, 0.95]` is not evaluable.
- Iteration 7000 is the sole G1 pass gate; iteration 3000 is diagnostic only. Snapshots 1000–7000 support timelines.
- The evaluator must fail closed on provenance, iteration, schema, row-count/order-contract, required-output, nonfinite metric, or archive-manifest mismatch and must never overwrite an output directory.
- Static cameras are image-ID-sorted quarter positions `((n-1)*q)//4` for `q in {1,2,3}`; no replacement or crop.
- Output is `/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1/<confirmation_id>/` plus a sibling `.tar.gz` and `.tar.gz.sha256` containing all PNG/SVG/PDF, colored PLY, CSV/JSON, and manifest files.
- Supporting modules, C1 gradients, lifecycle behavior, baseline assets, tags, and existing experiment results are out of scope.

---

## File Structure

- `reliability/offline_g1.py`: immutable input contracts, snapshot/Core-checkpoint join, mesh validation, triangle-surface distance query, hashes.
- `reliability/g1_metrics.py`: binary labels, ROC/PR, AUROC/AUPRC, fixed-grid risk-coverage, state/timeline summaries, hard G1 gate.
- `reliability/g1_visualization.py`: fixed palettes/scales, camera selection, colored PLY writer, chart writers, required-artifact inventory.
- `scripts/diagnostics/evaluate_d0_g1.py`: CLI orchestration, model/camera restoration for `override_color` renders, report/manifest/archive publication.
- `tests/test_g1_offline_inputs.py`: malformed inputs, row joins, topology/order contract, GT isolation.
- `tests/test_g1_geometry.py`: exact face/edge/vertex distance and invalid triangle/center handling.
- `tests/test_g1_metrics.py`: tie-safe metrics, prevalence, risk direction, sensitivity isolation, hard gate.
- `tests/test_g1_visualization.py`: deterministic indices/colors/names and archive inventory.
- `tests/gpu/test_g1_override_render.py`: real renderer color override contract without training/backward/state writes.

### Task 1: Freeze the snapshot/checkpoint/provenance join

**Files:**
- Create: `tests/test_g1_offline_inputs.py`
- Create: `reliability/offline_g1.py`
- Modify: `reliability/__init__.py`

**Interfaces:**
- Produces: `G1IterationInputs`, `load_g1_iteration(run_dir: Path, iteration: int) -> G1IterationInputs`, `sha256_file(path: Path) -> str`.
- `G1IterationInputs.centers` is finite-filtered `float64 [M,3]`; `finite_row_indices` maps results back to original checkpoint/snapshot rows; `snapshot` contains all `SNAPSHOT_FIELDS` before filtering.

- [x] **Step 1: Write failing join tests**

```python
def test_load_g1_iteration_joins_same_post_topology_rows(tmp_path):
    run = write_core_fixture(tmp_path, iteration=3000, point_count=4)
    joined = load_g1_iteration(run, 3000)
    self.assertEqual(joined.original_point_count, 4)
    self.assertEqual(joined.centers.dtype, np.float64)
    np.testing.assert_array_equal(joined.finite_row_indices, [0, 1, 3])
    np.testing.assert_array_equal(joined.snapshot["N"], [0.1, 0.2, 0.3, 0.4])

def test_load_g1_iteration_rejects_row_count_mismatch(tmp_path):
    run = write_core_fixture(tmp_path, iteration=7000, point_count=4)
    rewrite_snapshot_with_rows(run, 7000, 3)
    with self.assertRaisesRegex(ValueError, "snapshot/checkpoint row count"):
        load_g1_iteration(run, 7000)
```

Also cover legacy tuple rejection, wrong checkpoint iteration, missing/extra snapshot fields, non-1D fields, duplicate output path, preservation/reporting of nonfinite center row indices, and exact equality between snapshot fields and their checkpoint `core_state` counterparts so a same-count row permutation cannot pass.

- [x] **Step 2: Run tests and verify RED**

Run: `python -B -m unittest tests.test_g1_offline_inputs -v`

Expected: import failure for `reliability.offline_g1`.

- [x] **Step 3: Implement the minimal immutable loader**

```python
@dataclass(frozen=True)
class G1IterationInputs:
    iteration: int
    original_point_count: int
    centers: np.ndarray
    finite_row_indices: np.ndarray
    rejected_center_indices: np.ndarray
    snapshot: Mapping[str, np.ndarray]
    checkpoint_sha256: str
    snapshot_sha256: str

def load_g1_iteration(run_dir, iteration):
    payload = torch.load(
        Path(run_dir) / f"chkpnt{int(iteration)}.pth",
        map_location="cpu", weights_only=False,
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("G1 requires a versioned Core checkpoint")
    if int(payload.get("iteration", -1)) != int(iteration):
        raise ValueError("checkpoint iteration mismatch")
    capture = payload["gaussian_state"]
    xyz = capture[1].detach().cpu().numpy()
    with np.load(
        Path(run_dir) / "d0_evidence" / f"iteration_{iteration:06d}.npz",
        allow_pickle=False,
    ) as data:
        snapshot = {name: np.asarray(data[name]) for name in SNAPSHOT_FIELDS}
    # Validate exact fields, common leading dimension, trailing scalar shapes,
    # and exact snapshot/core_state arrays before constructing the finite-center
    # row map without filtering any other field.
```

- [x] **Step 4: Run focused and existing checkpoint tests**

Run: `python -B -m unittest tests.test_g1_offline_inputs tests.test_core_runtime tests.test_d0_diagnostics -v`

Expected: all PASS.

- [x] **Step 5: Commit**

```bash
git add reliability/offline_g1.py reliability/__init__.py tests/test_g1_offline_inputs.py
git commit -m "feat: add offline G1 input contracts"
```

### Task 2: Add full-triangle closest-distance evaluation

**Files:**
- Modify: `reliability/offline_g1.py`
- Create: `tests/test_g1_geometry.py`

**Interfaces:**
- Produces: `ValidatedMesh`, `load_valid_mesh(path: Path) -> ValidatedMesh`, `closest_triangle_distances(points, mesh, chunk_size=65536) -> np.ndarray`.
- Open3D selects the closest triangle/point through its BVH; returned closest points and original centers are converted to float64 before Euclidean distance is computed.

- [x] **Step 1: Write synthetic face/edge/vertex RED tests**

```python
def test_closest_triangle_distance_covers_face_edge_and_vertex(self):
    mesh = ValidatedMesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64),
        triangles=np.array([[0, 1, 2]], np.int64),
        rejected_triangle_count=0,
    )
    points = np.array([
        [0.25, 0.25, 2.0],   # face
        [0.50, -2.0, 0.0],   # edge
        [2.0, 2.0, 0.0],     # vertex (1,0,0) or (0,1,0)
    ], np.float64)
    np.testing.assert_allclose(
        closest_triangle_distances(points, mesh, chunk_size=2),
        [2.0, 2.0, np.sqrt(5.0)], rtol=0, atol=1e-6,
    )
```

Add tests proving invalid indices, nonfinite vertices, and exact zero-area triangles are rejected and counted; all-invalid mesh and nonfinite query points raise errors; chunk sizes produce identical values.

- [x] **Step 2: Run the focused geometry tests and verify RED**

Run: `python -B -m unittest tests.test_g1_geometry -v`

Expected: missing mesh/query symbols.

- [x] **Step 3: Implement validation and chunked surface queries**

```python
def closest_triangle_distances(points, mesh, chunk_size=65536):
    points64 = np.asarray(points, dtype=np.float64)
    scene = o3d.t.geometry.RaycastingScene()
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.triangles),
    )
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    distances = np.empty(points64.shape[0], dtype=np.float64)
    for start in range(0, points64.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), points64.shape[0])
        query = o3d.core.Tensor(points64[start:stop].astype(np.float32))
        closest = scene.compute_closest_points(query)["points"].numpy()
        delta = points64[start:stop] - closest.astype(np.float64)
        distances[start:stop] = np.linalg.norm(delta, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("nonfinite GT distance")
    return distances
```

- [x] **Step 4: Run tests with the server dependency versions**

Run: `python -B -m unittest tests.test_g1_geometry -v`

Expected: all PASS on CPU; no CUDA allocation.

- [x] **Step 5: Commit**

```bash
git add reliability/offline_g1.py tests/test_g1_geometry.py
git commit -m "feat: compute full-mesh Gaussian GT distances"
```

### Task 3: Implement frozen G1 metrics and hard gate

**Files:**
- Create: `reliability/g1_metrics.py`
- Create: `tests/test_g1_metrics.py`

**Interfaces:**
- Produces: `binary_curves`, `fixed_risk_coverage`, `summarize_states`, `evaluate_g1_gate`.
- `evaluate_g1_gate(snapshot, distances, iteration) -> dict` always records primary/sensitivity metrics but sets `g1_evaluable/g1_pass` only for iteration 7000.

- [x] **Step 1: Write metric and anti-cherry-picking RED tests**

```python
def test_primary_gate_uses_n_against_better_component(self):
    distances = np.array([0.01, 0.02, 0.08, 0.12])
    report = evaluate_g1_gate(
        snapshot={"N": [0.0, 0.1, 0.8, 0.9],
                  "A": [0.1, 0.2, 0.6, 0.7],
                  "S": [0.9, 0.8, 0.6, 0.5],
                  **valid_reliability_and_state_fields(4)},
        distances=distances,
        iteration=7000,
    )
    self.assertTrue(report["g1_evaluable"])
    self.assertGreater(report["primary"]["auroc_n"], 0.60)
    self.assertGreaterEqual(report["primary"]["auroc_gain"], 0.03)

def test_iteration_3000_can_never_pass(self):
    report = evaluate_g1_gate(perfect_snapshot(20), separated_distances(20), 3000)
    self.assertIsNone(report["g1_pass"])
    self.assertEqual(report["role"], "early_diagnostic")
```

Also test score ties, all-equal scores, prevalence 4.9%/5%/95%/95.1%, strict `>0.05`, N rejection direction, rP/rG retention direction and validity masks, coverage grid, 100% single Bypass/Abstain stop, finite-output enforcement, and sensitivity labels not affecting `g1_pass`.

- [x] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_g1_metrics -v`

Expected: import failure for `reliability.g1_metrics`.

- [x] **Step 3: Implement dependency-free tie-safe metrics**

```python
PRIMARY_THRESHOLD_M = 0.05
COVERAGES = np.arange(0.05, 1.0001, 0.05)

def evaluate_g1_gate(snapshot, distances, iteration):
    labels = np.asarray(distances) > PRIMARY_THRESHOLD_M
    prevalence = float(labels.mean())
    curves_n = binary_curves(np.asarray(snapshot["N"]), labels)
    curves_a = binary_curves(np.asarray(snapshot["A"]), labels)
    curves_one_minus_s = binary_curves(1.0 - np.asarray(snapshot["S"]), labels)
    gain = curves_n["auroc"] - max(
        curves_a["auroc"], curves_one_minus_s["auroc"]
    )
    evaluable = 0.05 <= prevalence <= 0.95
    hard_prediction_pass = curves_n["auroc"] > 0.60 and gain >= 0.03
    role = "primary_gate" if int(iteration) == 7000 else "early_diagnostic"
    return {
        "role": role,
        "g1_evaluable": evaluable if role == "primary_gate" else None,
        "g1_pass": (evaluable and hard_prediction_pass)
                   if role == "primary_gate" else None,
        "primary": {"threshold_m": 0.05, "prevalence": prevalence,
                    "auroc_n": curves_n["auroc"], "auroc_gain": gain},
        # Include complete ROC/PR, sensitivities, risk and state summaries.
    }
```

- [x] **Step 4: Run metrics plus arbitration tests**

Run: `python -B -m unittest tests.test_g1_metrics tests.test_arbitration tests.test_reliability_evidence -v`

Expected: all PASS.

- [x] **Step 5: Commit**

```bash
git add reliability/g1_metrics.py tests/test_g1_metrics.py
git commit -m "feat: add frozen G1 metrics and gate"
```

### Task 4: Produce deterministic scientific artifacts

**Files:**
- Create: `reliability/g1_visualization.py`
- Create: `tests/test_g1_visualization.py`
- Create: `tests/gpu/test_g1_override_render.py`

**Interfaces:**
- Produces: `camera_quartile_indices`, `scalar_colors`, `state_colors`, `write_colored_ply`, `cast_gt_depth`, `write_gt_overlay`, `write_metric_figures`, `required_artifacts`.
- Fixed scalar maps use `viridis` for `[0,1]`; GT distance uses `magma` over `[0,0.10]`; states use a constant five-row uint8 palette ordered Bypass/Consensus/Prior-led/Geometry-led/Abstain.

- [x] **Step 1: Write deterministic artifact RED tests**

```python
def test_camera_indices_are_frozen_for_406_views(self):
    self.assertEqual(camera_quartile_indices(406), (101, 202, 303))

def test_gt_colors_saturate_above_ten_centimeters(self):
    colors = scalar_colors(np.array([0.0, 0.05, 0.10, 1.0]), "gt_distance")
    np.testing.assert_array_equal(colors[2], colors[3])

def test_colored_ply_keeps_every_finite_center(self):
    write_colored_ply(path, centers, colors)
    self.assertEqual(PlyData.read(path)["vertex"].count, len(centers))

def test_gt_overlay_uses_the_frozen_camera_without_crop(self):
    overlay = write_gt_overlay(camera, one_triangle_mesh(), gaussian_rgb, path)
    self.assertEqual(overlay["camera_image_name"], camera.image_name)
    self.assertEqual(overlay["image_shape"], [camera.image_height, camera.image_width, 3])
```

The GPU test must patch/inspect the renderer call to prove `override_color` is used under `torch.no_grad()`, no backward/optimizer step occurs, and rendering does not mutate captured Gaussian parameters.

- [x] **Step 2: Run and verify RED**

Run CPU: `python -B -m unittest tests.test_g1_visualization -v`

Run GPU on AutoDL: `python -B -m unittest tests.gpu.test_g1_override_render -v`

Expected: missing visualization module/functions.

- [ ] **Step 3: Implement palettes, PLY, CSV, figure writers and required-artifact inventory**

```python
STATE_PALETTE = np.array([
    [127, 127, 127],  # Bypass
    [44, 160, 44],    # Consensus
    [31, 119, 180],   # Prior-led
    [255, 127, 14],   # Geometry-led
    [214, 39, 40],    # Abstain
], dtype=np.uint8)

def camera_quartile_indices(count):
    if int(count) < 4:
        raise ValueError("at least four cameras are required")
    return tuple(((int(count) - 1) * q) // 4 for q in (1, 2, 3))
```

Write every chart once as PNG, SVG and PDF from the same matplotlib figure object; write the plotted arrays to CSV and plot metadata/ranges to JSON. Use an explicit noninteractive backend and close every figure.

Implement GT depth without a GUI: build pixel-center camera rays from `Fx/Fy/Cx/Cy`, rotate directions and origins by `camera.get_calib_matrix_nerf()[1]` (the repository's camera-to-world transform), call `RaycastingScene.cast_rays`, and overlay finite GT hits/contours on the same full-frame Gaussian render. Record the selected image name, COLMAP ID, intrinsic matrix, camera-to-world matrix, hit fraction and depth range in JSON; never change the selected camera because of the hit fraction.

- [x] **Step 4: Implement and verify renderer integration**

Restore the checkpoint into `GaussianModel`, sort train cameras by COLMAP image ID with image name as deterministic tie-breaker, call existing `gaussian_renderer.render(..., override_color=color_tensor)` under `torch.no_grad()`, and save the three full-frame PNGs for every field. Do not modify renderer/CUDA source.

Run: `python -B -m unittest tests.test_g1_visualization tests.gpu.test_g1_override_render -v`

Expected: all PASS and pre/post model tensor hashes identical.

- [ ] **Step 5: Commit**

```bash
git add reliability/g1_visualization.py tests/test_g1_visualization.py tests/gpu/test_g1_override_render.py
git commit -m "feat: add deterministic D0 G1 visualizations"
```

### Task 5: Build the fail-closed CLI, manifest and archive

**Files:**
- Create: `scripts/diagnostics/evaluate_d0_g1.py`
- Modify: `tests/test_g1_offline_inputs.py`
- Modify: `tests/test_g1_visualization.py`

**Interfaces:**
- CLI requires `--run-dir`, `--source-root`, `--gt-mesh`, `--output-root`, `--confirmation-id`, `--iterations 3000 7000`, `--expected-commit`, `--expected-dataset-sha`, `--expected-gt-sha`.
- Exit codes: `0` completed/evaluable PASS, `1` completed/evaluable FAIL, `2` malformed or not evaluable. The report is still written for exits 1/2 when output publication can be completed safely.

- [ ] **Step 1: Write CLI RED tests**

Test rejection of an existing output directory, GT under the run directory, wrong hashes/commit/iteration, missing 1000–7000 timelines, incomplete artifacts, archive path traversal, and post-read mutation. Test that `manifest.json` lists every generated file with bytes/SHA256 and the tar contains exactly the manifest inventory plus no absolute paths.

- [ ] **Step 2: Run and verify RED**

Run: `python -B -m unittest tests.test_g1_offline_inputs tests.test_g1_visualization -v`

Expected: CLI/archive contract failures.

- [ ] **Step 3: Implement orchestration and publication**

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_root) / args.confirmation_id
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    # Fingerprint run/source/GT before reads; validate metadata and inputs;
    # evaluate 3000 then 7000; write temporary sibling directory;
    # render figures; build manifest; verify inventory; atomically rename;
    # create deterministic tar.gz, hash it, then re-fingerprint all inputs.
```

The archive must include at minimum `report.json`, `manifest.json`, `inputs.json`, metric CSV/JSON, all chart PNG/SVG/PDF, all field PLYs, three PNG views per field at 3000/7000, overlays, and timeline outputs.

- [ ] **Step 4: Add a static GT-leakage regression gate**

Run:

```bash
rg -n "gt[_ -]?mesh|mesh_aligned|PRIMARY_THRESHOLD_M" \
  train.py reliability/collector.py reliability/shadow.py reliability/diagnostics.py
```

Expected: no matches. Matches are a hard failure; do not whitelist a training-path reference.

- [ ] **Step 5: Run the complete CPU suite and compile checks**

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B -m py_compile \
  reliability/offline_g1.py reliability/g1_metrics.py \
  reliability/g1_visualization.py scripts/diagnostics/evaluate_d0_g1.py
git diff --check
```

Expected: all tests PASS, compilation PASS, no diff errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/diagnostics/evaluate_d0_g1.py reliability tests
git commit -m "feat: add fail-closed offline G1 evaluator"
```

### Task 6: Run the existing 500-iteration exploratory offline smoke

**Files:**
- Modify after evidence: `progress.md`
- Modify after evidence: `findings.md`
- Modify after evidence: `task_plan.md`

**Interfaces:**
- Consumes the immutable run `/root/autodl-tmp/ambisur_runs/Tool_Room/d0-shadow-smoke-500/d0_shadow_r2_seed0_500_20260915T084720Z`, its iteration-500 snapshot/checkpoint, and the existing Tool Room GT.
- Produces an explicitly `exploratory=true`, `g1_pass=null` bundle. It must not pretend that iteration 500 is 3000 or 7000.

- [ ] **Step 1: Push an exact clean evaluator commit and verify the server checkout**

Before giving the command to the user, record the literal commit SHA in the execution message. The server preflight must require that exact SHA, a clean tree, Python 3.10.21, Open3D 0.18.0, the verified dataset SHA, and GT SHA `31547a31069f736792d4b13fff76c73483f59239dbbabe1971e115f9ab17171d`.

- [ ] **Step 2: Run focused CPU/GPU tests on AutoDL**

Run the four new CPU modules and `tests.gpu.test_g1_override_render`; stop on any failure. Record wall time and peak GPU for the rendering test.

- [ ] **Step 3: Run the evaluator in explicit exploratory iteration-500 mode**

The CLI must require `--exploratory --iterations 500`; without `--exploratory`, iteration 500 must be rejected. Inputs and existing run files are read-only, and the output confirmation ID must be newly frozen before launch.

- [ ] **Step 4: Deep-audit the bundle**

Verify input before/after hashes, rejected center/triangle counts, row alignment, distance finiteness, metric finiteness, fixed camera IDs, exact required-artifact inventory, tar member list, archive SHA, no training process, and clean Git. Open representative PNGs and verify that PNG/SVG/PDF/CSV share identical plotted data.

- [ ] **Step 5: Record evidence without claiming formal D0/G1**

Use hypothesis language for visual/metric interpretation. Record exact paths, hashes, resource use and any stopped gate in the three planning files.

- [ ] **Step 6: Commit the verification record**

```bash
git add task_plan.md findings.md progress.md
git commit -m "docs: record exploratory G1 evaluator smoke"
```

### Task 7: Freeze formal D0 readiness and stop for training approval

**Files:**
- Modify: `task_plan.md`
- Modify: `progress.md`
- Create: `docs/superpowers/plans/2026-09-16-d0-7k-execution.md`

**Interfaces:**
- Produces a literal-sha, literal-path confirmation contract and user-operated Bash sequence for Tool Room r2/seed0/7000 with default refresh 1000, checkpoints 3000/7000, and no GT argument.
- Formal execution may start only after the evaluator smoke, full regression, clean push, output nonexistence, dataset/prior hashes, disk and single-training-process gates pass and the user explicitly approves that run.

- [ ] **Step 1: Write the execution plan with exact command construction**

The plan must use a preflight-created confirmation JSON containing the eventual literal implementation commit, normalized no-GT training command, source/private-view/run/output paths, dataset/prior/GT hashes, seed 0, resolution 2, refresh interval 1000, iterations 7000, testing iterations 1000–7000, checkpoint iterations 3000/7000, and proof that target paths do not exist.

- [ ] **Step 2: Define formal completion audits**

Require exit 0, one completion marker, snapshots 1000–7000, checkpoints 3000/7000, finite evidence, refresh count 7, last refresh 7000, exact snapshot/checkpoint row joins, input hashes unchanged, zero GT reference in training log/config/checkpoint, clean Git, and one optimizer step per eligible iteration.

- [ ] **Step 3: Define post-training offline evaluation**

Only after training completion, invoke the evaluator with the frozen GT SHA and a new diagnostics output. The G1 result comes solely from iteration 7000; archive and report paths must be printed for user download.

- [ ] **Step 4: Run plan self-review**

Check every approved G1 requirement maps to an implementation task; scan both plans for unfinished markers, vague cross-task references, or unbound paths; verify signatures and state names across tasks.

- [ ] **Step 5: Stop and request explicit formal 7k authorization**

Do not create a tag and do not launch training merely because evaluator implementation or the exploratory smoke passed.

## Plan Self-Review Record

- Spec coverage: input isolation, same-row topology join, full finite-center domain, full valid triangle mesh, primary/sensitivity labels, prevalence, hard AUROC gate, risk directions, state/timeline diagnostics, mesh alignment preflight, deterministic views/scales, all requested formats, archive/hash, and formal-run authorization are each assigned above.
- Placeholder scan: operational paths are fixed by the approved server layout or generated and frozen by confirmation contracts; future literal commit SHA is deliberately an execution-time frozen value and may not be guessed in this design-time plan.
- Type consistency: snapshot keys match `reliability.diagnostics.SNAPSHOT_FIELDS`; Core checkpoint fields match `reliability.runtime.build_checkpoint_payload`; Gaussian centers are capture index 1; state order matches arbitration IDs 0–4.
