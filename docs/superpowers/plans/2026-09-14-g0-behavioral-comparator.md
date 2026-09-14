# G0 Behavioral Schema-3 Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user selected inline, single-agent work and user-operated AutoDL commands; do not dispatch subagents or control the server directly.

**Goal:** Add a read-only, explicit schema-3 G0 audit in which observable behavior and structural safety remain hard gates while all 1,926 internal Gaussian/Adam numerical summaries remain complete diagnostics.

**Architecture:** Keep schema-1/schema-2 report assembly and `evaluate_triplet_report` unchanged. Put schema-3 contract validation and pure gate classification in a focused module, then branch `build_report`/CLI only when both `--topology-aware` and `--behavioral-g0` are explicit. A pre-launch, hash-pinned confirmation contract prevents a retrospective E0 replay from becoming a confirmation PASS.

**Tech Stack:** Python 3.10.21, PyTorch 2.7.1+cu128 for checkpoint loading, standard-library `unittest`/JSON/hashlib/pathlib, RTX 4090 AutoDL for user-operated verification. Local dependency-free pure tests run on the available Windows Python; checkpoint integration tests run on AutoDL.

**Spec:** `docs/superpowers/specs/2026-09-11-g0-behavioral-equivalence-design.md`; highest-priority method contract `docs/research/ambisur-reliability-routing-design.md` §13.

## Global Constraints

- Scope is the read-only comparator, its tests, this plan and evidence docs. Do not modify `train.py`, loss, renderer/CUDA, optimizer, topology, data, old result directories or tags.
- Default schema 1 and explicit `--topology-aware` schema 2 must retain their report content and old all-summary-hard decisions. The 2026-09-10 formal schema-2 FAIL remains immutable.
- Schema 3 requires explicit `--topology-aware --behavioral-g0`; behavioral mode without topology-aware mode is malformed before artifact loading.
- Keep factor exactly `2.0`, with no confirmation CLI override. L1/PSNR at `500,1000,5001,7001,8000`, pre-topology log/PLY count, post-topology checkpoint count and fixed-shape app Tensors are numerical hard gates using the existing self-repeat envelope.
- The 13 capture and 12 Adam-moment Gaussian fields must yield 25 unique fields and exactly 1,926 unique, finite, deterministically named summary scalars in the frozen Tool Room r2/8k confirmation. Their numerical outliers are diagnostics only; field presence, dtype, trailing shape, optimizer structure/step, emptiness and finite checks are hard.
- Confirmation requires a pre-launch JSON contract and independently recorded SHA256. The new E0 output must have been absent at preflight, and the contract must name frozen B1/B2, canonical source/prior roots and the exact new E0 path. A CLI-supplied SHA alone does not prove chronology; the user-visible preflight record must be frozen before launch.
- Existing B1/B2 and first E0 are read-only. The first E0 can be replayed only as exploratory/retrospective; it cannot establish revised G0. Unseen E0 is a separately approved future run, not part of this implementation authorization.
- Use `apply_patch` locally. Any proposed `git add/commit/push` or AutoDL command is a future execution step requiring explicit user approval. Server commands are given one at a time for user copy/paste and output review.

## File responsibility map

| File | Responsibility |
|---|---|
| `scripts/diagnostics/compare_feature_off.py` | Leave existing schema-1/2 and envelope evaluator untouched; schema-3 may import its pure `evaluate_triplet_report`/`evaluate_scalar_triplet`. |
| `scripts/diagnostics/behavioral_g0.py` (new) | Pure schema-3 gate/outlier logic plus read-only contract and immutable-input fingerprint helpers; no Torch or file mutation. |
| `scripts/diagnostics/audit_feature_off_triplet.py` | Minimal explicit schema-3 report/CLI branch; reuse existing topology-aware summaries, exact invariants, fixed Tensor distances and log loader. |
| `tests/test_behavioral_g0.py` (new) | Dependency-free RED/GREEN tests for hard/diagnostic split, completeness, factor, contract fields/hash and retrospective prohibition. |
| `tests/gpu/test_feature_off_triplet_audit.py` | Existing Torch synthetic-artifact fixtures extended for schema-1/2 compatibility, schema-3 assembly, safety and CLI integration. |
| `tests/gpu/test_feature_off_dispatch.py` | Extend the existing feature-off dispatch test to prove one legacy-path invocation and no Core-path invocation; pair with exact source-diff review. |
| `task_plan.md`, `findings.md`, `progress.md` | Record TDD evidence and provenance without presenting hypotheses as verified conclusions. |

---

### Task 1: Freeze schema-3 pure hard/diagnostic decision

**Execution status (2026-09-14):** Tests-first RED observed as the planned missing-module import; minimal pure evaluator then passed 21/21 combined local tests, compile and `git diff --check`. No test-only commit or server action was taken.

**Files:** Create `tests/test_behavioral_g0.py`; create `scripts/diagnostics/behavioral_g0.py`.

**Interfaces:** `evaluate_behavioral_report(report: dict, expected_diagnostic_names: set[str]) -> dict`. It consumes schema-3 `exact_invariants`, hard `numeric_fields`, hard `scalar_metrics`, `diagnostic_scalar_metrics` and `diagnostics`; it produces `hard_equivalent`, `equivalent`, `factor`, `exact_failures`, `numeric_failures`, `numeric_results`, `scalar_results`, `diagnostic_results`, `diagnostic_outliers` and grouped counts. Existing `finalize_gate` will consume `equivalent`.

- [ ] **Step 1: Write RED tests for one hard pass plus a diagnostic outlier.** In a new `unittest.TestCase`, construct a minimal report with `schema_version=3`, one passing `evaluation[500:train].psnr` scalar and one outlying `capture.xyz.channel_000.mean` diagnostic. Assert `hard_equivalent=True`, `equivalent=True`, the outlier remains in `diagnostic_results`, and family/count are `capture:1`. Use the actual existing scalar rule:

```python
hard = {"name": "evaluation[500:train].psnr", "b1": 20.0, "b2": 21.0, "e0": 20.5}
outlier = {"name": "capture.xyz.channel_000.mean", "b1": 0.0, "b2": 1.0, "e0": 4.0}
report = {"schema_version": 3, "exact_invariants": [], "numeric_fields": [],
          "scalar_metrics": [hard], "diagnostic_scalar_metrics": [outlier],
          "diagnostics": {}}
gate = evaluate_behavioral_report(report, {outlier["name"]})
self.assertTrue(gate["hard_equivalent"])
self.assertEqual([x["name"] for x in gate["diagnostic_outliers"]], [outlier["name"]])
```

- [ ] **Step 2: Write RED negative tests.** In separate subtests change the hard scalar E0 value to `24.0`, add a failing fixed-Tensor pair from `tests/test_compare_feature_off.py::_field`, add an exact invariant mismatch, remove the diagnostic, duplicate its name, set one value to `NaN`, and pass an extra unexpected diagnostic name. Each must produce a hard failure, except the first outlier-only case. Assert factor is exactly `2.0` and the public function has no `factor` parameter.

- [ ] **Step 3: Run dependency-free RED.** Run `python -B -m unittest tests.test_behavioral_g0 -v`. Expected: import/attribute error for `evaluate_behavioral_report` only; existing `python -B -m unittest tests.test_compare_feature_off -v` remains green. Record exact output, then stop if unrelated tests fail.

- [ ] **Step 4: Implement minimal pure decision.** Reuse the frozen envelope evaluator on hard collections only; evaluate diagnostics separately without including their numerical failures in the Boolean. Validate exact expected diagnostic-name set, uniqueness and finite B1/B2/E0 values before evaluation. A readable implementation shape is:

```python
def evaluate_behavioral_report(report, expected_diagnostic_names):
    if report.get("schema_version") != 3:
        raise ValueError("behavioral evaluator requires schema 3")
    metrics = report["diagnostic_scalar_metrics"]
    names = [item["name"] for item in metrics]
    if len(names) != len(set(names)) or set(names) != set(expected_diagnostic_names):
        raise ValueError("incomplete or duplicate diagnostic summary names")
    for item in metrics:
        if not all(type(item.get(role)) in (int, float) and math.isfinite(item[role])
                   for role in ("b1", "b2", "e0")):
            raise ValueError("nonfinite diagnostic summary")
    hard = evaluate_triplet_report(report, factor=2.0)
    diagnostic_results = [evaluate_scalar_triplet(m["name"], m["b1"], m["b2"], m["e0"], factor=2.0) for m in metrics]
    outliers = sorted((r for r in diagnostic_results if not r["passed"]), key=lambda r: r["name"])
    return {**hard, "hard_equivalent": hard["equivalent"],
            "diagnostic_results": diagnostic_results,
            "diagnostic_outliers": outliers,
            "diagnostic_outlier_counts": group_by_validated_field_and_statistic(
                outliers, set(expected_diagnostic_names))}
```

`group_by_validated_field_and_statistic(outliers: list[dict], expected_diagnostic_names: set[str]) -> dict` is a new helper in the same file: parse the `.<channel_NNN|row_l2>.<statistic>` suffix of each already-validated summary name, require the parsed full name to belong to `expected_diagnostic_names`, then increment nested `family/field/statistic` integer counts. The one-field pure-test fixture supplies its sole valid name, while real confirmation supplies exactly 1,926 names from 25 frozen fields. Do not group by arbitrary substring guesses. A malformed summary is a hard validation error, not an ignored outlier.

- [ ] **Step 5: Run GREEN and review.** Run `python -B -m unittest tests.test_behavioral_g0 tests.test_compare_feature_off -v`; require zero failures and stable schema-1/2 comparator tests. Then `python -B -m py_compile scripts/diagnostics/behavioral_g0.py` and `git diff --check`. Suggested review checkpoint/commit after authorization: `test: define behavioral G0 hard and diagnostic gates`, followed by `feat: add pure schema-3 G0 evaluator`.

### Task 2: Validate the pre-launch confirmation contract

**Execution status (2026-09-14):** Four contract/fingerprint tests produced the planned missing-interface RED while the six prior pure tests stayed green. Minimal read-only implementation then passed the combined 25-test local run with one Windows symlink-permission skip; that case and real Torch artifacts remain for AutoDL.

**Files:** Modify `tests/test_behavioral_g0.py`; modify `scripts/diagnostics/behavioral_g0.py`.

**Interfaces:** `load_confirmation_contract(path: Path, expected_sha256: str) -> dict` reads only; `confirmation_invariants(contract: dict, run_directories: dict[str, Path], run_contracts: dict[str, dict], resolved_configs: dict[str, dict], fingerprints_before: dict[str, str], *, iteration: int, evaluation_iterations: list[int], expected_baseline_commit: str, expected_e0_commit: str, expected_dataset_sha: str, expected_prior_sha: str) -> list[dict]` returns exact-invariant records for `build_report`. `fingerprint_immutable_inputs(contract: dict, run_directories: dict[str, Path], iteration: int, *, roles: tuple[str, ...] = ("b1", "b2", "e0")) -> dict[str, str]` hashes sorted consumed run files and canonical source/prior trees without writing; the explicit B1/B2 subset permits preflight while the new E0 directory is absent.

- [ ] **Step 1: Write RED contract tests using temporary paths.** Build a canonical JSON object with `contract_version=1`, `confirmation_id="g0-schema3-unseen-e0-test"`, `preflight_utc="2026-09-14T00:00:00Z"`, `e0_output_absent_at_preflight=true`, `paths={b1,b2,e0}` (resolved absolute strings), `canonical_source_root` and `canonical_aligned_prior_root` (resolved absolute strings), preflight `canonical_source_tree_sha256`/`canonical_prior_tree_sha256` and `baseline_artifact_fingerprints={b1,b2}`, `commits={b1,b2,e0}`, dataset/prior SHA, `seed=0`, `resolution=2`, `iteration=8000`, `evaluation_iterations=[500,1000,5001,7001,8000]`, and role-specific `normalized_commands`. Write it under the temporary diagnostics root and pass `hashlib.sha256(path.read_bytes()).hexdigest()`. Assert a valid contract yields only passing invariants and no input file changes.

- [ ] **Step 2: Write RED mutation tests.** Independently mutate one of SHA, role path, role commit, command, data/prior SHA, source/prior tree SHA, seed, resolution, iteration/evaluation list, missing `preflight_utc`, and `e0_output_absent_at_preflight=false`; set E0 `start_utc` earlier than preflight time. Wrong SHA or malformed JSON must raise `ValueError`; mismatching evidence must yield named failing hard invariants. Assert the old E0 path cannot pass a contract naming the new E0 path. Fingerprint a temporary file set, edit one byte, then assert the second digest differs; a missing or empty consumed file must raise rather than be skipped.

- [ ] **Step 3: Run RED.** Run `python -B -m unittest tests.test_behavioral_g0 -v`. Expected: missing loader/validator interface errors only; record output.

- [ ] **Step 4: Implement read-only validation.** `load_confirmation_contract` must reject absent/non-object JSON, non-64-character lowercase SHA, and computed SHA mismatch before loading run artifacts. `confirmation_invariants` must compare exact role paths/commits/commands and source hashes with `g0_run_contract.json`, compare seed/resolution/schedules with launcher `training_config` and E0 resolved metadata, compare the initial source/prior tree digests and B1/B2 artifact digests against preflight contract values, read `start_utc.txt` under the supplied E0 run directory to verify `preflight_utc < E0 start_utc`, and require the absence attestation. Use existing `_exact`-shaped records so the unchanged comparator can evaluate them:

```python
def _contract_exact(name, actual, expected):
    return {"name": name,
            **{role: actual[role] for role in ("b1", "b2", "e0")},
            "expected": {role: expected[role] for role in ("b1", "b2", "e0")}}
```

The content-tree helper must hash each relative path plus its file content in sorted order, reject links escaping either canonical root and reject missing/empty consumed run files; the preflight uses this same deterministic algorithm to record the two canonical tree hashes. The common consumed-run allowlist is `train.log`, `chkpnt{iteration}.pth`, `point_cloud/iteration_{iteration}/point_cloud.ply`, `app_model/iteration_{iteration}/app.pth`, `cfg_args`, `cfg_opts`, `g0_run_contract.json`, `exit_code.txt`, `start_utc.txt`, `end_utc.txt`, `gpu_peak_mib.txt`, and `launcher.log`. Require `resolved_config.json` and `run_identity.json` only for the E0 role; B1/B2 legacy runs are not expected to have them. Existing B1/B2 launcher contracts expose `training_config.semantic_seed`, `resolution`, `iterations`, `evaluation_iterations`, not separate top-level values; verify those four fields per role, and check E0 resolved metadata in addition. Explicitly encode this role-specific manifest in the fingerprint, rather than silently skipping an absent expected file. Add a RED test for a symlink escaping the canonical tree. The code cannot independently prove when a user wrote a file; the separately recorded SHA from the approved preflight is the chronology anchor. Do not silently accept a contract whose expected SHA is omitted or is computed at invocation time without that record.

- [ ] **Step 5: Run GREEN and review.** Run `python -B -m unittest tests.test_behavioral_g0 -v`, compile the new module, and `git diff --check`. Proposed commit after authorization: `feat: validate pre-launch G0 confirmation contract`.

### Task 3: Assemble schema-3 evidence without changing schema 1/2

**Execution status (2026-09-14):** User approved a test-only sync; commit `5832336992860f9dc04cb13ed873f2a1aded062f` contains only `tests/gpu/test_feature_off_triplet_audit.py` and was pushed. On clean AutoDL `research/core-routing` with Torch 2.7.1+cu128, the focused suite ran 22 tests: 16 pre-existing tests passed and all six new tests raised the expected `TypeError: build_report() got an unexpected keyword argument 'behavioral_g0'` (exit 1). This verifies the missing interface, not yet the six behavioral assertions. No training ran. Continue only with the approved read-only comparator implementation; require a later AutoDL GREEN before closing Task 3.

**Component GREEN observed; Task 3 not closed:** User separately approved a two-file commit/push, `502025a61d301ce4b6c601ea25e618af7109341a`, containing only `audit_feature_off_triplet.py` and `test_feature_off_dispatch.py`. AutoDL fast-forwarded clean `research/core-routing` to that exact commit and ran the two focused Torch modules: 27/27 PASS, exit 0, post-test clean, no training. The subsequent full `unittest discover` on that same clean commit passed 61/61, exit 0, without training. This verifies six schema-3 assembly/safety tests plus legacy dispatch and covered regressions. Local pure tests remain 25 OK (one Windows symlink skip). Field-integrity edge tests, schema-3 CLI and real-artifact checks are still outstanding; no G0 PASS or D0/C1 authorization follows from this component result.

**Later evidence (2026-09-14):** An empty fixed app Tensor produced the expected AutoDL RED at test-only `c81aabc` and passed focused/full (1/1, 62/62) after the schema-3-only fix `ff7a877`. Exploratory CLI tests were RED at `f6704dd` and GREEN (3/3, full 75/75) at `a995142`. A read-only historical replay reproduced the exact frozen schema-2 FAIL report SHA `7f513603171c28ae546760e3a979616aa7bb39cdec404e9c06a3bf6428fd9e5f`; the old E0 schema-3 exploratory report retained all 1,926 summaries and 343 outliers but remained `g0_equivalent=false`, with unchanged input hashes. Formal confirmation tests were RED at `ac91026` and GREEN (3/3, full 78/78) at `301e69781861c32e61100251e5ebe150940a4e7c`. These are comparator/component tests only: no new pre-launch contract or unseen E0 exists, and historical G0 remains FAIL.

**Files:** Modify `tests/gpu/test_feature_off_triplet_audit.py` and `tests/gpu/test_feature_off_dispatch.py`; modify `scripts/diagnostics/audit_feature_off_triplet.py`; use `scripts/diagnostics/behavioral_g0.py` from Tasks 1–2.

**Interfaces:** Extend `build_report(..., behavioral_g0=False, confirmation_contract=None)` and add `behavioral_summary_name_set(topology_evidence, capture_tensors, optimizer_tensors) -> set[str]`. Schema 3 has hard `scalar_metrics` (2 topology counts + 10 L1/PSNR), unchanged fixed app `numeric_fields`, `diagnostic_scalar_metrics` (1,926 summaries), and hard exact/safety/confirmation invariants. Default/schema-2 return dictionaries keep their original keys and values.

- [ ] **Step 1: Write Torch-fixture RED tests for mode dispatch.** Reuse `_write_synthetic_triplet` at iteration 8. Assert default `build_report(..., exploratory=True)` is schema 1, `topology_aware=True` alone is schema 2, and `topology_aware=True, behavioral_g0=True` is schema 3 with `diagnostic_scalar_metrics` separate. Assert behavioral without topology-aware raises before `load_checkpoint` is called (patch that function to throw if touched).

- [ ] **Step 2: Write RED field-integrity and safety tests.** On a valid miniature triplet assert count metrics and evaluation remain hard, every `capture.*`/`optimizer.*.exp_avg*` summary moves unchanged into diagnostics, and `appear_ab` remains a direct hard `numeric_field`. Delete one expected field, duplicate one summary name, alter trailing shape/dtype or optimizer step, remove/empty a required artifact, remove `Training complete.`, add a duplicate evaluation line, add a nonfinite Tensor, and set E0 peak above `22*1024` MiB in separate fixture copies. Require each condition to be a hard failure or malformed audit; diagnostic numerical outliers alone must not fail. The fixed confirmation expects exactly 25 internal fields/1,926 summaries; the small synthetic fixture validates the same per-field name formula without pretending to have 1,926 entries.

- [ ] **Step 3: Run expected AutoDL RED.** Because Windows Python lacks the project Torch runtime, send one bounded, read-only command to the user after a test-only commit is approved: `python -B -m unittest tests.gpu.test_feature_off_triplet_audit -v`. Accept RED only for the new behavioral keyword/field/helper gaps; 16 pre-existing focused tests must remain PASS. Stop and diagnose any unrelated failure.

- [ ] **Step 4: Implement the minimal schema-3 branch.** After the existing `build_topology_aware_tensor_evidence` call, derive diagnostic names independently from the frozen 13 `CAPTURE_NAMES[1:14]` and six Adam moment groups (`xyz`, `f_dc`, `f_rest`, `opacity`, `scaling`, `rotation`) times `exp_avg/exp_avg_sq`, with nine statistics for each channel and row-L2. Require 25 field prefixes, names unique and exactly 1,926 names for real confirmation; require each Tensor finite/nonempty and original dtype/trailing shape equal across roles. Partition by these validated names rather than numerical value. Keep schema-1/2 returned report construction literally behind the old branch:

```python
if behavioral_g0 and not topology_aware:
    raise ValueError("behavioral G0 requires topology-aware mode")
if behavioral_g0:
    report["schema_version"] = 3
    report["diagnostic_scalar_metrics"] = gaussian_metrics
    report["scalar_metrics"] = count_metrics + evaluation_metrics
    report["expected_diagnostic_names"] = sorted(expected_names)
```

For schema-3 only, add hard checks for one completion marker, nonempty required artifacts, exactly one train evaluation at each frozen boundary, `training_path=legacy`, Core flags false, no unsupported/nonfinite optimizer state and same step counters. Reuse existing `optimizer.structure` exact check and `run.final_points_matches_ply`. Resource hard gate uses existing per-run peak/wall evidence: each peak below the design's 22 GiB cap and E0 wall time no more than twice the larger B1/B2 wall time; record optional sampled memory as diagnostics, never invent absent B1/B2 samples. For the no-extra-backward gate, extend `tests/gpu/test_feature_off_dispatch.py` to count the selected training-path invocation when all Core flags are off and confirm it is the single legacy invocation; separately review `git diff d6f15c8891a53800d5e3100f95817a7dd7f98e2f a26082154889ed539322425347af5a57a859a52f -- train.py` to confirm the executed legacy backward/step block was not altered. Record both pieces as source/dispatch evidence; optimizer step equality alone cannot prove backward-call count. Stop if source review finds an extra backward or the test cannot establish legacy dispatch. Preserve schema-2 diagnostic JSON unchanged.

- [ ] **Step 5: Run GREEN and review.** On AutoDL require `python -B -m unittest tests.gpu.test_feature_off_triplet_audit -v` PASS, `python -B -m unittest discover -s tests -p 'test_*.py' -v` PASS, compile/CLI PASS, and clean worktree after committed sync. Locally require dependency-free comparator tests, `git diff --check`, changed-file allowlist and no modifications under training/renderer/CUDA. Suggested commits after authorization: `test: lock schema-3 report partition and safety` then `feat: assemble behavioral G0 evidence`.

### Task 4: Integrate the CLI and prove backward compatibility

**Files:** Modify `tests/gpu/test_feature_off_triplet_audit.py`; modify `scripts/diagnostics/audit_feature_off_triplet.py`; update `task_plan.md`, `findings.md`, `progress.md` with observed evidence.

**Interfaces:** Add `--behavioral-g0`, `--confirmation-contract`, `--expected-confirmation-sha` to `main(argv=None)`. Exploratory schema-3 may omit confirmation fields but `finalize_gate(..., exploratory=True)` must keep `g0_equivalent=false`. Non-exploratory schema-3 requires both contract arguments. Schema-1/2 CLI output and gate stay unchanged. For non-exploratory schema 3, `main` fingerprints all consumed run artifacts and the canonical source/prior trees before and after loading; it requires unchanged digests and an initial canonical digest equal to the preflight contract. These checks enter the hard report before `g0_equivalent` is computed.

- [ ] **Step 1: Write RED CLI tests.** Using temporary synthetic run dirs, call `main` with both schema-3 flags plus `--exploratory` and assert schema 3, `audit_completed=true`, `g0_equivalent=false`, full diagnostic set and zero hard failures for identical role evidence. Call behavioral alone and assert malformed return `2` with no output file; call non-exploratory schema-3 without either contract argument and assert return `2`. With a valid hash-pinned contract, assert confirmation uses only hard gates; mutate contract hash and assert return `2`, never PASS. Mutate an input file between the before/after fingerprint calls and assert `g0_equivalent=false` with a named immutability hard failure.

- [ ] **Step 2: Run RED on the test-only commit.** AutoDL command: `python -B -m unittest tests.gpu.test_feature_off_triplet_audit -v`. Expected only missing CLI flag/keyword integration failures; record server branch, exact HEAD, test exit and clean status before production changes.

- [ ] **Step 3: Implement CLI dispatch and report gate.** Parse the three new flags; validate impossible flag combinations immediately after `parse_args`, before resolving or loading artifact paths. For non-exploratory schema 3, load/hash-check the contract, calculate immutable fingerprints before/after `build_report`, and append exact hard invariants for initial canonical tree SHA and each unchanged digest. Then evaluate via `evaluate_behavioral_report(report, expected_names)` and `finalize_gate`. For schema 1/2 retain the current call to `evaluate_triplet_report(report)` and current `finalize_gate` path. In schema 3 emit `hard_equivalent` plus diagnostic outlier counts in the JSON and console summary; do not make diagnostic outliers affect `exit_code`.

```python
if args.behavioral_g0:
    gate = evaluate_behavioral_report(report, set(report["expected_diagnostic_names"]))
else:
    gate = evaluate_triplet_report(report)
report["gate"] = finalize_gate(gate, exploratory=args.exploratory)
```

Before using `expected_diagnostic_names`, Task 3 must have derived and checked them from frozen field layouts; a report-provided list alone is not trusted. For non-exploratory schema 3, append passing/failing confirmation-contract invariants to the same hard exact list before this gate call.

- [ ] **Step 4: Verify old modes and historical FAIL.** Run focused and full AutoDL suites, `python -B -m py_compile scripts/diagnostics/compare_feature_off.py scripts/diagnostics/audit_feature_off_triplet.py scripts/diagnostics/behavioral_g0.py`, and `python -B scripts/diagnostics/audit_feature_off_triplet.py --help`. Re-run the existing schema-2 formal triplet read-only into a *new diagnostics path*, using the same old role paths/flags, and compare its report SHA with frozen `7f513603171c28ae546760e3a979616aa7bb39cdec404e9c06a3bf6428fd9e5f`; it must remain FAIL. If output SHA differs, inspect exact JSON diff and stop before any unseen run. Replay that same old E0 in schema-3 `--exploratory`; diagnostic outliers must be retained, and `g0_equivalent` must remain false. Verify old artifacts unchanged and server worktree clean. No comparator-only PASS can bypass the schema-3 internal immutability hard invariants.

- [ ] **Step 5: Review and proposed commit.** Review only comparator/tests/docs diffs; request code review, verify no old behavior regression, then propose `feat: expose explicit schema-3 behavioral G0 audit`. Do not create a stage tag or push without user authorization. This is a comparator component gate, not G0 PASS.

### Task 5: Freeze unseen-E0 preflight and stop for approval

**Files:** Update `task_plan.md`, `findings.md`, `progress.md`; create a new diagnostics-side confirmation contract only on AutoDL through a future user-operated command. Do not write under the repository, canonical data, B1/B2 or any existing run.

**Interfaces:** Contract JSON version 1 with the exact keys from Task 2. The separate preflight output records the contract's byte SHA256, current clean comparator commit, approved spec commit, absent new E0 path, immutable B1/B2 hashes and the user-visible unique confirmation id.

- [ ] **Step 1: After local and AutoDL comparator GREEN, propose one no-training preflight command.** It must first check clean `research/core-routing`, exact approved comparator HEAD, frozen `c0-baseline`, no active training, disk/GPU/runtime, canonical dataset SHA `aad92aa2e0f0d072756b3a56c686d5c1d35f448811ce60ca4360c67dbc3ef255`, aligned-prior SHA `69a21ab8756f43834a5357f27ca6cf40c6b7b15695e0e8cade1914fe70956977`, and B1/B2 required artifact hashes. Derive a unique confirmation id with `uuid.uuid4().hex`; ensure the new E0 output path and private-view path do not exist. Do not create either run path during preflight.

- [ ] **Step 2: Write and freeze the contract outside runs.** The future user-operated preflight writes JSON under `/root/autodl-tmp/ambisur_diagnostics/`, records its SHA256 and UTC time in the returned terminal output, and records exactly the B1/B2/new-E0 directories, resolved canonical source/prior roots and their independent content-tree SHA256 values, role commits (`d6f15c8891a53800d5e3100f95817a7dd7f98e2f` twice; `a26082154889ed539322425347af5a57a859a52f` once), normalized per-role commands, data/prior SHA, seed 0, resolution 2, iteration 8000 and evaluation list `[500,1000,5001,7001,8000]`. Return the output for assistant review; freeze its path/hash in a reviewed planning record before any E0 launch.

- [ ] **Step 3: Stop at authorization boundary.** Ask for separate approval of exactly one unseen E0 8k run. If approved, provide its launcher command only after reading the preflight output and verifying the new paths still absent. After it completes, audit the new E0 safety/input hashes, run one schema-3 confirmation against frozen B1/B2, and stop on any hard failure. Do not start D0/C1 or create a tag without further approval.

## Verification and acceptance checklist

- [ ] All new pure tests are observed RED before implementation and GREEN afterward; old schema-1/2 tests never regress.
- [ ] AutoDL focused and full standard-library suites, compile and CLI checks pass on an exact clean commit. The server is operated by the user, one command/output review at a time.
- [ ] Historical schema-2 report remains the original FAIL with the original report SHA; old E0 schema-3 replay is exploratory only.
- [ ] Schema-3 hard evidence and all 1,926 diagnostics are complete, named and finite; numerical outlier count may be nonzero without changing the hard Boolean.
- [ ] Only after comparator freeze may a new pre-launch contract and new E0 be requested. Revised G0 is not claimed until the unseen triplet passes every hard gate.
