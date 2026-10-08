# Utility transfer implementation: code-qualification handoff

## Status and identity

- Implementation commit: `66b7d161b1e908cc5219f4b5722744b2ee3e27cb`.
- Task 7 server receipt: exact `18bbbf1b14d15d1b4979e8bbd88e4d30f9ce6977`, 37 focused / 395 discovery tests OK, clean checkout.
- Task 8 RED/GREEN: `9a07fad` / `51c2a9a`; review regression RED/fix: `bcde20e` / `66b7d16`.
- Tasks 8–9 code qualification: user-operated AutoDL receipt at exact `c701424c1b1f5a9006e6f19776769ee7bc8cb299` is **PASS**. This is not real-data or scientific qualification.
- No production method, frozen Tool Room statistics, Evidence v4, training, gradients, topology or routing changed.

## Authoritative AutoDL receipt

The user supplied the exact-commit server output on 2026-10-08: CUDA/Open3D PASS; 57 focused tests OK; 415 discovery tests OK; qualification exit 0 and clean detached checkout. The intentional argparse error text is a tested failure path; discovery reports OK. Training, real Utility GT access and C1 remained unstarted.

This resolves the local dependency limitation below. Execution remains pinned to `c701424c1b1f5a9006e6f19776769ee7bc8cb299`; a documentation-only receipt commit does not require repeating the same code qualification. Preserve the frozen statistical engine and DA3 snapshot. No real geometry release or mesh admission is inferred.

## Historical local evidence

Bundled Python was used; the user's Conda environments were neither invoked nor modified.

- Focused Tasks 1–8 plus frozen complementarity/confirmation/publication regressions: **142 tests OK, 6 skipped**.
- Discovery: **278 tests, 22 errors, 37 skips**; every error is `ModuleNotFoundError: torch`. This is **not** full-suite qualification.
- All six Utility CLI help surfaces pass; nine changed Python files compile; `git diff --check` passes.
- The intentional argparse required-arguments message is a tested failure path, not an unexplained discovery failure.
- Torch/Open3D/CUDA behavior and Linux exclusive directory publication must be qualified on AutoDL; local synthetic success cannot establish real mesh admission.

## Review and decisions

One fresh read-only review covered Task 8 plus whole-branch contract integration from `8980d58`. Two Important findings were reproduced and fixed under TDD, with no Critical or Minor findings:

1. Retain canonical GT identity before admission through evaluation/publication, rather than accepting changed post-admission bytes as a new baseline. Initial bad identity returns `INCONCLUSIVE`; subsequent mutation publishes nothing and preserves the access log.
2. Supply known protected GT/target identities to every repeated prior reload, including authorization and token verification, to block replaced GT aliases before reading bytes.

Execution rulings, with their costs if wrong:

- Qualification transport is three ordered path/SHA handles; output/staging/log must be the exact preregistered paths. Cost: request-adapter correction before external execution, not a changed statistical protocol.
- Protected reference reads precede existing run-binding validation with `verify_record_files=False`; default callers still verify references. Cost: guarded-adapter maintenance/security regression, covered by order and alias tests.
- Streaming file fingerprints avoid loading gigabyte checkpoints only for hashing. Windows descriptor checks use dev/inode/size/mtime; full path-generation checks also retain ctime over consumption. Cost: platform-specific identity regression; Linux is still an explicit qualification gate.
- Review did not judge real geometry release/human approval, scientific outcomes, datasets, training or actual GT/backend/server validity. These remain external gates; cost of treating them as proven would be unauthorized GT access or unsupported claims.
- Missing local dependencies originally blocked full qualification; the exact-commit AutoDL receipt above now resolves that code-only limitation. Preserve local evidence as history, not as a current server blocker.

## Schemas, interfaces and unchanged protocol

- Source audit / DA3 snapshot / prior-transfer confirmation / run qualification / firewall token / transfer report: existing schema 1; D0 runtime 2, Evidence 4, temporal 1.
- Fixed scene/seeds: Utility Room, 0/1/2; one frozen DA3 snapshot; six checkpoint/snapshot joins at 3000/7000.
- Sole comparison: `M0=[A,1-S]`, `M1=[A,1-S,1-r_p]`; no r_g, temporal feature, new N, alternate solver, fold, bootstrap or threshold.
- 3000 remains direction stability/descriptive only; 7000 remains the unique performance gate.
- Evaluator CLI: repository, exact commit, prior confirmation path/SHA, geometry release path/SHA, source root, GT path, output root, safe diagnostic ID, three repeated qualification path/SHA pairs. No scientific overrides.
- Exactly seven files: inputs.json, mesh_admission.json, report.json, seed_folds.csv, risk_bins.csv, bootstrap.csv, manifest.json. No PNG/PLY/checkpoint copies/archive.

## Historical prepared code-only qualification

The authorized final server gate is complete as recorded above. The broader preparation command below is retained as history, not a request to repeat tests and not permission to run the evaluator on real data:

```bash
"$PY" -B -m unittest \
  tests.test_utility_snapshot tests.test_utility_source_cli tests.test_utility_da3_cli \
  tests.test_g1_prior_transfer tests.test_g1_prior_transfer_confirmation \
  tests.test_g1_prior_transfer_confirmation_cli \
  tests.test_prior_transfer_assets tests.test_prior_transfer_assets_cli \
  tests.test_utility_gt_firewall tests.test_g1_prior_transfer_cli \
  tests.test_g1_complementarity tests.test_g1_complementarity_cli \
  tests.test_g1_confirmation tests.test_g1_publication \
  tests.test_g1_geometry tests.test_g1_offline_inputs -q
"$PY" -B -m unittest discover -s tests -t . -q
```

Record exact output, exit codes, skips, static checks and clean status. Only synthetic tests are invoked. Do not run any source/DA3/qualification/evaluator CLI against real assets during this code gate.

## Unchanged external stops

- Preserve Gate E v1 and its SHA; it cannot qualify new-code runs. A replacement canonical confirmation needs separate approval and must precede all real run/view/state/launcher/probe targets.
- No Utility seed has been launched by this implementation.
- No real geometry G-C candidate/terminal release exists; `NO_SEMANTIC_REPAIR_JUSTIFIED` does not release Utility GT.
- Geometry release, first semantic GT access/probe and each training seed require their own approval.
- No DA3 rerun, geometry intervention, five-state arbitration, production routing or C1 is authorized.
