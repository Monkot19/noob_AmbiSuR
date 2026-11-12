# Dense four-source routing Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; one fresh whole-branch review at completion. User approved design and immediate implementation.

**Goal:** Four full native images /29pairs, available-source shared1/2support hybrid target and cached-GT evaluation.
**Architecture:** Extend original routing CLI in explicit dense mode; reuse project_depth/query_matches/pair_geometry/regrid_mvs/paired_target_stats and serializers. Preserve legacy modes and all training files.
**Tech Stack:** Python/NumPy/Torch/PIL, existing official RoMa precise runtime on user AutoDL.
**Spec:** ../specs/2026-11-12-dense-four-source-routing-design.md

## Global Constraints
D main directory/isolated branch,4×584×876/29pairs, original7K sources; maxangle<=2 then maxsourcecount/maxsupportcount/highangle tie; existingcycle1px/overlap.5; unique minimum/exacttie fallbackDA3. No GT model input/training/newweights/newenv/newbaseline/smoke. Existing failure outputs immutable.

## Review Focus
- MissingDA3/current/spread or badprojection must not discard unrelated usable candidates.
- Single shared support/source-priority cannot select different supports per source or based on costs.
- Same DA3-valid full-image denominator despite currentmissing; each source paired count independent.
- Query chunks cover every pixel once/model.match onlyonceperpair; real donor result types serialize.
- Preflight verifies original manifests/photos/poses/depths; random fallback strata valid; old evaluator outputs unchanged.

### Task 1: Shared dense decisions / geometry
**Files:** utils/observed_correspondence.py,tests/test_candidate_selector.py
**Interfaces:** dense_routing_evidence(costs,projection_valid,evidence,candidate_depths,seed=0,cycle_max_px=1.,overlap_min=.5,znear=.01); pair_geometry(...,compute_sensitivity=True).
- [x] Failing regressions: missing source/single support, source-first subgroup, alltie/invalid DA3 fallback, randomsource validity, sensitivity-off equals core output.
- [x] Implement minimal shared helper, optional sensitivity calculation; GREEN8/8 shared+legacy tests.

### Task 2: Dense capture and shared evaluation roundtrip
**Files:** scripts/evaluate_depth_sources.py,experiments/depth_sources/toolroom_four_source_dense_v2.json,tests/test_candidate_selector.py
**Interfaces:** routing-match --capture/--mvs in explicit denseconfig; dense source preflight, query chunks, pair cache and record build; original routing-score dispatch by audit_type.
- [x] RED missingdensequery/preflight/serializer/bootstrap; GREEN8newdense regressions including missingcurrent/DA3,full-gridchunks,actualNPZ/CLI/JSON/CSV/PNG.
- [x] Reuse actualquery/regrid/projection/geometry andpairedstats;unified92tests83pass9skip15.787s, actualdonorproducerincluded. Real2,046,336pixel/33photo/29pair preflight6.797s, noGT/model.

### Task 3: Verify and handoff
**Files:** docs/research/four-source-routing-launch.md,project-handoff.md,rootledgers.
- [ ] Actual local nonGT fourmap+29photo metadata preflight and available donor producer check (no GT/newmodel); full unified CPU entry oncegreen.
- [ ] One fresh read-only wholebranch review; necessary fixes only, preserveold numerical outputs.
- [ ] Fixed codecommit, no-install server command/originalcache reuse/estimatedoutputs, Bashsyntaxcheck; push/bundle and concise userhandoff. Servermodel/GTscore/training not locallyrun.
