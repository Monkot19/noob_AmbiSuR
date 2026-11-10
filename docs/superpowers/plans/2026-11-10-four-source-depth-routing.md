# Four-source depth routing implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; TDD in the existing unified test entry, one final independent review.

**Goal:** Implement the approved 29-pair, four-reference offline support/routing audit without training changes.
**Architecture:** Keep old match/score schema behavior. Add routing-match and routing-score modes to the existing CLI. Reuse old 8 pair NPZ and frozen candidate depths, read additional cameras through the existing COLMAP adapter, compute 21 new pairs only. Shared geometry/ranking lives in observed_correspondence.py; paired statistics and serializers remain common.
**Tech stack:** Python3.10-compatible NumPy/Torch; existing RoMaV2 precise environment. No new dependencies.
**Spec:** ../specs/2026-11-10-four-source-depth-routing-design.md (user confirmed).

## Global constraints
- D directory branch codex/four-source-depth-routing; old branch/results/data read-only.
- Original4x16384 pixels, current/DA3/MVS/spread arrays frozen from successful8-pair result; no local MVS regrid replacement.
- Exactly29 original neighbor pairs;8 reused,21 new. No support expansion, no new depth inference or GT queries.
- Support gates: original forward validity,cycle<=1 nativepx,overlap>=.5,valid positive pair triangulation; rank angle descending/stable original index,maximum2. Old first2 uses identical gates.
- Shared selected supports for all compared candidates; invalid projection makes comparison unavailable, never deletes a bad support per candidate. Separate common3/common4 domains.
- Four-source unique mean-cost winner, ties/insufficient evidence abstain. No teacher mask or training threshold tuning.
- Per-reference/protocol/common-domain seeded source-label permutation preserves quotas incl abstain; abstain target falls backDA3 for offline target metrics.
- No train/scene/ALR/renderer/loss/LR edits; original score behavior preserved; server user-run only.

## Review focus
- Old cache identity and referencepixel/K/pose agreement; cannot attach old pair to a new geometry.
- New-camera conversion must match native producer's row-world-to-camera pose/K/center convention, including text fallback.
- Candidate-invalid selected support must not shrink evidence and create a false winner.
- Spread missing must not suppress3-source comparison or create an invalid random target.
- Cached-GT results must use schema/serialization shared with original evaluator, no NaN JSON or silent recovery after failure.

## Task1: shared support/ranking core
Files: utils/observed_correspondence.py,tests/test_candidate_selector.py.
Interfaces: pair_geometry(...)->dict angle/epipolar/positive/sensitivity; routing_evidence(candidate_costs,projection_valid,support_evidence,seed)->old_first2/new_best2 common3/common4 records.
- [ ] Write meaningful missing-helper tests for variable supports, stable angle selection, source-independent masks, ties/missing-spread and quota-preserving random labels; run RED.
- [ ] Implement minimal helpers, reuse normalized triangulation and pose/ray functions; run GREEN and old correspondence regressions.

## Task2: production-shaped CLI/cache/score
Files: scripts/evaluate_depth_sources.py,experiments/depth_sources/toolroom_four_source_routing_v1.json,tests/test_candidate_selector.py.
Interfaces: routing-match(config,source,existing,baseline,vendor,output); routing-score(matches,gt-cache,output). Existing refers successful8-pair directory; baseline supplies frozen neighbor/camera metadata identity. Read original 4NPZ+8pairNPZ; new source camera from original COLMAP loader, verify against baseline camera archive/native referenceK/pose.
- [ ] RED actual cached producer roundtrip/score serializer, invalid-cache and real camera-conversion tests.
- [ ] Implement data preflight before model load; reuse official model initialization; add new21calls only, freeze29record manifest and evidence. Preserve old functions; share initialization rather than copy model runner.
- [ ] Score frozen arrays with paired_target_stats/strict serializers, output per-source/region and old/new intersection/native domains. Fixed models/targets and random labels saved beforeGT.
- [ ] GREEN targeted tests and full existing unittest entry; actual downloaded nonGT producer preflight only, no realGT/model run.

## Task3: release and user handoff
Files: docs/research/four-source-routing-launch.md,root planning files,project-handoff.md.
- [ ] One read-only whole-branch review, verify production paths unchanged; meaningful fixes RED/GREEN if required.
- [ ] Pin production commit; copy-paste Bash uses existing environment/cache, new output and exitcode; check Bash syntax/actual CLI.
- [ ] Commit/push isolated branch; supply offline Gitbundle fallback as prior GitHub failures warrant. No automatic server execution.

Root progress.md is the existing execution ledger (reuse over a second bookkeeping framework). No milestone implies scientific benefit before actual GPU/GT evaluation.
