# selector_v3_spread Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. Steps use checkboxes. User has approved specification and instructed immediate implementation; do not request another plan approval.

**Goal:** Generate normal-guided propagated position proposals through the existing selector and evaluation pipeline.
**Architecture:** One shared pure tensor propagation/builder, small backward-compatible selector extensions, training/capture producers shared. Original losses/renderer/position executor unchanged.
**Tech Stack:** Existing Python>=3.10, PyTorch, unittest; no new dependencies.
**Spec:** ../specs/2026-10-30-selector-v3-spread-design.md and source-adaptation report linked there.

## Global Constraints
- Host integerKinv;5x5/10/.6/.99/.01 constants; firstsupportRGB and existing102400 cap; old gray min-NCC unchanged.
- Original DA3 losses; allfalse proposal means A-only, not rawDA3 fallback. Source runs every activecall; early79moves retained.
- D isolatedbranch; no data/output/baseline mutations, newlosses, servercode edits or unapproved training.

## Review Focus
- Partially invalid bilinear supports never manufacture candidate depth; original normal stencil remains rawDA3.
- Anchor NCC belongs to rendered mixed plane, not individual Gaussian; RGB/NCCscale and different source sizes align.
- Invalid/grazing/negative transfers cannot inject NaN/Inf or hallucinate depth; no anchor means falsemask.
- Old source/default stream/RNG and source-off render calls remain equivalent.
- Training/observer use same producer; rawDA3 still consumed by ALR and directdepthloss; literal target/ID order persists.

## Task1: shared matching and proposal contracts
- Files: utils/candidate_selector.py,arguments/__init__.py,tests/test_candidate_selector.py.
- [x] Add failing tests: source CLI/validation, optionalproposal target changes and falsemask A-only/no source callback, RGB matching mean agrees three realgray calls, old stream default same.
- [x] RED (missing CLI/proposal and RGB shape error) -> GREEN; source option/sharedroundtrip/RGB/proposal masks preserve old path; frozen matcher comparison exact.

## Task2: common propagation and anchor producer
- Files: newutils/depth_spread.py,same testentry.
- [x] RED missing module -> GREEN analytic tiltedplane/noanchor/normalboundary/invalid/grazing and actual RGB anchor->selector targets/RNG.
- [x] Implement common propagate_depth/build_spread_proposal with approved constants/validity/ray adaptations.
- [x] Actual source sizes/NCCscale/RGB-to-gray reference comparison and producer chain verified; feature commit after integration is included to avoid publishing unwired source CLI.

## Task3: train/capture integration
- Files: train.py,scripts/evaluate_candidate_actions.py,same tests; optional smallconfig.
- [x] RED missing proposal training/observer -> GREEN production helper/producer/observer/CLI coverage, rawprior/cache/movement preserved.
- [x] Wire sharedproducer, pre-move preview and call-local sourcecache, explicit proposal kwargs; originalmetadata/score unchanged.
- [x] Existing native class extended; local56tests48pass8native skips/11.07s. Syntax3.10, actualcaptureCLI and originalALR/score AST+renderer/Gaussian/kernel diff verified.

## Task4: review and handoff
- [x] Independent wholefeature review against fixed8b1def9/donor source: noCritical/Important; independent9newCPU pass11.415s, no edits or GT/server execution.
- [ ] Verify diff against frozen losses/renderer/Gaussianexecutor and cleancommits; publish fixedcode/handoff if ready, no remoteexperiment execution.
- [ ] Update projectledgers and concise final five-part report, including annotations1/2/3; provide next staticcapture reuse instructions, not automatic30K.
