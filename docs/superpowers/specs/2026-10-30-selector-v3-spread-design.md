# selector_v3_spread approved specification

2026-10-30: user explicitly approves the [source-to-interface proposal](../../research/2026-10-29-selector-v3-spread-code-adaptation.md) and authorizes implementation. This specification records that approval; it does not replace the earlier historical proposal.

- Work in D main directory, isolated codex/selector-v3-spread branch from43e2ea0; preserve all previous branches/output/baseline.
- Only new candidate position source: model rendered-depth anchors, first saved support, RGB-NCC/roundtrip confidence; cap uses existing multi_view_sample_num102400. Existing candidate budget16384 and two-support min-NCC joint comparison unchanged.
- DP-GS donor dc1ce835:5x5 neighborhood,10 steps,confidence>.6, all25 normal cos>.99, mask threshold>.01, no hard anchor reinjection; integer Kinv host rays and invalid-transfer rejection as approved adaptations.
- proposal_depth/proposal_valid are distinct from raw DA3 prior_depth/prior_normal. Both original depth loss and ALR targets remain DA3. Source invalid means no additional candidate comparison/E/relocation; A labels and baseline losses remain. SH is a label rule, not a fallback matching operation.
- Source spread applies to every active selector call. Suggested run settings already approved: same native7K/App,seed0,r2,30K,early cutoff15000/interval100 gives79relocations. No training is automatically started by local implementation.
- One shared utils/depth_spread.py producer. Reuse current selector geometry/photometry/sampler, training helper, capture/score and test entry; no customrenderer/CUDA/weights/checkpoint fields or score-core changes.
- One source CLI defaultda3; missing old option meansda3. Original sampling identity/global RNG remain unchanged. Source callback cache only within spread call; reference no-grad preview before relocation, main differentiable render after.
- Invalid spread returns finite zero placeholder plus false mask, neverNone/rawDA3 fallback. Bilinear proposal sampling requires fully valid support.
- No fixed-normal NCC/curves/gradientfusion/LR change, no propagated depthloss now. User annotation2 records propagated loss as future separately approved ablation.
- CPU and existing native integration tests; actual GPU time/memory and geometric benefit remain unverified until server execution. Reuse static capture/score before deciding new30K. No extra smoke framework or baseline rerun.

Annotation1/3 clarification: a sampled Gaussian without usable projected proposal is removed before the added geometric/NCC comparison. A-positive labels persist whether or not propagation exists; A-negative rows simply receive no new E.
