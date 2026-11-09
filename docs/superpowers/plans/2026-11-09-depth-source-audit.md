# Offline depth source audit implementation plan

User approved the preceding source/admission check on2026-11-09 and confirms AutoDL is running. Execute inline in D on codex/depth-source-audit. No new training method or server auto-start. Spec: docs/research/2026-11-06-position-admission-and-supervision-redesign.md plus user-confirmed paired gain/enrichment/coverage comparison.

## Deliverable and interfaces
- Extend existing scripts/evaluate_candidate_actions.py capture with opt-in full depth-map records. Original behavior default unchanged. Same native7K/17views/spread producer/renderer, no optimization or model writes. Full maps include camera/units/source identity and model,DA3,spread(+mask), fixed-normal gate evidence if external MVS cache supplied.
- One reusable scripts/evaluate_depth_sources.py entry: infer official pinnedMVSAnywhere from existing savedcameras+fixedneighbors; score frozen fullmaps+MVS using existing GTQueries/TriangleDistance producer. No GT import in inference/admission code path.
- Add tests to tests/test_candidate_selector.py shared entry; only meaningful common-support, targetreplacement/gain/invalidmask and actualproducer-tensor normalization checks.
- Config experiments/depth_sources/toolroom_native7k_v1.json: original17views,seed0,MVShero480x640/defaultone refinement,7recordedneighbors, noGTalignment. NCCprobe16384pixels/camera,baselinepatchradius3/maxcost.5, common-valid mean over same2recordedsupports, fixedcurrentrenderednormal, compareQ with both current andDA3. It is a diagnosticNCCproxy, not an observed-correspondence gate or approvedtrainingmask. No margin fitting or Aexclusion.
- Launch guide for user: new isolated runtime/vendor/weights on data disk, fixedcommit checkout, MVS inference then originalnative7K fullmapcapture. No baseline retrain/mesh extraction.

## Steps
- [x] Read actualinputs, resourceinventory and originalcode. 7K fullmap17 missing; local savedtrace7001 is native7K beforefirstoptimizer, onefullref+support+probe only. Windows rasterizer unavailable; AutoDL capture required.
- [x] RED meaningful numerical and realtensorproducer-interface tests; implement minimal functions and captureflag; GREEN sharedentry.
- [x] Read pinned donor dependency/model-loader implementations; minimal standalone inferenceadapter with stockforward/preprocess, separateenvironment, datasetreadonly.
- [x] Freeze inputs/config before GT. Existing7001 fullreference pre/postdepth exact; no GT query yet and no substitution with Gaussiannearest stats. Complete17capture requires AutoDL.
- [ ] Independent read-only review, verify production train/renderer/model untouched, preserveoldresearchdocs/history, commit/push isolatedauditcode, prepare exactservercommands.
- [ ] Complete17source/admission score only after actualgeneratedMVS/fullmaps available; no fabricatedresults or GT-drivenretries.

## Review focus
Depthz vs rayrange; c2w/w2c and Kresizecopies; originalDA3scale and model/world units; same valid domains vs nativecoverage; fit-viewreuse; stricttargetreplacementidentity; invalidNaN not becoming quality; noinputoverwrite; noGT inmask; scientificnegative separated from interfacefailure.

## Errors / resource limits
- multi_view.json is JSONLines, wholefile json.loads failed during inventory; corrected using actualproducer lines. No GT touched.
- Native CUDA renderer unavailable locally. GPU RTX5060 has about2.6GiB free; avoid interfering with activeapps. User confirms AutoDL for read-only capture.
