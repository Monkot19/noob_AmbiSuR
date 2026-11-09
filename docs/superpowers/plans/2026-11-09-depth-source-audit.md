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
- [x] Independent read-only review complete; production paths unchanged. Prior research preserved in8c3805b, audited code fixed cae81969d5eee7f97e5a4fd9d8151b300742fbb4. Server commands pinned; release/push verification recorded in progress.md.
- [ ] Complete17source/admission score only after actualgeneratedMVS/fullmaps available; no fabricatedresults or GT-drivenretries.

## Review focus
Depthz vs rayrange; c2w/w2c and Kresizecopies; originalDA3scale and model/world units; same valid domains vs nativecoverage; fit-viewreuse; stricttargetreplacementidentity; invalidNaN not becoming quality; noinputoverwrite; noGT inmask; scientificnegative separated from interfacefailure.

## Errors / resource limits
- multi_view.json is JSONLines, wholefile json.loads failed during inventory; corrected using actualproducer lines. No GT touched.
- Native CUDA renderer unavailable locally. GPU RTX5060 has about2.6GiB free; avoid interfering with activeapps. User confirms AutoDL for read-only capture.

## Server interface failure: donor CLI parsing (2026-11-09)
- Evidence: capture/dependencies/freeze succeeded; inference exits2 because pinned OptionsHandler.parse_and_merge_options reads the outer audit CLI. Donor options.py309 exposes ignore_cl_args=False and321 parses sys.argv.
- [x] Add one regression to the shared selector tests using the actual pinned donor OptionsHandler and the production constructor call under an audit argv; observe exit2 before the fix.
- [x] Set ignore_cl_args=True only on the adapter configuration call. Keep YAML, weights, forward, cameras and scientific protocol unchanged; do not edit the vendor or training code.
- [x] Run shared tests72/64pass8nativeSkip and actual donor-boundary verification; independent read-only review found no material issues. Commit/push release next. Preserve failed output; restart infer only into a new sibling directory, referencing existing capture/runtime/vendor/weights and recording their identities. No repeated capture, installation, training or GT scoring.
