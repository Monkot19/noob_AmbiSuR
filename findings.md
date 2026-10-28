# AmbiSuR 新项目 Findings

## 2026-10-26 Native baseline completion and resume consumer fix

Returnednativebaselinefull30Klog:50:21,PSNR31.072280120849612,L10.016401121392846107,N1291587. Exitcode/checkpointfilelisting pending. FoundoriginalresumeAppweights reads newdestination/latest;second-run fix isoneline selectingcheckpointparentandexactfirst_iter. ActualGaussian.capture/restore+Appsave/load+actualtrainingASTbranch serializedCPUintegration demonstratedRED(wrongApp8vs.2)→GREEN,nonzeroAdamstep/momentsandnextupdatepreserved. Suite24run19pass/5native skipped;independentscopedreviewnoissues. Nativebaselinebranch whole-tree unchanged;RNG/cameraqueue/Appoptimizer omissions remainaccepted.

## 2026-10-26 Native baseline zero-change protocol approved

User approves pureC0 nativecheckpoint route. C0train already exposescheckpoint_iterations andsavescapture+iteration plus7KAppweights afteroptimizerstep;safe_state hardcodesseed0,no --seedCLI. No baselinefile/module/hook added. NativeRNG/cameraqueue/Appoptimizer omissions remain disclosed;latercandidate branches sharemodel/Adam startingstate and restartprocedure,not guaranteedcontinuousbaseline randomtrajectory. Baseline runs full30K;newbranch reference atd6f15c8 andnewoutput only. Old53K/full-state-stop proposals superseded.

## 2026-10-26 Shared prefix boundary

User selects preserving selector-free7Kprefix. Actual train saves ordinaryPLY before densification/trim/optimizerstep,while checkpoint is after. Usingiterations7000 wouldskipstep7000 viaiteration<opt.iterations;keep total30000andstopseparately aftercompleted7000. ExistingPLY alone andlegacycheckpoint lackingRNG/cameraqueue cannot establish exactcontinuation. Shortspec ready for review;no production edits or newGPUjob.

## 2026-10-26 Baseline lineage audit (current)

Before budget launch user questions early-trajectory confounding. Both actual downloaded runs have7KPLY/30KPLY/30Kapp.pth but no7Kchkpnt. Original capture savesGaussian/Adam/densification stats,notRNG orcamera queue. A16384-onlyPLYrestart is not a fair comparison withcontinuousold4096. Recommend discussing one new shared7Kprefix plus two matched23Ksuffixes(53Ksteps) for budget attribution,or explicitly exploratory single30K. No launch/checkpoint implementation yet;fixedbaseline unchanged.

User receipt confirms1219 current training inputs match C0 manifest,0missing/0mismatched. d979c07 parent is exactly d6f15c8;62e9117 is not ancestor of trained3d85faf. Official remote main verified at88d64054;C0's8different Python sources have identical AST and4native sources identical noncomment tokens. All1689baseline files remain,1680byte-identical;only4existing production files changed for selector/seed integration. Protected1630files matchC0;old24reliability sources absent. Current training/test blobs match trained commit. Existing ignored138code files are allpyc;no ignored .py/.so/.pyd/native source;old runtime/config/arbitration source-less resolution returnsNone. Fresh suite23run/18pass/5native skipped;no fresh GPU acceptance. Full audit report follows;do not reinterpret as full7K trajectory equivalence or an identified cause of early divergence.

## 当前增补：实际C0配对诊断（2026-10-23）

C0下载mesh SHA与正式报告相同；C0/new input、cameras、neighbors相同。正确PSNR31.07406044→31.10275154，仅5个训练视角。复用原核心与GT样本，均值复算最大差4.62e-13mm。z<4m平均Completeness31.7489→29.9042mm，上部84.1396→119.6272mm；最坏1%点贡献整体增量4.5284mm。完整性差距88.03%已在raw中；最坏点84.67%满足至少一相机GT first-hit 1cm可见容差。不能主要解释为后处理或完全没拍到。

关键混杂：上部局部盒Gaussian中心7000轮已为C0 29904/new14205，30000为28431/15848；selector从7001启动，日志600轮首次点数分叉。中心数不等于7K表面误差，但最终差距不能全归因于补选。种子均为0，未证明数值不确定性或具体代码路径为根因。下一批建议只读核对1219实际图像/深度/置信/对齐文件；暂不调4096或重训baseline。详见结果报告末节与独立analysis目录。GT仅用于离线最终评价诊断。

> 当前二周目主目录为 `D:\research_Space\noob_AmbiSuR`；旧 Core 内容为历史，最新状态见文末及 `docs/research/project-handoff.md`。

本文件只记录新项目通过代码、测试和实验得到的事实。设计假设不得写成已证实结论。

## Confirmed Inputs

- Spec：`docs/research/ambisur-reliability-routing-design.md`
- Fork：`https://github.com/Monkot19/noob_AmbiSuR`
- GPU：RTX 4090 24GB
- 主场景：Tool Room、Utility Room
- 开发场景：Tool Room seed 0

## Evidence Table

| Claim | Status | Evidence/Path | Next check |
|---|---|---|---|
| feature-off 与 baseline 等价 | untested | | E0 |
| N 比 A 或 1-S 更能定位几何风险 | hypothesis | | D0/G1 |
| 双可靠性能校准风险 | hypothesis | | coverage-risk |
| Abstain 能减少错误选边 | hypothesis | | C2 vs C3 |
| 参数路由改善几何 | hypothesis | | C3–C5 |
| 生命周期改善稳定性与几何 | hypothesis | | C5 vs C6 |

## Code Findings

每项记录：文件与行号、可复现证据、是否已测试、是否与设计冲突。不要只写“可能有问题”。

## Experiment Findings

每项必须引用对应 manifest、commit、scene 和 seed。不得把单次运行写成稳定结论。

## Decisions

| Date | Decision | Evidence | Consequence |
|---|---|---|---|



## 2026-10-22 Clean second-run migration
User approves c0-baseline branch plus selector-only port and necessary seed/metadata reuse. Old records above are historical. The original dirty checkout stays unchanged. No baseline/GT/storage operation.
- Server disk evidence supplied by user: overlay30G used25G/free5.3G, persistent50G used36G/free15G; no inode issue. pip cache nearly empty, conda tarballs106.7MB. Three environments (base,ambisur,da3-v1) and VSCode CLI3.8G explain major visible use; da3-v1 is an environment, not proof of duplicate DA3 weights. Overlay du43G versus df25G means visible directory totals cannot be treated as reclaimable writable-layer bytes.
- Do not relocate/delete Conda or DA3 envs, model weights or old results. Next no-GPU inventory can focus on VSCode versions/active executable paths and installed rasterizer provenance. No remote cleanup executed.

## 2026-10-23 Server cache and installed extension evidence
- User-supplied no-GPU receipt shows six Stable VSCode Server versions under /root/.vscode-server/cli/servers totaling3.8G; no matching executable links were printed. This alone does not prove no other client references a version.
- Fresh read-only local code --version returns1.131.0, commit e4c7e7b1d6d060162f4aa7f8225271b67ce1df75, matching the server671M Stable directory; retain that version and propose only the five explicitly listed older directories, after closing all remote VSCode clients and using the web terminal. Approximate candidate directory total3.1GiB, actual recovery requires df.
- User's active ambisur imports rasterizer wrapper and binary from /root/miniconda3/envs/ambisur/lib/python3.10/site-packages and exposes rasterize_gaussians_with_evidence=True. Thus branch/source migration alone has not selected a baseline-only installed extension; API presence alone does not prove numerical interference.
- c0-baseline ext.cpp exports only rasterize_gaussians/rasterize_gaussians_backward/mark_visible (corrected after server receipt: markVisible is the C++ symbol and Python wrapper method, not the extension export); setup.py uses CUDAExtension/BuildExtension. Propose building pinned baseline extension into a separate small data-disk target and using per-process PYTHONPATH for the second run, preserving the first-run package/environment. Before no-GPU build, verify nvcc/toolkit and target architecture; explicit TORCH_CUDA_ARCH_LIST is needed without visible GPUs. No installation, code changes, deletion or remote execution this step.

### 2026-10-23 Build receipt and assertion correction
- User successfully removed the five explicitly named old VSCode versions: system df now22G used/8.3G available/73%; CLI672M. Actual recovery is approximately3G, no Conda/model/results cleanup needed now.
- Pinned c0 wheel built and independently installed at /root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python; wheel SHA256 56a2e694c96145e81081d15486679e8bb3fb2eb3b95d2d3ba6e1538a0de37346. Import and wrapper/binary target assertions passed before the erroneous markVisible assertion; absence of old evidence export was not yet checked because execution stopped first.
- Root cause verified with git show c0-baseline:.../ext.cpp: m.def("mark_visible", &markVisible). Baseline wrapper markVisible calls _C.mark_visible. Agent's handoff command used the wrong extension name; no evidence of a build failure. Correct only the one-shot validation command and reuse the same target; native GPU tests/training remain pending.
- Subsequent user receipt confirms corrected check passed with wrapper/binary in the independent target, all baseline exports callable and old evidence export absent. This establishes import/export selection only; it does not establish GPU rendering, selector integration or reconstruction benefit.

### 2026-10-23 AutoDL native acceptance and next run configuration
- User-pasted full output confirms exact3d85faf95951b4806eee18498841e66551b93320 on RTX4090, baseline extension selected from U365Ym/python, all23 unified tests pass in2.875s with no skips. Native producer/renderer/ALR backward+optimizer, normal-coordinate, feature-off/e0 and callback checks actually executed. Synthetic active helper27.8098ms/17607168peak allocated bytes is not a real-scene time/memory estimate. Tensorboard absence did not prevent existing console/file metadata tests.
- Read-only frozen Tool_Room_r2 cfg_args/cfg_opts and train.log confirm406views,r2,eval=False,seed0 baseline defaults,30k,DA3,unc_from_iter7000,original ray/trunc/depth settings. No baseline checkpoint/resume required; candidate first activates at7001. Dataset source path in old cfg is historical and cannot be assumed still present.
- Original project task plan requires canonical uploaded data to stay read-only: baseline loader can write sparse_da3_aligned/0/points3D.ply. Reuse a qualified writable Tool Room working view (or discuss a minimal private copy); do not launch directly against canonical data. Need current server path, not another scientific audit or new smoke framework.
- Directory receipt resolves an existing c0 r2 working-view parent. To preserve old input state, use a new thin view; image/DA3 array links do not duplicate bulk data, and an independent aligned sparse copy absorbs loader's unconditional PLY rewrite. Fresh actual-parser comparison to frozen cfg_args/cfg_opts passes, with only intended path and new selector/seed fields differing. Directory existence alone is not proof of training success; launch command checks required input files before starting.

## 2026-10-23 Primary D workspace and running evidence

- User's latest excerpt reaches23760/30000, usually sampled4096 with nonzero additions; iteration21400 has sampled0. No cause inferred for that empty pool. total_alr_weight is the full rendered supervision weight, not the newly added weight or geometric improvement. Final metrics remain unknown.
- Inventory: D candidate-selector-v1/62e9117 had8modified+16untracked files; C second-run/3775135 was clean; both share D/.git. Byte-exact backup preserves24files/987252bytes and54historical files. Complete bundle6121675bytes verified, SHA256193c8c88f628e6e881bdad7a873a6c96851ea7f3e797cb05fe2a88d39d0e9b34. Archive7b3154b pushed; original branch remains62e9117.
- After user detached C, D normally switched to second-run. Fresh checks:1697tracked files present,8source/test files match C after LF normalization,192ignored hashes unchanged,C raw code unchanged. Existing D suite18pass/5native skips;8syntax checks pass. No new method/framework/dependency or server operation.

## 2026-10-23 First selector30K completion-log receipt

- User returns30000/30000,final Gaussian/app_model save and Training complete. New train PSNR31.102751541137696,L1 0.0163764713332057,points1295772,progress elapsed1:14:56. Exitcode file and saved-file existence are not yet returned; no full artifact qualification claimed from a log excerpt.
- Fresh read-only fixed Tool_Room_r2 train.log: PSNR31.098477172851563,L1 0.016407448239624502,points1297143,elapsed42:54. Computed deltas: PSNR+0.0042743682861328125dB,L1-0.00003097690641880313(-0.1887978%),points-1371,observed elapsed ratio1.7466977(+32:02). These are training-view log values; not held-out rendering/geometry improvement. Timing was not a contemporaneous paired benchmark, so the full difference cannot be causally assigned to the selector.
- Final selector record sampled4096/comparable3143/added1463,total_alr_weight255571.625. Added denotes labels in that iteration, not new primitives or an accumulated unique count; total weight includes baseline labels. It establishes activity, not correct-geometry coverage or extra supervision weight by itself.
- Read fixed extraction log and actual extract_general.py: baseline used max_depth5.0,voxel_size0.005,sdf_trunc_scale4.0,num_cluster2;mesh/tsdf_fusion.ply and tsdf_fusion_post.ply already exist. Original extraction log spans roughly3minutes after model load, not a promised new-run duration. Proposed next batch only extracts the new model with identical settings and existing renderer; no baseline retraining/re-extraction,GT access,metric pipeline expansion or code change performed.
- Subsequent user receipt confirms train.exitcode0 and nonempty30K point_cloud.ply307M/app.pth15K; user authorizes extracting the mesh for baseline geometry comparison. Extraction can retain tested server3d85faf and standalone baseline extension, without fetching local documentation commits or rebuilding. Existing extract_general.py writes method mesh/train renders into its model_path and uses the method's private input view; baseline output is not a write target. Standard render(opt=None) remains supported by the shared label helper.
- Current clean repository evaluators are dataset-specific DTU/TnT; they do not establish an already-approved ScanNet++mesh protocol. Geometry comparison requires shared GT frame/units and an equal evaluation domain; generic two-mesh discrepancy alone does not identify which geometry is correct. No GT content or metric result read in this handoff.

## 2026-10-23 Mesh protocol
ScanNet++ has no verified official reconstruction mesh benchmark; AmbiSuR formal geometry uses DTU/TnT. User requests formal ScanNet++ adaptation: complete surface mean distances/CD and strict-threshold PRF at1/2/5cm(main5cm),1M area samples/seed0. Old TnT scalar definitions inspected; archive closest-point kernel can be isolated without reliability framework. Current local Python3.12 lacks Open3D/SciPy; native verification must be explicit. No actual GT read.

Formal evaluator evidence: Open3D0.19.0 on external isolated Windows Python3.12 runtime passes8/8. Analytic40mm/identity/missing surface/triangle interior and real PLY→CLI→existing metadata writer→JSON/CSV verified; strict-threshold PRF matches existing TnT function on same arrays. No real mesh/GT inspected; actual high-resolution cost/frame and geometry benefit await server. GT full-surface scope includes unseen surfaces, declared before results.

## 2026-10-23 First formal Tool Room result
User provides Completed log and full6-row CSV at evaluator4010fd0: baseline CD40.715599347687984mm,selector46.02748794544195mm (+13.046322988870841%); accuracy+5.964331533667264%,completeness+18.418124068287643%. F1@1/2/5cm +0.2192956064/+0.4510731129/+0.1150998749 percentage points.5cm recall-0.282pp. Mixed result, no stable/overall geometry benefit established; no automatic new modules or protocol changes. Threshold counts vs mean distance differ mathematically, aggregate values do not identify spatial cause.

## 2026-10-23 Budget hypothesis: observed logs vs inference
Read already-downloaded run metadata/log only.115 selector log snapshots:sample/comparable/added medians4096/2699/983,added mean1011.93,logged added/sample25.1426%; only2 zero-sample snapshots. Selector executes EVERY iteration7001..30000,200 is log interval. No proof budget4096 sufficient/insufficient: actual eligible pools/rendered extra mass not logged. More important competing hypotheses: invisible/low-contribution additions and full-image normalization; best-source NCC selection precision; plane position evidence vs unchanged normal-only ALR action. Uniform shared-depth prior loss already exists. No root cause claimed, no new training/code.

## 2026-10-23 Downloaded artifact identity and pool-size evidence
New postmesh hash and localGT hash exactly match formal results;JSON/CSV exact. Run has1577files,406RGB/depth/normal each,complete7000/30000PLY lengths. Original local baseline differs from formal C0 mesh (hash/size),user is downloading C0. New/original input,cameras,neighbors hashes match but7K counts/PSNR already differ before selector active; not proof of mechanism failure. Three final snapshot pool estimates129416/741704/164701 mean4096 sampling3.165%/.552%/2.487%. Derived conditional expected extraALR scales min(1,B/M),so fixed budget changes per-view extra supervision strength;not proof of final metric improvement. Combined debug masks normalized per-image cannot isolate extra dose.

## 2026-10-26 Native checkpoint scalar root cause
Serverf41failedatfirsttorch.loadbeforecontinuedstep: NumPycoremultiarray.scalar rejectedbyrestrictedloader. ActualgetNerfppNorm radius flows through Scene.cameras_extent→Gaussian.spatial_lr_scale→capture. PriorCPUfixtureincorrectlyusedPythonfloat2.5;usingactualnormalizationproducerreproducesthesameerrorlocally. Explicitweights_only=False load preservestheself-producednativecheckpointtypes/valueswithoutproducerchanges;owntrustedcheckpointonly. ExistingAppweightsstate_dict contains tensors andloadsunchanged. Thisisconsumercompatibility,not scientificnegativeevidence/budgeteffect/checkpointcorruption.

## 2026-10-27 Download archive verified locally
4749files7373232294Bpassed:13PLYpayloads/finitevertices/faceindices,5PTHCRC+CPUload,4683fullJPEGdecodes,406RGB/depth/normalperrun. Native7Kcheckpointmatcheslaunchrecordedc0264cf...sha;restores7000/16fields/1421298Gaussians/6Adamstates/7groups. Threecameras/neighbors/initialcloudsidentical,twooptimizationcfgsonlybudgetdiffers. Pair/extractionqueue/alltrain/extractexitcodes0. Baseline1578files3421450805B;pair3171files3951781489B. Fullmanifestsoutsidearchiveinanalysis/shared_native7k_archive_20261027_v1. Serverfullhashcomparisonnotyetperformed;noGT/geometryevaluation/deletion. Cleanupcandidatearchivedrawmesh+renders/debug/pg_view3.13GB,optionalpostmesh1.82GB;keepnativeprefix/models/metadata/data/environments.

## 2026-10-27 Shared7K three-model formal result
ExistingformalCPUevaluatorcompleted75.441s/exit0/cleancode28e/1000000samplesseed0/sameprotocol. CDmm baseline40.69819792766138,409641.10463587491704(+0.998663%),1638440.64842342453039(-0.122301%). Accuracy35.777223→36.042003/35.413509;Completeness45.619172→46.167268/45.883338. F1@5cm.7799548308→.7796694056/.7804795412. Higherbudgetbeats4096butoverallnearbaseline,completenessstillworse,trainPSNRlower;no stablebenefitclaim/singlecausality. Nativecheckpointstateomissionsremainexplicit. InputsSHAagreewithverifiedarchives/9CSVrowsagreewithJSON. Userwaivesserverfullhashgate;stage1cleanupscript3.13GBprovided,serverexecutionpending.

## 2026-10-28 V2 training receipt
early formal result: cleanHEAD5fde7cb,CLIexit0/132.273029s,4models/12CSVrows. Old3metrics/hashes andGTsample/protocol exactsame. earlyAcc34.679897284020335/Comp44.80903192113209/CD39.744464602576215mm,F1@5cm.7877394904236319. CDgain2.34342888%vsC0/2.22384719%vsV1/.87854650%vsfullV2;F1gainvsfullonly.048591pp. PSNR30.850548934936526 (+.158683vsfull,+.010351vsV1). Recall@1cm slightlybelowfullV2, notallmetricimprove. Download1584files1952775223B,train/extractexit0,79exactscheduledrelocationevents325258ops. Prefixcounts differ fromfullV2 atall79events (full323721ops),despitealloldcfg_opts equal/cameras equal/Python3.10.21Torch2.7.1cu128CUDA12.8RTX4090bothclean. Causeunresolved;don'tclaimexactshared15Ktrajectoryorallgaincausedbycutoff. Noearlyregionalanalysisyet. Report2026-10-28-selector-v2-early-results.md.
User now explicitly approves trying v2_early. This supersedes the prior proposal-only status: implement one optional exclusive cutoff with0default preserving fullV2,15000fornewrun; selected candidate/geometry/ALR/optimizer implementation unchanged. Keep Dmain via newisolatedbranch, same testentry, direct7K→30KAutoDL launch. No new baseline, damping, candidate thresholds or diagnostics approved.
Follow-up user asks whether the idea is wrong. Source re-check finds pinned Sparse2DGS arguments/__init__.py defaults iterations=7000,densify_until_iter=5000 (official raw af15587 fetched); actual train.py164/168/202 gatesSGU inside iteration<densify_until_iter. Therefore donor has final2000 steps withoutSGU by default, whereas approved V2 has relocation through29900. Host actualcfg densify_until_iter15000. A proposed next single-variable check is V2-early: same16384/V1/ALR/LR/targets/interval/checkpoint, relocation only7100..14900(79events), continue ordinary optimization+V1 to30K. This is host-phase alignment hypothesis, not donor proof that15000 is optimal or late relocation caused losses. Alternatives damped relocation or candidate-rule redesign add different assumptions. No user selection/approval yet; do not implement or train automatically. Also retain: jointNCC win notGT correctness proof,regional gains do notvalidate per-candidate precision,donor uses rendered targets/sparseviews/SGU insteadADC versusourDA3/denseviews/retainedADC.
Regional analysis complete: same1MGT samples,semantic vertices/order/XYZ match,94groups/no segment conflicts;5.2043%mixed kept andREMOVE kept. V2vsV1 GTcompleteness >=1cmgain8.5644%/harm7.1609%,gaincontribution-3.606208/harm+3.103049mm(86.05%cancel). FixedV1best10% errorstratum mean worsens1.693mm,worst10% improves7.177mm;descriptive notcausal. Main wall69comp60.839→55.247mm,wall71 72.056→78.492mm;spatialmaps locate upper/boundary patches. Accuracy partition by closest GT triangle reproduces exact formal accuracies and totalCDdelta-.5516903780534314mm:wall69-.609137mm,allothersnet+.057447mm. Ceiling/rack completeness slightly worse but dualcontribution improves;do notlabel them uniformlybad. Door68 andpipes59/6 negative contributors. Camera plates selected maximalGTvisibility independentof error,96x64GTfirsthit coarse;not ALRresidual/depthJPEGmetric. Same reusable analysisartifact,CSV/JSON/6plates/spatialmap;no GTtrainingcache ormethodchanges. Full report2026-10-28-selector-v2-regional-analysis.md.
Formal V2 result now available: existing CLI exit0/76.14767s,1M seed0/fullsurface/point-to-triangle/1,2,5cm unchanged. C0/V1 exact old metrics and mesh hashes,GT samplehash/protocol reproduce;JSON/CSV9rows match. V2 Accuracy34.81328776609012mm/Completeness45.38017832686387mm/CD40.096733046477mm/F1@5cm.787253580173376; vsC0 CD -.6014648811843841mm(-1.477866%),F1 +.72987494pp; vsV1 CD-.5516903780533937mm(-1.357225%),F1+.67740390pp. Both distances and all thresholdPRF improve, but gain modest and trainPSNRlower, no statistical/cross-scene claim.229exact scheduled relocation events,227nonzero,1019840move operations(not uniqueGaussians);114common logevents moved==added. Provenancegit_dirty=true due only4 progress docs, evaluator/config unchanged; preserve identity as recorded. Full evidence in2026-10-28-selector-v2-geometry-results.md.
V2 trainPSNR30.691865921020508 vs V1-16384 30.840198135375978 and nativeC0 31.072280120849612. N1225318 vs V1 1275897 (-50579, -3.96419%); lower count does not establish improved or damaged geometry. Nonzero relocation in supplied late log proves action execution for those events, not unique Gaussian coverage or all229 events. Final30000 excluded as specified. Full training completion log received; exitcode/model files and formalgeometry still pending. Reuse unchanged mesh extraction/evaluation, no new method changes inferred from trainingPSNR.

## 2026-10-27 Current negative/mixed result interpretation

Latest implementation: user authorized V2 and routine autonomous fixes; D branch codex/second-run-selector-v2 codea114ed5. Unified30 tests24CPU pass/6native skips; fresh independent review same results/no material or minor findings. Original selector core AST unchanged after optional target-return normalization, old GaussianModel methods/ALR unchanged, renderer/CUDA/metadata/evaluator/extractor byte-preserved. Actual Camera calibration method added to test fixture after real normal producer exposed missing interface; no production workaround/algorithm change. Handoff bash/CLI checks pass; no server launch or geometry benefit yet. Approved adaptation keeps DA3/ADC/full-active scheduling and inherited baseline behaviors; efficacy/GPU await actual30K.


V2 source-to-code audit: official af15587 train.py18387B SHAa43e72a019e3fb4b95f1fec2f7c8322514c61d692fbb1ac87da11228d0cdb210 and scene/gaussian_model.py24911B SHAc5ab6a89fc43db44161a4bbe39c0099f82a760830a0cd5dbfd9a93ccb96c44b3 match local cache byte-for-byte. Referenced old per-repo source_manifest.json path is absent, so verify raw bytes directly rather than assuming manifest survives. Donor train has no densify_and_prune call: SGU substitutes for its ADC, while V2 explicitly retains inherited ADC. Host selector final ids/points_p/accepted remain aligned after validity slicing at231/233/255; use these exact rows, not extra.nonzero paired with differently ordered points. Host helper89..122 has insertion slot before line117 main render and before any this-step backward; avoid writing afterforward or after topology. Host replace_tensor resets whole moments; use a new selected-row in-place method instead. Existing metadata already serializes all option fields, so no schema/writer changes required.


User subsequently approves the joint-plane comparison plus relocation route and names it selector_V2 (V2=V1+relocation). Short spec records preserved V1 semantics/ALR and proposes full active-phase 100-step events7100..29900, same evaluated DA3 target, execution before fresh main render, selected xyz row moments/stat resets with nonselected state/Adam step retained. Full-phase timing is an explicit adaptation beyond donor's densification branch. These execution details await short-spec confirmation; production code remains c351 and no training started.


Latest user steering defers resumed-off control in favor of a materially effective correction module. Read-only execution of actual SH functions gives native 7K A=213195/1421298 and native30K A=193737/1291587 (both~15%, not GT errors). Actual xyz LR at7K/15K/30K=1.85442685e-4/5.43095711e-5/5.43095711e-6, rotation1e-3. Fixed-center prior-normal patch comparison is defined by d=-n_P^T X_G, so it changes neighborhood warp while preserving center depth; this is an adaptation, not donor-validated. Actual Sparse2DGS SGU changes xyz only, every100 in topology stage, using rendered rather than DA3 geometry. Its whole xyz replacement creates the final Parameter a second time atL336; executing the actual method confirms active parameter has no state/grad. Therefore E-only relocation needs an explicit local-state integration decision, not blind whole-optimizer copying. Both method alternatives remain unapproved; no training/source changes.

Newofflineevidence:16KpostGTsampleerrorshavegain-3.177554mm/harm+3.441720mm→net+.264165mm;>=1cmgain7.3627%/harm7.5947%,notnooutputchange. Fixedbaselineworst1%432.145→406.560mm;other99%mean41.7149→42.2401mm. z<4mmean29.1791→29.6198,z>=4m88.3434→88.1490;do notreusepreviousupper-regionfailureclaim. Morebudgetgives4.01575xlabels,similar27.1/27.2%acceptance;totalweightonly2.916%higherbutcannotisolateextraorgradient. Full-planecomparisonchangesposition+normal,whileextratermisonlybaselineALRnormalmask;actualnormal/ALRcounterexampleprovesdepth-onlyNCCadvantagecanbeunactionableforaddedALR. Thisisdesigngapnotprovenmaindriver. Raw16KCDgain.44%shrinksformalpost.12%;cannotfixbycherrypickingraw. Continuousbaselinevsresumedcandidateslacksresumed-offcontrol;nativeRNG/cameraqueueomissionsprecludeuniquecause. Noalgorithmchangesapproved/implemented.

## 2026-10-28 early regional evidence and claim boundaries
Same offline script with explicit baseline/fullV2/early names; unchanged 1M GT sampler/triangle distances. Completeness67.567s,Accuracy24.265s,both exit0;six formal means and twoCD partition sums reproduce. earlyvsC0 completeness gain-3.922091/harm+3.111951/net-.810141mm,79.344% cancellation;>=1cm gain8.5219%/harm7.4519%. CDvsC0-.953733:wall69-.589879,othersnet-.363854,wall89+.185199. Full/early pointwise GTdistance min42.038757mm (early44.809032) is outputcomplementarity ONLY,not attainablemesh/fullCD/theoreticalbound. No ceilingproof;single scene/seed+resume/prefix caveats remain. Sparse2DGS actualSGU code already NCCcompare/positionreplace/phasegate;ConfidenceMattersGS pinned ef3e071 actualtrain230..266 read,weight conf**((iter-3000)/3000),depth/directnormal/depthnormal. ExacttrainSHA a0f6f690b3ee81a4782b6374746f4890fff664fd0661efae92993105a302fcab. Newreport2026-10-28-selector-v2-early-regions-and-claims.md;no arbitrary publicationpercentbar/no independentmaincontribution established.No training/methodchanges.

## 2026-10-28 DA3 position/normal and A-E-M-R evidence
Fixed5views,151832GTfirst-hit imagegridpoints,stride4,r2sameproduction normal/depthloader;11selectedinputhashes equaltrainmanifest. Pairedv2audit54.62s exit0/125rows/source1514fbbb9cd2ff54f9fe76291d970c0656394dda65d65598bc7502c4d1aeb667. SameC0hitdomainprior/C0zMAE(mm):wall89 58.748/45.976;structure8 130.830/43.977;pipe59 114.915/128.190;wall71 61.087/64.204;wall69 32.802/28.897;ac12 65.749/53.604. Medianpriornormaldegrees9.354/9.883/10.060/5.123/7.929/11.323. DSC06528highestconfquartile:pairedprior98.174mm,C027.137mm,normal5.709deg,priorcloser9.69%. Actualearlyconfquantileratio0.0 means finiteconfpass;notrelativebenefitprob. Not acceptedcandidateprecision/causalproof;GTposemesherrors/finalTSDFdomain caveats. ActualAoriginalSH,Eselectiononly~A,ALR=A|E,actualmoveM=Eonrelocationevents;futureindependentRrequiresallowAintersection AND explicitselect/movefrequencies toavoidchangingALRfrequencyunnoticed. Dpositiveuniformscale leavesnormalizedproductionnormalsunchanged(testrealproducer). PinnedDPGSnormal-edgeanchorpropagation andVCRdetachcosweights realsource re-read. DatasetvaliditysupportedScanNet++officialandInDepthWeTrust5.1;targetcandidateauditbeforechoosingconditionalfix,no newtraining. Report2026-10-28-da3-failure-regions-and-action-selection.md.
