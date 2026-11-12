# AmbiSuR Core 实施计划状态

> **2026-11-12 当前批次已完成：** 下载核对、原四来源评分一次成功、支持／来源／区域分析与独立终审完成。新最大角策略仍不支持分区loss或30K；见docs/research/2026-11-12-four-source-routing-results.md。下方Core与旧任务仅历史，不恢复实施。

> **2026-11-10 下一批对应检查规格已准备，待审阅：** 用户同意继续源码/接口/资源预检。拟RoMaV2固定95c996...、4参考/8原支持对、原65536像素；先用固定图像对应残差比较现有位置，再复用匹配做COLMAP式固定相机三角化。已读供体match/sample/半像素/特征加载/相关算子以及COLMAP3.11.1线性求解，实际12图header与供体坐标函数通过。权重未下载、依赖未安装、网络/GT未运行、生产代码未改。服务器torchvision0.22.1低于供体>=0.23、本地仅1.9GB显存空闲；需明确隔离运行环境。规格docs/superpowers/specs/2026-11-10-observed-correspondence-probe-design.md。

> **2026-11-10 非GT失败证据调查（用户批准继续，已完成）：** 复用现有17视角/冻结抽样/生产warp与NCC/GTQueries，只读检查DSC06348、DSC06669、天花板及成功对照。先保存不依赖GT的每支持代价、相关符号/纹理、投影与模型深度分歧、位置代价曲线和观测夹角，再按既有GT定义描述条件收益。原正式评分不覆盖；新产物放analysis/depth_source_failure_evidence_20261110_v1。无GT调门槛、训练mask、loss/训练/renderer修改；不能将探索性分组称已验证准入。步骤：输入/复用接口核对→冻结证据定义与计算→原查询评价/原分数复现→可查看报告及局限。

> **本批已完成：** 17原probe/统计精确复现，非GT特征及分位数先冻结；17参考+34支持GT网格各一批查询退出0、无失败重跑。发现小基线弱位置分辨与大基线错误光度偏好两类不同条件；219条双支持天花板仍91.78%伤害，简单门槛不足。独立终审无实质问题，生产源码不变。报告docs/research/2026-11-10-depth-source-failure-evidence.md；下一批真实对应/支持选择或新规则仍待讨论，不自动训练。

> **2026-11-10 最新讨论（优先于下方历史计划）：** 用户认可继续查明非GT证据条件与深度来源质量的关系，并讨论类似IndoorGS的分区监督；本轮仅核对论文、公开代码与现有17视角结果，没有新分区规则、loss或训练实施授权。下一项仍是DSC06348、DSC06669和天花板误接受的观测证据调查。

> **当前（2026-10-26）：** 用户已批准原生C0零源码改动的新30K/seed0baseline，内置选项保存7Kcheckpoint，旧output保留。先交付启动命令；后续二周目4096/16384共享该模型/Adam起点，承认原生RNG/相机队列缺失。不追加基线模块或cloudsmoke。

## Goal

在不混入 Supporting 模块的前提下，完成可复现、可消融的 AmbiSuR Core：观测校准、双可靠性、五状态仲裁、参数级梯度路由和 Gaussian 生命周期，并通过 G0–G2。

## Current Phase

Phase 0：新项目接管与 baseline 锁定。

## Phases

### Phase 0：接管与规格核对
- [ ] 读取 AGENTS、交接说明、baseline audit 和最终设计稿
- [ ] 记录实际 baseline commit、环境和数据路径
- [ ] 核对代码能否提供所有公式输入
- [ ] 生成详细 Core implementation plan
- **Status:** in_progress

### Phase 1：E0 与测试基础
- [ ] 锁定 C0 baseline tag
- [ ] 建立 CPU 单元测试和 GPU smoke 入口
- [ ] 验证所有新增开关关闭时等价
- **Status:** pending

### Phase 2：D0 影子证据
- [ ] 实现 A/S/N 统计
- [ ] 实现双可靠性、K、Delta
- [ ] 实现五状态和日志，但不改变训练
- [ ] 通过 G1
- **Status:** pending

### Phase 3：C1–C3 粗粒度执行
- [ ] C1 需求门控
- [ ] C2 整组强制仲裁
- [ ] C3 增加 Abstain
- **Status:** pending

### Phase 4：C4–C5 参数路由
- [ ] 分离 g_B 与 g_P
- [ ] C4 参数组直接加权
- [ ] C5 冲突投影
- **Status:** pending

### Phase 5：C6 生命周期
- [ ] Normal/Confirmed/Repair/Protected/Quarantine/Probation
- [ ] 持续计数、冷却和两阶段提交
- [ ] optimizer/EMA/state 拓扑迁移
- **Status:** pending

### Phase 6：G2 主验证
- [ ] Tool Room 与 Utility Room 快速确认
- [ ] Baseline/Core 三 seeds
- [ ] 效率、显存、几何与外观守门
- **Status:** pending

### Phase 7：Supporting 决策
- [ ] 仅在 G2 通过后新建独立计划
- **Status:** pending

## Errors Encountered

| Date | Error | Attempt | Resolution |
|---|---|---:|---|

## Scope Guardrails

- 当前计划禁止实现曝光、Ray-Color、Ray-Normal、自适应截断。
- GT mesh 不进入训练。
- C0–C6 每阶段必须对应独立 commit 和验证 tag。
- 服务器不修改代码。



## 2026-10-22 Clean second-run migration
User approves c0-baseline branch plus selector-only port and necessary seed/metadata reuse. Old records above are historical. The original dirty checkout stays unchanged. No baseline/GT/storage operation.
- [x] Adapt tests and observe clean-base RED.
- [x] Minimal selector/metadata port with no old scientific dependencies.
- [x] Local validation, baseline-boundary check and one review; AutoDL pending.
- [x] Clean-base RED and minimal selector/metadata port.
- [x] Local scoped tests, compile/whitespace, baseline source boundary and one review.
- [x] AutoDL native validation: user receipt at3d85faf onRTX4090,23tests passed/zero skips. Separate real-data30k discussion remains pending.

### 2026-10-23 Approved version handoff
- User confirms next batch and reports GPU server rented. Finish local commit/push of reviewed selector-only migration and provide exact server commit/extension/unified-test commands; no method changes or baseline rerun.
- Fresh precommit validation:23 tests,18CPU pass/5native skips;8 code files compile; diff whitespace and unchanged baseline source boundary pass. Prior whole-migration review remains applicable; only server receipt documentation changed since review.
- [x] Commit/push exact migration; feature commit d979c070925a9dc9a5f5ce3a5563de56614fa192 verified against origin ref with a clean local tree. Documentation follow-up records this receipt.
- [x] User-operated AutoDL native test run at3d85faf95951b4806eee18498841e66551b93320:23tests passed in2.875s,zero skips.
- [x] Discuss first real-data run and resolve input: user supplied existing c0_candidate_r2_30k_seed0_20260902T082615Z/colmap_undistorted. Prepare thin per-run view from it (link read-only inputs, independently copy aligned COLMAP) and new output; tested training commit remains3d85faf. Fresh real-parser comparison confirms frozen baseline cfg_args/cfg_opts fields unchanged except input/output paths and new seed/selector/budget fields.
- [x] User executed Tool Room/r2/seed0/30k/budget4096 launch and returned active logs through23760; final completion and reconstruction benefit remain pending.

## 2026-10-23 D drive primary workspace takeover

- [x] User confirms30k is running and returns23760/30000 with active selector records; launch achieved, final completion/benefit still unverified.
- [x] Read-only inventory and user-approved migration; preserve24files,54ignored historical records,complete bundle and original branch history.
- [x] User completes C detach after tool permission denial; D switches normally to second-run at3775135, C retained detached/clean.
- [x] Verify1697tracked files,8code/syntax files,192ignored hashes; unified local tests18pass/5native skips. No data/baseline/server modification.
- [x] Update primary-directory rules and recovery/handoff documents; old Core preserved on archival branch.
- [x] One scoped review passed with no Critical/Important/required Minor; production diff empty.
- [x] Documentation published as3d27ba2 with matching remote refs; D clean on second-run,C detached3775135/clean,1699tracked files,8core hashes and192ignored files verified; Git connectivity and24-file original backup rechecked. No production change.
- [x] Receive final30000/30000 log with Gaussian/app save and Training complete; compute exact training-log deltas against fixed baseline.
- [x] User returns exitcode0,30K PLY307M and app.pth15K; approves new-model mesh extraction and comparison. Training completion confirmed.
- [ ] User-operated existing extract_general.py atiteration30000 with frozen baseline5.0/0.005/4.0/2; output only inside new method run. Receive extract exitcode/log and raw/post meshes.
- [ ] Define matching ScanNet++GT coordinate/observable domain and use applicable existing distance/metric primitives for geometry comparison; no DTU/TnT domain substitution or old detector-label metric. C cleanup remains unapproved.

## 2026-10-23 Formal mesh evaluation
User authorizes shared surface evaluator plus AmbiSuR PRF; confirms1/2/5cm, main5cm.
- [x] Read existing DTU/TnT and archived point-to-triangle helper; write scoped spec/plan.
- [ ] TDD shared evaluator, focused native integration, review, publish and AutoDL handoff.

- [x] Unified evaluator implemented;8/8 native tests pass/zero skips, Python3.10 grammar and whitespace pass; one review seed finding fixed RED→GREEN.
- [ ] Server native acceptance and first formal Tool Room metrics using existing meshes; no baseline rerun.

- Evaluator published:4010fd01e4b5bbe651ac4e221e00532bf99f4ff9, origin branch hash matched; D clean after push. Server native tests and formal GT metrics pending, no training changes. Closing receipt is documentation-only; server can pin4010fd0.

## 2026-10-23 First formal geometry receipt
- [x] Server8/8 tests/zero skips; formal paired mesh evaluation completed, full CSV received.
- [x] Preserve mixed outcome: F1@5cm+0.1151pp,CD+13.0463%,Accuracy/Completeness worse; no overall benefit claim.
- [ ] User self-downloads full run/evaluation; original files retained. Next focused analysis discussed before any code/training change.

- [x] Initial4096-budget hypothesis analysis: actual115 log records and producer/consumer source checked, competing coverage/selection/action explanations recorded; no causal conclusion.
- [ ] Discuss one narrow follow-up after download completes; no new method or training authorized this turn.

- [x] Verify downloaded new-run/evaluation identities and numerical reports;compute final-snapshot3-view candidate pools and inspect existing debug images.
- [ ] User downloading actual formalC0 baseline;paired spatial comparison waits for matching asset. No rerun or method changes approved.

## 2026-10-23 Actual C0 paired analysis completed
- [x] Verify actual C0 SHA and actual C0/new configs, logs, input/cameras/neighbors.
- [x] Reuse original evaluation core/identical GT samples for offline height/tail diagnosis; original means reproduce to4.62e-13mm, official files unchanged.
- [x] Separate raw/post effects, check GT first-hit visibility and existing same-view products.
- [x] Establish pre7001 distribution/log divergence; correct paired PSNR to+0.0286911dB on5training views; preserve causal uncertainty.
- [x] Record artifacts/handoff; no method, training or baseline edits.
- [ ] Discuss user-operated read-only1219-file input hash check against C0 manifest; noGPU/GT/baseline rerun.
- [ ] Decide next single-variable experiment after input/early-trajectory investigation;4096 remains a hypothesis.

## 2026-10-26 Requested baseline-lineage and line-by-line source audit
- [x] User receipt:1219 input files,0missing/0mismatched;current view matches C0 manifest.
- [x] Verify C0/feature/current parent graph and compare all changed production lines.
- [x] Compare C0 to locally recorded official upstream;separate baseline adaptations from first-run additions.
- [x] Trace imports/config defaults/native source,metadata and feature-off path;report any residual or unintended behavior without fixing it.
- [x] Record concrete evidence and limitations;no method changes/new experiments.

Audit completed:directC0parent,no old62e9117ancestor;officialmain88d64054 verified,8Python AST/4native token equivalence;all changed production lines reviewed,protected1630files exact. Source audit/JSON saved;fresh18CPU passed/5native skipped. Oldpyc retained,no source/native residue loaded. No next experiment authorized by this audit;early trajectory cause remains open.

## 2026-10-26 Approved budget16384 direct30K
- [x] User explicitly selects16384 and requests direct30K,no new smoke/safety framework.
- [x] Reuse fixed3d85faf training,Tool Room/r2/seed0/same inputs and schedules;prepare separate writable aligned view and output without production edits.
- [ ] User launches16384 run and returns train PID/log;no baseline rerun.
- [ ] Complete30K,then reuse same extraction and formal geometry evaluator against frozenC0/4096 outputs.

User steering before launch:questions uncontrolled pre7K divergence and fairness of7K resume. Verified both downloaded runs have7K PLY but no7Koptimizer checkpoint. Hold launch handoff;discuss exploratory single30K versus shared7Kprefix/two23Ksuffixes(total53K). No checkpoint implementation,training or reset approved yet;existingbaseline remains read-only.

## 2026-10-26 Shared selector-free prefix selected
- [x] User agrees a selector-free7000prefix must be saved;write short concrete state/iteration contract.
- [ ] Review minimal save/restore/stop scope in docs/superpowers/specs/2026-10-26-shared-7000-prefix.md before production edits.
- [ ] Implement only approved boundaries,reuse existing tests;no separatecloudsmoke.
- [ ] Produce actual7Kprefix onAutoDL with total30K schedules andselectoroff;no existingbaseline output replacement.

## 2026-10-26 Approved native-C0 baseline execution
- [x] User approves nativeC0/no baseline changes and existingcheckpoint limitations;supersedecomplete-state/stopdraft.
- [x] Preserve currentdocs;create baselinebranch directly atd6f15c8 and prove whole-tree equality.
- [x] Commit/push docs-onlyresearchupdates and baselinebranch;944f676 research/d6f15c8baseline verified,no newworktree,forcepush or C0tagchange.
- [x] Handoff directnative30K launch withcheckpoint7000/privateinput/newoutput/logexitcode;no test reruns or CUDA rebuild.
- [x] Receive user-operated baseline launch/completion and7Kcheckpoint assets;candidate restoration work stays onsecond-runbranch.

## 2026-10-26 Native baseline completion / minimal resume consumption
- [x] User returns exactd6f15c8checkout,launchPID11962 andfull30K Trainingcomplete log(50:21,PSNR31.07228012,N1291587).
- [x] Receive train.exitcode0 and7Kcheckpoint1074511797B/App14349B/30KPLY320315108B listing.
- [x] Fix onlysecond-run Appweights parent/exactiteration via existingcheckpointAPI;follow scopedplan2026-10-26-native-checkpoint-resume.md.
- [x] ExistingCPU serialization/Adam integration case,unifiedsuite24run19pass/5native skips andscopedreviewpassed;no separateGPUsmoke,baselineedits orselectorchanges.

- [x] Fixedcandidatef41dac0b and sequentialpairedlaunch guide:actualCLI andBash-n verified;no newGPUtest.
- [ ] User-operated4096/16384 shared7K continuation;each23000steps,thenreusemeshcomparison.

## 2026-10-26 Server checkpoint loading compatibility failure
- [x] Receivef41pairPID14168/firstarmloadfailure;userauthorizesautonomousfix.
- [x] TraceactualNumPyradiusproducer,correctexistingfixture,observeRED→GREEN;onlynativeconsumerweights_only=False;19pass/5native skips/reviewpassed.
- [x] Fixedcommitc3510b0andfreshv2pairedrestartguideprepared/validated;publishonsecond-runbranch,preservefailedf41/baseline.
- [ ] Receiveuser-operatedserverrestart/completion;noSSHaccess.

## 2026-10-27 Shared7K paired training receipts
- [x] Receive4096full23Kcontinuationlog/global30Kcompletionandsaves;checkpointloadcompatibilityfailureovercome.
- [ ] Confirm4096exitcode/nonemptymodels and16384automaticqueueprogress/completion.
- [ ] Afterpaircompletion,reusemesh extraction/settings andformalpairedgeometryevaluationfornativeC0/4096/16384;no newmethod/diagnosticframework.

## 2026-10-27 Three-model mesh comparison handoff
- [x] Receive4096exitcode/artifacts/16384automaticstartandfinal23Kcompletionlog.
- [x] Userauthorizesnewbaseline/4096/16384mesh extraction;reuseexistingextractorandparameters,actualCLI/Bashsyntaxchecked.
- [ ] User-operatedqueuechecks16384exitcode/artifactsandextractsthreeserially;receivequeue0/nonemptymeshes.
- [ ] Userdownloadsfullnewbaselineandpairoutputs;localformalgeometrycomparisonviaexistingevaluator.

## 2026-10-27 Download verification before storage cleanup
- [x] Locateuserprovidedbaseline/pairwrappers;alltrain/extract/queueexitcodes0andcriticalfilespresent.
- [x] Read-onlyverifyPLYpayloads/PTHzipCRC+CPUload/rendercoverage;4749filehashesandsharedprefixSHA match.
- [x] Archiveverification.json/twofullmanifestsoutsidesourceoutputs;write2026-10-27-shared7k-download-verification.md;serverfullcomparisonpending.
- [x] Userwaivesfullserverhashcomparisonandauthorizesfastcleanup;deliverexact3.13GBstage1scope/script,retainmodels/postmesh;serverexecutionreceiptpending.

Localverificationcomplete;noformalgeometryevaluationorservercleanupperformed. Nextserverreadonlymanifestcomparisonandconcretecleanupconfirmationareseparatefromresearchanalysis.

## 2026-10-27 Direct geometry analysis after hash waiver
- [x] Reuseunchangedevaluator/protocol/verifiedinputs forbaseline4096/16384;exit0,clean28eidentity,JSON/CSV/inputSHAverified.
- [x] Recordmixedresults:4096CD+0.998663%,16384CD-0.122301%vsbaseline;no clearoverallbenefit.
- [x] Receiveservercleanupstage1df/output:50G/42Gused/8.3Gavailable84%;noagentSSHaccess.
- [ ] Discussnextboundedanalysis/methodscope;no automaticnewtrainingorarchitecture.

- [x] Stage2completeuserreceipt:50G/41Gused/10Gavail81%;duinventoryreceived.

- [x] Stage3receipt:50G/37Gused/14Gavailable74%;D0/G0/Utilitymodels/auditrecordsretained.

## 2026-10-27 Result mechanism analysis
- [x] Actualsource/donorread,115logpairs,6reusedkernelpost/rawqueries/sameGTsample,parallelplanecounterexample;factsandcausalboundariesrecorded.
- [ ] Discussresumed-off23Kcontrolandpotentialnormal-onlycandidate;no newmethodcode/trainingscopeapprovedyet.

## 2026-10-27 User prioritizes a stronger correction action
- [x] Accept explicit deferral of resumed-off control; do not make it a prerequisite for method exploration.
- [x] Re-read actual SGU position update, training schedule and optimizer replacement; verify fixed-center normal homography mathematically.
- [x] Execute existing SH selectors on native 7K/30K artifacts and existing LR function; save aggregate evidence without GT access.
- [x] User chose V2=V1+relocation; subsequent implementation authorization recorded below.

## 2026-10-27 selector_V2 route selected
- [x] User selects V2 = V1 + relocation; retain original ALR and V1 comparison, defer resumed-off control.
- [x] Write short V2 spec with explicit proposed schedule, pre-render execution and selected-row state handling; no production edits.
- [x] User confirmed source-based concrete changes; V2 branch and implementation plan created.

## 2026-10-27 V2 source-to-code adaptation requested
- [x] User basically accepts V2 intent, asks for donor-source-based concrete code changes before implementation.
- [x] Re-read pinned Sparse2DGS train/update_points/world-unprojection/optimizer code and current selector/render/training/densification/metadata/test interfaces.
- [x] Fetch two pinned official raw files in memory and confirm byte equality with read source cache; no dependency installation or donor execution.
- [x] Deliver concrete function signatures, tensor contracts, insertion order and donor-to-host differences in code-adaptation report; no production code edits.
- [x] User approved concrete five-file changes; branch created and implementation completed.

## 2026-10-27 selector_V2 implementation authorized
- [x] User approves source-based concrete changes and authorizes autonomous routine fixes while away.
- [x] Create D branch codex/second-run-selector-v2 at ef333bb; preserve original V1/C/baseline.
- [x] TDD one five-file implementation using existing test entry, local CPU verification, one independent review.
- [x] Commit/push fixed V2 code and deliver existing-prefix direct30K AutoDL handoff; no local GPU/server execution claim.
- [x] User returns actual V2 Training complete log (2026-10-28): global30K, relocation active, 53:07, train PSNR30.691865921020508.
- [x] Downloaded V2: train/extract exitcode0, launch settings and nonempty Gaussian/App/raw/post mesh checked;229 relocation log events.
- [x] Run unchanged formal mesh evaluator on nativeC0/V1-16384/V2 (exit0,76.15s); old-model metrics/input hashes/commonGT/protocol reproduce exactly,9CSV rows agree with JSON.
- [x] Record V2 CD40.096733mm/F1@5cm.787254: CD -1.47787% vsC0/-1.35722% vsV1, both distances improve; trainPSNR decreases. Limited single-scene/seed result, next scope requires discussion.

## 2026-10-28 User authorizes existing-output regional analysis
- [x] Reuse exact common GT sampler and triangle-distance producer for paired post-mesh regional completeness changes; preserve formal domain and results.
- [x] Validate existing semantic annotation mapping; keep unknown/mixed/REMOVE in full statistics, avoid inventing labels.
- [x] Use recorded cameras/GT first-hit visibility to select representative gain/loss views; inspect six existing RGB/normal comparison plates and spatial maps.
- [x] Add fixed-V1-error strata, spatial bins and production Accuracy partition; reproduce all six formal directional means and exact globalCD difference decomposition.
- [x] Record quantitative partitions, view-selection rules and limitations; next experiment remains discussion only, no training/method changes.

## 2026-10-28 Follow-up research decision discussion
- [x] Recheck donor actualSGU gate and missing pinnedarguments:7000training/5000cutoff; compare host15000topology andV2through29900.
- [x] User approves v2_early: only relocation7100..14900; ordinary V1/ALR optimization continues to30K, same native7K/budget16384/seed0/LR/interval.
- [x] D isolatedbranch and shortspec/plan;3-code-file changea52ab904, real boundarytestsRED→GREEN,31tests25CPU pass/6native skips; all otherselector functionAST unchanged.
- [x] Verify pinnedlaunch CLI/Bash, all existingV2optimizationvalues identical, onefreshreview nofindings/31tests25pass6skip; codea52ab904/handoffdocad8ecc7 ready.
- [x] Publish isolatedbranch: initialremote78312ce matcheslocal via ls-remote; fixedcodea52ab904/fixedlaunchdocad8ecc7 available. Deliver shortpinned-doc launcher.
- [x] User returns earlyglobal30K Trainingcomplete log:58:27,PSNR30.850548934936526,N1226364,late selector active.
- [x] Downloaded early:1584files,train/extractexit0,nonemptyGaussian/App/raw/postmesh,exact79events7100..14900 andoriginalextractionsettings verified.
- [x] Four-model unchanged formal evaluationexit0/132.27s,old3scores/hashes/commonGT reproduce,12CSVrows agreeJSON. earlyCD39.744465mm/F1@5cm.787739,CDgain2.34343%vsC0/.87855%vsfullV2,trainPSNR30.850549.
- [x] Record prefixtrajectory caveat:79movedcounts differ beforecutoff despite equaloldoptions/cameras/recordedruntime; no causaloverclaim or unapprovedretraining.
- [ ] Discuss nextscope:earlyregionalgains/harms notyetmapped; do notchange method orstarttraining automatically.

## 2026-10-28 Early regional interpretation and publication scope
- [x] Reuse offline regional program with explicit reference/candidate roles; verify role selection on real formal report and preserve prior outputs.
- [x] Reproduce formal directional means, map early gains/harms, and quantify descriptive output complementarity (not a theoretical bound).
- [x] Verify closest primary literature/source overlap and record defensible method/paper claim boundaries.
- [x] Deliver findings and next decision for discussion; no training or method changes.

## 2026-10-28 User requests failure-region/DA3/ALR linkage analysis
- [x] Read actual A/E/relocation/ALR interfaces; distinguish stage completion from established gain.
- [x] Check local prior identity and compare existing DA3 depth/derived normals with visible GT surface on a small fixed view set, using production geometry functions, no training output.
- [x] Read primary dataset/method sources and available actual source for relevant borrowing; map conditional interventions to measured cause.
- [x] Discuss whether to freeze/transfer or first change method; no training/method change without user decision.

## 2026-10-28 Approved static candidate check; server OFF
- [x] Preserve prior analysis documents in2c2fb1f; create D isolated branch codex/selector-static-audit, keep all training branches/output.
- [x] Scope existing native7K plus C0/fullV2/early30K,5 illustrative+12 systematic views, budget16384/seed0; no15K rerun or historical replay claim.
- [x] One actual-selector capture/CPU score entry, existing test entry, production metadata/geometry reuse; training/selector/renderer unchanged.
- [x] Real selector→NPZ→GT/BVH/normal→CSV/JSON tested.37 tests:30CPU pass/7native skips; Python3.10 syntax and CLI verified. Review FP32 false mismatch reproduced RED→GREEN, literal production targets exported, re-review no blocker.
- [x] Publish pinned capture/score handoff; first remote13e12874ee5e92e671a5db31933d05cf8170f328 matches local via ls-remote.
- [x] User-opened server real CUDA capture and subsequent local scoring completed2026-10-29; never start server automatically.

## 2026-10-29 COLMAP text compatibility
- [x] Reproduce source-format omission; real reader regression RED→GREEN, binary priority/order/layout preserved. Audit-only fix9603ef6.
- [x] Actual local406 poses/intrinsics match C0;46 existing tests39pass7native skip, Python3.10 parse, independent narrow review no blocker.
- [x] Server fixed capture static_v2 exit0/68arrays decoded, local score exit0/2815rows; preserve failedstatic_v1, no training/data conversion.

## 2026-10-29 Real candidate outcomes
- [x] Read downloaded capture identity/68NPZ/counts without changing inputs; actual4snapshots/17views/clean9603ef6/exit0.
- [x] Execute unchanged score on localGT, clean cacb570/exit0;2815JSON/CSV rows agree, GTsha matches formal geometry evaluation.
- [x] Summarize view-group/accepted-rejected/object evidence, proper denominators and nearest/ray-z/normal boundaries; preserve real outputs.
- [x] Independent claim/numerical review noCritical/Important; actual68NPZ/2815rows and denominators corroborated, no repeatGT or significance claims. Handoff complete; next method/training remains discussion-only.

## 2026-10-29 Literature-grounded candidate redesign research
- [x] Screen24 named2025–2026 works using primary venues/arXiv/author/publisher sources; explicitly separate main/workshop/preprint/author-page venue evidence, no exhaustive-coverage claim.
- [x] Inspect12 actual official source repositories/18 source files; preserve commits/line ranges/fileSHA in output candidate_literature_20261029_v1/source_read_manifest.json. Fifteen localPDF texts extracted; not all full-read, external trainers not run.
- [x] Map target-source/conditional-NCC/position action to real train.py and candidate_selector interfaces; preserve A/ALR/ray/renderer contracts. No algorithm implementation.
- [x] Rank anchored normal-guided candidate targets first, conditional position comparison/ambiguity curves second, relative-structure backup; distinguish DP-GS depthloss from host hardrelocation and code/paper coordinate differences.
- [x] Independent narrow evidence/source review: noCritical/Important;12repo/18fileSHA and DP-GS/APDe/MP-SfM/host action boundaries checked, no GT/training rerun. Deliver research for discussion; next route notyetapproved.

## 2026-10-29 selector_v3_spread: source-to-interface proposal only
- [x] User chooses depth propagation candidate generation first, names selector_v3_spread; conditional fixed-normal NCC is deferred. Must report proposed code/interfaces before implementation.
- [x] Re-read pinned DP-GS propagation, RGB confidence and5x5 normal-edge producer; host candidate/training/ALR/capture interfaces. Current production files identical to approved early a52ab904; later audit adds tests/entry only.
- [x] Draft specific donor-to-host proposal: rendered-surface RGB/roundtrip anchors, firstsupport/102400pixel cap;5x5/10step/.6/.99/.01; explicit Kinv and invalid-transfer adaptation, proposal/prior separation, baselineA fallback; one shared module/fiveexistingfiles.
- [x] Independent narrow feasibility review noCritical/Important; hostALR/depthconsumers, mixedrenderplane anchors, donor differences, callback/cache andcapture passthrough checked. Oral proposal ready; no product code/branch switch/GPU/GT/training. Specific defaults await user approval.

Research-only read correction: initially looked for Codex reference under brainstorming/references (missing); actual using-superpowers/references/codex-tools.md read. No dependency/source modification.

## 2026-10-30 Approved selector_v3_spread implementation
- [x] User approves concreteadaptation; annotation1/3 clarify no proposal skipsnewmatch/E, preservesA; annotation2 propagatedlossfutureonly. D branchcodex/selector-v3-spread from43e2ea0, spec/inlineplan saved.
- [x] TDD sharedmatcher/source/proposal and commonproducer: missingAPI/module/RGB shape RED->GREEN; tiltedplane/noanchor/boundary/grazing, exactfrozengraymatcher, RGBchannelmean/NCCscale/source sizes/RNG checked.
- [x] Realtraining/capture integration RED missingproposal->GREEN; producer->selector->literalworldtarget->observer, rawDA3/cache/mainrender order;56tests48pass8native skips,3.10/CLI/frozenAST checks pass.
- [x] AppropriateCPU checks and independentreview complete, fixedcode8b1def9; native8unavailable locally, no GPU/benefit claim. Handoff0b6b255/newbranch published, ls-remote matches, clean; Bash/config checked, no automaticservertraining.
- [x] 2026-11-02 userserverreceipt17viewcapture/exit0/manifest; logcounts summarized, OMP0fallback1 noted without rerun. Nativeunittestreceipt/downloadvalidation/GTquality remain pending, nextsame-score comparison not30K.

## 2026-11-02 Downloaded spread scoring and comparison
- [x] Decode17NPZ using productionvalidator, verifyclean8b1def9/exit0 andsame native7k inputs/options/views/supports. All38005rows exactsame currentgeometry/rawnormals on oldIDs; receipts saved outsidecode.
- [x] Run unchangedscore once, clean6de5c44/exit0/17.298s,462JSONCSVrows agree, GTsha matches oldformal. No repeatedGTscore ortraining.
- [x] Weighted comparison usingcount/visible_count/normal_count, accepted/rejected and5/12 groups, objects/views/overlapcounts. Positivecandidate signal withselectiondomain/coverage/localharm caveats; report saved.
- [x] Independent narrowresultreview foundnomaterialissue; actual17pairedNPZ/metadata,462CSVJSONrows, allweightedtables/object/view/coverage/intersection audited withoutGT/rerun. Finalreport/ledgers saved; no newmethod or30K beforediscussion.
- Gitwriteerror: add/index andCOMMIT_EDITMSG permissiondenied, ACL explicitDeny; no permissionchange/bypass. Pre-score pendingnotespatch preserved outside, source restoredclean beforeGT; finaldocupdates remainuncommitted. Get-Content Raw+TotalCount invalid corrected toTotalCount only. No scorefailure/retry occurred.

## 2026-11-02 Approved spread30K handoff
- [x] User approves unchangedsource route fromnative7kto30K/seed0/r2/16384/early15000; reusepublished8b1def9, no sourcechanges/baseline/smoke.
- [x] Adapt originalearlylaunch withonealgorithmsourceflag/newview+output/OMP1; bothouterBashblocks+innerbody parse, actualtrainCLIand alloptimizationoptions matchcompletedearlyexceptsource. Fixedguide saved; no localGitpermissionbypass.
- [x] Userreturns30Kcomplete/81:17/exit0/PSNR30.8618515/N1279368; trainingreceipt recorded, notgeometryclaim.
- [x] Downloadextractexit0/model+rawpostmeshnonempty/headers/79moves verified; originalTSDF5m/.005m/4/2, no re-extraction ornewtraining.

## 2026-11-02 Spread finalmesh evaluation
- [x] Verifydownloadedtrain/extractexit0, models/meshheaders,79events/60273moves, cleangit8b1def9, actualopts onlyspreadsource vs early;406cameras/supports identical, sameextract/protocol/evaluatorsource5fde7cb. Receipt outsidecode; noGTqueryyet.
- [x] RunexistinggeometryCLI onceC0/early/spread, exit0/evaluator67.553s/wrapper68.666s, frozen1Msamples/seed0/fullsurface/1,2,5cm; no failure/retry/newtraining.
- [x] C0+earlyoldmetrics/inputSHA andGTsample/protocol/origin exactreproduce;9JSONCSVrowsagree. V3CD41.161527 worseC0by1.13845%/early3.56543%, reportnegativewithoutcausaloverclaim.
- [x] Independentnarrowreviewno materialissue; exactcontrols/protocol/GTsample/9CSVJSON/percentages/trainlogs andmechanismboundaries corroborated withoutGT/rerun. Report/ledgers+artifactcopy saved, docsremainpendingGitpermission, no methodorparameterchanges.

## 2026-11-02 Approved V3 failure analysis
- [x] Reuseexistingregionalcorecompleteness+accuracy-only exit0, exactsixmeans/GTpoints/CDdecomposition; sameannotations/newoutput. Existingraw2meshdiagnostic originalevaluator exit0, rawgap81.92%postgap; no algorithm/training/TSDFchanges.
- [x] Trace actualhostCUDA/selector/relocation/ALR/depthloss and pinnedDP-GSdc1ce835 source. Prioritymismatches documented; actualCPUpropagationlimits+analyticalmix-depthcounterexample explicitlynotreal-scenecauseproof.
- [x] Quantify17NPZangles/viewagreement,115logE/ALRmass,regions/strata/raw-vs-post. Mainnormalangle39.43deg despitexyz-onlyaction; oneCLIflagchangesEandALRtoo. No newGPUcapture, noGTfailure/retry.
- [x] Independent read-only review found no material issue: regional sums/strata, actual17NPZ angles/support counts, hostCUDA/action/loss and pinned donor boundaries verified without GT rerun. Finalreport/ledgers saved; next method remains discussion-only.

## 2026-11-02 Five-remedy discussion (not implementation approval)
- [x] Re-read host selector/propagation/train and pinned APDe/DP-GS/MP-SfM/normal-integration production functions; verify primary paper pages. CVF PDF opens403; use available arXiv primary HTML and existing pinned source, no repeated download.
- [x] Formulate separate remedies: fixed-current-normal action gate; surface-level soft depth update; exclusive local depth target; separate ALR/position decisions and effective sampling; anchored discontinuity-aware propagation. Prioritize a single action-gate experiment, keep surface-loss route an alternative and anchor solver later. No source changes, tests, GT evaluation or training authorized/performed.

## 2026-11-03 Approved valid-pool + position-check implementation
- [x] User approves summarized hard-relocation repair package; D branchcodex/selector-v3-position-check created normally from6de5c44, old dirty Markdown preserved, no ACL changes. Spec/inlineplan saved; no renewed approval needed.
- [x] TDD existing shared selector/CLI/real NCC and effective sampling; frozenV3 default-off regression exact. Added8meaningfulCPUcases, original6featurecases RED then GREEN.
- [x] Thin observer actualsubsettarget validation, minimal eventlog, original native-test reused withbothflags; actualCPUtraining helper checks71/15K cutoff, rawprior and sourcecache. NativeGPU8skipped locally.
- [x] Whole-suite65:57PASS/8native skips, Python3.10/source scope and independent finalreview9newCPUcases passed after supportmaskfix. Production6172bc45 committed; bothouter Bashblocks/inner and actualCLI guide passed. Branch firstpublishedHEAD4ef887e matchesremote/cleanstatus; no auto serverrun.

## 2026-11-03 Position-check training receipt
- [x] Read actualuserattachment: traincomplete/exit0/75:20/N1277504/PSNR30.8500263. Parse22tailrows, allsampled==proposal_usable; no global/geometry overclaim.
- [x] Append originalmesh extraction withonepass full79event andcounter/file/version checks; no newmethod/model orsmoke. LocalBash validation before delivery.
- [x] Serverextractionexit0/rawpostmeshreceipt andlocalactual6172download verified, reusedformal4modelmesh evaluator; no newbaseline/training.

## 2026-11-03 Downloaded position-check formal geometry analysis
- [x] Userauthorizes analysis. Namedpath isold8b1def9, foundactual6172bc45 in siblingselector_v3_position_check/selector_v3_position_check_2026_11_3. Train/extractexit0 andcleanserveridentity verified; inputs read-only.
- [x] Fast sizes/PLYheaders/full79events:413554checked/315800moved/76.3625%pass, originalspread60273moves. Train/extract0/clean6172, actualopts onlytwoflags,406cameras/supportsexact, evaluator/protocol/metadata bytes identical5fde7cb. No GT queryyet.
- [x] OriginalCLI once4models/exit0/106.139s evaluator106.985s wrapper, cleanfd73074; no GT failure/retry. Allold3metrics/meshSHA andGTsample/protocol/origin exact,12CSVJSONrowsagree.
- [x] 12CSVJSON/3oldcontrolmetrics/hash/protocol/GTsampleexact verified; analysisreport negativewithcausalboundaries, independentreadonlyreviewno materialissue (79events/115frames/metadata included), handoff/ledgers/artifactreport saved. No automaticnewmethod/training.

## 2026-11-04 Position-check failure explanation only
- [x] Re-read selector gate, batch relocation, CUDA surface mixing, persistent DA3/ALR and pinned DP-GS execution source. Existing formal metrics/logs only; no new GT/training.
- [x] Separate confirmed changes from causal hypotheses; note batch coupling with explicit mathematical counterexample, not actual trajectory proof. Review failure-mechanisms report before delivery.

- [x] Independent narrow source/report review found no material issue; added early positive hard-action counterevidence and persistent-loss shared-control boundary. Analysis delivered without new methods or experiments.

## 2026-11-04 User-approved direct output investigation
- [x] Inventory actual current/oldspread/early/C0 downloads: meshes, Gaussian PLY,406 render/depth/normal JPEGs,115 debug/pg_view frames,logs. No metric depth or historical target arrays.
- [ ] Reuse existing regional core on current result; reproduce official directional means and locate actual harm.
- [ ] Reuse evaluator for raw-vs-post diagnostic and examine existing RGB/debug images/logs; no training, method changes or historical-causality claims.
- [ ] Verify numeric artifacts, independent review, report evidence-supported versus untestable hypotheses.

- [x] Direct output: original regional completeness+accuracy reproduced all6 formal means and exactCD delta; raw3model diagnostic exit0 with exactoldcontrols;406JPEG full-view scan;9view door first-hit thin adapter successful. No GT failures/retries.
- [x] Main new damage located at door68: +1.034933mm CD vsoldspread, others net-.417332. Mesh signed-z median-96.8mm, DA3abs21.6/newmesh127.1mm;11400debug alreadywrong. Independent report/data review pending.

Independent read-only review passed:95regional delta rows exact, oldraw metrics/hashes exact,406JPEG summaries and9view share exact,4camera hashes equal,ray script/sourcehash/coordinate/common-hit logic verified. No material issues; no GT rerun/training or reviewer edits.

## 2026-11-04 Follow-up: door failure mechanism and modest remaining gains
- [x] Re-read actual depth_spread producer and saved config. RawDA3 confidence ratio0.0 means binary depth/ALR mask passes all finite conf; cannot claim confidence excludeddoor.
- [ ] Reuse existing output/score/propagation producer to examine door structure/anchoring and real training debug domain; no new training/production changes.
- [ ] Decompose allregional gains and harms versusspread,early,C0 and report global weighting, not net0.417 as isolated improvement.
- [ ] Verify artifacts and narrow independent review, separate current evidence from unavailable historical targets.

## 2026-11-04 User-approved process trace30K implementation
- [x] Userselects existing7K→30K observation of unchanged current method; D isolated codex/selector-process-trace, spec+inlineplan saved.
- [x] Shared observer record extraction, trace events realCPU RED/GREEN.
- [x] runpy launcher, readonly A-only fixed probe and CPUhelper tests; original train/algorithm remain identical6172.
- [x] Unified tests63=55pass8nativeSkip; independent review no material issues; Python3.10 syntax/CLI/Bash blocks pass. No auto-serverrun.
- [x] Implementation fixed cd6475526b8cccc50e3ce797bcea4a2c66bea664; launch guide placeholders replaced and independently reviewed. Push/final identity recorded below.

Follow-up analysis and direct-output checklists above completed with independently reviewed reports/artifacts. This does not authorize new scientific method changes.


2026-11-04 user server receipt: cd647552 trace run completed30K/23000 remaining steps in74:32, exit0, N1279429, train L1=.01710001826286316/PSNR30.836247634887698, App save/complete12:20:20. No manifest/files received yet; cannot claim96/24 verified. Append original TSDF command preceded by cheap96/24/79/nonempty checks, no new methods or training. Misaddressed GS-SDF context withdrawn; AmbiSuR unchanged. Server extraction/user checks pending.


## 2026-11-04 Downloaded process-trace analysis
User provides actual selector_v3_spread_process_trace/selector_process_trace_2026_11_4/cd647552_seed0_b16384_until15000_30k_v1. Approved analysis only: validate96/24/79/files/config, original formal mesh protocol once, inspect actual anchor/proposal/move/immediate surface/fixed-probe late coverage with existing scorer/prior-grid/region core. GT offline only; no production edits/training/deletion. Preserve original output readonly and write unique analysis/evaluation artifacts. Stage1 inventory pending.

Downloaded trace inventory exit0/1.263s:96reference/24probe/408NPZ/3,876,519,869bytes. All schema1 validated; actual move subset/target order and post-executor coordinates exact, late no move.79events415426checked/317598moved. Clean actualcd647552, optimization exactly prior6172 andcameras/support bytes exact. Model/mesh files found, original exit0. No GT query yet. Proceed frozen full-surface evaluator once with baseline/early/position/trace.

Trace analysis results so far: frozen formal4model evaluator exit0/107.56s, old3 metrics/input identities and GT sample/protocol exact. New trace CD40.5629326/F1@5 .785306509 vs C040.6981979/early39.7444646/oldposition41.7791282. Original regionalcore twoentries exit0 reproduced six directional means; door CD delta vs early+.03148, wall69+.39854/ceiling66+.36709/wall71+.12181. Main process adapter exit0/50.85s, no GT failures/retries: fixed door no old front-bulge/lateA-collapse; initialabs21.79→8000abs48.77→30000abs40.33mm while angle14.62→4.33→3.96deg and Aweight.928→.949→.954. Actual moved projected-door2509 records only42.29% nearest improve, mean1.80mm/median-.757mm. Same-center surface change63.85%below1mm. Propagation replay3events exactmask/maxdepthdiff<7.2e-7m. Thin detailCSV duplicated each row in ALL/door loop; originalsource/rawCSV preserved, exact-identical dedup published without GT rerun; summary counts were correct. Additional actual wall/ceiling diagnostics running, no training/methodchanges.


Process-trace analysis complete: all formal/control/input/protocol checks exact, region decomposition and actual process/target/pixel diagnostics saved. Additional three main harms matched-domain values current→spread:53.573→53.122,127.705→127.289,51.374→50.271mm; rawDA3 differs58.959/99.401/44.068. Jointunc averages.469/.201/.627, cannot inferA-only outsidefixeddoor. Independent readonly review no material findings; original adapter duplicate detail fully identical dedup noGT rerun and original source/data preserved. Three execution source snapshots/hash exact; live adapter parameters moved to adjacent JSON and future duplicate emission fixed, no GT rerun. Report/handoff/AGENTS updated; production unchanged. No new method or training authorized.


## 2026-11-04 Dense-view route discussion and literature
User identifies6failure issues and asks whether sparse-view donors transfer poorly, prioritizes accurate position targets and actual rendered surface effect. Discussion/source review only, no method edits/experiments. Verify sparse premise; distinguish dataset total view count from local identifiable evidence and existingDA3 redundancy. Academic MCP unavailable; OpenAlex fallback429, use primary arXiv/CVF/publisher/official source. Candidate new refs GVGS2026, CoMVS-GS2026, MonoMVSNet2025/SPE-MVS2026; read actual available source before recommending implementation.

- [x] Dense-route discussion: verified DP-GS sparse settings and actual selector two-source interface; read 11 non-README pinned donor files (GVGS4/MonoMVSNet7), full CoMVS mechanism/ablation. No method changes/GT queries/training. Evidence report saved; independent narrow review requested.

Dense-route independent read-only review complete: no material issues; statistical boundaries, selector/DA3 interfaces, MonoMVSNet/GVGS source and all11SHA verified. CoMVS external paper evidence remains primary-agent review, not independently refetched. Production unchanged; discussion only.

## 2026-11-04 Expanded literature request
User explicitly requests broader ideas beyond CoMVS/MonoMVSNet. Research/source downloads only. Search calibrated MVS/reflection/weak texture; joint normal-depth/planar structure/BA; surface execution/opacity; model-independent geometric priors. Deduplicate old24-paper review. Read actual official implementation for priority modules and pinSHA. No training/loss/method edits or GT queries. Academic MCP unavailable; prior OpenAlex429 so use primary CVF/arXiv/official sources.

- [x] Retain user priority: candidate location, better geometry, actual surface; include rendering only where geometrically connected.
- [x] Build31-entry deduplicated catalog with explicit reading tiers;44 new source files+3 reused pinned PAGaS files,33 files key implementation excerpts read. Fifteen source repositories including renderer submodule; placeholder SPE tree not counted as implementation.
- [x] Draft expanded survey tying calibrated image correspondences / pretrained MVS / shared surface execution / CoMe component separation to existing interfaces. No production changes, inference, GT query, weights or training.
- [x] Narrow independent read-only evidence review: no material issues; qualified AmbiSuR appearance/SSIM enable+raw-loss<0.5 condition and CoMe Python/paper scope, not fused CUDA audit. Four ablation groups/source derivation verified;47 source/12 paper hashes exact. Research report, catalog and review receipt archived. No production changes or experiment execution.


## 2026-11-06 Position admission and DA3 supervision follow-up
User critique identifies missing position correction vs extra ALR rule and new-target vs DA3 loss relation; does not accept propagation as established root cause. Authorized discussion/literature/source research while away only.
- [x] Recheck actual ALR/depth/candidate interfaces and known6findings.
- [x] Research concrete correspondence ambiguity, geometric support, depth-prior replacement/completion, normal and depth separation; read available official implementations.
- [x] Compare2–3 coherent minimal routes with explicit loss, masks, A/ALR relation and fallback; donor mechanisms vs our hypotheses separately identified.
- [x] Narrow evidence review and artifact verification, no production method/training change.


## 2026-11-09 Admission benefit and spread anchors Q&A
- [x] Read actual spread builder/propagation and train prior-normal producer.
- [x] Check official pinned MVSAnywhere demo/hub inference, not README alone.
- [x] Define offline target-gain/enrichment/coverage vs real training-benefit distinction; discussion only, no implementation/inference/GT evaluation.


## 2026-11-09 User-approved source/admission check
Current phase: implement minimal reusable offlineaudit and prepare AutoDL forward-only commands. Fixednative7K checkpoint and rawphotos/DA3 local present; only onefullnative7K reference cached, not17. User confirms server running. Dbranchcodex/depth-source-audit preservesprioruncommittedresearch. Plan docs/superpowers/plans/2026-11-09-depth-source-audit.md. NoGT/evaluation/newtraining yet.


2026-11-09发布回执：离线采集／推理实现固定`cae81969d5eee7f97e5a4fd9d8151b300742fbb4`，分支`codex/depth-source-audit`，操作见`docs/research/depth-source-audit-launch.md`。统一71项63通过／8native跳过（5.791s），独立复核无剩余实质问题；官方MVS模块仅导入验证，尚无权重加载／真实前向／GT质量结果。train／arguments／scene／utils／renderer／submodules保持87ec3c2，baseline／数据只读；服务器由用户执行17视角只读采集＋独立环境MVS推理，非30K训练。


2026-11-09用户返回服务器空间：系统盘余8.3G、数据盘余30G，无需清理／扩容。仅更新启动指南：10GiB保守空间预留检查、TMPDIR数据盘化；Python实现仍cae81969，训练／数据／baseline未动。准备用户执行前向采集与MVS推理，尚无实际服务器结果。


2026-11-09服务器接口失败已修复：原capture／dependencies／freeze成功，MVS初始化因供体解析宿主CLI退出2；固定修复`fc75cf58f4830e88fb40149ddd8c569f66538a33`使用官方ignore_cl_args=True，科学配置／权重／forward不变。真实供体回归修复前复现退出2、修复后通过；统一72项64通过8native跳过，独立复核无实质问题。原失败记录保留，仅恢复infer至同根mvs_cli_fix_v2，新退出码inference_cli_fix_v2.exitcode；不重复capture或安装／训练／GT，实际GPU前向尚待用户执行。恢复命令见docs/research/depth-source-audit-launch.md。


2026-11-09用户批准支持视角上限修复，固定实现b769fd555bd570ecd8a2aa6818e9f75546b0a237：mvs_supports=7作为最大值，保留真实邻居顺序；15视角取7，DSC06669取5，DSC06681取4；不填充／重选。统一74项66通过8native跳过，实际17视角／127图片header检查通过，源代码独立窄审无实质问题。原capture／配置／评分／训练／DA3／renderer不改。已有self-stack推理完成前3张后因旧exact7断言退出；原partial与日志保留。新批次仅infer至mvs_support_cap_v4，复用缓存与环境，仍保留60秒self-stack。命令见docs/research/depth-source-audit-launch.md；用从fc75制作的离线bundle避开GitHub503。实际4/5源GPU／17完整manifest／GT质量尚未验证。


2026-11-09服务器成功回执：b769fd55支持视角上限版MVS推理，用户日志完整列出17视角480x640及inference_support_cap_v4.exitcode=0；DSC06669实际5源／DSC06681实际4源均完成，其余15视角7源。辅助DINO／ResNet缓存复用，原capture及旧partial／失败记录保留。实际manifest／17NPZ下载完整性与GT精度／准入收益尚未本地检查；不据此宣称几何收益，不启动训练。服务器归档根/root/autodl-tmp/ambisur_runs/Tool_Room/depth_source_check/20261109_v1；用户下载目标D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/depth_source_check/20261109_v1。之后仅用原evaluate_depth_sources.py score，capture来自原capture子目录，MVS来自mvs_support_cap_v4。


## 2026-11-10 Downloaded17 source/admission evaluation
User supplies actual root D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/depth_source_check_17views_after_v3_position_check_from_NVSAnywhere/20261109_v1. Resume approved existing plan, no new methods/training. Directory contains17native7K capture records and17MVS support-cap results; original stage exits0, supports15x7/1x5/1x4. Source pipelineb769, capturecae819; oldfailedrecords preserved. No evaluation running at intake, repo cleanHEADdb5361a.
- [x] Preflight actual full-depth NPZ/masks/K/transforms, camera+support hashes/config/units and source identities without GT scoring; freeze output path/protocol and input receipt. Do not compare against30K current or infer GT labels.
- [x] Run existing evaluate_depth_sources.py score once, 17same-native7K, original threshold/seed/budget, originalGTQueries + semantic regions. Save unique analysis/depth_source_comparison_20261110_v1, preserve any failure; no automatic GT-after-failure retry.
- [x] Summarize four-source common-domain/nativecoverage and NCC accepted/rejected/random pairedgain/harm/coverage; separate illustrative vscoverage and targetquality vs actualsurface benefit. Independent read-only review, report and handoff.


2026-11-10输入预检完成：下载449774436bytes；17完整深度图+17MVS，实际15x7/1x5/1x4，capture0/infer0，config与本地完全相同；原生7K/1074511797byte checkpoint身份，4metadatahash与本地baseline相同，全部K/pose一致（max4.77e-7），MVS480x640。GTmesh31547a...历史身份一致，segments/annotations路径已冻结；尚无GT查询。唯一输出analysis/depth_source_comparison_20261110_v1/score，复用原score入口，seed/门槛/预算不变。


2026-11-10离线17视角评价完成：真实下载路径depth_source_check_17views_after_v3_position_check_from_NVSAnywhere/20261109_v1，实际状态为原生C0的7K。固定clean a4155ac原score一次40.97秒退出0，5369行／3237汇总，GT无失败重跑。四者共同域3059926条／36.62%可见记录，MAE当前59.296／DA3 71.781／spread55.371／MVS66.722mm，不能把条件域排名扩展到全场景。NCC接受GT后59412条，平均g0+2.312／gP+15.776mm，但改善DA3比例63.72%低于GT前等预算随机64.18%，伤害>5mm率33.26%高于随机31.78%；全图仅实际抽样替换目标收益0.11216mm／覆盖0.71095%，不是训练收益。墙71局部MVS强，天花板原DA3更准且准入危险。两个NCC视角全部重用于MVS输入。独立只读终审无实质问题，CSV/JSON逐字段与区域计数校验通过。报告docs/research/2026-11-10-depth-source-and-admission-results.md，产物analysis/depth_source_comparison_20261110_v1；没有新增方法／训练／loss／预算／GT标签，不直接授权30K。
# 2026-11-10 approved observed correspondence batch

## 2026-11-10 user accepts four-source routing direction
- [x] Record route approval: RoMa fixed correspondences compare current/spread/DA3/MVS; triangulated depth not a training teacher; support identifiability first; surface execution later.
- [x] Draft staged specification and exact reusable code interfaces; distinguish accepted route from proposed numerical settings and future training authorization.
- [x] Independent builder check; clarified GT-before per-source-label permutation random control and identical old/new-support decision rules. Present concrete first-batch scope for user review. No implementation/model run/GT score/new training in this drafting turn.

## 2026-11-10 approved first batch implementation
User confirms concrete spec. Dbranchcodex/four-source-depth-routing from89c502d,inlineexecution.
- [x] Shared support/ranking TDD; old correspondence behavior preserved.
- [x] CLI/cache/camera/score roundtrip TDD, fixed29pairs/8reuse21new,actualnonGTpreflight. Unified84:75pass/9skip.
- [x] Unified verification,oneindependentreview(noCritical/Important),fixedcommit/launch/bundle/push/userhandoff. No training/GPU/GT execution here. Server actual21newpairs stillpending userexecution, not part oflocalcompletion claim.

## 2026-11-12 server receipt and next evaluation
- [x] User returned fixedc4546a3/finalmanifest15K/output74M/exit0; record forward completion, no quality claim.
- [x] Download verified29pairs/8reuse21new/33NPZ/frozen65536IDs/four-source targets; originalrouting-score onceexit0. No automatic rerun/newtraining.

## 2026-11-12 downloaded routing evaluation
- [x] Verify supplied four_source_routing_v1 manifests/33NPZ/sourcearrays,29pair receipts and fixedc4546a3; freeze analysis/four_source_routing_4views_20261112_v1 before reading GT grids.
- [x] Existingrouting-score once with original4cachedGTgrids; exit0/14560rows/65536pixels. No threshold edits/newmeshqueries/model/training.
- [x] Compared common/full-frozen policies, source/ceiling/conditioning and exploratoryagreement/margins; original14560+224rows exact. Independentreview noCritical/Important; report/handoff complete. Nextmethod/experiment remainsdiscussiononly.

## Downloaded GPU evidence evaluation (approved existing specification)
- [x] Verify downloaded v3 manifest/12NPZ/config/original65536IDs/input hashes and shared domains without inspecting GT values; freeze unique analysis output.
- [x] Run existing correspondence-score once against original four cached GT grids; no mesh queries, threshold edits, reruns after GT failure or training.
- [x] Summarize same-domain source quality, observed-vs-NCC admission, pair/joint triangulation and numerical conditioning; independent read-only review completed without material findings, report and handoff updated. Supplementary JSON export failure retained; no GT rerun or recovery.

- [x] Preserve prior analysis/spec as 30b7dad, create D branch codex/observed-correspondence-audit.
- [x] RED meaningful unified regressions; implement offline helper/CLI/config.
- [x] GREEN unified tests, actual input preflight, independent narrow review.
- [x] Fixed implementation7c698a3, AutoDL launch and probe/source/code bundle handoff; no automatic server/model run.

## 2026-11-12 user corrects full-image routing objective
- [x] Read review and inspect actual routing masks/score/model-query persistence; exact native584x876,511584/view,2046336fourviews; sampled16384/view only3.2026%.
- [x] Verify four complete current/DA3/spread maps and originalMVS predictions exist; paircache stores only sampledquery outputs,notdense RoMa fields. No GT values read/newmatching/scoring.
- [x] User chose29pool/maxangle<=2; draft concrete denseallpixels/availablecandidate/shared1support/DA3fallback spec. No productionchanges/newbatch before review.
- [x] User approved fullspec including sourcecount/sharedsupport priority;implemented53600fd andverified,serverpending.

## 2026-11-12 dense implementation approved
Userapprovedconcretespec/sourcecountpriority. Dbranchcodex/four-source-dense-routing fromef3e32c,inline3tasks:shareddecisionsTDD;densecapture/scorerTDD;verification/review/fixedhandoff. No newGT/model/trainexecuted.

- [x] Dense implementation complete53600fd:unified92/83pass9skip,actual2046336inputpreflight,independentreview2P1fixedwithCLIRED/GREEN+narrowrecheck;launchsyntaxverified. No realGPU/GTquality/newtraining.
