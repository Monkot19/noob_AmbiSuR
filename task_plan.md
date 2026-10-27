# AmbiSuR Core 实施计划状态

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
- [ ] Discuss two alternatives and choose next method with user; no implementation or new training authorized yet.
