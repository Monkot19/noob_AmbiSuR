# AmbiSuR Core 实施计划状态

> **当前（2026-10-23）：** D 盘 `D:\research_Space\noob_AmbiSuR` 已接管 `codex/second-run-selector`；旧 Core 计划为历史。30K退出码0、点云和app权重非空已确认，train PSNR基本持平。用户已同意同参数提取新mesh并比较几何；提取结果和几何收益仍待验证。

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
