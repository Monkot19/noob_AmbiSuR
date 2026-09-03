# AmbiSuR 新项目 Progress

## Session Log

### 2026-09-01 read-only audit and planning
- **Status:** audit_complete_waiting_for_approval
- [x] 读取指定 skills 及 Codex 适配说明
- [x] 完整读取 AGENTS、README、研究交接、baseline audit、最终设计、服务器流程、manifest 模板与三份规划文件
- [x] 核验 Git、目录、训练/参数/model/renderer/CUDA/DA3/ALR/densify/checkpoint/evaluation 代码
- [x] 写入公式输入映射、规格差距与 E0/D0/C1–C6 TDD 实施计划
- **Scope:** 仅只读审计；只更新三份规划文件，不改源码、不提交、不推送、不启动 AutoDL
- **Tool note:** 初始系统无 `python` 命令，已用 bundled Python 完成 session catchup；用户随后安装的系统 Python 3.14.7 已可启动，但当前环境仍无 `torch/pytest`，不属于项目约定的 Python 3.12 验证环境。
- **Interim evidence:** 已读取完整 `train.py`；确认 DA3 depth、ALR 双 backward、multi-view trim、densify/prune 与 optimizer 时序。
- **Interim evidence:** 已读取参数系统、Scene、renderer、AppModel 与完整 GaussianModel；确认 Core 无开关/状态，严格 Gaussian pooling 缺 renderer 输出，C6 缺拓扑映射与 checkpoint 状态。
- **Interim evidence:** 已核对 DA3 预处理/加载、Camera 张量设备与邻接表；确认现有相机/深度/confidence/内参可作为可靠性输入，但 DA3 normal 与 Gaussian pooling 需新增计算路径。
- **Interim evidence:** 已核对 renderer Python/C++/CUDA forward 接口；确认 `w=alpha*T` 仅存于 kernel 局部，严格证据 Pool 需要可选 CUDA Gaussian 累加输出。
- **Interim evidence:** 已核对 CUDA backward；确认 Ray-Color→SH→xyz 泄漏、ALR 重复触发风险，以及 DA3 plane-depth 梯度会覆盖 pos/scale/opacity。
- **Interim evidence:** 已核对 checkpoint、TSDF mesh 和外观/DTU/TnT 评价；仓库缺 ScanNet++ 专用配置与评价入口。系统 Python 已安装为 3.14.7，但不等同于项目约定的 3.12 环境。
- **Spec audit:** 发现 C1 prior-loss 定义/权重、K 的共同支持有效性、生命周期 `s/ell` 符号，以及 C1–C5 screen-space densification proxy 归属缺口；按 brainstorming 规则一次只请求一个确认，首问将是 C1 是否严格沿用 baseline prior 公式。
- **Readiness:** 当前无项目测试套件；系统 Python 3.14.7 无 torch/pytest；训练 seed 实际固定为 0 但未进入 CLI/manifest。E0 尚不具备直接按 TDD 开工条件。
- **Git evidence:** `main@d6f15c8891a53800d5e3100f95817a7dd7f98e2f` 与 `origin/main` 一致；`upstream/main@88d64054f53e0eba9ce49198282edf4a67fc8ca8`；无 tag、无根 `.gitmodules`/gitlink。当前仅三份获授权规划文件为 modified，源码未改。

### 2026-09-01 C1 prior contract clarification A
- **Status:** recorded_waiting_for_k_joint_validity
- [x] 用户正式批准 A：Core C1–C6 沿用 baseline DA3 depth `depth_weight=0.1` 与 Dual-End ALR `unc_weight=0.1`，保持原启用时序、权重及有效区域。
- [x] 记录设计 §7.1 新 `L_Pd/L_Pn` 与 baseline prior 的冲突；设计稿原文未修改；新 prior 移为 Core 后独立 Supporting 候选，禁止混入 `c1-need-gate`。
- [x] 锁定 residual contract：`g_baseline_total=g_B_residual+g_P_clean`；`g_C1=g_B_residual+N_i*g_P_clean`；clean prior 不含 Ray-Color且 SH/曝光为0。
- [x] 锁定 C1–C5 densification proxy 使用 baseline total-gradient，C6 前不提前改变 topology；每轮最终只有一次 optimizer step。
- [x] 锁定 GPU 单步 oracle 边界：1000/1001、5000/5001、7000/7001、15000/15001；逐组比较 gradient、viewspace proxy 和 step 后参数，并覆盖 ALR on/off。
- [x] 锁定停止条件：固定容差失败或需要 `out_observe/out_all_map` 等近似时立即停止，不进入 C2。
- **Scope:** 只更新 `task_plan.md`、`findings.md`、`progress.md`；没有修改设计稿、方法源码、Git refs 或实验状态。

### 2026-09-02 K joint-validity architectural clarification A
- **Status:** recorded_and_verified_waiting_for_next_clarification
- [x] 用户批准 `Z_i^PG=sum(sg(w)m_PG)`、`V_i^PG=1[Z_i^PG>tau_Z]`，复用 `tau_Z=1e-4`，不新增超参数。
- [x] 设计稿 §3.2/§5.4 已补入 `K_raw` 有效观测、0→1首次初始化、invalid 不更新 EMA、历史 K 仅日志可见和新点未初始化合同。
- [x] 设计稿 §6/§11.2 已补入仲裁优先级：Bypass/单方可靠不依赖 joint gate；双方可靠时所有 K 分支要求 `V_pg=1`；D0/C3–C6 joint-invalid→Abstain，C2→`g_B`，Delta 不选边。
- [x] D0 计划增加 `Z_pg/V_pg`、整体/双方可靠条件 coverage、joint-invalid Abstain、scene/stage/N-quantile coverage 与离线 GT error 分组；coverage 塌缩只报告并停止。
- [x] 实施计划增加 zero support、0→1/1→0 EMA、历史 K 隔离、Bypass/单方优先级、D0/C2/C3+ 行为和 feature-off 测试。
- **Scope:** 仅文档同步；方法源码、Git refs、tag 和训练保持不变。

### 2026-09-02 lifecycle `s_i`/`ell_i` architectural clarification A
- **Status:** recorded_and_verified_waiting_for_next_clarification
- [x] 用户批准 `s_i` 固定为经过迟滞的稳定五状态仲裁结果，`ell_i` 固定为生命周期状态；所有 clone/split/prune gate 只消费 `ell_i`。
- [x] 设计稿 §8/§8.1/§8.2 已补入 `ell_i=M(s_i)` 只在 evidence refresh 执行、刷新间保持，以及 `d_i` 只在 evidence refresh 计数的合同；原 §8.1 的 gate 枚举已由 `s_i` 修正为 `ell_i`。
- [x] 新 Gaussian 在 topology 提交后强制 Probation 至少 500 个 optimizer iteration；期间可记录但不能消费 `s_i` 解除保护，满 500 轮后只在第一次 evidence refresh 按当时稳定 `s_i` 映射。
- [x] C6 计划增加五状态映射、非刷新不映射、gate-domain、499/500/下一刷新、`d_i` refresh-only、Probation `g_B`、topology migration 与 checkpoint/resume 测试。
- [x] 本次澄清当时没有授权 truncation 改动，并把设计 §8.2 的 Probation `trunc_sigma=2.0` 冲突记录为下一项；该项现已由下一节的方案 A 正式闭合。
- **Scope:** 仅修改设计稿和三份规划文档；没有修改方法源码、Git refs/tag 或启动训练。

### 2026-09-02 Probation truncation architectural clarification A
- **Status:** recorded_and_verified_waiting_for_final_plan_approval
- [x] 用户批准 Core C6 不设置或迁移逐 Gaussian truncation；Probation 原样继承 baseline 全局 `--trunc_sigma/--disable_trunc`，当前默认值 `2.0` 不是生命周期动作。
- [x] 设计稿 §8.2 已明确 Core 不新增 per-Gaussian truncation Tensor、不按 `ell_i` 改写全局配置、不修改 Python/C++/CUDA truncation 接口；§9.4 保留为 G0–G2 后独立 Supporting 候选。
- [x] C6 计划增加 C5/C6 全局 renderer settings 等价、lifecycle/checkpoint 无 truncation Tensor 的测试和停止条件。
- [x] 至此本轮只读审计识别出的四项 architectural specification gaps（C1 prior、K joint validity、`s/ell`、Probation truncation）均已有正式决策。
- **Scope:** 仅修改设计稿和三份规划文档；没有修改方法源码、Git refs/tag 或启动训练。

### 2026-09-02 user-provided ScanNet++ baseline/data audit
- **Status:** old_tool_room_protocol_and_data_contract_verified_waiting_for_da3_dataset_and_run_approval
- [x] 以只读方式核验 `D:\research_Space\output\AmbiSuR_original\ScanNetpp`；未移动、覆盖或删除任何旧结果。
- [x] 从 `Tool_Room_r2/r4/train.log` 恢复真实 30k 训练命令；从 `cfg_args/cfg_opts` 恢复 resolution、DA3/ALR/Ray-Color/truncation 和其他默认配置。
- [x] 从 `extract_general.log` 恢复统一 mesh 参数：`max_depth=5.0, voxel_size=0.005, sdf_trunc_scale=4.0, num_cluster=2`。
- [x] 核验本地 Tool source：406 images/poses，名称集合与旧输出 406 cameras 完全相同，PINHOLE 1752x1168，内参一致；但本地 source 不含旧 DA3 depth/conf/aligned model。
- [x] 核验本地 Utility source：147 images/poses，但相机仍为 loader 不支持的 `OPENCV_FISHEYE`，尚未形成可训练 `colmap_undistorted`。
- [x] 建立 canonical upload 与 per-run working view 合同；GT 独立目录，旧协议无 `split.json/--eval`；发现 loader 会写 source `points3D.ply`，因此禁止直接训练 canonical asset。
- **Scope:** 只读外部结果/数据审计并更新规划文档；没有运行训练、生成 DA3、评价 mesh、修改方法源码或 Git refs。

### 2026-09-02 Tool Room upload and first-run authorization
- **Status:** c0_reproduction_accepted_waiting_for_git_and_e0_approval
- [x] 用户报告已按 frozen path 上传 Tool Room canonical source/GT；截图只验证顶层目录可见，内部文件/数值/checksum 仍待终端 preflight。
- [x] 用户明确授权至少先运行一个实验；当前把授权严格限定为 Tool Room `seed 0 / -r 4 / 8000 iterations` clean-baseline，新输出目录，跨过 1001/5001/7001 分支。
- [x] Utility Room 暂不作为本次运行前置条件；G2 多场景与最终主实验前仍必须补齐。
- [x] 恢复 user-operated terminal 合同：助手给命令，用户执行并回传完整输出，复核后再给下一条；助手不直接控制服务器。
- [x] 用户回传只读 preflight：commit/diff、406 image/depth/conf、原/对齐 COLMAP、scale、GT、shape/finite、PyTorch/CUDA 与 `train` import 均通过。
- [x] preflight 唯一 FAIL 来自错误的 Python 3.12 断言；`environment.yml` 实际要求 Python 3.10 + PyTorch 2.7.1+cu128，服务器 3.10.21/2.7.1+cu128 正确。pytest 缺失不阻塞 legacy baseline，但 E0 前必须补齐。
- [x] 用户执行 working-view/manifest/8k 启动命令：private aligned copy 通过，dataset manifest SHA256 为 `aad92aa2e0f0d072756b3a56c686d5c1d35f448811ce60ca4360c67dbc3ef255`。
- [x] 进程 PID `9305` 以 `LAUNCH_STATUS=RUNNING` 返回；日志确认 406/406 cameras、原始 `sparse/0` pose、私有 `sparse_da3_aligned/0` priors 和 `cameras_extent=3.394348192214966` 已加载。
- [x] 训练正常达到 `8000/8000`，保存 iteration 7000/8000 point cloud 与 iteration 8000 app model，并打印 `Training complete.`；进度耗时约 5:21，末段 points=1,209,624。
- [x] 完成性审计通过：Git/data/GT 均未变化；全日志 0 error/NaN/Inf；7k/8k artifacts 完整；3k/7k 指标和 PLY vertex 已提取。
- [x] 新/旧 `-r 4` 的 7k PSNR 仅差 `+0.009570 dB`，L1 相对差约 `-0.85%`，PLY vertices 相差 `+0.625%`；接受 path-check，但不标记完整 C0 reproduction。
- [x] 发现旧 r4 `cfg_args resolution=2` 与 train log `-r 4` 内部冲突；后续只以 train command + 新 manifest/cfg 为协议依据。
- [x] 用户明确批准下一次 Tool Room `-r 2 / 30000 iterations / seed 0` 正式 C0 候选运行；授权不扩展到 tag、源码修改或 E0。
- [x] C0 candidate 正常达到 `30000/30000` 并打印 `Training complete.`；run path 为 `attempt_20260902T082615Z`，用时 50:13，最终 points=1,297,647、L1=0.0164226247、PSNR=31.0740604。
- [x] C0 training completion audit 通过：exit 0；Git/data/GT unchanged；0 errors/NaN/Inf；peak GPU 11,966 MiB；7k/30k artifacts 与 cfg 完整。
- [x] 新/旧 r2 的 7k PSNR 差 `+0.001024 dB`、PLY vertices 差 `-0.597%`；30k PSNR 差 `-0.02442 dB`、points 差 `+0.0389%`。接受训练路径。
- [x] 2026-09-03 按冻结旧 r2 参数启动 mesh 提取；runtime gate PASS，launcher PID `2317`、mesh PID `2320`，首段日志确认 iteration 30000 与 406/406 cameras。
- [x] Mesh exit 0；406/406 render/TSDF 完成；全日志 0 error/NaN/Inf；峰值 5,480 MiB；Git、canonical source 与 GT 未变化。
- [x] 新/旧 raw mesh vertices/faces 差 `-1.364%/-1.050%`，post 差 `-1.032%/-0.918%`；接受该运行作为当前 frozen C0 reference，但不声称与 provenance 不完整的历史输出 bitwise 等价，也不替代 E0 feature-off oracle。
- **Scope:** C0 运行授权已结束；2026-09-03 新授权仅覆盖 Git baseline tag、累计分支及四份文档的提交/推送，不覆盖方法源码、D0/C1 或其他实验。

### 2026-09-03 Git baseline locking
- **Status:** completed_waiting_for_e0_plan_approval
- [x] 用户批准 annotated `c0-baseline` 精确指向 C0 SHA `d6f15c8891a53800d5e3100f95817a7dd7f98e2f`。
- [x] 远端预检确认 `origin` 尚无同名 `c0-baseline` tag 或 `research/core-routing` branch，未覆盖既有 ref。
- [x] 本地创建 `c0-baseline`，并从相同 baseline SHA 创建/切换累计分支 `research/core-routing`；四份获批文档修改随分支保留。
- [x] 四份文档已由 commit `6145c5787b3d6453a07a28da94bc2f44e26bcc47`（`docs: add Core audit and implementation plan`）提交；`research/core-routing` 与 annotated `c0-baseline` 已推送到 `origin`。
- **Scope:** 不修改方法源码，不安装依赖，不启动 E0/D0/C1 或服务器实验。

### Project bootstrap
- **Status:** complete
- [x] 交接包已在仓库根目录并完整读取
- [x] 最终设计稿已核对
- [x] baseline candidate commit 已记录
- [x] 本轮识别的 architectural specification gaps 已闭合
- [x] Git baseline tag 与累计分支已按批准策略建立
- [x] Core implementation plan 已获用户批准；当前执行授权严格限于 E0

### 2026-09-03 E0 implementation authorization
- **Status:** paired_500_failed_rng_trace_excluded_pending_baseline_repeat
- [x] 用户批准完整实施计划，并授权当前仅实施 E0 测试、default-off 配置/调度、显式 seed、复现元数据和只读 comparator。
- [x] 开工前确认 `research/core-routing@59ffca971782b44439c19f7bc18ea3490b1d452c`、upstream 同步且工作树 clean。
- [x] CoreConfig/seed/runtime/comparator tests 均先观察到目标 RED，再完成最小 GREEN；19 项非 GPU suite 在 bundled Python 通过。
- [x] Commit `68922cc` 保存 test-first harness；commit `b2c46db` 保存纯 E0 配置/runtime/comparator foundation；`train.py` 尚未修改。
- [x] 在 AutoDL clean `research/core-routing@b2c46db49e3465da7ff5cfda56a7ddd30be6f02c`、Python 3.10.21 以标准库 `unittest` 运行 integration test，精确 RED 为 `train.py` 缺少 `build_checkpoint_payload`；未安装依赖、未启动训练。
- [x] Commit `580adeb` 先增加 metadata integration 合同；commit `0601056` 只接线已观察 RED 所需的 runtime helper、`training(..., core_config=None)`、feature-off legacy dispatch 与 legacy tuple checkpoint。
- [x] AutoDL 第二次 integration：legacy dispatch、tuple checkpoint、explicit CoreConfig 三项通过；metadata 唯一精确 RED 为 logger 二参数签名（commit `02ac3b9700ea53a9f723ef3f62c3c8cac1b15d42`）。
- [x] 最小接线已在本地完成：logger 写 resolved config/run identity，CLI 显式传递 seed/CoreConfig；19 项非 GPU suite、AST parse 与 `git diff --check` 通过。
- [x] AutoDL 在 clean `research/core-routing@a7d04d4bbd28aa025f1d09373e8e7d1e615bf688`、Python 3.10.21 上以标准库 `unittest` 验证四项 integration GREEN（4/4，0.097 s），测试后工作树 clean。
- [x] AutoDL clean `research/core-routing@7223f919e8e015f1b1eed2d94d6855aed3b4eb29` 完整 23 项 component suite PASS（0.211 s）；CLI help 返回 0，八个 flags 全部存在，测试后工作树 clean。
- [x] 用户单独批准第一次 E0 500-iteration paired experiment：同一 frozen Tool Room canonical snapshot/seed/config，baseline 与 E0 all-off 串行、各自新建 private view/output；不含 GT/mesh/tag/D0/C1/8k。
- [x] 首次 baseline launcher 在训练前被过严数据门停止，无 run root/训练结果；只读审计确认原始 pose 是完整 txt/PINHOLE，DA3+aligned 是 binary，scale 正确、无 split、Git clean，数据无需重传。
- [x] 第二次 launcher 也在训练前停止：错误统计 depth 全目录 812；审计确认训练所需 depth/conf `.npy` 各 406 且与 images 逐名严格对应，另 406 个 depth `.jpg` 只是预览，无 run root。
- [x] 第三次 launcher 在重复 `git fetch origin --tags` 时遇到 GitHub 443 timeout（130.66 s）；branch fetch 已成功，但流程仍在 run root/训练前停止，return 128。
- [x] Baseline half PASS：`pair_20260903T090116Z`，`d6f15c8891a53800d5e3100f95817a7dd7f98e2f`，500/500，exit 0，0 error/nonfinite，200,000 points，peak 4,769 MiB，canonical source/Git clean，PLY SHA `01407a4d…f0c5c`。
- [x] Exact E0 `a26082154889ed539322425347af5a57a859a52f` 同 pair 500 完成：训练/metadata/artifact/source/Git均PASS，服务器已切回 `research/core-routing`（head `9f75c97…`，clean，behind remote 3 commits）。
- [x] Strict equivalence FAIL：checkpoint 首个 mismatch=`_xyz`；L1 `+2.2673e-5`、PSNR `-0.0050125 dB`、PLY SHA不同；points=200,000、peak delta=0、wall delta=-1 s。
- [x] Partial field audit：private prior PLY、`knn_f`、`features_rest`、`max_weight` exact；learned params与 densification proxy 均 finite 但不同，定位到训练阶段分叉。
- [x] 修正版只读审计完成：spatial LR、optimizer hyperparameters/steps、共同配置 exact；app model、SH rest 及其 optimizer moments exact；已训练参数及 moments 分叉。
- [x] Baseline/E0 fresh-process RNG sentinel 完全一致：Python/NumPy/Torch CPU/Torch CUDA state SHA 以及 500-step camera trace SHA 全部相同；排除 seed/logger RNG 消耗与相机顺序差异。脚本 exit 0，并恢复服务器 `research/core-routing@9f75c970b3aea0694934424cd98a3e05c7705162`，工作树 clean、当时 behind origin 3。
- [ ] 下一最小诊断需用户另行批准第二次 exact-baseline 500 self-repeat；在区分 baseline CUDA 非确定性 hypothesis 与 E0 非 RNG 副作用 hypothesis 之前，禁止修改实现、调整容差或运行 8k/D0/C1。
- **Scope:** 不实现 D0/C1，不修改 renderer/CUDA，不创建 tag，不安装依赖；当前实验授权仅限 paired-500，不含 8k/正式实验。

## Experiment Readiness

- **当前阶段：** Phase 0、Tool Room C0 锚点和 E0 工程组件门已完成；E0 paired-500 严格等价失败，当前停在系统化诊断，D0/C1–C6 未开始。
- **最早可启动的下一阶段：** 仅在用户批准后运行第二次 exact-baseline 500 self-repeat，用已有结果测量 baseline 自身重复性；这不是方法实验，不使用 GT/mesh，也不授权 8k/D0/C1。
- **E0 当前门：** RNG、camera trace、配置、optimizer 超参数与输入初始化证据一致；仍需 baseline self-repeat 区分 GPU baseline 非确定性与 E0 非 RNG 副作用。若 baseline self-repeat 逐位一致，则继续查 E0 副作用；若 baseline 自身分叉，则先报告其噪声包络并请求用户确认新的 G0 数值等价判据，绝不静默放宽。
- **首个诊断实验：** E0/G0 通过后运行 D0 Tool Room seed 0（正式时序到 7k），只记录证据/状态；G1 不通过则停止。
- **首个方法实验：** D0/G1 通过且 C1 single-step gradient oracle 全部通过后，才运行 C1 Tool Room seed 0 quick。C2–C6 依次按上一阶段 tag 晋级，不能并行跳级。
- **当前数据事实：** 服务器 Tool source/DA3/GT 已通过文件数、basename、数值、COLMAP、scale 与 hash preflight，并完成 C0；Utility 尚未上传且本地版本仍需从 FISHEYE 转为 PINHOLE/SIMPLE_PINHOLE。它不阻塞 E0/D0/C1 Tool quick，但阻塞 G2 多场景结论。旧协议为全部相机训练，无 `split.json`，且旧结果没有 ScanNet++ GT geometry metric。
- **当前环境事实：** 本机 Python 3.14.7 无 torch/pytest，不能充当项目验证环境；服务器已验证为仓库锁定的 Python 3.10.21、PyTorch 2.7.1+cu128、CUDA 12.8/RTX 4090。服务器 pytest 尚未安装，E0 前补齐。

## Verification Log

| Date | Stage | Commit | Command | Result | Manifest |
|---|---|---|---|---|---|
| 2026-09-01 | read-only audit scope | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` + uncommitted planning docs | `git diff --name-only`; excluded-source diff | only `task_plan.md`, `findings.md`, `progress.md`; no source diff | n/a |
| 2026-09-01 | planning-doc syntax | same | `git diff --check` | exit 0; only LF→CRLF notices, no whitespace error | n/a |
| 2026-09-01 | plan requirement coverage | same | E0/D0/C1–C6 section matrix; placeholder assertion | all stages contain files/functions, failing tests, CPU/GPU, feature-off, Tool quick, gates and commit/tag fields; no placeholder marker | n/a |
| 2026-09-01 | local test readiness | same | `python --version`; import discovery for torch/pytest | Python 3.14.7 starts; `torch=None`, `pytest=None`; no project tests run | n/a |
| 2026-09-01 | C1 clarification A record | same | approved-A content assertion; `git diff --check`; excluded-source/design diff; `git tag --list` | all requested A/oracle constraints present；diff check exit 0；only three planning docs modified；design/source diff empty；no tag | n/a |
| 2026-09-02 | K joint-validity clarification A | same + four uncommitted docs | four-file contract assertion；math delimiter parity；placeholder scan；`git diff --check`；excluded-method diff；tag list | contract coverage pass；164 math delimiters/even；no placeholders；diff check exit 0；only design + three planning docs modified；no method source diff；no tag | n/a |
| 2026-09-02 | lifecycle `s_i`/`ell_i` clarification A | same + four uncommitted docs | lifecycle contract assertion；old-`s_i` gate absence；math delimiter parity；placeholder scan；`git diff --check`；changed-file allowlist；status/tag | contract coverage pass；old lifecycle-enum `s_i` gate absent；166 math delimiters/even；no placeholders；diff check exit 0；exactly design + three planning docs modified；no method source/untracked file；no tag | n/a |
| 2026-09-02 | Probation truncation clarification A | same + four uncommitted docs | four-file contract assertion；stale-unresolved scan；math delimiter parity；placeholder scan；`git diff --check`；status/tag | approved global-baseline-only contract present；no stale unresolved architectural text；166 math delimiters/even；no placeholders；diff check exit 0；exactly four docs modified；no method source/untracked file；no tag | n/a |
| 2026-09-02 | old Tool Room protocol and upload data contract | same + four uncommitted docs | old `r2/r4` log length/existence；Tool image/pose count；document marker assertions；placeholder scan；`git diff --check`；changed-file allowlist；tag list | old logs remain 2228/2232 bytes；Tool has 406 images/406 poses；contract markers pass；no placeholders；diff check exit 0；exactly design + three planning docs modified；no method source/untracked file；no tag | n/a |
| 2026-09-02 | server Tool preflight | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | user-operated read-only Git/data/runtime validator | Git/data/CUDA/train import pass；406 matched finite priors；Python 3.10.21 and torch 2.7.1+cu128 match `environment.yml`；preflight Python-3.12 assertion identified as validator error；pytest absent but non-blocking for baseline | pending run manifest |
| 2026-09-02 | Tool baseline r4 8k launch | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | user-operated isolated working-view + manifest + `nohup python -u train.py ... -r 4 --iterations 8000` | completed 8000/8000 in about 5:21；saved 7k/8k point clouds and app model；completion audit pending | `/root/autodl-tmp/ambisur_runs/Tool_Room/baseline-pathcheck-r4-8k/d6f15c88/seed_0/attempt_20260902T074545Z/manifest.md` |
| 2026-09-02 | Tool baseline r4 8k completion audit | same | user-operated full-log/config/artifact/hash audit + local old-r4 comparison | accepted as path-check；0 errors；canonical unchanged；7k PSNR delta +0.009570 dB；point delta +0.625%；not a 30k C0 reproduction | same |
| 2026-09-02 | Tool C0 candidate r2 30k training | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | user-operated monitored `-r 2 --iterations 30000` run | training reached 30000/30000 and printed complete；final PSNR delta vs old r2 -0.02442 dB，point delta +0.0389%；completion audit pending | `/root/autodl-tmp/ambisur_runs/Tool_Room/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z/manifest.md` |
| 2026-09-03 | Tool C0 r2 training completion audit | same | user-operated exit/resource/hash/config/full-log/artifact audit | accepted training path；exit 0；0 errors；canonical/GT unchanged；peak 11,966 MiB；7k/30k deviations small；mesh pending | same |
| 2026-09-03 | Tool C0 r2 mesh launch | same | user-operated safety-gated `extract_general.py --max_depth 5.0 --voxel_size 0.005 --sdf_trunc_scale 4.0 --num_cluster 2` | runtime gate PASS；PID 2320 running at launch；iteration 30000 and 406/406 cameras loaded；completion audit pending | `$RUN_DIR/mesh_manifest.md` |
| 2026-09-03 | Tool C0 r2 mesh completion audit | same | user-operated process/exit/full-log/PLY-header/resource/Git/input-safety audit | PASS；exit 0；406/406 render+TSDF；0 errors/nonfinite；peak 5,480 MiB；raw/post vertices vs old `-1.364%/-1.032%`；C0 reference accepted, tag pending | same + `$RUN_DIR/mesh_manifest.md` |
| 2026-09-03 | Git C0 baseline lock | `c0-baseline -> d6f15c8891a53800d5e3100f95817a7dd7f98e2f`; docs `6145c5787b3d6453a07a28da94bc2f44e26bcc47` | local ref/type/scope checks；non-force push branch and annotated tag | branch/tag first push succeeded；remote verification and final clean-worktree audit follow status commit | n/a |
| 2026-09-03 | E0 TDD foundation + AutoDL train integration | tests `68922cc`,`580adeb`; implementation `b2c46db`,`0601056`,`eaebd8d`; docs `a7d04d4` | bundled Python 19-test suite；AutoDL Python 3.10.21 targeted integration RED→GREEN | 本地 19/19；AutoDL 两次目标 RED 后最终 4/4 GREEN；尚未运行真实训练或 feature-off numerical comparator | n/a |
| 2026-09-03 | E0 full component/CLI gate | `7223f919e8e015f1b1eed2d94d6855aed3b4eb29` | AutoDL stdlib discovery + `train.py --help` exact flag presence | 23/23 PASS；8/8 flags present；post-test clean；等待 paired 500 experiment 授权 | n/a |
| 2026-09-03 | E0 paired-500 baseline half | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | Tool Room `-r 2 --iterations 500 --test_iterations 500 --checkpoint_iterations 500`，private aligned copy + source hash guards | PASS；exit 0；0 error/nonfinite；200,000 points；train PSNR 19.0524567；peak 4,769 MiB；canonical/Git clean | `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-paired-500/pair_20260903T090116Z/baseline_d6f15c88/manifest.md` |
| 2026-09-03 | E0 paired-500 all-off half | `a26082154889ed539322425347af5a57a859a52f` | identical Tool Room protocol + `--seed 0`/Core default-off；semantic checkpoint/app compare + read-only comparator | training/metadata/source/Git PASS；strict equivalence FAIL at `_xyz`, L1/PSNR/PLY SHA；8k stopped pending diagnosis | `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-paired-500/pair_20260903T090116Z/e0_a2608215/manifest.md` |
| 2026-09-03 | E0 remainder/RNG read-only audit | baseline `d6f15c8`; E0 `a260821` | existing checkpoints/configs + fresh-process post-logger RNG states + simulated 500-step camera trace；no training | spatial LR/config/optimizer hyperparameters and RNG/camera trace exact；learned states diverge during training；baseline self-repeat still required | same pair; sentinel `/root/autodl-tmp/e0-rng-sentinel.hTwRyR` |

## Cloud Runs

| Date | Scene | Stage | Commit | Seed | Result path | Decision |
|---|---|---|---|---:|---|---|
| 2026-09-02 | Tool Room | baseline-pathcheck-r4-8k | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/baseline-pathcheck-r4-8k/d6f15c88/seed_0/attempt_20260902T074545Z` | path-check accepted；not C0 reproduced |
| 2026-09-02 | Tool Room | c0-candidate-r2-30k | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z` | complete C0 reference accepted；annotated tag pending user approval |
| 2026-09-03 | Tool Room | E0 paired-500 baseline | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-paired-500/pair_20260903T090116Z/baseline_d6f15c88` | baseline half PASS；等待同 pair E0 all-off |
| 2026-09-03 | Tool Room | E0 paired-500 all-off | `a26082154889ed539322425347af5a57a859a52f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-paired-500/pair_20260903T090116Z/e0_a2608215` | training PASS；strict equivalence FAIL；RNG/camera trace 排除，等待 baseline self-repeat |

## Current Blocker

1. Tool Room C0 reference、`c0-baseline` 与 `research/core-routing` 已完成 local/remote 锁定；Git 基线不再是 E0 blocker。
2. 完整 Core 计划已批准；当前执行边界为 E0。已授权 paired-500 已结束且 strict FAIL；下一次 baseline self-repeat、8k、D0/C1 与正式实验均尚未授权。
3. Utility 未上传不阻塞本次 Tool run，但 G2 跨场景与最终主实验前必须上传并完成 PINHOLE/SIMPLE_PINHOLE undistortion；ScanNet++ GT evaluator 仍需在解释几何结果前冻结。
4. 新服务器 Python/PyTorch/CUDA 与项目 import 已验证；当前 E0 suite 可由标准库 `unittest` 完整执行，pytest 缺失不再阻塞 E0 component 验证，后续若测试使用 pytest-only fixture 再单独申请安装。
5. 当前本地分支为 `research/core-routing`；`main` 与 `c0-baseline` 均保持 baseline SHA，服务器 C0 commit 保持 clean。
