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
- **Status:** g0_A_approved_pending_written_spec_review
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
- [x] 用户批准第二次 exact-baseline 500 self-repeat；范围限定为同 GPU/snapshot/seed/config、新 private view/output 和对既有 baseline/E0 的只读比较，不含 GT/mesh/tag/8k/D0/C1。
- [x] Baseline self-repeat run `baseline2_d6f15c88_20260903T094958Z` 完成且 gate PASS：500/500、exit 0、0 error/nonfinite、200,000 points、canonical/Git clean，服务器恢复 `research/core-routing@287ff08086e687fd8467ea5054ade98e28b8901f`。
- [x] Baseline-1 与 baseline-2 的 PLY/checkpoint SHA 不同，而 baseline cfg_opts 与三次 app model exact；baseline 自身非 bitwise deterministic 已获直接证据。其 L1/PSNR 波动与 E0 相对 baseline-1 的波动处于相近量级。
- [x] 三方 Gaussian 参数与 densification proxy 统计已取得：E0 两个 pair 与 baseline self-repeat 的 RMSE/mean-absolute 整体同阶，且 `knn_f/features_rest/max_weight` 三次 exact。
- [x] 三方报告在 `spatial_lr_scale` scalar JSON 分支再次触发 NumPy `bool_` 序列化错误；已改用统一 Python scalar 转换，并只读补跑缺失 remainder，未重复训练。
- [x] Remainder exit 0：scalar、optimizer hyperparameters/state keys/steps、strict first-difference、app 与 metrics 全部取得；服务器仓库仍 clean `research/core-routing@287ff08086e687fd8467ea5054ade98e28b8901f`。
- [x] 三组 learned parameter/proxy/optimizer moment 误差总体同阶；E0 未出现结构性特有偏离。E0 与 baseline-2 的 L1/PSNR 差仅 `-9.4168e-6/-0.0007416 dB`，但这仍不是预注册 G0 判据。
- [x] 用户于 2026-09-04 选择方案 A：exact 不变量保持严格；已更新字段/proxy/optimizer moment 逐字段分别以 RMSE/MAE 检查 `d_E<=2*d_B`，`d_B=0` 时 exact；标量指标用绝对差。方案已同步进设计 §13、计划和 findings。
- [x] 只读回代现有 500 三方审计 JSON：32 个字段 × 2 种距离共 64 门全部落在批准界内，最大比值约 `1.99`；L1/PSNR 最近 baseline 比值约 `0.293/0.174`。该数据在规则冻结前已观察，只能标为探索性标定，原 strict comparator 结果仍为 FAIL，G0 尚未通过。
- [x] 用户已确认方案 A 书面规格并要求尽快推进；`writing-plans` 详细计划写入 `docs/superpowers/plans/2026-09-04-g0-triplet-equivalence.md`。用户既定禁用多代理，因此执行路径固定为 inline。
- [x] 三方 gate TDD、服务器 hardened gate、真实 checkpoint probe 与 500 versioned exploratory replay 均已完成。
- [x] Task 1 已完成 RED→GREEN：新增 API 首次运行精确因 import 缺失失败；最小实现后目标 comparator suite 13/13 PASS，全部 28 项本地 non-GPU tests PASS，`py_compile`/diff check 返回 0。Commit `286d67f`；既有两方 strict comparator 保持原行为。
- [x] Task 2 的六项 AutoDL CPU-Torch 合同测试已先写入；本机无 Torch 因而整类 skip。下一步在服务器 clean exact test commit 上观察缺少 `audit_feature_off_triplet` 的预期 RED，尚未实现 extractor。
- [x] 用户已明确批准冻结的独立 Tool Room 8k baseline/baseline-repeat/E0 确认组；factor 保持 `2.0`，不因 500 replay 的最大比值 `1.9854507624` 事后放宽。
- **Scope:** 当前可执行 G0 8k preflight、三次串行训练、逐次审计和最终只读 comparator；仍不实现 D0/C1，不修改方法源码/renderer/CUDA，不创建 tag，不安装依赖或启动其他正式实验。

## Experiment Readiness

- **当前阶段：** Phase 0、Tool Room C0、E0 工程、topology-aware comparator RED→GREEN、AutoDL 组件门及 B1/B2 schema-2 dry audit 均已完成。当前冻结并推送规格/计划/证据文档；E0 8k 尚未启动，D0/C1–C6 尚未开始。
- **下一项：** 文档提交推送并让服务器 fast-forward 到精确 approval-record commit；随后按既有授权运行 E0 8k launch gate，不改变 factor、数据、B1/B2 或训练协议。
- **E0 当前门：** 500 三方结果仍仅为探索性；8k B1/B2 是确认组的自重复参照。topology-aware helper、显式 schema v2 integration、AutoDL full GREEN 与 B1/B2 dry audit 已完成；文档同步与 server clean/exact-commit 门通过后即可启动 E0，并在完成后同时检查 count/summary 数值门及配置/输入/trailing-shape/optimizer structure/资源/错误 strict gates。
- **首个诊断实验：** E0/G0 通过后运行 D0 Tool Room seed 0（正式时序到 7k），只记录证据/状态；G1 不通过则停止。
- **首个方法实验：** D0/G1 通过且 C1 single-step gradient oracle 全部通过后，才运行 C1 Tool Room seed 0 quick。C2–C6 依次按上一阶段 tag 晋级，不能并行跳级。
- **当前数据事实：** 服务器 Tool source/DA3/GT 已通过文件数、basename、数值、COLMAP、scale 与 hash preflight，并完成 C0；Utility 尚未上传且本地版本仍需从 FISHEYE 转为 PINHOLE/SIMPLE_PINHOLE。它不阻塞 E0/D0/C1 Tool quick，但阻塞 G2 多场景结论。旧协议为全部相机训练，无 `split.json`，且旧结果没有 ScanNet++ GT geometry metric。
- **当前环境事实：** 本机 Python 3.14.7 无 torch/pytest，不能充当项目验证环境；服务器已验证为仓库锁定的 Python 3.10.21、PyTorch 2.7.1+cu128、CUDA 12.8/RTX 4090。服务器 pytest 尚未安装，E0 前补齐。

### 2026-09-04 G0 Task 2 checkpoint comparator

- [x] AutoDL expected RED：clean `research/core-routing@32f192997ac11b8f03d2b20c9d8656e437fa62f6`，6/6 新测试均因缺少 `scripts.diagnostics.audit_feature_off_triplet` 报错，符合 TDD 合同。
- [x] 本地最小实现已写入；role-specific exact invariant 新测试先失败后通过，comparator 定向 suite 14/14 PASS，脚本与相关测试 `py_compile` PASS，`git diff --check` 无 whitespace error。
- [x] AutoDL 对 `de732f2` 的首轮 GREEN 已执行；本机仍缺 Torch/NumPy，不能用本地 skip 代替服务器证据。完成前 review 后产生的修正需要新的服务器复验。
- [x] 500 artifact replay 已在只读输入上通过；尚未启动 8k、尚未改方法源码或创建 tag。
- [x] AutoDL 首轮实现验证：`de732f2` 定向 6/6、完整 39/39、compile、post-test clean 均 PASS。
- [x] 完成前单代理 code review 对照真实 optimizer/metadata producer；发现 `knn_f` 无 state 与 `git_commit/git_dirty` 两项覆盖缺口，已在 replay 前暂停。
- [x] 三项回归合同与一项 Torch-optional import 合同均先观察 RED；最小修正后本地 4/4 新测试与 14/14 comparator 测试 PASS。服务器复验尚未执行，故 Task 2 仍未关闭。
- [x] AutoDL hardened 复验关闭 Task 2：clean `3db69bb` 上定向 10/10、完整 43/43、compile、post-test clean 与真实 B1 checkpoint schema probe 全部 PASS；下一步只读 replay 三个既有 500-run artifact。
- [x] Task 3 replay：报告 `/root/autodl-tmp/ambisur_diagnostics/e0-g0-a-500-replay-3db69bb.json`，SHA256 `5598ab13…cb0126b`；exact `0` failure、32 字段/64 tensor 门与 2 scalar 门全部通过，且 `exploratory=true/g0_equivalent=false`。下一步必须单独批准独立 8k triplet。
- [x] 用户已单独批准 Task 4 独立 8k triplet；对 500 最大比值接近 2.0 的担忧已记录。factor 继续冻结为 `2.0`，8k 超界时按失败处理而不事后放宽。
- [x] Task 4 Step 2：用户回传 preflight PASS，Git/commit/tag/runtime/GPU/磁盘/数据 manifest/目标目录合同满足；随后按冻结顺序分别启动 B1、B2。

### 2026-09-04 G0 8k baseline pair strict-gate stop

- [x] B1 完成：exit=0，454 s，peak=11,968 MiB；pre log/PLY=1,502,365，post checkpoint=1,360,857；5 个评价点齐全，canonical/prior hash 不变。一次性 deep audit 的 pre/post 混用断言经 baseline 代码和用户只读 reconciliation 纠正，PASS；方法/审计器源码未改。
- [x] B2 完成：exit=0，439 s，peak=10,626 MiB；最终日志/launcher 点数=1,509,961；`B2_COMPLETION_GATE=PASS`，恢复 clean `research/core-routing@5fc8866d6afe287b4e27a341b2a9ecb69d266c74`。
- [x] 识别同口径 pre-topology 数量 exact 门失败（差 7,596），停止 E0。两次都是 baseline；不能据此说 E0 实现错误。完整三元 comparator 未运行，factor=2.0 保持冻结。
- [x] B1/B2 只读对照：actual args/opts/normalized command/launcher contract、optimizer structure/step、输入记录 hash 均相同，health errors=[]；post checkpoint 分别 1,360,857/1,366,889，全部依赖点数的 capture/Adam 张量仅第一维 shape 不同；最早显示差异在首次 densify 的 600 轮（差 1）。
- [x] 用户批准 topology-aware G0 原则：目标为 feature-off 不增加超出 baseline 自重复范围的偏差；动态 topology count/第一维 shape 不再 exact，其余 provenance/schema/structure/safety 门保持 exact。
- [x] 用户批准具体方案 A；最终设计 §13 与独立书面规格已同步：pre/post count 独立 scalar envelope；Gaussian-indexed Tensor 每通道/row-L2 的 mean、population std、7 quantiles 各自独立判门；trailing shape/dtype exact；fixed app 直接 RMSE/MAE；无 row matching/pad/truncate。
- [x] 用户书面 review 后已编写并批准最小实施计划；factor=2.0 不调整，D0/C1 未开始。计划中的 RED→GREEN、AutoDL suite 与 B1/B2 dry audit 已完成。
- [x] 用户已回复“规格通过”；implementation plan 已创建并自审，路径为 `docs/superpowers/plans/2026-09-04-g0-topology-aware-comparator.md`。
- [x] 用户已批准 inline execution；test-only RED commit 已推送并由用户在 AutoDL 执行。
- [x] 用户批准 inline execution；本地 baseline comparator 14/14 PASS、相关 py_compile 与 diff check 返回 0。当前专用 `research/core-routing` 分支原地执行（普通 checkout，非 linked worktree），保留已批准文档改动。
- [x] Task 1 先写 6 项 CPU-Torch 行为测试：固定 summary/quantile、行置换不变、scalar/empty/nonfinite 拒绝、dtype/trailing shape 保留、不同 leading count 独立指标、incompatible channels 不对齐。生产 comparator 尚未修改，等待 AutoDL 观察 RED。
- [x] test-only RED commit `59ae1c9` 已只包含 `tests/gpu/test_feature_off_triplet_audit.py` 并推送到 `origin/research/core-routing`；本地无 Torch，12 项仅能确认 collect/skip，不能冒充 RED。下一步由用户在 AutoDL Python 3.10/Torch 环境运行定向 suite，预期新增 6 项仅因两个接口缺失而 ERROR，既有 6 项继续 PASS。
- [x] AutoDL expected RED 已观察：clean `59ae1c9` 上 12 项中既有 6 项 PASS，新增 6 项仅因 `summarize_gaussian_tensor` / `gaussian_summary_metrics` 尚不存在而 ERROR；return code 1 符合预注册预期，post-test worktree clean。首次 GitHub HTTP/2/RPC 失败未运行测试；启用 network turbo 后重试成功。
- [ ] Task 2 summary helper GREEN：本地最小实现、静态/非 Torch 回归后提交推送，再由用户在 AutoDL 运行同一 12 项 suite。
- [x] Task 2 Steps 1–4：只在只读 auditor 新增固定 quantile、CPU-float64 canonical scalar summary、channel/row-L2 summary、role-wise metrics 与 bounded raw SHA diagnostics；未接入 `build_report`/CLI。local Torch suite 12 项全部明确 skip，既有 comparator 14/14 PASS，相关 `py_compile` 与 `git diff --check` 返回 0（仅 CRLF warning）；不能把 local skip 计作 GREEN。
- [ ] Task 2 Steps 5–6：审查 planned-file diff 后提交/推送 helper candidate，再由用户在 exact commit 的 AutoDL Torch 环境运行 focused 12-test GREEN。
- [x] Task 2 Step 5：单文件 helper candidate `a1643abc31e0dd423a363b9a5ca12d21ce90918a`（`feat: add topology-invariant Gaussian summaries`）已推送；commit 只含 `scripts/diagnostics/audit_feature_off_triplet.py` 138 行，未混入设计/规划文档。
- [x] Task 2 Step 6：AutoDL clean exact `a1643abc31e0dd423a363b9a5ca12d21ce90918a`，Python 3.10.21/Torch 2.7.1+cu128/CUDA available；focused suite 12/12 PASS（0.042 s），return code 0，post-test clean。Task 2 正式完成。
- [x] Task 3 完成：pure topology-aware report assembly、optimizer moment classification、单一 summary failure 与显式 schema-v2 mode 已按预注册 RED→GREEN 实现并完成 AutoDL 组件验证。
- [x] Task 3 Steps 1–3：新增 4 项 Torch integration tests，覆盖 pure evidence、optimizer moment/step/unknown suffix、真实 synthetic artifact 的 schema1/schema2 分流及 CLI 输出；新增 1 项非 Torch independent-q99 failure gate。production integration 未改。本地 Torch 16 项明确 skip，非 Torch comparator 15/15 PASS，test compile/diff check 返回 0。
- [x] Task 3 Step 4：两份测试文件已提交/推送，并在 AutoDL exact test commit 上观察到预期 RED 后才实施 schema 2。
- [x] Task 3 Step 4a：test-only `8c54a539df4de93e4b8dfe664ab150aa0a246966` 已推送；commit 仅含两份测试文件共 280 行。本地 comparator 15/15 PASS、test compile/diff check 返回 0。
- [x] Task 3 Step 4b：AutoDL focused suite RED 已按预注册缺口完成并复核。
- [x] Task 3 Step 4b：AutoDL clean `8c54a539...` 共运行 31 项；27 项 PASS，4 项仅因 assembly helper、`topology_aware` keyword、CLI flag 缺失而 ERROR；return code 1、post-test clean，完全符合预注册 RED。
- [x] Task 3 Steps 5–6 candidate：pure evidence、schema 1/2 显式分流与 CLI wiring 已完成；dependency-free local suite 32/32、comparator 15/15、compile/diff check PASS。Full discovery 仅因本机已知缺 Torch/NumPy 产生 2 import ERROR，16 Torch tests skip；不计作 GREEN。
- [x] Task 3 Step 7：单文件 integration candidate `e781fef23f4f2adec5382808108e7e7e3331e11a`（`fix: make G0 comparison topology-aware`）已推送，未混入训练/方法或规划文档。
- [x] Task 3 Step 8：AutoDL clean `e781fef23f4f2adec5382808108e7e7e3331e11a` focused 16/16 PASS（0.139 s）、full discovery 54/54 PASS（0.449 s）；compile/help rc=0、CLI flag present、post-test clean、`training_started=NO`。
- [x] Task 4 Step 1：不可变 B1/B2 artifact 的显式 `--topology-aware --exploratory` summary-only dry audit 与现有报告 post-validation 均已 PASS；schema/cardinality/resource/hash/clean-status 合格，`g0_equivalent=false`，未把 B2-as-E0 解释为 G0。
- [x] Task 4 Step 1a 环境修正：用户报告 AutoDL 无 `/usr/bin/time`、仅有 Bash `time`。原 dry-audit 未执行；修订命令使用 Python 标准库在 audit 同一进程内记录 wall time 与 Linux `ru_maxrss`，其余 artifact hash、schema、cardinality、factor=2.0 和 clean-status 门保持不变。
- [x] Task 4 Step 1b：对既有 report 的轻量 post-validation 返回 0；91 exact、1 fixed numeric、1,938 scalar、25 Gaussian fields（13 capture + 12 Adam moments）、1,926 summaries 与资源记录全部命中，repo clean。核心 audit 未重跑，E0/训练未启动。
- [x] Task 4 Step 2：cardinality/output review 通过；report 954,155 bytes，wall 91.350 s、peak RSS 3961.5 MiB，无缺失或重复 summary；factor=2.0、字段和统计均未事后调整。
- [x] Task 4 Step 3：冻结的 topology-aware G0 规格、计划与证据文档进入独立 `docs:` 提交并推送；提交范围白名单仅为七份文档，不包含方法/训练源码，不创建 tag。
- [ ] Task 4 Step 4：让 server fast-forward 到该文档提交，复核 exact commit、clean status、B1/B2/数据 hash、磁盘和无并行训练后，才启动已批准的 E0 8k。
- **Scope:** 当前只允许修改只读 comparator helper、对应测试与规划记录；不改训练/renderer/CUDA/方法设计，不创建 tag，不直接操作服务器，不启动 E0/D0/C1。

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
| 2026-09-04 | G0 方案 A 文档化与 500 探索性回代 | docs branch `a2c6e5a` + 当前文档 diff；实验仍为 baseline `d6f15c8` / E0 `a260821` | 解析既有两份三方审计输出；逐字段重算 RMSE/MAE 2× envelope；文档合同检查 | 32 字段/64 数值门探索性通过，最大比值约 1.99；不构成 G0 PASS；无训练/方法源码/tag | 三个既有 500 run，路径见 findings |
| 2026-09-04 | G0 500 versioned exploratory replay | comparator `3db69bb`; runs `d6f15c8`/`d6f15c8`/`a260821` | versioned read-only audit + frozen reconciliation assertions | exact 0 failure；64/64 tensor checks + 2/2 scalar checks PASS；max ratio 1.98545；exploratory only，G0 remains pending | `/root/autodl-tmp/ambisur_diagnostics/e0-g0-a-500-replay-3db69bb.json` |
| 2026-09-10 | G0 8k topology-aware B1/B2 dry audit | comparator `e781fef`; B1/B2 frozen runs, B2 reused only as exploratory E0 role | schema-2 summary audit + stdlib resource record + immutable artifact after-hash + corrected JSON post-validation | audit/post-validation rc=0；91 exact、1 fixed numeric、1,938 scalar、25 Gaussian fields、1,926 summaries；0 failures；wall 91.350 s、peak 3961.5 MiB；repo clean；`exploratory=true/g0_equivalent=false` | `/root/autodl-tmp/ambisur_diagnostics/g0_8k_topology_aware_dry_e781fef_r2.json` |
| 2026-09-10 | Tool Room G0 8k E0 feature-off | E0 `a26082154889ed539322425347af5a57a859a52f`; approval docs `3a1ef87` | user-operated private-view 8k train + completion/hash/metadata/repository audit | E0 run PASS：exit 0、五个评价点、feature-off metadata、输入 after-hash、0 errors、clean restore；L1 `0.02973617`、PSNR `26.029318`、pre points `1,496,374`、peak `11,902 MiB` | `/root/autodl-tmp/ambisur_runs/Tool_Room/g0-triplet-8k/g0_8k_r2_seed0_20260904_v1/e0_a2608215` |
| 2026-09-10 | Tool Room formal topology-aware G0 | comparator `e781fef`; B1/B2 baseline `d6f15c8`; E0 `a260821` | non-exploratory schema-2 audit + stdlib resource + immutable before/after hash + corrected post-validation | valid report but `G0=FAIL`：105/105 exact PASS、fixed/evaluation/count PASS；343/1,938 scalar FAIL，全部 capture 122 + optimizer 221；wall 90.728 s、peak 3947.1 MiB；report SHA `7f513603...e5f` | `/root/autodl-tmp/ambisur_diagnostics/g0_8k_topology_aware_confirmation_e781fef.json` |
| 2026-09-10 | G0 architectural review：目标选择 A | planning docs only；未提交 | 用户确认 G0 首要目标为 feature-off 的可观测训练语义/质量等价，而非内部轨迹逐项一致 | 目标语义已批准；内部 summary 的 gate 角色和独立 confirmation 政策仍待批准；现有 G0 仍按旧合同 FAIL，D0/C1 未授权 | n/a |
| 2026-09-10 | G0 architectural review：内部层选择 A | planning docs only；未提交 | 用户确认 1,926 个 capture/Adam 数值 summaries 改为 diagnostic-only，结构/dtype/step/finite/error 保持 hard | 内部层角色已批准；独立 confirmation 政策仍待批准，尚未改设计稿/comparator 或回写 G0 | n/a |
| 2026-09-10 | G0 architectural review：独立确认选择 A | planning docs only；未提交 | 用户批准冻结新合同后只补跑 unseen E0 8k，并复用冻结 B1/B2 envelope | 当前 E0 降格为 retrospective evidence；新 E0 才可确认修订 G0，完整 gate/spec/comparator 仍待批准，D0/C1 未授权 | n/a |
| 2026-09-11 | G0 behavioral schema-3 规格同步 | six documentation files；未提交 | 将用户批准的完整 hard/diagnostic 划分、schema-1/2 兼容、显式 schema-3、旧 FAIL 保留与 unseen E0 协议同步到最高优先级设计、历史 schema-2 spec、新 behavioral spec 和三份规划文件；自审补充 pre-launch confirmation contract，防止已见 E0 被 CLI 误提升 | 仅文档变更；allowlist=6、`git diff --check`、placeholder/trailing/code-fence、918 个 math delimiters parity、8 个 contract markers 与 stale-current-state scan 均 PASS。书面规格等待用户 review；comparator/训练/方法源码/tag/服务器实验均未改，D0/C1 仍暂停 | `docs/superpowers/specs/2026-09-11-g0-behavioral-equivalence-design.md` |
| 2026-09-14 | G0 behavioral schema-3 书面规格确认与 TDD 计划 | local `research/core-routing@3a1ef87`; documentation only | 用户确认书面规格；按 `writing-plans` 将纯 hard/diagnostic 判定、确认合同、报告/安全门、CLI 兼容和 unseen E0 preflight 拆为五个任务 | 计划已写并完成文档自审，待单独执行批准；未改 comparator 或方法源码，未提交/推送/运行服务器实验，旧 schema-2 FAIL 与 D0/C1 blocker 不变 | `docs/superpowers/plans/2026-09-14-g0-behavioral-comparator.md` |
| 2026-09-14 | G0 schema-3 比较器 TDD 启动 | local `research/core-routing@3a1ef87`; existing 7 documentation changes preserved | 用户仅批准比较器 TDD；核对 skills、AGENTS、规格、计划与 Git 工作树，当前分支已隔离于 baseline | Task 1 纯判定器 RED 将先行；未授权 commit/push、AutoDL、新 E0 或 D0/C1 | `docs/superpowers/plans/2026-09-14-g0-behavioral-comparator.md` |
| 2026-09-14 | G0 schema-3 Task 1 纯 gate RED | local Python 3.14.7；仅新增 `tests/test_behavioral_g0.py` | baseline `tests.test_compare_feature_off` 15/15 PASS；新测试运行因缺少 `scripts.diagnostics.behavioral_g0` 得到唯一预期 `ModuleNotFoundError`，rc=1 | 可补最小纯判定器；尚无 GREEN，不能称组件完成 | local unittest output |
| 2026-09-14 | G0 schema-3 Task 1 纯 gate GREEN | local Python 3.14.7；new `scripts/diagnostics/behavioral_g0.py` | `tests.test_behavioral_g0` 6/6 + 既有比较器 15/15 合计 21/21 PASS；`py_compile` 与 `git diff --check` rc=0 | 仅 Task 1 本地纯判定器验证；无 Torch/AutoDL 集成、commit/push、新 E0 或 G0 PASS | local unittest output |
| 2026-09-14 | G0 schema-3 Task 2 合同/指纹 TDD | local Python 3.14.7；tests-first 4 个新用例 + 纯模块扩展 | 旧 6 项继续 PASS，新 4 项因缺少目标接口先 ERROR；补最小实现后纯比较器+旧比较器 25 项 OK，其中 symlink 逃逸因 Windows 权限 skip；编译/diff check rc=0 | 只确认本地纯逻辑；Linux symlink 与真实 artifact 仍待服务器验证 | local unittest output |
| 2026-09-14 | G0 schema-3 Task 3 Torch RED 准备/环境诊断 | new `tests/gpu/test_feature_off_triplet_audit.py` 六个目标测试；未改报告组装源码 | 本机较早的全仓 64 项产生 2 个既有依赖导入 ERROR、21 skips；`find_spec(torch/numpy)=None`，`train.py` 与相关既有测试无本轮 diff；新 Torch 测试语法编译成功但未观察 RED | 必须先获得 test-only 提交/推送授权，用户按约定在 AutoDL 运行并回传完整输出；不能提前实现 Task 3 生产分支 | local test output |
| 2026-09-14 | G0 schema-3 Task 3 AutoDL Torch RED | test-only `5832336992860f9dc04cb13ed873f2a1aded062f`；clean `research/core-routing`；Torch `2.7.1+cu128` | 用户执行 focused `unittest`：22 项中原有 16 项 PASS，新 6 项均在 `build_report(..., behavioral_g0=True)` 因参数不存在抛 `TypeError`；rc=1，post-test clean，`training_started=NO` | 缺失接口 RED 已观察；尚未验证六个行为断言，Task 3 只能继续本地 comparator 实施，之后必须 AutoDL GREEN | 用户回传 `fcbf1d19-ce5a-4578-b94c-f7fccf4f928c/pasted-text.txt` |
| 2026-09-14 | G0 schema-3 Task 3 本地组装候选 | 未提交的 `scripts/diagnostics/audit_feature_off_triplet.py` 和 `tests/gpu/test_feature_off_dispatch.py`，保留原有全部未提交文档/纯模块 | 新分支仅在 `behavioral_g0=True` 时构造 schema 3；本地纯比较器 25 项 OK（Windows symlink 1 skip）、语法编译与 `git diff --check` rc=0；`git diff d6f15c8 a260821 -- train.py` 显示原训练 backward/step 主体未改 | Torch 集成和剩余字段完整性/空 Tensor 等测试尚未验证；不提交/推送、不称 Task 3 GREEN、不启动新 E0/D0/C1 | local code/test output |
| 2026-09-14 | G0 schema-3 Task 3 AutoDL focused GREEN | 用户批准的双文件 commit `502025a61d301ce4b6c601ea25e618af7109341a`；clean `research/core-routing` | AutoDL Torch `2.7.1+cu128` 上 `tests.gpu.test_feature_off_triplet_audit` + `tests.gpu.test_feature_off_dispatch` 共 27/27 PASS、rc=0、post-test clean、`training_started=NO`；含新增 6 项 schema-3 组装/安全门和 1 项 legacy dispatch | 仅组件级 GREEN；完整 discovery、字段完整性/空 Tensor、CLI/旧报告只读回放及 unseen E0 仍待验证；不回写 G0，不进入 D0/C1 | 用户回传 `741cf0d8-ba98-487c-b7ea-898d94735632/pasted-text.txt` |
| 2026-09-14 | G0 schema-3 Task 3 AutoDL full regression GREEN | 同一精确提交 `502025a61d301ce4b6c601ea25e618af7109341a`；服务器 clean `research/core-routing` | 用户执行 `python -B -m unittest discover -s tests -p 'test_*.py' -v`：61/61 PASS、rc=0、post-test clean、`training_command_invoked=NO` | 旧测试与当前组装测试的完整回归门通过；字段/空 Tensor 边界、schema-3 CLI/确认合同、历史报告回放和 unseen E0 尚未通过，不能称 G0 PASS | 用户回传 `60fdd5da-9672-4179-8896-58a9f21d987f/pasted-text.txt` |
| 2026-09-14 | G0 schema-3 空 app Tensor 预期 RED | test-only `c81aabc8eacd06400a296b776eb992e40f705ce7`，只改 `tests/gpu/test_feature_off_triplet_audit.py` | AutoDL exact HEAD 跑单项测试，得到 `AssertionError: ValueError not raised`；1 FAIL、rc=1，post-test clean，`training_started=NO` | 证明 `appear_ab` 全空时现有 schema-3 接受伪等价；仅本地补了 schema-3 专用非空检查，尚待服务器 GREEN；旧 G0 仍 FAIL | 用户本轮回传 |
| 2026-09-14 | G0 schema-3 空 app Tensor GREEN | 单文件修复 `ff7a877c63a8a1c3227ea1e3d5bedf315b15228e`，只加 `build_report` 的 schema-3 非空门 | AutoDL exact HEAD 单项 1/1 PASS、全仓 discovery 62/62 PASS、rc=0、post-test clean、`training_started=NO` | 边界修复回归门通过；schema-3 CLI/确认合同、历史报告回放和 unseen E0 仍未完成，正式 G0 保持 FAIL | 用户回传 `86c2bacc-b266-48ed-9212-7466b0535da0/pasted-text.txt` |
| 2026-09-14 | G0 schema-3 CLI 接线 RED | test-only `f6704ddb459db8161fc82970dc1a98265ebe0501`，只增三项 CLI 测试 | AutoDL clean exact HEAD 三项均因 argparse 不认识 `--behavioral-g0` 而 `SystemExit(2)`；3 ERROR、rc=1、无训练 | 预期缺失接口 RED；本地仅接 exploratory schema-3 CLI，正式确认仍 fail-closed，Torch GREEN 尚未观察 | 用户回传 `b3ac854e-3528-4675-8a43-6f6780653332/pasted-text.txt` |
| 2026-09-14 | G0 schema-3 exploratory CLI GREEN | 三文件比较器/纯测试 `a9951424675d15026d57ff04f14d8cf9986aa1d4`；服务器 clean exact HEAD | AutoDL 三项 targeted CLI 3/3 PASS、全仓 75/75 PASS、rc=0、post-test clean、`training_started=NO`；正式确认入口仍 fail-closed | 仅 exploratory CLI 及回归门通过；历史 schema-2 报告回放、正式合同/不可变性接线和 unseen E0 未完成，G0 仍 FAIL | 用户回传 `ef2e5809-56f5-41a0-9bce-6dbafa185990/pasted-text.txt` |
| 2026-09-14 | G0 历史报告只读回放 | clean `research/core-routing@a9951424675d15026d57ff04f14d8cf9986aa1d4`，冻结 B1/B2/旧 E0 | 新 diagnostics 路径重算 schema-2：rc=1、SHA 完全复现 `7f513603...e5f`、343 numeric FAIL；同一旧 E0 的 schema-3 exploratory：1,926 diagnostics、343 outliers、hard exact/numeric 0 failure、`g0_equivalent=false`；输入 run artifact 前后哈希一致、repo clean、无训练 | 旧 FAIL 未被改写且 retrospective 不能晋级；正式合同接线与独立 unseen E0 仍待完成 | 用户本轮回传 |
| 2026-09-14 | G0 正式合同接线 RED | test-only `ac910266b97247517900968ed2f42c6e2c344b6e`，仅 `tests/gpu/test_feature_off_triplet_audit.py` | AutoDL exact clean HEAD 三项：有效合成合同被未实现入口阻断导致输出不存在、错误 SHA 尚未读到、输入指纹函数未接入；2 ERROR+1 FAIL、rc=1、无训练 | 预期正式接线 RED；本地仅有 read-only CLI 合同/指纹接线候选，需 AutoDL GREEN 和完整回归，不能开始 unseen E0 | 用户本轮回传 |
| 2026-09-14 | G0 正式合同接线 GREEN | 单文件比较器 `301e69781861c32e61100251e5ebe150940a4e7c` | AutoDL clean exact HEAD 合成正式合同、错误 SHA、输入突变 3/3 PASS；全仓 78/78 PASS、rc=0、post-test clean、`training_started=NO` | 比较器组件已具备正式合同/前后指纹硬门；实际 unseen E0 的预运行合同尚未冻结，未启动训练/正式审计，G0 仍 FAIL，D0/C1 未授权 | 用户回传 `17624b84-60f1-4735-addd-15303063188d/pasted-text.txt` |
| 2026-09-14 | G0 unseen E0 pre-launch 合同冻结 | clean `research/core-routing@ff319d5a4ddc35ec48914111899da3146d86fc31`；`c0-baseline -> d6f15c8`；ID `g0_schema3_unseen_e0_20260914T080510Z_4fc982625600` | AutoDL 无训练预检：旧 B1/B2 指纹分别 `817eb436c074a19769f5a1f181a477d6845b0e51eec0adb58ee5814de16d4eb9`、`113ff302087d12a4bfdf5150f02138edcb54f1043a134433d9b18875645ad18b`；canonical source/prior 当场内容树哈希已写合同；新 run/view 均不存在；空闲约 40 GB、RTX 4090、无训练进程；合同 SHA `16ec0f8c9161ec5254c337ab918eda096023af07aceca5e6a2a80a6a823b0ab3`、独立预检记录 SHA `9f2051937e740bdfbe501601ad51c17eeffc09f52dd0f6f20bd0a13ac920882f` | 仅合同冻结，未建 view/run、未启动 E0；须先核对并单独批准一次 8k 训练，然后用该合同进行正式 schema-3 G0；当前 G0 仍 FAIL | `/root/autodl-tmp/ambisur_diagnostics/g0_schema3_unseen_e0_20260914T080510Z_4fc982625600.{confirmation,preflight}.json` |
| 2026-09-14 | G0 schema-3 unseen E0 训练 | E0 code `a26082154889ed539322425347af5a57a859a52f`；launcher restore target `ff319d5a4ddc35ec48914111899da3146d86fc31` | 用户操作同构 private view、冻结 Tool Room r2/seed0/8k 命令；completion/input/repository gates | exit 0；`Training complete.` 一次；start/end `08:16:36Z/08:23:50Z`；peak 10,770 MiB；iteration 8000 L1 `0.0297586594`、PSNR `26.0253731`、pre-topology points `1,500,565`；dataset/prior after-SHA 匹配；clean branch restore | `/root/autodl-tmp/ambisur_runs/Tool_Room/g0-schema3-unseen-8k/g0_schema3_unseen_e0_20260914T080510Z_4fc982625600/e0_a2608215` |
| 2026-09-14 | 正式 behavioral schema-3 G0 | comparator `ff319d5...` 含 `301e697...` formal wiring；frozen contract SHA `16ec0f8c...b0ab3` | 非 exploratory、topology-aware、hash-pinned confirmation；B1/B2 + unseen E0；审计前后不可变指纹 | `G0=PASS`：schema 3、`g0_equivalent=true`、hard exact/numeric failures `0/0`；346 internal numerical outliers 保留为 diagnostic-only；report SHA `3f6f7e7a9bf302e192ffe48694a1df0ac05c9cb957da76bd909d0b7a6f9879a8`；无训练重启、服务器 repo clean（仅落后 origin 1 commit） | `/root/autodl-tmp/ambisur_diagnostics/g0_schema3_unseen_e0_20260914T080510Z_4fc982625600.formal-schema3.json` |
| 2026-09-15 | D0 TDD 第一批纯合同测试准备 | local `research/core-routing@9db5e1fa9c8fb3921cd0f766255d2627be6ca55a` | 用户授权 D0；新增 test-only `tests/test_reliability_evidence.py`、`tests/test_arbitration.py`，锁定动态 SH、A/S/N、confidence、joint validity/K EMA、五状态优先级和 `H_enter=3` | 尚未改生产代码；本机缺项目 Torch，先做语法/diff 检查，必须由用户在 AutoDL clean commit 运行并观察因目标模块缺失产生的 RED 后才能补最小实现；未启动训练/C1/tag | local pending test-only diff |
| 2026-09-15 | D0 纯证据/仲裁 RED 已观察与最小 GREEN 候选 | test-only `946031a8a9270690ded136e3768cd1e885ee32ac`；AutoDL Torch 2.7.1 | 用户运行两模块，精确因 `reliability.evidence`、`reliability.arbitration` 缺失产生 2 个 import ERROR、rc=1、post-test clean、无训练；随后本地仅新增这两个模块并给 `CoreConfig` 加冻结阈值字段 | 生产候选覆盖当前 22 个纯合同测试目标；本地仅完成 `py_compile` 与 `git diff --check`，因无 Torch 尚不能称 GREEN，须提交后由用户运行 AutoDL focused/full suite；renderer/CUDA/train/topology 均未改 | local uncommitted production candidate |
| 2026-09-15 | D0 纯证据/仲裁 GREEN | `7c3f3b6b4d71b8c1610f1537c8458fdef664775e`；AutoDL Python 3.10.21/Torch 2.7.1+cu128 | focused evidence/arbitration/CoreConfig 26/26；全仓 `unittest discover` 100/100；`py_compile`；`git diff --exit-code` | 全部 PASS，服务器 clean、无训练。D0 第 1 步完成；仅证明纯 A/S/N/confidence、joint K/EMA、候选五状态与 `H_enter=3`，尚未证明 P/G 重投影、renderer/CUDA accumulation、shadow isolation 或 G1 | 用户回传 `dc1f7b91-99c7-4ba8-b0c4-492087411b58/pasted-text.txt` |
| 2026-09-15 | D0 第 2 步 geometry/topology test-only 准备 | local after `7c3f3b6b4d71b8c1610f1537c8458fdef664775e` | 新增 `tests/test_reprojection_reliability.py` 与 `tests/test_topology_migration.py`，锁定 5% foreground-occlusion、frame/positive/finite mask、relative depth、normal sign invariance、P/G T-V-r 分离、first-history invalid、`new_to_old=-1` 新点清零 | 仅测试和进度文档；目标纯函数及 topology 模块尚未实现，需 AutoDL 观察缺失接口 RED；未改 renderer/CUDA/train/GaussianModel、未训练 | local pending test-only diff |
| 2026-09-15 | D0 第 2 步 RED 已观察与最小 GREEN 候选 | test-only `4e8342752f419eca663f8d6f195f67346a9b1c55`；AutoDL clean Torch runtime | 新测试精确因四个 evidence 接口缺失及 `reliability.topology` 不存在产生 2 import ERROR、rc=1、无训练；本地随后仅扩展纯 `evidence.py` 并新增纯 `topology.py` | 实现 projection 后 validity/error、P/G reliability 组合、history stability、`TopologyChange/migrate_tensor`；尚仅 `py_compile`/diff 检查，须 AutoDL focused/full GREEN；未接入 GaussianModel topology 或 renderer/CUDA/train | local uncommitted production candidate |
| 2026-09-15 | D0 第 2 步 geometry/topology GREEN | `b76ebb377b83d71d25368a1a279314a3fbcbd65e`；AutoDL Python 3.10.21/Torch 2.7.1+cu128 | focused 33/33；全仓 `unittest discover` 111/111；`py_compile`/静态检查；server Git clean | 全部 PASS、无训练。D0 第 2 步完成；只证明纯重投影/reliability/history/topology migration 合同，尚未接入 renderer/CUDA、GaussianModel 或训练 shadow path | 用户回传 `efe910e2-b5f0-4fc6-b9f9-cc3b9be3378d/pasted-text.txt` |
| 2026-09-15 | D0 第 3 步 renderer/CUDA accumulator test-only 准备 | local after `b76ebb377b83d71d25368a1a279314a3fbcbd65e` | 审计确认现有 `GaussianRasterizer.forward` 只有五输出，CUDA blend loop 在接受贡献后以 `alpha*T` 更新透射率且未暴露逐 Gaussian 权重；新增 GPU RED 锁定 optional `[E,H,W]` values/validity、`[P,E]` numerator/denominator、single-Gaussian rendered-alpha oracle、empty-E、legacy五输出、stop-gradient | 计划采用独立 forward-only/no-grad evidence 入口；不改现有 backward，严禁用 `out_observe/out_all_map` 近似。当前仅测试/文档，须先在 AutoDL 观察缺失 keyword RED | local pending test-only diff |
| 2026-09-15 | D0 第 3 步 CUDA accumulator RED 已观察与最小 GREEN 候选 | test-only `5d06b4a0eaf562a5c3bb619471dcdb16eac69a41`；AutoDL CUDA runtime | legacy 五输出 1 项 PASS；其余 4 项均精确因 `GaussianRasterizer.forward` 不接受 `evidence_values` 而 ERROR，rc=1、server clean、无训练 | 本地候选新增独立 pybind forward-only/no-grad 入口与真实 `alpha*T` `[P,E]` atomic sums；legacy Python autograd/backward 返回合同未改。尚须 AutoDL 重编译扩展并运行 focused/full GREEN；未接入 `gaussian_renderer` 或训练 | local uncommitted production candidate |
| 2026-09-15 | D0 第 3 步 CUDA accumulator GREEN | `5e653b4b521b29a7850573fb9d2d2a7b3499ddf7`；AutoDL Torch 2.7.1+cu128/CUDA Toolkit 12.8.93/RTX 4090 | private temp copy wheel build rc=0；新 binding present；focused CUDA oracle 5/5；全仓 116/116；`py_compile`/diff check；server clean | 全部 PASS、无训练。证明真实 `alpha*T` weighted sums、legacy五输出 exact、empty-E、输入合同和 detached outputs；D0 第 3 步完成，尚未证明 shadow runtime/gradient/topology isolation | 用户回传 `df2f2bf6-5cb9-4122-b1fe-cbe4def03c3a/pasted-text.txt` |
| 2026-09-15 | D0 第 4 步 EvidenceAccumulator state test-only 准备 | local after `5e653b4b521b29a7850573fb9d2d2a7b3499ddf7` | 新增 CPU RED 锁定 refresh snapshot、H-enter=3、joint-invalid 历史隔离、survivor/new-row topology migration、versioned state round-trip 与无 GT/mesh 输入 | 当前仅测试/文档，生产 `EvidenceAccumulator/EvidenceRefreshInputs` 尚不存在；未改 renderer/train/GaussianModel，未训练 | local pending test-only diff |
| 2026-09-15 | D0 第 4 步 state RED 已观察与最小 GREEN 候选 | test-only `910b1c1b74976063eb3418680fb3e47e1e415803`；AutoDL Torch runtime | 模块精确因 `EvidenceAccumulator` 缺失产生 1 import ERROR，rc=1、server clean、无训练 | 本地候选仅在 `reliability/evidence.py` 组合已验证纯函数，保存 K/迟滞/几何历史，支持显式 topology migration 与 versioned state dict；尚须 AutoDL focused/full GREEN | local uncommitted production candidate |
| 2026-09-15 | D0 第 4 步 persistent state GREEN | `a90dd92bfe2bb16a01305005bb6c5e5e0585638f`；AutoDL Python 3.10.21/Torch 2.7.1+cu128 | focused state/evidence/geometry/arbitration/topology 38/38；全仓 121/121；静态/工作树检查 clean；`training_started=NO` | Step 4 完成：snapshot detached、joint-invalid K history 隔离、H-enter=3、显式 topology migration 与 versioned state round-trip 均获服务器证据；尚未接 high-level renderer 或训练 shadow path | 用户回传 `eb718cd7-c9a6-46ea-a6aa-a6f90a52f871/pasted-text.txt` |
| 2026-09-15 | D0 第 5 步 high-level renderer adapter test-only 准备 | local after `a90dd92bfe2bb16a01305005bb6c5e5e0585638f` | 新增真实 GPU boundary RED：feature-off 公共返回键不变；可选 evidence 输入返回 `[P,E]` detached numerator/denominator；Gaussian 参数 `.grad` 与 densification state 不写；单边输入 fail-closed | 当前仅测试/文档，`gaussian_renderer.render` 尚不接受 evidence 参数；须先在 AutoDL 观察预期 keyword RED，再补最小适配。未改训练、未启动实验 | local pending test-only diff |
| 2026-09-15 | D0 第 5 步 renderer adapter RED 已观察与最小 GREEN 候选 | test-only `6f2dc4c9eff7af296928868f2e31f460b612d787`；AutoDL Torch/CUDA runtime | feature-off 1/1 PASS；两个 evidence 用例均精确因 high-level `render()` 不接受 `evidence_values` 而 ERROR，rc=1；server clean、无训练 | 根因位于公共 renderer signature/五输出固定解包，底层七输出 CUDA path 已由 Step 3 验证。本地候选只在显式 evidence 请求时转发/暴露两项 detached sums；feature-off 不传新 keyword。尚须 AutoDL focused/full GREEN | 用户本轮回传 |
| 2026-09-15 | D0 第 5 步 high-level renderer adapter GREEN | `bff78e3dec4b1c833b2dd44374efbe72bbde27af`；AutoDL Torch 2.7.1+cu128/RTX 4090 | evidence binding present；focused renderer/CUDA/feature-off 13/13；全仓 124/124；static/safety clean；`training_started=NO` | Step 5 完成：显式 evidence path 可由 public renderer 消费且 detached；feature-off 公共返回合同、参数 `.grad` 和 densification state 未写。尚未接训练 refresh/runtime | 用户回传 `c04f8c16-5927-48fa-b19d-e7da3ec6c64a/pasted-text.txt` |
| 2026-09-15 | D0 第 6 步 shadow runtime/checkpoint/diagnostics test-only 准备 | local after `bff78e3dec4b1c833b2dd44374efbe72bbde27af` | 新增 CPU RED 锁定 interval=1000、同轮去重、collector+refresh 全程 no-grad、Parameter/grad/Adam/densification proxy 不写、topology migration、runtime state round-trip、legacy/Core checkpoint 双 schema 和 no-GT `.npz/.jsonl` 不覆盖日志 | 仅测试/文档；`reliability.shadow`、`reliability.diagnostics` 与 `parse_checkpoint_payload` 尚不存在。未解除 `train.py` Core guard、未训练 | local pending test-only diff |
| 2026-09-15 | D0 第 6 步 shadow runtime/checkpoint/diagnostics RED 已观察与最小 GREEN 候选 | test-only `50b3bb1fdb749109dd42c9f22698148451071e25`；AutoDL Python 3.10.21 | 三个测试模块分别精确因 `reliability.shadow`、`reliability.diagnostics`、`parse_checkpoint_payload` 缺失而 import ERROR；rc=1、server clean、无训练 | 预期接口缺失 RED。随后本地只新增 shadow 调度/状态、no-GT 不覆盖 writer 和 checkpoint 双 schema parser；尚须 AutoDL focused/full GREEN。未改 `train.py`、renderer/CUDA、optimizer/topology 执行 | local uncommitted production candidate |

## Cloud Runs

| Date | Scene | Stage | Commit | Seed | Result path | Decision |
|---|---|---|---|---:|---|---|
| 2026-09-04 | Tool Room | G0 8k B1 | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/g0-triplet-8k/g0_8k_r2_seed0_20260904_v1/b1_d6f15c88` | training complete；save-order reconciliation PASS；pre=1,502,365/post=1,360,857 |
| 2026-09-04 | Tool Room | G0 8k B2 | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/g0-triplet-8k/g0_8k_r2_seed0_20260904_v1/b2_d6f15c88` | completion PASS；pre=1,509,961/post=1,366,889；baseline-pair topology exact FAIL，配置/来源/optimizer 对照相同，STOP before E0 |
| 2026-09-10 | Tool Room | G0 8k E0 | `a26082154889ed539322425347af5a57a859a52f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/g0-triplet-8k/g0_8k_r2_seed0_20260904_v1/e0_a2608215` | training/completion PASS；pre=1,496,374；旧 schema-2 下因 343 个内部 summary envelope FAIL，结论保持 FAIL；对后续 schema-3 仅作 retrospective evidence |
| 2026-09-02 | Tool Room | baseline-pathcheck-r4-8k | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/baseline-pathcheck-r4-8k/d6f15c88/seed_0/attempt_20260902T074545Z` | path-check accepted；not C0 reproduced |
| 2026-09-02 | Tool Room | c0-candidate-r2-30k | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z` | complete C0 reference accepted；annotated tag pending user approval |
| 2026-09-03 | Tool Room | E0 paired-500 baseline | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-paired-500/pair_20260903T090116Z/baseline_d6f15c88` | baseline half PASS；等待同 pair E0 all-off |
| 2026-09-03 | Tool Room | E0 paired-500 all-off | `a26082154889ed539322425347af5a57a859a52f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-paired-500/pair_20260903T090116Z/e0_a2608215` | training PASS；strict equivalence FAIL；RNG/camera trace 排除，等待 baseline self-repeat |
| 2026-09-03 | Tool Room | E0 baseline self-repeat | `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` | 0 | `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-baseline-self-repeat/baseline2_d6f15c88_20260903T094958Z` | run gate PASS；证明 baseline 非 bitwise；用于方案 A 的 500 探索性标定 |

## Current Blocker

1. Tool Room C0 reference、`c0-baseline` 与 `research/core-routing` 已完成 local/remote 锁定；Git 基线不再是 E0 blocker。
2. 历史 schema-2 G0 的 343 项内部 summary FAIL 原样保留；按后来批准且预先冻结的 behavioral schema-3 合同，独立 unseen E0 正式审计已经 `g0_equivalent=true`，hard exact/numeric failures `0/0`，因此 Phase 1 的 behavioral G0 门已通过。346 项内部 outlier 仍是诊断证据，不解释为内部轨迹一致。用户已经授权 D0 TDD；当前 D0 Steps 1–5 已在 AutoDL 分别通过 pure formulas/arbitration、geometry/topology、真实 CUDA accumulator、persistent state 和 high-level renderer adapter 的 focused/full 回归门。当前阻塞是训练 shadow runtime、checkpoint/topology 接线和日志尚未完成；C1、Supporting、阶段 tag 和正式 D0 训练仍未获准。
3. Utility 未上传不阻塞本次 Tool run，但 G2 跨场景与最终主实验前必须上传并完成 PINHOLE/SIMPLE_PINHOLE undistortion；ScanNet++ GT evaluator 仍需在解释几何结果前冻结。
4. 新服务器 Python/PyTorch/CUDA 与项目 import 已验证；当前 E0 suite 可由标准库 `unittest` 完整执行，pytest 缺失不再阻塞 E0 component 验证，后续若测试使用 pytest-only fixture 再单独申请安装。
5. 用户回传的最新服务器验证提交为 clean `research/core-routing@bff78e3dec4b1c833b2dd44374efbe72bbde27af`。当前本地只准备 Step 6 shadow runtime/checkpoint/diagnostics RED 及同步记录；不修改 baseline/结果资产，不创建 tag，也不启动 D0/C1 实验。
