# AmbiSuR Core Implementation Plan

> **For agentic workers:** 实施时必须先读本文件与 `docs/research/ambisur-reliability-routing-design.md`，使用 `superpowers:test-driven-development`；按用户选择使用 `superpowers:executing-plans`。除非用户明确授权，不使用多代理。每个阶段完成后使用 `superpowers:requesting-code-review`，声称完成前使用 `superpowers:verification-before-completion`。

**Goal:** 在不混入 Supporting 模块的前提下，完成可复现、可消融的 AmbiSuR Core：观测校准、双可靠性、五状态仲裁、参数级梯度路由和 Gaussian 生命周期，并通过 G0–G2。

**Architecture:** 保留 feature-off legacy 训练路径；Core-active 路径由 `EvidenceAccumulator`、纯函数五状态机、`GradientRouter` 和 `LifecycleManager` 组成。证据刷新通过可选、forward-only 的 CUDA Gaussian evidence accumulator 精确汇聚 `sg(alpha*T)`；连续参数只做一次 optimizer step，离散 topology 在 step 后提交并用显式索引映射迁移状态。

**Tech Stack:** baseline runtime 以仓库 `environment.yml` 为准：Python 3.10、PyTorch 2.7.1+cu128、CUDA 12.8 custom extension、NumPy 1.26.3/Open3D；当前 E0 测试兼容标准库 `unittest`，无需为此安装 pytest；单张 RTX 4090 24GB。设计稿 §1 的 Python 3.12/PyTorch 2.8 是待批准修正的环境描述，不作为运行阻断条件。

**Spec:** `docs/research/ambisur-reliability-routing-design.md`

## Global Constraints

- 设计稿是最高优先级合同；已识别缺口必须逐项获用户确认，不得静默选算法。
- 本计划只实现 Core；禁止新增解耦曝光、local Ray-Color、Ray-Normal、逐 Gaussian truncation。
- D0 只记录证据/状态，不能改变 loss、Parameter `.grad`、optimizer、densify/prune 或 topology。
- C0–C6 严格嵌套；每一阶段只增加表中唯一能力。
- GT mesh 仅离线 D0 标签/最终评价读取，不得进入训练、checkpoint 或 evidence cache。
- baseline 结果目录只读；所有 smoke/实验使用 commit/stage/seed 唯一新目录。
- 上传的数据集目录是只读 canonical asset；由于 baseline loader 会写 `sparse_da3_aligned/0/points3D.ply`，任何训练都必须以 canonical asset 构建独立 working view，并把 `--source_path` 指向该 view，禁止直接指向上传目录。
- 默认 feature-off 必须走现有 legacy path，保留 baseline loss、Ray-Color、ALR、checkpoint 和 topology 时序，直到 E0 证明等价。
- **正式澄清 A（2026-09-01 已批准）：** Core C1–C6 的 prior contract 固定为 baseline `depth_weight=0.1` DA3 depth 与 `unc_weight=0.1` Dual-End ALR，保持原启用时序、权重、confidence/uncertainty 有效区域；C1 唯一方法变量是逐 Gaussian 以 `N_i` 门控 clean prior geometry gradient。
- 设计 §7.1 的新 `L_Pd/L_Pn` 不修改原文、不进入 Core C1–C6；暂列 Core 后独立 Supporting 候选。未来实施必须有独立阶段、tag 和消融，禁止混入或回写 `c1-need-gate`。
- C1 必须满足 `g_baseline_total = g_B_residual + g_P_clean`、`g_C1 = g_B_residual + N_i*g_P_clean`；`g_P_clean` 不含 Ray-Color，对 SH/曝光严格为 0，baseline Ray-Color 与其他非先验贡献全部留在 `g_B_residual`。
- C1–C5 的 viewspace/densification proxy 固定为 baseline total-gradient，不随 N/仲裁路由，不得在 C6 前改变 topology。
- **K joint-validity 澄清 A（2026-09-02 已批准）：** `Z_i^PG=sum(sg(w)*m_PG)`，`V_i^PG=1[Z_i^PG>tau_Z]`，复用 `tau_Z=1e-4`。只有当前 `V_i^PG=1` 才计算有效 `K_raw`、初始化/更新 K EMA并允许 K 仲裁；历史 K 永远不能绕过当前 validity。
- Bypass 与单方可靠状态不依赖 `V_i^PG`。双方均可靠但 joint-invalid 时：D0/C3–C6 为 Abstain，C2 回退 `g_B`；`Delta` 只记诊断，禁止选边。C4–C6 继承 C3 状态，不重新解释 K。
- **生命周期状态澄清 A（2026-09-02 已批准）：** `s_i` 只表示经过迟滞的稳定五状态仲裁结果，`ell_i` 只表示生命周期状态；所有 clone/split/prune 门只消费 `ell_i`。正常映射 `ell_i=M(s_i)` 仅在证据刷新执行，刷新间保持不变。
- 新 Gaussian 在 topology 提交时强制 `ell_i=Probation`，至少保持 `C_prob=500` 个 optimizer iteration；期间 `s_i` 可计算/记录但不得覆盖 `ell_i`。满 500 轮后只在下一次证据刷新按当时稳定 `s_i` 完成首次映射；`d_i` 只按证据刷新计数，生命周期变化时重置为 1。
- **Probation truncation 澄清 A（2026-09-02 已批准）：** Core C6 不设置或迁移逐 Gaussian truncation 状态；Probation 与全部 Gaussian 一样原样沿用 baseline 全局 `--trunc_sigma/--disable_trunc`（当前默认 `trunc_sigma=2.0`）。Core 不新增 truncation Tensor、不按 `ell_i` 改写全局参数、不修改 renderer/CUDA truncation 接口；设计 §9.4 仅为 G0–G2 后可独立立项、tag 和消融的 Supporting 候选。
- 任何 GPU 实验前记录 40 位 SHA、空 `git status --short`、`git diff --exit-code=0`、resolved config、seed、scene、命令和环境。
- 完整 Core 实施计划仍未获最终批准；不得修改方法源码或创建/移动 branch/tag/commit/push。用户已于 2026-09-02 单独批准先运行一次 Tool Room clean-baseline；该有限授权只覆盖服务器只读 preflight 通过后的 `seed 0 / -r 4 / 8000 iterations` 新目录运行，不授权 E0 方法改动或其他实验。
- 服务器操作固定采用 user-operated terminal：助手一次给出可审计命令，用户复制执行并回传完整输出，助手复核通过后才给下一步；助手不直接控制浏览器、SSH 或服务器终端。

---

## Goal

在不混入 Supporting 模块的前提下，完成可复现、可消融的 AmbiSuR Core：观测校准、双可靠性、五状态仲裁、参数级梯度路由和 Gaussian 生命周期，并通过 G0–G2。

## Current Phase

Phase 0 已完成；2026-09-03 起仅按用户批准的 E0 边界进入测试与 default-off 工程实现。

## Phases

### Phase 0：接管与规格核对
- [x] 读取全部指定文件与 skills
- [x] 记录 baseline candidate、Git/remote/tag/submodule 状态
- [x] 核对所有公式输入、renderer/CUDA/gradient/topology/checkpoint/evaluation 路径
- [x] 生成详细 Core implementation plan
- [x] 用户批准 C1 prior contract 澄清 A 与 residual oracle 约束
- [x] 用户批准 K joint-validity 澄清 A、EMA 与 C2/C3 行为
- [x] 用户批准生命周期 `s_i`/`ell_i` 澄清 A、Probation 释放时序与 `d_i` 计数语义
- [x] 用户批准 Probation truncation 澄清 A：Core 保持 baseline 全局配置，不新增逐 Gaussian/state-driven truncation
- [x] 本轮审计发现的 architectural specification gaps 已逐项闭合
- [x] 从只读旧结果恢复 Tool Room r2/r4 baseline 训练与 mesh 提取命令、完整 cfg 和运行时摘要
- [x] 用户报告已按 frozen contract 上传 Tool Room canonical source/GT，并批准一次 `-r 4` 8k clean-baseline
- [x] 用户终端只读 preflight 验证 Git、406 image/depth/conf、COLMAP、aligned scale、GT、CUDA 与 `train` import；唯一失败为 preflight 错把 Python 3.12 当要求，仓库实际锁定 3.10.21
- [x] 用户最终批准完整实施计划，并授权当前仅执行 E0；D0/C1、renderer/CUDA、阶段 tag 和服务器实验仍未授权
- [x] 用户于 2026-09-03 在已审计的 C0 run 目录启动冻结协议 mesh 提取；runtime gate 通过，launcher PID `2317`、mesh PID `2320`，启动日志已加载 iteration 30000 模型与 406/406 cameras
- [x] Mesh completion audit：exit 0，406/406 render/TSDF 完成，错误/NaN/Inf 为 0，峰值 5,480 MiB，Git/canonical source/GT 不变；raw/post mesh 相对旧 r2 的 vertices 差 `-1.364%/-1.032%`、faces 差 `-1.050%/-0.918%`
- [x] 接受 `main@d6f15c8891a53800d5e3100f95817a7dd7f98e2f` 的 Tool Room r2/30k/seed0 运行作为当前可复现 C0 锚点；这不是对历史输出的 bitwise 等价声明，也不替代 E0 feature-off 严格等价测试
- **Status:** complete

### Phase 1：E0 与测试基础
- [x] 锁定 `c0-baseline` annotated tag 于 `d6f15c8891a53800d5e3100f95817a7dd7f98e2f`，并从该提交创建累计分支 `research/core-routing`
- [x] 建立 19 项可由标准库 `unittest`/pytest 共同执行的非 GPU 合同测试，以及服务器 `train.py` integration test 入口
- [ ] 验证所有新增开关关闭时等价
- **Status:** in_progress_paired_500_iteration_experiment_approved_waiting_for_launch

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
| 2026-09-01 | `planning-with-files` 会话衔接脚本无法通过 `python` 启动（命令不存在） | 1 | 改用 Codex workspace bundled Python；脚本成功且未报告未同步上下文 |
| 2026-09-01 | 首次 Git 审计脚本被 JavaScript/PowerShell 反引号组合解析失败 | 1 | 删除格式字符串中的反引号后只读命令成功 |
| 2026-09-01 | bundled Git 的 `git submodule status` helper 缺少 `basename/sed/git-sh-setup` | 1 | 发现仓库无 `.gitmodules`；改用系统 Git与目录证据交叉核验，不修改 PATH |
| 2026-09-01 | 阶段完整性检查脚本的 PowerShell 变量冒号/JavaScript 模板插值冲突 | 2 | 改用 PowerShell 字符串拼接后检查通过；不影响仓库内容 |
| 2026-09-01 | A 澄清覆盖检查中的 Markdown 反引号截断 JavaScript 模板字符串 | 1 | 覆盖断言改用不含 Markdown 反引号的稳定子串；脚本未执行、未改变仓库状态 |
| 2026-09-01 | A 澄清覆盖检查用 SimpleMatch 未跨越 Markdown code-span 反引号 | 1 | 对 shape/optimizer 文本改用允许 code-span 的 regex；内容本身已存在 |
| 2026-09-02 | K 澄清自审补丁未匹配设计稿末尾汇总表的精确空格 | 1 | 补丁事务未应用；改用 `rg` 取得精确行后拆分为两个小补丁 |
| 2026-09-02 | K 合同覆盖断言用 SimpleMatch 未跨越 `otherwise -> g_B` 的 Markdown code spans | 1 | 改用不含反引号的独立稳定子串；合同文本本身已存在 |
| 2026-09-02 | 首次读取 Chrome control skill 时误把 `chrome/...` 写成 `browser/...` | 1 | 根据 available-skills 根映射定位正确文件；未初始化或操作任何浏览器，随后按用户约定恢复为 user-operated terminal |
| 2026-09-02 | runtime correction 的首个跨三文件补丁因 findings 上下文匹配失败而整笔未应用 | 1 | 确认无部分写入后改为逐文件小补丁；以 `environment.yml` 和用户终端输出为证据 |
| 2026-09-02 | path-check 记录验证脚本把点数差精确期望手算成 0.624740%，与实际 0.624754% 不符 | 1 | 用 `(1166901-1159656)/1159656` 重新计算；文档只写四舍五入 0.625%，内容无需修正 |
| 2026-09-02 | C0 完成记录的首个多文件补丁使用了错误的连字符状态文本，事务未应用 | 1 | 用 `rg` 读取实际 underscore 状态后拆分为逐文件补丁；无部分写入 |
| 2026-09-03 | 切回 Core 分支时设计稿被另一 Windows 进程独占，Git 无法 unlink；用户侧已有两个内容相同的 stash | 1 | 停止强制操作；确认 working/stash/remote blob 相同后由用户解除锁并恢复到 clean `research/core-routing`，未丢失内容 |
| 2026-09-03 | 首次 E0 skill/plan 读取调用的 JavaScript 封装含非法字符，脚本未启动 | 1 | 改为较小的规范 `exec_command` 调用；只读读取成功，仓库未受影响 |
| 2026-09-03 | metadata 测试把 `Path("data/scene")` 的字符串结果硬编码为 POSIX `/`，Windows GREEN 首轮失败 | 1 | 生产逻辑保持不变；测试改为独立的 `str(Path(...))` 平台原生期望，随后 19 项非 GPU suite 全部通过 |
| 2026-09-03 | AutoDL E0 integration test 在 clean `research/core-routing@b2c46db49e3465da7ff5cfda56a7ddd30be6f02c` 上 RED | 1 | Python 3.10.21 成功导入 `train.py`，随后因 `train` 尚未导出 `build_checkpoint_payload` 精确失败；确认目标接口缺口后才进入最小 `train.py` 接线 |
| 2026-09-03 | 本地把 GPU integration 目录包含进 bundled Python discovery，因该运行时无 `torch` 导入失败 | 1 | 栈追踪确认失败发生在 `train.py:import torch`，不是 E0 回归；随后只运行 19 项非 GPU suite（PASS），GPU integration 继续由既有 AutoDL Python 3.10/torch 环境验证，不安装本地依赖 |
| 2026-09-03 | AutoDL 第二次 E0 integration test 在 clean `research/core-routing@02ac3b9700ea53a9f723ef3f62c3c8cac1b15d42` 上 RED | 1 | legacy dispatch、tuple checkpoint 与 `core_config=None` 三项通过；唯一失败精确为 `prepare_output_and_logger()` 仍是二参数签名。确认元数据接线缺口后才增加 logger/CLI wiring |
| 2026-09-03 | AutoDL E0 integration GREEN 在 clean `research/core-routing@a7d04d4bbd28aa025f1d09373e8e7d1e615bf688` 上完成 | 1 | Python 3.10.21 标准库 `unittest` 4/4 PASS（0.097 s）；logger 实际创建临时输出并写元数据，测试后工作树仍 clean。下一步是完整 component suite/CLI 检查，不把 integration GREEN 写成 G0 数值等价 |
| 2026-09-03 | AutoDL E0 full component gate 在 clean `research/core-routing@7223f919e8e015f1b1eed2d94d6855aed3b4eb29` 上完成 | 1 | 标准库 `unittest` 23/23 PASS（0.211 s）；`train.py --help` 返回 0，seed、shadow 与六级 Core flags 全部存在，测试后工作树 clean。仅证明工程组件门，不证明 500-step/8k feature-off 数值等价 |
| 2026-09-03 | 本地 PowerShell 将未引用的 annotated-tag peel 表达式 `c0-baseline^{}` 误解析，commit 查询失败并产生无关输出 | 1 | 未修改仓库；改用跨 shell 稳定的 `git rev-list -n 1 c0-baseline`，确认 peeled commit 为 `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` |

### 2026-09-03 E0 paired-500 authorization

- [x] 用户批准 Tool Room、`-r 2`、seed 0、500 iterations 的 baseline/E0 all-off 成对实验。
- [x] baseline 固定为 `c0-baseline@d6f15c8891a53800d5e3100f95817a7dd7f98e2f`；E0 固定为批准记录提交后的 `research/core-routing` HEAD。
- [x] 两次运行使用同一 canonical snapshot、各自独立 private working view 和新输出目录；保存 iteration 500 point cloud/checkpoint。
- [x] 本次不使用 GT、不提取 mesh、不创建 tag、不启用 Core、不并行占用 GPU；baseline 成功后才启动 E0。
- [ ] 两次运行完成后执行只读 comparator；任何错误、非有限值、输入污染或差异立即停止，不进入 8k。

## Scope Guardrails

- 当前计划禁止实现曝光、Ray-Color、Ray-Normal、自适应截断。
- GT mesh 不进入训练。
- C0–C6 每阶段必须对应独立 commit 和验证 tag。
- 服务器不修改代码。
- 当前授权仅允许 E0：测试框架、default-off Core 配置/调度、显式 seed、复现元数据和只读 comparator；禁止 D0/C1、renderer/CUDA、Supporting、阶段 tag、依赖安装和服务器实验，后者均须另行批准。

## Planned File Boundaries and Interfaces

实施时先用以下接口锁定边界；名称/类型改变必须回到计划审查，不在编码中临时漂移。

```text
reliability/config.py
  CoreConfig: all Core capabilities default False; seed defaults 0
  CoreConfig.enabled_features() -> tuple[str, ...]
  CoreConfig.validate() -> None — reject non-nested stage combinations

reliability/evidence.py
  EvidenceAccumulator.refresh(iteration: int, cameras, gaussians, render_fn, pipe, background: Tensor) -> EvidenceSnapshot
  EvidenceAccumulator.on_topology_change(change: TopologyChange) -> None
  EvidenceAccumulator.state_dict() -> dict[str, Tensor]
  EvidenceAccumulator.load_state_dict(state: dict[str, Tensor], point_count: int) -> None
  compute_pg_consistency(weighted_support: Tensor, depth_error_sum: Tensor, normal_error_sum: Tensor, tau_z: float=1e-4) -> PGConsistency
  PGConsistency fields: Z_pg:[P] float32, V_pg:[P] bool, K_raw:[P] float32 — K_raw elements are consumable only where V_pg=True
  KEMAState.update(raw: Tensor | None, valid: Tensor) -> None
  KEMAState fields: value:[P] float32, initialized:[P] bool, current_valid:[P] bool
  EvidenceSnapshot fields:
    A,S,N,T_p,V_p,r_p,T_g,V_g,r_g,Z_pg,V_pg,K_raw,K,K_initialized,delta,M_obs,M_p,M_g — each leading dimension P

reliability/arbitration.py
  candidate_state(snapshot: EvidenceSnapshot, cfg: CoreConfig) -> Tensor[P,int8]
  ArbitrationStateMachine.update(candidate: Tensor[P,int8]) -> Tensor[P,int8]

reliability/gradient_router.py
  GradientRouter.collect(base_loss, prior_loss, gaussian_inputs, app_inputs, screen_inputs) -> SeparatedGradients
  GradientRouter.route(stage: str, grads: SeparatedGradients, evidence: EvidenceSnapshot, state: Tensor) -> dict[str, Tensor]
  GradientRouter.route_c1(g_b: Tensor, g_p: Tensor, need: Tensor) -> Tensor
  GradientRouter.assign_once(routed: dict[str, Tensor]) -> None

reliability/topology.py
  TopologyChange.new_to_old: Tensor[P_new,int64] — surviving old index; -1 for new Gaussian
  TopologyChange.is_new: Tensor[P_new,bool]

reliability/lifecycle.py
  LifecycleManager fields: lifecycle_state:[P]int8, state_duration:[P]int32, born_iteration:[P]int64
  LifecycleManager.update_from_arbitration(stable_state: Tensor[P,int8], iteration: int, evidence_refresh: bool) -> Tensor[P,int8]
  LifecycleManager.gate_candidates(candidates: TopologyCandidates, iteration: int, evidence: EvidenceSnapshot) -> TopologyCandidates
  LifecycleManager.on_topology_change(change: TopologyChange, iteration: int) -> None
  LifecycleManager.state_dict() -> dict[str, Tensor]
  LifecycleManager.load_state_dict(state: dict[str, Tensor], point_count: int) -> None

train.py
  select_training_path(cfg: CoreConfig) -> Literal["legacy", "core"]
  build_checkpoint_payload(gaussian_state, iteration, cfg: CoreConfig, core_state=None) -> tuple | dict
```

文件责任：`evidence.py` 只算/缓存证据；`arbitration.py` 只做状态；`gradient_router.py` 只处理已分离梯度；`lifecycle.py` 只门控离散事件；`diagnostics.py` 只序列化无 GT 训练诊断；`scene/gaussian_model.py` 只负责 Parameter/Adam 的 topology 操作并返回 mapping；`train.py` 只编排调用。

## Universal Test and Experiment Protocol

每个阶段都执行以下顺序，任一步失败即停止，不启动后续成本更高的步骤。

1. CPU focused tests：运行该阶段列出的精确 test files。
2. CPU full suite：`python -m pytest tests -m "not gpu" -q`。
3. 静态合同：Core flags 默认/嵌套、无绝对数据路径、训练模块不引用 GT mesh、Supporting 未新增。
4. AutoDL CUDA/component tests：`python -m pytest tests/gpu -m gpu -q`。
5. AutoDL 100–500 iteration smoke：Tool Room、seed 0、新输出目录；证据/路由组件用 GPU integration test 以合成 iteration/state 覆盖实际激活分支，避免缩改正式 30k 时序。
6. Feature-off comparison：同一环境、输入、seed、iteration 与 c0；比较 loss trace、checkpoint tensor、Gaussian count、峰值显存和 wall time。
7. Tool Room seed 0 quick：使用正式时序和相同 mesh/evaluation protocol。
8. 只有阶段门槛通过才创建不可移动 tag；然后才进入下一阶段。

通用停止条件：feature-off 差异无法解释；NaN/Inf/CUDA error；额外隐式 backward；`.grad` 非有限；状态长度不等于 P；resume 不一致；dirty worktree/manifest 缺失；写入 baseline 目录；峰值显存≥22GB 或 Core 预计耗时>2×；单阶段混入 Supporting/无关重构。

## Frozen ScanNet++ Data Contract

旧结果只读根目录 `D:\research_Space\output\AmbiSuR_original\ScanNetpp` 已核验。Tool Room r2/r4 均使用 406 张图、同一 COLMAP/DA3 先验和默认训练参数；区别仅为 `-r 2` 与 `-r 4` 及输出目录。新服务器上传目录固定为 canonical asset，不直接作为训练工作区：

```text
/root/autodl-tmp/ambisur_data/
├── source/ScanNetpp/
│   ├── Tool_Room/colmap_undistorted/
│   │   ├── images/                         # 406 images；文件名/大小写与 COLMAP 完全一致
│   │   ├── sparse/0/
│   │   │   ├── cameras.txt                 # PINHOLE/SIMPLE_PINHOLE；binary 三件套也可
│   │   │   ├── images.txt
│   │   │   └── points3D.txt
│   │   ├── estimated_depths/
│   │   │   └── <image_filename>.npy        # 例 DSC06274.JPG.npy
│   │   ├── estimated_confs/
│   │   │   └── <image_filename>.npy
│   │   ├── sparse_da3/0/                   # 原始 DA3 COLMAP model；用于 provenance/复算
│   │   └── sparse_da3_aligned/0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       ├── points3D.bin
│   │       └── trans.json                  # 至少含 scale
│   └── Utility_Room/colmap_undistorted/    # 同一结构；相机必须先转成 PINHOLE/SIMPLE_PINHOLE
└── gt/ScanNetpp/
    ├── Tool_Room/mesh_aligned_0.05.ply
    └── Utility_Room/mesh_aligned_0.05.ply
```

训练 source 与 GT 物理分离；GT 根目录不得进入 training config、cache 或 checkpoint。C0/E0 为复现旧协议，`colmap_undistorted` 根目录不得放 `split.json`，且命令不传 `--eval`，即全部相机用于训练；若以后需要 held-out appearance split，必须作为独立评价协议决策，不能静默混入 C0–C6。

上传完成后创建每次运行独立的 view：

```text
/root/autodl-tmp/ambisur_work/data_views/<dataset_snapshot>/<run_id>/colmap_undistorted/
```

view 中 `images/estimated_depths/estimated_confs/sparse` 可只读链接 canonical asset，`sparse_da3_aligned` 必须复制到该 run 私有目录，因为当前 `scene/dataset_readers.py::readColmapSceneInfo` 会无条件写 `points3D.ply`。运行前记录 canonical 文件清单、总字节数和 SHA256 manifest；任何 DA3 重新生成都产生新的 dataset snapshot ID，不得冒充旧 snapshot。

旧结果恢复出的命令合同：

```bash
python train.py --source_path <working_view>/colmap_undistorted --model_path <new_run_dir> -r 2
python train.py --source_path <working_view>/colmap_undistorted --model_path <new_run_dir> -r 4
python mesh_extract/extract_general.py --model_path <new_run_dir> --max_depth 5.0 --voxel_size 0.005 --sdf_trunc_scale 4.0 --num_cluster 2
```

建议 `-r 4` 仅用于首次 8k 基线路径/smoke，正式 C0–C6 主协议候选为旧结果中分辨率更高的 `-r 2`；该分辨率角色须在第一次运行前由用户批准并写入 manifest。

## Strict Nesting Contract

| Level | Inherits | This level's only new experimental variable | Computed but non-executing |
|---|---|---|---|
| C0 / E0 | verified baseline | none; default-off harness and metadata only | none |
| D0 | E0 | shadow evidence/state logging only; no training variable | A/S/N, P/G reliability, K/Delta, five states |
| C1 | D0 | multiply clean **baseline DA3 depth + Dual-End ALR** geometry gradient by N；新 `L_Pd/L_Pn` excluded | reliability/states/lifecycle |
| C2 | C1 | coarse forced dual-reliability arbitration, no Abstain | fine group table/projection/lifecycle |
| C3 | C2 | execute Abstain in coarse routing | fine group table/projection/lifecycle |
| C4 | C3 | replace coarse geo action with parameter-group table | projection/lifecycle |
| C5 | C4 | conflict projection on retained dual paths | lifecycle |
| C6 | C5 | reliability-driven lifecycle/topology execution | Supporting remains absent |

任何 level 的配置 diff 除该行唯一变量、配套参数/日志/测试外必须为空；否则该 run 不能归入该消融级别。

## Stage E0 — Baseline Lock and Feature-Off Equivalence

**唯一变量：** 无方法能力；只建立测试、显式复现元数据和 default-off dispatch。`c0-baseline` 仍指向验证过的 baseline commit，而不是 E0 scaffolding commit。

**Files/functions:** modify `arguments/__init__.py::{ModelParams,OptimizationParams,get_combined_args}`, `train.py::{training,__main__,prepare_output_and_logger,select_training_path,build_checkpoint_payload}`, `utils/general_utils.py::safe_state`; create `reliability/config.py::CoreConfig`, `tests/test_core_config.py`, `tests/test_seed_contract.py`, `tests/gpu/test_feature_off_dispatch.py`, `scripts/diagnostics/compare_feature_off.py::{load_run,compare_runs}`.

**先写失败测试：**

```python
def test_all_core_flags_default_off():
    cfg = CoreConfig()
    assert cfg.seed == 0
    assert cfg.enabled_features() == ()

def test_nonnested_core_flags_are_rejected():
    cfg = dataclasses.replace(CoreConfig(), enable_gradient_projection=True)
    with pytest.raises(ValueError, match="requires parameter routing"):
        cfg.validate()

def test_core_off_selects_legacy_training_path():
    assert select_training_path(CoreConfig()) == "legacy"

def test_feature_off_checkpoint_uses_legacy_tuple_schema(legacy_components):
    payload = build_checkpoint_payload(*legacy_components, CoreConfig())
    assert isinstance(payload, tuple) and len(payload) == 2
```

**CPU command:** `python -m pytest tests/test_core_config.py tests/test_seed_contract.py -q`；随后运行通用 CPU full suite。

**最小实现步骤：**

- [ ] 在干净 `main@d6f15c8891a53800d5e3100f95817a7dd7f98e2f` 用现有代码跑 Tool Room seed 0 baseline，不修改代码；确认 7001 之后 ALR/Ray-Color 路径可运行。
- [ ] baseline 前验证 source view：406 个 image/COLMAP pose/DA3 depth/conf 文件名一一对应；相机模型为 PINHOLE/SIMPLE_PINHOLE；`sparse_da3_aligned/0/trans.json` 和 `points3D.bin` 存在；canonical asset checksum 未变化。
- [ ] baseline 通过后才建议创建 annotated `c0-baseline` 指向该 40 位 SHA；若失败，使用 systematic-debugging 单独形成 baseline fix 方案，不能把修复混进 C1。
- [x] 从 `c0-baseline` 创建累计分支 `research/core-routing`；规划文件改动在该分支提交，main 不再推进方法代码。
- [x] 添加 default-false Core flags、shadow mode、`--seed` 默认 0 与 nested validation；AutoDL 已证明 all-off 选择 legacy、checkpoint 保持 tuple 且训练入口显式接收 CoreConfig。
- [x] 输出 `resolved_config.json` 和运行身份，并把显式 seed/CoreConfig 接入 CLI；AutoDL integration 4/4 GREEN。尚未进行训练数值等价验证。
- [x] 实现只读 comparator，比较两次新目录，不接触 baseline assets。

**Feature-off / CPU / GPU：** CPU 检查 flags、seed、schema；AutoDL 500 iter c0 vs E0 all-off；Tool Room seed 0 快速等价运行建议 8k，必须覆盖 densify(600+)、trim(1000+)、Ray-Color(5001+) 与 ALR(7001+)。

**晋级：** G0 通过；默认 loss/grad/checkpoint/Gaussian count 在确定的容差合同内等价；无额外 backward/显存增长。**停止：** baseline 本身在 7001 路径失败、DA3 artifact 缺失/重生成但未建立新 snapshot、source 被写、无冻结的分辨率/evaluator、测试环境不可用或工作树不净。

**Commits/tags:** `test: add Core feature-off equivalence harness`; `feat: add disabled Core configuration and run metadata`. Tag only `c0-baseline` on the verified main commit; E0 scaffolding不另打 tag。

## Stage D0 — Shadow Evidence and Five-State Diagnostics

**唯一变量：** 计算/记录证据与状态；训练数值和 topology 必须与 E0 feature-off 相同。D0 不进入 C0–C6性能表、不打规定阶段 tag。

**Files/functions:** create `reliability/evidence.py::{EvidenceAccumulator.refresh,on_topology_change,state_dict,load_state_dict}`, `reliability/arbitration.py::{candidate_state,ArbitrationStateMachine.update}`, `reliability/diagnostics.py::write_snapshot`, `reliability/topology.py::TopologyChange`, `scripts/diagnostics/evaluate_d0.py::{load_snapshot,label_geometry_risk,evaluate}` and the listed CPU/GPU tests; modify `arguments/__init__.py::OptimizationParams`, `train.py::{training,build_checkpoint_payload}`, `gaussian_renderer/__init__.py::render`, `scene/gaussian_model.py::{densify_and_clone,densify_and_split,prune_points,densify_and_prune,capture,restore}`, extension wrapper `GaussianRasterizer.forward`, `rasterize_points.{h,cu}::RasterizeGaussiansCUDA`, and `cuda_rasterizer/{forward.h,forward.cu,rasterizer.h,rasterizer_impl.cu}::{FORWARD::render,Rasterizer::forward}`.

**先写失败测试：** dynamic SH degrees; `out_observe>0`; S count/angle; N boundaries; robust confidence; P/G reprojection masks; T/V separation; first-history invalid; EMA/new-point reset; joint support/K validity/EMA transitions; all arbitration branches and Henter=3; CUDA weighted sums; D0 leaves parameters/optimizer/topology unchanged.

```python
@pytest.mark.parametrize(("degree", "count"), [(1, 3), (2, 8), (3, 15)])
def test_sh_non_dc_counts_are_3_8_15(degree, count):
    sh = torch.ones(2, 15, 3)
    assert active_non_dc(sh, degree).shape == (2, count, 3)

def test_pixel_hits_are_binarized_per_view():
    hits = torch.tensor([[100, 0], [1, 0], [0, 8]])
    assert torch.equal(view_count(hits), torch.tensor([2, 1]))

def test_shadow_refresh_does_not_write_parameter_grad(shadow_fixture):
    before = clone_parameters_and_optimizer(shadow_fixture.gaussians)
    shadow_fixture.refresh()
    assert_parameters_and_optimizer_equal(before, shadow_fixture.gaussians)

def test_new_gaussian_does_not_inherit_parent_ema(evidence_state):
    change = TopologyChange(new_to_old=torch.tensor([0, 1, -1]), is_new=torch.tensor([False, False, True]))
    evidence_state.on_topology_change(change)
    assert not evidence_state.ema_valid[-1]

def test_zero_joint_support_has_no_arbitration_k():
    result = compute_pg_consistency(
        weighted_support=torch.tensor([0.0]),
        depth_error_sum=torch.tensor([0.0]),
        normal_error_sum=torch.tensor([0.0]),
        tau_z=1e-4,
    )
    assert torch.equal(result.Z_pg, torch.tensor([0.0]))
    assert not result.V_pg.item()
    snap = make_snapshot(N=0.8, H_p=True, H_g=True, V_pg=False, K=torch.ones(1))
    assert candidate_state(snap, CoreConfig()) == ArbitrationState.ABSTAIN

def test_invalid_joint_refresh_keeps_history_but_not_current_validity(k_ema_state):
    k_ema_state.update(raw=torch.tensor([0.8]), valid=torch.tensor([True]))
    historical = k_ema_state.value.clone()
    k_ema_state.update(raw=None, valid=torch.tensor([False]))
    assert torch.equal(k_ema_state.value, historical)
    assert not k_ema_state.current_valid.item()

def test_first_valid_joint_observation_initializes_k_ema(k_ema_state):
    assert not k_ema_state.initialized.item()
    k_ema_state.update(raw=torch.tensor([0.35]), valid=torch.tensor([True]))
    assert k_ema_state.initialized.item()
    assert torch.equal(k_ema_state.value, torch.tensor([0.35]))

def test_joint_invalid_both_reliable_is_d0_abstain():
    snap = make_snapshot(N=0.8, H_p=True, H_g=True, V_pg=False, K=0.9)
    assert candidate_state(snap, CoreConfig()) == ArbitrationState.ABSTAIN

def test_bypass_and_one_sided_states_ignore_joint_validity():
    assert candidate_state(make_snapshot(N=0.5, H_p=True, H_g=True, V_pg=False), CoreConfig()) == ArbitrationState.BYPASS
    assert candidate_state(make_snapshot(N=0.8, H_p=True, H_g=False, V_pg=False), CoreConfig()) == ArbitrationState.PRIOR_LED
    assert candidate_state(make_snapshot(N=0.8, H_p=False, H_g=True, V_pg=False), CoreConfig()) == ArbitrationState.GEOMETRY_LED

def test_feature_off_never_computes_or_consumes_joint_gate(core_off_spy):
    core_off_spy.run_iteration()
    assert core_off_spy.pg_consistency_calls == 0
    assert core_off_spy.arbitration_calls == 0
    assert_core_off_matches_legacy(core_off_spy)
```

**CPU command:** `python -m pytest tests/test_reliability_evidence.py tests/test_arbitration.py tests/test_topology_migration.py tests/test_d0_isolation.py -q`；CUDA oracle 使用 `tests/gpu/test_evidence_accumulator_cuda.py`。

**最小实现步骤：**

- [ ] 实现 pure evidence formulas、EMA valid masks 与 state machine；所有 tensors detach。
- [ ] 实现 `Z_pg/V_pg/K_raw/K EMA`：`V_pg=0` 时不构造可消费的 `K_raw`、不更新 EMA；历史值与 `k_ema_valid` 分开保存，新点 initialized=False。
- [ ] 扩展 renderer forward 的可选 pixel-evidence 输入和 `[P,E]` 累加输出；empty input branch 不执行 atomics，backward 忽略该输出。
- [ ] 先渲染所有训练视图的 G maps/out_observe，再逐 source 计算 P/G/PG pixel evidence，第二遍 render 累加 Gaussian numerator/denominator；CPU/offload streaming 限制显存。
- [ ] refactor topology 返回 `new_to_old`; legacy action order/random call order不变；D0 只迁移 state，不 gate action。
- [ ] 训练写无 GT `.npz/.jsonl`；除原有指标外记录 `Z_pg/V_pg`、`P(V_pg=1)`、`P(V_pg=1|H_p=H_g=1)`、joint-invalid Abstain 比例和按 scene/stage/N-quantile 的 coverage；离线 evaluator 才读取 GT mesh并比较 joint-valid/invalid 几何误差。
- [ ] Core-active checkpoint 用 versioned dict 保存 evidence/state；feature-off 保持 legacy tuple。

**Feature-off / CPU / GPU：** all D0 flags off 重跑 E0 500；GPU accumulator 对 oracle；500-iter D0 smoke 用 refresh interval 100 仅作工程测试；Tool Room seed 0 正式 D0 运行到 7k，默认 refresh 1000，GT 只离线分析。

**晋级：** G1：N AUROC>0.60 且比 `max(A,1-S)` 对应较优单项至少+0.03；risk-coverage合理、状态不被 Bypass/Abstain单一占据；D0 reconstruction/grad/topology等价。**停止：** G1失败、joint coverage 近零或导致几乎全 Abstain、state collapse、CUDA accumulator不等于 oracle、任何 GT 路径进入训练/checkpoint。coverage 失败只报告，不得自动改用 Delta、降低 `tau_Z` 或绕过 validity。

**Commits/tags:** `test: specify Core evidence and arbitration`; `feat: add forward-only Gaussian evidence accumulation`; `feat: add D0 shadow reliability diagnostics`. No D0 tag.

## Stage C1 — Observation-Calibrated Need Gate

**唯一变量：** `g_P_geo` 乘 `N_i`；不使用双可靠性选边、Abstain、细粒度表、projection或lifecycle。

**Files/functions:** create/modify `reliability/gradient_router.py::{GradientRouter.collect,route_c1,assign_once}`, `tests/test_gradient_router_c1.py`, `tests/gpu/test_single_step_router.py`; modify `train.py::training`, `arguments/__init__.py::OptimizationParams`, `reliability/config.py::CoreConfig.validate`.

**正式实施澄清 A（approved 2026-09-01）：**

- C1–C6 沿用 baseline `depth_weight=0.1` DA3 depth 和 `unc_weight=0.1` Dual-End ALR，保持 `train.py` 的原启用时序、权重、confidence threshold、`rendered_unc.detach()` mask 与其他有效区域定义。
- 设计 §7.1 新 `L_Pd/L_Pn` 与 baseline prior 不同且权重不闭合；为维护 C0–C6 单变量消融，本 ladder 不实施它。该公式保留在设计稿原文，作为 Core 后独立 Supporting 候选；未来必须使用独立阶段/tag/消融，不能修改已有 C1 结果或 tag。
- 对 optimizer Gaussian groups 定义 `g_baseline_total = g_B_residual + g_P_clean`；其中 `g_P_clean` 只含上述 baseline depth/ALR 对 xyz、rotation、scaling、opacity 的几何梯度，不含 Ray-Color，且对 SH、appearance/exposure 严格为 0。
- C1 只计算 `g_C1 = g_B_residual + broadcast(N_i)*g_P_clean`。SH、appearance/exposure 和未纳入 prior 的组始终取 `g_B_residual == g_baseline_total`；viewspace/densification proxy 绕过路由，直接使用冻结的 baseline total-gradient。

**失败测试与实现：**

```python
def test_c1_n_zero_and_one_gate_prior_geo_gradient():
    g_b = torch.ones(2, 3)
    g_p = torch.full((2, 3), 2.0)
    routed = GradientRouter.route_c1(g_b, g_p, torch.tensor([0.0, 1.0]))
    assert torch.equal(routed[0], g_b[0])
    assert torch.equal(routed[1], g_b[1] + g_p[1])

def test_c1_need_one_reproduces_baseline_total_gradient(router_fixture):
    grads = router_fixture.collect_with_baseline_residual()
    routed = router_fixture.route_c1(grads, need=torch.ones(grads.point_count))
    assert_gradients_close(routed, grads.baseline_total)

def test_prior_gradient_to_sh_and_exposure_is_zero(router_fixture):
    grads = router_fixture.collect_prior()
    assert torch.count_nonzero(grads.sh) == 0
    assert grads.exposure is None

def test_router_assigns_grad_once_and_optimizer_steps_once(router_fixture):
    router_fixture.run_iteration()
    assert router_fixture.optimizer_step_calls == 1
    assert router_fixture.backward_calls == 0
```

**CPU command:** `python -m pytest tests/test_gradient_router_c1.py -q`；单步 gradient oracle 使用 `tests/gpu/test_single_step_router.py`。

**GPU 单步 gradient oracle（必须先失败再实现）：** 使用相同初始 Gaussian/AppModel/Adam state、camera、随机状态和 renderer 输入，分别执行 untouched legacy 单步与 C1 单步；每个 case 同时测试 `N=zeros(P)` 与 `N=ones(P)`。实际代码采用 strict `>`，因此必须覆盖边界前后：

| Iterations | DA3 depth | Ray-Color | ALR | 目的 |
|---|---:|---:|---:|---|
| `1000`, `1001` | off → on | off | off | depth `iteration > 1000` 边界 |
| `5000`, `5001` | on | off → on | off | Ray-Color `iteration > 5000` 边界 |
| `7000`, `7001` | on | on | off → on | ALR `iteration > unc_from_iter`，默认 `unc_from_iter=7000` |
| `15000`, `15001` | on | on → off | on | Ray-Color 当前表达式在默认 `densify_until_iter=15000` 时包含 15000、从 15001 关闭 |

每个 iteration case 必须逐项比较：

- `g_baseline_total` 与 `g_B_residual+g_P_clean`；`N=1` 的 C1 与 baseline total；`N=0` 与 `g_baseline_total-g_P_clean`。
- xyz `[P,3]`、rotation `[P,4]`、scaling `[P,3]`、opacity `[P,1]`、SH DC/rest，以及 AppModel/曝光参数的 gradient/None 状态。
- `g_P_clean` 的 SH 和 AppModel/曝光必须为 exact zero/None；Ray-Color 不得进入 `g_P_clean`。
- `viewspace_points`、`viewspace_points_abs` 和 `viewspace_point_tensor` 的 densification proxy 必须等于 legacy baseline total，不受 N 影响；有 ALR 和无 ALR case 都要覆盖。
- 从相同 Adam state 执行唯一一次 `optimizer.step()` 后的所有参数；记录每组 max-abs、max-rel、cosine error。

固定验收容差：有限 float32 tensor 使用 `torch.testing.assert_close(rtol=1e-5, atol=1e-7)`；zero/None 合同用 exact assertion；optimizer step 次数必须严格为 1。若 untouched baseline 自身重复运行不能满足该容差，或 decomposition 任一组/边界失败，先报告原始误差并停止，不得事后放宽容差。

- [ ] 按批准的 baseline prior contract 分离 depth/ALR 与其余贡献；先以原两次 backward 语义取得 `g_baseline_total`，再用 `ray_reg=0` 的 prior graph 取得 `g_P_clean`，定义 `g_B_residual=g_baseline_total-g_P_clean`。这样 `N=1` 回到 baseline（包括既有隐式 Ray-Color 残差），而路由的 prior 不含 Ray-Color。
- [ ] 用 `autograd.grad` 收集分量，不调用 `.backward()` 累积到 Parameter；执行 finite check/prior clip/N mask，手动赋 `.grad` 后一次 step。先用单步 GPU oracle 证明 residual decomposition 与 baseline 两次 backward 等价。
- [ ] C1始终保留 SH/exposure 的 `g_B_residual`；C1–C5 的 screen-space densification proxy 使用冻结的 baseline total-gradient 定义，不由 N/状态路由，确保 topology 新变量只在 C6 引入。
- [ ] 禁止用 `out_observe`、`out_all_map` 或其他 forward proxy 近似 `g_baseline_total/g_P_clean`；若 autograd 分离无法满足固定容差，形成阻断报告并停止 C1，不得进入 C2。

**验证：** feature-off 500；CPU N=0/1/中间值；GPU single-step gradient oracle；500-iter integration smoke；Tool Room seed 0 production quick 10k（至少3k active routing）并用同协议提取/evaluate。

**晋级/停止：** 全部 strict-boundary oracle 在固定容差内通过；日志中 sampled `||gP_after||/||gP_before||` 与 N 一致；一次 step；无灾难性>1% quick几何退化/外观越过 G2 guardrail。任何 Ray-Color 泄入 `g_P_clean`、SH/曝光受 N 影响、proxy 被路由、residual/step 参数不等价、使用近似或 feature-off变化即停，禁止进入 C2。

**Commits/tag:** `test: specify C1 need-gated gradients`; `feat: add C1 observation-calibrated need gate`; 通过后 tag `c1-need-gate`。

## Stage C2 — Coarse Dual-Reliability Forced Arbitration

**唯一变量：** 在 C1 上加入无 Abstain 的整组 `geo={pos,scale,opa}` 双可靠性强制选边/融合；SH/exposure仍为gB。

**Files/functions:** modify `reliability/gradient_router.py::GradientRouter.route_c2`, `reliability/arbitration.py::candidate_state`, `tests/test_gradient_router_c2.py`, `train.py::training` diagnostics and `reliability/config.py::CoreConfig.validate`.

**失败测试：** table-driven Bypass/Consensus/Prior-only/Geometry-only/conflict-P/conflict-G/both-untrusted；所有 K 分支要求 `V_pg=1`；双方均可靠但 `V_pg=0` 回退 `g_B` 且不消费 Delta/K；`Delta>0` 边界；双方打平/无效回退 gB；同一 coarse state 权重应用到全部 geo groups。

**CPU command:** `python -m pytest tests/test_gradient_router_c2.py tests/test_arbitration.py -q`。

- [ ] 实现 `route_c2` 严格按 design `11.2`：Consensus/冲突分支同时要求 `V_pg=1`；joint-invalid 落入 `otherwise -> g_B`，不调用 Abstain/Delta fallback/projection/parameter table。
- [ ] 记录 source choice、状态占比、gB/gP norm 与选边正确率，不改 topology。

**验证：** C1 flags + dual reliability；feature-off和C1回退测试；CPU全表；GPU 500；Tool Room seed0 10k，比较C1且保持相同其他配置。

**晋级/停止：** 每个分支实际出现或用诊断证明数据不触发；选边/冲突指标有合理信号且无系统退化。all-one-source、无可靠性有效点、delta符号错误或配置差异>唯一开关即停。

**Commits/tag:** `test: specify C2 coarse arbitration`; `feat: add C2 coarse dual-reliability arbitration`; tag `c2-coarse-arbitration`。

## Stage C3 — Coarse Abstention

**唯一变量：** 在 C2 上增加 Abstain；几何梯度为0，SH/exposure保留gB；topology仍不由状态门控。

**Files/functions:** modify `reliability/arbitration.py::{candidate_state,ArbitrationStateMachine.update}`, `reliability/gradient_router.py::GradientRouter.route_c3`, `tests/test_arbitration.py`, `tests/test_gradient_router_c3.py`, `reliability/diagnostics.py::write_snapshot`.

**失败测试：** `|Delta|<=delta` conflict、both-low、K threshold、Henter hysteresis；双方均可靠但 `V_pg=0` 必须 Abstain，即使历史 K 高或 `|Delta|>delta`；单方可靠与 Bypass 不受 `V_pg` 影响；Abstain xyz/rotation/scale/opacity全0、SH/app等于gB；C2 flag off时绝不 Abstain。

**CPU command:** `python -m pytest tests/test_arbitration.py tests/test_gradient_router_c3.py -q`。

- [ ] state machine输出稳定五状态；`route_c3`使用完整C3粗粒度公式，所有 K 分支要求当前 `V_pg=1`，joint-invalid 落入最终 Abstain。
- [ ] 记录 Abstain coverage、oracle regret、持续时间/抖动率。

**验证：** feature-off/C2 fallback；CPU全分支；GPU 500；Tool Room seed0 10k，唯一配置差异为 abstention开关。

**晋级/停止：** Abstain非零且不支配全部高需求点，冲突子集错误选边/风险不劣于C2，稳定性/外观守门。状态塌缩、频繁抖动、Abstain仍有geo gradient即停。

**Commits/tag:** `test: specify C3 abstention`; `feat: add C3 abstention`; tag `c3-abstain`。

C4–C6 只消费 C3 已产生的稳定状态；其 executor/lifecycle 不得绕过当前 `V_pg`，也不得用历史 K 或 Delta 重新给 joint-invalid Gaussian 选边。

## Stage C4 — Parameter-Group Routing

**唯一变量：** 把C3整组geo动作细化为 design `7.3` 的 pos/scale/opa/SH直接加权；不投影。

**Files/functions:** modify `reliability/gradient_router.py::{GradientRouter.route_c4,clip_prior_by_group,split_groups,reassemble_groups}`, `train.py::training`; create `tests/test_gradient_router_c4.py`.

**失败测试：** 五状态×四组完整权重表；pos把xyz+rotation共用clip尺度；prior clip≤`2*median(||gB||)`；empty/None/nonfinite；prior SH/app恒0；C3开关组合回退原粗粒度结果。

**CPU command:** `python -m pytest tests/test_gradient_router_c4.py -q`。

- [ ] 实现 group extraction/reassembly和C4 table；`knn_f`保持base/None，不赋予设计外动作。
- [ ] logging记录每状态/组的norm、zero fraction和prior clip rate。

**验证：** feature-off/C3 fallback；CPU矩阵；GPU 500；Tool Room seed0 10k，对比C3。

**晋级/停止：** 所有组权重与oracle逐元素一致，opacity/SH边界无泄漏，显存/时间守门且quick无系统退化。任何projection提前出现或prior更新SH/exposure即停。

**Commits/tag:** `test: specify C4 parameter routing`; `feat: add C4 parameter-group routing`; tag `c4-parameter-routing`。

## Stage C5 — Conflict Gradient Projection

**唯一变量：** 仅在C4保留双路的组上加入 design `7.4` projection。

**Files/functions:** modify `reliability/gradient_router.py::{GradientRouter.route_c5,project_conflicts}`, `reliability/diagnostics.py::write_gradient_stats`; create `tests/test_gradient_projection.py`.

**失败测试：** negative/positive/zero dot；zero dominant；Prior-led dominant P；Consensus dominant按r；Bypass/Geometry/Abstain不投影；投影后辅助与主导dot≥数值容差。

**CPU command:** `python -m pytest tests/test_gradient_projection.py -q`。

- [ ] 顺序锁死为 finite→prior clip→projection→state mask→assign once。
- [ ] 记录pre/post conflict rate、投影范数和按组分布；不改变C4权重表。

**验证：** feature-off/C4 fallback；CPU数学测试；GPU 500；Tool Room seed0 10k，对比C4。

**晋级/停止：** 数学不变量全过，冲突率下降且无NaN/过大范数，quick稳定。主导梯度被改、单路状态被投影、重复step即停。

**Commits/tag:** `test: specify conflict projection`; `feat: add C5 conflict gradient projection`; tag `c5-conflict-projection`。

## Stage C6 — Reliability-Driven Gaussian Lifecycle

**唯一变量：** 在 C5 上增加完整 lifecycle 层：稳定仲裁状态到生命周期状态的刷新时映射、Probation 的 `g_B` 覆盖，以及 clone/split/protect/quarantine/prune 控制；不实现逐 Gaussian truncation。

**Files/functions:** create/modify `reliability/lifecycle.py::{LifecycleManager.update_from_arbitration,gate_candidates,on_topology_change,state_dict,load_state_dict}`, `reliability/topology.py::{TopologyCandidates,TopologyChange,migrate_tensor}`, `tests/test_lifecycle.py`, `tests/test_topology_migration.py`, `tests/gpu/test_lifecycle_integration.py`; modify `scene/gaussian_model.py::{densification_candidates,apply_topology_change,prune_points,capture,restore}`, `train.py::{training,build_checkpoint_payload}`, `reliability/gradient_router.py::GradientRouter.route`, `reliability/diagnostics.py::write_lifecycle_stats`.

**失败测试：** 五种 `s_i`→`ell_i` 映射；非刷新 iteration 不映射；所有 topology gate 只读 `ell_i`；z递推/失败清零；cooldown；`d_i` 只按 evidence refresh；Quarantine 三条件；Probation 精确边界和下一刷新释放；Probation 只路由 `g_B`；C5→C6 的全局 truncation settings 完全相同且 lifecycle/checkpoint 无 per-Gaussian truncation Tensor；Confirmed/Protected prune veto；step-before-Parameter-replace；clone/split/prune 后 Adam/EMA/state/counter/birth-iteration 长度和值；checkpoint/resume 保留剩余 Probation 与 `d_i`。

```python
def test_optimizer_step_happens_before_topology_commit(loop_spy):
    loop_spy.run_iteration()
    assert loop_spy.events.index("optimizer.step") < loop_spy.events.index("topology.commit")

def test_confirmed_and_protected_are_never_pruned(lifecycle):
    candidates = prune_all_candidates(2)
    states = torch.tensor([LifecycleState.CONFIRMED, LifecycleState.PROTECTED])
    assert not lifecycle.gate_prune(candidates, states).any()

def test_new_children_start_probation_with_neutral_evidence(lifecycle_fixture):
    lifecycle_fixture.clone_parent(0)
    child = lifecycle_fixture.point_count - 1
    assert lifecycle_fixture.manager.lifecycle_state[child] == LifecycleState.PROBATION
    assert lifecycle_fixture.manager.state_duration[child] == 0
    assert not lifecycle_fixture.evidence.ema_valid[child]

@pytest.mark.parametrize(("arbitration", "expected"), [
    (ArbitrationState.BYPASS, LifecycleState.NORMAL),
    (ArbitrationState.CONSENSUS, LifecycleState.CONFIRMED),
    (ArbitrationState.PRIOR_LED, LifecycleState.REPAIR),
    (ArbitrationState.GEOMETRY_LED, LifecycleState.PROTECTED),
    (ArbitrationState.ABSTAIN, LifecycleState.QUARANTINE),
])
def test_stable_arbitration_maps_only_at_evidence_refresh(manager, arbitration, expected):
    before = manager.lifecycle_state.clone()
    manager.update_from_arbitration(tensor_state(arbitration), iteration=200, evidence_refresh=False)
    assert torch.equal(manager.lifecycle_state, before)
    manager.update_from_arbitration(tensor_state(arbitration), iteration=200, evidence_refresh=True)
    assert manager.lifecycle_state.item() == expected

def test_probation_expires_only_on_first_eligible_evidence_refresh(new_child_manager):
    manager = new_child_manager(created_at=100)
    manager.update_from_arbitration(tensor_state(ArbitrationState.BYPASS), iteration=599, evidence_refresh=True)
    assert manager.lifecycle_state.item() == LifecycleState.PROBATION
    manager.update_from_arbitration(tensor_state(ArbitrationState.BYPASS), iteration=600, evidence_refresh=False)
    assert manager.lifecycle_state.item() == LifecycleState.PROBATION
    manager.update_from_arbitration(tensor_state(ArbitrationState.BYPASS), iteration=610, evidence_refresh=True)
    assert manager.lifecycle_state.item() == LifecycleState.NORMAL

def test_lifecycle_duration_counts_refreshes_not_iterations(manager):
    manager.update_from_arbitration(tensor_state(ArbitrationState.BYPASS), iteration=100, evidence_refresh=True)
    assert manager.state_duration.item() == 1
    manager.update_from_arbitration(tensor_state(ArbitrationState.BYPASS), iteration=101, evidence_refresh=False)
    assert manager.state_duration.item() == 1
    manager.update_from_arbitration(tensor_state(ArbitrationState.BYPASS), iteration=200, evidence_refresh=True)
    assert manager.state_duration.item() == 2

def test_topology_gate_uses_lifecycle_not_stable_arbitration(probation_fixture):
    probation_fixture.stable_state[:] = ArbitrationState.BYPASS
    gated = probation_fixture.manager.gate_candidates(
        clone_all_candidates(1), iteration=300, evidence=probation_fixture.evidence
    )
    assert not gated.clone.any()

def test_c6_probation_preserves_baseline_global_truncation(render_settings, probation_fixture):
    before = (render_settings.trunc_sigma, render_settings.disable_trunc)
    probation_fixture.manager.update_from_arbitration(
        tensor_state(ArbitrationState.BYPASS), iteration=300, evidence_refresh=True
    )
    after = (render_settings.trunc_sigma, render_settings.disable_trunc)
    assert after == before

def test_lifecycle_state_and_checkpoint_have_no_per_gaussian_truncation(manager):
    assert not hasattr(manager, "trunc_sigma")
    assert "trunc_sigma" not in manager.state_dict()

def test_split_clone_prune_migrate_adam_and_all_core_state(topology_fixture):
    topology_fixture.apply_clone_split_prune()
    p = topology_fixture.gaussians.get_xyz.shape[0]
    assert topology_fixture.all_leading_dimensions() == {p}
```

**CPU command:** `python -m pytest tests/test_lifecycle.py tests/test_topology_migration.py -q`；强制 topology 事件的 GPU integration 使用 `tests/gpu/test_lifecycle_integration.py`。

- [ ] 把baseline candidate生成与apply分开，保持candidate判据不变。
- [ ] `LifecycleManager` 保存 `lifecycle_state=ell`、`state_duration=d` 和 `born_iteration`；只有 `evidence_refresh=True` 才执行五状态映射和 `d` 递推，普通 iteration 不得更新二者。
- [ ] 所有 clone/split/prune gate 只消费 manager 保存的 `ell_i`，不直接比较 `s_i`；提交后 GaussianModel 返回 mapping，统一迁移 Evidence/Arbitration/Lifecycle/Adam。
- [ ] 新点 EMA/history 清零并强制 Probation；`iteration-born_iteration<500` 时只用 `g_B` 且禁 clone/split/普通 prune，满 500 后仍等到第一次 evidence refresh 才按当时稳定 `s_i` 映射。
- [ ] Repair/Protected/Quarantine 按合同；生命周期状态改变时 `d_i=1`，未改变且发生刷新时才加一。
- [ ] C6 不增加 truncation 字段、开关或 topology migration 项；`arguments.ModelParams`、`gaussian_renderer.render` 与 extension 的全局 `trunc_sigma/disable_trunc` 传递保持 C5/baseline 原样，Probation 不写它们。
- [ ] Core-active顺序改为一次step旧参数→到期lifecycle action→迁移；feature-off仍保持legacy时序，直到另有baseline兼容决策。
- [ ] checkpoint 恢复 `ell_i/d_i/born_iteration`、剩余 Probation 语义和全部 lifecycle counter；不添加 trunc tensor 或 CUDA trunc 改动。

**验证：** feature-off/C5 fallback；CPU topology/state tests；GPU 500须由integration fixture强制clone/split/prune；Tool Room seed0正式quick建议15k以覆盖7k–15k topology窗口；随后Utility Room seed0确认。

**晋级/停止：** 无 state/Adam 错位、无旧 grad 丢失、两阶段计数真实发生；逐 Gaussian 验证 `s_i`/`ell_i` 枚举无混用、Probation 不早释、`d_i` 无 iteration 级偷增；C5/C6 renderer truncation settings 与 checkpoint schema（除合法 Core lifecycle state 外）不变；C6 相对 C5 有可重复几何/稳定性信号且效率守门。任何 P 长度错位、resume 不等价、保护点被剪、Probation 连分裂、满 500 轮即在非刷新点释放，或新增/改写 truncation 状态与接口即停。

**Commits/tag:** `test: specify lifecycle and topology migration`; `refactor: expose baseline topology candidates and mappings`; `feat: add C6 reliability lifecycle`; tag `c6-lifecycle`。

## Post-C6 G2 Confirmation (No New Method Variable)

- [ ] 固定 `c6-lifecycle`，Tool Room与Utility Room各3 seeds完整30k；同一输入视图、mesh提取、评价协议。
- [ ] 报告逐scene/seed几何、PSNR/SSIM/LPIPS、runtime、峰值显存、Gaussian数、state/action指标。
- [ ] G2：两场景平均几何约≥3%；单场景系统退化≤约1%；PSNR下降≤0.3dB、LPIPS恶化≤0.01；时间≤2×、显存<22GB。
- [ ] G2失败不进入Supporting；只Tool有效则降级主张并停止堆模块。

## Git State and Remaining Policy

1. `main` 锁定于已验证 baseline `d6f15c8891a53800d5e3100f95817a7dd7f98e2f`；官方代码祖先为 `upstream/main@88d64054f53e0eba9ce49198282edf4a67fc8ca8`。
2. 用户于 2026-09-03 批准并创建 annotated `c0-baseline`，精确指向上述 40 位 baseline SHA；该 tag 不得移动。
3. 累计分支 `research/core-routing` 已从该 baseline 创建并切换；设计稿与三份规划文件在此分支以 `docs: add Core audit and implementation plan` 提交，不推进 main。
4. C1–C6 tag 只在 CPU+GPU smoke+Tool seed0 quick通过后创建：`c1-need-gate`, `c2-coarse-arbitration`, `c3-abstain`, `c4-parameter-routing`, `c5-conflict-projection`, `c6-lifecycle`。
5. D0无用户指定阶段tag；不新增/移动实验结果已引用tag。每个tag记录40位SHA和验证manifest。
