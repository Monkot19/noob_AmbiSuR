# AmbiSuR 可靠性路由项目交接说明

## 1. 项目目标

**最新实际结果与复用规则（2026-10-09，取代下方访问前状态）：** 用户在b454b127运行id-recovery1，首次GT访问已发生，mesh ADMITTED，六组seed/iteration计算完成，116.131秒后发布INCONCLUSIVE。发布manifest SHA `c53441b9cff3c351ed2c4ecf5e439679d3f661597964516acbcd5dc2b122e5c0`及其所有文件经用户只读审计通过；原因是聚合器报bootstrap2000合同不匹配，seeds为空、macro为null。只读本地合成诊断复现：原生产函数返回形状(2000,)的ndarray，Utility聚合器错误地要求list；Tool Room原报告函数会先调用既有JSON规范化。不是缺少replicate，也不构成信号否定。修复及GT后同参数恢复尚待明确授权，严禁再用访问前恢复或删除原输出。

**用户强化复用要求：** 依根目录AGENTS.md新增“真实生产接口复用门”，必须验证真实producer → 适配/聚合 → 原规范化 → consumer/发布，禁止只靠预制列表或mock断言接口可复用。共享科学核心、实际记录格式和目标命名统一；不为场景重写统计、不扩展无关测试。当前仅记录规则和诊断，不修改方法/评价代码，不重跑任何实验。

**最新恢复状态（2026-10-09）：** 格式恢复尝试也在GT前因冻结合法`.probe`名称被拒绝而停止。用户只读审计确认失败凭据SHA `1618c56f6d9cfac4a6825a71d8443e6609b71639878176fd7c72b8415102afb0`、前次amendment SHA `02b475fe2b1981d7719c382d8bd53967686df8a8b4f18c1648f65a0e8f45dec2`，三个原始一次性目标均不存在。后续只能以两份确切handle进行显式`.id-recovery1`恢复，保存两轮旧审批/修订/失败记录，不重用旧记录名，不自动重试。原confirmation、三seed、输出路径和全部统计合同不变；GT准入和科学结果仍未知。局部修复41项测试及既有核心范围内104项回归通过（1本地可选后端跳过），一次范围受限审阅无问题；不得将此当成真实GT评价完成。

**当前优先级（2026-10-09 用户决定）：** Geometry Q因范围/预算暂停，全部文档/结果保留，不是科学否定，也不产生geometry termination/release。唯一优先任务为已合格Utility seeds0/1/2的冻结 `1-r_p` 独立验证。用户现已明确授权[最小 prior-transfer-only 解耦修订](../superpowers/specs/2026-10-09-utility-prior-transfer-gt-decoupling-amendment.md)与一次冻结 GT 评价；首次尝试因读取器误要求缩进 JSON 而在 GT 访问前停止。原 source/snapshot SHA 正确、实际生产格式为紧凑 JSON；首次访问日志/输出/staging 均不存在。仅修复格式读取，保留并绑定旧审批/失败记录，新的明确恢复仍须检查相同一次性目标与冻结核心，不自动重试。Utility结果不得回流Geometry候选恢复/选择/调整；GT访问后Utility不再是Geometry未触碰确认场景。无DA3/训练重跑、新N、r_g路由、五状态、生命周期、30k或C1授权。

**论文优先级澄清：** 五状态仲裁、参数路由与Gaussian生命周期是主要贡献目标；N/高误差检测是前置辅助。Q研究并非检测必需前置，不能继续以其复杂化阻塞Utility已有资产评价。冻结predictive信号不等于GT-free部署公式或训练动作收益；当前r_g仍只是内部一致性telemetry。未来若不依赖Q研究一致性驱动五状态，须另批语义/动作/消融，不可静默恢复原几何可靠性授权。Consensus为空的原因仍未逐门定位，不归咎于7k或强行调阈值造状态。

**跨场景工程原则（2026-10-08 用户确认）：** 公共评价/审计核心只维护一套，场景差异通过配置与薄适配层表达，只为新增合同补测试。完整规则见根目录 `AGENTS.md` 的“跨场景复用与统一评价”。必须在上下文恢复后保留此原则；当前 Utility 直接复用 Tool Room 冻结统计核心，禁止重新开发或重选。

以 AmbiSuR 官方实现为 baseline，在 ScanNet++ 紧凑、高反射室内场景上研究：观测校准的歧义需求、外部先验与内部几何双可靠性、拒绝式仲裁、参数级梯度路由和可靠性驱动的 Gaussian 生命周期。

优先目标是会议论文；保留期刊扩展与硕士毕业论文兜底。当前主张限定为“改善高反射室内场景的整体表面重建”，不声称已经对人工标注反光区域做独立定量证明。

## 2. 代码来源

- 官方实现：`https://github.com/Fictionarry/AmbiSuR`
- 用户 fork：`https://github.com/Monkot19/noob_AmbiSuR`
- 新项目开始时必须在这里补写实际基线 commit：`BASELINE_COMMIT=<运行 git rev-parse HEAD 后填写>`
- `main` 只保存可复现 baseline；开发从 `research/core-routing` 开始。

## 3. 环境与算力

- 平台：AutoDL
- GPU：单张 RTX 4090 24GB
- PyTorch：2.8.0
- Python：3.12
- 系统：Ubuntu 22.04
- CUDA：12.8
- Core 峰值显存建议低于 22GB，训练时间不超过 baseline 约 2 倍。

## 4. 数据

- ScanNet++ Tool Room：`d415cc449b_Tool_Room`
- ScanNet++ Utility Room：`0a5c013435_Utility_Room`
- 本地数据根目录：`D:\dataset\ScanNet++\data\data`
- 本地 baseline 结果：`D:\research_Space\output\AmbiSuR_original`，只读
- AutoDL 数据根目录：首次配置时填写到实验 manifest，不得猜测或写死在代码中
- 后续泛化：AmbiSuR 官方 DTU 15 scenes、Tanks and Temples 6 scenes

Tool Room seed 0 是开发与 D0 标定场景；Utility Room 和其他 seeds 用于确认。论文必须披露 Tool Room 的开发用途。

## 5. 已确认的方法边界

Core 顺序：

1. 连续高端 SH 歧义量 A。
2. 有效视图数与方向离散度得到观测充分度 S。
3. 得到外部帮助需求 N。
4. 分别计算外部 `(T^P,V^P)` 与内部 `(T^G,V^G)`。
5. 计算可靠性、一致性和可靠性差。
6. 输出 Bypass、Consensus、Prior-led、Geometry-led、Abstain。
7. C2 开始执行粗粒度动作；C3 增加拒绝；C4 参数组路由；C5 冲突投影；C6 生命周期。

Supporting 仅在 Core 通过后实施：

- CoMe 式解耦曝光；
- 可靠前表面局部 Ray-Color；
- 可靠前表面局部 Ray-Normal；
- 生命周期状态驱动的逐 Gaussian 截断。

## 6. 实验门槛

- G0：关闭新功能时复现 baseline；无 NaN/Inf、显存增长和重复 backward。
- G1：N 的高几何误差 AUROC 大于 0.60，且比 A 或 1-S 中较优者至少提高 0.03。
- G2：两个反光场景平均几何改善约 3%，单场景系统退化不超过约 1%；PSNR/LPIPS 分别不恶化超过 0.3dB/0.01。
- G3：Supporting 每个模块最多进行默认配置、一次有依据调整、一次跨场景确认。
- G4：DTU/TnT 多数场景不得系统退化。

## 7. 本地—GitHub—AutoDL 闭环

本地 Agent 只在隔离分支/worktree 修改代码，完成单元测试、代码审查和短步检查后提交并推送。AutoDL checkout manifest 中记录的明确 tag/commit，运行短步 GPU smoke 或正式实验。实验输出连同 manifest、配置、日志、指标和环境快照下载回本地。分析脚本读取结果但不修改原始输出；新结论写入 `findings.md`，阶段状态写入 `progress.md`。

禁止行为：服务器临时改代码、只记录分支名不记录 commit、用脏工作树跑正式实验、结果出来后移动 tag、一次提交混入多个消融模块。
