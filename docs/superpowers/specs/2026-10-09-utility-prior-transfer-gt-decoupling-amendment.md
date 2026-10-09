# Utility prior-transfer 首次 GT 访问：最小解耦修订案

**2026-10-09，待用户确认；仅书面交付，尚不授权代码修改或GT访问。** 对[已批准Utility规格](2026-09-30-prior-risk-utility-transfer-design.md)仅修订§3与§7的GT准入依赖。Geometry Q按用户范围/预算决定暂停，保留文档和结果；不是科学否定、不是 `NO_ACTION_SPECIFIC_SIGNAL`，不生成geometry termination/release。

## 1. 唯一协议改动

旧规则要求首次Utility GT访问前取得geometry G-C或科学终止release。新增一个**只适用于冻结 `1-r_p` prior-transfer 的人工审批路径**：原v3 confirmation、三seed合格资产，加不可变的 `utility_prior_transfer_access_amendment` 补充记录和独立GT执行授权。该路径不要求Q的结论，也不给任何geometry候选访问Utility的权限；原geometry release路径仍保持原合同，不用伪造release或全局关闭防火墙。

**信息隔离的代价明确披露：** 首次GT解析后Utility不再是Geometry的未触碰确认场景。Utility GT、prior-transfer的指标/报告/标签及其派生结论均不得用于恢复、选择、筛选、修改或校准任何geometry候选、公式、常数、阈值或预算，也不能充当geometry验证证据。今后若恢复Geometry，须独立批准并改用未触碰的独立确认数据；这次Utility结果不提供恢复依据。暂停/工程失败/归档不能触发科学终止release。

## 2. 保留已完成资产，用补充记录绑定评价代码

不改写、不替换原confirmation或资格文件，不倒签“训练前”记录，不重跑DA3/训练。已完成资产仍按原始v3身份资格审计；补充记录只在首次GT内容访问之前创建。

- 原confirmation SHA：`8ae37c8d0d884936c9f8416b9fd833f05f2c4c03efc7e073c8f78dccaa7653a1`；训练commit：`c701424c1b1f5a9006e6f19776769ee7bc8cb299`。
- 冻结DA3 snapshot SHA：`307b176e41111af403a565db94cfe8ada0a7d739361e1e380dcfaca08f49fc22`。
- seed0/1/2资格SHA分别为 `fbad915581157be0c89e24ae13e605ddc9f1fe03a1495dc94dc44e61ce59a1c9`、`d0b3eae76beab3261097239d23d9f70b6e1bc96b66ebc099c1b34f4279391bf4`、`7688190747a595c23c766b4d66d55b4516c66157c2e5eb1ba5db096874de445a`，均为QUALIFIED、GT NONE；首次执行仍需重核实际字节/指纹。
- GT仅已有文件身份：24,490,351 bytes，SHA `213dbdfff9ba992000039533463e4fcd941708d9fd8fe077df53a8495b63cd75`。没有解析/对齐/覆盖通过的结论。

补充记录采用canonical JSON+detached SHA，必须绑定：本修订及用户确认/GT执行授权的可审计记录；原v3的path/SHA；三份资格path/SHA；source/snapshot/GT原身份；新**评价**exact clean commit与未改统计核心的文件哈希；原protocol与mesh_admission的完全相同副本；原probe output/staging/access-log路径；`PAUSED_SCOPE_BUDGET`、无geometry候选准入、Utility信息禁止回流。固定schema、精确字段及白名单在最小适配中落实；未知/缺失字段拒绝，不接受任意override。

新评价commit与训练commit必须分别记录，不能把新commit写成训练来源。当前CLI硬性要求评价commit等于原confirmation commit，此处仅用补充记录显式允许**经过最小准入适配且统计核心未变**的新评价commit；资产资格绑定仍使用原v3。冻结统计核心以原训练commit的Git对象内容及文件哈希比对；不以“新代码通过测试”代替身份一致性。

原输出/staging/access-log在补充记录创建和执行时均须不存在；不要求已完成的run/view/state再不存在，也不能重做pre-run confirmation。补充记录/授权/原confirmation/资格的guarded reload、alias/symlink/hardlink安全、完整输入指纹、first-access日志先于mesh解析、最终再验证、原子发布和禁止覆盖保持有效。没有已批准补充记录和明确GT执行授权时，现有防火墙继续阻止访问。

## 3. 科学、mesh准入与输出全部不变

直接调用冻结 `probe_g1_prior_complementarity.evaluate_iteration`，通过现有 `g1_prior_transfer` 薄适配与聚合；不修改/复制统计实现。唯一 `M0=[A,1-S] → M1=M0+[1-r_p]`；全部finite、严格同行join后 `V_p=True` 的共同样本域；`V_p=False`未知、不得当最大风险；标签严格 `distance>0.05m`，不crop/过滤/重对齐。原始 `1-r_p` 的AUROC、AUPRC、risk quintiles、Spearman及各fold结果仍按既有协议完整报告。

五空间slabs、fold-local标准化/class weight、固定Newton/IRLS及全部solver常数、0.5m paired voxel bootstrap、2,000 replicates、`SeedSequence([20260930,iteration,seed])`及三seed宏聚合原样复用。3000仅方向稳定性门+描述性报告；7000唯一性能主门：各seed coverage≥0.80、原方向门（两时点high-low/Spearman≥0；7000 high-low≥0.05）、每fold数值独立性、各seed pooled OOF gain≥0、三seed均值≥0.02、宏95%下界>0.005。没有新增主指标、候选、阈值或选择机会。

原mesh_admission逐字段保持：米制/identity变换，固定最多50,000 sparse-point minhash，distance median≤0.05m/p90≤0.15m/within0.10m比例≥0.80；147相机各8×6射线，aggregate hit≥0.80、至少90%相机hit≥0.50；全finite非退化triangle，evaluation_filtering=NONE。首次真实解析若身份/坐标/覆盖异常，INCONCLUSIVE并立即停止；不得修mesh/换域/改门。

只产生现有compact `inputs.json`、`mesh_admission.json`、`report.json`、`seed_folds.csv`、`risk_bins.csv`、`bootstrap.csv`、manifest，加必要补充记录/首次访问/执行及发布审计；不生成图像、PLY、大型tar、训练资产或新候选。有效评价的三种冻结结论仍为 `PRIOR_RISK_TRANSFER_SUPPORTED`、`NO_CROSS_SCENE_REPLICATION`、`INCONCLUSIVE`。先冻结全部结果再讨论简单检测方案/C1；任何结果都不自动授权新N、r_g路由、五状态、生命周期或30k。

## 4. 最小适配与测试，不重开探针开发

获批后只改confirmation补充记录验证、GT firewall准入/token及现有CLI的绑定/路径选择；不改生产方法、statistical core、三seed聚合和mesh审计算法。复用现有canonical、guarded-read、指纹及原子发布工具，不新建通用框架。旧记录/default路径继续fail-closed，不能直接用暂停状态调用旧geometry release接口。

新增测试仅覆盖六组合同：①暂停本身/未批准补充记录不解锁GT；②补充记录与三seed/原v3/新评价commit及未改core的正确绑定；③任一身份、科学字段、授权或输出路径变动拒绝；④任何伪科学终止/geometry准入请求拒绝；⑤日志先于解析、别名/变异和既存目标拒绝；⑥CLI经新路径仍委托同一模型/统计/聚合与compact出版，并在mesh失败时不拟合。使用合成fixture和现有测试依赖，无真实GT。

已有 `test_g1_complementarity`、`test_g1_prior_transfer` 只作回归；confirmation/firewall/CLI运行直接受影响的测试，不跑无关全套，不增加Tool Room探针。本轮只交付书面修订；测试与适配尚未执行。

## 5. 论文主线与本验证的关系

用户进一步澄清：论文主贡献目标是五状态仲裁→参数路由→生命周期；高误差检测量是前置辅助，不应为它无限扩建Q研究。Q此前回答的是action-specific geometry authorization，不是N检测必需组成；暂停Q不等于放弃主线，也不能把它变成prior-transfer的前置任务。

本验证PASS只支持冻结 `1-r_p` 的跨场景预测信息；原始风险指标与M0/M1 OOF诊断必须分开。用GT拟合的OOF模型不是可直接放进GT-free训练的部署公式。后续可以另行批准简单、GT-free检测/动作的最小规格与对baseline的配对实验，不必先完成Q；但不能把高 `1-r_p` 直接解释为增加不可靠先验梯度的授权，也不能自动交换A、运行30k或宣称重建收益。

一致性可以作为未来策略假设的输入，但当前r_g未验证外部正确性或安全授权；恢复五状态需要另行明确一致性与动作语义、反例和配对消融，不能跳回原“可靠性授权”称谓。已有Tool Room soft-v4/7000稳定Consensus=0、Geometry-led=22；原因尚未被逐门/迟滞分解定位，不能归因为7k短，也不要求人为凑齐状态。这个未解诊断不阻塞本次prior-transfer，不启动新研究线或Q。

## 6. 缺口、审批与批准后直接执行的步骤

**没有新数据或训练缺口。** 目前缺的是用户批准本准入修订、最小适配/必要TDD落地和新评价commit、带SHA的补充记录，以及明确的首次GT访问/评价授权。GT对齐/覆盖尚未经真实解析，这不是预先假定会通过的条件。

请先确认本修订；确认后以一次明确授权覆盖“最小准入适配/必要验证、补充记录创建及一次冻结三seed GT probe”，不再插入Geometry研究或新统计审批。若只批准协议、未授权代码/GT，则仍停在该边界，不推断权限。

获批且获得上述授权后，直接交付一个user-operated AutoDL流程：checkout已提交的exact评价commit → focused资格/原v3及三seed指纹重核 → 创建/重载canonical补充记录并验证空probe目标 → 写入first-access审计 → **原mesh准入** → 原evaluator六个seed×iteration测量及三seed7000聚合 → compact原子发布/输入不变审计 → 冻结结论。中间不安排新DA3、训练、Geometry或大型包。

任一合同异常停止并报告，不自动重试/重划fold/更换solver/改门；不能计算时INCONCLUSIVE，前置安全错误若连合法报告目标都不可建则仅保留操作错误审计，不伪造科学报告。科学门正常计算但未通过则如实冻结NO_CROSS_SCENE_REPLICATION，不反复调整。该修订与后续授权仅适用于prior-transfer，绝非普遍GT解除。
