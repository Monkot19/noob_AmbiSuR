# Geometry G-A Gate 0 四项规格修订案（待审批）

状态：**仅提案，未生效**。对应已批准精确子规格 `2026-09-30-geometry-stage-ga-exact-subspecification.md`，以及实施计划 `2026-10-08-geometry-stage-ga-tdd.md` 的 Gate0。原规格正文、公式、常数、门槛、计划状态均未修改。本稿不生成 fixture、预测或 outcome，不构成 `NO_ACTION_SPECIFIC_SIGNAL`、geometry termination 或 Utility GT release。

2026-10-09审阅补件：具体参数附件已另写为 `2026-10-09-geometry-ga-execution-annex-proposal.md`，含逐项推荐值、生成规则及其超出旧文字的解释影响。它同样未获批准；请同时审阅，不能把本稿的字段清单当作已完成执行闭包。下文“另行提交附件”描述原2026-10-08阶段，现由上述待审补件承接，不表示Gate0已关闭。

建议分别审批四项，不把它们捆绑成科学结果。第1、2项需要明确的实验合同变更；第3项需要另行完成并审核逐字段冻结附件；第4项需要明确计算图和数值语义。审批前不实施任何一项。

## 0. 不变范围与审议含义

- 仍研究一个 cluster 的一次固定几何 action 是否有益且安全；不是逐 Gaussian 因果效果，不允许 batching。
- 不复用/翻转/包装 `r_g` 或 `1-r_g`；其仅是内部一致性 telemetry。Evidence 保持v4。
- 不重新选择 `Q_g` 五组件、其组合、授权阈值、action 解法或 benefit/safety 门。下面明确列出的备选常数仅是**待审批建议**，不是现行值。
- 不读取 Utility GT，不开展 Tool Room 干预或 synthetic 执行。Utility seed0训练与本稿独立，不需要先通过 Geometry Gate0；Utility GT仍须已有防火墙的独立释放合同。
- 文档矛盾/未定义量属于设计闭包阻塞，不是候选被实验否定。只有未来获批合同下完成的有效实验，才可能产生科学正/负结论。

## 1. 20° sector 与逐 family coverage 的必然冲突

### 原规则

§3.2要求至少8个支持相机、P/H各至少4个；§9.1只有12个相机，方位间隔30°；§9.2的 ill-conditioned family只保留20°扇区内相机；§10.3同时要求每个family validity coverage≥0.60。原总域为8×16=128对，overall coverage分母128。

因此该family至多保留1个相机，16个variant必定全部invalid。不能用零分数、补相机或忽略family在运行后修补。

### 建议改动 A（推荐；必须显式批准）

用以下条文替换相关 inventory/coverage/bootstrap/budget 条文：

> 总登记库存仍为128例。ill-conditioned的16例预先定义为“支持不足、必须拒绝”的validity合同对照，不是112个预测性 branch pairs 的一部分。每例必须 `V_Q=False,Q_g=null,authorized=False`，不得执行未定义的action，也不产生伪造的 `Y_C/tau_C`。逐例报告拒绝原因；任何一例被授权均为合同失败。
>
> 预测性库存固定为另外7 families×16=112对，不按实际结果增删。overall validity coverage为 `valid/112`，仍要求≥0.80；这7个family各仍要求≥0.60。报告另附128例登记库存和16例对照的拒绝率，禁止混称112/128的覆盖率。其他invalid预测例仍留在112分母内。
>
> 预测指标仅使用valid预测对；bootstrap在预先指定的7个预测family内各抽16个variant，仍为PCG64 seed20260930、2000 replicates、原percentile区间及classless不重抽规则。16个支持不足对照不进入预测bootstrap。最多112×2×8=1792个optimizer steps，原2048上限不增加；其余资源上限不变。

### 备选 B（不推荐）

保留128个预测对和每family coverage门，另设计能够保留至少8个真实支持视角的 ill-conditioned fixture，并重新预注册其相机几何。这会改变负对照机制，不能靠在20°扇区复制相机或假称视角独立来满足计数；需要新的唯一布局与支持验证规格。

### 理由及实验含义

A保留“未知不能授权”的严厉合同，不强迫一个刻意支持不足的family同时成为有效预测样本。但它**改变预测总体、coverage分母和bootstrap分层库存**，并非纯措辞修正。未来结论只覆盖7类有机会满足支持条件的预测任务，不能声称在支持不足family上验证了benefit预测。若不接受这个含义，应选择B并另行审阅具体布局；不能保持矛盾直接运行。

## 2. 平面 lattice 无法构造32-row cluster

### 原规则

§2.1固定32个邻居，最远距离≤`R_C=0.05 m`；§9.1单层平面grid间距0.02m。内部anchor的半径内共21个中心（5+10+6），不足32。刚性旋转不改变距离，边界更不能补足。

### 建议改动 A（推荐；必须显式批准）

仅建议将 **cluster半径上限由0.05m改为0.07m**；保留grid0.02m、32个不同原始row、distance/index排序、无padding/duplicate、guard0.10m及action cap0.01m。普通内部平面anchor的第32近邻距离不超过 `sqrt(10)*0.02≈0.063246m`，因此可满足0.07m上限。边界anchor仍需按附件的确定性布局审查，不能据此声称所有surface/anchor已经可行。

建议新条文：

> `R_C=0.07m` 仅用于32-row pre-action cluster membership的上限。guard继续按到实际成员的最小距离≤0.10m构造，不从新半径重新推导。manifest冻结具体anchor/rows；选择不看outcome；不足32即invalid，不替换anchor。

### 备选 B

保留半径0.05m，把grid间距改为0.015m，使普通平面内部anchor有足够候选。这会改变Gaussian密度、重叠、内部残差和渲染计算量，比A牵动更多输入。禁止靠同位置复制点填满32行。

### 理由及实验含义

A保持32-row干预单元和初始采样密度，但**扩大允许的干预空间尺度**；实验不再是严格0.05m半径cluster的研究。B保持半径但改变离散表示和信息密度。两者都不是实现修复，均须批准；本稿不应用任何新值。若拒绝A/B，必须先提出完整不同的合法fixture布局，不能直接开始TDD。

## 3. Fixture 与 horizon 的执行闭包

### 原规则

§2.3要求canonical anchor manifest；§6固定8步和step4单次生产topology事件；§9给出三种surface、八families及旋转。却未唯一规定surface分配、有限grid/anchor、三角化、camera命名、随机数、horizon loss、Adam及topology选择量。直接引用生产默认值会把未审批的loss/DA3/调度隐含带入synthetic研究。

### 建议改动（先批准附件冻结制度，不擅选新的科学常数）

在§9之后添加以下强制条文：

> 在研究代码/TDD实施前，另行提交并批准一份 `G-A execution-contract annex`。附件必须是可机械验证的literal表格/JSON schema及生成规则；不能只写“沿用默认”“适当选取”或留TBD。附件与spec/addendum SHA共同绑定confirmation。附件未批准时仍为 `SPECIFICATION_BLOCKED`，不得生成结果或termination。

附件必须一次性填满以下字段：

| 对象 | 必须冻结的具体定义 | 禁止的替代 |
|---|---|---|
| 128登记例与预测/合同域 | 逐family surface映射、variant集合、各例唯一anchor；同构旋转须保持label-free | 看过outcome换surface/anchor或删除难例 |
| 几何与原始row | 平面范围/边界包含规则，圆柱角域/轴域/三角化，corner共享边处理，row排序、颜色/尺度/opacity/SH初始化；guard/far必须非空的构造或明确不可计算规则 | padding、随机补点、空区域当零误差 |
| 相机与哈希 | cx/cy、坐标轴/handedness、up/look-at退化处理、唯一camera_name；cluster_key十进制ASCII索引用逗号连接且无空格，整个 `fixture_id|cluster_key|camera_name` UTF-8 | 不同语言tuple repr、隐式Python hash |
| 残差扰动 | 哪个实际模型/观测张量、坐标系、施加次序；冻结mask来源；“错位的内部观测”不得与实时渲染被重复计数 | 把参考surface/参考normal输给Q或action |
| horizon目标与参数 | 唯一逐步loss及权重；明确哪些参数有梯度、哪些固定；所有Adam groups、lr/schedule、betas、eps、moments初值、zero_grad和step次序 | 按baseline大训练默认自动启用先验/ALR等 |
| replay与随机数 | Python/NumPy/TorchCPU/CUDA逐例seed派生、exact8相机调度、随机背景/采样开关；proposal/held-out是否参与horizon训练必须明确 | 验证期间重抽相机或不同branch调用数 |
| step4事件 | survivor/clone/split/prune各4个variant的固定调用、候选统计的来源/累计时段、全部阈值/排序/随机draw、mapping与disappearance定义 | 用Y或指定treated分支强造失败/选择children |
| 失败归类 | missing field、dtype/derivative不支持、mapping损坏均是工程不可计算；不把其转成有效负例或GT release | 补默认值继续跑 |

### 理由及实验含义

这是对**如何形成可审批、唯一执行合同**的具体修订，不是本稿替用户选择新的horizon科学定义。本项审批后仍需撰写并审批填满的附件；**Gate0.3尚不能因此标为已闭合**。如果希望一次审批即实现，必须先提供并审阅该literal附件，不能仅凭本清单声称完整冻结。

采用不同loss/优化组/拓扑门可改变cluster级 `tau_C(H=8)`，因此本稿不臆选Adam或把它们称为“无关工程默认”。唯一action、Y、horizon8、benefit/safety原门保持原文；任何附件提议的实质更改必须另列影响并获批。

## 4. 可微 residual、归一化及 dtype 边界

### 原规则

§4要求固定权重归一化和exact autograd `J`；§5.1使用未定义objective的逐camera `g_v`；总则默认float64。生产collector/reprojection是detached/no_grad。去掉生产隔离或用零J/有限差分action都不符合批准合同。

### 建议改动（推荐明确以下唯一测量定义；待审批）

用以下条文补齐§4.1、§5.1–5.2及dtype例外，不更改五个Q组件或action解法：

> 对partition `B=P/H`，在delta0冻结原始有效component集合 `I_B`。每个component保留其源camera、残差family及非负pre-action renderer贡献权重 `w_j`。使用**partition内全部component、全部family的同一个分母** `W_B=sum_{j in I_B} w_j`；不得逐view或逐family重新归一化。`W_B`不足或非有限时invalid。定义 `R_B,j(delta)=sqrt(w_j/(W_B+epsilon)) * e_j(delta)/tau_family`。
>
> P中的 `n=|I_P|` 同时用于原H/b公式与逐camera objective：`F_v(delta)=sum_{j in I_P,source(j)=v} R_P,j(delta)^2/(2*n)`，`g_v=grad_delta F_v(0)`；故 `sum_v F_v=||R_P||²/(2*n)`。原q_dir的非零梯度/支持规则保持不变。
>
> `L_H(delta)=sqrt(sum_{j in I_H} w_j*(e_j(delta)/tau_family)^2/(W_H+epsilon))`。不再额外除component数。P/H各自分母冻结于0，delta_C虚拟评价只改变渲染/可微残差，不改变inventory、mask或权重。冻结component在delta后不可计算则该例unknown，不允许丢弃component或填零。

**实施边界建议：** 允许独立research-only可微语义adapter，不改生产collector、D0、CUDA或Evidence。它只接当前model/cameras/frozeninventory与共享translation，不接参考surface、GT、DA3或post-outcome。生产语义平价测试必须覆盖坐标变换、符号不变normal、遮挡、support和invalid。

**dtype例外建议（明确是数值合同修订）：** 现有CUDA renderer保留其原生float32 forward/backward；渲染输出立即cast float64，再构造残差、权重归一化、J/H/b及solve/score。J是该混合精度计算图的autograd导数，不宣称renderer本身float64或解析实数精确。不得detach渲染输出；必须先验证所需depth/normal路径确实有正确非零导数。若后端不支持，结论是measurement `INCONCLUSIVE`，不是科学负例，也不临时写CUDA或改action。有限差分只可作将来获批的合成导数核验，永不用于实际action/J替代。

### 备选

坚持全链float64、不接受renderer例外，则先确认现有后端能力；不能满足就停在计算合同不可实现，不开展另一个CUDA实现项目。选择逐view/family归一化则会改变优化目标和camera权重，属于另一候选测量合同，需重新写唯一proposal，不能执行时任选。

### 理由及实验含义

新增定义消除工程师各自选择梯度尺度的自由度。原先的“normalized weight”未唯一确定这些值；因此**首次明确分母/逐view objective会确定实际action和Q的测量含义**，不是声称与所有旧解释数值等价。混合精度例外降低后端门槛，但并不保证可以求得所需导数；合成平价与后端资格仍需以后单独授权。

## 5. 审批后顺序与当前停止线

建议审批记录分别列出：1A/1B、2A/2B、3附件制度与另行审批、4归一化/梯度/dtype方案。不得只写“Gate0全部自动通过”。

若获批，下一步只是把批准的修订写入正式addendum，并补齐第3项literal附件交审；更新实施计划必须引用新SHA和清楚的新inventory。附件获批、四项闭合、TDD实施授权均满足后才可写/运行研究测试。synthetic完整执行另行审批；Tool Room更在之后。

现在的实际读写仅为既有规格/生产接口的只读检查和本Markdown/日志写入。**没有实验启动；本提案通过不证明Q有效，提案被否决也不证明Q无效。** Utility GT访问、geometry release、五状态、生产路由和C1仍不授权。
