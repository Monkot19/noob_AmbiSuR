# Geometry G-A 简短设计可行性复核（仅书面，待选择）

日期：2026-10-09。审查基线：`80399fd634abe558092465682a70ef479aa8ce29` 的[待批附件](2026-10-09-geometry-ga-execution-annex-proposal.md)，不是实验结果，也不是已生效的新规格。

**后续状态：** 用户已选择缩小、拓扑后置，并原则认可数学及检查方向。精确公式/validity/阈值含义、固定后端策略/容差/计时及阶段审批点现一次合并交付于[最小可行性合同](2026-10-09-geometry-ga-minimal-feasibility-contract.md)，仍待具体审批。下文保留原复核，不再作为重复选择或执行授权；任何不同的检查细节以新待审合同为唯一提案。

**建议：缩小范围，不开始128例建设。** 数学尺度问题确实存在；后端关键导数及成本仍未获验证。先审批下面唯一数学修订及最小检查方案，之后另行授权检查；通过也只证明工程路径可能可行，不证明 `Q_g` 能预测有益/安全更新。

## 1. 已接受与仍待批的边界

- 1A原则接受：16例只作“支持不足拒绝对照”；不能称为有效观测条件下的病态几何预测验证。112例预测域也仍受其余待批项限制，不能现在运行。
- 2A原则接受：cluster半径0.05→0.07m；32成员及0.10m guard规则不变。空间尺度与成员集合可能变化，不能宣称干预对象与原设计相同。
- 3A/4A未批准；本复核不批准它们，不改已批准子规格、TDD、生产代码、公式实现或常数。无GPU、synthetic/Tool Room干预、GT读取、geometry release、五状态或C1。
- Utility三seed资产合格不是GT release。工程不可行、规格矛盾或本次缩小建议都不能产生 `NO_ACTION_SPECIFIC_SIGNAL` 或解除Utility GT防火墙。

## 2. 数学核对：重复平均改变了什么

为避免与现有证据 `A` 混名，用 `G=JᵀJ`、`c=Jᵀr` 表示数据曲率及梯度，`f_j=e_j/τ_family`。待批4A先定义

\[
W=\sum_j w_j,\quad R_j=\sqrt{w_j/(W+\epsilon)}f_j,\quad J=\partial R/\partial\delta.
\]

weights、mask与inventory在pre-action冻结。因权重已按总和归一化，`G,c`已经来自**加权平均意义**的残差目标，不是未经平均的n项总和。随后主规格再使用

\[
H=G/n+\lambda I,\quad b=c/n,\quad
\delta_{raw}=-(G+n\lambda I)^{-1}c.
\]

这是严格代数等价；`λ=10⁻³`实际变成相对于该归一化数据目标的 **`nλ`阻尼**。不是仅把同一个优化问题乘上一个无关的正数。沿数据曲率特征值 `a_k>0` 的方向，相对无阻尼Gauss–Newton更新的保留比例为 `a_k/(a_k+nλ)`；弱信息方向衰减最强。0.01m步长上限是后续另一个机制，不能消除该尺度效应。

若把相同components复制m份并重新总权重归一化，`G,c`基本不变，n变成mn，阻尼却增加m倍；像素分辨率/同质采样密度也可能产生这种变化。固定 `W+ε` 使复制不严格不变，但 `W>10⁻⁴, ε=10⁻⁸` 时该归一化floor效应远小于n倍阻尼，不能用它解释后者。复制不是增加独立信息；这里是尺度一致性思想实验，不是统计独立性论证。

**`q_cond`的问题是另一层：** 待批定义从含阻尼H取特征值，故

\[
\rho_H=\frac{a_{min}+n\lambda}{a_{max}+n\lambda+n\epsilon},\qquad
q_{cond}=\operatorname{clip}(\rho_H/0.05,0,1).
\]

随着n增大，`ρ_H`趋近 `λ/(λ+ε)`（约1），不是趋近数据本身的良好可辨识性。例如数据曲率 `diag(1,0.1,0)` 是秩亏的；n=256时比例约0.204，已经让 `q_cond=1`。这只是手算例，不是候选实验结果。即便删去/n，只要仍用含阻尼矩阵，阻尼也可能改善这个数值而未增加观测信息。

合理保守包括固定λ抑制弱方向、0.01m上限、support/validity及held-out下降要求；**未经设计的n依赖阻尼，以及用正则化的各向同性代替几何信息，是意外保守/乐观混合**。小更新往往使 `q_virt`变小，却使 `q_mag`更接近1、`q_cond`偏高，所以不能简单说总Q只会更保守。`g_v`的1/n共同缩放在无ε的单位化中可抵消；实际 `||g_v||+ε` 和非零判定仍可能被不必要的小尺度影响。

### 唯一待批准修订：一次平均，分离数据条件与数值求解条件

不是提供多个候选择优；建议把以下内容作为**一个不可拆开执行的尺度/conditioning修订案**：

\[
F=\tfrac12\|R_P\|^2,\quad H=G+\lambda I,\quad b=c,\quad
\delta_{raw}=-(G+\lambda I)^{-1}c,
\]
\[
F_v=\tfrac12\sum_{source=v}R_{P,j}^2,\quad g_v=\nabla F_v(0),\quad
q_{cond}=\operatorname{clip}\!\left[\frac{\lambda_{min}(G)}{\lambda_{max}(G)+\epsilon}\,/\,0.05,0,1\right].
\]

- 保留λ、ε、所有τ、0.01m cap、0.05 condition缩放、Q几何平均及原benefit/safety/性能门；n仍用于≥256component的inventory门，但不再平均已总权重归一化的目标。`L_H`原加权RMS不改。
- `H`原数值有限/可解门保留，但只说明正则化求解可用，**不证明观测充分**。有限、秩亏G可得到 `q_cond=0`；这与其他validity失败产生unknown不同。非有限矩阵/导数仍unknown，不填零。原validity门不因低分而放宽。
- G只检查相对各向异性，不能独自证明信息绝对强度或外部正确性；其余support/virtual门也尚无预测能力证明。不能把此数学修正写成科学PASS。
- 理由：让固定λ对应固定归一化目标，避免仅改变inventory就改变动作；让conditioning反映数据而非人为阻尼。**这会改变原动作与唯一Q公式，必须由用户批准后才能写回规格/实现。** 新旧行为不可宣称等价，不允许先看实验再决定是否采用。
- 坐标仍米制，残差以既定τ无量纲化，G与λ均使用该平移坐标基准。复制不变量只承诺上面的ε-floor容差，不承诺不同观测分布/场景的动作相同。

## 3. 最小后端检查方案：一个案例，尚未授权执行

**目的**仅检查真实native渲染→reprojection的关键导数是否正确且能在预算内形成三列J；不计算paired `Y_C`、tau或科学性能，不建设128例、八步optimizer、topology、结果发布系统。

**拟用一个固定小案例：** 96个Gaussian，32成员+64非成员，0.02m规则点阵、0.015m尺度、0.8opacity、固定倾斜primitive normal；8个PINHOLE环形相机，4P/4H，不batch cluster。低分辨率64²查导数，**同一案例**512²查成本；focal随分辨率同比缩放以保持同一FoV。沿用待批附件的frame、插值、τ、native float32→float64、完整正权inventory定义。具体96行、相机矩阵、固定非零残差和probe分量ID将在获准写最小检查脚本时先冻结为只读fixture；不得从运行结果挑选像素/步长/normal来取得通过。本方案批准不等于3A完整fixture/horizon方案批准。

独立参照及检查顺序：

1. **手算层**：一个仿射depth残差+一个非平行normal残差的float64闭式导数，对照残差装配、三列J与G/c，atol=10⁻¹⁰、rtol=10⁻⁸。它不代替native后端检查。
2. **权重/前向层**：生产 `forward.cu` 按 `features*alpha*T` 汇RGB，黑背景下用成员color=1/其余=0的**全场景**indicator render有望直接得到 `sum_C(Tα)`。查与独立CPU小场景逐ray、前后顺序累乘的权重和，atol=5×10⁻⁵、rtol=5×10⁻⁴；CPU必须由fixture的3D参数与相机独立计算投影footprint/α及排序，不读取native中间贡献再加总。不能使用只渲染cluster、alpha-only或out_observe替代。若无法独立对齐贡献截断/early termination，停止，不新建CUDA功能。这是源码支持的路线假设，尚未实证。
3. **native关键导数层**：冻结3个预先指定的非边界depth/mv-normal/dn标量及其有效inventory；共享translation三轴autograd导数，对照**仅forward重算的中心有限差分**。参照对象明确为action使用的加权 `R_j=sqrt(w_j/(W+ε))*e_j/τ`，两侧复用δ=0的冻结w/W/mask。步长固定 `h=10⁻³,5×10⁻⁴,2.5×10⁻⁴m`；最后两档差分彼此及各自与autograd均须满足atol=5×10⁻³、rtol=2×10⁻²，且手算预期非零方向不被detach。若跨sorting/culling/abs/clamp/validity边界导致参照不稳定，则工程检查不确定，停止，不选最好h或替换component。差分只作独立检查，**禁止作为action/J替代实现**。
4. **成本层**：512²、8view、96行的全部合法components，不抽样；形成全部三列J或精确累计G/c并核对第3步分量。研究adapter只能复制可微语义并与生产forward对齐，不能移除生产D0 no_grad。不能用gradient-of-summed-loss冒充完整J；每一轴残差导数才是J列。custom autograd是否支持有效批量reverse/JVP未定；只有正确原生路线可接受，不能靠新CUDA项目、近似J或降分辨率绕过。

**建议工程预算与停止门（均待批）：** 一台现有4090、peak allocated GPU≤6GiB、进程RSS≤4GiB、输出≤20MiB；整次检查硬上限15分钟（含import/setup/参照），512²完整导数块≤30秒。一次固定warmup加3次相同计时，取最慢值，报告component数和峰值；每次重新建图，统计包括全场景渲染、residual/J/G/c，CUDA同步后读时钟。不能把warmup或缓存图算作新求导。只读代码/合成fixture，写新小receipt，不触碰任何真实数据或GT。任一身份、前向、导数、finite、资源门失败即停止，无自动调参/重试。

本段native差分容差**明确替换附件中尚未批准的通用gradient容差**，区别于仍使用5×10⁻⁵/5×10⁻⁴的前向对齐容差；用于float32forward差分噪声，且须在执行前批准，绝不能根据结果放宽。未获独立导数参照、预算不合格或backend不支持只能报告 `BACKEND_CHECK_NOT_QUALIFIED`，不是 `NO_ACTION_SPECIFIC_SIGNAL`。

通过也仅表明此小案例工程可行：96行→真实fixture1681+行、12views与更大inventory **不能直接外推预算**；须在任何预测试验前另有真实规模资格门。通过不批准3A的horizon目标，不批准完整系统，更不证明几何授权构念。

## 4. 拓扑范围：已有证据、缺口、建议

源码核对而非本轮重跑：

| 现有测试 | 能支持什么 | 不能支持什么 |
|---|---|---|
| `tests/test_topology_composition.py`、`test_topology_migration.py` | 行映射组合、new/survivor、reset及显式父系继承合同 | 真实训练触发次数、配对outcome或渲染/optimizer因果隔离 |
| `tests/gpu/test_gaussian_topology_mapping.py` | 实际clone/split选择函数传递parent indices、repeat次序及split后prune映射 | postfix/prune被recorders替换；没有验证原生参数追加/裁剪、Adam replacement或后继渲染 |
| `tests/test_transition_diagnostics.py` | 时间状态及映射后的lineage记录 | cluster干预下的后代RMS outcome、安全收益或跨branch拓扑干扰 |

真实clone/split缺席时，只能说“执行了一个允许拓扑的策略，但该事件未发生”；不能说“验证了clone/split后的geometry-authorization”。rowcount变化也不能单独识别clone/split，因为同时append/prune可能抵消。必须以后分别记录事件和ancestor-descendant映射。不得改梯度阈值、scale、随机样本或延长horizon来强造事件。

**当前核心问题是固定初始cluster的共享translation能否被正确测量，以及候选对短程更新是否有用；拓扑不是回答前一问题的必要条件。建议后置。** 最小后端检查完全固定拓扑，无optimizer；它不需要topology family或lineage outcome。既有合同测试保留，不重复开发。

若以后选择固定拓扑的**科学**首阶段，必须另批范围修订：删去16例topology-lineage预测family，原112预测→96（6family），另保留16支持拒绝对照，总112；所有分支8步禁止densify/clone/split/prune，ancestor mapping恒identity。按6family而非7familybootstrap；性能/benefit/safety阈值、horizon与候选不改变。此为总体和干预条件变化，须重新预注册，不能复用旧128/112例PASS定义。3A的horizon/fixture仍须单独批准。

其正结果最多说明**冻结topology的共享translation授权**；不能推广到生命周期、split/prune后安全、生产全局coupling、五状态或C1。拓扑后续需独立事件合同和原生端到端验证，不能自动恢复。本次仅建议，未修改正式域/门槛，也未运行任何分支。

## 5. 供选择的继续/缩小/停止决策

| 选择 | 理由及下一项可授权工作 | 仍不能推断/执行 |
|---|---|---|
| 继续完整线（目前不建议） | 须先批准唯一数学修订、完整3A/4A及真实规模backend资格，再考虑128例/拓扑合同 | 数学合理不是计算可行；现在不应投入完整框架 |
| **缩小（建议）** | 先决定§2数学修订，再审批§3单案例检查方案；另行授权才写/执行最小检查。过关后再审固定拓扑科学子规格，而非自动扩建 | 未解决的构念、horizon选择和真实规模成本仍待验证；绝不承诺科学成功 |
| 停止/归档当前候选 | 若不接受语义修订，或认为独立导数/成本路线不值得投入，可不建系统 | 这是设计/预算决策，不是已冻结科学阴性；不能据此签geometry终止release或读Utility GT |

请分别决定：**是否批准§2唯一公式/conditioning修订；是否原则接受§3检查方案及工程容差/预算（执行仍单独审批）；是否接受拓扑后置的范围方向。** 1A/2A的原则接受已记录；完整3A/4A不因此通过。Utility GT release及任何实验/方法修改仍分别待授权。
