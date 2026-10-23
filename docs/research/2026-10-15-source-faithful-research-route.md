# 以 AmbiSuR 现有接口验证候选选择与深度传播

日期：2026-10-15。研究综合答复；不替换已批准设计，不是实施规格或训练授权。本地 HEAD 为 `62e9117dfb2e20a06b257d166d8a6636d37845cc`。用户允许必要时在独立临时目录下载、修改、运行论文代码；本轮下载了梯度方法源码并运行 NumPy 合成数学探针，未运行方法训练、Torch/CUDA 或访问 GT/已有结果。

## 1. 当前问题不是模块太少，而是论证、复用和检验对象混在一起

当前有四个需要先澄清的问题。

1. “观测条件影响 SH 估计”被过度延伸成“现有 S 能校准几何错误/先验需求”。前者有条件统计依据，后者尚无充分证据。此前没有修改 raw A，应明确这是研究路线调整，而非声称已经完成 A 的校准。
2. 基线已经有 PGSR 的反投影、往返一致性、单应性 patch warp、NCC 和 Gaussian 标签渲染。它们是需要复用的基础设施，不是再借 VA-GS 后新增的贡献。
3. 候选比较、匹配度量替换、梯度协调、独占参数更新被提前堆进一个实验，无法归因。用户指定的首要假设应只检验新增选择能否让现有 ALR 更有效。
4. 传播监督确有几何来源，但必须说清锚点、法线、DA3 深度和两类置信度的分工；不能只说“补位置约束”就宣布互补。

因此当前路线收敛为：**先用 NCC 几何假设比较补充 SH 未命中的先验监督对象，完整复用标签渲染与 ALR；再独立适配 DP-GS 的传播损失和 VCR 的权重；只联合确有收益的项。** 第一版不加入固定特征网络、额外归属门槛、梯度准入、两步预览、QP 或新五状态。

这不是否定这些候选的研究价值，而是让每次实验回答一个问题。原实现的 best-source、位置更新、视角数量等任务选择不能仅凭静态阅读被认定为缺陷，亦不能默认改成“更严格”就更好。

## 2. A/S 的理论结论保留到哪里，为什么当前不修改 A

### 2.1 统计依据是条件性的，不是几何错误检测定理

局部固定几何/混合条件下，以非 DC SH 系数表示外观参数 β；DC 等干扰项需已固定或正确消除。设 `y=Yβ+ε`、`E[ε]=0`、`Cov(ε)=σ²I`，且 Y 满列秩。OLS 给出：

\[
\hat\beta=(Y^\top Y)^{-1}Y^\top y,\qquad
V=\operatorname{Cov}(\hat\beta)=\sigma^2(Y^\top Y)^{-1}.
\]

本地 `compute_weighted_sh_norm('equal')` 实际为非 DC 系数平方和的平方根。在忽略其数值 clamp 的条件模型中，\(A^2=\|\hat\beta\|^2\)，因此：

\[
\mathbb E[A^2\mid Y]=\|\beta\|^2+\operatorname{tr}(V).
\]

这是从 OLS 及二次型期望直接推出的条件恒等式；OLS 来源见 [Stanford CS229 讲义](https://cs229.stanford.edu/summer2023/cs229-notes1.pdf)。它说明相同真实方向外观下，不同采样设计/噪声可产生不同估计能量。一般加权能量则为 `β^T Wβ+tr(WV)`。

秩亏时无唯一 OLS；最小范数解会丢失不可观测分量。正则化时还存在偏差，不能一概说“支持少必然高能量”：

\[
\mathbb E\|\hat\beta_\lambda\|^2
=\|K_\lambda\beta\|^2+\operatorname{tr}(V_\lambda),
\quad K_\lambda=(Y^\top Y+\lambda I)^{-1}Y^\top Y.
\]

因此高/低 SH 都有多种原因：真实方向外观、噪声、病态采样、正则化与联合几何—外观优化。上述公式没有把它们唯一映射成错误几何。真实 GS 还包含可变 visibility、alpha 混合、曝光、几何耦合和优化偏差，不能把条件线性模型当成实际训练的完整生成模型。[AmbiSuR 方法与 SH 指标](https://arxiv.org/html/2605.12494v1)。

本轮 NumPy 例中，同样 2 个观测、相同 β/噪声，仅改变设计矩阵，预测平方能量由 .175 变为 .6725；Monte Carlo 分别为 .17491/.67086。它验证条件恒等式，未验证实际高斯的几何错误。

### 2.2 从这个结论不能跳到现有标量 S

需要分别证明四件事：观测影响估计；所选 S 能代表有关影响；校准后的 A 更能预测目标；被新增选中的对象使用先验后确有重建收益。当前只充分支持第一层。视角数/方向离散度或 PUP 的局部敏感性不能自动替代完整 Y、噪声与 nuisance 耦合，更不能直接成为错误概率。

此前被冻结的 N 在原协议未过 G1，仅说明那个公式/目标未得到支持，不否定一切观测校准方法。当前也不借该结果重新开 Geometry Q 或调整冻结公式。

**当前研究决定：不部署新的 S–A 算式，保留 raw A 与官方双端阈值，改进“何时追加先验”的选择决策。** 新路线不检验 A 校准假设，不能命名为已完成的观测校准方法。

观测条件并未消失：候选要有实际有效的支持照片和有效投影才能比较。但这不是把旧 S 换个名字。将来可把有效支持数等作为诊断分层，观察哪些条件下比较有用；未经结果不加入新的 S 分数、Fisher 门槛或混合公式。

## 3. 首要假设与最小技术路线：只改选择，保留先验优化

### 3.1 必须明确这是尚未验证的假设

H1：**AmbiSuR 的现有先验优化能帮助一部分 SH 未命中的高斯；用候选几何比较找出其中一部分、追加原有 ALR 监督，可以改善最终重建。**

不能把“SH 漏掉了错误高斯”“NCC 更好就一定受益”“ALR 会实现候选的全部几何变化”预先作为已证事实。H1 同时需要最终收益以及扩大 SH 比例/随机补选等简单对照。

### 3.2 对 A 的联系是明确的集合扩展，不是 raw A 修正

令官方双端 SH 标签为 `b_i^A∈{0,1}`。从当前参考相机可投影、且 `b_i^A=0` 的高斯中抽取固定预算候选 T；使用独立随机生成器，避免改变原训练的采样序列。A 已命中的对象不重复比较，也不取消其原监督。

\[
b_i^{\rm new}=b_i^A\lor e_i,\qquad e_i=0\quad(i\notin T).
\]

A 继续提供原有选择，新比较专门补其补集。无需先添加“A 异常附近必有漏检”的空间生长假设；此前提出的 A 邻域种子可以成为后续计算预算对照，当前不作为必要模块。

这属于对基线选择策略的增量，而不是声称发明更好的 SH 能量指标。若直接增大官方阈值范围也能获得相同收益，复杂候选比较的价值就未被证明。

### 3.3 先按 Sparse2DGS 的比较机制适配

对高斯 i，在参考相机建立两个局部平面：

- G_i：当前高斯位置和由最短轴获得、按参考相机定向的法线。
- P_i：在同一参考投影位置读取 DA3 深度反投影得到的位置，以及现有 DA3 深度导出法线。须使用基线真实的深度尺寸、尺度和坐标变换，不把 DA3 深度当 ray length。

采用列向量外参 `X_s=R_sr X_r+t_sr`，参考平面 `n^T X_r+δ=0`，单应性为：

\[
H_s=K_s\left(R_{sr}-\frac{t_{sr}n^\top}{\delta}\right)K_r^{-1}.
\]

从同一参考 patch 经两种平面 warp 到支持真实照片，比较 NCC。基线已有这些几何与采样工具。不能把该单应性或往返检查作为我们新增的贡献。

保留 donor 的 best-source 比较形式：在相同有效支持照片集合上，分别计算两候选最小代价 `c_G=min_s ell_s(G_i)`、`c_P=min_s ell_s(P_i)`，**允许二者最优源不同**；这正是原始 best-source 选择，不改为两视角都胜出、同一赢家或梯度共识。

三视图 donor 的参考视角有两个源；迁移到多视图场景时，首版从基线已有真实邻近相机列表取两个源，沿用其 patch 设置，缺少可用源时不追加标签。这是相机列表适配，不宣称更改视角数/聚合会更好；也不取虚拟相机生成的图像作为真实匹配观测。

\[
e_i=\mathbf1[i\in T]\,\mathbf1[c_P<c_G]\,\mathbf1[c_P<0.5].
\]

.5 是 donor SGU 绝对代价阈值的移植起点，不是已验证的 AmbiSuR 最优值。支持集合、边界、相机可见性按已有投影有效性处理；不能只对当前 G 先用“匹配很好”筛掉困难对象再做比较。匹配失败、空集合或无效平面不追加标签，原标签保持。

Sparse2DGS §3.4 明确比较 Gaussian 原语解释和 rendered depth/normal 解释的 NCC 后选择位置更新；`train.py:170–202` 与 `update_points` 给出源码路径。本项目替换为 DA3 候选、将输出改接基线标签而非直接位置替换，这两点是明确的适配，**不是完整 SGU 的原样复现**。[论文 §3.4](https://arxiv.org/html/2504.20378v1)、[比较调用](https://github.com/Wuuu3511/Sparse2DGS/blob/af15587eb186ca7ab12a2cf795775fa0f473f25d/train.py#L170)、[更新源码](https://github.com/Wuuu3511/Sparse2DGS/blob/af15587eb186ca7ab12a2cf795775fa0f473f25d/scene/gaussian_model.py#L476)。

原 SGU 在其 DTU Table5 中 CD 从1.201降至1.152，相比 ADC 的1.234更好。这支持测试选择性几何比较，但不证明改成标签+ALR仍保留收益，更不预测本项目数值。

### 3.4 两个 NCC 用途不同，但不需要两个匹配模块

| 使用位置 | 计算对象 | 结果如何使用 | 来源/新增边界 |
| --- | --- | --- | --- |
| 基线 NCC loss | 当前渲染几何产生的 warp | 可导损失，持续优化当前几何 | 基线 PGSR 继承；保留 |
| 候选选择 | 当前原语与 DA3 候选产生的两种 warp | no_grad 比较，输出额外二值标签 | 借 Sparse2DGS 假设比较；接现有 ALR |
| DP 锚点 | 当前渲染深度的几何往返与 NCC | 形成锚点 confidence map | 独立适配 DP 原机制 |

同一观测代价可被用于优化当前状态，也可用于多个显式假设之间的比较。这一用途区分有 donor 的明确实例，不是为了新增反投影模块找理由；它也不产生统计独立验证集。DP 原文 §3.1 亦明确区分把一致性当 loss 与用它估计 depth confidence。[DP-GS §3.1](https://arxiv.org/html/2607.03765v1)。

## 4. 标签渲染接口完全保留，不增加独占归属门槛

本地真实链为：

`compute_weighted_sh_norm → b_A → input_all_map[:,5] → rasterizer → out_all_map[5:6] → rendered_unc.detach() → loss_unc`。

令 `w_i(p)=alpha_i(p)T_i(p)`，现有标签图就是：

\[
M_b(p)=\sum_i w_i(p)b_i.
\]

把 `b_A` 换成 `b_new`，仍用原通道、原 blending 和原 ALR：

\[
L_{\rm ALR}=\lambda(t)\,\operatorname{mean}_p
\operatorname{sg}(M_{b^{\rm new}}(p))\,Q_{\rm DA3}(p)
\,[1-n_{\rm DA3}(p)^\top n_{D_\theta}(p)].
\]

其中 Q_DA3 是现有先验置信度准入，normal 路径、alpha 因子与 mean 归一化均按基线保留。第一版不另加一个额外 ALR loss，避免重复计数与改变已有调度。[renderer 通道](https://github.com/Fictionarry/AmbiSuR/blob/88d64054f53e0eba9ce49198282edf4a67fc8ca8/gaussian_renderer/__init__.py)、[ALR 调用](https://github.com/Fictionarry/AmbiSuR/blob/88d64054f53e0eba9ce49198282edf4a67fc8ca8/train.py)。

像素与选中高斯有确定的加权贡献对应，但通常不是一一对应。即使 mask 只来自一个选中高斯，深度法线残差仍依赖其他混合高斯，梯度可能到达它们；这与现有区域监督一致，**当前不要求只更新选中高斯，因此不构成额外准入门槛**。撤回此前 top-contributor、主导占比与独占参数路由作为第一版条件的建议。

代码适配时须同步 renderer 标签与训练侧重算的 unc_mask，避免两个地方继续各算各的 SH-only 标签。已有 requires_grad_ 切片赋值风险需要核实，但不因本实验擅自修复参数分离；若未来修复，必须单独控制其收益。

还需处理真实时序：标签必须在对应 render 之前准备好，不能修改已渲染的 map。因为本候选 G_i 取原语参数、P_i 取已有先验，不依赖参考视图的混合深度，可以先在 no_grad 中完成候选比较，再将同一 b_new 传入主渲染与 ALR。若实现选择依赖首遍 render 的 radii/depth，则必须显式计入第二次标签渲染，不能声称只改 Python 变量已更新掩码。支持图深度可复用已有绘制/缓存接口，额外计算成本仍需实测。

额外标签与原标签不交叠，固定几何下有 `M_new=M_A+M_E`。关闭模块或没有额外标签时复原原标签；无需新的 CUDA 特征通道、遮挡系统、持久五状态或生命周期。新增标签实际效用还受原 Q_DA3 影响，应记录有效掩码增量，避免“选中了很多但 ALR 全被先验置信度挡掉”。

## 5. NCC 与固定特征：有后续研究空间，没有当前必须混用的证据

### 5.1 先尊重实际 NCC 实现

基线 `utils/loss_utils.py::lncc` 与 Sparse2DGS 的同名函数实际使用平方相关：

\[
\ell=1-\frac{[(x-\bar x)^\top(y-\bar y)]^2}
{\|x-\bar x\|^2\|y-\bar y\|^2+10^{-8}},
\]

再 clamp。代码不是严格的 `1-rho`；两份代码的有效 mask 阈值分别是 .9 与1，SGU 还有 .5 的选择阈值。应记录差别，不将其静默统一。[Sparse2DGS NCC](https://github.com/Wuuu3511/Sparse2DGS/blob/af15587eb186ca7ab12a2cf795775fa0f473f25d/utils/point_utils.py#L72)。

忽略 epsilon 时，中心化与归一化使相关系数对 patch 的仿射亮度变化有不变性；平方后连反号对比也不区分。常量 patch 方差为零，实际源码使代价为1，不提供可区分的匹配支持。反射、饱和、局部非仿射变化和重复纹理超出这种不变性。NumPy 同式例核对了这些性质，但未执行 PyTorch conv2d producer。

### 5.2 固定特征真正新增的是描述子，不是反投影

FatesGS/VA-GS 冻结了学习到的图像描述子，用它的匹配约束替代/补充 RGB/gray 的光度匹配。它们的潜在不同在于描述空间、上下文与训练得到的匹配性质，反投影本身已经存在。冻结还避免描述子在当前场景训练中主动改变来迎合错误几何，但不保证反射域泛化或正确对应。

| 原文实验 | 数据 | 支持什么 | 不支持什么 |
| --- | --- | --- | --- |
| FatesGS Table4，rank+smooth 无 feature2.56，加 feature1.37 CD | DTU little-overlap | 该方法中 feature 对 depth constraints 有互补作用 | 不是 feature 替换 NCC 的等预算实验 |
| VA-GS Table4/8，无 feature F1 .53，完整 .54；时间15.9→20.9min | TNT，RTX4090 | 在保留 photometric alignment 的框架中 feature 有增量和成本 | 不能据此推出 feature 更适合本项目候选选择 |
| Sparse2DGS SGU1.201→1.152 CD | DTU，原框架 | NCC 假设比较已有正实证 | 不证明高光区域全部可匹配，也不证明标签适配有效 |

来源：[FatesGS](https://arxiv.org/html/2501.04628v1)、[VA-GS](https://arxiv.org/html/2510.11473v2)、[Sparse2DGS](https://arxiv.org/html/2504.20378v1)。这些是论文原数值，不是本项目结果。

**当前统一用 NCC 完成两个候选比较，基线 NCC loss 原样保留。固定特征与 C_H/g_H 均退出第一版。** 后续若有可定位的 NCC 失效，保持候选、视角、warp、预算与 ALR 不变，仅把 selector 的描述子/代价替换为冻结特征，才能判断该替换的价值。未经这个对照，不新增不同位置各用一种匹配方式的组合。

## 6. 独立适配 DP：DA3 深度和先验置信度都继续使用

### 6.1 四种数据的职责不能混淆

| 输入 | 当前作用 | 传播分支作用 |
| --- | --- | --- |
| DA3 depth D_P | 初始化、现有 depth loss、ALR 的法线来源 | 继续存在；先提供 depth-derived normal，非默认传播锚点 |
| DA3 confidence Q_P | 现有 depth/ALR 先验准入 | 原路径不删除；不等同 DP 的观测一致性 confidence |
| 当前渲染 depth D_G | 当前几何、基线一致性 | 被几何往返+NCC 支持的像素成为锚点 |
| normal prior n_P | 当前 ALR | 给出锚点到邻域的局部传播关系 |

所以用户对“用基线多视角一致性找锚点，再用 DA3 法线传播”的理解基本正确，但锚点深度取 **当前渲染深度**；保留所有原有 DA3 深度/置信度监督。DP 原文用预测 normal，换用 DA3 depth-derived normal 是本项目明确的输入适配，其误差相关性不同，不能继承原文收益。

### 6.2 传播式直接来自局部平面几何

使用 z-depth 与非单位针孔射线 `r_i=K^{-1}p_i`，三维点 `X_i=d_i r_i`。同一局部平面满足 `n^T(X_i-X_j)=0`，因此：

\[
d_i=\frac{n^\top r_j}{n^\top r_i}d_j.
\]

等价微分形式来自 `X(u,v)=D(u,v)r(u,v)` 的切向条件 `n^T∂_u X=0`：

\[
\partial_u\log D=-\frac{n^\top\partial_u r}{n^\top r},
\quad
\partial_v\log D=-\frac{n^\top\partial_v r}{n^\top r}.
\]

法线定义局部深度变化关系，锚点提供积分/位置条件；这不是凭空构造 loss，也不是把两份单点深度取平均。离散局部平面式见 DP §3.1 Eq6；同法线本身不足以证明两点共面，不能跨墙/椅子等不同表面任意连边。[DP 原文](https://arxiv.org/html/2607.03765v1)。

### 6.3 与 DA3 的互补和冗余可以推导，而非预设

若 D_P 在 patch 内恰好对应这一平面，`a_ij=(n^T r_j)/(n^T r_i)=D_P(i)/D_P(j)`，从渲染锚点得到：

\[
D_{\rm prop}(i)=a_{ij}D_G(j)
=D_P(i)\frac{D_G(j)}{D_P(j)}.
\]

因此它沿用先验的局部形状比例，以渲染锚点校正局部位置尺度。若 `D_P=s D_true`、法线正确而锚点 `D_G(j)=D_true(j)`，传播恢复理想平面真深度；若锚点等于 D_P(j)，则传播目标就是 D_P(i)，没有产生不同的几何目标。上述是理想条件的推导，非真实 DA3 全局误差模型。

本轮用1.3倍偏差的理想平面验证：真锚点传播最大误差 `4.4e-16`；先验锚点传播与原先验最大差 `8.9e-16`。这是合成代数，不是论文效果复现。

normal 由 D_P 导出并未创造一份独立测量；收益若存在，来自选择与耦合使用已有信息。即使传播目标等于 D_P，改变覆盖、权重和调度也可能有优化效果，所以不能把一切收益归为“新位置关系”。应有同传播掩码/同权重、监督原 D_P 的控制，分辨目标变化与监督范围变化。

### 6.4 真实代码如何接，哪些原选择先保留

DP `train.py:227,291–304` 对采样 RGB patch 求 NCC，把 `(2−ell_NCC)/2 * exp(−roundtrip_error)` 按 `valid_indices` scatter 到 `[HW,3]`，再对 RGB 取均值得到锚点 mask。三列是 RGB 通道。基线当前只返回过滤后 gray NCC 向量/mean loss，不能作为这张 map 直接传递。

第一移植保留原 RGB confidence producer、相机选取和 patch 设定，只复用基线 warp/相机接口、补足索引 scatter；不改原基线 gray NCC loss。DP 此处用抽取的一个 nearest_cam，并不是 Sparse2DGS 的双源 best-source。这里 gray 与 RGB 不同有明确 donor 来源，不能为了“统一”而悄悄改变锚点算法。若改 gray confidence，是后续另一个对照。

DP `scene/cameras.py:85–97` 用5×5法线相似度>.99形成 normal-edge mask；`train.py:300–335` 用 unfold/传播权重迭代，在 no_grad 下生成目标，然后计算 masked depth L1。先保留它，不换成新分割器、全局平面图或新不确定性模型。

\[
L_{\rm prop}=\frac{\sum_p M_{\rm prop}(p)
|D_G(p)-\operatorname{sg}(D_{\rm prop}(p))|}
{\sum_p M_{\rm prop}(p)}.
\]

将它加在现有 DA3 depth loss 旁，不替换 ALR。没有锚点/最终 mask 空时新增项为零；计算退化显式退出。源码 depth ratio 的 get_pos 使用 arctan 坐标，应先保留作者实现作参考、用合成平面检查其与本地 z-depth/normal 坐标的关系；不能静默换成另一射线定义后仍说原样复现。必要坐标适配与 donor 近似需分别记录。[DP 训练](https://github.com/hanl2010/DP-GS/blob/dc1ce835ea3c6bb7887333c50df7a13497a45c09/train.py#L227)、[传播工具](https://github.com/hanl2010/DP-GS/blob/dc1ce835ea3c6bb7887333c50df7a13497a45c09/utils/graphics_utils.py#L212)。

保留当前初始化、原深度/ALR、拓扑与其他损失，不连带移植 opacity decay 或异常深度边缘平滑。有限有效输入中的聚合选择不提前“改进”。源 normal-edge 不是跨表面连接的完备证明；其局限记录在实验中，不以新增复杂几何分类为前置。

DP Table4 在同样 normal+opacity-decay 条件下，加传播 CD1.55→1.13；这是移植动机。但原实验没有 AmbiSuR 的 DA3 深度监督，故不能从该数据推出加入当前基线必然有收益。

## 7. VCR 以及梯度方法的位置：分别作对照，当前不强行串联

VCR 的 detach normal-agreement 权重可以单独加到现有 ALR：`exp((dot(n_direct,n_P)−1)/tau)`，tau>0；坐标/符号/单位法线对齐，保留现有 DA3 confidence。它有直接 donor 实现，但只度量当前模型与先验的方向一致，不称为真实概率。本轮不因“平行平面偏移”提前扩展该模块；先测忠实权重适配是否有用。[VCR cos_weight](https://github.com/HLinChen/VCR-GauS/blob/aa715d19bfacfa9d491f477c572eab1839dcee3e/tools/loss_utils.py#L135)。

乘权重本身并不是问题。此前 N 的问题是目标语义和组合依据不充分；VCR 的权重有明确残差、作用位置与来源实验，但移植后仍需验证。

本轮实际下载并阅读以下算子和训练调用，未仅看 README：

| 方法 | 固定版本/源码 | 算子内容与适配判断 |
| --- | --- | --- |
| PCGrad 官方 TF | [c5fbd7c8，PCGrad_tf.py](https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/PCGrad_tf.py) | 逐任务梯度展开、负内积投影、合并；数学基准，不将 TF wrapper 接入 PyTorch |
| PyTorch PCGrad 复现 | [e987ac60，pcgrad.py](https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/e987ac603fa1accd386820a985a6dc2fd92dec5b/pcgrad.py#L44) | _project_conflicting、_pack_grad；wrapper 多次 backward/清零需与基线分离，不能直接包上训练 |
| ConFIG 官方 | [94862437，grad_operator.py](https://github.com/tum-pbs/ConFIG/blob/94862437f451f175673bce9c85f3e14bd9182c21/conflictfree/grad_operator.py) | 归一化梯度矩阵的 lstsq/pinv，再定长度；PCGradOperator 直接接梯度矩阵，固定遍历并求和 |
| CAGrad 官方 | [dc3d4815，toy.py](https://github.com/Cranial-XIX/CAGrad/blob/dc3d48152b6196945cfd56144879b9d42353b095/toy.py#L299)；nyuv2/utils.py | 两任务 scalar 或多任务 simplex 求解、平均梯度+修正、rescale；NYUv2 版本写三任务是场景设定，需要数量适配 |
| RotoGrad 官方 | [f9249856，rotograd.py](https://github.com/adrianjav/rotograd/blob/f92498563f2ac47e736b20b1d86be860e993dcdd/rotograd/rotograd.py#L330) | 共享 backbone/heads、representation hooks、可训练旋转与幅度；不是 Gaussian rotation，也不是 HarmoGS 双梯度角度操作的替代实现 |

若后续需要梯度对照，优先简单 PCGrad，再 ConFIG；CAGrad 在这些对照确有必要后加入；RotoGrad 暂不移植。ConFIG 的 PCGradOperator 与随机顺序/平均版本在顺序和归约尺度上不同，比较时应注明；不能让平均/求和变化冒充算法收益。ConFIG 的最小二乘解也不能在相反梯度等秩不满足条件下保证严格共同下降。

CLEAR Eq7 的算子在**人为设定 tau_c=0、omega=1、epsilon忽略且主梯度非零**时，退化为辅助梯度对主梯度的一次 PCGrad 式单向投影：

\[
g_P'=g_P-\frac{\min(0,g_B^\top g_P)}{\|g_B\|^2}g_B.
\]

这是算子的数学对应，不是引用继承，也不是完整 PCGrad。原 CLEAR `omega=s(1−q)`；即使 tau_c=0、q=0，omega一般也不是1。例如 cosine=−.5 时 omega=.5，仍保留部分负内积；本轮数值例也核对了这一点。不能声称只去掉容忍度和证据就自动得到全投影。[CLEAR Eq7/11](https://arxiv.org/html/2608.02206v1)。

**这些全部退出 H1 首轮。** 用户提出的诊断顺序成立：先看新增选择与原 ALR 的端到端效用；只有候选有区分而收益不稳定时，再查先验内容、优化路径和冲突。梯度分析是诊断工具之一，不再优先开发两次 optimizer 模拟。

## 8. 当前完整研究路线与可证伪的创新边界

| 阶段 | 实际新增 | 首要验证 | 简单反解释/对照 |
| --- | --- | --- | --- |
| 候选选择 | 原有 NCC 比较额外的 DA3 平面，输出新增标签 | baseline+selector 是否改善最终几何 | 扩大 SH 选择、同数量随机补选；记录 ALR 有效掩码量 |
| 传播独立 | 真实 RGB confidence 图、DA3 normal 输入、传播 depth loss | baseline+prop 是否优于原 DA3 depth loss | 同传播掩码/权重但目标为原 DA3 depth，区分覆盖和目标 |
| VCR 独立 | 在原 ALR 上加 donor agreement 权重 | 简单先验加权是否已足够 | 只用原 SH；同权重开关，先不与 selector 捆绑 |
| 联合 | 只组合前面有明确收益的项 | 是否互补且成本可接受 | 各单项与联合；无需数学超加性才能证明有效 |
| 条件后续 | 仅针对已定位失败替换匹配或加梯度协调 | 失败是否真的被新机制改善 | NCC vs feature；原梯度和 vs PCGrad/ConFIG |

标签实验中原有 depth loss、ALR residual/权重/调度、视角训练、拓扑与正常梯度流程均固定。传播实验中先不加 selector/VCR。关闭功能时 G0 等价；数值退化时只取消新增项，不修改 baseline 管理动作。

没有独占梯度门、best-source 仍按 donor、没有新状态，适配范围比上稿小。真正需要验证的接口只有：候选平面/相机坐标；同一 b_new 进入 renderer 与 ALR；NCC 索引与 RGB confidence map；DA3 normal 的坐标与传播深度约定。实际 producer→consumer 边界应在有 Torch 的隔离环境用小型合成输入贯通，不以手写 mock 或纯 NumPy 数学核对替代。

候选阶段的潜在贡献是改进 AmbiSuR 的先验准入覆盖；传播阶段的潜在贡献是适配与有效利用既有先验的邻域关系。反投影、可见性、NCC、标签渲染、PCGrad、DP 传播公式和 VCR 权重均应明确归于原作者。A+B 组合可以形成有效工作，但需展示任务适配、与简单替代的差异、端到端收益及联合价值，不能把所有借来的基础设施逐项列为新创新。

当前可以向导师解释为：**原 SH 指标保留；以显式几何假设比较探索其遗漏的可受益先验监督对象，监督仍由原 ALR 执行。另一个独立问题是如何利用现有先验的局部关系，由受观测支持的渲染锚点约束邻域深度。两者先分别验证，再决定是否组合。** 这条路线不再声称已解决 A 的观测校准、所有几何错误识别或梯度冲突。

## 9. 复核证据、运行边界与旧报告关系

源码读取包括本地 SH/标签/ALR/NCC、Sparse2DGS 比较和更新、DP confidence/normal-edge/传播，以及五个梯度仓库的实际算子。固定版本、哈希和具体路径另存 `2026-10-15-research-source-manifest.json`；上一轮八仓库来源仍保留并核验，不把旧下载全量文件当成全量精读。

已运行 NumPy 合成探针：条件能量期望、秩亏最小范数、平方 NCC 退化、标签混合及非独占导数、锚点传播的尺度校正/冗余、CLEAR 全量与默认软校正的区别。它们检查数学，不是 donor PyTorch 执行。本地捆绑 Python 没有 torch/tensorflow/scipy，未为研究答复安装这些依赖，也未运行仓库 setup、模型下载或训练。故本轮能报告静态接口可行性，不能声称实际适配已运行通过。

本文件取代10-13报告中第一版所附的 ownership/特征梯度/两步预览前置建议；没有删除其审计历史或数学更正，也不静默覆盖原批准设计。未来正式方法代码、实验、GT 访问与运行环境仍按项目相应合同推进。

阶段只读审阅对主体未发现 Critical/Important；唯一 Minor 为审阅时来源 manifest 尚在生成，现已补齐并经脚本核对。审阅没有认证 Torch/CUDA 接口、运行成本、收益或新颖性，相关边界保留。
