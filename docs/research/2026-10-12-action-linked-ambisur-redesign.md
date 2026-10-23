# 2026-10-12 AmbiSuR：从 SH 异常筛选到有证据的几何修正与梯度融合

研究备忘录；不是批准设计的修订，也不是实验或实现授权。当前算法、冻结结果、Utility 恢复记录和 Geometry Q 阶段均不改变。

## 1. 本轮结论与检索边界

用户指出的两处问题成立：PUP 的局部敏感度不能直接认证几何观测充分度或几何错误；用可靠性权重对照片/先验几何取折中，也没有充分回答更新该走哪个方向。本轮不再把这两种推理作为方法成立的证据。

本轮以截图去重后的 22 篇为入口，补充 28 条相关记录，共 50 条；本轮检查相关方法/实验的 22 条，沿用上一轮已检查相关全文证据的 11 条，其余 17 条只到主源元数据/摘要或基线相关筛选。详见 [文献清单](D:/research_Space/noob_AmbiSuR/docs/research/2026-10-12-literature-inventory.md) 与对应 JSON。这不是 50 篇逐页精读，也不是已经找尽 2025–2026 的论文。检索覆盖表面重建、几何/先验置信度、反射和低照度、失败检测、梯度冲突、结构传播与跨领域验证重加权；未挂载学术 MCP，使用公开主源。未读到细节的论文不承担具体方法论证。

较上一轮，更适合作为主线的是 2025–2026 的 Sparse2DGS、FatesGS、VA-GS、CLEAR、HarmoGS，以及 2026 的法线引导深度传播。Ref-NeuS 是 ICCV 2023，DebSDF 是 TPAMI 2024；它们保留为历史机制依据，退出“最近最适配主线”的位置。HarmoGS、CLEAR、Confidence matters、GPU-SDF 等未确认正式 venue 的记录仍标作预印本，不因年份新就视为已验证方案。

推荐研究问题：**超出 SH 异常范围，识别有观测或结构证据支持的具体几何修正；再用同一证据决定先验梯度的准入、保留分量与执行方向。** 选择和优化都围绕修正是否可信，而不是先计算一个通用风险值、再另造五类状态。

## 2. 先纠正基线描述，否则很容易提出已经存在的“创新”

[AmbiSuR §3.3–3.4](https://arxiv.org/html/2605.12494v1) 与本地审计表明：

| 基线位置 | 已有机制 | 合理的研究增量 |
|---|---|---|
| SH ambiguity selector；train.py 459–463 | 非 DC SH 范数的高端和低端分位筛选 | 用具体修正的支持证据覆盖 SH 未命中的几何；与官方双端筛选比较 |
| DA3 initialization + depth supervision；train.py 435–443 | DA3 初始化及置信度 mask 下的深度监督 | 先验准入的验证/结构性使用，而非“首次加深度” |
| ALR；train.py 474–475 | DA3 深度导出的法线和 rendered depth normal 的区域监督 | 对原语位置/朝向施加结构约束，或对其梯度做有证据的融合 |
| parameter separation | 先验作用于受选几何，限制外观/部分参数通道 | 在几何参数内部控制实际更新方向，不把已有分离重新命名 |
| Ray-Color / primitive truncation | ray 内一致性及截断 | 拓扑若改动，需要说明与已经接受的几何证据如何共用 |

本地当前 Core 的连续 A 是上端分数，不能代替官方双端 selector 作为唯一比较基线。未来研究比较必须同时记录二者，固定选中预算和相同优化动作，否则“新检测器更好”可能只是恢复了低端分支或增加了监督数量。

本地布尔切片 requires_grad 等实现风险尚需验证。修复这类代码行为归于实现正确性，不是科学创新。DA3 已经影响模型初始化和训练，所以内部一致性与 DA3 先验不能宣称统计独立。

冻结 Tool Room 7k 的 N/A AUROC 是 .547598/.545626，增益 .001972，原 G1 已失败；当前研究不能用调阈值、延长训练或改名 S 覆盖这个事实。Utility 的接口失败是 INCONCLUSIVE，不提供新方法效益的结论。当前 r_g 仅是一致性 telemetry，没有资格直接当真值或保护锚点。

## 3. PUP、S 与“有效约束”究竟缺少什么

[PUP，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Hanson_PUP_3D-GS_Principled_Uncertainty_Pruning_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf) 支持用当前位置/尺度下的渲染敏感度选择压缩时应保留的原语。它没有证明低信息 Gaussian 的位置错误，更没有证明它适合接受 DA3 法线修复。

对任一局部信息或覆盖分数，都要分清：

1. **约束是否可信**：对应点、可见性、相机和先验是否可能有污染或系统偏差？
2. **约束能区分什么**：沿哪一个位置/转动方向改变几何，其他视角能否区分？颜色/SH 的重新拟合能否掩盖几何变化？
3. **这条修正是否有帮助**：可信先验提出的具体更新，在其他证据上是改善、破坏，还是无法判定？

低灵敏度只对应第二问的一种局部回答；它不是第一问或第三问的答案。重复视角可以提供大量像素却没有新的三角测量方向；有偏观测可以产生很强的 Hessian；正确模型在驻点的梯度也可为零。高/低敏感度都没有几何真假的一一对应。

SH 正交性说明非 DC 系数能量关联整球面的视角颜色变化。[AmbiSuR 的推导](https://arxiv.org/html/2605.12494v1) 不把它等同几何误差；低能量端也包含训练折中的经验观察。因此 A 适合做候选线索，不能承担完整错误标签。给 A 乘一个未校准 S，只是把两个意义不同的代理合并，并没有补出缺少的真实性证据。

建议将 S 的研究拆开，暂不坚持一个标量“充分度”：
- 可信支持视角与匹配是否有效；
- 几何假设的可区分程度（代价是否平坦/多峰）；
- 替代假设或更新方向是否改善支持证据。

若仍需信息矩阵解释，应在外观 nuisance 消元和可信残差模型下分析局部方向；这仍只能声称条件局部可辨识性。它不是检测器性能或真实几何正确性的证明。

## 4. 主源机制与消融：可以借什么，不能借什么

以下数值均为原论文自己的实验，未在 AmbiSuR 复现；不同协议的 CD/F1/PSNR 不作横向排名。

| 主源 | 核查的机制/结果 | 对本项目的含义与边界 |
|---|---|---|
| [Sparse2DGS，CVPR 2025，§3.2–3.4/表2–5](https://arxiv.org/html/2504.20378v1) | 固定颜色/特征抑制外观吸收几何误差；采样原语局部点，直接约束位置/朝向；比较 individual Gaussian 与 rendered surface 的 NCC 后选择更新。表2共有基准 CD1.201，DGPR1.142，SGU1.152，两者1.125 | “提出替代几何→观测比较→动作”的直接来源。其替代假设是 rendered surface，不是 DA3；换成 DA3、固定特征和可信支持是我们的待验证适配。单纯 MVS 初始化+2DGS CD1.27 并未超过 MVS1.26，完整方法1.13 |
| [FatesGS，AAAI 2025，方法/表4](https://arxiv.org/html/2501.04628v1) | 局部深度排序/平滑保整体，冻结多尺度特征重投影补细节。小重叠 DTU CD：基准2.47，排序+平滑2.56，三者1.37 | 组合应纠正各自失败模式。未报告全部 factorial，不能将该表当严格统计超加性证明。冻结特征比训练颜色更难通过 appearance 参数作弊，仍可能误匹配 |
| [VA-GS，NeurIPS 2025，§4.2/表4/补表8](https://arxiv.org/html/2510.11473v2) | 平面 patch 多视角对齐，区分出界可见性与回投影遮挡权重，加入固定特征。TnT F1 full.54，去 feature.53，去 photo.50，同时去二者.36 | 给 support validity 提供具体机制；遮挡判据依赖当前深度，会拒掉部分正确大修正。RTX4090 full20.9min、去 feature15.9min 是该论文成本，不能许诺本项目同成本 |
| [CLEAR，2026-08 预印本，式6–14/表5](https://arxiv.org/html/2608.02206v1) | 每 Gaussian/参数组比较 LR–HR 梯度；真实 LR 一致性验证 HR prior，证据共用于梯度校正、细节路径和增密。渐进消融 PSNR18.74→19.72→20.09→20.42，冲突率.243→.162 | 借证据到梯度/拓扑的贯通。HR detail demand 不是错误几何；真实 LR anchor 的默认优先级不适合反射几何。soft correction 也不能声称严格消除所有原始冲突。未有独立 mesh 证据 |
| [HarmoGS，2026-05 预印本，梯度协调/表4](https://arxiv.org/html/2605.13073v1) | 双视角 photometric 梯度在张成平面内旋转，保模长并正交；几何属性衰减、增密和持续冲突剪枝联动。full PSNR23.98/LPIPS.125；去 harmonization23.50/.131；去 conflict ADC23.84/.128 | 借不同视角冲突及属性层处理。不是 photo–normal prior 融合的现成证据；冲突可来自遮挡、反射、先验错误或模型表达不足，不能单独判错 |
| [Confidence matters，2026-08 预印本，表3](https://arxiv.org/html/2608.06117v1) | PGSR+VGGT深度/法线先验及网络置信度；三场景消融 F1.40 vs 去 confidence.37。完整6场景TnT平均 .50/.50，DTU CD.52/.53 | 置信度过滤有证据但不是普遍大幅收益或校准概率。向当前 rendered depth 对齐先验会有模型偏差；基础 confidence mask 已在 AmbiSuR，不够新 |
| [CoMe，ECCV 2026，式9/表3](https://arxiv.org/html/2603.24725v2) | 正 confidence 的残差加权与 log 正则、geometry/appearance loss 分工及 densification。渐进增加 confidence 时 ScanNet++ F1 .625→.655 | 直接几何指标支持 confidence 使用，但自由像素最优 C=β/Lrgb：描述拟合残差，不自动描述几何真值；正值不限定[0,1]。ray variance 与 AmbiSuR 既有机制重叠 |
| [UA-GS，EGSR 2025，§4/表4](https://diglib.eg.org/bitstream/handle/10.2312/sr20251192/sr20251192.pdf) | 用不同于颜色 A 的 SH 字段建模深度/法线视角 uncertainty，先验和 NCC 约束受其调节。无 uncertainty guidance CD.53，PGSR+prior.49，full.47 | 已有 DebSDF 向 GS 的近期移植，所以仅“加 prior uncertainty”很难构成新贡献；预测器也会跟模型自身残差走 |
| [GPU-SDF，2026-02 预印本，§III-B/C/表III](https://arxiv.org/html/2602.23926v1) | 翻转预测一致性估先验不稳定性，结合 edge distance field 和 multiview constraints。ND-SDF Chamfer6.4；加 depth+normal uncertainty6.0，precision72.1→73.7 | 可借输入扰动后的先验一致性检测；稳定偏差仍会漏检，法线/相机坐标必须正确变换。SDF专用边缘场不直接搬GS，书面 KL-inspired loss 需单独审查 |
| [法线引导高置信深度传播，ECCV 2026，§3.1/表4](https://arxiv.org/html/2607.03765v1) | 从几何+NCC 支持的当前 rendered depth anchors 出发，以局部同平面法线关系传播位置约束。normal+opacity decay CD1.55，加 propagation1.13，完整1.02 | 弱纹理区域可由结构关系得到位置，而非直接平均单目深度。当前锚点可能一致但错误；同平面和法线误差、距锚点太远都是明确边界 |

## 5. 推荐主线 M1：从检测“风险高”改为验证“具体修正得到支持”

### 5.1 提出两种几何解释，而不是融合两个位置

对 Gaussian i：
- 当前假设 G_i：原语中心及局部平面/朝向；
- 先验假设 P_i：用 DA3 的有效深度与法线构造局部候选 patch。只在适配研究中构造候选，不立即写回参数；
- 支持视角 H_i：来自其他训练相机，不参与当前局部候选拟合的视角。它们仍可能已经进入既有 DA3 的多视角输入和 baseline 训练，不能据此声称严格独立或 clean held-out。这里是支持视角验证，不是测试集或 GT；若未来要求严格生成/验证隔离，需要另行设计输入划分，不能重用已有全视角输出宣称隔离。

借 [Sparse2DGS](https://arxiv.org/html/2504.20378v1) 的采样 primitive 与 SGU 思想，在相同有效支持上比较 C_H(G_i) 和 C_H(P_i)。C_H 可为鲁棒的固定特征 patch 对齐代价，采用 [FatesGS](https://arxiv.org/html/2501.04628v1)/[VA-GS](https://arxiv.org/html/2510.11473v2) 的跨视角几何投影方式。

拟议证据 B_i=C_H(G_i)-C_H(P_i)。B_i>0 表示“这个先验替代在所选代理观测中更受支持”，没有资格直接改名“真实几何误差”或概率。检查支持视角子集一致性、重投影边界、候选间共同可见支持、候选局部扰动代价形状，防止候选通过丢弃困难像素取得虚假收益。

固定 feature/visibility/归一化约定后，只让几何投影改变；对验证动作暂不让颜色/SH 重新拟合。原论文在各自数据上的 feature robustness 不能保证镜面、饱和或暗部匹配正确。需保留“不足以区分”结果，尤其代价平坦或多峰、支持太少、两种候选都差时。

### 5.2 与 A 的具体关系

A 的用途是廉价地提出光度异常候选或分配审计预算；新增通道必须允许 A 未命中的 Gaussian 被支持证据发现，否则扩大不了覆盖。研究时需抽查全体或分层候选，不能只在 A 高的一批里验证。

保留官方 A 分支，并逐项测新通道：
- A 高但先验没有其他证据支持：不能自动加强先验；
- A 低/未命中，但 P 比 G 稳定地解释支持视角：是最应检查的额外修正；
- 观测弱且 P/G 无法区分：不是已经找到了错误，转入有限结构路线或不判定。

“更多错误高斯”要用固定选中预算下的新增 TP/FP、prior可用性、执行后局部改善三种结果分别报告。召回更大而 precision 崩溃或先验不可用，不算方法成功。

先验置信度和 GPU-SDF 式扰动一致性可用于 proposal validity；它们不是 B 的替代。DA3 的旋转/翻转测试需正确变换 intrinsics、poses、normal components 和度量尺度。第一版不建议加自由 uncertainty 网络：baseline已有conf mask，新增预测器容易学习“与当前模型不同”而不是“先验错误”。

## 6. 推荐主线 M2：以同一支持证据决定梯度准入和融合

### 6.1 数学连接：选择与执行都预测同一个代理目标的改善

令几何参数块 θ_i 为位置和旋转局部坐标。g_P=∇L_prior 是拟议先验修正的梯度，g_H=∇C_H 是支持视角代价的梯度。固定预条件器 P 时：

C_H(θ_i−ηP g_P)=C_H(θ_i)−η g_HᵀP g_P+O(η²).

这给出三个可检验结论：正内积表示足够小的该步在当前代理上有下降趋势；负内积表示局部冲突；近零表示当前一阶证据不能裁决。该式来自 Taylor 展开，不是新的几何真实性定理。

[Ren 等，ICML 2018 §3.2](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf) 利用验证梯度和训练样本梯度的关系决定重加权，提供跨领域方法来源。原文需要 clean/unbiased validation；本项目的固定特征验证并不是干净几何 GT，因此不能继承其正确性保证。这里只移植“根据外部验证效果选择更新”的原理，不移植每步昂贵的完整双层优化。

有限候选比较 B_i 与无穷小梯度检验不等价：前者说明替代几何在所用代理上代价较低；后者检查实际局部更新对代理的影响。2026-10-13 更正：这只能支持另设更新检验的可能性，不能证明 M2 必需；原稿“必要性”的判断过强。应先测 M1 单独增量，再确认是否存在有害更新。即使正确先验对应更低代价，非凸性、渲染混合和参数化也不保证局部更新朝向该几何；反之，若法线 ALR 对平面偏移梯度为零，准入模块也不能补出位置信息。

### 6.2 借 HarmoGS 与 CLEAR 的哪一部分

在几何参数块中先区分：
1. 同一观测通道内，不同相机的 photo/feature 梯度冲突；
2. 观测证据与 DA3 prior 梯度冲突。

HarmoGS 给出第一类的按属性协调和拓扑联动；CLEAR 给出第二类的 Gaussian 级证据路由。不能把两类梯度一次混在一起、再只算一个 cosine。也不预设某个通道永久优先：饱和反射的 photo 梯度可能误导几何，而有系统偏差的先验也可能误导。

拟议区别：受保护方向来自 M1 的可信支持证据/结构锚点，而不是比较未校准 r_p−r_g 后取较大者。photo-only 的 SH/color 通道可以保持外观拟合；geometry 的先验与 photo 分量由支持证据共同检验。这是 baseline 已有 parameter separation 上的方向控制增量。

### 6.3 一个具体且可审计的候选算子

以 d 表示实际下降方向，即 θ←θ−ηd。d_0 是候选融合后、包括规定 optimizer preconditioning/momentum 的实际方向。对经核验的支持梯度 h_j，研究局部投影：

min_d 1/2 ||d−d_0||_M²
subject to h_jᵀd≥0,  ||d||≤Δ.

约束集合来自支持证据，M 是事先规定的参数度量；Δ 是局部步范围。不能偷偷借 GT 决定保护集。单一过原点半空间、M=I 且步长约束未激活时（2026-10-13 更正：原稿漏写该成立条件）：

d*=d_0−min(0,hᵀd_0)/(||h||²) h.

同时处理所述欧氏步长约束时，先令 y=d_0−min(0,hᵀd_0)/(||h||²)h，再令 d*=y（||y||≤Δ）或 d*=Δ y/||y||（||y||>Δ）。此式要求 Δ≥0、M=I、单一过原点半空间；h=0 时只做球投影。一般 M、非齐次约束或多保护项不在此闭式声明内。用户反例 h=d_0=(1,0), Δ=.1 时，正确结果是(.1,0)，原稿结果(1,0)违反步长约束。此更正只修正文档数学，不改变批准算法。

2026-10-13 范围更正：这里 h/g_H 若来自固定特征代价，则只检验“新增先验更新与特征证据的兼容性”；它不自动覆盖 RGB–prior 冲突或支持视角内部协调。半空间保护的是代理损失的一阶趋势，不是把某些参数冻结，也不是认证高斯正确。多视角协调/保护集/QP退到后续候选，第一增量以基线回退和新增分支准入为主。具体代码审计和当前建议见 2026-10-13 的后续报告。

这是半空间投影/PCGrad 家族已有数学，不是我们发明的新 solver。[PCGrad](https://arxiv.org/abs/2001.06782) 和 [ConFIG，ICLR 2025](https://arxiv.org/html/2408.11104v3) 是必要比较对象。可能的贡献在“怎样获得受保护约束、怎样与 SH 漏检修正共享证据、怎样适配实际原语参数/优化器/拓扑”，而不是投影公式本身。

不同 protected constraints 会产生不同活跃约束/允许子空间，动作由优化问题产生，不需要先排列出恰好五种状态。完全相反且都要求严格改善的两条梯度可能没有公共非零方向；半空间非增约束总含零，但零更新没有修复效用。应报告停止/退让，不能硬造一个“折中就都好”的说法。

强制原始 noisy photo loss 不增加可能恰好阻止正确高光修复，因此只保护有明确证据支持的项。未进入保护集的 photometric 几何分量也不能随意放大，要研究其消融和失败条件。

### 6.4 原论文公式不是免责依据

HarmoGS 原文的修改后梯度正交，不保证二者相加同时下降原始损失。独立代数例：
g1=(1,0), g2=10(cos120°,sin120°)，两侧各转15°，
得到修改后正交梯度和 d≈(−1.6223,9.9181)，而 g1ᵀd<0。
这反驳的是无条件 common-descent 推论，不否定原文实证结果；掩码/衰减可能改善其实际行为。

ConFIG 附录 A.3 需要看 gradient matrix 的秩：参数维数大于损失数不能独自保证满行秩。g2=−g1 时，严格公共下降不存在。其高维神经网络实践直觉不能直接移植到每 Gaussian 的位置3维、opacity1维或旋转局部块。

Adam 下原梯度内积可以与实际更新效果异号。例如 gH=(1,1), gP=(2,−1) 原内积1；P=diag(.1,10) 后内积−9.8。公式审计需面对优化器真实方向和多项累积，不把 raw gradient cosine 当最终执行证据。小步 Taylor 也只保证局部一阶代理趋势，非全程/GT mesh 保证。

## 7. 更适合暗部/弱纹理的结构候选 M3：先验法线提供关系，可靠位置提供锚点

仅用固定特征并不能在所有暗部验证位置。[ECCV 2026 的深度传播](https://arxiv.org/html/2607.03765v1) 给出与 baseline ALR 不同的利用方式：先用有多视角支持的位置作 anchor，再让法线定义邻域位置之间的关系。

同一局部平面上，取 z-depth rays r=K⁻¹p、法线 n：
nᵀ(d_i r_i)=nᵀ(d_j r_j)，
所以 d_i=(nᵀr_j)/(nᵀr_i) d_j。

这不是对两条 depth 取平均。法线约束平面方向，anchor 决定平面位置；只使 normals 相似无法确定平面沿法向的偏移。暗区域没有有效 feature 时，可以由邻近 anchor 加结构关系得到一个可检验的修正 proposal。

适配候选：
- 由 M1 的有效跨视角证据或可核验 SfM 支持确定 anchor，避免原文当前渲染“自一致”直接等价正确；
- 只在法线连续且有局部同平面证据的 patch 内传播；图像 edge 只能作弱线索，反射边缘不能被直接认定几何边缘；
- 先验法线的稳定性/跨视角变换一致性决定允许传播范围，不要求 Manhattan 场景全局成立；
- 传播 uncertainty 不得越来越“自信”：对 f(d_j,n)=a(n)d_j 的一阶误差传播，独立近似下 Var(d_i)≈a² Var(d_j)+J_n Σ_n J_nᵀ，再计同平面模型误差；相关误差需保留交叉项；
- 当 nᵀr_i 接近零、跨越边界、normal不稳、anchor不受支持或传播距离太远时退出。传播不能从没有证据的区域创造正确性。

这种约束可作用于原语采样点 x_ik=μ_i+R_i diag(s_i)ξ_k：
L_plane=Σ_k ρ(n_Pᵀ(x_ik−X_anchor)).
采样 primitive 的连接来自 [Sparse2DGS](https://arxiv.org/html/2504.20378v1)；plane的位置/法线关系来自上述传播几何；具体结合是待验证提案。它直接影响中心与旋转等结构自由度，区别于只监督 rendered depth normal，也区别于添加 baseline已有 depth loss。

点到平面约束只限制法向距离，不能确定切向位置；切向细节仍需其他观测。若不具备平面适用性，可研究 FatesGS 的局部 depth ranking 作为较弱关系，但排序已有方法成熟，单独加它不能充当充分创新，也不能默认相对排序比 DA3 metric depth 总可靠。

M3 是解决 M1 观测证据失效的结构备选，不应第一版就与各种网络/损失一起堆入。应先确认失败来自弱纹理且有可靠邻近 anchor，而非相机误差或完全不可观测。

## 8. 拓扑动作：共享证据可以合理，但不是现在必需的第三创新

[IndoorGS](https://openaccess.thecvf.com/content/CVPR2025/papers/Ruan_IndoorGS_Geometric_Cues_Guided_Gaussian_Splatting_for_Indoor_Scene_Reconstruction_CVPR_2025_paper.pdf) 的类型由来源决定（SfM、line、plane-like），来源约束直接解释为什么 ADC 不同。它不证明动态双可靠性应有五状态。其 Table3 full F1.776、去 customized ADC.711，说明动作机制要单独验证。

[CLEAR](https://arxiv.org/html/2608.02206v1)/[HarmoGS](https://arxiv.org/html/2605.13073v1) 的可借点是同一准入证据对梯度和 densification 生效。否则一个被判不可信的 photo/prior 方向可能仍通过 view-space gradient 触发 clone，重新扩增错误。

[GeoSplat §3.1.2](https://arxiv.org/html/2509.05075v1) 限制 normal 方向更新并鼓励 surface growth，但它假定当前原语已接近表面。离面原语的修复恰好需要法向位移，不能在修复阶段一律投到 tangent。结构动作应区分证据已经支持的 repair 位移与后续局部 growth，避免提前锁死错误位置。

[GSPrior](https://arxiv.org/html/2603.19682v1) 的窄带、离带删除与投影依赖当前深度生成的 TSDF；这不是独立真值。它可以作为另一 ALR 方案比较，不宜与当前轻量主线同时引入。低信息或梯度冲突不构成直接删除理由，完整性代价需独立评价。

第一候选保持 topology 原逻辑，先研究检测+方向。之后若证据显示增殖在扩增被拒修正，再研究 accepted-gradient 统计及身份/Adam迁移；不可静默重写既有 C0–C6 合同。

## 9. 三个可选择的整体方案及优先级

| 方案 | 组合及 baseline 挂点 | 新增连接 | 适用性与风险 |
|---|---|---|---|
| I：观测验证的选择+梯度融合（优先研究） | SH selector + ALR/parameter separation；Sparse2DGS/FatesGS/VA-GS + CLEAR/HarmoGS/验证梯度思想 | 有限替代假设的观测支持与局部梯度准入共享证据；保护由证据决定，不固定 photo/prior主导 | 无需更换渲染表示；feature/occlusion proxy会错，暗部无法保证覆盖。性能与成本未知 |
| II：结构 anchor 修正+融合（条件备选） | ALR 与 SH 漏检覆盖；2026 depth propagation + Sparse2DGS primitive constraints + 同一方向准入 | 法线提供关系、anchor提供位置；传播不确定性和更新约束相连 | 适合有局部平面/anchor的弱纹理；复杂曲面/薄结构/错误anchor可能害人 |
| III：表示改造（暂不作为当前主线） | appearance/geometry representation；Ref-GS/GS-2DGS/SpecTRe-GS/Geometry Gaussians/GS-SDF等 | 分离反射、opacity或引入场表示，处理优化路由无法修复的表达瓶颈 | 多为元数据/摘要筛选；更换表示或增加传感器，工程和对照代价大，不能把完整方法效果预先归给路由 |

优先级只基于机制吻合、基线增量清晰和现有单RTX4090/24GB约束；不代表 I 已被证明最有效。预算应测真实峰值显存、平均/总时间、预处理与刷新开销，目标仍为可接受的约2倍以内，但现在不能承诺。VA-GS 作者仓库另提示完整 feature/mesh 流程 RAM>80GB，进一步说明不能照搬整套代码。Ren2018完整元优化需要额外训练开销，第一候选仅研究周期/小块一阶审计。

不把 ConFixGS/DualSplat 的名字直接贴模块；没有全文细节核查的项只能继续筛选。GIP-GS/UncertainGS 的出版摘要已提示 reflection-specific guidance/uncertainty+structural priors 有近邻；新颖性必须与其全文进一步对照，不能目前认证“首次”。

## 10. 证明方案不是故事：需要怎样的区分性实验

本节为未来预注册建议，不启动任何实验、不读取GT、不改变当前冻结门槛。

### 10.1 先把选择和动作分开

共用 evaluator/statistical core，比较：
1. 官方双端 SH + 原 ALR；
2. 新 selector + 原 ALR；
3. 原 selector + 新 fusion/operator；
4. 新 selector + 新 fusion/operator。

主控选中数量、prior次数、支持视角预算、初始化、训练步与采样种子；另报实际总时间。计算/数据量不同需等开销对照，不能以更多 backward 当理论贡献。对 M1 附加“全局直接加入同一feature loss”对照，区分更强监督与更好的选择/准入。

2×2 可以测交互，但贡献不要求收益大于两单项之和。两项联合优于baseline且机制清楚、有可复现效益可以有研究价值；不能从一次正结果承诺发表。Sparse2DGS 自身两模块数值呈收益重叠，仍优于单项，这很正常。

### 10.2 选择证据到底在预测什么

分开报告：
- 几何错误检测（固定预算 precision/recall、新增TP/FP，含低/高A分层）；
- 先验可用性（错误Gaussian中prior是否提供正确位置/法线；已有correctGaussian是否被先验拉坏）；
- 动作效用（执行更新前后局部几何误差/全局mesh质量，GT仅作批准诊断）；
- 不判定比例和区域。丢弃最难区域不等于找到更多错误。

“预测错误”“预测修复需求”“预测执行后改善”不是同一标签。若 B/gradient-affinity只预测 proxy improvement，须再检验与GT geometry对应；若失败，不能把代理改名可靠性绕过。

### 10.3 梯度来源与必要性的比较

必要对照：普通loss权重、PCGrad、CLEAR式固定photo主导、HarmoGS式无条件旋转、ConFIG式共同下降、证据选保护集但无有限验证、完整候选。可分阶段做，不要求一开始全开昂贵网格。

记录原始/协调/实际 optimizer方向的内积；不同相机与photo/prior冲突分开；位置/旋转/尺度分开；接受/拒绝的原因。检验更少conflict是否只是导致几何近乎冻结，以及是否损失 completeness/细节。q=0、g=0、near-antiparallel、局部秩缺失、遮挡边界和新clone缓存过期必须有定义。

打乱候选验证视角/假先验、去共享证据或把控制记录错配 Gaussian 是机制反证；若效果不受影响，声称 evidence-driven 的因果解释很弱。独立支持视角不等于统计独立，需报告同DA3/同影像的相关误差。

### 10.4 结构与拓扑单独证实

II需要 anchor-only、normal-only、关系传播、传播+uncertainty边界；固定监督数量。近/远anchor、同平面/边界、强/弱纹理、正确/系统偏prior分组。若没有anchor便凭法线“补位置”，必须拒绝该解释。

Topology后续单独做同selector/operator、仅accepted-gradient统计改变的对照；检查完整性和原语预算。不能同时改mesh extraction使F1改善，然后归因gradient fusion。不同数据集仍使用各自官方评价，统一 provenance/schema/report接口。

Tool Room 已用于冻结探针；Utility 已访问GT，不是新方法 untouched holdout。未来确认需明确未污染评价域，不用当前GT结果反复选择特征、阈值或保护来源。若支持代理效果和真实mesh改善不一致，明确负结果而非新增状态救场。

## 11. 可给导师的答复

目前不应声称已经建立了“观测充分度S→更准确错误检测N→五状态→更好优化”的完整证据链。PUP只给条件局部渲染敏感度依据，旧N与状态数没有独立科学证明，冻结结果也不支持继续包装。

拟议的新问题是：SH只描述视角颜色异常，不能告诉我们哪条几何修正更可信。我们参考近期GS表面重建的固定特征/几何假设比较，把DA3从默认正确的监督变成待验证proposal；再借CLEAR/HarmoGS与验证梯度思想，使同一支持证据既扩展候选覆盖，也约束实际几何更新。对于无法通过特征观测验证的弱纹理区域，另研究可靠位置anchor与法线关系的结构传播。这样每一步对应一个可检验对象，动作由几何约束和梯度方向产生。

贡献若成立，应写为“AmbiSuR中观测验证的几何修正选择与执行的一体化适配”，而不是“新不确定性分数/五状态命名”或“发明梯度投影”。已借方法逐一署名；跨论文适配、退化处理、实际优化器/原语约束及独立实验是可能的新增贡献。文献证明了来源与条件合理性，尚未证明这一组合有效，也不能保证论文新颖性或录用。

## 12. 核查记录与未完成事项

- 主源相关方法/实验核查22条、前轮相关证据11条、浅筛17条；具体逐篇状态在清单，不以数量冒充全文精读。
- 主文 PDF 页面目视核查：Sparse2DGS第7页表2/3；HarmoGS第9页表4（full LPIPS .125）；CLEAR第7页表5；GPU-SDF第7页表III；UA-GS第9页表4。网页表格核查 FatesGS、VA-GS、CoMe、Confidence matters、GSPrior等。
- 独立合成代数审计：Harmo原损失反例、e log(e/U)极值、预条件器内积符号、单半空间投影、Taylor一阶代理下降。它们只证明公式边界，没有训练或效果证据。
- GPU-SDF书面“KL-inspired”项 e log(e/U)（固定U>0）的导数log(e/U)+1，最小e=U/exp(1)，并非残差零；没有检查实际代码clamp/normalization，不宣称其执行实现必有错误。取稳定性检测原则，暂不复制损失。
- OracleGS全文确认是VGGT global attention而非先前猜测的MVSplat/cost-volume；且在generated chunks上得出的attention不等于真实视角几何真值，不将其“oracle”名称当保证。
- 下载不完整PDF未当全文用，公开HTML替代；无新依赖。元数据审计发现并修正草稿中DebSDF错误编号：正确2308.15536；GIP-GS与UncertainGS以直接出版社URL替代不可解析DOI跳转。没有把草稿错误作为对用户已交付结论。
- 尚未复现任何候选，未核查所有近邻源码、补充材料和全部新颖性；需要在选定具体候选后针对 nearest-neighbor 全文做贡献对照。
- 研究文档新增与根台账更新；既有批准设计与10月9日历史备忘录均保留，生产代码/数据/评价资产不变。

## 13. 用户补充 Zotero 后的本地全文核查（同一轮延续）

用户提供 C:/Users/DELL/Zotero/storage 后，按题名匹配在只读源目录找到全部22篇目录论文和DA3原论文，完整解析23个对应PDF并保存来源路径、页数、SHA-256和重复附件信息。完整提取不等于逐页阅读；新增重点方法/实验核查6篇：DualSplat、ConFixGS、GS-2DGS、VCR-GauS、DN-Splatter、PGSR。最终50条记录的状态为 M22/R11/B17；第1/12节及清单已同步更新。

本地来源还解决了此前部分下载不完整问题：Confidence matters20页、CoMe34页、GSPrior17页等本地完整PDF可读。HarmoGS本地24页与公开抓取17页长度不同，但首页明确同为2605.13073v1，正文式6–9一致；不从页数不同推断版本更新或会议录用。其他附件的页数/版本不一也不能凭Zotero文件名认证出版日期。

### 13.1 新核查到的直接近邻和证据

| 来源 | 实际机制与实验 | 如何影响推荐 |
|---|---|---|
| [ConFixGS §3.4–3.5/表4](https://arxiv.org/abs/2605.09688)；本地JWJGXFR7，28页 | 用G0的depth和opacity做投影scaffold，真实support RGB检验diffusion pseudo-target；无opacity覆盖时confidence=0；confidence共用于photometric梯度和densification。nuScenes full PSNR19.03，去confidence17.46 | 是“同一证据连接优化/拓扑”的直接2026近邻，必须引用。它验证生成appearance，依赖G0深度建立对应且默认support photo可信；本提案验证两种geometry并研究reflective情况下保护源选择。不能把一般propose-and-validate或共享confidence声称新贡献 |
| [DualSplat §3.3–3.5/表4/6](https://arxiv.org/abs/2604.21631)；本地CT2BUEAP，10页 | 首次保守重建暴露transient failure；FiT3D feature与SAM2 instance residual形成pseudo-mask，第二阶段prior先于self-consistency；base PSNR19.043，直接mask22.604，full23.421；这些是NVS指标 | 提醒在线残差可能被过拟合“抹去”，候选证据应在受控/冻结阶段生成。其high-recall masking偏好是去transient；静态反射/暗部不可据此当transient删除，double reconstruction成本也不直接搬 |
| [GS-2DGS §3.3–3.5/表4](https://arxiv.org/abs/2506.13110)；本地CQ9ML5ZW，11页 | foundation depth+normal监督，随后PBR与deferred shading。2DGS C-L1 .0481/PSNR26.23；加geometry .0084/25.52；加PBR .0074/25.86；full .0068/26.76 | 原始photo下降与geometry改善可冲突，直接支撑“不永久保护photo梯度”的适配动机。该表不能证明梯度融合会成功；表示能力分离也可以是收益主因。原文“单目模型不受反射影响”不能视为一般理论保证 |
| [VCR-GauS §3.3/表4](https://arxiv.org/abs/2406.05774)；本地MPPLZDRW，26页 | D-Normal由depth反投影邻域差分得到，影响position/normal；confidence用rendered/predicted normal cosine。TnT F1 full.40，去D-Normal.30，去confidence.36 | AmbiSuR使用rendered depth normal，不能声称它的ALR仅有rotation梯度或首次实现normal-to-position。其confidence依赖当前模型normal，也会把正确但与错误模型不一致的prior降权；验证替代geometry仍有研究动机 |
| [DN-Splatter §4.1–4.2/表4](https://arxiv.org/abs/2403.17822)；本地9SW55VJH，11页 | monocular depth以sparse SfM做scale/shift对齐，edge-aware log depth loss；显式primitive normal与normal smoothness。单VR room sensor数据F1：depth-only86.6，depth+normal87.7，再加smooth87.1 | 可借可靠SfM约束候选对齐，避免与当前wrong geometry强行对齐；边缘权重及平滑不必然更好。该单sensor场景结果不能外推RGB-only反射重建 |
| [PGSR §IV-B/表IV](https://arxiv.org/abs/2406.06521)；本地FJ3EB3TL，20页 | local plane/normal一致，multi-view homography+NCC及回投影遮挡。TnT F1 full.52；去multi-view.32；去occlusion estimation.28 | 再次证明“有效支持”要处理遮挡，而不是只数view；但模型深度遮挡判据也可能屏蔽大修正，原文推断被错屏蔽点最终都会收敛不能当作已证明保证 |

Ref-GS的deferred attribute blending与directional factorization、SurfaceSplat的SDF/GS双向互补也做了有限方法定位，尚不升格为完整方法+实验核查。其他目录论文有完整本地文本可供后续定向阅读，不能因已提取就称已精读。

### 13.2 更新后的贡献边界

第一研究候选应进一步收紧：**利用对当前/候选几何的对比证据，选择具体修正；在反射光度与可能错误的几何先验之间，对原语几何的实际更新方向做验证和受约束融合。**

| 近邻已有能力 | 拟议新增适配；均未验证 |
|---|---|
| Sparse2DGS：Gaussian vs rendered surface比较、feature/primitive geometry优化 | DA3/结构候选与current geometry对比，独立支持视角、共同支持域、不判定机制；与official双端A遗漏覆盖相连 |
| ConFixGS/CLEAR：支持观测验证generated appearance，同confidence控制梯度/增密 | 验证geometry hypothesis而非只验证appearance；photo anchor可能误导几何，保护来源由几何证据确定；直接检验实际optimizer方向 |
| HarmoGS：view-specific photo协调 | 分开within-source/cross-source，几何position/rotation局部坐标；不把modified-gradient orthogonality当原始loss common descent |
| VCR-GauS/AmbiSuR：depth-normal coupling和confidence | 不重复法线作用于position；结构版本的新增是有位置anchor的关系约束，减少对可能有偏absolute prior depth的依赖 |
| 2026 depth propagation：current-render depth anchors + normal propagation | 独立支持anchor审查、传播uncertainty/退化边界，并与同一个实际更新保护机制连接；公式本身已有 |

加入ConFixGS后，“统一证据驱动”这几个字不足以支撑创新。需要上述问题/输入证据/保护条件/执行粒度的实质适配以及对近邻的直接对照。如果最后只是固定feature loss+PCGrad，更诚实的结论是常规组合，研究贡献仍未建立。

完整来源清单与本地附件manifest可追溯；本地PDF只读，未修改Zotero数据库、附件、模型或评价数据。本节提升文献支撑和否定边界，不把文献结果当本项目收益。

