# 候选模块的目标构造与动作判别：文献和源码调研

日期：2026-10-29。状态：研究建议，尚未批准为实现规格。主仓库起点c63748fc7a5e5cd386599f98a9cce5b896118a16；本轮不改训练、selector、renderer、评价协议，不训练、不重新访问GT。

## 结论先行

建议优先研究**用有观测支持的位置作锚点、借先验法线生成位置候选**，替代当前“原始DA3绝对深度就是重定位目标”的做法。第二优先级是**固定法线后判别位置变化**，再视需要引入匹配代价曲线；这两步分别处理目标质量与判别混杂，不能靠再乘一个置信分数同时解决。

最贴近的可移植生产代码是DP-GS，几何依据来自法线积分与MP-SfM；APDe-MVS为NCC歧义判别提供更直接的实现参照。它们支持研究路线，不证明移植到406视角、已有DA3监督的AmbiSuR一定有效。第一次实验应只改变一个环节，不能把下文所有建议一次装进训练。

## 1. 已有证据限定了什么问题

真实静态检查不是最终mesh收益的因果分解。12扩展视角接受候选最近表面改善比例为7K37.70%、C0终态31.64%、V2终态40.35%、early终态32.54%；候选减原位置的平均误差分别+7.963/+20.861/+12.104/+20.178mm。NCC接受组比拒绝可比较组有6.52—10.11个百分点的描述性富集，但两组不是随机对照。

early法线有效共同域36474条中，只有法线改善36.22%，两者均改善22.53%。这说明联合候选胜出不能同时证明位置动作和法线动作有益；不等于法线绝对准确。候选轴法线不是ALR实际D-normal，GT first-hit所属面也不保证就是原Gaussian所属面。完整数据见[静态结果](2026-10-29-static-candidate-results.md)。

当前代码事实：`utils/candidate_selector.py:225`采样DA3深度，228行`points_p = rays * depth_p[:, None]`；同时换成先验法线。68行的判别取两个共同有效支持视角的各自最小NCC，先验较低且小于0.5即接受。`train.py:93`起的真实调用将E并入ALR标签，并在重定位事件把E中心替换到先验位置。

因此我们至少有三个不同问题：候选绝对位置偏差；联合比较不能区分位置和法线贡献；低匹配代价仍可能歧义。预算或学习率可以改变干预量，却不会直接使错误目标变正确。

## 2. 检索与阅读范围

本轮筛选24项2025—2026相关工作，兼顾GS、SfM、MVS、法线积分。使用CVF/作者原文/arXiv/出版社信息和本地Zotero；当前工具没有可调用的Scopus等学术数据库接口，不能声称检索了那些数据库，也不声称穷尽2025—2026论文。15份本地PDF作文本提取，提取不等于逐页精读；下表分开标注阅读程度。

实际下载12个官方仓库，查看18个非README源文件的关键实现，固定commit和文件SHA。仓库、文本与`source_read_manifest.json`放在本地output的`analysis/candidate_literature_20261029_v1`，不进入训练包。没有运行这些外部训练程序或编译它们的CUDA扩展。APDe出版社全文访问受限，依据为出版社摘要/条目与实际CUDA源码；Variational Graph-based Normal Integration的OpenReview遇到验证页，未纳入核心证据。

| 工作与原始来源 | 发表身份 | 本轮实际核对程度 | 对当前问题的用途/取舍 |
|---|---|---|---|
| [DP-GS](https://arxiv.org/html/2607.03765v1) | 作者页标注ECCV2026 | 方法、消融、实际传播和训练代码 | 首选候选目标来源；原文是传播深度损失，非硬重定位 |
| [MP-SfM](https://openaccess.thecvf.com/content/CVPR2025/html/Pataki_MP-SfM_Monocular_Surface_Priors_for_Robust_Structure-from-Motion_CVPR_2025_paper.html) | CVPR2025 | 方法/补充推导、实际积分求解代码 | 3D位置锚点与法线结构分工；不是换一个置信乘法 |
| [Discontinuity-aware Normal Integration](https://arxiv.org/html/2507.06075v1) | ICCV2025 | 方法/推导、官方求解代码 | 局部平面、射线和断层的理论依据 |
| [Continuous Components Normal Integration](https://openaccess.thecvf.com/content/WACV2026/html/Milano_Towards_Fast_and_Scalable_Normal_Integration_using_Continuous_Components_WACV_2026_paper.html) | WACV2026 | 方法、实际分量表示/CG更新 | 连续块相对尺度，后续加速/跨大区域时才考虑 |
| [APDe-MVS](https://www.sciencedirect.com/science/article/pii/S0031320325012567) | Pattern Recognition，2025在线/172卷2026 | 出版社摘要/信息、DepthToWeak实际CUDA | 代价曲线而非单次NCC胜出；不照搬整套MVS |
| [MAtCha](https://openaccess.thecvf.com/content/CVPR2025/html/Guedon_MAtCha_Gaussians_Atlas_of_Charts_for_High-Quality_Geometry_and_Photorealism_CVPR_2025_paper.html) | CVPR2025 | 结构保持方法、现有仓库深度排序函数及真实调用 | 相对结构监督备选；公开代码新增排序项与原文核心要区分 |
| [Sparse2DGS](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Sparse2DGS_Geometry-Prioritized_Gaussian_Splatting_for_Surface_Reconstruction_from_Sparse_Views_CVPR_2025_paper.html) | CVPR2025 | 本地原文、实际SGU位置更新 | 已有借鉴；目标来自模型表面，与我们原始DA3目标不同 |
| [FatesGS](https://arxiv.org/abs/2501.04628) | AAAI2025 | 原文、深度排序/特征损失实际代码 | Vis-MVSNet特征可作后续匹配消融，投影本身不新颖 |
| [Stereo Anywhere](https://stereoanywhere.github.io/) | CVPR2025 | 方法、实际双分支代价体/尺度对齐代码 | 单目/立体两类证据可互补，但完整学习系统移植较重 |
| [GSPrior](https://arxiv.org/html/2603.19682v1) | CVPR2026 | 方法、训练/TSDF更新代码 | 自身多视角TSDF作约束；第一版不新增TSDF生命周期 |
| [IndoorGS](https://openaccess.thecvf.com/content/CVPR2025/papers/Ruan_IndoorGS_Geometric_Cues_Guided_Gaussian_Splatting_for_Indoor_Scene_Reconstruction_CVPR_2025_paper.pdf) | CVPR2025 | 本地方法与算法；未定位官方实现 | 线/SfM约束位置与平面区域，不是大小阈值排列组合；工程较重 |
| [Confidence Matters](https://arxiv.org/html/2608.06117v1) | 2026预印本，未确认正式venue | 本地原文、实际加权训练代码 | 置信调制不能替代有益性检验，非第一修复 |
| [In Depth We Trust](https://arxiv.org/html/2604.05715v1) | CVPR2026 3DMV workshop | 原文方法；未定位官方代码 | 不一致区域局部绝对约束+相对深度变化；GAL指空间导数 |
| [SwiftNDC](https://arxiv.org/html/2602.22565v1) | 2026预印本，CVPR Findings活动列表 | 方法；未定位官方代码 | SfM校正先验深度，可作第二种目标生成器，需重实现/依赖适配 |
| [PAGaS](https://arxiv.org/html/2604.22129v1) | CVPR2026 3DMV workshop | 方法、实际逐像素depth参数构造 | 控制可优化自由度有启发；临时像素Gaussian不同于本项目持久Gaussian |
| [PMNI](https://openaccess.thecvf.com/content/CVPR2025/papers/Pei_PMNI_Pose-free_Multi-view_Normal_Integration_for_Reflective_and_Textureless_Surface_CVPR_2025_paper.pdf) | CVPR2025 | 原文摘要/方法范围、官方仓库勘查，未读核心源码 | 反射/弱纹理法线积分；联合位姿/SDF整套较重 |
| [GS-2DGS](https://arxiv.org/abs/2506.13110) | CVPR2025 | 本地方法、官方补充材料，未读核心源码 | 反射场景先验/PBR；不直接解决我们目标准入 |
| [DepthSplat](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_DepthSplat_Connecting_Gaussian_Splatting_and_Depth_CVPR_2025_paper.html) | CVPR2025 | 原始条目/摘要筛选 | 学习式多视角深度，不是当前最小插入模块 |
| [GSRecon](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_GSRecon_Efficient_Generalizable_Gaussian_Splatting_for_Surface_Reconstruction_from_Sparse_ICCV_2025_paper.html) | ICCV2025 | 原始条目/摘要筛选 | 几何/特征聚合需整套训练表示，暂后排 |
| [QuickSplat](https://arxiv.org/abs/2505.05591) | ICCV2025 | 原始摘要/方法范围筛选 | 学习式初始化/增密，改变当前对照太多 |
| [Modeling Depth Ambiguity](https://arxiv.org/html/2606.02552v1) | 2026预印本 | 原始方法范围筛选 | 边界可能多深度模式；不把两个深度加权平均当真值 |
| [CLEAR](https://arxiv.org/html/2608.02206v1) | 2026预印本 | 本地原文；未定位官方实现 | 梯度协调不先行：当前目标本身尚不够好 |
| [HarmoGS](https://arxiv.org/html/2605.13073v1) | 2026预印本 | 本地原文；未定位官方实现 | 同上；不能靠协调梯度证明目标正确 |
| [Gaussian Wrapping](https://arxiv.org/abs/2604.07337) | ECCV2026 | 原始摘要/方法范围筛选 | 表示、法线场、renderer与mesh抽取均改变，暂不移植 |

## 3. 优先路线：重建候选位置的来源

### 3.1 为什么这与我们的实测问题直接对应

我们不应把“DA3某处深度有偏”扩展成“不能用DA3”。可以有选择地使用它给出的局部结构。DP-GS从有跨视角支持的模型深度出发，借外部法线传播；MP-SfM把多视角3D点、原深度先验和法线积分分别放入目标，而非假定同一先验的所有属性同样可信。它们与本轮位置/法线质量不同的发现相符。

先验法线通常约束局部表面如何变化，可信位置锚点确定它在哪里。以参考相机的z-depth记号，r_i=K^-1[u_i,v_i,1]^T，X_i=d_i r_i。同一局部平面、相同相机坐标的法线n满足

\[
n^T(d_i r_i-d_j r_j)=0,
\qquad
d_i=\frac{n^T r_j}{n^T r_i}d_j.
\]

这是射线—平面关系；DP-GS/IronDepth传播与ICCV2025法线积分提供直接来源，不是从位置正确性凭空推出的分数。要求同一连续局部表面、正深度、非掠射且分母远离0；不能跨墙/椅边界。仅有法线时，将所有d同乘常数仍满足关系，因此需要位置/尺度锚点。前向平面n=(0,0,1)且射线z分量为1时，传播就是邻点共享锚点深度：修正局部起伏，却不能凭空修正全区域共有的平行偏移。

这个公式保证的是满足相应平面关系，不保证更接近真实表面。锚点错、法线错、跨边界或多路径累积都会传错。DP-GS自己也讨论远离可靠区域时传播精度下降，不能把它宣传成暗部通解。

### 3.2 真正可借的DP-GS代码

固定[train.py](https://github.com/hanl2010/DP-GS/blob/dc1ce835ea3c6bb7887333c50df7a13497a45c09/train.py#L226)226—335：几何往返误差权重×逐像素RGB-NCC置信，阈值形成起始锚点；用先验法线局部传播，最后约束渲染深度。数组(HW,3)的3是RGB通道，不是三个支持视角；未采样像素置信初始化0。默认5×5邻域、10次传播、阈值0.6，这些是作者配置，不能直接宣称适合Tool Room。

[get_depth_candidate_weights/get_pos](https://github.com/hanl2010/DP-GS/blob/dc1ce835ea3c6bb7887333c50df7a13497a45c09/utils/graphics_utils.py#L212)注明来自IronDepth，生产实现采用arctan归一化坐标和像素+0.5，并有近零/非有限处理；它不与上面精确K^-1射线公式逐位相同。适配时要明确沿用哪种坐标约定并验证几何，不能复制后假称与现有ray接口天然一致。原代码反复传播平均，并非每轮严格冻结初始锚点。

DP-GS表4在三张小重叠视角DTU上，已有法线+opacity decay时，增加传播CD由1.55到1.13；再加边缘平滑到1.02。这是作者模块消融，不是把全部2.79→1.02都算给传播，更不是对本项目可预期的27%收益。

### 3.3 锚点与目标如何适配到AmbiSuR

建议第一版仅讨论下列来源中的一个：优先沿DP-GS，使用现有PGSR几何往返+patch匹配支持的**当前渲染深度**作起始锚点；DA3深度继续提供原有先验法线，不把它整张绝对深度当重定位真值。若实际锚点覆盖不足或明显自洽错误，备选才是MP-SfM式原始SfM轨迹点锚定。

原始SfM三角化点须从原始sparse及观测轨迹读取；`sparse_da3_aligned`里的DA3稠密初始化点不能改名成额外三角化证据。模型锚点/SfM/DA3都复用照片，不称统计独立。DA3深度置信也不是MP-SfM所用的深度方差或法线协方差，禁止直接冒充精度矩阵。

推荐先尝试将传播结果作为**替代位置候选生成器**：已有E比较和重定位接口处理得到的target。这个连接比DP-GS原始深度损失更贴近当前模块，但必须如实标为本项目适配假设：像素表面深度转成Gaussian中心目标，并不能由原论文的loss消融直接背书。若希望首先忠实验证DP-GS，则应只移植其传播深度损失、关闭新增重定位；这是另一个实验，不与前者一次组合。

## 4. 第二路线：让证据与动作对应，再处理NCC歧义

### 4.1 当前联合比较有一个可明确写出的混杂

令C(d,n)为同一参考patch在相同共同有效支持上的NCC代价；d表示参考中心的z-depth。当前在比较

\[
C(d_P,n_P)<C(d_G,n_G).
\]

两边同时改位置与法线，因此胜出可能主要来自法线改善。C是投影几何的复合函数，不具有“总差下降⇒任一坐标方向均改善”的性质。更明确地，有完全不依赖近似的代数恒等式：

\[
C(d_G,n_G)-C(d_P,n_P)
=\underbrace{C(d_G,n_G)-C(d_G,n_P)}_{\text{固定中心的法线变化}}
+\underbrace{C(d_G,n_P)-C(d_P,n_P)}_{\text{固定法线的位置变化}}.
\]

例如这三个代价依次为0.20、0.05、0.10，当前联合规则仍接受0.10的候选，但在同一先验法线下，从原位置移动到先验位置反而使代价增加0.05。这个反例解释的是代理目标中的动作混杂，没有假定NCC就是真实几何误差。用MP-SfM的分块优化作方法论参照、APDe固定法线扫描深度的源码作实现参照，可以提出本项目的**条件比较**：

\[
\Delta_{pos}=C(d_G,n_P)-C(d_*,n_P),
\qquad
\Delta_{normal}=C(d_G,n_G)-C(d_G,n_P).
\]

第一式在同一法线下比较当前位置与拟移动位置，第二式固定中心，只改变平面朝向。无需新renderer：现有`_oriented_plane`与`_source_cost`已可构造经过同一中心、朝向不同的平面，再用现有单应性warp。这也回答“固定位置后是否还能NCC比较法线”：可以，平面单应性依赖法线和点到平面距离，固定中心不使整块patch变换固定。

这两式是本项目待验证的适配定义，不是声称某篇论文已验证了相同的AmbiSuR规则。比较改善的仍是匹配代理；不能据此保证XYZ或ALR最终有效。特别是ALR监督的是渲染深度导出的法线，不是直接把n_G换成n_P；因此第二式也不是ALR更新效用的严格检验。

若实施，优先把条件比较只用于**额外位置动作的准入**，ALR仍走现有A∪E接口；不要又用同一个联合胜出标签自动授权所有动作。是否同时改变E的产生方式，需要单独批准，不能悄悄修改。

### 4.2 APDe-MVS给出了比单个NCC值更具体的依据

[DepthToWeak实际CUDA](https://github.com/whoiszzj/APDe-MVS/blob/7c7ce3dda39cf11dbcca11546dabeb95e387c815/APD.cu#L2103)固定当前平面法线，围绕当前深度按视差取61个假设，用选定支持视角权重汇总NCC，可加几何代价；再分析局部极小值数量、当前深度靠近哪个极小值、最小代价及替代极小值分离情况。它不是给低NCC额外乘一个任意置信值。

可借的机制是“比较候选周围的代价形状，识别平坦/多峰歧义”，不是立刻复制61个采样、全部门槛、patch变形、SAM、CUDA求解器。采样数和深度域应在适配规格中明确；缩小曲线是简化重实现，不叫原模块无改动移植。多假设应使用可比较的有效支持，不能因为部分深度出界、剩余支持变少就制造假低谷。尖锐错误反射匹配仍可能发生，所以曲线信息增加而非真值保证。

当前`min_s C_s(P)<min_s C_s(G)`的两个极小值甚至可能来自不同支持照片；这是继承SGU式最佳视角选择的机制特性，不先宣布它是bug。第一版不同时换成所有视角平均/多数票/特征距离，否则无法分清是哪项改变有效。

实际`utils/loss_utils.py:106`的lncc采用`cross*cross/(ref_var*nea_var+eps)`，也就是近似1−rho²的代价。忽略数值eps时，对非零方差patch，只要另一patch=a×本patch+b、a非零，代价就可为0，甚至不区分相关符号；重复纹理的错误深度也能满足相同关系。因此低NCC本身不可能唯一确定真实深度。这是由现有测量定义得到的性质，不是声称当前损伤已查明来自负相关；不据此擅自换signed NCC。

## 5. 为什么暂不优先上梯度融合、特征网络或更复杂表示

FatesGS的实际`get_feat_loss_corr`使用Vis-MVSNet预训练特征、跨视角采样、余弦相关与可见性；不是DINO。它支持“特征是另一种匹配证据”的实验候选，不证明比当前NCC更能裁决DA3位置，尤其高光和重复结构仍可能错。Stereo Anywhere通过双分支相关体、置信与学习式更新融合证据，说明问题可被处理，却不是换一行相似度公式即可迁移的模块。先复用现有NCC，减少同时更换目标和测量器。

MAtCha公开仓库的`compute_depth_order_loss`及真实训练调用提供相对排序监督；原论文更核心的是charts、法线/曲率保持与对齐。若无可用锚点，借排序而非错误绝对位置是备选，但排序不能定位平行平面的绝对偏移，额外法线损失又可能重复已有ALR，不优先叠加。

GSPrior用自身深度融合TSDF并分阶段约束Gaussian；源码还含作者本地数据路径，不能整段拿来运行。PAGaS的逐像素临时球形Gaussian深度自由度、Gaussian Wrapping的表示/renderer/extraction，以及IndoorGS的线几何和不同ADC均超出本轮最小替换。它们有科学价值，但不是用更多模块掩盖目标误差。

CLEAR/HarmoGS或PCGrad能处理更新间冲突，但目标方向若本身错，协调不会自动产生正确目标。“In Depth We Trust”的GAL则是图像空间深度导数匹配，不是参数梯度冲突融合；不混同两种梯度。

## 6. 对真实代码的修改边界：尚未执行

| 环节 | 当前真实接口 | 建议变动/保留 |
|---|---|---|
| A及原ALR | `compute_sh_ambiguity_labels`、标签渲染、原先验损失 | 保留原A公式与A通道；E仍补原~A池，不能声称改进了A公式 |
| 候选目标输入 | `train.py`准备`prior_depth/prior_normal` | 位置候选另供`proposal_depth`，先验法线/原ALR数据独立命名；不可直接覆盖原`prior_depth` |
| 位置目标 | `_select_candidate_geometry`的DA3采样/`points_p` | 只在有效传播域采样新target，复用原ray/世界坐标/ID顺序 |
| 对应和NCC | `_source_cost/_plane_homography` | 复用PGSR继承接口；条件比较/曲线如获批才新增，非另写一套投影 |
| 位置执行 | `relocate_selected`、existing targets | 复用现有参数/优化器状态处理；生成的表面target适不适于中心仍需实验 |
| 原ALR目标 | `selector_data['prior_depth']`及`prior_normal` | 当前下游还会读取；分开位置目标避免意外把法线监督也改了 |
| 离线检查/正式评价 | 既有capture/score、mesh evaluator | 同一入口补一种目标来源薄适配；不创建新可靠性框架，不改GT/单位/评价域 |

本轮所谓“站在基线上”是保留A的已有作用、投影/标签/损失与生命周期接口，扩展未命中区域的可修正候选；不是把A强行乘S。观测约束弱并不能单独证明几何错或DA3好，因此不回到N=1-S(1-A)的论证。

## 7. 建议实施顺序与可以下的结论

1. 用户先选择候选目标生成路线或更忠实的DP-GS传播loss路线。推荐前者，作为现有early的单一target-source消融；先不连同APDe/特征/梯度/学习率一起变化。规格需要锁定锚点规则、normal/ray坐标、传播域和位置执行，不能仅凭这份报告宣称已批准。
2. 复用现有静态入口，在相同7K/视角/样本预算上看新target是否比旧DA3target减少位置损伤；没有新的GT训练信息或大规模诊断体系。检查不是最终收益证明，也不允许据GT逐区域调门槛。若目标本身没有改善，不值得马上消耗一轮30K。
3. 若目标可用，直接进行已讨论的30K续训与固定mesh/渲染评价，复用基线输出。条件位置比较作为单独后续消融；曲线只在NCC歧义仍是主要限制时追加，不把完整候选系统作为首个实验。
4. 成功需要最终Accuracy/Completeness/CD、1/2/5cm P/R/F1和渲染质量共同判断，而非接受数更多或静态更近比例单项。重定位截断/预算等先保持既有early设置，是否采用不同设置须先讨论；单场景后再Utility及多seed，不能凭文献收益替本项目宣布完成。

当前最有依据的研究假设是：**保留多视角支持的位置尺度，借先验补局部结构；选择时区分结构改变与位置移动，再决定哪种动作被允许。** 这是由本项目静态负结果和现成方法共同提出的可检验方向，没有“必然明显收益”或理论上限结论。

## 8. 源码证据索引

下列均读取非README实际实现；更多固定路径/行段/SHA见外部`source_read_manifest.json`（12仓库18文件）。没有代码执行成功声明。

- DP-GS `dc1ce835`：train210—335、graphics_utils212—254，置信图/传播/深度loss。
- MP-SfM `fa496408`：image/integration136—208、233—285、380—488，深度/法线/稀疏位置项与真实CG求解。
- ICCV法线积分 `f68e47cf`：main29—170、379—459，generic ray约束与不连续项；bini文件为其BiNI基线，不能当新方法主体。
- WACV法线积分 `0c157564`：main932—1005、ComponentDecomposition1—200，分量log尺度更新；仓库依赖其wacv2026分支submodule，本轮未安装/运行。
- APDe-MVS `7c7ce3dd`：APD.cu2103—2250，固定法线深度曲线及可靠性分类。
- MAtCha `b119fd96`：depth142—223与train_with_charts真实调用，排序损失及调度。
- Sparse2DGS `af15587e`：gaussian_model476—501，xyz替换、rotation/scale注释。
- FatesGS `a721d318`：feat_utils259—389与train75—111，Vis-MVSNet特征与排序/相关。
- Stereo Anywhere `c4aa114b`：stereoanywhere107—197，双分支相关/置信/对齐。
- Confidence Matters `ef3e0715`：train230—266，已有模型深度对齐与conf幂加权。
- GSPrior `dcb7c89f`：train148—157/198—220及gaussian_model的TSDF函数上下文（后者只作关键词/上下文检查）。
- PAGaS `22ff538b`：splat_ops1—145，depth/inverse/logdepth参数化，非本项目高斯池更新。

独立窄范围审查：未发现Critical/Important。核对DP-GS动作/坐标/表4、MP-SfM约束/精度、APDe曲线、主仓库ALR耦合、精确分解与NCC不唯一性、关键既有静态数字；12仓库18文件SHA全部匹配。审查未全面重读24篇、重算GT或运行外部训练。
