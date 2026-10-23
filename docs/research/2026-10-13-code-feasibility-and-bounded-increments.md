# AmbiSuR 增量改进：论文依据、代码接口与回退边界

日期：2026-10-13。性质：研究答复与源码可行性审计，不是已批准设计的修订，不授权实现、训练、GT 访问或恢复已有实验。

2026-10-15 用户纠偏后的当前路线见 [复用基线的研究综合稿](2026-10-15-source-faithful-research-route.md)：首轮仅以 NCC 候选比较扩展 Gaussian 标签并复用 ALR；取消 exclusive ownership、固定特征、逐视角准入与两步 optimizer preview 的前置地位，保留 donor best-source 等选择先测试。传播和 VCR 独立适配。下文保留为当时的审计/候选历史，不将其全部视为当前必要模块。

## 1. 先纠正结论，再讨论移植

本轮结论是：用户对 ALR、冲突来源、数学漏项和实施顺序的质疑成立。AmbiSuR 的 ALR 已由 PGSR 平面深度反传到位置与方向；固定特征梯度只能检验特征证据；候选几何代价低不证明梯度准入模块必需；保护集与 QP 应后置。本轮查验了公开源码，而不再只根据论文标题判断“可移植”。

近期候选中，VCR 的先验权重、DP-GS 的传播损失、FatesGS/VA-GS 的特征采样有可辨认的接口。Sparse2DGS 的整个更新函数不能直接搬。HarmoGS/CLEAR 暂未找到可核查的官方实现，只能作为论文指导的重实现候选。

建议把研究拆成可单独检验的增量：**SH 引导的先验监督扩展；有证据需求后再加新增更新准入；另测法线引导的锚定深度传播。** 不预设三个都必须出现，不继续维护未经证明的新 N 或恰好五个状态。本文件不会覆盖现有批准规格；若之后实施，需将选中的增量写成单独规格。

## 2. 源码审计范围与可追溯性

实际集成目标是本地 `62e9117dfb2e20a06b257d166d8a6636d37845cc`；用户公开 fork main 本轮解析为 `d6f15c8891a53800d5e3100f95817a7dd7f98e2f`，二者不能视作同一状态。下表是第三方审计固定版本，非永久最新版本。

| 来源 | 固定 commit | 实际核查接口 | 判断 |
| --- | --- | --- | --- |
| [AmbiSuR 官方](https://github.com/Fictionarry/AmbiSuR/tree/88d64054f53e0eba9ce49198282edf4a67fc8ca8) | `88d64054` | renderer、ALR、PGSR round-trip/NCC、CUDA forward/backward | 本地继承的真实基础 |
| [VCR-GauS](https://github.com/HLinChen/VCR-GauS/tree/aa715d19bfacfa9d491f477c572eab1839dcee3e) | `aa715d19` | trainer、cos_weight、depth-normal；子模块 `9ee293dc` 的 ray-plane depth | 权重可借，深度渲染不等同 PGSR |
| [Sparse2DGS](https://github.com/Wuuu3511/Sparse2DGS/tree/af15587eb186ca7ab12a2cf795775fa0f473f25d) | `af15587e` | compute_hom、update_points、特征渲染 | 借比较逻辑；整套 SGU/特征 CUDA 不直接移植 |
| [FatesGS](https://github.com/yulunwu0108/FatesGS/tree/a721d3181ace57f79e3e65db4a45b7501a21798e) | `a721d318` | FeatExt、get_feat_loss、grid_sample | 固定特征损失可适配 |
| [VA-GS](https://github.com/LeoQLi/VA-GS/tree/171125583c30bb7a6748784c49abc3431e27f6d0) | `17112558` | geometric_consistency、feature loss、相机选择 | 可见性与特征接口可借；不是现成多视角梯度协调 |
| [DP-GS](https://github.com/hanl2010/DP-GS/tree/dc1ce835ea3c6bb7887333c50df7a13497a45c09) | `dc1ce835` | normal_guided_depth_loss、get_depth_candidate_weights、get_pos | PGSR 分支上的损失级适配候选 |
| [TriaGS](https://github.com/cogniboticslab/triags/tree/179d146bacb5c30fd50ff190d55c0a0e6d5c770f) | `179d146b` | compute_tgpc_loss | 循环构造证据疑问，暂不推荐移植 |

GitHub API 本轮 403；改用公共 Git 与 commit 绑定的 raw 文件。文件 SHA256、来源 URL、审计版本保存在配套 `2026-10-13-code-source-manifest.json`。只读取/提取公开代码到系统临时目录，未执行第三方 setup 或训练代码，没有下载权重或数据。静态可行性不等于 CUDA 编译通过、显存满足或已有指标收益。

[HarmoGS 原文](https://arxiv.org/html/2605.13073v1) 的复现清单写代码将在发表后开放；[CLEAR 作者页](https://qz1-boy.github.io/) 当前该条目提供论文链接，未提供可核查代码链接。这里的结论是“本轮未找到可核查官方实现”，不是断言网络上任何实现都不存在。

上一轮清单含 22 个截图标题与 28 个补充记录，方法/相关实验本轮核查 M22、此前证据复用 R11、摘要/元数据筛选 B17；不宣称全文精读了 50 篇，也不宣称穷尽所有 2025–2026 论文。

## 3. ALR 和 VCR：位置耦合已经存在，新增价值要另找

### 3.1 用户对 PGSR 深度的记忆正确

本地 `gaussian_renderer/__init__.py:165–174` 先计算相机坐标法线、位置和到局部平面的距离：

\[
\delta_i=|n_i^\top\mu_i|,\qquad
\bar n(p)=\sum_i w_i(p)n_i,\quad
\bar\delta(p)=\sum_i w_i(p)\delta_i,
\quad w_i=\alpha_iT_i.
\]

在相机法线朝向与符号约定下，CUDA `forward.cu:449` 给出：

\[
D_{\rm PGSR}(p)=-\frac{\bar\delta(p)}{\bar n(p)^\top r_p+10^{-8}}.
\]

`backward.cu:491–495` 显式计算距离分子和法线分母的导数。Python 再对这个深度反投影点的邻域作叉积，得到 `depth_normal`；`train.py:474–475` 用 DA3 深度导出的法线监督它。因此位置、旋转以及混合权重都在可达反传路径上。[PGSR 原文](https://arxiv.org/abs/2406.06521)、[AmbiSuR renderer](https://github.com/Fictionarry/AmbiSuR/blob/88d64054f53e0eba9ce49198282edf4a67fc8ca8/gaussian_renderer/__init__.py)。

这里“无偏深度”是论文的平面深度名称，不能扩展为在任意遮挡/透明混合下统计无偏，或等价真实表面。应撤回“ALR 只能改方向”及“VCR 才首次让法线监督位置”的表述。

### 3.2 VCR 与 PGSR 并非同一个混合公式

VCR 固定子模块 `9ee293dc` 的 `auxiliary.h:262` 在传入射线 dirs 的分支先求每个高斯的射线—平面交点。若接受的贡献均通过 inside 选择，其深度为：

\[
d_i=r_z\frac{n_i^\top\mu_i}{n_i^\top r};
\qquad D_{\rm VCR}=\frac{\sum_iw_id_i}{\sum_iw_i}.
\]

源码还提供未传 dirs 时使用中心深度的分支，以及分子 inside 选择；完整一般式须将分子写为 `sum_i w_i s_i d_i`、分母仍为 `sum_i w_i`。上述 ray-plane 分支中，PGSR 是混合平面属性后取商，VCR 是各自求交后混合。一般不相等，二者均依赖位置与方向。不能为了增加已经存在的耦合替换整个 rasterizer。[VCR 固定 CUDA 源码](https://github.com/HLinChen/diff-gaussian-rasterization/blob/9ee293dcae4fe112b5300ed690203f90fe2e026a/cuda_rasterizer/forward.cu#L374)。

### 3.3 可以先借 VCR 的置信度，不必借整个深度实现

`VCR-GauS/trainer.py:266–279` 用 **detach 的直接渲染法线** 与先验法线计算权重，去加权 **深度导出法线** 的监督；`tools/loss_utils.py:135` 的实际式子是：

\[
w(p)=\exp\!\left(\frac{\hat n_{\rm direct}(p)^\top\hat n_P(p)-1}{\tau}\right),\quad\tau>0,
\quad w\text{ detach}.
\]

源码 τ<=0 时返回全一权重。可在现有 ALR 中对先验项增加这个权重，不改 PGSR 深度。必须将单位法线变换到同一坐标系并统一符号：现有直接渲染法线在相机坐标，ALR 的 depth-normal 路径使用世界坐标，直接点积不成立。[VCR 训练实现](https://github.com/HLinChen/VCR-GauS/blob/aa715d19bfacfa9d491f477c572eab1839dcee3e/trainer.py#L266)、[权重实现](https://github.com/HLinChen/VCR-GauS/blob/aa715d19bfacfa9d491f477c572eab1839dcee3e/tools/loss_utils.py#L135)。

它是低成本的迁移基准，不自动构成主创新。该权重度量当前模型与先验的方向一致；模型错误时反而可能压低正确先验，不能解释成先验真实概率。也不能解决“两个平行但位置不同的平面”这一问题。

### 3.4 位置可反传，不等于位置可辨识

理想单平面满足 `n^T X=d`。对射线 `r_p`，其点是：

\[
X_p(d)=\frac{d}{n^\top r_p}r_p.
\]

改变 d 会改变深度和位置，但每个这样的平面导出的单位法线仍是 n。纯法线损失沿平面的整体法向偏移可能不变。这个反例直接说明：**有位置导数路径，不代表所有位置自由度都被法线约束辨识。** 本轮解析例与数值例验证了这个事实；它不是对真实混合场景一切方向的断言。

AmbiSuR 已有 DA3 深度 L1、光度和多视角约束，不应说整个基线没有位置监督。DP-GS 的意义需体现为从有支持的深度锚点传递位置关系，以及相对现有深度先验的互补收益。

## 4. 第一增量：让 A 真正决定搜索，再以修正证据扩展 ALR

### 4.1 原样保留基线的双端 SH 命中

记官方高/低端 SH 选择为 \(\mathcal U_A\)。这里 A 是用户所说的 SH 线索简称；不要把当前 Core 的单侧连续 A 偷换成官方双端 selector。

第一版保留原有 \(\mathcal U_A\)、ALR、损失权重和调度。A 另外作为 **搜索种子**：只在其投影足迹邻域、满足前表面与区域边界检查的局部候选中，审计未被 A 命中的高斯。用固定 patch 半径或一层邻接和固定计算预算定义 \(\mathcal B_A\)，而非把整个场景送入昂贵验证。

\[
\mathcal E\subseteq \mathcal B_A\setminus\mathcal U_A,
\qquad\mathcal U_{\rm candidate}=\mathcal U_A\cup\mathcal E.
\]

这给 A 三个具体作用：触发局部审计、限制预算、保留已命中分支。它不再被塞进未经校准的概率公式，也不以 A 硬过滤所有新增高斯。边界是：远离种子的漏检可能不能覆盖；局部错误是否聚集，是**我们的待测假设**，不是 Sparse2DGS 已证明的定理。

必须做等预算随机种子或全局采样对照，并比较直接扩大官方 SH 选择比例。若这些简单方法一样有效，应承认 A 引导并无额外价值，而不强行讲深层联系。

### 4.2 借 Sparse2DGS 的是假设比较，不是复制整个 SGU

在同一参考 patch 和相同支持像素上，对比当前渲染深度解释 G 与已有 DA3 深度解释 P。只有可比较且 P 的固定特征代价稳定较低时，才让此处成为扩展监督候选。

Sparse2DGS 的 `utils/point_utils.py::compute_hom/compute_hom_v2` 和 `scene/gaussian_model.py::update_points` 有几何假设的 NCC 比较。实际 SGU 比较的是渲染/原语相关候选，不是 DA3；三个视图结构写死，且只更新 xyz，旋转/尺度更新被注释，并替换优化器位置张量。`train.py:170–202` 的候选比较还取源视角中的最小 NCC 代价，不能将其描述为所有支持视角一致通过。我们的 DA3 候选、SH 种子和有限范围都是适配假设。不能说源代码已经包含这些新逻辑。[Sparse2DGS 实现](https://github.com/Wuuu3511/Sparse2DGS/blob/af15587eb186ca7ab12a2cf795775fa0f473f25d/scene/gaussian_model.py#L476)。

其主特征分支还有改造的 8 通道 surfel rasterizer，不能直接移入 AmbiSuR 的 3D Gaussian renderer。先借 FatesGS/VA-GS 的深度反投影与固定特征采样，避免同时更换表示。

### 4.3 C_H 的明确来源与定义

冻结的特征图 \(F_r,F_s\) 来自 FatesGS/VA-GS 的 VisMVSNet 编码器机制。源码 `FeatExt` 输出 32 通道，深度反投影到世界坐标，再投影到支持相机，`grid_sample` 后计算余弦差异。固定权重并不等于固定采样位置，因此可以对几何反传。[FatesGS 特征代码](https://github.com/yulunwu0108/FatesGS/blob/a721d3181ace57f79e3e65db4a45b7501a21798e/utils/feat_utils.py#L352)、[VA-GS 特征代码](https://github.com/LeoQLi/VA-GS/blob/171125583c30bb7a6748784c49abc3431e27f6d0/utils/feat_utils.py#L363)。

对参考像素 p，定义：

\[
X_r(p;D)=R_r^\top[D(p)K_r^{-1}\tilde p-t_r],
\quad q_s(p;D)=\pi(K_s[R_sX_r(p;D)+t_s]),
\]
\[
C_{rs}(D)=\frac{1}{\sum_p m_p}
\sum_p m_p\left[1-
\frac{F_r(p)^\top F_s(q_s(p;D))}
{\|F_r(p)\|\|F_s(q_s(p;D))\|}\right].
\]

式子采用列向量外参 `X_cam=R X_world+t`；本地行向量、像素中心、特征分辨率和 `align_corners` 必须适配，不能逐字照抄。零特征范数和空支持直接退出。

G/P 比较使用两者都有效的固定支持、同一权重、同一归一化。不能让某候选把代价大的像素移出图像后凭更少支持“胜出”。第一版可规定两个有效支持视角各自有 `C_rs(G)-C_rs(P)>δ` 才接受；δ 和预算需在实现规格中预注册。两个视角是范围/成本选择，并非正确性定理。

这只表示 P 在此特征代理上更受支持。反射、重复纹理或共用先验输入可能造成共同错误；它不是几何错误概率，也不构成统计独立验证集。差异不足、无有效支持、出界、遮挡/边界混合无法核查时，新增集合为空，沿用基线。

VA-GS 当前训练代码 `train.py:253–259` 在 feature 阶段把多个邻居缩减为最后一个，再计算 loss；本方案如需逐支持视角检查，必须分别调用代价，不能声称直接复制即可。FatesGS 的 visibility helper 还含排序后按 UV 重复计数选样本的逻辑，其正确分组需单独验证；不把此类 helper 的名称当成可靠可见性的证明。优先复用基线/VA 的 round-trip scaffold，保留其实际失效边界。

### 4.4 可见性与高斯归属必须分开

基线 `train.py:284–316` 已做深度反投影/round-trip，后续包含 patch homography、`grid_sample` 与 NCC。VA-GS `geometric_consistency` 也比较源视图深度回投影误差；可复用此类检查排除明显遮挡与出界。[VA-GS 检查实现](https://github.com/LeoQLi/VA-GS/blob/171125583c30bb7a6748784c49abc3431e27f6d0/utils/graphics_utils.py)、[AmbiSuR 训练实现](https://github.com/Fictionarry/AmbiSuR/blob/88d64054f53e0eba9ce49198282edf4a67fc8ca8/train.py)。

但是这两者都不证明“先验深度属于目标高斯”。本地 `radii>0` 只代表投影足迹可见；`out_observe` 是前景透射条件下的像素命中计数；当前 evidence pooling 是 alpha*T 加权统计，少量后景贡献归一化后也可能得到高分。

第一版只覆盖可验证的单前表面 patch。若要提出 **高斯级** 新增选择，仍需得到实际贡献归属，例如受接受 alpha*T 定义一致的 top-contributor ID、主导占比与边界检查；这些是新增工程接口，不是 VA-GS 现成证明。若只实现像素层验证，应把结论限制为“扩展 ALR 的像素监督”，不能升格为“找到错误高斯”。混合不清晰时退出，不优先开发通用遮挡推理系统。

### 4.5 先只增加监督，不强行增加融合器

已通过的额外候选可以定义 detach 的额外监督权重 W_E，沿用已有 DA3-normal 与 PGSR depth-normal 的 ALR 残差：

\[
L_E=\lambda\,\operatorname{mean}_p
W_E(p)\,[1-n_P(p)^\top n_D(p;\theta)].
\]

原 ALR 的 W_A 不改、不重复计数。要声称更新只作用于 E，必须单独产生 L_E 的几何梯度，并在新分支按高斯/参数组屏蔽；像素权重并不自动实现参数归属。第一版 E 可按当前步计算且不持久缓存，在发生参数/拓扑管理的迭代取消新增分支，避免冒充已经实现状态迁移。位置/旋转作为额外分支目标来自 AmbiSuR 的参数分离意图，不来自“这些参数已被证明正确”的保护集。

当前 ALR 的 `requires_grad_` 布尔切片赋值不能作为可靠选择性冻结证据；这是既有接口风险，需与研究创新分开。本轮不修它。功能关闭时保留实际基线；任何未来冻结修复若改变基线行为，必须单列控制，不能把工程修复收益归于新方法。

先测这个扩展 ALR 能否改善基线。若主要误差是平行平面位置偏移，它可能无效，此时梯度准入也不能补出信息，应转向第 7 节的锚定深度约束。

## 5. 第二增量是可选的先验更新准入，不能提前宣称必需

### 5.1 g_H 并不需要写出巨大 Jacobian

令 \(\theta\) 为选定几何参数，\(h_s=\nabla_\theta C_{rs}(D_\theta)\)。反传链是：

`feature cosine → bilinear sampling → source projection → reference backprojection → PGSR plane_depth → Gaussian parameters`。

FatesGS/VA-GS 已实现前半链，AmbiSuR CUDA 已实现后半链；自动微分计算向量—Jacobian 乘积，不需要构造完整 Jacobian。`grid_sample` 分片可微，投影/除法需深度非零；可见性、排序、掩码切换使“全局平滑”不成立。掩码冻结、局部排序稳定的步内可以用一阶分析。Toy 解析/有限差分误差约 `1.6e-10` 只核对数学链，不是本地 CUDA gradient check。

实际还有一项污染风险：本地 CUDA Ray-Color 会在 backward 中隐式加梯度。若不隔离这条路径，`autograd.grad(C_rs,theta)` 可能并非纯特征代价导数。诊断分支须禁用隐式注入或提供隔离接口，且避免重复 backward 重复注入；不能仅凭损失名称声称得到 g_H。

### 5.2 泰勒式成立，但要限定优化器和目标

对规定方向 d，小步满足：

\[
C_{rs}(\theta-\eta d)=C_{rs}(\theta)-\eta h_s^\top d+O(\eta^2).
\]

连续可微时余项可写 `o(η)`；二阶受界才用 `O(η²)`。这是普通微积分结果，不需特定 GS 论文授权。[Ren 等 ICML 2018 Eq12](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf) 已用验证梯度与训练梯度关系来重加权，提供方法学先例；其干净验证集假设不等于我们的特征代理。

只有实际更新就是 `d=P g_P` 时，`h_s^T P g_P` 才有此前解释。Adam 含历史动量、梯度依赖的二阶状态以及主损失累积，不能通常写成固定 P 乘当前先验梯度。原始梯度 cosine 更不能直接保证最终 Adam 更新有益。

### 5.3 第一版可执行规则：比较有/无新增分支的实际候选一步

若第一增量已产生有害更新证据，可按同一参数 θ 与同一优化器状态 Ω，无副作用预览：

\[
\theta_b=\operatorname{Step}(\theta,\Omega,g_b),\quad
\theta_t=\operatorname{Step}(\theta,\Omega,g_b+g_E).
\]

这里 g_b 是实际基线总梯度，g_E 是新增且已隔离的几何梯度。冻结拓扑，预览保持实际 optimizer 的原始参数与状态语义：本地四元数在 `get_rotation` 渲染激活中归一化，optimizer 并无 step 后参数归一化；因此不能额外改变原始四元数。两种候选均通过同一渲染激活求代价；预览不更新 Ω，不累计 densification 统计。

两种候选都必须在原固定支持上继续满足可见性、贡献归属与边界检查；若更新后支持丢失，直接取消新增分支，不能缩小 mask 抹掉不利样本。在同一有效支持和权重上，对各支持视角分别检查：

\[
a=\mathbf1\{C_{rs}(\theta_t)\le C_{rs}(\theta_b),\;\forall s\in\mathcal V_{\rm valid}\},
\qquad g_{\rm commit}=g_b+a g_E.
\]

本地 `train.py:534–585` 在 backward 与 optimizer.step 之间还可能执行 densify/prune、multi-view trim 和 reset_opacity，替换 Parameter 或修改 Adam 状态。第一版仅在确定不会发生任何这些参数/拓扑管理的迭代启用预览与新增分支；其他迭代不预览、不新增，完整沿用原管理顺序，不能为了预览跳过 baseline 的管理动作。或者未来明确把预览放在管理完成后的实际 step 边界并重新绑定参数/梯度/Ω，但这不是本次默认。

正式只执行一次 step。无足够有效支持或任一候选产生非有限值时 a=0；拒绝只取消新增分支，保留当前状态下基线更新。数值容差须声明，不偷换“严格不增”。第一版可验证合并的新增 batch 提案，避免各点单独通过后合并产生新的相互影响。

这个规则是借鉴验证式重加权后提出的 **适配假设**，不是 HarmoGS/CLEAR 已提供的 AmbiSuR 插件。它比直接 raw-dot 准入更接近真实优化器行为，但要付出候选渲染和无副作用 step 预览成本；静态审计不能保证 4090 预算。先记录梯度/有限步的分歧频率，再决定是否值得实现。

同一证据用于候选发现与实际更新检查的关系是明确的：前者比较目标几何 G/P，后者比较真实优化器将走到的两个状态。二者都只检验特征代理，不承诺 GT 改善。

### 5.4 多视角冲突没有被一个内积自动解决

不同视角需保存各自 \(h_s\)，不能先平均再声称每视角无害：`h_1^T d=3, h_2^T d=-1` 的均值为正，但第二视角受损。上面的逐视角有限步准入会拒绝；它 **没有协调各视角自身的训练梯度**。

HarmoGS 的 photo-view 梯度协调仍是另一候选。若未来引入，应明确 `g_RGB,r`、`g_RGB,s` 与特征 `h_s` 的不同目标，先在照片通道内部验证其作用，再决定怎样合入先验。CLEAR 的观察 LR 对生成 HR 证据路由可启发准入，但其超分设定和 mesh 目标不同，不应声称已经解决本项目的照片—DA3 冲突。[HarmoGS](https://arxiv.org/html/2605.13073v1)、[CLEAR](https://arxiv.org/html/2608.02206v1)。

### 5.5 “目标好、局部更新不合适”只有可能性证据

考虑非凸代理 `f(x)=(x-1)^2(1+4x+5x²)`：当前 x=0 代价 1，候选 x=1 代价 0，但 f'(0)=2，小步下降走向负方向，暂时降低 f 却远离候选。它证明 **有限目标比较不推出局部更新朝向候选**，不证明真实 AmbiSuR 经常发生此事。

[AbsGS §3.2](https://arxiv.org/html/2404.10484v1) 分析 alpha 混合中像素梯度抵消，说明渲染耦合有实在优化后果；其结论用于增密，不是“正确先验必然产生错误 ALR 梯度”的定理。此前将可能性写成第二模块的必要性，是过强结论，已修正旧报告。

必须先做 M1、再查新增更新是否损害代理/几何。若没有实际伤害，或准入后收益不变，应删除 M2，而不是为故事保留它。

## 6. 投影公式更正与暂不保护参数的理由

用户反例正确。原式只解了半空间，没处理激活的步长球。对 **M=I、单一齐次半空间、Δ≥0**：

\[
y=d_0-\frac{\min(0,h^\top d_0)}{\|h\|^2}h,
\quad d^*=\begin{cases}y,&\|y\|\le\Delta,\\
\Delta y/\|y\|,&\|y\|>\Delta.\end{cases}
\]

`h=0` 只投影到球。h=d0=(1,0)、Δ=.1 时结果(.1,0)。一般度量 M、非齐次边界、多半空间不能直接沿用这个闭式。本轮 1000 个随机例的 KKT 数值残差小于 `6.4e-16`；旧稿已补条件与正确公式。

[A-GEM ICLR2019 §4 与附录 C](https://arxiv.org/html/1812.00420v2) 对梯度半空间投影有明确公式/证明；[PCGrad](https://arxiv.org/abs/2001.06782) 是相关比较项。但保护 `h^T d≥0` 只是保护代理损失的一阶趋势，不是认定 h 对应的高斯正确，也不是将参数冻结。哪些几何参数可信，并未被这些论文证明。

因此保护集、QP、梯度旋转、多状态路由都后置。第一版有/无新增分支的实际步准入已足以形成清楚动作；只有证明二元拒绝损失太多有益修正，才研究柔性投影。不能先加复杂 solver，再倒推其必要性。

## 7. DP-GS：先按原机制适配位置约束，再讨论扩展

[DP-GS ECCV2026](https://arxiv.org/html/2607.03765v1) 从几何/光度一致性取得较有支持的渲染深度锚点，以先验法线把它传播到低置信区域。其代码基于 PGSR，这一点比换成 SDF/新 primitive 更贴近 AmbiSuR。

### 7.1 插在现有深度损失旁，不替换 ALR

同一局部平面内，z-depth rays `r=K^{-1}p`：

\[
n^\top(d_i r_i)=n^\top(d_j r_j)
\Rightarrow d_i=\frac{n^\top r_j}{n^\top r_i}d_j.
\]

法线提供位置之间的关系，锚点提供平面偏移。它不是照片位置与先验位置的加权折中，也不是单点加强法线 loss。

DP-GS `train.py:300–335` 的实际机制是：根据 confidence 选择渲染深度锚点，利用法线比值与邻域 unfold 多步传播；在 `no_grad` 下形成目标 \(D_{\rm prop}\)，再用掩码 L1 监督可导渲染深度：

\[
L_{\rm prop}=\frac{\sum_p m_{\rm prop}(p)
|D_\theta(p)-\operatorname{sg}(D_{\rm prop}(p))|}
{\sum_p m_{\rm prop}(p)}.
\]

第一移植候选保留 renderer、初始化、SH/ALR、现有 DA3 depth loss 与 topology，在 depth loss 旁增加此项；复用已有 PGSR round-trip/NCC 几何与采样接口取得锚点。可先复用 DA3 导出的法线，避免把 Metric3D/MASt3R 等整套新输入一并搬入。[DP-GS 训练代码](https://github.com/hanl2010/DP-GS/blob/dc1ce835ea3c6bb7887333c50df7a13497a45c09/train.py#L300)。

真实 confidence producer 并不相同：DP-GS 在 `train.py:227,282–292` 对采样 patch 的 RGB 三通道分别求 NCC，生成 `(2−ncc)/2 * roundtrip_weight`，按保留的 `valid_indices` scatter 到 `[HW,3]`，未采样处为零；三列是 RGB 通道，不是三个支持视角。AmbiSuR 当前用灰度 NCC，并在过滤后只生成向量/均值 loss，没有这个逐像素图。薄适配必须保留真实像素索引并构造图，不能把现有 scalar loss 直接传入传播；先按原 RGB confidence 机制适配，若选择灰度版本须另行注册对照，不能继承原阈值 .6 的校准。

传播损失的深度梯度也可经混合权重流向 opacity/scale，而不仅是位置/旋转；保留源机制时不应声称实现了参数冻结或先验分离。若后续限制参数组，应单独消融，不能把该限制隐含在“位置约束”措辞中。

换用 DA3 深度导出法线会改变先验误差结构，不能继承原论文的数值收益。特别是深度先验的高频噪声会放大到法线，需与原预测法线机制区分；当前不下载/新增运行任何模型。

### 7.2 先闭合必要接口，不把工程修正包装成创新

`utils/graphics_utils.py::get_pos` 实际使用 `arctan((u+.5-cx)/fx)`；本地 PGSR 反投影是针孔 z-depth rays。应核对射线定义、半像素和坐标系，不能把角度当归一化射线坐标直接复用。源码分母近零时用 sign-clamp，恰好零仍可能为零，并会将 NaN/Inf 比值设为 1；第一适配应显式把退化传播标无效。[DP-GS 几何工具](https://github.com/hanl2010/DP-GS/blob/dc1ce835ea3c6bb7887333c50df7a13497a45c09/utils/graphics_utils.py#L212)。

还需保证：无锚点/最终 mask 空时新增 loss 为零；不跨越检查到的区域边界；掠射或非有限传播退出；世界法线变换到射线所在相机坐标；normal 必须归一化；target detach。原论文也明确远离高置信区域的传播精度下降，不能声称能处理所有暗区。

这些是迁移正确性与回退合同，不是主贡献。第一版不增加 uncertainty propagation、新网络或全局平面状态，也不把异常边缘平滑第二模块一起搬入。先测 loss-only 适配的净收益。

它可以独立先测；若确有收益，再检验是否专门用于第一增量无法区分且存在邻近锚点的区域。将 M1 特征支持用于 DP 锚点，是另一阶段的共享证据假设，不要提前声称已闭合全算法。

## 8. 回退的准确含义

功能关闭时应与实际基线等价。功能开启但当前新增候选/传播/准入无效时，额外分支为零，执行 **当前模型与当前 optimizer 状态下** 的基线动作。

这不保证曾接受新更新后的轨迹回到独立 baseline run，也不保证最终指标不会下降。局部准入只保护所用支持代理，且 topology/未来渲染仍可能受之前更新影响。未来必须在 G0 后测端到端几何与图像指标，不能以“可回退”替代效果验证。

不制作人工高光 mask。GT 只用于已经获准的最终评价或 D0 标签；本文件不批准新的 GT 访问、Utility 恢复、Geometry Q 重启或任何实验。

## 9. 最小实验问题，而不是一次堆满模块

| 顺序 | 唯一新增能力 | 关键对照 | 所回答的问题 |
| --- | --- | --- | --- |
| 接口前置 | 关闭等价、梯度来源/归属验证 | 实际基线 vs 关闭新分支 | 收益是否混入工程修复或隐式注入 |
| P1 | A 种子附近，经特征假设比较扩展 ALR | 扩大 SH 比例；同预算随机/全局种子；不验证而扩展 | A 是否提供有效搜索，验证是否提高新增监督效用 |
| VCR 对照 | ALR 中方向一致性权重 | 现有 ALR | 简单成熟先验权重能否解释主要收益 |
| P3 单独 | 原机制的锚定传播深度 L1 | 现有 DA3 depth L1；同范围平滑；anchor-only | 收益是否来自 normal+anchor 的关系约束 |
| P2 条件试验 | 有/无新增分支的逐视角有限步准入 | P1；平均视角准入；只记录不执行 | 有害额外更新是否存在，准入是否有实际净收益 |
| 联合实验 | 仅联合已证明有用的两项 | 单项、联合、独立简单叠加 | 共享证据/作用区域是否有超出简单叠加的价值 |

指标应记录新增候选覆盖、有效支持/拒绝率、额外梯度范数、各视角代价变化、锚点覆盖、时间显存，以及原有正式几何/图像指标。收益无需数学上“超加性”才能构成有效方法，但需明确优于基线且不能由更大监督范围/预算单独解释。不因单场景正结果跳过跨场景与多 seed。

此前冻结 N/A 对比只显示很小提升且未过 G1，不能当成新方案有效证据。不能以已 inconclusive 的 Utility 发布接口问题为理由自动重跑。

## 10. 新论文也要过机制审计：TriaGS 的暂缓理由

本輪补查 [TriaGS，WACV2026 / 2025 预印本](https://arxiv.org/html/2512.06269v1)，不是只检索 2024 工作。其 `compute_tgpc_loss` 将参考深度点 X 投影到邻居，再用同一相机矩阵和这些投影重三角化，未在该函数内取得独立图像对应或邻居深度点作为三角化测量。[固定实现](https://github.com/cogniboticslab/triags/blob/179d146bacb5c30fd50ff190d55c0a0e6d5c770f/utils/multiple_view_loss.py#L153)。

理想一致计算下，由同一 X 产生的投影满足 `A X_h=0`，重三角化应还原 X；因此其残差不能直接当成独立支持证据。外部 round-trip mask 会筛点，但不改变此函数的三角化观测构造。有限精度、归一化、其他 loss 可使实际行为不同；这里不推断整篇论文实验无效，只说明该模块作为本项目“新证据”仍缺必要解释，暂不移植。新近与可执行不是充分的科学依据。

### 10.1 另外两篇 2026 工作：筛选依据与未推荐理由

沿 CLEAR 作者页追查了 [DOC-GS，2604.06739](https://arxiv.org/html/2604.06739v1) 与 [PairDropGS，2605.12072](https://arxiv.org/html/2605.12072v1) 的方法部分。作者页该条目无代码链接，补充标题与 GitHub 定向检索仍未定位可核查官方实现，不能标成已验证移植。

DOC-GS 将 floater 渲染近似为雾幕来使用 dark-channel 线索，但原文明确是解释性类比；Eq15 将整幅图的异常比例加给当前所有可见高斯，并非精确像素—高斯归属。其主评价是 NVS 图像质量，不为当前几何修正提供独立位置验证，故暂不搬入 pruning。

PairDropGS 用两次 dropout 渲染的低频图像一致性和 stop-gradient 约束来稳定训练；它可以启发扰动稳定性诊断，但两份模型都可能共享错误结构，没有深度锚点就不提供绝对几何位置。此时引入随机 dropout 还改变基线训练与拓扑统计，超出本次小增量范围。

这些筛选展示为什么“有新的 2026 论文”仍不等于值得加一个模块。没有直接适配证据的候选保持在清单，不能用论文自身的可靠性措辞证明我们的指标是几何真值。

## 11. 给导师的可核查论证与创新边界

能够严谨说明的主线是：**保留 AmbiSuR 的 SH 触发与现有 ALR，用可比较的跨视角证据检验未命中的具体先验修正；对于单点法线不能确定的平面偏移，用有支持的位置锚点和法线关系补约束。** 梯度准入只在观察到有害额外更新时添加，不作为预设主贡献。

机制来源各有分工：Sparse2DGS 提供候选几何比较；FatesGS/VA-GS 提供固定特征和可见性采样；VCR 提供方向一致性权重；DP-GS 提供锚点—法线—深度传播；Ren 提供验证更新的思路。公式成立条件、代理边界和本地 CUDA 接口另行核查。

这仍是待验证的适配方案。不能因为 A+B 有可能有效就说已创新；需要展示每项依附的位置、关掉后回退合同、相对简单对照的收益、联合是否利用同一证据而非随意加 loss。若 P1 无用而 P3 有用，主线就应收敛到现有深度/法线先验使用方式，不强留 A 检测故事；若 P2 无收益，就删除它。发表层面的新颖性仍需与这些来源明确比较。

## 12. 本轮验证记录

源码/论文分析完成，第三方源文件哈希与代数复核见配套 manifest。数学复核范围：单半空间+球的 KKT；平行平面法线不辨识偏移；固定特征投影链有限差分；逐视角平均隐藏冲突；非凸有限候选与局部方向不等价。

没有实际 Torch/CUDA 梯度测试、方法实现、模型下载、训练或新几何评价；因此不声称可运行或已改善基线。生产代码保持未改；旧研究稿已修正投影条件、梯度目标和必要性表述。

阶段只读审阅发现并已闭合三处接口/条件问题：DP RGB confidence 的真实索引 scatter；实际一步预览避开参数/拓扑管理；VCR 温度非正的全一分支。审阅还校正了 PGSR epsilon 符号、VCR dirs/inside 分支与本地原始四元数语义。限定复核无剩余 Important/Minor，不等于方法效果或运行预算获验证。
