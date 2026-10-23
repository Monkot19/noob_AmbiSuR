# AmbiSuR 候选选择模块：首版规格

日期：2026-10-21。状态：**本书面规格已获用户确认；2026-10-22五文件本地实现/CPU检查完成，AutoDL原生验证待执行。** 源码核对点：`62e9117dfb2e20a06b257d166d8a6636d37845cc`。用户要求后续行动先讨论，实际重建收益优先，继续使用已有 baseline 固定 output。

## 1. 目标、适用范围与来源

检验 H1：原 SH 未命中的一部分 Gaussian，可以通过几何候选比较被补选，追加原 ALR 后改善最终表面重建。NCC 胜出不等于真实几何正确或保证 ALR 有益；最终以重建结果判断。

只接入“候选比较→临时二值标签→现有标签渲染→原 ALR”。本规格经书面批准后，作为二周目 selector 的独立合同；旧 N、双可靠性、五状态、路由/生命周期及其 C1–C6、N-AUROC 门不适用于本模块。旧结果保留；数据只读、真实接口复用、明确 commit/配置/seed、GT 仅用于获准离线评价等规则继续适用。不恢复 Utility 旧评价或 Geometry Q。

依据为 Sparse2DGS 的双几何解释 NCC/best-source 比较，以及 AmbiSuR/PGSR 已有投影、NCC、标签渲染和 ALR。供体两个解释改为当前原语与 DA3 平面，直接位置替换改为追加标签；不是完整 SGU 原样复现。论文与源码证据已记录在 [10-15 路线](../../research/2026-10-15-source-faithful-research-route.md) 和对应来源 manifest；不把已有投影/遮挡/NCC 作为新贡献。

## 2. 已确认的首版配置

| 项目 | 合同 |
| --- | --- |
| 开关 | 新增独立 `enable_candidate_selector=False`，不放进旧 `CoreConfig` 阶梯；首版只支持旧 Core 全关、`use_mono=False` 且有 DA3 输入的路径 |
| 时序 | 开关开启且 `iteration > max(1000,unc_from_iter)` 才执行，复用原ALR的外层深度分支与内层启用条件；默认从7001开始，每轮当前训练参考视角计算一次 |
| 候选数 | `candidate_selector_budget=4096`；每轮最多、无放回；不足取全部；不是理论最优值 |
| 支持相机 | 当前相机既有 `nearest_id` 有序列表前两个真实相机；不重排、不随机选、不补虚拟视角；列表不足两个时整轮保留基线 |
| 图像/patch | 原始灰度图、各相机自己的 `ncc_scale` 和内参；沿用 `multi_view_patch_size`，现有值3即7×7 |
| 选择阈值 | best-source 的严格 `c_P<c_G` 且 `c_P<0.5`；0.5为供体移植起点，不新增margin/融合分数 |
| 标签有效期 | 当前参考相机、当前迭代；不保存到模型/checkpoint，不跨相机/迭代累积，无EMA或拓扑迁移 |

不同场景使用同一实现和参数化测试。场景身份/路径/分辨率/seed由既有配置提供，不复制 solver、evaluator 或测试算法。正式实验前的参数改变另行讨论，不在观察结果后隐式修改。

## 3. 候选、平面与抽样

1. 用原 `compute_weighted_sh_norm('equal')`、原双端分位阈值、低端上限及严格比较生成 `b_A:[N] bool`；保留全部已有命中，不修改 raw A。将原重复标签公式提取成一个公共函数，renderer 与训练端复用，不增加第三份公式。
2. 从 `b_A=False` 且中心有限、参考相机正向可投影、整个参考 patch 在图内的 Gaussian 建池。投影使用实际相机矩阵；不借用主渲染后才产生的 radii，也不把可投影称为真实可见。不加 opacity、尺度、A邻域或贡献占比门。
3. 均匀无放回抽样至多4096；候选之后因先验/支持无效退出时不补抽。专用局部 Torch Generator 按 `SHA256('ambisur-selector-v1|seed|iteration|camera_uid')` 前8字节little-endian并取低63位设种子；在候选ID升序池上抽样。不得消费全局 Python/NumPy/Torch RNG；同环境可复现，不承诺CPU/CUDA抽样逐位一致。
4. G平面取 Gaussian 位置及 `get_normal(reference)` 的世界系最短轴法线，按当前行向量接口变换：`X_G=xyz@world_view_transform[:3,:3]+world_view_transform[3,:3]`，`n_G=n_world@world_view_transform[:3,:3]`。P平面在同一亚像素参考坐标，双线性读取与训练分辨率对齐的 DA3 **z-depth**，按 `X_P=D_P K^{-1}[u,v,1]^T` 反投影；已有 `render_normal(reference,D_P)` 当前实际输出**参考相机系法线**，采样/归一化后直接用，不能再做world→camera旋转。其底层 `depth2point_world` 虽名含world，但世界变换已注释、实际return xyz_cam；必须以源码和非单位相机姿态测试为准。深度尺寸对齐沿用基线 bilinear/align_corners=True；不把z-depth当单位射线长度。
5. 两法线均朝参考相机，保持 `d=-n^T X>0`，平面约定 `n^T X+d=0`。正常量/深度/坐标必须有限；法线范数、平面距离及齐次分母退化以 `1e-6` 数值检查拒绝，不用 clamp 把无效候选变有效。中心需在相机近面之后；边缘无有效深度邻域或无法构造法线时取消该候选。

参考 DA3 深度/法线预处理只算一次，并在本轮原 ALR 中复用；不改变原 confidence 阈值、初始化或深度监督。源视角深度见下一节，不能用DA3源深度悄悄替换。

## 4. 有效支持与 NCC 比较

对每个已采样候选、两个固定源，分别计算 G/P：

- 相对外参按 `X_s=R_sr X_r+t_sr`；用 `H=K_s(R_sr-t_sr n^T/d)K_r^{-1}` 及既有 `patch_offsets/patch_warp` 生成源 patch，各源按自己的灰度图宽高归一化，`grid_sample(...,align_corners=True)`。整个参考/源patch坐标有限且在图内，平面交点在源相机前方；padding值不当作有效观测。
- 取得该迭代当前模型的源 `plane_depth`，用既有 `get_points_depth_in_depth_map` 采样候选中心对应深度；其源Z>0.1和范围规则原样复用，单位是现有场景内部单位。采样深度须有限且正。沿源射线回投参考，要求回投深度有限、正、在近面之后；按基线使用**参考全分辨率像素**往返误差与 `multi_view_pixel_noise_th`。若原 `wo_use_geo_occ_aware=True`，同基线不执行误差阈值，其他有效性仍保留。该检查只是当前模型条件下的可用性代理，不保证真实无遮挡。
- 将两假设几何/数值有效性取交集为共同源集合 `V_i`；先固定两个源，每候选交集剩一个也可比较，空集则 `e_i=0`。不要求两源都胜出，也不补选列表后面的源。
- 使用现有 `lncc` 返回的**原始有限代价**。它是平方相关形式；其另一个返回值 `cost<0.9` 服务于当前几何优化，不用来预筛G或定义共同源。否则当前几何很差的候选会提前被剔除。P是否足够好由0.5阈值决定；弱纹理等无法区分时不追加。

\[
c_G(i)=\min_{s\in V_i}\ell(G_i,s),\quad
c_P(i)=\min_{s\in V_i}\ell(P_i,s),\quad
e_i=\mathbf1[V_i\ne\varnothing]\mathbf1[c_P(i)<c_G(i)]\mathbf1[c_P(i)<0.5].
\]

最优源可不同；比较在no_grad内，只输出标签，不以NCC产生额外训练梯度。

## 5. 标签时序、成本与回退

每轮：**原相机抽样/原背景→计算b_A与候选→两源深度及比较→b_new→一次主参考渲染→原损失/ALR→原优化与拓扑**。`b_new=b_A|e` 为 `[N] bool`、同设备、detached；原命中始终保留。新 `render(...,ambiguity_labels=None)` 可选参数只替换原第5通道标签；训练端同轮 `unc_mask` 使用同一 `b_new`，不重算SH-only标签。

首版没有跨迭代缓存；最多增加**两个no_grad源深度前向**，只在仍有可比较候选时进行，`ray_reg`关闭，无源法线渲染。它们不能替换原多视角loss需要的可导源渲染，也不提前消费/改变原随机源选择。本规格不承诺复用这两个前向降低原loss开销。4096×2假设×2源最多16,384个源patch比较；源渲染/图像读取/全量候选投影的成本另计，需实测。

原ALR residual、权重/confidence、归一化、调度、backward、参数管理及其它损失/拓扑保持不变；不修复旧 `requires_grad_` 语义风险，不声称梯度只到选中点，不新增CUDA通道、参考预渲染、梯度投影或新loss。

| 情况 | 行为 |
| --- | --- |
| 关闭或尚未到启用轮 | 原路径，不建候选/局部Generator、不预处理先验、不额外渲染 |
| 空池、少于两个真实源、正常数据中先验无效、共同源为空、比较未通过 | 对应候选e=0；整轮均无追加时等于b_A |
| 标签长度/device/dtype不符、缺少声明必需的DA3数据或不支持的模式、模型非有限、代码异常/NaN | 明确报错停止，不用catch-all回退掩盖接口/模型故障 |

不写baseline输出，不读取GT，不写checkpoint标签状态。旧checkpoint格式与原metadata/serializer复用；配置记录显示新开关，不能因旧Core路径名仍为legacy就声称新方法关闭。

## 6. 最小改动与验证边界

| 文件 | 唯一职责 |
| --- | --- |
| `utils/candidate_selector.py`（新增） | 公共SH标签函数、候选抽样/平面/有效支持/NCC决策；依赖现有工具，不依赖旧reliability模块；避免与renderer循环导入 |
| `arguments/__init__.py` | 仅增加独立开关和预算两个字段，复用现有解析/metadata |
| `gaussian_renderer/__init__.py` | 可选标签输入及第5通道，None保留原语义；复用公共SH标签函数 |
| `train.py` | 启用/输入准备、两源调用、生成标签时序与同标签消费；保持原随机选择和原损失路径 |
| `tests/test_candidate_selector.py`（新增） | 一套参数化测试入口，CPU数学/决策与条件GPU真实接口测试同文件；不按场景复制 |

现有warp/NCC/深度采样函数直接调用，必要的维度与坐标适配留在selector。不因适配顺手重构公共几何工具、旧框架、评价器或CUDA。

同一测试入口覆盖三组有意义的证据：①已知平面/相机的投影与往返，含**非单位旋转/平移及真实render_normal生产者**以检查G/P法线坐标、原NCC真实返回类型、best-source/空源/差G代价、预算/局部RNG；②真实标签producer→真实renderer→`rendered_unc`→原ALR消费与梯度，关闭及e=0对照；③条件GPU smoke覆盖7001后的活动路径、有限性和时间/显存。纯字典/mock/数学恒等式不能作为唯一接线证据；复用既有测试作为回归，不按场景重建。

仅记录抽样数、共同源非空数、追加数、总有效ALR标签权重及运行成本，沿用日志。单个第5通道只给M_new，不能把它的权重质量称为“新增量”；在上述真实接口检查中可用e-only标签额外前向验证新增监督传递，正式训练不为诊断例行多渲染、不加通道或新统计系统。

## 7. 首个实验与下一步

**2026-10-22用户评价修订：** 本节补齐基线评价协议，不改变§1–6已批准的候选算法或五文件边界。baseline直接用已有固定output；新方法独立目录，优先对齐原数据/DA3、r2、seed、30k预算及提取设置，不重训/续训baseline。

| 数据集/用途 | 场景与分辨率 | 报告指标 | 直接复用的既有入口 |
| --- | --- | --- | --- |
| DTU表面几何 | 24、37、40、55、63、65、69、83、97、105、106、110、114、118、122；r2，2DGS预处理 | d-to-s↓、s-to-d↓、CD↓；CD=(mean_d2s+mean_s2d)/2 | `scripts/eval_dtu/evaluate_single_scene.py`→`eval.py`，沿用观测mask/坐标/采样与距离处理，不改成全域平方距离替代 |
| TnT表面几何 | Barn、Caterpillar、Courthouse、Ignatius、Meetingroom、Truck；r2，Neuralangelo预处理 | Precision↑、Recall↑、F1↑ | `scripts/tnt_eval/run.py`及其evaluation/config，沿用原对齐/裁剪和逐场景距离阈值 |
| Mip-NeRF360新视角渲染 | indoor:bonsai/counter/kitchen/room，r2；outdoor:bicycle/garden/flowers/stump/treehill，r4 | PSNR↑、SSIM↑、LPIPS↓，分室内/室外报告 | `mesh_extract/extract_general.py`生成同口径渲染→`metrics.py`，后者沿用VGG版LPIPS；保持官方eval划分、图像编码与数值域 |

[基线论文§4.1/E.2](https://arxiv.org/html/2605.12494v1)与本地源码共同支持上述分工；论文均值是文献背景，不是我们本地baseline已复现的成绩。几何与渲染结果分别报告，PSNR或中心误差AUROC/NCC/追加数不单独证明重建收益。正式均值列明实际场景数与缺失项，不把汇总脚本的缺失占位值当真实结果。

当前先做Tool Room首轮30k，不自动扩大为全部30个benchmark场景：训练日志train PSNR/L1保留为快速进度；离线渲染补齐PSNR/SSIM/LPIPS时，baseline和新方法必须使用相同明确相机列表/分辨率/编码，结果另存。当前baseline为eval=False，test配对图像为0，只有406张train JPEG渲染；不能直接运行metrics.py得到有效测试指标，不能混用旧JPEG和新PNG宣称同协议提高。可从已有冻结模型补渲染同口径图像到独立评价目录，不重新训练；具体视角与成本在该批次讨论，训练视角结果明确标注而不冒称新视角测试。

Tool Room/Utility属ScanNet++，不能套用DTU/TnT专用评价数据或阈值；其最终表面协议在获准离线评价前单独明确，复用适用的现有工具，不搭新框架、不自动恢复旧GT探针。metrics.py会向所给根目录写results/per_view，补充评价不得把只读baseline目录当可写目标。Mip运行脚本目前启用use_mono，与首版DA3-only输入合同不同，正式迁移时先讨论，不静默改模式或使用该脚本重新训练baseline。

先通过新增边界检查和活动路径smoke，再讨论首次方法训练；有初步收益后比较扩大SH选择与同数量随机补选。DP/VCR独立实验和多seed/跨场景属于后续讨论。当前不把旧G1/G2数字自动移作本模块通过门，也不为开跑先建立新的统计框架。

本轮只交付本规格及根记录；下一步为用户审阅本文件。书面确认后才写最小TDD实施计划并讨论实施；当前不创建分支、改代码、运行测试/实验、提交或推送。
