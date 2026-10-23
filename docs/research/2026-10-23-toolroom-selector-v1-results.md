# Tool Room：二周目selector_v1首次正式几何评价

## 来源与身份

2026-10-23用户粘贴AutoDL日志及完整metrics.csv六行。服务器8项评价测试全部通过/零跳过，用时10.591s；正式进程PID12484随后正常完成，日志明确打印Completed并列出四个JSON/CSV产物。此前PID消失不是异常退出证据。本文为用户回传记录的整理，不冒称原始服务器文件已下载或校验。

- Baseline训练commit：`d6f15c8891a53800d5e3100f95817a7dd7f98e2f`。
- Selector训练commit：`3d85faf95951b4806eee18498841e66551b93320`，Tool Room/r2/seed0/30K。
- 评价commit：`4010fd01e4b5bbe651ac4e221e00532bf99f4ff9`。
- 协议：`experiments/mesh_geometry/scannetpp_surface_v1.json`；完整表面，1M面积采样/seed0，点到三角面，未平方均值，无ICP/crop/截断。PRF严格小于1/2/5cm，主列5cm。
- 输出：`/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/evaluation/toolroom_pair_seed0_30k_v1`；日志为相邻同名`.log`。

## 结果（原始比例和毫米）

| Metric | Baseline | Selector | Selector−baseline |
|---|---:|---:|---:|
| Accuracy mm ↓ | 35.124416906255874 | 37.21935357981245 | +2.0949366735565746 |
| Completeness mm ↓ | 46.30678178912009 | 54.835622311071454 | +8.528840521951366 |
| Surface Chamfer mm ↓ | 40.715599347687984 | 46.02748794544195 | +5.311888597753963 |
| Precision@1cm ↑ | 0.284718 | 0.288533 | +0.003815 |
| Recall@1cm ↑ | 0.287127 | 0.287689 | +0.000562 |
| F1@1cm ↑ | 0.28591742582692864 | 0.28811038189100724 | +0.0021929560640785994 |
| Precision@2cm ↑ | 0.508735 | 0.516255 | +0.007520 |
| Recall@2cm ↑ | 0.507029 | 0.508585 | +0.001556 |
| F1@2cm ↑ | 0.5078805673660418 | 0.5123912984953748 | +0.004510731129332979 |
| Precision@5cm ↑ | 0.791756 | 0.797179 | +0.005423 |
| Recall@5cm ↑ | 0.767304 | 0.764484 | −0.002820 |
| F1@5cm ↑ | 0.7793382497453593 | 0.7804892484947137 | +0.001150998749354426 |

相对baseline：Accuracy增加5.9643%，Completeness增加18.4181%，CD增加13.0463%。F1@1/2/5cm分别增加0.2193/0.4511/0.1151个百分点，不能把0.001151写成提高1.15个百分点。

## 当前结论与边界

结果混合：阈值内命中率部分改善，但两方向平均距离均恶化，当前不能声称获得稳定或全面的几何收益。5cm主列F1确有小幅上升，不应隐藏；距离明显退化也不能因F1上升而忽略。没有多seed统计，不能给微小差值附加显著性结论。

两类指标不矛盾：PRF只看是否跨过阈值，不区分阈值外错误有多远；均值包含距离大小。现有汇总不足以判定退化来自少数严重偏移、缺失区域、浮面、表面面积权重变化或提取后处理；不能先宣布任何一种根因，更不能直接认定是候选NCC或ALR机制失效。

本次对比资产是服务器C0 mesh，不把先前本地original训练日志的PSNR拼成同一组正式配套成绩。原协议/全部指标/原产物保留，不按负结果改阈值/裁剪/截断，不自动重训或叠加VCR、梯度模块、深度传播。下一批分析须先讨论，优先使用现有mesh同视角检查，确认具体退化位置后再决定是否需要额外数值定位。

## 用户自行下载归档

用户明确自行下载，不提供SSH、不由agent连接。建议保存以下三部分，不写入原AmbiSuR_original目录：

1. `/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/3d85faf_seed0_30k_v1/`：训练模型、mesh、日志、配置等完整run。
2. `/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/evaluation/`：结果目录及相邻日志。仅20KB左右的表格/元数据不能替代大模型/mesh备份。
3. `/root/autodl-tmp/ambisur_runs/Tool_Room/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z/`中的本次参考mesh、配置和训练/提取日志，归档为独立`baseline_reference_c0/`。本地original与这份服务器C0资产尚未证明相同，不能默认用前者代替。GT若已有本地副本，可按results.json的SHA256核对，无需盲目重复下载。

建议本地父目录：`D:\research_Space\output\AmbiSuR_second_run\ScanNetpp\Tool_Room\selector_v1\`，保留服务器下`3d85faf_seed0_30k_v1/`与`evaluation/`的相对结构。原始JSON/CSV直接下载，勿只保存截图；下载后再核对大小/mesh哈希，当前不声称本地归档已完成。服务器原件暂不删除。

## 训练日志与代码的初步机制分析（非根因判定）

用户询问4096是否过少，并告知大output下载较慢。只读发现新run的日志/配置已在本地 `D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/selector_v1_2026_10_23/3d85faf_seed0_30k_v1` 可读；大mesh/完整评价归档仍不宣告完成。

从实际train.log解析115条每200轮记录（7200..30000）：sampled中位4096、comparable2699、added983；added均值1011.9304，sampled=0共2条。日志样本合计comparable/sample=0.6336615908，added/comparable=0.3967826956，added/sample=0.2514259541；这些不是全训练逐轮或去重覆盖统计。train/extract.exitcode均0。实际参数unc_weight0.1、unc_decay1、unc_conf_thresh_ratio0、预算4096、7x7灰度patch、两个nearest支持、cP<cG且cP<0.5，与规格一致。

代码事实：7001..30000每轮当前参考相机都抽样，200只是日志间隔；预算不代表总训练只检查4096个。A阴性且中心在前/patch在图内的池先抽样，不是按参考视角alpha贡献优先。新标签只当前迭代/相机有效，不累积。候选胜出比较同时依赖平面法线和距离；真正追加的ALR仍只有深度导出法线残差，原全局DA3深度L1不因补选而改变。Sparse2DGS update_points直接替换位置，与本适配动作不同。

固定同一模型/相机时，标签渲染线性：DeltaM(p)=sum_{i in extra} T_i(p)*alpha_i(p)。额外ALR为lambda/(HW)*sum_p DeltaM(p)*confidence(p)*(1-nP dot nD)。因此added计数不能替代额外权重/残差/梯度量，total_alr_weight包括原A，不能证明新增剂量。实际池大小/可见贡献/跨轮重复未记录，不能单凭4096/总Gaussian数判定预算充分或不足。

优先待判别假设：①抽样标签的有效渲染覆盖/剂量不足；②NCC胜出但不代表目标几何或法线可靠（两源best-source、无margin、0.5供体阈值）；③平面位置证据没有在新增法线动作中兑现。平行移位平面保持深度导出法线，构成③的数学反例；不等于声称法线不能影响位置，也不证明该反例主导本次退化。CD恶化不能只用“作用太小”解释，必须检查伤害来源。PSNR是train指标，且先前original日志与服务器C0资产未证明同一身份，不能拼成严格配套正式渲染表。

待讨论动作：下载完整后优先少数相同视角的mesh/G-P patch/额外mask检查；若只做数值对照，单改预算并跑一项30K，保持其余不变，区分更多监督改善/放大伤害/无明显变化；不同时改阈值/支持视角/损失。不已获执行授权，不自动实验或恢复旧诊断架构。

## 下载完成后的首轮本地核对与预算分析

用户确认下载完成，随后明确服务器C0 baseline尚未下载并开始下载它。当前可读归档父目录为 `D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/selector_v1_2026_10_23/`。

### 产物与对照身份

- 新run1577文件/2,324,851,119字节；renders/renders_depth/renders_normal各406张。7000/30000高斯PLY长度严格等于header+N*62*4，分别367487684/321352988字节；train/extract.exitcode均0。
- 新post mesh597628670字节，SHA256=`23255cc3c6fc0e9d2ef37d6a624e3ad6f0545e183a802a1631d10e4c8a7588c4`，与正式results.json完全匹配。
- 本地GT104149767字节，SHA256=`31547a31069f736792d4b13fff76c73483f59239dbbabe1971e115f9ab17171d`，与正式记录完全匹配。仅核对原文件哈希，没有重算距离或读取GT生成标签。
- JSON和CSV全部指标逐值相等；正式CPU评价耗时87.2829s，Open3D0.18.0/NumPy1.26.3/clean4010fd0。
- 原本地original post mesh611443570字节，SHA256=`b3ce07c8830d1ead18d3c279ac7a6d5f525e9184978e8b17f355aacc3c0a6243`；正式C0605468503字节，SHA256=`396676490395879cbefe49c776b05f4094641c2a17835d4250e2812c70ec7e61`。明确不是同一资产，不拿original替代正式C0。
- 新run与本地original的input.ply/cameras.json/multi_view.json分别逐文件哈希一致。cfg_args仅输入/输出路径不同；cfg_opts仅新开关/预算/显式seed字段不同。但original与新run在3000/7000已存在PSNR差异，7000高斯数1486111 vs1481799；选择从7001才开始，这些早期差异不能归因于selector。不能据此认定具体随机性/环境原因，也不要求重训baseline。

### 最终模型的候选池

只读memmap最终Gaussian PLY，用AST提取原 `GaussianModel.compute_weighted_sh_norm` 的真实函数并调用真实 `compute_sh_ambiguity_labels`，不复制SH公式、不加载CUDA场景模块。恢复保存相机/实际876x584图像，按原pool规则计算三个预先按训练日志选取的视角；相机float32恢复在边界可能有少量差异，属于最终快照估算，非历史逐轮或真实可见性统计。

| Camera | A阴性且前向/patch在图内的池 | 4096/池 |
|---|---:|---:|
| DSC06293（末轮日志视角） | 129416 | 3.164987% |
| DSC06337（最高added日志视角） | 741704 | 0.552242% |
| DSC06676（极低comparable日志视角） | 164701 | 2.486931% |

最终模型N1295772，原A标签194367（约15%）。三个池里opacity<0.01比例约11.80%/9.21%/14.24%；没有证据把整个池说成几乎透明，遮挡/贡献仍需真实渲染。不能把三个视角当全场景代表性分布或历史覆盖率。

### 预算对新增监督的可推导影响

固定模型与参考视角，pool大小M，无放回均匀抽B；每个候选的G/P判断在该固定模型下与同批其他候选无关。标签通道为alpha线性合成，掩码在ALR中detach。因此，若DeltaL_full表示同一模型上全pool比较所形成的额外ALR，当前额外损失满足 E[DeltaL_B|model,view]=min(1,B/M)*DeltaL_full。不是训练最终收益公式，也不能把它直接推广为Adam更新期望。

这意味着预算不只节省计算，还把不同视角的新增监督乘上不同的抽样比例；上述三视角相差约5.73倍。4096不足成为有具体机制与数据支持的候选解释，但不能认定为唯一原因或声称16384必然改善，因为更多错误准入同样可能放大伤害。没有修改预算、加抽样补偿或新训练。

### 已有可视化的限度

查看30000_DSC06293、15600_DSC06337、20000_DSC06676三幅已保存debug图：源照片/渲染图、直接法线、深度法线和总掩码均存在；未做GT空间误差定位。右下掩码是A|extra且每图除自身max；pg_view是原NCC训练采样，不是selector G/P比较证据。故现有图不能还原原A/extra各自的绝对监督质量，也不能从亮红区域宣布补选已覆盖到位。20000轮added=1但总掩码仍广，直观说明两者不可混同。

当前不重算评价，不修改代码/训练参数。待C0到齐后，用正确参考资产做同视角空间对比；若需要新增渲染/距离分析或单预算30K对照，先讨论范围。
