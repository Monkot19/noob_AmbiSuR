# Tool Room：二周目selector_v1首次正式几何评价

> **最新状态：实际C0已下载核验，配对空间诊断已完成。** 各轮记录按发生顺序保留；以末节为当前结论。下一步优先核对训练输入身份，暂不增加预算或启动训练。

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

## 实际C0到齐后的配对诊断

用户再次确认下载完成后，读取正式C0及新run现有产物，复用原评价核心进行离线空间诊断；没有训练、重新提取mesh、修改方法或改变正式评价域。此前“尚缺C0”“未读取GT距离”的记录只描述前一轮状态。

### 资产与计算一致性

- 实际C0目录：`D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/baseline/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z`。post mesh大小605468503字节，SHA256与正式报告的`396676490395879cbefe49c776b05f4094641c2a17835d4250e2812c70ec7e61`完全相同。
- C0与新run的`input.ply`、`cameras.json`、`multi_view.json`分别字节相同。配置差异仅输入/输出路径及新增seed/selector/预算字段。实际C0和新run均在主入口重置seed0，不能将模块顶层`setup_seed(22)`误当成实际seed不同。
- 复用原生产评价核心与同一组1M GT面积样本，样本SHA256为`efb1f2f330fdabdffa0d4db5f426b411eedfa670cb030b9727c9ffbed46212b9`。本地Open3D0.19.0与服务器0.18.0的原均值指标最大差仅`4.62e-13 mm`。
- 新产物独立保存在`D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/selector_v1_2026_10_23/analysis/toolroom_pair_seed0_30k_diagnosis_v1`。原run、baseline、GT、正式JSON/CSV未写入。GT仅用于离线最终评价诊断；局部坐标文件不得进入训练、可靠性缓存、训练mask或checkpoint。

### 空间定位：均值退化主要集中在上部

对同一组GT点分别查询到两个post mesh的距离，再按世界坐标z事后分层。分层只解释整体成绩，不替换正式评价，也不据此裁掉上部。

| GT分层 | GT面积样本占比 | C0平均Completeness mm | Selector平均Completeness mm | 对整体均值差贡献 mm |
|---|---:|---:|---:|---:|
| z<4m | 72.2129% | 31.748934 | 29.904191 | −1.332142 |
| z>=4m | 27.7871% | 84.139603 | 119.627232 | +9.860983 |
| 全部 | 100% | 46.306782 | 54.835622 | +8.528841 |

z<4m各0.5m分层的平均完整性距离均改善；不表示其中每个点都改善，也不能认定改善由selector造成。最明显退化集中在上部角落，特别是x约5.5–7m、y约2–3.5m、z约4–5m。逐点距离增量最大的1% GT样本对整体均值差贡献+4.528411mm，占净退化约53.1%。Completeness中位数19.5971→19.5102mm略改善，但P99为363.611→530.335mm，解释了F1略升与平均距离变差为何可以同时出现。

### 后处理与完全未观测都不是主要解释

保持同一GT样本，补查raw mesh的完整性方向，不替代正式post mesh评价：

| Completeness mm | C0 | Selector | Selector−C0 |
|---|---:|---:|---:|
| raw mesh | 42.752573 | 50.260804 | +7.508231 |
| post mesh | 46.306782 | 54.835622 | +8.528841 |
| 后处理新增距离 | +3.554209 | +4.574818 | +1.020609 |

差距约88.03%在raw mesh中已存在，不能主要归因于只保留两个连通域。raw仍包含渲染和TSDF融合影响，不能进一步宣称训练本身贡献了88%。

对逐点退化最大的1%（10000个GT位置），使用保存相机与GT first-hit射线检查：84.67%在至少一个相机满足1cm可见容差，2cm为89.25%。多数最坏点不能简单解释为根本没拍到。有限GT mesh和容差不证明充分纹理或可靠NCC。DSC06627可查看其中大量位置；已查看双方保存的同名法线图与源图。

### 关键混杂：区域分布差异早于selector启动

在相同事后诊断包围盒`x∈[4,7), y∈[2,3.5), z∈[4,5)`统计现有Gaussian中心：

| 区域内Gaussian中心数 | C0 | Selector run |
|---|---:|---:|
| iteration7000 | 29904 | 14205 |
| iteration30000 | 28431 | 15848 |

候选比较与补选从7001才启动（`utils/candidate_selector.py::candidate_selector_active`）。7000时已相差15699个区域内中心，不能把最终差距都归因于selector后续删掉或错误监督了上部高斯。中心数不是覆盖率、几何误差或7K mesh成绩；没有7K表面评价，不能宣布7K表面已差同等幅度。

进度日志在100轮已出现打印loss微差，600轮首次记录点数差异206119/206134，早于1001轮深度监督和7001轮补选。舍入日志不能定位首个浮点分叉；固定输入功能关闭等价测试也不证明完整7K训练轨迹逐位相同，不能排除移植代码运行路径或环境差异。

两run的Python3.10/Torch2.7.1+cu128/CUDA12.8/GPU类别一致；相同seed不足以证明逐位复现。原CUDA backward有多处`atomicAdd`，例如607/624/650行，浮点累加次序与后续增密阈值放大是合理待查假设，**未被确认为本次根因**。[PyTorch 2.7复现说明](https://docs.pytorch.org/docs/2.7/notes/randomness.html)

### 正确配套PSNR只覆盖5个训练视角

| 30000轮日志 | 实际C0 | Selector |
|---|---:|---:|
| train PSNR | 31.074060440063477 | 31.102751541137696 |
| train L1 | 0.016422624699771404 | 0.0163764713332057 |

配套PSNR差为+0.0286911dB；此前+0.0042744dB来自另一份本地original，不是正式C0。`train.py:631`只选择索引5/10/15/20/25的五个训练相机，不能称406视角均值或新视角PSNR。这一很小差值不足以支持明显渲染收益。

### 下一步只核对输入，暂不调整4096

已证明初始点云、相机、邻居相同，但下载output没有全部训练图像、深度、置信图和对齐scale。`scene/dataset_readers.py:123–128`消费`estimated_depths`、`estimated_confs`及`trans.json`中的scale；`input.ply`相同不能代替这些身份检查。

建议下一批利用C0已有`dataset_sha256.txt`，只读核对新工作视图406原图+406深度npy+406置信npy+1个对齐trans.json，共1219文件。不用GPU、GT、依赖安装或baseline重训。以下仅打印检查结果；待用户讨论该批次后在服务器执行。

```bash
python -B - <<'PY'
from pathlib import Path
from collections import Counter
import hashlib

baseline = Path('/root/autodl-tmp/ambisur_runs/Tool_Room/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z')
current = Path('/root/autodl-tmp/ambisur_work/data_views/second_run_selector/toolroom_r2_seed0_30k_v1/colmap_undistorted')
records = []
for line in (baseline / 'dataset_sha256.txt').read_text().splitlines():
    expected, relative = line.split(None, 1)
    relative = relative.removeprefix('./')
    if relative.startswith('images/'):
        group = 'images'
    elif relative.startswith('estimated_depths/') and relative.endswith('.npy'):
        group = 'depth'
    elif relative.startswith('estimated_confs/') and relative.endswith('.npy'):
        group = 'confidence'
    elif relative == 'sparse_da3_aligned/0/trans.json':
        group = 'alignment'
    else:
        continue
    records.append((group, relative, expected))
counts = Counter(group for group, _, _ in records)
assert counts == {'images': 406, 'depth': 406, 'confidence': 406, 'alignment': 1}, counts
assert len({relative for _, relative, _ in records}) == 1219
print('Selected:', dict(counts), flush=True)
missing, mismatched = [], []
for index, (_, relative, expected) in enumerate(records, 1):
    path = current / relative
    if not path.is_file():
        missing.append(relative)
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        mismatched.append((relative, expected, actual))
    if index % 200 == 0:
        print('Checked:', index, flush=True)
print('Total:', len(records), 'Missing:', len(missing), 'Mismatched:', len(mismatched))
for relative in missing[:40]:
    print('MISSING', relative)
for relative, expected, actual in mismatched[:40]:
    print('DIFFERENT', relative, 'C0=' + expected, 'current=' + actual)
raise SystemExit(1 if missing or mismatched else 0)
PY
```

全部匹配也只证明当前工作视图与C0记录一致；若软链接目标曾在训练后变化，不能追溯证明开训时逐字节相同。发现差异先查身份；全部相同再调查7001前轨迹，并讨论预算单变量实验。现有剂量推导支持预算假设，但不支持“4096就是主因”或“扩大必然修复上部”。不自动叠加梯度/VCR/DP模块。

诊断数值见独立分析目录的`diagnosis.json`、`raw_post_diagnosis.json`、`primitive_7000_30000_counts.json`、`early_training_log_comparison.json`、`worst_region_gt_visibility.json`、`analysis_receipt.json`。高度贡献与raw/post分解闭合误差小于1e-9mm；跨场景继续复用现有正式评价核心。
