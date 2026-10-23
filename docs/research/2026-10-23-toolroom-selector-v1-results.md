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
