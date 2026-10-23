# 二周目正式mesh几何评价 v1

用户于2026-10-23要求可复用正式评价，并确认1/2/5cm，5cm为主列；落实此前整表面、面积采样、点到三角面方案，不修改训练。

## 范围与选择

一个公共实现 `scripts/evaluate_mesh_geometry.py`、一个测试入口 `tests/test_mesh_geometry.py` 和协议JSON；场景/mesh/模型commit/输出通过CLI提供。选择直接三角面距离，避免目标采样密度形成最近邻误差。不将DTU观测掩码/TnT专用crop/ICP套到ScanNet++；原脚本不变，协议名 `scannetpp_surface_v1`。

Accuracy=mean(d_recon_gt)，Completeness=mean(d_gt_recon)，Surface Chamfer=(两者)/2；距离不平方。Precision(t)=mean(d_recon_gt<t)，Recall(t)=mean(d_gt_recon<t)，F1=2PR/(P+R)，P=R=0时F1=0。PRF判定/调和平均与现有TnT函数一致，但不冒称完整官方协议。使用真实旧函数的小输入作数值对照，不移植旧框架。

## 冻结参数

- 每个表面按面积均匀采样1,000,000点，seed0，局部NumPy PCG64。GT只采一次供所有模型复用；重建采样流与模型顺序无关。
- 面积概率选面，sqrt重参数化均匀重心采样；分块面积/采样/查询，不整体展开上千万个三角形。
- 输入米，`unit_scale_to_m`支持显式单位转换；输出毫米和[0,1]PRF。阈值0.01/0.02/0.05米，主列0.05米，全部发布。
- 输入已在相同坐标系；完整表面、不裁剪、不ICP、不截断距离、不去连通块。零面积面移除并记录；空mesh/非有限坐标/非法索引报错。
- 沿用归档offline_g1的Open3D closest-point计算方式：Float32几何/查询，Float64采样/累计；不是全双精度求解。所有模型共同减去GT包围盒中心以改善数值条件，仅共同平移，记录原点，不拟合配准。

## 输入输出与验证

至少一个命名模型，每个显式模型commit；任意数量模型同一核心顺序处理，一次只持有一个重建mesh/BVH。记录解析协议、命令、评价代码Git身份、模型身份、路径/文件SHA256、mesh计数/面积、共同GT采样摘要和库版本；复用 `utils.run_metadata.write_run_metadata`。

仅新独立输出目录，已有目录拒绝覆盖；原始mesh只读，不保存巨型样本点云。JSON结构化指标及相对首模型的candidate-baseline差值，CSV每模型每阈值一行。没有results.json就不能称评价完成。

测试覆盖手算方向/严格阈值/零F1、面积比例/三角形内采样、真实Open3D同面/平移/缺面/面内点、真实PLY→CLI→既有serializer→JSON/CSV、输入字节不变/重复输出拒绝。缺Open3D明确native跳过，不冒称通过。

## 来源与结论边界

- AmbiSuR附录E.2：https://arxiv.org/html/2605.12494v1#A5.SS2
- `scripts/eval_dtu/eval.py`：双向均值；`scripts/tnt_eval/evaluation.py::get_f1_score_histo2`：PRF。
- Open3D：https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html
- ScanNet++坐标说明：https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation

这是正式Tool Room几何评价的明确适配协议，不是ScanNet++官方或DTU/TnT分数；不与论文表格横向混算。完整GT中不可见表面也计入Completeness。单场景seed0结果不替代Utility/多seed确认。
