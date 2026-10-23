# 正式mesh几何评价：共用入口与Tool Room首轮

## 协议与依据

入口：`python scripts/evaluate_mesh_geometry.py --help`。协议：`experiments/mesh_geometry/scannetpp_surface_v1.json`。
用户已确认同时报告1/2/5cm，5cm为主列；1M面积均匀采样/seed0，双方共用同批GT点，完整表面、共同原始坐标、无ICP/裁剪/距离截断。输入米，距离结果毫米；PRF为0–1，不是百分数。

| 输出 | 定义 | 趋势 |
|---|---|---|
| accuracy_mm | 重建表面采样点到GT三角面的平均距离 | 越低越好 |
| completeness_mm | GT表面采样点到重建三角面的平均距离 | 越低越好 |
| surface_chamfer_mm | 两个未平方均值的算术平均 | 越低越好 |
| precision | 重建→GT距离严格小于阈值的比例 | 越高越好 |
| recall | GT→重建距离严格小于阈值的比例 | 越高越好 |
| f1 | 2PR/(P+R)，二者全零时记0 | 越高越好 |

AmbiSuR的指标依据为[附录E.2](https://arxiv.org/html/2605.12494v1#A5.SS2)；本地 `scripts/eval_dtu/eval.py` 的双向距离、`scripts/tnt_eval/evaluation.py::get_f1_score_histo2` 的PRF定义。与后者的真实标量函数通过小输入核对；采样、点到面后端采用本协议，不等于完整DTU/TnT官方脚本。

[Open3D距离查询](https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html)支持非封闭三角网格的unsigned最近表面点。CPU Float32 BVH/查询、Float64采样/累计，所有模型共同减去GT包围盒中心以改善数值条件；仅共同平移，不拟合配准。

[ScanNet++文档](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)说明DSLR COLMAP与扫描对齐且为metric；输入仍须沿用同一坐标链，不能仅凭包围盒相近宣布配准正确。GT文件名0.05是简化到5%，不是评价阈值。

这是我们固定的正式ScanNet++适配评价，不是官方ScanNet++排行榜协议。DTU/TnT正式论文对照继续运行原有脚本，不使用此入口替代其观察掩码/crop/配准/点云采样。完整GT的未观察区域也计入完整性，必须据此解释结果。

## 验证与运行

依赖现有NumPy/Open3D；无需GPU、渲染器、训练或重新提取baseline。服务器先运行同一个测试入口（native项目必须零跳过）：

```bash
python -B -m unittest tests.test_mesh_geometry -v
```

参数示例（任意工作目录运行时使用脚本和config的绝对路径）：

```bash
python -u -B scripts/evaluate_mesh_geometry.py \
  --config experiments/mesh_geometry/scannetpp_surface_v1.json \
  --scene ScanNetpp/Tool_Room \
  --gt /root/autodl-tmp/ambisur_data/gt/ScanNetpp/Tool_Room/mesh_aligned_0.05.ply \
  --mesh baseline=/root/autodl-tmp/ambisur_runs/Tool_Room/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z/mesh/tsdf_fusion_post.ply \
  --mesh selector=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/3d85faf_seed0_30k_v1/mesh/tsdf_fusion_post.ply \
  --model-commit baseline=d6f15c8891a53800d5e3100f95817a7dd7f98e2f \
  --model-commit selector=3d85faf95951b4806eee18498841e66551b93320 \
  --output /root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/evaluation/toolroom_pair_seed0_30k_v1
```

发布前必须checkout交付的明确评价commit，并记录 `git rev-parse HEAD`、`git status --short` 和 `git diff --exit-code`。不要在服务器改代码；训练模型commit与评价代码commit分别记录。此文档示例本身不代表服务器已经执行。

输出目录必须不存在且位于输入mesh目录之外。失败后保留记录，查明原因后另行讨论重跑，不覆盖旧目录。CPU运行时间/内存取决于两份上千万顶点mesh和服务器CPU；脚本逐模型打印进度，GT/BVH复用，分块查询，不承诺未经实测的耗时。

## 结果与复用

- `resolved_config.json`：完整已解析协议/场景/命令/路径/模型commit。
- `run_identity.json`：评价代码Git commit/dirty状态、Python和NumPy/Open3D版本等，调用现有metadata writer。
- `metrics.csv`：模型×阈值，每行包含六项指标、主列标记及相对第一个模型的绝对差。
- `results.json`：完成后的结构化报告、两份mesh及GT的SHA256/计数/面积/包围盒、共同数值原点、GT采样点摘要、耗时和各模型差值。

差值统一为candidate-baseline：距离负值为改善，PRF正值为改善（0.01是1个百分点）。判断时同时看准确性和完整性，不用单独Precision或顶点数量代替总体几何。阈值全部报告，不看结果后选择阈值；未实现重采样置信区间，不把很小的Monte Carlo差值说成显著性。

换场景复用同一脚本/测试；提供不同 `--scene/--gt/--mesh/--model-commit/--output`。换单位使用显式协议配置 `unit_scale_to_m`，不自动猜测。任何评价域改变都另存版本化协议，禁止结果出来后静默裁剪。
