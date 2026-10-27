# 共享7K前缀：4096/16384正式几何结果

日期：2026-10-27。用户要求跳过服务器全目录逐字节对照，尽快清理并分析结果。本轮复用既有evaluate_mesh_geometry.py与scannetpp_surface_v1，不改方法、不加诊断框架、不重跑测试或训练。

## 执行与证据

- 评价代码：28e189057f667ea1b3f26fc46a255bca662aee62，启动时git_dirty=false；模型commit：baseline=d6f15c8，两候选=c3510b0。
- 正式评价退出码0，耗时75.44108089990914秒。三份输入SHA256与刚核验的归档清单相同，JSON/CSV共9行指标逐值一致。
- 沿用完整表面、100万面积样本/seed0、同一GT与坐标、点到三角面、1/2/5cm（主列5cm）、无ICP/裁剪/距离截断。
- 本地输出：

```text
D:\research_Space\output\AmbiSuR_second_run\ScanNetpp\Tool_Room\evaluation\shared_native7k_seed0_30k_v1
  results.json
  metrics.csv
  resolved_config.json
  run_identity.json
```

GT采样摘要为efb1f2f330fdabdffa0d4db5f426b411eedfa670cb030b9727c9ffbed46212b9。没有重提取mesh、修改本地原件或改变评价域。

## 主结果

| 模型 | Accuracy mm↓ | Completeness mm↓ | Surface Chamfer mm↓ | F1@5cm↑ | train PSNR dB↑ |
|---|---:|---:|---:|---:|---:|
| baseline | 35.777223 | 45.619172 | 40.698198 | 0.779955 | 31.072280 |
| b4096 | 36.042003 | 46.167268 | 41.104636 | 0.779669 | 30.822012 |
| b16384 | 35.413509 | 45.883338 | 40.648423 | 0.780480 | 30.840198 |

| 模型 | F1@1cm | F1@2cm | Precision@5cm | Recall@5cm |
|---|---:|---:|---:|---:|
| baseline | 0.287009 | 0.508091 | 0.793780 | 0.766603 |
| b4096 | 0.287853 | 0.509904 | 0.793475 | 0.766336 |
| b16384 | 0.287483 | 0.511517 | 0.795626 | 0.765899 |

train PSNR只来自既有训练报告的5个训练视角，不是held-out新视角质量。

## 能得出的结论

1. 4096相对新原生基线：Chamfer增加0.4064379473mm（+0.998663%），Accuracy与Completeness同时变差；F1@5cm下降0.028543个百分点；train PSNR下降0.250268dB。本轮没有整体收益。
2. 16384相对基线：Chamfer减少0.0497745031mm（-0.122301%），F1@5cm提高0.052471个百分点。Accuracy改善0.363714mm，但Completeness恶化0.264165mm；5cm Precision上升0.1846个百分点、Recall下降0.0704个百分点。改善主要体现在准确性，完整性没有同时改善，整体非常接近基线。
3. 16384比4096的Chamfer低0.4562124504mm，说明本次预算增大后结果更好，但不能由此宣布4096不足是根本原因或16K已解决问题。4倍预算尚未带来明确的整体重建收益；train PSNR仍低于基线0.232082dB。
4. 两候选从相同原生7K checkpoint接续，使两预算对照减少此前前7K轨迹差异的混杂。原生checkpoint不含RNG/相机队列/App优化器，仍不能把相对连续baseline的小差异全归因于候选模块。
5. 仅Tool Room/seed0且未做重采样置信区间，不声称差异具有统计显著性。此次结果不支持继续只靠增大预算追求收益。

下一批建议先利用三组已有同视角渲染/法线与mesh定位具体改善及缺失位置，再讨论候选补选与ALR动作是否匹配；不自动追加训练、梯度路由或可靠性体系。本轮没有开展新空间诊断。

## 服务器清理（用户已授权跳过全量哈希）

第一批只针对新nativebaseline/b4096/b16384的raw tsdf_fusion.ply、train/ours_30000、debug、pg_view，约3.13GB；保留共享7K checkpoint/App、7K与30K Gaussian、各组30K App、post mesh、配置日志、数据/先验/权重/环境。

命令已语法核对并保存在：

```text
D:\research_Space\output\AmbiSuR_second_run\ScanNetpp\Tool_Room\analysis\shared_native7k_archive_20261027_v1\server_cleanup_stage1.sh
```

Agent无服务器连接，命令由用户执行；未收到实际删除回执，不记录为已释放空间。原本地归档与全文件manifest保持原样。服务器哈希对照不再作为清理或结果分析门槛。
