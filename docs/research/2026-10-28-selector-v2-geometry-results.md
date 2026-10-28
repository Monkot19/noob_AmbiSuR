# selector_V2：原生7K续训至30K的首次几何评价

日期：2026-10-28。用户已下载V2并提供本地路径；本轮只快速核对关键产物、运行既有正式mesh评价与整理结果，不改算法、训练或评价协议。

## 下载与执行核对

- 下载目录：`D:\research_Space\output\AmbiSuR_second_run\ScanNetpp\Tool_Room\selector_v2\selector_v2_16384_with_baseline_checkpoint_2026_10_28\a114ed5b_seed0_b16384_30k_v1`。
- 共1584个文件、1972550922字节；训练与提取退出码均0。30K Gaussian 303880396字节、App 14349字节、raw mesh 671865650字节、post mesh 609997588字节均非空；PLY头顶点/三角面数与日志一致。此为关键产物检查，没有进行全目录逐字节对照、全部图片解码或全部PTH完整性验证。
- 固定方法a114ed5、seed0、原生7K checkpoint、预算16384、重定位间隔100，原生renderer路径与原提取5m/0.005m/trunc4/cluster2/no depth filter均与批准设置一致。train耗时53:07，最终N1225318。
- 实际重定位事件恰好7100..29900，共229次，227次非零；累计1019840次位置更新，均值4453.449782/事件、最大8705。此为操作次数，不能解释为独立高斯覆盖数。与selector日志重合的114次中moved均等于added。

## 正式评价及可复用证据

复用`scripts/evaluate_mesh_geometry.py`与`experiments/mesh_geometry/scannetpp_surface_v1.json`：完整表面、100万面积样本、seed0、同一米制对齐坐标、双向点到三角面、1/2/5cm严格阈值（主列5cm），不进行ICP/裁剪/距离截断。本设置为声明的ScanNet++适配，不称作ScanNet++官方协议或与DTU/TnT官方评测完全相同。

- 模型commit：C0=d6f15c8891a53800d5e3100f95817a7dd7f98e2f，V1=c3510b029247e783d3163b2c8e7ce5d1f182039d，V2=a114ed5b5ee1c265eea5ececc98268934eca6b2e。
- 执行代码HEAD=47fbde284c4782166ebdc95fe066edb2d5c90c00；身份记录git_dirty=true，启动时差异仅4份交接/规划文档，评价代码与配置对HEAD无差异。没有隐藏或改写此身份记录。
- 真实CLI退出码0，耗时76.147670秒。结果JSON/CSV9行逐值核对一致。
- 本次C0/V1输入mesh SHA和全部指标与前次正式结果精确相同；协议及共同GT采样摘要亦相同：`efb1f2f330fdabdffa0d4db5f426b411eedfa670cb030b9727c9ffbed46212b9`。
- 新输出目录：`D:\research_Space\output\AmbiSuR_second_run\ScanNetpp\Tool_Room\evaluation\selector_v2_native7k_seed0_30k_v1`，含results.json、metrics.csv、resolved_config.json、run_identity.json。旧结果未覆盖。

## 结果

| 模型 | Accuracy mm↓ | Completeness mm↓ | Surface Chamfer mm↓ | F1@5cm↑ | train PSNR dB↑ |
|---|---:|---:|---:|---:|---:|
| 原生C0 | 35.777223 | 45.619172 | 40.698198 | 0.779955 | 31.072280 |
| V1－16384 | 35.413509 | 45.883338 | 40.648423 | 0.780480 | 30.840198 |
| V2－16384 | 34.813288 | 45.380178 | 40.096733 | 0.787254 | 30.691866 |

| 模型 | F1@1cm | F1@2cm | Precision@5cm | Recall@5cm |
|---|---:|---:|---:|---:|
| 原生C0 | 0.287009 | 0.508091 | 0.793780 | 0.766603 |
| V1－16384 | 0.287483 | 0.511517 | 0.795626 | 0.765899 |
| V2－16384 | 0.288493 | 0.512983 | 0.802477 | 0.772597 |

V2相对C0：Accuracy减少0.963936mm（2.69427%），Completeness减少0.238994mm（0.52389%），Chamfer减少0.601465mm（1.47787%）；5cm Precision/Recall/F1分别增加0.8697/0.5994/0.729875个百分点。train PSNR下降0.380414dB。

V2相对V1－16384：Accuracy减少0.600221mm，Completeness减少0.503159mm，Chamfer减少0.551690mm（1.35722%）；5cm F1增加0.677404个百分点。train PSNR下降0.148332dB。全部1/2/5cm的Precision/Recall/F1点估计均高于两对照。

## 结论与下一步边界

1. V2在此次Tool Room/seed0上获得较V1更一致的几何改善：准确性与完整性同时改善，5cm精确率和召回率同时改善。不能再把本次结论写为“只有新增候选或执行动作，尚无任何几何收益”。
2. 改善幅度仍有限，Chamfer相对C0约1.48%，不能称为显著突破、统计显著或跨场景有效。train PSNR继续下降；这是原训练报告的5个训练视角，不是held-out PSNR、SSIM或LPIPS。
3. V2与V1共用7K恢复路径，因此二者对照更直接地反映“添加重定位及其局部状态处理”这一整体模块。单次实验仍不提供统计稳定性，也不将累计位置更新解释为独立覆盖或证明某个位置一定正确。连续C0对恢复分支仍有既有RNG/相机队列等恢复限制，用户此前明确暂缓off对照，不将其重新变成前置任务。
4. 三组使用相同后处理及评价域。当前结果不能证明哪些区域改善、是否主要受post处理影响、训练PSNR为何下降，或更改学习率/频率是否有效；本轮未新增这些诊断。
5. 下一步先向用户讨论是否利用已有同视角渲染/mesh定位收益与损失，再决定是否调整重定位窗口/频率等；不自动加训练、改LR、升级可靠性体系或清理服务器。
