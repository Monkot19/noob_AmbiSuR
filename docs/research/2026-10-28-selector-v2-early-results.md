# v2_early：首次正式几何评价

日期：2026-10-28。用户已下载early完整输出，本轮只做关键文件核对及复用既有正式评价，不改方法、不追加训练或评价协议。

## 输入核对

- 本地目录：`D:\research_Space\output\AmbiSuR_second_run\ScanNetpp\Tool_Room\selector_v2_early\selector_v2_early_16384_with_baseline_checkpoint_2026_10_28\a52ab904_seed0_b16384_until15000_30k_v1`。
- 1584个文件、1952775223字节；train.exitcode/extract.exitcode均0。30K Gaussian304139804字节、App14349字节、raw mesh661734141字节、post mesh602565606字节，均非空。post11564585顶点/22332426三角面，raw13031515顶点/23837151三角面；只是关键产物核对，没有全目录逐字节对照或全部图片/PTH完整性检查。
- 方法固定a52ab904、seed0、原生7K、预算16384、间隔100、截止15000；全日志精确79条重定位，7100..14900，累计325258次位置更新（非唯一高斯数）。原提取5m/0.005m/trunc4/cluster2、不启用depth_filter；cameras.json与完整V2完全一致。

## 正式评价执行

运行原`scripts/evaluate_mesh_geometry.py`及`scannetpp_surface_v1.json`，一次评价C0、V1-16384、完整V2、early。完整表面、同一坐标、100万面积样本/seed0、双向点到三角面距离、1/2/5cm严格阈值（主列5cm），无ICP/裁剪/距离截断。

- 评价代码HEAD5fde7cbe06bf385bb0635e03063d0e1e950f283a，启动git_dirty=false；真实CLI退出码0，耗时132.273029秒。
- 旧3模型全部指标及输入mesh SHA精确复现，协议和共同GT样本摘要相同：`efb1f2f330fdabdffa0d4db5f426b411eedfa670cb030b9727c9ffbed46212b9`。
- JSON/CSV共12行逐值一致。新结果目录：`D:\research_Space\output\AmbiSuR_second_run\ScanNetpp\Tool_Room\evaluation\selector_v2_early_native7k_seed0_30k_v1`，原正式结果未覆盖。

## 结果

| 模型 | Accuracy mm↓ | Completeness mm↓ | Surface Chamfer mm↓ | F1@5cm↑ | train PSNR dB↑ |
|---|---:|---:|---:|---:|---:|
| C0 | 35.777223 | 45.619172 | 40.698198 | 0.779955 | 31.072280 |
| V1－16384 | 35.413509 | 45.883338 | 40.648423 | 0.780480 | 30.840198 |
| 完整V2 | 34.813288 | 45.380178 | 40.096733 | 0.787254 | 30.691866 |
| v2_early | 34.679897 | 44.809032 | 39.744465 | 0.787739 | 30.850549 |

| 模型 | F1@1cm | F1@2cm | Precision@5cm | Recall@5cm |
|---|---:|---:|---:|---:|
| C0 | 0.287009 | 0.508091 | 0.793780 | 0.766603 |
| V1－16384 | 0.287483 | 0.511517 | 0.795626 | 0.765899 |
| 完整V2 | 0.288493 | 0.512983 | 0.802477 | 0.772597 |
| v2_early | 0.289611 | 0.514870 | 0.802540 | 0.773475 |

- early相对C0：Accuracy减少1.097326mm、Completeness减少0.810141mm、Chamfer减少0.953733mm（2.34343%）；F1@5cm增加0.778466个百分点；train PSNR低0.221731dB。
- early相对V1：Accuracy减少0.733612mm、Completeness减少1.074306mm、Chamfer减少0.903959mm（2.22385%）；F1@5cm增加0.725995个百分点；train PSNR高0.010351dB，仍非常接近。
- early相对完整V2：Accuracy减少0.133390mm、Completeness减少0.571146mm、Chamfer减少0.352268mm（0.87855%）；F1@5cm增加0.048591个百分点；train PSNR高0.158683dB。新增几何优势主要体现在平均完整性距离，不能把很小的F1增量夸大。
- 1cm阈值Recall0.289480略低于完整V2的0.289706，不能说每个指标全部提高；其余阈值的Precision/Recall/F1及两向平均距离更好。

## 归因边界：前15K不是逐条相同的训练轨迹

额外核对发现：完整V2与early前79次事件的迭代号相同，但moved计数79次均非逐条一致；操作合计分别323721、325258，首个7100分别3563、3574。说明两次实际训练在截止前已存在轨迹差异，不能宣称“完全相同前15K、只在15K之后分叉”。

已核对实际cfg_opts：全部旧优化参数完全相同，只有early新增candidate_relocation_until_iter=15000；cameras.json相同，run_identity均记录Python3.10.21/Torch2.7.1+cu128/CUDA12.8/RTX4090、git_dirty=false。同一原生checkpoint路径/原生renderer路径也已记录。此前已验证代码在截止前调度等价、selector/训练/优化器未修改。差异具体来源尚未查明，不将其直接认定为CUDA非确定性或代码错误；本轮未擅自补训练/重写恢复机制。

因此结论为：**本次单场景/seed0运行中，early是四组中整体几何指标最好的，训练PSNR也恢复到V1附近；结果支持保留early作为当前候选，但不足以严格证明全部改善由缩短窗口造成。** 连续C0与checkpoint恢复分支还保留既有恢复路径边界；不宣称统计显著、跨场景稳定或已达到发表级提升。

## 下一步待讨论

本轮尚未对early运行逐区域分解，不能直接说前次门旁墙/门/管道退化已被修复，或69号墙之外的收益已变广。若继续分析，应复用已存在的区域分析核心及相同GT分组，检查收益是否仍集中在墙69、主要负贡献区域是否改善，再决定是否做跨场景验证或修改方法。没有自动启动下一批训练、改变阈值/LR、GT区域保护或清理输出。
