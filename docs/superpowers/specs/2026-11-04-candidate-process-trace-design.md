# 既有7K到30K过程采集：只观察，不改方法

用户已批准为定位门面首次前鼓及后期纠正失败，从现有原生7K续训到30K，并允许增加debug入口。D目录隔离分支codex/selector-process-trace，既有6172bc45位置检查方法/参数保持原样，旧output与baseline只读。

## 三文件范围

| 文件 | 修改 |
|---|---|
| scripts/trace_candidate_training.py | 新的显式采集启动器，通过Python return observer观察原函数并用runpy运行原train.py；普通train.py启动完全不加载它 |
| scripts/evaluate_candidate_actions.py | 仅把原observe_selector的record构造抽成共享selection_record；schema1静态格式/含义不变 |
| tests/test_candidate_selector.py | 在原统一入口验证真实CPU producer→共享记录→trace、实际move子集、复制而非引用、RNG/Parameter保持、事件时序/退出恢复 |

不改train.py、arguments、生产selector、propagation、renderer、Gaussian执行器或优化器；无新loss/LR/门槛/人工mask。GT不上服务器采集路径，不额外backward。记录耗时可能改变浮点非确定轨迹，不声称GPU逐位相同。

## 采样与数据

- 7001记录初态；7100..14900每100轮记录79个实际重定位事件；15000..30000每1000轮记录后期标签/表面，共96个参考事件。普通非采样迭代仅跳过记录。
- 生成端：实际preview plane_depth、DA3 depth/normal、normal_valid、真实anchor索引/被测试索引与cost/confidence/roundtrip，以及最终proposal/valid；可用原propagate_depth离线精确重放，不再增加传播算法。
- 选择端：复用schema1 record，另存真实move_accepted/targets、fixednormal cost及共同支持；joint ALR accepted与实际move区别明确，后15K实际move为空。
- 执行端：原helper主渲染的plane_depth、depth_normal、rendered_unc和实际移动坐标；同一参考帧可比较移动前后表面，不把中心距离当表面收益。
- 固定诊断视角DSC06642：7001及每1000轮，额外只读no_grad前向保存depth/normal/RGB和明确A-only的mask。它不计算候选、不重定位、无额外反传；固定A-only probe不冒充该视角的联合ALR掩码。实际参考事件另存真实A并E的rendered_unc。
- 记录NPZ/JSON与清单，复用原NumPy写法/metadata。每轮独立文件，既有文件不覆盖；预估额外数百MB到约1–2GB，实际由压缩率决定，不能预先保证耗时/字节量。

## 明确边界

只读观察验证真实函数返回与实际后渲染；无梯度分项归因、无optimizer回滚/模拟动作、无额外baseline。若候选生成提前回退，仍记录实际空动作及已有字段，不伪造缺失目标。完整原7K恢复App/种子语义保持既有边界，不宣称checkpoint保存原RNG/相机队列。服务器不自动启动；完成后提供新view/output的固定提交命令。
