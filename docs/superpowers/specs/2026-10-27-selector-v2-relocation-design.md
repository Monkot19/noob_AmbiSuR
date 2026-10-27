# selector_V2：V1 + 重定位

日期：2026-10-27。状态：**已按用户授权在codex/second-run-selector-v2实现，固定代码a114ed5；统一30项24CPU通过/6原生跳过，独立审查无待修问题。服务器训练尚未启动。** [源码适配与收据](../../research/2026-10-27-selector-v2-code-adaptation.md)、[实施计划](../plans/2026-10-27-selector-v2-implementation.md)、[AutoDL直接30K指南](../../research/selector-v2-native7k-launch.md)。

## 1. 意图与范围

检验：V1找到的先验替代平面，除了扩展ALR标签外，若实际落实其位置目标，能否改善最终重建。正式名称为`selector_V2`，定义`V2 = V1 + 重定位`。保留V1，新增独立开关；不把V1改名成V2，不将任何效果保证写入方法主张。

用户选择了“保留完整平面比较，增加位置动作，再接原ALR”；“固定中心、只换先验法线进行比较”保留为备选，不在本轮实现。新同恢复selector-off对照暂缓，继续复用现有原生7K checkpoint和30K baseline输出。

## 2. 保持的V1合同

沿用[已批准V1规格](2026-10-21-candidate-selector-design.md)：原SH标签A、从A未命中池抽样、固定两个支持视角、G/P完整平面、原NCC/best-source判定与全部有效性/失效回退，标签E当前迭代有效，ALR使用A并E。候选与基线的全部loss、权重、学习率、原有拓扑调度、初始化、Ray-Color及渲染器保持原算法。

V2首轮实验显式设置预算16384、seed0、r2、DA3、总迭代30000，从现有原生7K checkpoint续训7001—30000。不修改V1的全局默认预算4096。GT、数据集及已有baseline/实验输出只读。

## 3. 待确认的执行细节（本次提议）

| 项目 | 建议合同 |
|---|---|
| 开关 | `enable_candidate_relocation=False`；开启时必须同时开启V1，沿用DA3-only条件。关闭时等于现有V1路径，不产生目标坐标或位置写入 |
| 重定位间隔 | `candidate_relocation_interval=100`；首轮固定100，不同时调参 |
| 启停 | V1已激活，`iteration % interval == 0`且`iteration < iterations`；当前默认7100、7200、…、29900。**覆盖整个启用阶段是适配提议，供体源码将SGU放在其densification分支内；不声称供体验证过我们的30K窗口。** |
| 目标 | 只对本轮补选E中的高斯，使用本次NCC实际评估过的同一DA3候选位置，全量替换中心；不插值、不另加步长/置信门或新loss |
| 顺序 | 原相机/背景选择 → 原V1比较得到A/E及候选位置 → 事件轮立即重定位 → 新主参考渲染（标签A并E）→ 原ALR/其他loss/反传/拓扑/step。源深度前向用于移动前的比较，主loss所需可导渲染按原路径重新执行 |
| 位置写入范围 | 只改E行；不改变点数、Parameter对象身份、旋转/尺度/opacity/SH；不沿用穿过densify/prune的旧索引，不建立跨迭代候选缓存 |
| Adam | 若xyz状态已存在，仅清零E行`exp_avg`、`exp_avg_sq`；全局`step`、参数组LR、非E行和其他参数组状态保留。没有状态则不人为创建。保留全局step符合原增点时零初始行矩的方式，不宣称是逐行重启完整Adam |
| 原增密统计 | 仅清零移动行已有的`xyz_gradient_accum`、`xyz_gradient_accum_abs`、`denom`、`denom_abs`、`max_radii2D`、`max_weight`，随后按移动后的本轮渲染正常累计，避免旧位置统计参与新位置增密；不改变原增密规则 |

以参考相机行向量接口为准：`X_P = D_P * K_r^{-1}[u,v,1]^T`，`X_world = (X_P - world_view_transform[3,:3]) @ world_view_transform[:3,:3].T`。直接复用已通过选择的X_P，不重新采样深度、不再使用另一份归一化坐标。空E或有效性失败时不移动；正常缺证据仍走V1回退，代码/模型异常沿V1明确报错。

## 4. 依据与不能提前声称的结论

[Sparse2DGS §3.4及Table5](https://arxiv.org/html/2504.20378v1)支持比较后直接更新位置、每100步执行的机制；[固定源码update_points](https://github.com/Wuuu3511/Sparse2DGS/blob/af15587eb186ca7ab12a2cf795775fa0f473f25d/scene/gaussian_model.py#L476)实际只替换xyz。供体目标来自渲染表面，本方案来自DA3，是适配而非原样复现。供体整块替换Parameter的优化器副作用不照搬；上述局部状态处理是本次明确提出的工程适配。

ALR本身可通过深度法线影响位置；新增价值是显式中心重定位，并非“ALR完全没有位置梯度”。联合平面NCC胜出不保证位置/法线分别正确，重定位加ALR也不保证一步实现完整胜出平面。最终是否有用以相同正式mesh协议衡量。

## 5. 最小实现、Git与验证范围

- 确认本文后，在D盘从当前二周目提交建立`codex/second-run-selector-v2`；保留原V1分支和历史，C盘临时worktree不动。基线分支/源码不改。
- 复用`utils/candidate_selector.py`的唯一选择核心。V1三元组接口保留；V2事件轮额外获取通过候选的ID和同一次比较的目标位置，不复制NCC/投影实现。
- `scene/gaussian_model.py`增加局部位置/状态更新入口，`train.py`接入上述顺序，`arguments/__init__.py`增加开关/间隔；不改renderer/CUDA、checkpoint格式或评价器。
- 复用`tests/test_candidate_selector.py`，仅补坐标转换、真实选择producer到重定位consumer、未选行/其他参数/Adam状态保持、无移动回退与调用时序的新增边界测试。运行现有相关CPU回归；本地原生扩展缺失如实标注，不搭新诊断框架、不追加独立短步GPU实验。
- 复用现有metadata与日志，仅在原selector日志增加重定位计数；不记录逐高斯历史或额外GT标签。代码及新增边界验证后提交并推送固定commit，再交付AutoDL直接至30K命令；不在服务器临时改代码。
- 新实验另建`selector_v2`输出；复用既有mesh提取和正式evaluator，比较当前baseline、V1-16384、V2。指标仍为Accuracy/Completeness/Surface Chamfer及1/2/5cm PRF（5cm主列），训练视角PSNR保留其原含义。

## 6. 结合真实源码补充（用户审阅要求）

具体函数、行位置、张量接口及供体差异见[源码适配清单](../../research/2026-10-27-selector-v2-code-adaptation.md)。其中明确：供体用SGU替代ADC，而本轮保留V1的ADC；不能拷贝三视图repeat分组；必须成对返回ids[accepted]/points_p[accepted]，不以extra.nonzero重排；位置/动量/stat使用原地索引写入，保留当前Parameter。主渲染前接入无需新增渲染。日志细化为事件轮加一行moved计数，以记录原200轮日志未覆盖的7100/7300等事件；既有metadata自动记录新字段，无需修改serializer。

本规格及源码改法已获用户确认并实施；当前收据见顶部链接，不重复请求已给出的实施授权。
