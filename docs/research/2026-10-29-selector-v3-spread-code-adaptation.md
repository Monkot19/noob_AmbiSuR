# selector_v3_spread：供体源码到宿主接口的适配提议

状态：2026-10-29用户选择先做传播候选深度，指定名称selector_v3_spread，并要求代码修改前口头汇报。本文件是具体待审提议，不是实施授权。此次仅阅读/规划，未修改算法、创建训练分支或启动实验。

## 1. 已核对的起点

- 主目录D:/research_Space/noob_AmbiSuR，当前codex/selector-static-audit，起点8d2cab7。train.py、candidate_selector.py、arguments、renderer、GaussianModel、run_metadata与已运行early的a52ab904生产逻辑相同；后续静态审计增加独立入口及测试，不含一周目算法。
- 固定供体DP-GS为dc1ce835ea3c6bb7887333c50df7a13497a45c09；重新阅读train.py70—81、145—335、561—567，graphics_utils.py212—254，scene/cameras.py83—100。不是只读README。
- 本模块仅改变额外候选的位置来源；固定法线NCC、代价曲线、梯度融合、学习率调整均留待另一消融。建议训练对照采用现有v2_early，保持相同7K前缀、预算16384、重定位间隔100和截止15000，训至30K；这些具体设置仍待本次审阅确认。

## 2. 供体真正做了什么

| 供体真实位置 | 源码语义 | 宿主适配 |
|---|---|---|
| train145—154 | 取当前渲染plane_depth，gt_normal/edge是外部估计先验 | 取重定位前当前模型深度；法线沿用本项目DA3深度导出法线，不下载新法线模型 |
| train189—227 | 单邻居渲染、几何往返、exp(-pixel_error)，置信图初始0 | 锚点用已有第一支持视角，几何阈值复用现有1px；最终候选仍原来的两个支持视角 |
| train234—292 | 几何可用像素中最多sample_num个，逐RGB patch NCC | 上限复用multi_view_sample_num=102400；与Gaussian候选预算16384分开，不把SH标签当锚点正确性 |
| train291—305 | c=mean_RGB((2-cost)/2)×exp(-error)，c>0.6为起始锚点 | 采用相同置信定义；不是mean后的标量训练loss，也不把DA3 confidence冒充此c |
| cameras83—97 | 5×5邻域与中心法线cos均>0.99时，normal_edge=1 | 沿用此中心邻域门控，不能声称它认证物体边界；不增加SAM/语义mask |
| graphics_utils212—254、train309—331 | 邻居法线计算深度传递系数；5×5、10次；mask阈值0.01 | 保留传播/平均/更新mask结构，只有射线坐标及无效值适配，详见下文 |
| train333—335 | 传播结果作深度loss，不直接改Gaussian中心 | 不移植此loss；传播结果作候选输入，再走已有NCC选择和重定位，这是本项目适配 |

第一支持视角来自当前已冻结邻居顺序；供体每轮随机选一个邻居，我们不再动宿主训练相机/RNG队列，固定用已有第一邻居。这是可复现的支持视角适配，不宣称与供体逐步等价。候选比较的两个支持视角、min NCC、0.5阈值仍保留。

## 3. 建议的具体数据流

1. `_render_training_reference`仍先准备原DA3 `prior_depth`及`prior_normal`。
2. spread启用且selector有效时，以no_grad取得重定位前的参考plane_depth、rendered_normal、rendered_distance；源深度仍由现有callback取得。
3. 在完整像素网格做几何往返筛选，再用局部、独立命名的随机流，从有效池最多抽102400像素做RGB-NCC。只给实际匹配通过的像素写入起始锚点，其他置信为0。不从~A限定锚点来源：锚点与待修正Gaussian是不同用途。
4. 原DA3法线、当前模型深度、起始锚点进入一次5×5/10步传播，返回`proposal_depth[H,W]`和`proposal_valid[H,W]`。
5. 原selector仍在~A中抽16384候选；当前位置解释不变，另一解释的位置改采proposal_depth，朝向仍原DA3 prior_normal。仍执行联合NCC比较，本轮不加固定法线的第二步。
6. 传播无效区域不补原始DA3位置候选；本次不产生E，保留原A路径。有效域边缘的双线性采样要求参与插值的像素都有效，避免把0与有效深度混成目标。
7. 得到E后仍标签A∪E→原mask渲染→原ALR；重定位事件仍只移动E并复用现有局部Adam/stat处理。主渲染在移动后照常执行。

锚点的NCC平面使用同一次预渲染的rendered_normal/rendered_distance，其位置为plane_depth×相机射线；不能把个体Gaussian中心的旧NCC代价贴到混合渲染深度上，假定两种几何已对应。共享_source_cost处理的是一般给定平面，因此可以复用计算而保持这一区别。

推荐selector有效的每次调用都采用spread来源（默认7001—30000），重定位仍仅7100—14900共79次；15000后继续由传播候选产生E但不移动。若只在79次移动事件生成传播目标、其他轮次仍用DA3，会变成时变混合目标来源，首版不采用这个省算方案。具体时间开销需实际GPU检查，不能由供体稀疏场景时间直接推断。

此设计是新的候选来源消融；E集合自然会变化，不能声称ALR监督像素或总剂量逐位不变。保留的是其损失公式、先验目标、权重、schedule及标签接口。

## 4. 必须明确的坐标、传播与回退约定

### 射线与法线

本项目的plane_depth为相机z-depth，使用现有整数像素K^-1射线，法线为参考相机坐标；`render_normal`虽然调用名为depth2point_world的函数，实际返回的是camera XYZ，不能按函数名误做世界坐标转换。最终世界目标继续使用原selector变换。

建议传播系数采用宿主真实射线：

\[
w_{i\leftarrow j}=\frac{n_j^T r_j}{n_j^T r_i},\qquad
d_{i\leftarrow j}=w_{i\leftarrow j}d_j.
\]

注意法线来自邻居j，与供体张量展开方式对应。DP-GS get_pos实际采用arctan及像素+0.5，不与宿主整数K^-1射线相同；不直接复制该函数。选用宿主射线是本次提议中的显式坐标适配，待用户确认；不同时新增两个坐标模式或调参开关。

### 传播更新

保持供体“起始锚点→有效邻居传递→平均→更新mask”的顺序，normal cos门槛0.99、mask更新门槛0.01、5×5/10步首版固定。供体并不在每步重新写回初始锚点深度，本版也不增加硬冻结锚点、阻尼、深度融合权重或额外平滑loss。

对于几何不成立的分母、负传递深度或非有限项，按宿主有效性约定取消这条传递；不沿用供体将NaN/Inf系数替换为1后继续传播的行为。有效邻居的平均规则与正深度传递保持；零分母/没有可用邻居时返回无效域，不编造目标。这些是必要的输入/坐标合同适配，要有小型真实几何测试，不包装成另一个创新。

5×5法线相似门控仅为供体启发式：法线相似的不同物体仍可能接近，法线噪声也可能导致传播范围不足。本轮不另加边界检测系统，先通过锚点数/传播有效像素数了解是否实际起作用。

### 先验和回退

- `prior_depth`、`prior_normal`继续指原DA3输入。当前train.py的直接深度loss与ALR都会读取它们，绝不覆盖成proposal。
- proposal只决定候选局部表面位置与后续重定位目标；不从proposal重新导出prior_normal，否则同时改变两项先验。
- 无锚点、传播未覆盖、normal/depth无效时，仅取消新增E/重定位，原A和原有训练继续；不强制使用已知存在偏差的原DA3位置作为fallback。
- spread关闭时，旧V1/V2/early默认函数、返回结构、候选采样流与训练RNG均保持；不能仅靠开关名声称等价。

## 5. 拟修改的文件、接口及复用方式

| 文件 | 具体提议 | 为什么需要 |
|---|---|---|
| `utils/depth_spread.py`，唯一新增公共模块 | `propagate_depth(...)`纯张量传播；`build_spread_proposal(...)`准备锚点并调用它 | 训练和静态采集共享同一实现，不复制场景版传播器 |
| `utils/candidate_selector.py` | 提取`_roundtrip_geometry`给现有_source_cost和锚点复用；同一_source_cost增加RGB patch支持，原gray调用保持；现有select接口追加可选proposal_depth/proposal_valid；原采样函数追加有旧默认值的stream参数供锚点复用 | 不另写几何投影/NCC/随机采样；显式区分法线原始深度与位置来源 |
| `train.py` | 只在_render_training_reference接spread准备；当前调用内源深度缓存复用；原ALR及深度loss不改；原selector日志增加锚点/有效域计数 | 保持移动前生成证据、移动后主渲染的执行顺序 |
| `arguments/__init__.py` | 一个`candidate_depth_source`参数，默认da3，可选spread；旧opt缺字段按da3 | 新旧方法复用同一训练入口、保存配置直接记录来源 |
| `scripts/evaluate_candidate_actions.py` | 给现有capture增加同一个候选来源覆盖项，调用真实build_spread_proposal，传入observer；score/GT/BVH/表格统计不动 | 用已有入口检查真实新目标，不造新的评测脚本 |
| `tests/test_candidate_selector.py` | 在原测试入口增加传播/接口真实链路用例，既有回归复跑 | 无另起一套测试框架 |

可选新增一个小型实验配置，供静态capture选择spread，继承原view/seed/预算/快照字段；不是复制一份evaluator。没有train_v3.py、scene_spread.py、renderer修改、新CUDA编译、新权重或新checkpoint字段。run_metadata按namespace自动保存新参数，无需改schema。

建议具体接口（名字和参数以本次审阅后规格为准）：

```python
# utils/depth_spread.py
propagate_depth(model_depth, prior_normal, rays, anchor_mask, normal_valid,
                *, patch_size=5, steps=10,
                normal_cos=0.99, mask_threshold=0.01)
# -> proposal_depth [H,W], proposal_valid [H,W]

build_spread_proposal(gaussians, reference, supports, prior_depth, prior_normal,
                      reference_geometry, opt, *, seed, iteration,
                      support_depth_fn)
# -> proposal_depth [H,W], proposal_valid [H,W], small_stats

# existing selector entry points retain original positional arguments/results
select_candidate_labels_with_targets(...,
    proposal_depth=None, proposal_valid=None)
# Original prior_depth remains available for prior-normal stencil validity.
```

供体传播常数作为模块具名配置固定在commit，不首版开放成调参面板；102400锚点上限、7×7 NCC patch、1px几何门槛则复用宿主已有opt。纯传播的5×5与NCC的7×7不是同一个patch参数，不能混写。

proposal_depth/proposal_valid必须同时提供或都省略；旧省略路径完全使用原DA3位置。提供时原prior_depth只用于原法线/stencil合同，proposal有效性单独判断。原采样默认stream保持`ambisur-selector-v1`，锚点用新stream，不改候选ID抽样序列或训练全局RNG。

spread无锚点也要返回全false的proposal_valid及有限占位深度，不能返回None：None在兼容接口中表示旧DA3路径，否则会误把“无传播证据”变成继续raw DA3重定位。

现有capture中的prior_world将继续记录实际被评估/返回的位置目标，来源写入元数据；confidence列仍是DA3 confidence，不改名冒称传播置信。传播锚点置信是内部RGB/几何量，只报告必要计数，不混入旧conf分层。score及旧NPZ记录保持可读。

### 渲染与匹配复用的实际边界

当前selector在主render前重定位，而基线多视角loss在主render后计算，不能直接拿后者标量/图当本轮移动前证据。也不能在旧有带梯度render后原地改xyz再继续backward。

因此spread新增一次参考no_grad几何预渲染；源深度callback在单次调用内缓存，锚点与最终两支持候选共用，避免第一源重复渲染。缓存只在本次selector内生效，不能跨移动或训练轮次复用旧几何。主render保持原参数；关闭spread时不增加预渲染/缓存/随机数消耗。

锚点RGB与现有gray最终候选共用patch_warp/lncc/往返函数；若ncc_scale非1，需要将RGB显式适配至对应NCC分辨率并验证K/坐标，不能假定get_image()[0]已经是灰度图尺寸。本项目实际默认1也不能成为接口省略理由。

## 6. 最小验证和后续执行边界

- 不在本轮写实现或测试代码；先交付口头方案，用户确认后才建`codex/selector-v3-spread`隔离分支，D目录继续为主，旧baseline/V1/V2/early分支和outputs保留。
- 只沿用现有单元测试入口，补已知倾斜平面传递、无锚点/无效域/normal边界、旧方法关闭等价和RNG、真实传播producer→selector→世界target、训练helper仍把原DA3交给ALR/深度loss、capture共享同一producer。CPU可验证几何/接口，不冒称真实GPU renderer通过。
- 原生GPU验证复用现有集成入口，不另写多轮smoke框架；具体上卡和运行范围另行讨论。本地缺原生扩展的事实不改变。
- 先复用已有静态capture/score看目标是否实际可用，再决定30K；不新增GT训练缓存、可靠性状态、图像诊断体系或重训baseline。
- 日志仅增加起始锚点数、传播有效像素数/可用候选数，继续原sampled/comparable/added/moved/iteration_ms；未测量前不报时间或显存保证。

## 7. 本次请用户审阅的具体取舍

推荐组合：现有early作为对照；模型渲染深度锚点＋DA3法线；单个第一支持视角做锚点RGB置信、上限102400；供体5×5/10步/0.6/0.99/0.01；宿主K^-1射线；无有效传播只保留A；selector有效时均使用传播来源，原15000重定位截止保留。所有项目均为本次提议，不能把用户对模块名称/方向的同意当作对这些细节的批准。

不建议同时采用的替代：原样复制供体arctan坐标会混淆宿主深度含义；把传播当新loss改变了用户选定的候选生成路线；仅79次事件替换来源会引入时变混合；继续raw DA3位置fallback会模糊本次目标来源消融。若用户偏好这些替代，应先改规格，不静默选择。

独立窄范围可行性审查：无Critical/Important阻断。核对实际DP-GS三文件、宿主混合渲染平面、原DA3两处loss消费者、proposal接口/observer透传与共享计算边界；确认未混入固定法线NCC。未运行测试/训练/GT，也未验证GPU成本或收益。最终主审另行检查生产六文件与early相同、供体三文件与固定commit相同、仅文档发生变动。
