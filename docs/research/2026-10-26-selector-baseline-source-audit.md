# 二周目候选模块：基线来源与逐行差异审计

日期：2026-10-26。用户要求：确认实现是否直接基于原始baseline，而非沿用一周目训练代码。此次只审计、记录和运行既有本地测试，不改生产代码或启动实验。

## 结论与精确定义

**二周目训练提交直接建立在C0 baseline上；没有把一周目的Core/可靠性/五状态/拓扑追踪/CUDA证据实现继承进来。** 已逐行阅读C0→训练版本的全部生产代码差异、新增候选文件259行及metadata文件93行；未修改的文件通过Git blob全量核对，无需靠抽查推断。

这里的“原始”需要分清两层：C0不是官方文件的逐字节副本，包含中文注释、说明文档和编辑器设置；但对比官方源码后，8个不同Python源文件AST相同，4个不同C++/CUDA源文件去注释后的token序列相同，没有发现C0中的训练算法改动。因此不把“C0含审计注释”称为“一周目算法已经混入”。

候选算法本身确实从此前已批准的候选实现原样移植，seed/日志工具也复用必要实现；这不等于复用一周目的整套训练路径。`utils/candidate_selector.py`在旧归档和新训练提交中的blob均为`222e7a5c635b2d273f31ef8bc9f56069199a0dd9`，SHA256为`f5c4c2a51c496d2ab6eb4203c8e0745318d30cc352cd567d63e2cf3292deefeb`。

## 提交关系与实际训练身份

```text
d6f15c8891a53800d5e3100f95817a7dd7f98e2f  c0-baseline / origin main
  └─ d979c070925a9dc9a5f5ce3a5563de56614fa192  候选模块干净移植
      └─ 3d85faf95951b4806eee18498841e66551b93320  实际30K训练版本
          └─ 后续文档、目录接管与独立mesh evaluator
```

- `git show -s --format=%P d979c07`直接返回d6f15c8。
- `git merge-base --is-ancestor 62e9117 3d85faf`返回1；一周目末端不在二周目训练祖先中。二周目与旧归档的共同祖先为d6f15c8。
- 下载run的`run_identity.json`记录训练commit为3d85faf、dirty=false、seed0；实际argv只有原参数加候选开关/预算，没有旧Core参数。
- `launch.txt`记录独立扩展目标为`/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python`。此前用户已提供固定C0源码编译、路径/导出检查及服务器23项测试零跳过的回执。此次没有连接服务器或重新验证GPU二进制，不能将历史回执冒称本轮现场检查。

## C0与官方源代码的全量关系

官方仓库`https://github.com/Fictionarry/AmbiSuR.git`的main于本次通过`git ls-remote`核实为`88d64054f53e0eba9ce49198282edf4a67fc8ca8`，与本地upstream/main相同。

以下是官方→C0全部存在文本差异的生产代码文件，非抽样：

| 文件 | 验证 |
|---|---|
| arguments/__init__.py | Python AST相同 |
| gaussian_renderer/__init__.py | Python AST相同 |
| mesh_extract/extract_general.py | Python AST相同 |
| scene/__init__.py | Python AST相同 |
| scene/gaussian_model.py | Python AST相同 |
| train.py | Python AST相同 |
| utils/graphics_utils.py | Python AST相同 |
| submodules/diff-plane-rasterization-ambisur/diff_plane_rasterization_ambisur/__init__.py | Python AST相同 |
| 同一renderer内cuda_rasterizer/forward.cu | 非注释token相同，逐段diff核对 |
| 同一renderer内cuda_rasterizer/backward.cu | 非注释token相同，逐段diff核对 |
| 同一renderer内ext.cpp | 非注释token相同，逐段diff核对 |
| 同一renderer内rasterize_points.cu | 非注释token相同，逐段diff核对 |

官方文件没有被C0删除；C0没有额外新增Python/native算法源文件。其余新增项为文档/规则/进展与编辑器设置，不进入训练导入路径。Ray-Color、Primitive Truncation、ALR参数管理等代码在官方版本就存在，并非二周目偷偷带回一周目Supporting。

## C0→二周目：所有生产改动逐项核对

C0共有1689个已跟踪文件，训练版本保留全部1689个，其中1680个blob完全相同；9个被修改的旧文件中，4个是下面的生产文件，另外5个是AGENTS/handoff/根目录三个记录文件。

| 文件/当前行 | 逐行核对内容 | 对原路径的影响 |
|---|---|---|
| arguments/__init__.py:86、146–147 | 新增seed0、默认关闭开关、预算4096 | 原学习率、损失权重、SH阈值、增密/裁剪调度全未改 |
| utils/general_utils.py:121–143 | safe_state增加seed入参，三个固定0改成该入参 | 实际seed0与C0一致；不将顶层setup_seed(22)误当训练seed |
| gaussian_renderer/__init__.py:22、40–52 | 导入公共标签函数、增加可选ambiguity_labels与接口检查 | 默认None，旧调用仍可使用 |
| gaussian_renderer/__init__.py:167–168 | 第5通道SH标签改为公共函数或显式标签 | 原SH打分、分位阈值、低端上限/严格比较未改；布尔加法改为OR，同布尔真值；新增空模型处理与no_grad |
| train.py:22–23、89–124 | 新helper在启用后准备DA3、两个支持深度、候选标签，再作主渲染 | 7001前不建候选/不额外源渲染；活动后最多新增两个no_grad源前向 |
| train.py:125–133 | 原ALR公式提取为helper | residual、confidence条件、全图mean、权重与衰减公式逐项一致 |
| train.py:136–140 | 启动时校验支持模式，向logger传pipeline | 不计算旧Core，不改模型初始化 |
| train.py:227–236 | 当前视角nearest_id前两个临时传入；消费后删除属性 | 不重排、不改变原随机源抽样 |
| train.py:437–442 | 活动分支复用已对齐prior_depth | 插值模式/align_corners与原路径相同，不改变原深度loss |
| train.py:462–470 | 原SH标签或补选标签供unc_mask，调用原ALR公式 | 后续原参数管理、backward与屏幕梯度清零代码未修改 |
| train.py:504–512 | 每200轮输出计数与总标签权重/耗时 | 不新增训练loss或状态缓存 |
| train.py:590–611、695 | 写轻量metadata及显式seed传递 | 仍保留cfg_args/cfg_opts和原checkpoint；不导入CoreConfig |
| utils/candidate_selector.py:1–259（新增） | 全文件逐行阅读：SH标签、局部RNG、坐标/平面、源深度往返、NCC、回退与OR输出 | 与批准候选规格一致，no_grad输出临时标签；不含旧仲裁或生命周期 |
| utils/run_metadata.py:1–93（新增） | 全文件逐行阅读：JSON/路径转换、Git身份、Python/Torch环境查询、写两个JSON | 不操作Gaussian/optimizer/梯度；无随机抽样。含CUDA可用性/设备信息查询，不能描述为完全没有运行时操作 |

除此没有其他训练生产代码改动。`tests/__init__.py`为空，`tests/test_candidate_selector.py`为统一测试入口；研究来源manifest和报告不被训练导入。训练后新增的mesh evaluator/配置/测试用于离线评价，不在训练调用链中。

## 未带回的一周目代码与本地缓存

- 受保护的scene、submodules、mesh_extract、multi_view_priors、lpipsPyTorch、公共geometry/loss及环境声明共1630个文件，与C0逐字节相同。Gaussian优化器、初始化、clone/split/prune、参数/Adam迁移与原CUDA接口没有继承一周目改动。
- 一周目归档中的24个`reliability/*.py`均不在当前Git树；旧D0/G1诊断/GPU runner没有移入本次训练路径。
- 从train/renderer/参数/新增模块出发，静态本地源码import闭包为20个模块，没有reliability。候选文件的`_CORE_FLAGS`只是拒绝旧开关的字符串名单：发现旧开关则报错，不实例化或运行旧Core。
- D盘仍有138个Git忽略的代码类文件，全部为`.pyc`；没有忽略的`.py/.so/.pyd/.cu/.cpp/.h`。其中旧reliability目录仅缓存不等于旧源码恢复；标准PathFinder对旧runtime/config/arbitration子模块解析均为None，当前调用链也无相应import。没有擅自删除这些缓存；它们不是服务器Python3.10训练源码。
- 当前8个selector生产/测试文件blob与实际训练3d85faf完全相同。此次只新增审计文档，不改生产文件。

## 验证、边界与后续状态

本轮重新运行已有`python -B -m unittest tests.test_candidate_selector -v`：23项中18通过、5项因本地无AmbiSuR原生扩展跳过，测试运行1.438s；不声称本轮GPU通过。此前AutoDL23/23是历史回执。无需重建测试体系或重训baseline。

另需限定现有测试证明范围：`test_real_training_helper_off_ezero_and_active_path`比较helper与当前renderer的直接调用，并非分别启动官方train.py与改版train.py完整跑7000轮；它不能证明整段训练逐位相同。本轮AST/blob/调用链审计证明来源与改动边界，不能据此宣称早期轨迹差异已解释、候选算法无bug或重建一定有益。

输入方面，用户本轮回传406图像+406深度+406置信+1个对齐文件均匹配C0（1219/0/0）。结合源码结果，没有发现“一周目Core残留”或当前输入内容不一致的证据。7001前的分叉原因仍未确定，不自动宣布CUDA非确定性，更不把4096当成已确认根因。本次在源码审计处结束；下一批调查/实验继续按用户要求先讨论。

机器可读证据保存在独立归档：`D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/selector_v1_2026_10_23/analysis/toolroom_pair_seed0_30k_diagnosis_v1/source_audit_20261026.json`。不读取GT，不改原训练/评价output。
