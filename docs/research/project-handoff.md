# AmbiSuR 二周目交接入口

> **当前入口（2026-10-23）：** 主工作目录已改为 `D:\research_Space\noob_AmbiSuR`，分支 `codex/second-run-selector`。C 盘临时 worktree 保留 detached `3775135`，后续开发不再以 C 为入口。完整归档与换机步骤见 [主目录与恢复指南](second-run-workspace-and-recovery.md)。下方一周目记录仅供查阅，不作为当前派工。

## 当前方法与代码

> **2026-10-26最新已批准派工：** 用户要求baseline零源码改动并同意原生checkpoint限制。新建`codex/baseline-30k-checkpoint`仅指向C0/d6f15c8；运行原train.py、r2、30000轮、内置seed0、`--checkpoint_iterations 7000`，不加保存模块/钩子，不做smoke。新output为`/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1`；独立privateview位于`data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted`。后续4096/16384读取相同原生7K模型/Adam状态；不承诺RNG/相机队列恢复。下方53K/单次16384/完整状态早停提议均被本次[批准规格](../superpowers/specs/2026-10-26-shared-7000-prefix.md)取代。先交付原生baseline启动命令，尚待服务器回执。

直接运行命令见[原生C0启动指南](native-c0-30k-launch.md)。基线branch已在本地创建并验证整树相同；不在其上添加文档commit。D盘规则/规格/命令记录只提交到二周目分支。不要在基线进程运行期间切换服务器代码版本。

> **原生baseline完成日志已收到：** d6f15c8/PID11962，30000/30000、50:21、Trainingcomplete，trainPSNR31.072280120849612、L10.016401121392846107、N1291587。尚待exitcode及7Kcheckpoint/App/30KPLY列表，不冒称产物已核验。二周目已作唯一生产行修复：从checkpoint所属run、按first_iter加载Appweights，而非新output或latest30K；Gaussian/Adam原恢复逻辑不变。真实方法CPU集成用例RED→GREEN，24项19pass/5native跳过，独立review通过。未改baseline、未新开GPU测试/训练。

**后续产物回执已确认：** exitcode0；chkpnt7000.pth=1074511797字节，7Kapp.pth=14349字节，30KPLY=320315108字节。原生基线训练及关键产物完成，尚无新baseline几何成绩。现可交付二周目固定修复版本的4096/16384原生checkpoint续训命令。

二周目恢复修复固定`f41dac0b47f5d00397e346ff6cedad0c107471ff`，顺序运行4096/16384各7001..30000命令见[配对续训指南](native-7k-selector-pair-launch.md)。新output父目录`/root/autodl-tmp/ambisur_runs/Tool_Room/selector_from_native7k/f41dac0b_seed0_30k_pair_v1`，各有b4096/b16384。只是在本地准备/发布启动命令，尚无服务器候选续训启动回执；不把完成的baseline和未启动的候选混写。

> **2026-10-26当前派工：** 用户已批准候选预算16384、其余不变、直接30K，不增加smoke/安全框架。训练仍固定3d85faf；新输出`/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/3d85faf_seed0_b16384_30k_v1`，新private data view含`toolroom_r2_seed0_b16384_30k_v1`。CLI参数实验无生产代码变化；尚待用户启动回执，详见首轮结果报告末节。输入1219/0/0与源码审计已完成，不重复派工。

> **启动前最新转向：** 用户提出共同7K起点及4096可比性问题，尚未发出启动命令。现有两run只有7K PLY，没有完整7K checkpoint；不能单独将16384从PLY续训当公平预算对照。当前待讨论单次探索性30K或共享7K前缀/两个23K分支（总53K步），不自动实现恢复或启动训练。

- 从 `c0-baseline`（`d6f15c8891a53800d5e3100f95817a7dd7f98e2f`）仅移植已批准候选模块：保留原 SH 双端标签，从未命中池抽样4096，比较当前平面与 DA3 平面的跨视角 NCC，补选后复用原标签渲染与 ALR。
- 当前规格：`docs/superpowers/specs/2026-10-21-candidate-selector-design.md`；移植记录：`docs/superpowers/plans/2026-10-22-clean-selector-migration.md`。
- 方法提交 `d979c07`；服务器验证／训练固定 `3d85faf95951b4806eee18498841e66551b93320`。本地接管起点 `3775135690b2ea4edfdc50e3b19fba467d02d117` 与它仅有文档差异；本次也只更新规则和文档，实际当前 HEAD 用 `git rev-parse HEAD` 读取。
- 旧 D 未提交内容保存为归档 `7b3154b7daa7e03ca7dad5b0e5cf1260f4e1099d`，不是当前二周目实现；不将旧 Core、五状态、梯度路由或 Utility 恢复自动合并／恢复。

## 已验证与首轮30K完成日志

- AutoDL RTX4090／`3d85faf` 统一23项测试全部通过、零跳过，包含真实 renderer／ALR backward／optimizer step。
- D 接管后同一入口23项中18项通过、5项原生集成因本地缺少 AmbiSuR 扩展而跳过；8个代码文件语法检查通过。1697个已跟踪文件齐全，192个忽略文件未改变。
- Tool Room 首轮30K已由用户实际启动：406张训练图像、r2、seed0、`eval=False`、DA3、预算4096，保留原调度，候选／ALR从7001轮启用。
- 用户最新回传为30000/30000、保存Gaussian及app_model并打印 `Training complete`；日志总时长1:14:56，train PSNR31.102751541137696／L1 0.0163764713332057，点数1295772。末轮sampled4096／comparable3143／added1463，只证明补选分支活动，尚无几何收益结论。
- 服务器输出：`/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/3d85faf_seed0_30k_v1`。
- 输入工作视图：`/root/autodl-tmp/ambisur_work/data_views/second_run_selector/toolroom_r2_seed0_30k_v1/colmap_undistorted`。
- 独立基线扩展：`/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python`，通过进程级 `PYTHONPATH` 选择；旧一周目扩展保留。
- 已确认服务器环境为Python3.10／Torch2.7.1+cu128／CUDA toolkit12.8／GCC11.4／RTX4090，Python入口 `/root/miniconda3/envs/ambisur/bin/python`。下方旧环境字段不是当前环境。
- 本地主目录整理没有干预服务器作业。用户已回传 `train.exitcode=0`、30K点云307M和app.pth15K，训练完成及非空模型产物已确认；不重训或恢复checkpoint。

## 正式mesh评价入口（2026-10-23）

- 用户回传406视角渲染/TSDF/后处理结束，新mesh raw12962510、post11470565顶点；后续下载核验两run提取退出码均0，mesh身份匹配。顶点数量不代表几何收益。
- 用户授权可复用正式评价，并确认PRF阈值1/2/5cm、5cm主列。评价固定提交 `4010fd01e4b5bbe651ac4e221e00532bf99f4ff9`（已推送），统一入口 `scripts/evaluate_mesh_geometry.py`，协议 `experiments/mesh_geometry/scannetpp_surface_v1.json`，完整用法/既有资产命令见 [正式mesh评价指南](mesh-geometry-evaluation.md)。
- Accuracy/Completeness/Surface Chamfer + Precision/Recall/F1；1M面积样本、seed0、共同GT、点到三角面、毫米/0–1，不ICP/裁剪/截断。训练代码和DTU/TnT原脚本未修改。ScanNet++适配协议不冒称官方排行榜或DTU/TnT完全同口径。
- 外部独立本地环境Open3D0.19.0真实8项测试全过/零跳过，含原TnT函数数值对照、真实PLY/CLI/metadata/JSON/CSV和输入只读；审查seed偏差已RED→GREEN修复。Python3.10语法检查通过。后续服务器8项测试全过/零跳过（10.591s）及Completed/完整CSV已收到。用户自行下载原件后，agent进行了获准离线GT空间诊断，未连接服务器。

## 首次正式几何结果（2026-10-23）

- [完整结果与解释](2026-10-23-toolroom-selector-v1-results.md)：baseline→selector的Accuracy35.1244→37.2194mm，Completeness46.3068→54.8356mm，Surface Chamfer40.7156→46.0275mm（+13.0463%，变差）。
- F1@1/2/5cm均略升，5cm为0.77933825→0.78048925（+0.1151个百分点）；5cm Recall下降0.2820个百分点。结论为混合结果、尚无整体几何收益，不隐去正负任一方向。
- PID12484消失由正式Completed日志解释为正常结束；不重启评价。新run、完整评价及实际C0已由用户下载到D盘，双方postmesh/本地GT哈希与报告完全匹配，JSON/CSV一致；不由agent SSH连接。
- 训练、mesh与评价原件保全；阈值、评价域与方法不变。先归档，下一批定位/改动须讨论，不自动叠加模块或新训练。

## 最新本地分析（2026-10-23）

> **2026-10-26更新：** 用户已完成1219文件输入哈希比对，0missing/0mismatched。随后要求逐行确认基线来源；[完整源码审计](2026-10-26-selector-baseline-source-audit.md)确认d979c07直接父提交为C0，未继承旧Core；C0与官方88d6405的8个Python差异文件AST相同、4个native差异文件非注释token相同。4个既有生产文件加2个新模块逐行核对，1630受保护文件与C0逐字节相同。本轮18CPU测试通过/5native跳过；不是全7K轨迹等价证明。当前不再等待输入核对，不自动启动训练或归因数值随机性。

- 新归档位于 `D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/selector_v1_2026_10_23`；实际C0在相邻`baseline/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z`。原本地original不是正式C0，不混用成绩。正确C0 train PSNR31.07406044→31.10275154（+0.0286911dB），仅为5个训练视角。
- 最终快照三个既有debug视角的A阴性候选池约129416/741704/164701，4096覆盖3.165%/0.552%/2.487%。固定预算在不同视角改变新增ALR剂量，给预算不足提供具体机制线索，但尚无因果结论；总掩码图不能分离extra。
- 实际C0到齐后复用原评价核心/相同GT样本离线诊断：z<4m平均Completeness31.7489→29.9042mm，z>=4m为84.1396→119.6272mm；整体退化主要集中上部。完整性差距88.03%在raw已存在；最坏1%点84.67%满足至少一相机GT first-hit 1cm可见容差。不据此裁剪正式评价或生成训练mask。
- 关键混杂：上部局部包围盒Gaussian中心数7000轮已为C0 29904/new14205，selector从7001启动；日志600轮首次点数差异。中心数不等于7K表面误差，但最终差距不能全归因于补选。相机/初始点云/邻居哈希一致；全部图像/深度/置信/对齐scale待核验，不先认定数值随机性或4096是根因。
- 独立分析在新归档`analysis/toolroom_pair_seed0_30k_diagnosis_v1`。原均值复算最大差4.62e-13mm；正式原件、方法/训练代码均未改。下一批建议只读比对C0已有manifest与新工作视图1219文件，不用GPU/GT、不重训baseline；完整命令与边界见结果报告末节，待用户讨论。

## 接手后的下一步

1. 在 D 盘检查分支、HEAD、clean状态，阅读当前候选规格与根目录三个进展记录。
2. 用户已同意提取新模型mesh并与基线比较。先沿用baseline的 `max_depth=5.0, voxel_size=0.005, sdf_trunc_scale=4.0, num_cluster=2`，以服务器原3d85faf和独立扩展提取iteration30000；已有baseline mesh保持只读。已收到提取后处理日志及三个mesh路径；固定适配协议的服务器8项真实CPU测试和首轮正式评价均已完成；当前转向原产物归档及待讨论的退化定位，不重训或重提取baseline。
3. baseline `D:\research_Space\output\AmbiSuR_original` 与数据集保持只读；不重训、不覆盖、不移动。GT只用于获准离线评价。
4. DTU沿用CD及双向距离，TnT沿用Precision／Recall／F1，Mip360沿用PSNR／SSIM／LPIPS；当前Tool Room属于ScanNet++，不能套用DTU/TnT专用评价域。`eval=False` 的训练视角PSNR不冒称新视角指标。
5. C 清理、旧 Utility 恢复及新方法扩展均另行讨论；当前优先获得实际重建比较结果。

## 以下为历史交接记录

保留旧目标、环境和阶段约定供追溯；冲突时以上方当前入口、AGENTS和已批准候选规格为准。

## 1. 项目目标

以 AmbiSuR 官方实现为 baseline，在 ScanNet++ 紧凑、高反射室内场景上研究：观测校准的歧义需求、外部先验与内部几何双可靠性、拒绝式仲裁、参数级梯度路由和可靠性驱动的 Gaussian 生命周期。

优先目标是会议论文；保留期刊扩展与硕士毕业论文兜底。当前主张限定为“改善高反射室内场景的整体表面重建”，不声称已经对人工标注反光区域做独立定量证明。

## 2. 代码来源

- 官方实现：`https://github.com/Fictionarry/AmbiSuR`
- 用户 fork：`https://github.com/Monkot19/noob_AmbiSuR`
- 新项目开始时必须在这里补写实际基线 commit：`BASELINE_COMMIT=<运行 git rev-parse HEAD 后填写>`
- `main` 只保存可复现 baseline；开发从 `research/core-routing` 开始。

## 3. 环境与算力

- 平台：AutoDL
- GPU：单张 RTX 4090 24GB
- PyTorch：2.8.0
- Python：3.12
- 系统：Ubuntu 22.04
- CUDA：12.8
- Core 峰值显存建议低于 22GB，训练时间不超过 baseline 约 2 倍。

## 4. 数据

- ScanNet++ Tool Room：`d415cc449b_Tool_Room`
- ScanNet++ Utility Room：`0a5c013435_Utility_Room`
- 本地数据根目录：`D:\dataset\ScanNet++\data\data`
- 本地 baseline 结果：`D:\research_Space\output\AmbiSuR_original`，只读
- AutoDL 数据根目录：首次配置时填写到实验 manifest，不得猜测或写死在代码中
- 后续泛化：AmbiSuR 官方 DTU 15 scenes、Tanks and Temples 6 scenes

Tool Room seed 0 是开发与 D0 标定场景；Utility Room 和其他 seeds 用于确认。论文必须披露 Tool Room 的开发用途。

## 5. 已确认的方法边界

Core 顺序：

1. 连续高端 SH 歧义量 A。
2. 有效视图数与方向离散度得到观测充分度 S。
3. 得到外部帮助需求 N。
4. 分别计算外部 `(T^P,V^P)` 与内部 `(T^G,V^G)`。
5. 计算可靠性、一致性和可靠性差。
6. 输出 Bypass、Consensus、Prior-led、Geometry-led、Abstain。
7. C2 开始执行粗粒度动作；C3 增加拒绝；C4 参数组路由；C5 冲突投影；C6 生命周期。

Supporting 仅在 Core 通过后实施：

- CoMe 式解耦曝光；
- 可靠前表面局部 Ray-Color；
- 可靠前表面局部 Ray-Normal；
- 生命周期状态驱动的逐 Gaussian 截断。

## 6. 实验门槛

- G0：关闭新功能时复现 baseline；无 NaN/Inf、显存增长和重复 backward。
- G1：N 的高几何误差 AUROC 大于 0.60，且比 A 或 1-S 中较优者至少提高 0.03。
- G2：两个反光场景平均几何改善约 3%，单场景系统退化不超过约 1%；PSNR/LPIPS 分别不恶化超过 0.3dB/0.01。
- G3：Supporting 每个模块最多进行默认配置、一次有依据调整、一次跨场景确认。
- G4：DTU/TnT 多数场景不得系统退化。

## 7. 本地—GitHub—AutoDL 闭环

本地 Agent 只在隔离分支/worktree 修改代码，完成单元测试、代码审查和短步检查后提交并推送。AutoDL checkout manifest 中记录的明确 tag/commit，运行短步 GPU smoke 或正式实验。实验输出连同 manifest、配置、日志、指标和环境快照下载回本地。分析脚本读取结果但不修改原始输出；新结论写入 `findings.md`，阶段状态写入 `progress.md`。

禁止行为：服务器临时改代码、只记录分支名不记录 commit、用脏工作树跑正式实验、结果出来后移动 tag、一次提交混入多个消融模块。
