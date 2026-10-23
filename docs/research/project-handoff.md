# AmbiSuR 二周目交接入口

> **当前入口（2026-10-23）：** 主工作目录已改为 `D:\research_Space\noob_AmbiSuR`，分支 `codex/second-run-selector`。C 盘临时 worktree 保留 detached `3775135`，后续开发不再以 C 为入口。完整归档与换机步骤见 [主目录与恢复指南](second-run-workspace-and-recovery.md)。下方一周目记录仅供查阅，不作为当前派工。

## 当前方法与代码

- 从 `c0-baseline`（`d6f15c8891a53800d5e3100f95817a7dd7f98e2f`）仅移植已批准候选模块：保留原 SH 双端标签，从未命中池抽样4096，比较当前平面与 DA3 平面的跨视角 NCC，补选后复用原标签渲染与 ALR。
- 当前规格：`docs/superpowers/specs/2026-10-21-candidate-selector-design.md`；移植记录：`docs/superpowers/plans/2026-10-22-clean-selector-migration.md`。
- 方法提交 `d979c07`；服务器验证／训练固定 `3d85faf95951b4806eee18498841e66551b93320`。本地接管起点 `3775135690b2ea4edfdc50e3b19fba467d02d117` 与它仅有文档差异；本次也只更新规则和文档，实际当前 HEAD 用 `git rev-parse HEAD` 读取。
- 旧 D 未提交内容保存为归档 `7b3154b7daa7e03ca7dad5b0e5cf1260f4e1099d`，不是当前二周目实现；不将旧 Core、五状态、梯度路由或 Utility 恢复自动合并／恢复。

## 已验证与正在运行

- AutoDL RTX4090／`3d85faf` 统一23项测试全部通过、零跳过，包含真实 renderer／ALR backward／optimizer step。
- D 接管后同一入口23项中18项通过、5项原生集成因本地缺少 AmbiSuR 扩展而跳过；8个代码文件语法检查通过。1697个已跟踪文件齐全，192个忽略文件未改变。
- Tool Room 首轮30K已由用户实际启动：406张训练图像、r2、seed0、`eval=False`、DA3、预算4096，保留原调度，候选／ALR从7001轮启用。
- 用户最近回传为23760/30000，多条 sampled4096／added非零。仅证明新增分支在活动，尚无最终重建收益结论；零抽样记录不单凭这段日志推断原因。
- 服务器输出：`/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/3d85faf_seed0_30k_v1`。
- 输入工作视图：`/root/autodl-tmp/ambisur_work/data_views/second_run_selector/toolroom_r2_seed0_30k_v1/colmap_undistorted`。
- 独立基线扩展：`/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python`，通过进程级 `PYTHONPATH` 选择；旧一周目扩展保留。
- 已确认服务器环境为Python3.10／Torch2.7.1+cu128／CUDA toolkit12.8／GCC11.4／RTX4090，Python入口 `/root/miniconda3/envs/ambisur/bin/python`。下方旧环境字段不是当前环境。
- 本次目录整理不连接、停止、重启或更新服务器训练；完成需等待 `train.exitcode`、最终日志和30K产物回执。

## 接手后的下一步

1. 在 D 盘检查分支、HEAD、clean状态，阅读当前候选规格与根目录三个进展记录。
2. 接收本次训练最终产物，再与用户讨论复用原 mesh 提取及适用的评价入口，与已有固定 baseline 比较。
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

