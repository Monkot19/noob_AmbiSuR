# AmbiSuR 二周目项目规则

> 2026-11-09发布回执：离线采集／推理实现固定`cae81969d5eee7f97e5a4fd9d8151b300742fbb4`，分支`codex/depth-source-audit`，操作见`docs/research/depth-source-audit-launch.md`。统一71项63通过／8native跳过（5.791s），独立复核无剩余实质问题；官方MVS模块仅导入验证，尚无权重加载／真实前向／GT质量结果。train／arguments／scene／utils／renderer／submodules保持87ec3c2，baseline／数据只读；服务器由用户执行17视角只读采集＋独立环境MVS推理，非30K训练。

> **2026-11-09用户批准离线位置来源／准入检查：** 比较同原生7K的当前渲染深度、原DA3、spread与MVSAnywhere；GT只评价，不生成训练mask或调门槛。复用原capture／GTQueries／统计入口，补存17视角完整深度；MVS在独立环境推理。固定法线、同支持集平均NCC仅作离线准入对照探针，不等同已实现真实对应准入，不构成新训练mask授权。D隔离分支`codex/depth-source-audit`；既有训练、renderer、loss/LR/预算及baseline不改。用户已确认AutoDL开机，允许准备只读前向命令，服务器仍由用户执行，不自动启动训练。计划`docs/superpowers/plans/2026-11-09-depth-source-audit.md`。

> 2026-11-04过程数据最新结论：cd647552训练/提取/下载及96参考、24探针、79动作验证完成；原正式评价新CD40.562933，仅比C0好0.3324%，仍比early差2.0593%。本次未重现旧严重前鼓/后期A覆盖下降，不能利用本次轨迹宣布旧根因或采集算法收益；门A-only权重.954、位置误差仍约4cm。大面积墙/天花板传播目标仅较当前改善约.4–1.1mm。报告`docs/research/2026-11-04-process-trace-results-and-mechanisms.md`，独立只读复核通过；当前最好仍early，本轮只有分析，不授权新loss/LR/预算/训练或Utility泛化。

> 2026-11-04用户最新批准：从现有原生7K checkpoint直接到30K补过程记录，同时追踪后期纠正失败。D分支`codex/selector-process-trace`，只允许独立采集启动器/共享record抽取/统一测试与文档；6172bc45方法、原参数、原train/renderer/执行器不改，无新loss/LR/门槛/GT训练。96实际参考事件+24只读A-only固定探针，联合E与实际move分别记录。63项55通过8native缺失跳过，独立审查无实质问题；未验证GPU，服务器由用户启动，不增加smoke/新baseline前置。规格`docs/superpowers/specs/2026-11-04-candidate-process-trace-design.md`；固定提交及命令见`docs/research/selector-process-trace-native7k-launch.md`。

> 2026-11-03结果状态：6172bc45位置检查版已完成30K/提取/本地正式评价，CD41.779128mm，比early差5.12%、旧spread差1.50%，不能视作成功模块或自动追加预算/Utility/训练。当前最好仍early。报告`docs/research/2026-11-03-selector-position-check-geometry-results.md`；本轮评估不构成软传播loss或其他新方法的实施授权。

> 2026-11-03最新批准范围：在V3 spread上增加传播有效域内抽样与固定当前法线的位置执行检查，预算显式16384；联合E继续用于ALR，移动仅取检查通过子集，7100..14900/30K不变。D分支`codex/selector-v3-position-check`，生产固定`6172bc455184c2b77b7b199c7b66194319bd55fd`。两开关默认False，保留旧V3行为；无新loss/LR/传播/renderer/基线重训。规格`docs/superpowers/specs/2026-11-03-selector-position-check-design.md`，直接30K指南`docs/research/selector-v3-position-check-native7k-launch.md`；本地65项57通过/8原生依赖跳过，不声称GPU或收益通过，不自动启动服务器。

> 2026-10-28最新批准范围：v2_early仅将重定位截止于15000（7100..14900共79次），其余V1/ALR/原优化继续至30K。D主目录分支codex/second-run-selector-v2-early，原V2/V1/baseline保留。规格docs/superpowers/specs/2026-10-28-selector-v2-early-design.md；不额外加smoke、baseline或诊断框架作前置。此前默认全程V2行为保留，禁止把本实验截止值静默变成原V2默认。

## 主工作目录与当前范围（2026-10-23 用户确认，优先于旧规则）

- 二周目始终以 `D:\research_Space\noob_AmbiSuR` 为主工作目录，使用 `codex/second-run-selector`；后续修改、测试和提交均在这里完成。
- 用户选择在 D 盘通过分支隔离工作；未经再次讨论，不把主开发转移到 C 盘或新建其他主 worktree。
- C 盘 `C:\Users\DELL\.codex\worktrees\second-run-selector\noob_AmbiSuR` 保留在 detached `3775135`；清理由用户另行决定。
- 当前方法为干净 `c0-baseline` 加已批准候选选择模块，复用原标签渲染和 ALR；旧 Core、五状态、可靠性路由和 Utility 恢复仅属历史，不自动继续。
- 数据集与已有 baseline 输出只读，复用固定 baseline，不重新训练、覆盖或移动。AutoDL 当前训练固定 `3d85faf`，本地主目录整理不触碰服务器作业或版本。
- **2026-10-26用户最新授权例外：** 新增一次原生C0的Tool Room/r2/seed0/30K训练，原有参数保存7Kcheckpoint；新baseline分支整树必须与d6f15c8完全一致，不补保存模块/钩子或修改基线。仅写独立新output，旧baseline/4096继续只读保留。二周目恢复适配独立，不声称原生checkpoint保存RNG/相机队列。当前规格见`docs/superpowers/specs/2026-10-26-shared-7000-prefix.md`。
- 归档和换机入口见 `docs/research/second-run-workspace-and-recovery.md`；不复制覆盖 `.git` 或强制回退来迁移目录。

## 必读顺序

1. `docs/research/project-handoff.md`
2. `docs/research/baseline-audit.md`
3. `docs/superpowers/specs/2026-10-27-selector-v2-relocation-design.md`（当前V2规格，继承V1；旧 `ambisur-reliability-routing-design.md` 为历史参考）
4. 当前代码、配置、Git 状态与最近提交
5. `task_plan.md`、`findings.md`、`progress.md`

## Skill 工作流

- 每个新会话首先使用 `superpowers:using-superpowers`。
- 复杂任务使用 `planning-with-files`，持续维护根目录三个规划文件。
- 需要新增或改变方法行为时，先使用 `superpowers:brainstorming`；已确认设计不得静默重写。
- 修改代码前使用 `superpowers:writing-plans`。
- 开始独立功能开发时使用 `superpowers:using-git-worktrees`，或至少创建隔离分支。
- 每个功能/修复使用 `superpowers:test-driven-development`。
- 出现异常、NaN、测试失败或指标意外变化时先使用 `superpowers:systematic-debugging`。
- 完成一个阶段后使用 `superpowers:requesting-code-review`。
- 声称完成、通过或可运行前使用 `superpowers:verification-before-completion`。

## 一周目阶段规则（历史；二周目按上方当前范围执行）

1. 设计稿是已批准的规格合同。代码与规格冲突时先报告证据，不得擅自选择另一算法。
2. 第一阶段只实现 Core：观测校准、双可靠性、五状态仲裁、参数路由、生命周期。
3. 曝光、Ray-Color、Ray-Normal、自适应截断属于 Supporting；Core 未通过 G0–G2 前不得实现。
4. D0 只记录状态和指标，严禁影响训练梯度或拓扑。
5. C0–C6 必须严格嵌套；每一级只增加设计稿规定的唯一能力。
6. 每个模块独立 commit；禁止顺手修改无关模块。
7. 一个阶段若需要多个实现 commit，验证通过后再打阶段 tag。
8. 任一实验必须能由 commit/tag、配置、seed、数据场景和命令唯一复现。

## 数据与结果安全

- 用户本地 baseline 结果目录只读，禁止覆盖、移动或删除。
- 数据集不提交 Git；代码不得写死 Windows 或 AutoDL 的绝对数据路径。
- GT mesh 仅用于 D0 标签生成和最终评价，禁止进入训练、可靠性缓存或 checkpoint。
- 不制作或依赖人工高光 mask。
- 服务器不得临时修改训练代码；如需修复，先在本地分支提交并推送，再让服务器 checkout 明确 commit。

## 验证与提交规则

- 功能关闭时必须保持 baseline 等价；未通过 G0，不解释任何指标收益。
- CPU 单元测试、本地静态检查、服务器短步 GPU smoke、完整实验依次执行。
- commit 信息使用 `test:`、`feat:`、`fix:`、`refactor:`、`docs:` 前缀。
- 每次推送前记录 `git status --short`；工作树不干净不得启动正式云端实验。
- 服务器实验开始时记录 `git rev-parse HEAD` 和 `git diff --exit-code`。
- 不因单场景正结果跳过 Utility Room 或多 seed 确认。



## 2026-10-22 User-approved clean second run: supersedes old Core stage rules
Use c0-baseline plus approved candidate selector only, no reliability/Core/five-state/evidence/topology port. Preserve historical work and fixed baseline output. See docs/superpowers/specs/2026-10-21-candidate-selector-design.md and docs/superpowers/plans/2026-10-22-clean-selector-migration.md. Lightweight seed/metadata reuse is authorized. Use one test entry, then AutoDL native validation, no new diagnostic platform. Progress updates follow the user five-item format. Each new implementation or experiment batch is discussed first; later explicit approval supersedes the original migration batch's no-push/no-training restriction.

## 持续有效的复用约束

- 2026-10-27用户已批准selector_V2=V1+重定位及源码适配，授权立即实施和常规问题自主处理、事后报告。主目录仍D，当前分支codex/second-run-selector-v2，V1分支/历史及baseline不动。固定a114ed5；24CPU通过/6native跳过，不能称GPU或收益通过。当前下一步为按已批准预算16384/原LR/loss/间隔100/现有7K直接至30K，不另加smoke、恢复对照或诊断框架作前置任务；新的方法范围仍须讨论。

- 同一科学问题只维护一套公共统计与审计核心；新场景主要增加数据身份、路径、seed、单位和协议配置，格式差异通过薄适配层处理。
- 不按场景复制 evaluator、solver、空间分折、bootstrap 或状态审计；不为通用框架阻塞已批准的最小实验。
- 不同数据集沿用各自正式评价定义，统一执行与 provenance，不用同一指标替代 DTU/TnT 等官方协议。
- 真实接口复用须贯通 producer → 适配／聚合 → 原有类型规范化 → schema 校验 → 发布；新增边界用实际生产函数和真实返回类型验证，不只靠手写字典、list fixture 或 mock。
- 原 JSON／NumPy serializer 直接复用，类型转换不改变数值、随机序列、统计方法或门槛；只为新增格式或协议边界补测试，不重建统计框架。
- 接口异常与科学负结果分开记录；GT 后失败保留原记录，诊断后须获明确恢复授权，不删除一次性记录或自动重跑。
- 历史 Utility 的冻结 `1-r_p`、M0/M1、目标命名、单 seed／多 seed 聚合与发布合同仍约束其后续恢复；二周目工作不构成恢复授权。
