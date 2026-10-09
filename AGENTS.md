# AmbiSuR Reliability Routing Project Rules

## 必读顺序

1. `docs/research/project-handoff.md`
2. `docs/research/baseline-audit.md`
3. `docs/research/ambisur-reliability-routing-design.md`
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

## 范围与阶段规则

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

## 跨场景复用与统一评价（2026-10-08 用户确认）

- 同一科学问题只维护一套公共统计与审计核心；禁止为每个场景复制 evaluator、solver、空间分折、fold-local 预处理、bootstrap 或 checkpoint/state 审计实现。
- 新场景主要新增数据身份、图像数量、路径、seed、单位/坐标约定及预注册配置；必要的格式差异通过薄适配层处理。
- 只为新增数据格式、协议差异和 fail-closed 合同补测试；已有核心测试作为回归运行，不按场景重新开发。
- 当前 Utility 必须直接复用 Tool Room 冻结的 `1-r_p`、M0/M1 与统计核心，不得借通用化重新选择公式、模型、门槛或评价域。
- 不同数据集的正式评价定义可能不同；统一执行、provenance、结果 schema 与报告框架，不强行用同一几何指标替代 DTU/TnT 等官方协议。
- 不为追求通用框架而提前大规模重构、重复 Tool Room 探针开发或拖延已批准的最小 Utility 工作；先复用现有接口，在新增合同边界内推进。

### 真实生产接口复用门（2026-10-09 用户强化确认）

- 复用不止是调用同一 evaluator：必须贯通真实 producer → 场景适配/多 seed 聚合 → 原有类型规范化 → schema 校验 → 发布。新增适配层不得绕过原流程必需的步骤。
- 新增接口的最小集成测试必须实际调用已有生产函数，用小型合成输入获得真实返回类型/结构，再传给真实 consumer；手写字典、预先转换为 list 的 fixture 或 mock 不能作为唯一接口验证证据。不要求为每个场景重新建立统计测试框架。
- 原有 JSON/NumPy 规范化及生产记录 serializer 直接复用；producer 返回 ndarray、标量或实际规范格式时，验证长度、值、dtype/结构与既有合同一致，不能只判断 mock 的类型。类型转换不得改变数值、随机序列、统计方法或门槛。
- 至少覆盖真实冻结目标命名、记录格式、单 seed 到多 seed 聚合与最终发布路径；科学核心仍只维护一套，场景差异只来自经批准的配置。正式服务器执行前通过与新增边界对应的小型端到端测试，不以无关全套验证替代。
- 接口异常与科学负结果分开记录；不得为绕过接口错误修改候选、门槛或评价域。GT 后失败必须保留原记录，诊断后另获明确恢复授权，不删除一次性访问记录、不自动重跑。
