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

