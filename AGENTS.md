# AmbiSuR 二周目项目规则

## 主工作目录与当前范围（2026-10-23 用户确认，优先于旧规则）

- 二周目始终以 `D:\research_Space\noob_AmbiSuR` 为主工作目录，使用 `codex/second-run-selector`；后续修改、测试和提交均在这里完成。
- 用户选择在 D 盘通过分支隔离工作；未经再次讨论，不把主开发转移到 C 盘或新建其他主 worktree。
- C 盘 `C:\Users\DELL\.codex\worktrees\second-run-selector\noob_AmbiSuR` 保留在 detached `3775135`；清理由用户另行决定。
- 当前方法为干净 `c0-baseline` 加已批准候选选择模块，复用原标签渲染和 ALR；旧 Core、五状态、可靠性路由和 Utility 恢复仅属历史，不自动继续。
- 数据集与已有 baseline 输出只读，复用固定 baseline，不重新训练、覆盖或移动。AutoDL 当前训练固定 `3d85faf`，本地主目录整理不触碰服务器作业或版本。
- 归档和换机入口见 `docs/research/second-run-workspace-and-recovery.md`；不复制覆盖 `.git` 或强制回退来迁移目录。

## 必读顺序

1. `docs/research/project-handoff.md`
2. `docs/research/baseline-audit.md`
3. `docs/superpowers/specs/2026-10-21-candidate-selector-design.md`（当前规格；旧 `ambisur-reliability-routing-design.md` 为历史参考）
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

- 同一科学问题只维护一套公共统计与审计核心；新场景主要增加数据身份、路径、seed、单位和协议配置，格式差异通过薄适配层处理。
- 不按场景复制 evaluator、solver、空间分折、bootstrap 或状态审计；不为通用框架阻塞已批准的最小实验。
- 不同数据集沿用各自正式评价定义，统一执行与 provenance，不用同一指标替代 DTU/TnT 等官方协议。
- 真实接口复用须贯通 producer → 适配／聚合 → 原有类型规范化 → schema 校验 → 发布；新增边界用实际生产函数和真实返回类型验证，不只靠手写字典、list fixture 或 mock。
- 原 JSON／NumPy serializer 直接复用，类型转换不改变数值、随机序列、统计方法或门槛；只为新增格式或协议边界补测试，不重建统计框架。
- 接口异常与科学负结果分开记录；GT 后失败保留原记录，诊断后须获明确恢复授权，不删除一次性记录或自动重跑。
- 历史 Utility 的冻结 `1-r_p`、M0/M1、目标命名、单 seed／多 seed 聚合与发布合同仍约束其后续恢复；二周目工作不构成恢复授权。
