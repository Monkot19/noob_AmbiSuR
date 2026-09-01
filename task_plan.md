# AmbiSuR Core 实施计划状态

## Goal

在不混入 Supporting 模块的前提下，完成可复现、可消融的 AmbiSuR Core：观测校准、双可靠性、五状态仲裁、参数级梯度路由和 Gaussian 生命周期，并通过 G0–G2。

## Current Phase

Phase 0：新项目接管与 baseline 锁定。

## Phases

### Phase 0：接管与规格核对
- [ ] 读取 AGENTS、交接说明、baseline audit 和最终设计稿
- [ ] 记录实际 baseline commit、环境和数据路径
- [ ] 核对代码能否提供所有公式输入
- [ ] 生成详细 Core implementation plan
- **Status:** in_progress

### Phase 1：E0 与测试基础
- [ ] 锁定 C0 baseline tag
- [ ] 建立 CPU 单元测试和 GPU smoke 入口
- [ ] 验证所有新增开关关闭时等价
- **Status:** pending

### Phase 2：D0 影子证据
- [ ] 实现 A/S/N 统计
- [ ] 实现双可靠性、K、Delta
- [ ] 实现五状态和日志，但不改变训练
- [ ] 通过 G1
- **Status:** pending

### Phase 3：C1–C3 粗粒度执行
- [ ] C1 需求门控
- [ ] C2 整组强制仲裁
- [ ] C3 增加 Abstain
- **Status:** pending

### Phase 4：C4–C5 参数路由
- [ ] 分离 g_B 与 g_P
- [ ] C4 参数组直接加权
- [ ] C5 冲突投影
- **Status:** pending

### Phase 5：C6 生命周期
- [ ] Normal/Confirmed/Repair/Protected/Quarantine/Probation
- [ ] 持续计数、冷却和两阶段提交
- [ ] optimizer/EMA/state 拓扑迁移
- **Status:** pending

### Phase 6：G2 主验证
- [ ] Tool Room 与 Utility Room 快速确认
- [ ] Baseline/Core 三 seeds
- [ ] 效率、显存、几何与外观守门
- **Status:** pending

### Phase 7：Supporting 决策
- [ ] 仅在 G2 通过后新建独立计划
- **Status:** pending

## Errors Encountered

| Date | Error | Attempt | Resolution |
|---|---|---:|---|

## Scope Guardrails

- 当前计划禁止实现曝光、Ray-Color、Ray-Normal、自适应截断。
- GT mesh 不进入训练。
- C0–C6 每阶段必须对应独立 commit 和验证 tag。
- 服务器不修改代码。

