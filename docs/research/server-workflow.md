# 本地—GitHub—AutoDL 实验闭环

## 1. 总原则

本地是唯一代码开发源；GitHub 是版本交换与审计中心；AutoDL 只运行已提交代码；本地结果仓库只负责归档和分析。任何正式实验都必须绑定一个不可变 commit/tag 和一份 manifest。

## 2. 本地开发流程

### 首次建立 baseline

```bash
git clone https://github.com/Monkot19/noob_AmbiSuR.git
cd noob_AmbiSuR
git status --short
git rev-parse HEAD
git tag c0-baseline <已验证的baseline-commit>
git push origin c0-baseline
git switch -c research/core-routing c0-baseline
```

只有确认 `c0-baseline` 能复现实验后才能打 tag。若仓库已有用户注释提交，应以实际验证过的 commit 为准，不能机械选择最新 commit。

### 每个 C 阶段

1. 从上一验证点开始，只实现一个阶段的唯一增量。
2. 先写并运行失败测试，再实现最小代码使其通过。
3. 检查 `git diff --stat` 与 `git diff`，确认没有 Supporting 或无关重构。
4. 提交测试与代码。
5. 本地可运行的测试全部通过后推送。
6. AutoDL 完成 smoke 和阶段实验。
7. 只有验证通过才打阶段 tag。

示例：

```bash
git add arguments reliability tests train.py
git commit -m "feat: add C1 observation-calibrated need gate"
git push origin research/core-routing
```

服务器确认该 commit 通过后：

```bash
git tag -a c1-need-gate <commit> -m "C1 verified on Tool Room smoke"
git push origin c1-need-gate
```

不要用一个 commit 同时实现 C2 和 C3；不要在 C3 实验后重写 C3 历史并移动 tag。

## 3. 服务器代码同步

首次：

```bash
git clone --recursive https://github.com/Monkot19/noob_AmbiSuR.git
cd noob_AmbiSuR
git fetch --all --tags --prune
git checkout --detach <manifest中的commit或tag>
git submodule update --init --recursive
git status --short
git rev-parse HEAD
git diff --exit-code
```

后续不要直接 `git pull` 后运行。每次先 `git fetch`，再 checkout manifest 指定 commit。`git status --short` 必须为空，`git diff --exit-code` 必须返回 0。

如服务器代码需要修复：停止实验，在本地分支完成测试、commit、push，然后服务器 checkout 新 commit。服务器上的临时修改不能直接用于论文结果。

## 4. 实验层级

### 本地检查

- CPU 纯函数测试：A/S/N、可靠性组合、状态机、梯度投影、生命周期计数。
- 静态检查：开关默认关闭；没有硬编码数据路径；没有 Supporting 代码混入 Core。

### AutoDL GPU smoke

- 固定 Tool Room、固定 seed、100–500 iterations。
- 检查 CUDA 编译、forward/backward、NaN/Inf、峰值显存、状态张量长度。
- smoke 失败不能启动 30k。

### D0

- 5k–7k 只记录影子状态，不改梯度和生命周期。
- 输出 A/S/N/T/V/K/Delta 分布、状态占比和 GT 离线诊断。
- G1 不通过则回到证据定义，不继续实现 C6 掩盖问题。

### 阶段实验

- C0–C6 使用同一数据、seed、迭代数、mesh 提取和评价协议。
- 除本阶段唯一开关外，配置差异必须为 0。
- 快速筛选通过后再进行 Tool+Utility、3 seeds 主实验。

## 5. 输出目录

建议服务器目录：

```text
/root/autodl-tmp/ambisur_runs/
└── <scene>/
    └── <stage>/
        └── <commit8>/
            └── seed_<seed>/
                ├── manifest.md
                ├── resolved_config.json
                ├── env.txt
                ├── train.log
                ├── metrics.json
                ├── checkpoints/
                ├── mesh/
                └── diagnostics/
```

不得用相同目录重跑并覆盖旧实验。修复后重跑必须使用新 commit 目录。

## 6. 回传本地

每次至少下载：`manifest.md`、resolved config、环境快照、完整日志、指标、最终 mesh、诊断统计；需要续训时再下载 checkpoint。保持服务器目录的相对结构。

本地归档建议：

```text
D:\research_Space\output\AmbiSuR_reliability\
└── <scene>\<stage>\<commit8>\seed_<seed>\
```

下载完成后计算文件校验和或至少记录文件数量和大小。分析脚本只读原始回传目录，表格和图片写到独立 `analysis/` 目录。

## 7. 推荐节奏

不需要把每个小 commit 都送上云端完整跑 30k。推荐：本地单元测试 → 服务器 100–500 轮 smoke → 阶段性 Tool Room seed 0 → 通过门槛后 Utility Room → 最后 3 seeds。这样既节约经费，也保持证据链清晰。

