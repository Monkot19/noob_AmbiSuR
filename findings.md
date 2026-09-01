# AmbiSuR 新项目 Findings

本文件只记录新项目通过代码、测试和实验得到的事实。设计假设不得写成已证实结论。

## Confirmed Inputs

- Spec：`docs/research/ambisur-reliability-routing-design.md`
- Fork：`https://github.com/Monkot19/noob_AmbiSuR`
- GPU：RTX 4090 24GB
- 主场景：Tool Room、Utility Room
- 开发场景：Tool Room seed 0

## Evidence Table

| Claim | Status | Evidence/Path | Next check |
|---|---|---|---|
| feature-off 与 baseline 等价 | untested | | E0 |
| N 比 A 或 1-S 更能定位几何风险 | hypothesis | | D0/G1 |
| 双可靠性能校准风险 | hypothesis | | coverage-risk |
| Abstain 能减少错误选边 | hypothesis | | C2 vs C3 |
| 参数路由改善几何 | hypothesis | | C3–C5 |
| 生命周期改善稳定性与几何 | hypothesis | | C5 vs C6 |

## Code Findings

每项记录：文件与行号、可复现证据、是否已测试、是否与设计冲突。不要只写“可能有问题”。

## Experiment Findings

每项必须引用对应 manifest、commit、scene 和 seed。不得把单次运行写成稳定结论。

## Decisions

| Date | Decision | Evidence | Consequence |
|---|---|---|---|

