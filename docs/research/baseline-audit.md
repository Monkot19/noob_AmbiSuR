# AmbiSuR Baseline 代码审计与实现风险

本文件记录实施前已发现的风险。它们需要代码和服务器实验验证；未验证前不得在论文中称为 baseline 缺陷。

## 已确认的代码事实

1. `scene/gaussian_model.py::compute_weighted_sh_norm` 当前按 degree 3 写死 15 个非 DC SH 系数；新实现必须按当前 SH 阶数动态计算。
2. CUDA `out_observe[i]` 是单视图中满足前景透射条件的像素命中数，不是视图数；观测充分度必须先用 `out_observe>0` 转成每视图 0/1。
3. 当前 multi-view trim 每 1000 轮遍历训练相机并累计 `out_observe>0`，可复用其扫描时序，但方向覆盖需新增。
4. 当前曝光路径的 SSIM 使用 detached appearance image；迁移解耦 SSIM 时，亮度项必须允许梯度进入曝光参数。
5. 当前 Ray-Color 在 CUDA backward 隐式注入 `dL_dcolors`；标准 SH backward 可能通过视角方向把梯度泄漏到 means3D。
6. ALR 的额外 backward 可能导致同一 CUDA Ray-Color 在一轮内重复注入。
7. 当前 ALR 用布尔切片修改 `requires_grad_`，不能可靠实现逐 Gaussian 参数冻结。
8. 当前 `trunc_sigma` 是 CUDA 全局标量；逐 Gaussian 截断必须同步修改 Python、C++/CUDA forward 与 backward 接口。
9. 当前 densify/prune 在 optimizer step 前可能替换 Parameter；生命周期实现必须明确操作顺序，并迁移 Adam、EMA 和状态数组。
10. DA3 已参与初始化和监督；论文只能称两条证据通道，不能声称统计独立。
11. Gaussian 法线由最短尺度轴经 rotation 得到，并根据观察方向翻转符号。

## 实施前必须建立的测试

- 动态 SH 阶数测试：degree 1、2、3 的非 DC 系数数量分别为 3、8、15。
- `out_observe` 测试：像素命中数不能被直接解释为有效视图数。
- feature-off 等价测试：所有新开关关闭时，固定输入的损失和梯度与 baseline 一致。
- 状态机测试：五种状态的全部分支和持续计数。
- 梯度路由测试：外部先验对 SH/曝光梯度恒为 0；Abstain 几何梯度为 0、外观保留基础梯度。
- 投影测试：冲突投影后辅助梯度与主导梯度内积不再为负。
- topology migration 测试：clone/split/prune 后所有状态张量、Adam buffer 与 Gaussian 数量一致。
- 服务器 GPU smoke：100–500 轮内无 NaN/Inf、CUDA 越界和显存持续增长。

## G0 阻断条件

以下任一情况出现都必须停止方法实验：关闭功能仍改变 baseline；存在脏工作树；日志缺少 commit/config/seed；多次 backward 导致隐式重复梯度；densify 后状态长度不等于 Gaussian 数量；恢复 checkpoint 后状态或 optimizer 不一致。

