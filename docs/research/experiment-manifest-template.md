# AmbiSuR 实验 Manifest 模板

> 每次实验复制本文件到输出目录并填写。不得修改已经完成实验的 manifest；补充信息使用“追加记录”。

## 身份

- Experiment ID：`<scene>-<stage>-<commit8>-seed<seed>-<timestamp>`
- 日期时间与时区：
- 操作者：
- 目的：
- 实验层级：`E0 / D0 / E1 / E2 / E3 / E4`
- 消融阶段：`C0 / C1 / C2 / C3 / C4 / C5 / C6 / Supporting`

## Git

- Repository：`https://github.com/Monkot19/noob_AmbiSuR`
- Branch（仅说明来源）：
- Commit SHA（必须 40 位）：
- Tag：
- `git status --short` 输出：`必须为空`
- `git diff --exit-code`：`0`
- Submodule commit：

## 环境

- GPU：RTX 4090 24GB
- PyTorch：2.8.0
- Python：3.12
- Ubuntu：22.04
- CUDA：12.8
- 驱动版本：
- `pip freeze` 文件：`env.txt`
- CUDA 扩展重新编译：`是 / 否`

## 数据

- Dataset：ScanNet++
- Scene：
- 服务器数据绝对路径：
- 输入视图协议：
- 数据版本/校验信息：
- GT mesh 使用方式：`仅离线评价 / 本实验不使用`

## 运行配置

- Seed：
- Iterations：
- Config 文件：
- Resolved config：`resolved_config.json`
- 完整命令：

```bash
<粘贴实际执行命令，不使用“同上”>
```

## 功能开关

| 开关 | 值 |
|---|---|
| observation calibration | off |
| dual reliability | off |
| abstention | off |
| parameter routing | off |
| conflict projection | off |
| lifecycle | off |
| decoupled exposure | off |
| local Ray-Color | off |
| local Ray-Normal | off |
| state truncation | off |

只有当前阶段规定的开关允许改变。C0 所有新增开关必须为 off。

## 预期门槛

- 本次唯一变量：
- 必须通过的测试：
- 主要几何指标预期：
- 外观守门指标：
- 运行时间上限：
- 峰值显存上限：
- 停止条件：NaN/Inf、CUDA error、状态长度错位、脏工作树、配置不匹配。

## 实际结果

- Exit code：
- 训练时长：
- 峰值显存：
- 最终 Gaussian 数：
- 几何指标：
- PSNR/SSIM/LPIPS：
- 状态分布：
- 日志路径：
- Mesh 路径：
- Checkpoint 路径：

## 结论

- `通过 / 不通过 / 仅诊断`
- 是否允许进入下一阶段：
- 依据：
- 异常与失败：
- 下一步必须改变的唯一变量：

## 追加记录

- 下载到本地的日期：
- 本地归档路径：
- 校验信息：
- 分析脚本 commit：

