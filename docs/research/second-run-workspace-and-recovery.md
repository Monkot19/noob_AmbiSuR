# 二周目主目录、归档与换机恢复

2026-10-23 用户确认 D 盘作为二周目长期主目录。本次只整理本地工作区与文档，训练代码不变。

## 目录与分支

| 位置 | 用途 |
| --- | --- |
| `D:\research_Space\noob_AmbiSuR` | 主仓库，真实 `.git` 目录；2026-10-27起使用 `codex/second-run-selector-v2`，原V1分支 `codex/second-run-selector`保留在ef333bb；后续修改／测试／提交均在D完成 |
| `C:\Users\DELL\.codex\worktrees\second-run-selector\noob_AmbiSuR` | 保留的 linked worktree，固定 detached `3775135`；其 `.git` 文件指向 D，不能单独复制后当独立仓库使用 |
| `D:\research_Space\noob_AmbiSuR_workspace_backup_20261023T040116Z` | 原样文件、历史日志及 Git bundle 备份；不是数据或 baseline 输出目录 |

当前状态用 `git worktree list --porcelain` 核对。C 的保留不影响 D 分支前进；清理另行讨论，不自动 prune/remove/archive。

## 保全记录

- 原 D 分支 `codex/candidate-selector-v1` 仍为 `62e9117dfb2e20a06b257d166d8a6636d37845cc`。
- 归档分支 `codex/archive-before-second-run-20261023` 为 `7b3154b7daa7e03ca7dad5b0e5cf1260f4e1099d`，已推送 origin，保存8个已跟踪修改和16个未跟踪文件。它是历史快照，不是当前原生验证通过的二周目实现。
- 二周目接管起点 `3775135690b2ea4edfdc50e3b19fba467d02d117`；本次只改工作规则和文档，生产代码与它一致。
- 原192个忽略文件未清理且经哈希确认不变；其中54个非缓存历史记录另存ZIP。残留旧缓存不作为二周目实现依据。

备份目录包含：

- `working-tree.zip`：24个文件的原始字节，以 `files/相对路径` 保存；SHA256逐项校验通过。
- `ignored-history.zip`：54个历史日志／工作记录，SHA256逐项校验通过。
- `preservation-manifest.json`：原路径、大小、哈希、提交和代码指纹。
- `repository-preserved.bundle`：迁移前所有可达 refs 的完整历史，含旧分支、归档、二周目、tags和已有stash；`git bundle verify`通过，无前置仓库依赖。
- `unstaged.patch`、`staged.patch`、`refs-before.txt`、`worktrees-before.txt`、`bundle-heads.txt`和迁移执行记录。

bundle大小6121675字节，SHA256为 `193c8c88f628e6e881bdad7a873a6c96851ea7f3e797cb05fe2a88d39d0e9b34`。

ZIP保存原始换行字节；Git的 `core.autocrlf=true` 会正常规范化文本换行，因此归档和接管代码另按LF内容核对。原字节与Git内容两种表示均有保全。

百度云同步范围由用户设置。此备份位于主目录同级；如果只同步主目录，需要另外保留备份目录。GitHub恢复已推送分支，原始字节ZIP、忽略日志及stash的离线恢复需要该备份。

## 新电脑正常接手

目标目录尚不存在时执行：

```powershell
git clone --branch codex/second-run-selector https://github.com/Monkot19/noob_AmbiSuR.git D:\research_Space\noob_AmbiSuR
git -C D:\research_Space\noob_AmbiSuR branch --show-current
git -C D:\research_Space\noob_AmbiSuR rev-parse HEAD
git -C D:\research_Space\noob_AmbiSuR status --short
```

如果D目录已经通过百度云同步，先执行后三条和 `git remote -v`；不向非空目录再次clone或覆盖 `.git`。D是普通主仓库，新电脑接手不依赖C worktree存在；历史worktree登记按实际状态另行处理，不预先清理。

随后阅读 `AGENTS.md`、`docs/research/project-handoff.md`、当前候选规格和根目录 `task_plan.md`／`findings.md`／`progress.md`。数据、模型和Conda环境不随Git恢复，分别按交接记录确认，不把路径存在当作可运行证据。

原本机回归入口如下；新电脑应使用实际可用的Torch环境，不自动另建训练环境：

```powershell
Set-Location D:\research_Space\noob_AmbiSuR
& C:\Users\DELL\miniconda3\envs\pytorch\python.exe -B -m unittest tests.test_candidate_selector -v
```

本机接管后是18项通过、5项原生集成因缺本地扩展而跳过。服务器 `3d85faf` 的23项全通过是另一份回执，不混作新电脑测试结果。实际原生训练继续在AutoDL，环境与扩展路径见当前交接入口。

## 离线 bundle 恢复与旧文件查看

保留备份原件，恢复目标必须是新目录。以下独立目录只用于恢复核对，核对后再讨论主目录接管：

```powershell
git clone --branch codex/second-run-selector D:\research_Space\noob_AmbiSuR_workspace_backup_20261023T040116Z\repository-preserved.bundle D:\research_Space\AmbiSuR_recovery_check
```

bundle保存接管前二周目 `3775135` 及归档历史；之后的交接文档提交从GitHub获取。这样clone后的origin初始指向bundle，连接GitHub时须核对并显式更新remote，不能当作已经同步到远端最新。

旧文件可用 `git show codex/archive-before-second-run-20261023:相对路径` 查看；新clone中使用 `origin/codex/archive-before-second-run-20261023:相对路径`。ZIP解包到单独目录，不直接覆盖当前二周目文件。

## 验证与仍待事项

- D切换后1697个已跟踪文件齐全；8个核心代码／测试文件与迁移前C内容一致；192个忽略文件及C原始代码未改变。
- C detach首次因worktree `logs/HEAD`权限拒绝失败，由用户本机正常Git命令完成；没有改ACL或强制回退。
- D分支、HEAD、clean状态经明确断言检查，归档没有merge回二周目；迁移不改变算法参数或训练调度。
- 目录迁移不操作服务器作业，不重训baseline，不移动数据或已有输出。迁移时已知进度为23760/30000；后续训练状态以 `docs/research/project-handoff.md` 为准。
- C清理另行讨论；下一步仍是分析实际重建结果，不新增无关诊断体系。
