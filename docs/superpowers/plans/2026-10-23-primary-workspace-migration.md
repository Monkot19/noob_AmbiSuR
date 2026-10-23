# 二周目 D 盘主目录接管计划与记录

**Goal:** 保全D／C两边修改和历史，让D使用二周目分支，C保留，服务器和结果不动。

**Spec:** 用户2026-10-23确认的备份／归档／detach／切换／核对／文档方案。

**Architecture:** 保持原共享Git结构，用归档分支和外部ZIP／bundle保存现状；C固定提交后，D通过普通Git切换接管。只更新交接文档和工作规则。

**Workflow:** executing-plans、planning-with-files、using-git-worktrees和verification-before-completion；本任务不改变方法行为或新建测试框架。

## 约束与核对

- 不复制覆盖 `.git`，不强制回退，不移除C worktree，不移动／覆盖数据或baseline。
- 保存全部24个文件、54个忽略历史记录，确认原分支与归档可恢复；bundle完整、字节快照可校验。
- D核心文件与接管前C一致，忽略文件不变；测试结果区分本地原生跳过与实际服务器验证。
- 恢复文档不依赖新电脑C路径存在；不把正在运行的训练写成已完成。

## 执行记录

- [x] 只读核对D62e9117／8修改+16未跟踪，C3775135／clean／共享Git；说明方案并获确认。
- [x] 保存逐字节ZIP、忽略历史ZIP、192个忽略文件哈希、patch／refs／worktree状态。
- [x] 归档提交7b3154b并推送，原分支仍62e9117；完整bundle导出及verify通过。
- [x] 工具C detach遇日志权限拒绝，停止；用户本机完成正常detach，D普通git switch接管。
- [x] 核对1697tracked、8核心代码／语法、192ignored、C raw哈希不变。
- [x] D原统一入口18通过／5原生依赖跳过，无新脚本／安装／服务器复测。
- [x] 更新AGENTS、当前交接入口、换机恢复指南和根目录三个记录。
- [x] 一次范围限定审查通过，无Critical／Important／必须修复Minor；生产代码diff为空。
- [x] 文档已以3d27ba2提交／推送，远端引用一致；D主分支clean、C固定3775135且clean，1699个已跟踪文件齐全，Git连接性／备份及保全哈希复核通过。后续文档收据不改变生产代码。

备份和恢复详情见 `docs/research/second-run-workspace-and-recovery.md`。
