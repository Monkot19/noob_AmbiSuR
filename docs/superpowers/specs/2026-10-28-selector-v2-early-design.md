# v2_early：仅缩短位置重定位窗口

状态：2026-10-28用户确认“可以尝试一下这个v2_early”，执行此前明确提议，不另加方法。

- 主工作目录仍D盘，分支`codex/second-run-selector-v2-early`；保留V2、V1及原生C0分支和所有输出。
- 复用同一原生7K checkpoint及7K App、r2、seed0、预算16384、重定位间隔100，训练全局7001..30000。
- 唯一实验变量：重定位在global iteration<15000时允许，即7100..14900共79次；15000及以后不移动E，不清理其Adam/stat。
- 15000及以后仍正常运行V1候选补选、原SH标签合并、标签渲染、ALR及其他原有loss/优化，直到30K。选择定义/阈值/支持视角/采样RNG/目标/LR/生命周期/renderer/评价不变。
- 新CLI `--candidate_relocation_until_iter`，默认0表示无额外截止，完整保持已有V2调度；显式15000启用本实验。旧opt未带字段时按0处理。启用重定位时仅接受非负Python int，拒绝bool/float/负值；关闭重定位时不增加验证/行为。
- 只改`arguments/__init__.py`、`utils/candidate_selector.py`、`tests/test_candidate_selector.py`，复用已有真实训练helper和元数据writer。
- 复用同一测试入口，只补截止边界、79/229事件数、默认/旧配置兼容、真实helper在截止后保留V1输出且无重定位的检查。不新增测试入口/诊断框架/独立GPU smoke。
- AutoDL固定提交运行、私有data view及独立`selector_v2_early`输出；已有模型/数据/输出不覆盖，baseline不重训。
- 供体af15587默认7000总轮数/5000截止，实际update_points在该分支内；15000是宿主已有densify_until_iter，不是供体直接给出的最优值，也非GT选阈值。效果/后期损失原因均待本轮验证。
- 提取与正式评价沿用原配置及原程序。训练预计参考V2约1小时，实际日志为准；本地无原生扩展，不冒称GPU通过或服务器已启动。
