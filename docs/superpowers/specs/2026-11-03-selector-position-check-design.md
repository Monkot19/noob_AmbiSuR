# V3有效域抽样与固定当前法线的位置执行检查

状态：2026-11-03用户已批准上一条五点技术路线，授权开始实施；本文将已批准范围落实为接口合同，不追加方法。

主目录D，分支`codex/selector-v3-position-check`，起点`6de5c4419819f496fd88276ed539abe29f6835f1`；生产起点与已完成V3/8b1def9相同。已有待提交分析文档保留；本次普通Git切分支成功，未更改ACL。

## 已批准行为

1. 预算显式16384，不改全局默认4096；传播mask按原双线性四邻有效要求，在原~A/视野pool中先筛有效传播深度再抽样。后续原几何/法线/遮挡/NCC无效项仍退出，不补抽。
2. 原联合NCC与best-source/0.5门槛继续产生`E_ALR`，标签仍`A|E_ALR`；抽样域改变后标签数值可能改变，不声称ALR集合逐值冻结。
3. 只有请求重定位targets的事件轮，对联合接受行补算`C(points_p, normal_g)`。当前位置代价复用；重算经过points_p的平面距离，保留当前法线轴。两种位置仅在共同有效支持上按原best-source汇总，严格`min C_pos < min C_g`通过，不添加新绝对门槛/边际/多视角投票。
4. `move_accepted = accepted_joint & position_pass`；ALR extra不变。返回targets仍为(ids,world_xyz)，只含通过行，ID顺序及字面目标不变；空集合返回None。不改位置执行器/Adam处理。
5. 沿用7100..14900共79事件、interval100/until15000，总30000、原生7K起点/r2/seed0、原DA3 depthloss/ALR/LR/拓扑/renderer/TSDF/评价。15K后继续联合ALR，位置检查不执行。

## 开关与复用

- 两个默认False选项：`candidate_selector_valid_pool`、`candidate_relocation_position_check`。本次实验二者显式开启，source=spread。关闭二者保持旧V3/V2/V1路径及抽样流。启用改进要求selector与spread；with_targets用于只读观察时无需实际启用参数移动。
- 修改arguments、同一selector核心、train事件日志；复用source depth/gray，不为位置检查增加支持渲染。
- 原capture observer仅薄适配targets子集核验；schema1中的accepted继续表示联合ALR接受，不能改称移动标签。不新增审计系统或逐Gaussian日志。
- 原测试入口`tests.test_candidate_selector`覆盖真实NCC/参数执行/观察边界；不增加GPU smoke或新baseline前置。完成后交付固定commit与直接30K命令，不自动开服务器。

## 限制

本轮同时检验有效抽样与执行检查，不能将收益单归一个因素。NCC只是匹配代理，仍存在中心/表面差别及旧DA3持续目标问题；软传播loss、独立E_pos、锚点/边界求解器均不在范围。
