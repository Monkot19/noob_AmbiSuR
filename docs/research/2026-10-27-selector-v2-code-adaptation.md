# selector_V2：供体源码到现有代码的适配清单

日期：2026-10-27。状态：用户基本同意V2主旨，要求先汇报实际代码改法；**本文为修改前的源码适配说明，未实现V2。** 主仓库HEAD核对点06589b2，实际生产逻辑仍为c3510b02。短规格见[selector_V2](../superpowers/specs/2026-10-27-selector-v2-relocation-design.md)。

## 1. 本次真正读了哪些供体代码

固定Sparse2DGS commit `af15587eb186ca7ab12a2cf795775fa0f473f25d`。读取train.py158–211及scene/gaussian_model.py的初始化/view_indexs、get_point_from_depth、replace_tensor_to_optimizer、update_points正文；不是只读README。重新从官方raw地址取得下面两个文件，**与本地读到的缓存逐字节相同**，未安装依赖或执行新供体训练。

| 文件 | 字节 | SHA256 |
|---|---:|---|
| train.py | 18387 | a43e72a019e3fb4b95f1fec2f7c8322514c61d692fbb1ac87da11228d0cdb210 |
| scene/gaussian_model.py | 24911 | c5ab6a89fc43db44161a4bbe39c0099f82a760830a0cd5dbfd9a93ccb96c44b3 |

本轮旧source_manifest.json引用路径已不存在，因此直接核对固定commit原始文件，不假装旧manifest仍在。官方链接：[训练调用](https://github.com/Wuuu3511/Sparse2DGS/blob/af15587eb186ca7ab12a2cf795775fa0f473f25d/train.py#L163)、[位置更新](https://github.com/Wuuu3511/Sparse2DGS/blob/af15587eb186ca7ab12a2cf795775fa0f473f25d/scene/gaussian_model.py#L476)、[优化器替换](https://github.com/Wuuu3511/Sparse2DGS/blob/af15587eb186ca7ab12a2cf795775fa0f473f25d/scene/gaussian_model.py#L324)。

## 2. 保留的机制和必须适配的地方

| 供体真实做法 | 当前V1真实情况 | V2适配 |
|---|---|---|
| train.py180–200比较primitive几何与模型渲染几何，两支持源取最小NCC | utils/candidate_selector.py已有G/P平面、共同支持与best-source判定；P取DA3 | 完整复用V1比较，不移入第二套compute_hom或源视角选择 |
| get_point_from_depth将渲染深度反投影至世界坐标 | V1第212行的points_p在参考相机系，第231/233行与ids一起经过有效项筛选 | 返回同一points_p胜出行，经现有行向量外参逆变换后作为世界系目标 |
| view_indexs与repeat(3,1)对应三张初始化深度图的分组关系 | 当前高斯由7K模型继承，后续会增删，无该三组像素排列 | 使用当前真实高斯ID；不复制repeat(3,1)或引入初始化归属标签 |
| train.py168–202在原ADC位置执行SGU；入口没有densify_and_prune调用 | train.py549–554仍按原规则增密/裁剪 | 按V2=V1+重定位保留原ADC；这是“增加动作”的适配，非完整SGU复现 |
| update_points只实际写xyz，rotation/scaling更新注释掉 | 原ALR基于渲染深度法线，可影响位置与方向 | 新增动作只写xyz，法线仍用原ALR渐进优化，不硬拷贝先验旋转 |
| 整块new_xyz.clone后换Parameter，改变整块xyz的状态/grad | 当前Adam以self._xyz对象为state键 | 新增局部原地方法，不调用整张量replace_tensor_to_optimizer |
| SGU在backward后、optimizer.step前执行；全量换Parameter使原grad不沿新Parameter继续使用 | V1已有主渲染前的no_grad候选阶段 | 移动插在该阶段，主渲染基于新位置，避免旧位置梯度用于新位置 |

供体每100步执行，但放在其densification条件内；本轮7100..29900的全启用阶段窗口仍是明确的适配提议，不是源码给出的最优设置。保留ADC也意味着重定位后的点可在本轮后续被原规则clone/split/prune，不能宣称整轮点数或Parameter身份永远不变。

## 3. 五个文件分别怎么改

### A. utils/candidate_selector.py：复用比较，补上目标输出

当前函数select_candidate_labels位于155–259行，返回(labels, extra, stats)，实际NCC决策在255行。计划将其原函数体移入私有`_select_candidate_geometry(..., include_targets=False)`，不复制算法；原公开select_candidate_labels保留原签名并取前三项。增加薄入口`select_candidate_labels_with_targets(...)`调用同一核心，返回四项。原V1入口/非事件轮均不生成世界系目标。

第四项targets是None或`(ids:[M]int64, xyz_world:[M,3]float32)`，device/dtype跟当前模型一致，均no_grad、无跨迭代持有。所有原提前回退返回None；标签、stats、随机抽样身份字符串与原NCC运算完全不变。末尾仅在请求目标且存在胜出项时追加：

```python
# 修改示意，尚非生产实现；ids和points_p都来自原函数同一次有效项筛选。
move_ids = ids[accepted]
move_camera_xyz = points_p[accepted]
target_xyz = (move_camera_xyz - transform[3, :3]) @ transform[:3, :3].T
targets = (move_ids.detach(), target_xyz.detach())
```

**顺序不能改：** 如ids=[8,2,5]、accepted=[True,True,False]，目标顺序对应[8,2]；extra.nonzero()却返回[2,8]。把后者和原目标数组配对会错写位置。因此必须一起取ids[accepted]/points_p[accepted]，不能事后从bool标签重新凑坐标。

同文件增加`candidate_relocation_active(iteration,opt)`：关闭立即False；开启则检查V1已启用、间隔为正整数，沿用V1模式检查，并使用`candidate_selector_active(iteration,opt) and iteration % interval == 0 and iteration < opt.iterations`。不加新的置信度阈值或候选选择条件。

### B. scene/gaussian_model.py：增加一个局部位置更新方法

在现有optimizer管理方法附近增加`@torch.no_grad() relocate_selected(ids, target_xyz)`，输入来自上述targets。检查形状/设备/类型、索引范围、目标有限性及xyz参数绑定后，对M=0返回0；其他情况下：

```python
self._xyz.index_copy_(0, ids, target_xyz)
state = self.optimizer.state.get(self._xyz)
if state is not None:
    state['exp_avg'].index_fill_(0, ids, 0)
    state['exp_avg_sq'].index_fill_(0, ids, 0)
for name in ('xyz_gradient_accum', 'xyz_gradient_accum_abs',
             'denom', 'denom_abs', 'max_radii2D', 'max_weight'):
    getattr(self, name).index_fill_(0, ids, 0)
return ids.numel()
```

以上是动作示意，完整输入检查须在任何写入之前。不得写`self._xyz[ids].copy_(...)`或`state['exp_avg'][ids].zero_()`：高级索引可能得到副本。原地index_copy_/index_fill_才明确修改原存储。

保留Parameter身份、Adam全局step/LR、非选中行、其他参数组及其状态；不为无状态参数新造state。不调用当前325行replace_tensor_to_optimizer，它会清空整个xyz动量并换Parameter。清理选中行历史的目的，是避免继续沿旧位置累积动量推进，不等于为每行重启完整Adam；保留全局step也避免改变非选中行的偏置修正。现有378–391行cat_tensors_to_optimizer同样为新增行置零矩并保留既有step，是兼容现有优化器表示的参照，不是最优性证明。

六个统计数组均已存在于create/training_setup/capture/restore路径，不新增持久状态。当前525–529行densify处理零分母NaN为0；移动点将按移动后的本轮render重新累计统计，若无实际贡献仍保留原处理，不借此新建visibility门。重定位方法本身不改点数，后续增密逻辑照旧。

### C. train.py：只在主渲染前接入动作

现有_render_training_reference第89–122行已经具备合适位置：第103–106行是两源no_grad深度回调，第108–111行选择，第117行开始主参考渲染。事件轮改用上面新增薄入口，取得targets后立即调用gaussians.relocate_selected；非事件轮仍调用原select_candidate_labels。两者都将原labels/extra/prior_depth/prior_normal/stats交给现有后续流程。

```text
原相机/背景选择
→ 原V1候选、两源no_grad比较
→ 事件轮得到当前ID/世界系目标并立即重定位
→ 原主参考渲染（同一A并E标签）
→ 原深度loss、ALR backward、主loss backward
→ 原统计/增密/trim/opacity reset/optimizer.step
```

事件轮只走一次选择，不先跑V1再重跑一次选择，不增加源渲染或第二次主渲染。labels在移动前判定，SH系数与ID未变；移动后掩码由新主渲染自然生成。ALR仍消费该labels/prior_normal，不换loss、不新增冻结、不改470行后的参数管理或553行后的拓扑。

training开头第137行现有启动校验后增加重定位配置校验，避免只开新开关却没开V1时静默失效。日志保留原selector行，事件轮仅加一行`[relocation iteration] moved=M`，这样7100/7300等不落在原200轮日志间隔上的事件也可见；不存逐高斯历史文件。该记录比原短规格“仅在原日志加字段”更明确，不改变方法。

### D. arguments/__init__.py：仅增加两个字段

在已有enable_candidate_selector/candidate_selector_budget附近增加`enable_candidate_relocation=False`和`candidate_relocation_interval=100`。不改V1默认4096；首轮命令显式预算16384。原SH阈值、全部LR/loss、iterations/seed与增密参数不动。

### E. tests/test_candidate_selector.py：复用一套入口

新增边界只覆盖：原入口三元组与新入口前三项一致/不消费全局RNG；非单位相机与乱序ID的真实选择结果传给真实重定位方法；选中行位置/动量/stat正确，其他行/Parameter/step/参数组不变；关闭/非事件/空E不写；主render读到更新后位置，之后可正常backward/step。

沿用该文件已有的真实方法AST装载方式（原NativeCheckpointTests第502–507行）在CPU执行新增生产方法并使用真实torch.optim.Adam，不用手写目标字典或mock消费者作为唯一接口证据。原native集成用例可补活动路径，但本地缺扩展如实跳过，不另搭测试文件或依赖环境。代码未实现，因此本轮没有声称这些检查已通过。

## 4. 哪些现成接口直接不改

gaussian_renderer/__init__.py的ambiguity_labels接口、CUDA、_compute_alr_loss、原loss/NCC工具、capture/restore、7K/App恢复逻辑、mesh提取与正式评价器均复用。utils/run_metadata.py第23–35行已序列化optimization全部字段，新开关与间隔自然进入resolved_config；training_path仍是既有candidate_selector大类，以新开关、commit和输出名区分V2，无需新schema或writer。

关闭重定位时仍返回V1三元组、同一选择与原训练路径；开启但非事件轮同样不产目标或写位置。baseline关闭V1时不进入候选/重定位。直接赋值仅对E行；后续ALR/RGB/拓扑仍可通过模型耦合影响其他高斯，不声称整个训练过程只改变E。

## 5. 当前能确认什么与实施顺序

本轮证据是固定官方源码字节核对及当前函数/张量/调用顺序的静态核对，确认有明确接入点；**未修改训练代码、未运行V2测试或训练，不能宣称已可运行或有收益。**

确认这份具体改法后：在D盘从当前二周目分支建立codex/second-run-selector-v2，保留V1分支和baseline；先在原测试入口补新增边界，再按上述五文件接线、跑相关CPU检查、提交推送固定commit；交付AutoDL从现有7K直接训练到全局30K的新输出命令。现有baseline及V1结果只读复用，不补恢复对照或短GPU实验作为前置任务。
