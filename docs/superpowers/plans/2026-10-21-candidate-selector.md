# Candidate Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for proposed native execution; use superpowers:subagent-driven-development only if the user explicitly chooses it. Implement task-by-task with TDD and checkbox tracking. 2026-10-22用户确认native执行；算法规格不变，当前本地实现/CPU检查完成，AutoDL原生验证待执行。

**Goal:** 按已批准selector首版完成最小接线，必要检查后尽快运行一次新方法30k；开发日志比较PSNR/L1，正式评价沿用基线各数据集的几何和渲染指标。

**Architecture:** 一个公共selector和一套参数化测试；在主渲染前比较G/P，临时OR标签进入原第5通道和原ALR。不接旧Core，不建诊断/安全平台；两个no-grad源深度前向与原可导多视角路径分开。

**Tech Stack:** 现有PyTorch/CUDA扩展和unittest。优先已有兼容环境；当前捆绑Python无Torch，不在本计划自动装环境或重编基线。

**Spec:** [2026-10-21-candidate-selector-design.md](../specs/2026-10-21-candidate-selector-design.md)，用户已确认；批准前文件SHA为`c7db52d1e724d72f544dbfa78a31e7605c30f4f25b6dee378cf37ee2a61658e8`，此后只更新批准状态文字。

## Global Constraints

- 方法严格按规格：旧Core全关、use_mono=False、独立开关默认False、预算4096、两个固定真实nearest源、原7×7 patch、best-source严格cP<cG且cP<0.5、原SH标签OR追加；不改门槛或加入新模型。
- 时序为iteration>max(1000,unc_from_iter)，默认7001；标签只存当前参考相机/迭代，不建EMA/cache/checkpoint状态；候选无效不补抽，正常退化e=0，程序/模型错误不吞掉。
- 只动以下五个代码文件；文档/根规划记录除外。需要第六个代码文件、方法修改或新增依赖时先请示，不顺手修ALR/SH/曝光/CUDA。
- 复用已有投影、warp、NCC、深度采样、metadata与日志；不写场景专用测试/评价器/启动器。代码实施前使用隔离分支/worktree；保留当前未提交研究文件，不reset/清理它们。
- baseline固定output只读复用；不重训/续训baseline，不改旧Utility记录、不访问GT。只给新方法独立输出，首轮先拿30k训练日志与模型供分析。
- 2026-10-22用户修订：DTU报告双向距离/CD，TnT报告Precision/Recall/F1，Mip-NeRF360报告PSNR/SSIM/LPIPS；复用规格§7列明的现成评测入口，不把train PSNR作为唯一收益结论或新视角指标，不因补指标重训baseline。
- 所有代码/测试先在本地分支形成明确commit再到服务器；不得服务器热改代码。提交/推送、GPU执行和30k启动按用户行动批次确认，不以本计划自动派发。

## File Structure

| 文件 | 边界 |
| --- | --- |
| `utils/candidate_selector.py`（新增） | 公共SH标签、活动条件、抽样/平面/支持/NCC决策；不import renderer或旧reliability |
| `tests/test_candidate_selector.py`（新增） | 所有新增CPU/条件GPU测试和共享合成夹具；一个入口，无测试脚本生成器 |
| `arguments/__init__.py` | 仅增加enable_candidate_selector=False与candidate_selector_budget=4096 |
| `gaussian_renderer/__init__.py` | 末尾可选ambiguity_labels=None；共享SH函数；原第5通道消费 |
| `train.py` | 最薄的前渲染编排、先验复用、同标签ALR及原日志；必要的原逻辑同文件提取，保持运算不变 |

## Review Focus

1. 非单位相机姿态下G需world→camera，P的真实render_normal已是camera：Task1的pose/plane测试与Task2的真实normal生产者测试。
2. 源图尺寸/ncc_scale不同、patch越界或平面交点在后方，不能接受padding：Task1的shared-fixture参数化边界测试。
3. G的raw NCC>0.9仍可参与，且两假设仅有一个共同源时可比：Task1的cost/shared-support测试。
4. 关闭/7000前多渲染、全局RNG改变、e=0改变梯度：Task1的RNG/活动条件，Task2的真实renderer/ALR对照。
5. 标签行数/类型/device错误或源回调抛异常不得伪装正常回退：Task2的真实接口错误测试；每轮重新生成长度，无持久状态。

## Task 1: 一个公共测试入口，先观察真正的RED

**Files:** Create `tests/test_candidate_selector.py`。不写生产实现。

**Interfaces to pin:** 下列接口均在上述五文件内，不新增框架类型：

```python
# utils/candidate_selector.py
compute_sh_ambiguity_labels(gaussians, opt) -> Tensor  # [N] bool
candidate_selector_active(iteration: int, opt) -> bool
select_candidate_labels(gaussians, reference_camera, reference_gray,
    prior_depth, prior_normal, support_cameras, opt, *, seed: int,
    iteration: int, support_depth_fn) -> tuple[Tensor, Tensor, dict]
# 返回(b_new:[N]bool, e:[N]bool, stats)，stats仅sampled/comparable/added计数。
# prior_depth:[H,W]，prior_normal:[3,H,W]已在参考相机系。
# support_depth_fn(camera)->真实plane_depth:[1,Hs,Ws]；最多调用两个源。

# train.py：原代码同文件最小提取，供真实测试与训练共用
_render_training_reference(viewpoint_cam, gaussians, pipe, bg, app_model,
    opt, *, iteration: int, seed: int) -> tuple[dict, dict | None]
# 返回原render_pkg与selector_data；后者只含labels/extra/prior_depth/prior_normal/stats。
_compute_alr_loss(viewpoint_cam, render_pkg, metric_depth, metric_depth_conf,
    conf_thresh, iteration, opt, normal_from_metric_depth=None) -> Tensor
```

- [x] **1. 写CPU参数化测试。** `CandidateMathTests`共用解析平面/灰度图夹具：K的fx=fy=40、cx=cy=15.5、32×32；参考非单位Rz(30°)/平移，平面camera-Z=2，真实源基线0.1，另含不同图宽高/ncc_scale子例。调用真实patch_warp/lncc；检查已知平面重投影、sign-paired(n,d)不变、raw cost可用、空源/无效P不选、无全局RNG消费。用真实OptimizationParams解析器检查默认False/4096及启用参数，用原metadata生产者检查字段保留且旧Core仍全关；不支持的模式/错误预算应明确报错。内部函数可私有，不为每条断言造新公共API。
- [x] **2. 写决策和预算断言。** 固定代价仅用于决策单元，实际NCC另有上一步覆盖：
  `G=[.95,.10], P=[.20,.80], valid=[1,1] -> e=False`；`G=[.95,.90], P=[.20,.80] -> e=True`；共同有效仅源0仍True；空交集False；cP=.5或cP=cG均False。池5000最多4096且无重复，同seed/iteration/uid同抽样；小池全取，无效退出不补抽。`active(7000)=False, active(7001)=True`；unc_from_iter=0时1000仍False，1001才True。
- [x] **3. 写条件GPU真实接口测试。** `CandidateIntegrationTests`在同文件创建小型真实Camera/GaussianModel/renderer夹具：真实normal生产者验证非单位姿态；真实select返回labels→真实主render→真实ALR helper→一次backward/optimizer step，iteration=7001。检查`labels.dtype==bool`、shape/device/detach、原命中保留、`torch.testing.assert_close(M_new,M_A+M_e)`；关闭与e=0的loss/梯度对照；source callback至多2次且no_grad/ray_reg关闭，原多视角可导路径不被替代。附dtype/长度错误、异常传播子例。GPU集成必须使用真实生产返回值，mock不能是唯一证据。
- [x] **4. 运行并记录RED。** 有Torch的既有环境执行`python -B -m unittest tests.test_candidate_selector -v`，预期因新API/新标签参数缺失而失败；缺Torch、缺扩展或CUDA skips不算RED/GREEN。若本机仍无Torch，先报告现有环境缺口，使用已批准兼容环境，不自行安装。所有测试保存在同一文件；最多一次测试合同提交，不增加RED专用脚本。

## Task 2: 实现五文件，通过必要检查后直达30k准备

**Files:** Create `utils/candidate_selector.py`；Modify `arguments/__init__.py:139–163`、`gaussian_renderer/__init__.py:38,183–188`、`train.py:227–233,433–475`；Test同Task1文件。行号以HEAD62e9117为参考。

**Consumes/Produces:** 实现Task1的全部签名；renderer在现有参数末尾新增`ambiguity_labels=None`，返回结构不变；不改变CoreConfig、checkpoint或旧公共几何函数。

- [x] **1. 实现公共selector。** 直接照规格§2–4：原SH公式、升序A补集池、局部SHA种子Generator抽4096、G/P相机系平面、两个源原深度采样/往返/全patch有效性、共同源raw NCC best-source、返回OR标签。复用真实NCC/warp/采样，不移植供体位置替换。无源或候选耗尽不调用源深度回调；prior无效只退该候选，接口/模型非有限或回调异常报错。
- [x] **2. 接两参数与renderer。** 新开关/预算留在OptimizationParams；旧Core全关/DA3模式做最薄入口检查。公共SH函数替换原重复公式；None走原语义，有标签时验证[N]bool/device/detach并写原第5通道，不新增CUDA通道，不强制只更新选中点。
- [x] **3. 接train的实际编排。** `_render_training_reference`移动原参考render调用且保留全部原参数/时序，关闭时不增加selector工作；开启后先准备一次参考DA3深度/normal，调用selector，回调只做两个源no-grad深度前向，再携同一labels主render。原深度/ALR复用prior_depth/prior_normal；unc_mask用同labels。`_compute_alr_loss`仅原mult、normal、unc_diff、confidence和mean表达式同文件提取，不改backward/参数管理/Ray-Color/拓扑。新增统计沿用原日志，约每200轮输出一次，GPU计数仅输出时取标量；总ALR权重不冒充新增量。
- [ ] **4. 运行GREEN及现有回归。** CPU分组：`python -B -m unittest tests.test_candidate_selector.CandidateMathTests -v`；兼容GPU环境：`python -B -m unittest tests.test_candidate_selector tests.gpu.test_feature_off_dispatch -v`。后一条同时完成上述一次7001活动路径检查，**不再建独立500步冒烟实验/审批平台**。缺依赖/skips明确记录，不能冒充真实GPU通过。
- [ ] **5. 最小静态检查与一次阶段审阅。** 五文件py_compile、`git diff --check`、核对diff无第六代码文件/算法变化；一次完成阶段代码审阅，修复只限确认的实现偏差。通过后形成明确代码commit；不自动push/服务器派发。如修复需要改规格或新依赖，停止请示。无需为此次模块重跑baseline或旧D0/G1/Geometry统计。

## 30k直接交付路径

上述必要检查后，下一批动作就是新方法30k训练与结果分析，不另立诊断阶段。使用已有Tool Room工作数据视图与冻结DA3、r2/seed0；不要把会写points3D.ply的数据加载器指向只读canonical根。只核对实际工作视图/兼容环境/明确提交和新输出路径，不新建通用安全包装器。

以下命令中的两个路径变量由现有服务器资产填实；NEW_SELECTOR_OUTPUT必须是新目录，禁止指向baseline。不得猜测服务器路径；准备启动时一次讨论确认即可：

```bash
python -B train.py --source_path "$TOOLROOM_EXISTING_VIEW" \
  --model_path "$NEW_SELECTOR_OUTPUT" -r 2 --iterations 30000 --seed 0 \
  --enable_candidate_selector --candidate_selector_budget 4096 \
  --test_iterations 3000 7000 30000 --save_iterations 7000 30000
```

沿用原cfg_opts、resolved_config、run_identity与train.log；必要的一份实验清单记录commit、实际路径、配置/seed和完整命令。不复制通用launcher/monitor/统计框架，不为这次试跑建立新GT诊断。30k后先交付：7k/30k的train PSNR/L1、耗时、追加统计与独立模型路径、对照基线差值和明显异常；已有mesh提取随后复用，不等待新评价器。

现有固定r2基线：`D:/research_Space/output/AmbiSuR_original/ScanNetpp/Tool_Room/Tool_Room_r2`。30k train PSNR=31.098477172851563，L1=0.016407448239624502；原training_report取训练相机索引5/10/15/20/25模相机数的均值。新结果须比较同一相机顺序/数据/口径，不能把train PSNR写成全图集或新视角指标。PSNR首轮可分析，不单独证明几何收益；继续使用原mesh工具，GT相关评价另行讨论。

## 基线协议下的后续评价（2026-10-22修订）

- **DTU：** 精确15场景/r2按规格§7，原`evaluate_single_scene.py`调用`eval.py`产生`mean_d2s/mean_s2d/overall`，保留原mask/变换/采样/距离过滤；不要自行重写CD或套用旧D0的全Gaussian中心误差。
- **TnT：** 六场景/r2，调用实际主入口`scripts/tnt_eval/run.py`，读取`precision/recall/f-score`，保留原场景对齐/裁剪/config阈值；不是未完成的evaluate_single_scene包装文件，也不统一替换为一个5cm阈值。
- **Mip-NeRF360：** 九场景，室内r2/室外r4，复用`metrics.py`的PSNR/SSIM/VGG-LPIPS，报告每场景及室内/室外均值；不以这里现成run_mip360的use_mono训练选项改变首版DA3合同。原run_tnt也有场景专用设置，正式迁移时保持同配置对照并讨论，不在本轮自动运行benchmark训练。
- **当前Tool Room：** 先完成30k；train日志只作快速预览，补齐三渲染指标时用既有render_set与metrics函数在独立评价根目录处理同一组相机，必要时从旧模型重新渲染同编码图像，而非重新训练。当前test gt/renders均0，train gt为0、train renders为406张JPEG，不能把metrics.py直接指向baseline获得有效测试结果，更不能在baseline根目录写results/per_view。评价视角若属于训练集就如实标注；baseline与新方法保持相同图像编码、大小、相机身份及权重/指标版本。
- **几何结论与范围：** Tool Room/Utility不自动适用DTU/TnT协议，表面评价域/单位/阈值在离线评价批次确定；没有相应证据前只报告渲染变化，不宣称几何提高。DTU/TnT/Mip的论文数值不是本地基线成绩，也不立即派发30场景实验；已有模型/mesh优先复用，不添评价框架。
- **产物与汇总：** baseline补充结果、新方法结果写`EVAL_ROOT/baseline`与`EVAL_ROOT/selector`等独立路径，路径在评价批次填实；直接使用原指标producer返回/JSON字段，记录真实有效场景数。现有stat脚本对缺失结果有占位/分组行为，仅在完整产物条件下使用，不把缺失填零/10或错分组计作科学结果；不为此重建统计系统。

主线每次汇报固定五句，依次说明改动、实际验证（含未运行）、证据、可下结论、下一步范围/成本/输出位置；所有未运行指标明确标为待测，不以源码核对冒充通过。

## Plan Self-Review and Handoff

- [x] 规格§1–5映射到Task1/2；§6公共真实测试与有限日志映射Task1.1–3/Task2.4；§7及用户效率补充映射30k直接交付路径。
- [x] 接口名称/返回类型在两Task一致；五个Review Focus均有归属；五文件边界无新增架构、无新数值门；不变更已批准规格。
- [x] 只写计划，未执行测试、安装、分支、commit/push、训练或GT；不得把本清单当成已通过证据。

建议**当前会话串行实施（native）**：两步共用同一测试和标签接口，无需为每步启动独立实现/审阅代理。计划供用户核对并选择执行方式；确认后一次授权覆盖按清单实施及已讨论的必要验证，不逐函数重复询问。改规格/扩大范围或实际服务器批次仍先讨论。

## 2026-10-22 执行收据

- 已按五文件边界实现并经一次阶段审阅；两项Important（模型源深度NaN回退、原生ALR空验证）已修正并由同一审阅者限定复核关闭。
- 同一测试入口18项：14CPU通过，4原生项因缺AmbiSuR扩展未执行；加现有metadata/Core默认值/seed回归共27项，23通过、4跳过。五文件py_compile及git diff --check通过。
- 清单原生组合命令本地实际尝试后，既有tests.gpu.test_feature_off_dispatch因缺diff_plane_rasterization_ambisur导入失败；不是完整GREEN或feature-off/G0通过。
- 用户明确原生验证在AutoDL完成，沿用一代目既有环境/交接，不在本地安装扩展；一次只读SSH探测连接被关闭，未执行服务器测试/训练。
- 待讨论批次为明确版本提交/推送及AutoDL同入口验证，通过后进入一次30k新方法；不追加500步平台，不重跑baseline。
- AutoDL待执行统一命令：`python -B -m unittest tests.test_candidate_selector tests.gpu.test_feature_off_dispatch -v`；使用原项目兼容环境，必须确认四项原生测试实际运行而非跳过。
- 当前未commit/push/tag、未运行训练/指标、未访问GT或修改baseline；Task2.4/2.5未整体完成，不把CPU检查当作原生验收。
