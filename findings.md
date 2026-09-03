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

### 2026-09-01 审计状态

- 已完整读取规格与交接材料；以下代码事实将在本轮只读审计后补入，未核验前仍以 `docs/research/baseline-audit.md` 的“待代码验证风险”处理。
- 方法收益（N、双可靠性、Abstain、参数路由、生命周期）均继续保持 `hypothesis`，本轮不产生实验结论。

### 训练入口初步事实（已读源码，未运行）

- `train.py::training` 是唯一主训练循环；固定 `setup_seed(22)` 在模块加载时调用，CLI 没有 seed 参数（`train.py:46-52,611-638`）。
- 单次训练迭代先构造基础 `loss`，ALR 在 `train.py:403-445` 额外执行 `loss_unc.backward(retain_graph=True)`，随后 `train.py:450` 再执行 `loss.backward()`；该代码事实与“路由后一次参数更新”的 Core 规格不兼容，且需要 E0/GPU smoke 验证当前分支能否实际运行到该路径。
- `train.py:428-445` 对布尔切片的 `requires_grad_` 属性赋值，既不是逐元素梯度 mask，也没有调用 `requires_grad_(...)`；不能作为 C1–C5 参数路由基础。
- densify/prune 与 multi-view trim 均在 `optimizer.step()` 之前执行（`train.py:480-520`）；其中 `out_observe>0` 已被按视图二值化累计（`train.py:499-509`），可复用扫描时序，但 C6 必须重排拓扑动作到 step 之后并迁移附加状态。
- baseline 的 DA3 metric depth 监督从 `iteration>1000` 开启，用 per-view confidence 分位阈值 mask 后约束 `plane_depth`（`train.py:378-390`）；当前没有 DA3 normal loss、Gaussian 级可靠性或 stop-gradient 证据缓存。

### 参数、renderer 与 GaussianModel 事实（已读源码，未运行）

- 参数由 `arguments.ParamGroup` 根据实例字段自动注册；目前没有 Core 开关、证据刷新周期、可靠性/状态机/路由/生命周期阈值或 CLI seed（`arguments/__init__.py:19-45,47-153`）。默认 truncation、Ray-Color、DA3 depth 和 ALR 是 baseline 行为，E0 不能把它们误当新增 Supporting。
- `GaussianModel` 参数形状为 xyz `[P,3]`、DC `[P,1,3]`、non-DC SH `[P,K,3]`、scale `[P,3]`、rotation `[P,4]`、opacity `[P,1]`；均为 CUDA `nn.Parameter`（`scene/gaussian_model.py:176-205`）。优化器组名是 `xyz/knn_f/f_dc/f_rest/opacity/scaling/rotation`（`scene/gaussian_model.py:207-227`）。
- `compute_weighted_sh_norm` 把 non-DC 系数 reshape 为 `[P,15,3]`，确认 degree-3 写死风险（`scene/gaussian_model.py:587-616`）。设计要求应按当前 `active_sh_degree` 的 `(L+1)^2-1` 动态切片并在证据路径 stop-gradient。
- renderer 目前输出 RGB `[3,H,W]`、radii `[P]`、`out_observe` `[P]` int、`out_all_map` 派生的 primitive normal/depth normal/alpha/depth 等像素图；没有每像素 Gaussian blending weight/ID 或 Gaussian 加权统计输出（`gaussian_renderer/__init__.py:38-216`）。因此 S 只需现有 `out_observe`，但 `T^P/T^G/K` 的严格 `Pool_i(sg(w))` 需要新增 renderer/CUDA 统计接口，或规格批准的不同实现；不能从现有 Python 输出精确恢复。
- 当前 primitive normal 由最短正尺度轴及 rotation 得到，并按相机方向翻转（`scene/gaussian_model.py:149-167`）；`depth_normal` 由 `plane_depth` 的邻域反投影叉积得到且乘 detached alpha（`gaussian_renderer/__init__.py:23-36,210-212`）。
- clone/split/prune 直接在 `GaussianModel` 内替换所有 Parameter 并只迁移/扩展 Adam `exp_avg/exp_avg_sq`；函数不返回候选事件或旧到新索引映射，densification postfix 还会把所有 densification 累积器整体清零（`scene/gaussian_model.py:340-543`）。这是 C6 的接口级缺口。
- checkpoint `capture/restore` 是固定长度 tuple，只含 baseline Gaussian/optimizer 状态；不含任何 EMA、历史中心/法线、仲裁/生命周期/持续计数（`scene/gaussian_model.py:82-125`）。C6/恢复一致性需要版本化扩展且保持旧 checkpoint 可读。
- `AppModel.appear_ab` 实际为 `[1600,2]` 的每视图标量 scale/bias，不是设计稿 Supporting 里的每通道 `[V,2,3]`；本轮 Core 不修改该差异（`scene/app_model.py:9-15`）。

### DA3、相机与多视图输入事实（已读源码，未运行）

- DA3 预处理把每视图 depth/confidence 保存为 `.npy`，训练 loader 从 `estimated_depths/estimated_confs` 读取并用 `sparse_da3_aligned/0/trans.json` 的 scale 对 depth 对齐（`multi_view_priors/estimate_colmap.py:131-160`; `scene/dataset_readers.py:118-135,164-193`）。
- `Camera.depth_dict['depth']` 被构造成 CUDA `[H_P,W_P]` `nn.Parameter`，confidence 保留为 CPU float tensor；训练时 depth 立即 `detach()`，confidence 临时 `.cuda()` 并插值到 renderer 尺寸（`scene/cameras.py:106-112`; `train.py:381-389`）。Core 证据缓存必须显式 no-grad/stop-gradient，并避免误把 DA3 depth 参数加入梯度路径。
- 每个 Camera 已提供 CUDA `camera_center[3]`、`world_view_transform[4,4]`、内参和 `nearest_id`；邻接表由相机中心距离和中心射线夹角生成，默认最多 8 个邻居（`scene/cameras.py:51-146`; `scene/__init__.py:78-123`）。这些可提供 S 的方向向量和双可靠性的 `(v,u)` 候选对，但可靠性必须按规格重新做双向投影/遮挡检查，不能直接把现有 NCC mask 当成可靠性。
- DA3 normal 当前没有缓存；可在证据刷新时用现有 `utils.graphics_utils.normal_from_depth_image` 从深度和相机标定生成世界坐标 `[H,W,3]` normal（`utils/graphics_utils.py:78-85`）。该路径是普通 PyTorch 运算，D0/Core 证据计算应包在 `torch.no_grad()`。
- `Scene.cameras_extent` 已由训练相机中心半径计算，是设计中 `R_scene` 的现有来源（`scene/dataset_readers.py:49-70`; `scene/__init__.py:75-76`）。

### CUDA forward 接口事实（已读源码，未编译/运行）

- extension forward 只返回 `color[3,H,W]`、`radii[P]`、`out_observe[P]`、`out_all_map[7,H,W]`、`plane_depth[1,H,W]` 及内部 buffers（`rasterize_points.cu:35-130`; Python wrapper `__init__.py:48-113`）。没有可供 Python 重建每个 `(i,v,r)` 的 `alpha/T/w`。
- `out_observe[i]` 在每个被处理像素中仅当该 Gaussian 到达前 `T>0.5` 时原子加一（`forward.cu:349-409`），确认其语义是单视图像素命中数而不是视图数或权重和。
- 实际 blending weight 在 CUDA 局部变量中是 `alpha*T`，用于 RGB 与 7-channel `all_map` 混合（`forward.cu:371-392`），但没有汇聚回 Gaussian。严格实现 `Pool_i`、`M_i^P/M_i^G` 需要新增仅在证据刷新启用的 CUDA Gaussian 累加输出；建议输入预计算的 per-pixel evidence/mask，输出 numerator/denominator `[P,E]`，避免导出巨大的 contributor 列表。
- `trunc_sigma` 当前从参数系统到 Python settings、C++ bridge、CUDA forward/backward 全程是单个 float（wrapper `__init__.py:190-206`; `rasterize_points.h:18-75`; `forward.cu:273-297,364-369`）。Core C6 明确禁止逐 Gaussian truncation，因此本轮不得修改该接口。
- wrapper 的 debug forward 分支接收 8 个返回值且未绑定 `out_plane_depth`，而非 debug 分支接收 9 个；随后无条件返回 `out_plane_depth`（wrapper `__init__.py:97-113`）。这是现存 debug-mode 风险，E0 最小 smoke 应覆盖 `--debug` 或明确记录为 baseline blocker，但不得夹入 Core 方法 commit 的无关修复。

### CUDA backward 与梯度隔离事实（已读源码，未编译/运行）

- baseline Ray-Color 在 custom backward 中把 `ray_reg*(c_i-final_pixel)` 加入颜色梯度，再乘 `w=alpha*T` 原子累积到 `dL_dcolors`（`backward.cu:565-609`）；它没有 Python loss 值，也不对 alpha/T 传该正则梯度。
- SH backward 明确把颜色梯度通过 view direction 回传到 `means3D`（`backward.cu:18-139,389-395`），因此 Ray-Color 可能污染 xyz；ALR 两次 backward 又会重复触发这一隐式项。Core 实施必须保持 baseline feature-off 路径原样，但 C4/C5 的双路径梯度提取不能把隐式 Ray-Color 重复执行。
- plane-depth 上游梯度在 CUDA 中同时进入 `all_map` normal/distance 通道，并经 alpha 路径到 opacity、screen mean/conic，再进一步到 xyz/scale/rotation（`backward.cu:483-495,610-661`）。因此设计中的外部先验梯度天然覆盖 `pos/scale/opa`，需要显式分离和 mask，不能假设 depth loss 只更新 xyz/rotation。
- 现有 backward 为每个参数组输出完整 `[P,...]` dense gradient，适合在 Python 侧对已分离的 `g_B/g_P` 做 per-Gaussian mask、限幅和投影；不需要为了 C4/C5 修改 CUDA backward 数学，但必须避免 `requires_grad_` 切片和多次隐式 backward。

### Checkpoint、mesh 与评价流程事实（已读源码，未运行）

- 训练 checkpoint 是 `(gaussians.capture(), iteration)`，恢复时固定 tuple unpack 后重建 optimizer；appearance 权重另存/另载（`train.py:112-115,525-528`; `scene/gaussian_model.py:82-125`; `scene/app_model.py:17-30`）。Core 状态必须进入 Gaussian checkpoint，且旧 tuple 恢复测试是 E0 必测项。
- `mesh_extract/extract_adaptive.py::render_sets/render_set` 从最新/指定 PLY 加载 Gaussian，渲染 train-view `plane_depth`，以 Open3D TSDF 融合并写 `mesh/tsdf_fusion(_post).ply`；自适应脚本会根据相机估计重新设定 max depth/voxel size（`extract_adaptive.py:71-84,88-203`）。阶段比较必须固定同一提取脚本/参数/commit，且使用不同输出目录防覆盖。
- 外观评价由 `metrics.py::evaluate` 读取 `test/ours_*/renders` 与 `gt`，写 `results.json/per_view.json`，指标为 SSIM/PSNR/LPIPS（`metrics.py:24-100`）。DTU/TnT 有各自 mesh 评价脚本和运行封装（`scripts/run_dtu.py:26-46`; `scripts/run_tnt.py:26-46`）。
- 仓库没有 ScanNet++ 专用训练配置或 mesh GT 评价入口。用户随后提供的只读旧结果恢复了 Tool Room 的真实训练/mesh 提取命令与 cfg，但没有 Git SHA、环境、dataset/DA3 checksum 或 GT geometry metric；Utility Room 命令和统一 ScanNet++ evaluator 仍未冻结。
- 当前本机系统 `python` 为 3.14.7 且无 torch/pytest，只适合辅助只读脚本。仓库真实 baseline runtime 合同来自 `environment.yml:12,18-20`：Python 3.10、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、torchaudio 2.7.1+cu128；此前将设计稿 §1 的 Python 3.12/PyTorch 2.8 当作硬要求是错误的。设计稿环境描述待用户批准后另行修正，不影响方法公式。

### 旧 ScanNet++ baseline 结果与本地数据审计（2026-09-02，只读）

- 只读结果根目录：`D:\research_Space\output\AmbiSuR_original\ScanNetpp`。发现 Tool Room `Tool_Room_r2` 与 `Tool_Room_r4` 两次完整 30k 运行；`r2/r4` 是 `train.py -r 2/-r 4` 的图像分辨率缩放，不是 seed。两次 `cfg_opts` 相同，均为 `depth_weight=0.1`、`unc_weight=0.1`、`unc_from_iter=7000`、`ray_color_lambda=1e-5`、`disable_trunc=False,trunc_sigma=2.0`、`eval=False`。
- r2 命令为 `python train.py --source_path /root/autodl-tmp/data/ScanNetpp/Tool_Room/colmap_undistorted --model_path /root/autodl-tmp/output/AmbiSuR/Tool_Room_r2 -r 2`；30k 用时 42:54，最终 1,297,143 Gaussians，train-view PSNR 31.0985。r4 同命令改 `model_path` 和 `-r 4`；30k 用时 31:25，最终 1,052,487 Gaussians，train-view PSNR 32.7740。分辨率不同，PSNR/点数不能作为同协议优劣结论。
- 两次训练均读取 406/406 相机、原 COLMAP pose 和 `sparse_da3_aligned/0`，初始化 200,000 points，并成功经过 7001+ 的 ALR/Ray-Color/multi-view 路径直至 30k。两次 mesh 命令均为 `extract_general.py --max_depth 5.0 --voxel_size 0.005 --sdf_trunc_scale 4.0 --num_cluster 2`；已生成 mesh，但旧结果没有 ScanNet++ GT 数值评价文件。
- 本地 Tool source `D:\dataset\ScanNet++\data\data\d415cc449b_Tool_Room\dslr\colmap_undistorted` 含 406 images 和 406 COLMAP poses，文件名集合与旧输出 `cameras.json` 完全一致；相机为单一 `PINHOLE 1752x1168`，内参也与旧输出一致。source 共 406,295,584 bytes，但本地没有 `estimated_depths/estimated_confs/sparse_da3_aligned`，因此仅凭现有本地数据不能严格复现旧先验。
- 本地 Utility source 有 147 张 `resized_undistorted_images` 且与 147 个 pose 名称一致，但现有 `dslr/colmap/cameras.txt` 是 `OPENCV_FISHEYE`。当前 loader 只接受 PINHOLE/SIMPLE_PINHOLE（`scene/dataset_readers.py:103-113`），Utility 必须先建立经验证的 `colmap_undistorted`，不能直接上传现有 `colmap` 作为训练 source。
- DA3 runtime 文件合同来自 `scene/dataset_readers.py:115-130,164-226`：每张图必须有 `estimated_depths/<完整图像文件名>.npy` 和 `estimated_confs/<完整图像文件名>.npy`，例如 `DSC06274.JPG.npy`；`sparse_da3_aligned/0` 必须至少有 `points3D.bin` 与含 `scale` 的 `trans.json`。`scripts/run_da3_single.sh` 还会生成 `sparse_da3/0` 并对齐成 aligned model。
- 数据安全缺口：`readColmapSceneInfo` 在每次加载时无条件执行 `storePly(ply_path, xyz, rgb)`（`scene/dataset_readers.py:219-237`），会写/覆盖 source 中的 `points3D.ply`。canonical upload 必须保持只读；clean baseline 和 E0 必须从 per-run working view 启动，只复制可写的 `sparse_da3_aligned`，不能把 canonical upload 直接交给 `--source_path`。
- 2026-09-02 用户报告已将 Tool Room source 与 GT 上传到 `/root/autodl-tmp/ambisur_data/{source,gt}/ScanNetpp/Tool_Room`；截图可确认顶层 `images/estimated_depths/estimated_confs/sparse/0/sparse_da3/0/sparse_da3_aligned` 路径存在，但不能证明文件计数、basename 对齐、`sparse_da3_aligned/0` binary 三件套、`trans.json.scale`、GT mesh、数值有限性或校验和。以上在服务器 preflight 返回前只记为 user-reported，不记为已验证事实。
- Utility Room 未上传不阻塞第一次 Tool Room clean-baseline、后续 E0 feature-off、D0 或 C1 Tool seed-0 快速链路；它在 G2 的跨场景确认与最终 Tool+Utility/多 seed 主验证前必须补齐并先完成受支持相机模型的 undistortion。此为当前实验范围边界，不代表可以用单场景结果作最终结论。
- 用户回传的 AutoDL preflight 输出验证：服务器 `main@d6f15c8891a53800d5e3100f95817a7dd7f98e2f`、tracked diff 为空；Python 3.10.21、NumPy 1.26.3、PyTorch 2.7.1+cu128、CUDA 可用、RTX 4090 24GB、`train` import 成功。Tool 有 406 images/406 depth npy/406 conf npy，prior shape 均为 `(336,504)`、float64、全量 basename/finite 检查无报错，alignment scale `0.4221856859435143`，GT mesh 104,149,767 bytes，无 `split.json`。这是用户终端回传证据；正式运行仍须将原始输出和 manifest 写入新 run 目录。
- 首次 Tool `-r 4` 8k clean-baseline path-check 已正常跑到 `8000/8000` 并打印 `Training complete.`；日志进度耗时约 5:21，iteration 8000 前显示 1,209,624 points，已生成 iteration 7000/8000 point cloud 与 iteration 8000 app model。该事实证明 legacy 路径可完整跨过 1001/5001/7001 时序并保存，不等同于 C0 复现通过；仍需全日志错误扫描、7k 指标/点数与旧 r4 对照、配置和 canonical hash 完成审计。
- 8k 完成性审计通过：post-run Git tracked status 为空；canonical dataset manifest SHA 保持 `aad92aa2e0f0d072756b3a56c686d5c1d35f448811ce60ca4360c67dbc3ef255`，GT SHA 保持 `31547a31069f736792d4b13fff76c73483f59239dbbabe1971e115f9ab17171d`；全日志 `Traceback/RuntimeError/CUDA error/AssertionError/NaN/Inf` 均为 0；training-complete 与 7k/8k save 各出现一次，所需 artifacts 全部非空。
- 与旧 `-r 4` 同口径比较：新/旧 3k PSNR 为 `24.056405/24.129358`（差 `-0.072953 dB`），L1 为 `0.0374882/0.0373400`；新/旧 7k PSNR 为 `27.336051/27.326481`（差 `+0.009570 dB`），L1 为 `0.0258267/0.0260480`；新/旧 7k PLY vertices 为 `1,166,901/1,159,656`（新高 7,245，约 `0.625%`）。据此接受 legacy **path-check**，但因未跑 30k/mesh/GT evaluator，不标记 C0 reproduced。
- 旧 `Tool_Room_r4/train.log` 明确命令为 `-r 4`，且其 3k/7k 指标和点数明显匹配新 `-r 4` 而非旧 `-r 2`；但旧 `Tool_Room_r4/cfg_args` 却写 `resolution=2`。这是旧产物内部 provenance 冲突，具体覆盖来源当前未知；“后续某工具覆盖 cfg_args”只能作为 hypothesis。后续协议以实际 train command、运行日志和新 manifest/cfg 为准，旧单独 `cfg_args` 不得作为分辨率证据。
- Tool `-r 2` 30k C0 candidate 于服务器日志正常达到 `30000/30000` 并打印 `Training complete.`；用时 50:13，最终 Loss/Single/Geo/Pho=`0.00939/0.00097/0.00010/0.02130`，points=1,297,647，train L1=0.0164226247、PSNR=31.0740604。对照旧 r2 的 points=1,297,143、L1=0.0164074482、PSNR=31.0984772，新运行分别为 +504（约 +0.0389%）、L1 +0.0925%、PSNR -0.02442 dB；这是强复现信号，但在退出码、全日志、hash、3k/7k、peak GPU 和 artifacts 审计前不标记 C0 accepted。
- C0 training completion audit 通过：exit code 0；commit/status、canonical dataset SHA 与 GT SHA 全部保持；完整日志 0 traceback/runtime/CUDA/assertion/NaN/Inf；7k/30k saves 各一次；峰值 11,966 MiB，低于 22 GB 门槛；所需 train artifacts 均存在。新/旧 r2：3k PSNR `23.381071/23.494688`（-0.113617 dB），L1 相对 +1.478%；7k PSNR `25.939433/25.938409`（+0.001024 dB），L1 相对 +0.227%，PLY vertices `1,477,234/1,486,111`（-0.597%）；30k 差异见上一条。训练路径接受，仍需以冻结旧参数提取 mesh 后才接受整个 C0 candidate。
- 旧 r2 mesh 同口径参考：`tsdf_fusion.ply` 为 13,244,440 vertices / 24,171,471 faces / 671,829,276 bytes；`tsdf_fusion_post.ply` 为 11,741,805 vertices / 22,647,274 faces / 611,443,570 bytes。旧命令固定 `max_depth=5.0, voxel_size=0.005, sdf_trunc_scale=4.0, num_cluster=2`；新 C0 必须原样复用。
- 2026-09-03 用户回传的新 C0 mesh 启动证据：安全门确认 commit、clean worktree、训练 exit 0、磁盘与 runtime 后，`mesh_extract/extract_general.py` 以冻结参数 `max_depth=5.0, voxel_size=0.005, sdf_trunc_scale=4.0, num_cluster=2` 在同一 C0 run 目录后台启动；launcher PID `2317`、mesh PID `2320`。日志已确认读取 iteration 30000、私有 working view、406/406 cameras 与 `cameras_extent=3.394348192214966`。这只证明启动路径正确，不证明提取完成或 mesh 等价；完整 exit/log/PLY/hash 审计仍待回传。
- C0 mesh completion audit 已于 2026-09-03 通过：进程 exit 0，UTC `02:03:12–02:06:24`，峰值 GPU 5,480 MiB；406/406 render 与 TSDF fusion 均完成，`Traceback/RuntimeError/CUDA/device assert/AssertionError/NaN/Inf` 均为 0。Git 仍为 clean `d6f15c8891a53800d5e3100f95817a7dd7f98e2f`，GT SHA 保持 `31547a31069f736792d4b13fff76c73483f59239dbbabe1971e115f9ab17171d`，mesh 期间 canonical source/GT 新修改文件数均为 0。
- 新 C0 raw mesh 为 13,063,782 vertices / 23,917,760 faces / 663,653,267 bytes，相对旧 r2 分别 `-1.364029%/-1.049630%/-1.216977%`；post mesh 为 11,620,625 vertices / 22,439,335 faces / 605,468,503 bytes，相对旧 r2 分别 `-1.032039%/-0.918163%/-0.977207%`。旧输出没有 commit/environment/dataset manifest，不能作为 bitwise oracle；结合相同命令、当前完整 provenance、训练 30k PSNR 仅 `-0.02442 dB`、最终 Gaussian 数仅 `+0.0389%`、完整训练/mesh 无错误，接受新运行作为当前 frozen C0 reference。该接受只建立 Core ladder 的可复现锚点，不是 GT 几何质量结论；ScanNet++ evaluator 仍须在解释方法几何收益前冻结。

### 规格缺口与正式实施澄清（非实验结论）

1. **C1 的先验梯度合同已于 2026-09-01 选择 A 并正式澄清。** 设计 `7.1` 新定义 `L_P=lambda_Pd L_Pd+lambda_Pn L_Pn`，但未给 `lambda_Pd/lambda_Pn` 初值；现有 baseline 是 `depth_weight=0.1` 的 DA3 depth 加 `unc_weight=0.1` 的 Dual-End ALR，后者的 mask/公式与新 `L_Pn` 不同（`train.py:378-445`; design `:445-490,821-840`）。为维护 C0–C6 严格单变量消融，Core C1–C6 沿用 baseline 原有先验合同、启用时序、权重和有效区域；C1 唯一方法变量是对 clean baseline prior geometry gradient 逐 Gaussian 乘 `N_i`。设计稿原文不修改；新 `L_Pd/L_Pn` 暂列 Core 之后的独立 Supporting 候选，未来若实施必须新建独立阶段、tag 和消融，不能进入既有 `c1-need-gate`。
   - 必须满足 `g_baseline_total = g_B_residual + g_P_clean` 和 `g_C1 = g_B_residual + N_i*g_P_clean`。
   - `g_P_clean` 只含 baseline DA3 depth 与 Dual-End ALR 的几何梯度，不含 Ray-Color，对 SH/曝光梯度严格为 0；baseline Ray-Color 和所有其他非先验贡献归入 `g_B_residual`。
   - GPU 单步 oracle 必须验证 `N=1` 重现 baseline total、`N=0` 只移除 clean prior、SH/曝光不受 N 影响、C1–C5 densification proxy 保持 baseline total-gradient 定义且每轮只有一次 optimizer step。
   - oracle 必须覆盖 DA3-only、Ray-Color active 和 ALR+Ray-Color active 的 strict-boundary 前后，并逐组比较 xyz/rotation/scaling/opacity/SH、viewspace proxy、step 后参数以及有/无 ALR 路径。若无法在预先冻结的容差内等价，或需要用 `out_observe/out_all_map` 等近似代替严格梯度，立即停止，不得进入 C2。
2. **K joint-validity 已于 2026-09-02 选择 A 并正式补入设计。** 原设计的 `Pool_i` 规定分母不足应由后续 V 置零，但 `K_i` 没有 PG-overlap validity；当 `V^P=V^G=1` 而 P/G overlap 为零时，零 pooled error 会产生伪 `K\approx1`（原 design `:75-112,366-395`）。批准合同定义 `Z_i^{PG}=sum(sg(w)m^{PG})`、`V_i^{PG}=1[Z_i^{PG}>10^{-4}]`，复用已有 `tau_Z`。只有当前 `V_i^{PG}=1` 才计算有效 `K_raw` 并初始化/更新 K EMA；invalid 时保留但不得消费历史 EMA，新 Gaussian 初始为未初始化。
   - Bypass 和单方可靠状态不要求 `V_i^{PG}`；只有双方均可靠、需要判断一致/冲突时才要求它。
   - 双方均可靠且 joint-invalid：D0/C3–C6 为 Abstain，C2 回退 `g_B`；Delta 只诊断，不选边。
   - C4–C6 继承 C3 状态，不能自行用历史 K/Delta 绕过 gate。coverage 近零时只报告并停止，不得改用方案 B 或降低阈值。
3. **生命周期 `s_i`/`ell_i` 语义已于 2026-09-02 选择 A 并正式补入设计。** 原 §8.1 把五状态仲裁符号 `s_i` 与 `Normal/Confirmed/Protected` 生命周期枚举直接比较，而后文又以 `ell_i` 表示生命周期，原式会造成枚举域混用。批准合同固定：`s_i` 是经过 `H_enter` 迟滞的稳定五状态仲裁结果，`ell_i` 是生命周期状态；正常的 `ell_i=M(s_i)` 只在证据刷新执行，所有 clone/split/prune gate 只消费 `ell_i`。
   - 新 Gaussian 在 topology 提交时强制 `ell_i=Probation`，至少保持 `C_prob=500` 个 optimizer iteration；期间仍可记录 `s_i`，但不得覆盖 `ell_i` 或驱动先验强路由/topology。
   - 满 500 轮不会在刷新间隔中途释放；只在随后第一次证据刷新，按当时稳定的 `s_i` 完成映射。`d_i` 只计 `ell_i` 在证据刷新上的连续持续次数，状态变化时重置为 1，普通 iteration 不更新。
   - 这是 architectural specification clarification，不是已运行验证；实现仍须用状态映射、边界时序、gate-domain、checkpoint/resume 测试证明。
4. **Probation truncation 已于 2026-09-02 选择 A 并正式补入设计。** 设计 §8.2 原先在 Probation 行为内写有 `trunc_sigma=2.0`，§9.4 又把状态驱动逐 Gaussian truncation 列为 Supporting；用户明确禁止 Core 实现生命周期状态驱动的逐 Gaussian 截断。当前 baseline 的 `arguments.ModelParams` 定义全局 `disable_trunc=False,trunc_sigma=2.0`（`arguments/__init__.py:64-67`），renderer 从 Python settings、C++ bridge 到 CUDA forward/backward 也只接受全局 float（`gaussian_renderer/__init__.py:111-112`; `diff_plane_rasterization_ambisur/__init__.py::GaussianRasterizationSettings`; `rasterize_points.h::RasterizeGaussiansCUDA`; `cuda_rasterizer/forward.cu::FORWARD::render`）。批准合同固定：Core C1–C6 原样沿用该全局 baseline 配置，Probation 不读写独立 truncation 状态，不新增 per-Gaussian Tensor，不修改 Python/C++/CUDA 接口；§9.4 只保留为 G0–G2 后的独立 Supporting 候选。
5. **C1–C5 的 screen-space densification proxy 已获正式澄清。** renderer 的 `viewspace_point_tensor.grad` 是 baseline densification candidate 的输入（`train.py:476-486`; `scene/gaussian_model.py::add_densification_stats`）。C1–C5 必须固定使用 baseline total-gradient proxy，不随 N/仲裁路由，且不得提前改变 topology；C6 才增加生命周期对候选事件的执行门控。该项是已批准实施合同，但尚未通过 GPU oracle 验证。
6. `metric_depth_normal_weight_base`、camera depth optimizer、`_knn_f` 与 `max_weight` 在当前训练入口未被消费（`arguments/__init__.py:129-130`; `scene/cameras.py:195-272`; `scene/gaussian_model.py`）；不得把这些未使用字段当作已存在的 `lambda_Pn` 或 Core 状态来源。

### 测试与复现基础事实

- 仓库没有项目级 `tests/`、pytest 配置或 CPU 单元测试；命中的 `pyproject.toml` 仅属于 vendored COLMAP（`rg --files` 审计）。
- 当前 PowerShell 的 Python 3.14.7 环境没有 `torch` 或 `pytest`；本轮不能执行项目 CPU tests。E0 开始前必须提供与仓库兼容的测试环境，或在已部署的 AutoDL Python 3.10 环境补装 pytest 后运行；pytest 缺失不阻塞不含新增代码的 clean-baseline。
- `train.py` 模块加载先调用 `setup_seed(22)`，CLI main 随后调用 `safe_state()` 把 Python/NumPy/Torch seed 固定为 0；没有可记录的 seed CLI（`train.py:46-52,611-638`; `utils/general_utils.py:121-142`）。E0 应增加默认值为 0 的显式 seed 参数并证明默认 trace 不变。
- `prepare_output_and_logger` 只写 argparse Namespace 文本 `cfg_args/cfg_opts`，没有 resolved JSON、Git SHA、dirty state、环境或 seed manifest 自动记录（`train.py:533-556`）；正式实验前需补齐可复现元数据，但写入新的唯一输出目录，绝不触碰 baseline 结果目录。

## Experiment Findings

每项必须引用对应 manifest、commit、scene 和 seed。不得把单次运行写成稳定结论。

### E0 implementation facts（2026-09-03，尚未完成服务器验证）

- Commit `68922cc` 新增 E0 测试合同；commit `b2c46db` 新增 `reliability.config.CoreConfig`、`reliability.runtime::{select_training_path,build_checkpoint_payload,build_resolved_config,collect_run_identity,write_run_metadata}`、参数开关/seed、`safe_state(seed)` 和只读 comparator。Commit `0601056` 完成已观察 RED 所需的 legacy dispatch/signature/tuple checkpoint 接线；训练数值 feature-off 等价仍未验证。
- 本机 bundled Python 3.12（无 torch/pytest）以 `unittest` 执行 19 项非 GPU 测试全部通过；覆盖默认全关、六级严格嵌套、shadow、seed 重放、legacy tuple、元数据 Git dirty identity、run 文件只读比较和 PLY hash 差异。项目实际 Python 3.10/CUDA integration 仍待 AutoDL。
- `tests/gpu/test_feature_off_dispatch.py` 已先于 `train.py` 接线写入。AutoDL 在 clean `research/core-routing@b2c46db49e3465da7ff5cfda56a7ddd30be6f02c`、Python 3.10.21 上已观察到目标 RED：`ImportError: cannot import name 'build_checkpoint_payload' from 'train'`；这证明 E0 train integration 仍缺失，不是环境、数据或 CUDA import 故障，现可按 TDD 补最小入口接线。
- AutoDL 在 clean `research/core-routing@02ac3b9700ea53a9f723ef3f62c3c8cac1b15d42` 上已观察到第二个精确 RED：前三个 train 接口测试通过，只有 metadata integration 因 `prepare_output_and_logger` 仍为二参数签名失败。该证据授权的最小改动仅为 logger 写两份 JSON、CLI 显式传 seed/CoreConfig；不能据此声称 E0/G0 完成。
- AutoDL 在 clean `research/core-routing@a7d04d4bbd28aa025f1d09373e8e7d1e615bf688`、Python 3.10.21 上完成对应 GREEN：4/4 integration tests PASS（0.097 s），测试后工作树 clean。已验证范围仅为 train 模块 dispatch/signature、feature-off legacy tuple checkpoint 和 logger 元数据；尚未验证一次真实训练的数值/梯度/topology 等价。
- AutoDL 在 clean `research/core-routing@7223f919e8e015f1b1eed2d94d6855aed3b4eb29` 上完成 E0 full component gate：23/23 tests PASS（0.211 s），CLI help 返回 0 且八个 E0/Core flags 全部可见，测试后工作树 clean。事实边界仍是 component/CLI；G0 需要同 snapshot/seed/config 的 baseline 与 E0 feature-off 成对训练和只读比较。
- Paired-500 首次 launcher 在训练前因错误强制原始 `sparse/0/cameras.bin` 而安全停止；无 run root、Git clean。服务器只读审计确认原始 pose 为完整 `cameras/images/points3D.txt`（184 / 171,471,240 / 36,248,710 bytes），camera model=`PINHOLE`；`sparse_da3` 与 aligned prior 为 binary，aligned `points3D.bin=11,800,008 bytes`、`trans.json=623 bytes`、scale=`0.4221856859435143`，无 `split.json`。这与 `readColmapSceneInfo` 的 bin→txt fallback 一致，数据无需重传。
- 第二次 launcher 因把 `estimated_depths` 总文件数 812 与训练条目数 406 混淆而在训练前停止，无 run root。只读审计确认 812=406 `.npy` + 406 `.jpg` 预览；406 image 的期望 `<完整图像名>.npy` 与 depth/conf 集合完全相等（missing=unexpected=0）。训练合同只消费 `.npy`，预览文件不得计作缺失或额外先验。
- E0 paired-500 baseline 半程已在 `c0-baseline@d6f15c8891a53800d5e3100f95817a7dd7f98e2f` 完成：pair `pair_20260903T090116Z`，Tool Room `-r 2`、semantic seed 0、500 iterations；exit 0，training complete=1，0 error/nonfinite，initial/final 200,000 Gaussians，train L1 `0.0749632939696312`、PSNR `19.052456665039063`，PLY SHA `01407a4da71819f5c13477111f912a7dbcc8ccf6aa489c300c7612b1233f0c5c`，peak 4,769 MiB，canonical manifest SHA `aad92aa2e0f0d072756b3a56c686d5c1d35f448811ce60ca4360c67dbc3ef255` 前后不变，Git clean。服务器 detached HEAD 指向 baseline 是按协议临时 checkout，不表示 `research/core-routing` 分支被移动。
- 同一 pair 的 E0 all-off 半程在 exact `a26082154889ed539322425347af5a57a859a52f` 完成训练且 metadata PASS：resolved path=`legacy`、enabled features=[]、identity commit/dirty/seed 正确；exit 0、0 error/nonfinite、200,000 points、peak同为4,769 MiB、canonical/Git clean。但 strict state comparator 首个失败为 capture index 1 `_xyz`；E0 train L1=`0.07498596683144569`（baseline `+2.267286e-5`），PSNR=`19.047444152832032`（`-0.0050125122 dB`），PLY SHA=`1fa9e9f…6b75d`，故 comparator differences=`L1,PSNR,PLY SHA`，E0 gate FAIL。差异原因尚未验证；不得写成 metadata overhead、CUDA nondeterminism或 seed bug的结论。

## 公式输入—代码来源映射（2026-09-01 只读审计）

约定：`P` 是当前 Gaussian 数，`V` 是训练视图数，`H×W` 是当前 renderer 分辨率，`E` 是一次证据累加的像素通道数。下表的 Core 统计均应为 CUDA float32/bool/int tensor、`requires_grad=False`，默认每 `Delta_obs=1000` iteration 在证据刷新中更新；候选/稳定状态在刷新间保持不变。现有代码事实不等同于已验证运行结果。

| 设计量 | 现有变量/路径 | 目标形状、设备、更新频率 | 新 renderer/CUDA 输出 | 梯度边界 | 当前缺口 |
|---|---|---|---|---|---|
| 活动 SH 系数 `Theta_i` / 阶数 `L` | `GaussianModel._features_rest`, `active_sh_degree`；`scene/gaussian_model.py::compute_weighted_sh_norm` | 系数 `[P,(L+1)^2-1,3]` CUDA；A 刷新时读取 | 否 | A 分支 detach/no-grad | 现函数固定 `[P,15,3]` 且读取 max-degree storage；需按 active degree 动态切片 |
| `u_i,q_L,q_H,A_i` | 可由上述 SH tensor 和 `torch.quantile` 计算；当前 renderer 只算 Dual-End 二值 mask | `u,A:[P]` CUDA float32；每次证据刷新；`q_L/q_H` 标量 | 否 | 全部 stop-gradient；A 进入 EMA | 缺连续 A、退化分位保护、EMA/valid mask、degree 1/2 测试 |
| 单视图命中 `O_iv` | `render_pkg['out_observe']`; CUDA `forward.cu::renderCUDA` | 每视图 `[P]` CUDA int32；证据刷新遍历所有训练相机 | 否 | 非可微；先做 `>0` | 已有；必须禁止把像素命中数直接求和解释为视图数 |
| `o_iv,M_i,S_count` | `out_observe>0`；baseline trim `train.py:499-509` 已按视图累计 | `o:[P]` bool/临时，`M:[P]` int32，`S_count:[P]` float32；每次刷新 | 否 | no-grad | 需从 trim 中抽出只读统计，D0 不 prune |
| `d_iv,q_view,D_i,S_angle,S_i` | `Camera.camera_center[3]`, `GaussianModel.get_xyz[P,3]`; `scene/__init__.py` 已汇总相机中心 | `[P,3]` 临时和 `[P]` CUDA；每次刷新 | 否 | `mu_i` detach；S 进入 EMA | 缺方向和/离散度/`M<2` 分支及 `theta_c=30°` 测试 |
| `N_i` | 由 EMA 后 `A,S` | `[P]` CUDA；每次刷新重算、无二次 EMA | 否 | stop-gradient；C1 起只作梯度系数 | 缺纯函数与边界测试 |
| DA3 `D_v^P,C_v^P` | `Camera.depth_dict`; loader `scene/dataset_readers.py::readColmapCameras` | 原始 `[H_P,W_P]`; depth 当前 CUDA Parameter、conf CPU tensor；刷新时插值到 `[H,W]` CUDA | 否 | 必须 detach/no-grad；不得优化 DA3 | 已有原始量；需有限/正值 mask、per-view 5/95% 归一化与缓存策略 |
| DA3 normal `n_v^P` | 可复用 `utils.graphics_utils.normal_from_depth_image` 和相机标定 | `[H,W,3]` CUDA float32；刷新时生成或缓存 | 否 | no-grad | 当前未生成/缓存；需世界坐标单位化、边界/无效 normal mask 测试 |
| rendered `D_v^G,n_prim,n_depth,alpha` | `plane_depth[1,H,W]`, `rendered_normal[3,H,W]`, `depth_normal[3,H,W]`, `rendered_alpha[1,H,W]`; `gaussian_renderer.render` | 每视图像素图 CUDA；刷新预渲染 | 否（已有） | 证据副本 detach；训练 renderer 保持原 autograd | 已有；需避免把乘过 alpha 的 `depth_normal` 当单位 normal，可靠性前重新单位化并单独用 alpha mask |
| `(v,u)`、投影/遮挡 mask、`e_d^X,e_n^X,k^X` | `Camera.nearest_id` 提供候选对；`GaussianModel.get_points_from_depth/get_points_depth_in_depth_map` 和训练 multi-view 代码提供投影先例 | per-source pixel maps `[H,W]` CUDA；每次刷新、逐源/邻居流式计算 | 否 | 全部 no-grad | 现有训练 mask/`pixel_noise` 不等于规格的相对深度、5% 遮挡与世界 normal 误差；需新纯函数 |
| `sg(w_ivr)` 与 `Pool_i` | CUDA forward 局部 `alpha*T`; Python 不可见 | 建议可选 pixel evidence `[E,H,W]` → Gaussian sums `[P,E]` CUDA；仅证据刷新 | **是**：新增可选 forward-only Gaussian evidence accumulator；feature-off 传空 tensor、跳过原子累加 | 输出 detach，无 backward 槽；绝不写 Parameter `.grad` | 严格 `T^P/T^G/K` 的主要接口缺口；不能用 `out_observe` 或 `out_all_map` 近似 |
| `E_P_conf,E_P_mv,M_P,V_P,T_P,r_P` | DA3 conf、P 跨视图 pixel maps、上述 Gaussian sums | 全部 `[P]` CUDA；刷新；T 仅在有效统计时 EMA，V 按本次支持 | 需要同一 accumulator 的 numerator/denominator 列 | no-grad | 缺 Gaussian pooling、per-source `b_iv^P` 支持计数、T/V 分离和 zero-denominator 测试 |
| `E_G_mv,E_G_dn,E_G_stab,M_G,V_G,T_G,r_G` | rendered maps；中心 `get_xyz`; Gaussian normal `get_smallest_axis/get_rotation_matrix` | `[P]` CUDA；刷新；另存 `prev_xyz[P,3]`, `prev_normal[P,3]`, `hist_valid[P]` | mv/dn pooling 需要 accumulator；稳定性不需要 | no-grad；历史不继承新点 | 缺历史 state、首次无效、角度绝对内积、topology migration |
| `Z_i^{PG},V_i^{PG},K_i^{raw},K_i,Delta_i` | PG 同像素 depth/normal error + accumulator；`Delta=rP-rG` | 各 `[P]` CUDA；每次刷新；`tau_Z=1e-4`；K 仅 valid 时 EMA，Delta 即时派生 | PG support/K pooling 需要 accumulator | no-grad；历史 K 与当前 validity 分离 | joint-validity 合同已批准但未实现：invalid 不计算/消费 raw、不更新 EMA；新点 K 未初始化；需要 coverage/EMA/state 测试 |
| `H_P,H_G,hat{s},h,s` | 全部由 `N,T,V,K,Delta` | state/candidate `[P]` int8，run length `[P]` int16；每次刷新 | 否 | 非可微 | 缺五状态完整优先级、Henter=3、new-point neutral init 与全分支测试 |
| `g_B,g_P` | 当前 `loss`, depth loss, `loss_unc`; CUDA backward 返回各 Parameter dense grad | xyz `[P,3]`, rotation `[P,4]`, scale `[P,3]`, opacity `[P,1]`, SH DC/rest `[P,...]`; 每训练 iteration | C1–C5 不需新 CUDA backward | 按已批准 A 合同，用 `autograd.grad` 收集 baseline total 与 `ray_reg=0` 的 clean baseline prior，不累积 Parameter `.grad`；令 `g_B_residual=g_baseline_total-g_P_clean`，最终路由后只赋一次 `.grad`/step | 合同已闭合但尚未 GPU 验证；ALR 额外 backward 和隐式 Ray-Color 使 residual decomposition 数值/性能风险高，失败必须停止而非近似 |
| prior clip、C1/C2/C3/C4 weights、C5 projection | 无现有执行器；optimizer 参数组名见 `GaussianModel.training_setup` | 与各 Parameter 同 shape/device；每 iteration | 否 | mask/clip/projection 后赋 `.grad`; `gP_SH=0`, exposure 只 gB | 缺 `GradientRouter`、有限性检查、group median norm、coarse/fine routing/projection tests |
| lifecycle `ell,z,c,d,born_iteration` 与 topology mapping | baseline `densify_and_clone/split/prune`, trim prune；均直接替换 Parameter | `ell:[P]` int8；计数/生成轮 `[P]` int32/int64；candidate 每 densification event，cooldown 每 iteration，`ell=M(s)` 与 `d` 仅每 evidence refresh；新点 Probation 至少 500 iteration 并到下一刷新才释放；不含 truncation state | 否 | no-grad、optimizer step 后执行；Probation 在 C6 覆盖为 `g_B`；全局 baseline truncation 不变 | 缺候选/提交分离、旧到新映射、Probation、保护/隔离、Adam/EMA/state 统一迁移；`s/ell` 与全局 truncation 合同已批准但未实现/验证 |
| checkpoint/diagnostics | `GaussianModel.capture/restore` 固定 tuple；TensorBoard/PLY/JSON 现有输出 | Core state dict 随 checkpoint；diagnostic `.npz/.jsonl` 独立输出 | 否 | 不含 GT 派生标签 | 缺 versioned Core checkpoint、legacy tuple 兼容、offline D0 GT join、resolved config/commit/seed metadata |

结论：现有代码具备几乎全部**原始信号**（SH、相机、DA3 depth/conf、渲染 depth/normal/alpha、Gaussian 参数），但不能直接取得严格 Gaussian-weighted `T^P/T^G/K`，也没有历史、EMA、状态、梯度分离或 topology state。至少需要一个仅在证据刷新启用的 renderer/CUDA Gaussian evidence accumulation 输出；因此“设计量都能从现有代码直接取得”的答案是否定的。

## 规格—代码差距清单（2026-09-01）

| # | 文件与符号 | 当前实现（代码事实） | 规格要求 | 最小改动 | 主要风险 | 必需测试 |
|---:|---|---|---|---|---|---|
| 1 | `arguments/__init__.py::{ModelParams,OptimizationParams}`, `train.py::__main__/prepare_output_and_logger`, `utils/general_utils::safe_state` | 无 Core 开关/显式 seed；实际 seed 被 safe_state 固定 0；只写 Namespace 文本 | 独立嵌套开关、可复现 seed/config/commit | 增加六个 Core bool、shadow mode、证据参数与 `--seed=0`；校验非法非嵌套组合；feature-off 走 legacy 分支；写 resolved JSON | 默认值或 parser 语义改变 baseline | 默认全 off、非法组合拒绝、seed=0 trace、feature-off checkpoint/output 比较 |
| 2 | `scene/gaussian_model.py::compute_weighted_sh_norm` | 固定 15 non-DC coeffs，Dual-End 二值 | 动态连续 A、10/95 分位 | 抽 CPU/CUDA 纯函数，按 `active_sh_degree` 切片；保留 legacy 函数供 feature-off | active/max degree 混淆、分位相等 | degree 1/2/3=3/8/15、已知能量、常量输入、空 P |
| 3 | `train.py` multi-view trim；`gaussian_renderer.render`; `Camera.camera_center` | 每 1000 view-count 后直接 prune；无方向覆盖 | `O>0` 二值、M、方向离散度、S；D0 不动作 | `EvidenceAccumulator.refresh_observation` 复用全相机扫描；trim legacy 行为保持独立 | 把像素数当视图数；刷新时 topology 变化 | 大 footprint 不增 M、同向/反向方向、M<2、S 边界 |
| 4 | renderer wrapper/C++/CUDA `renderCUDA` | `w=alpha*T` 只在 kernel 局部 | 所有 Pool 精确使用 `sg(w)` | 可选 `pixel_evidence[E,H,W]` 与 `out_gaussian_evidence[P,E]`；空输入零开销分支；无 backward | 原子累加数值误差、接口错位、显存/时间 | CUDA vs CPU/gradient-oracle weighted sum、空输入 feature-off、NaN/shape/device |
| 5 | `scene/dataset_readers.py`, `scene/cameras.py`, `utils/graphics_utils.py`, training multi-view block | 有 DA3 depth/conf 与投影先例；无 P reliability | robust conf、P depth/normal reprojection、support/T/V | 新 `reliability/evidence.py` 纯函数 + streaming view-pair cache | 坐标系/align_corners/遮挡符号；缓存超显存 | identity/known transform、out-of-frame、behind-camera、5% occlusion、normal sign flip |
| 6 | `gaussian_renderer.render`, `GaussianModel.get_smallest_axis/get_rotation_matrix` | 有 G depth/two normals/primitive normal；无 history | G mv/dn/stability、history-valid、T/V | EvidenceAccumulator 保存上次中心/normal；topology 后旧点迁移、新点清零 | alpha-multiplied normal 未重归一化；argmin scale 变化 | first refresh invalid、move/rotation formula、abs dot、alpha mask、new-point history reset |
| 7 | design `3.2/5.4/6/11.1/11.2`; new `reliability/evidence.py::{compute_pg_consistency,KEMAState.update}` | 代码无 K/PG-overlap validity；原规格零分母可伪造 `K≈1` | 已批准 A：`V_pg=1[Z_pg>1e-4]`；仅 valid 更新/消费 K；joint-invalid 的 D0/C2/C3+ 行为固定 | accumulator 返回 `Z_pg/V_pg`；`KEMAState` 显式保存 value/initialized/current_valid；state/router 强制 gate；diagnostics 分层 coverage | 历史 EMA 泄漏、new-point 初始化为1、coverage塌缩、Delta绕过 | zero/partial support、0→1/1→0 EMA、历史 K 不消费、Bypass/单方不受影响、D0/C2/C3+ 表、feature-off隔离、coverage统计 |
| 8 | new `reliability/arbitration.py::{candidate_state,StateMachine.update}` | 无五状态/迟滞 | 完整优先级、Henter=3、Probation 中性 | int enum + 向量化分支 + candidate streak；所有阈值集中配置 | 分支优先级/边界 `>` vs `>=` | 表驱动覆盖每个 case、阈值等号、抖动序列、new/pruned migration |
| 9 | `train.py::training_report`; new diagnostics/offline evaluator | TensorBoard 仅 loss/PSNR；无 Gaussian evidence/state | D0 只记录，GT 仅离线标签 | 训练写无 GT 的 evidence/state snapshots；独立脚本读取 snapshot+GT mesh 输出 AUROC/AUPRC/risk/state metrics | GT 泄漏进 checkpoint；覆盖原结果 | training code grep/contract 禁止 GT path、snapshot schema、offline join、D0 gradients/topology unchanged |
| 10 | `train.py` depth/ALR blocks；design `7.1/11.2` | baseline depth + Dual-End ALR；ALR 单独 backward；设计新 `L_Pn` 不等价且权重未定义 | 已批准 A：C1 只增加 N 门控，保持 baseline prior 的时序/权重/有效区域；新 `L_Pd/L_Pn` 移至 Core 后 Supporting 候选 | 严格抽取 clean baseline prior geometry gradient；设计稿不改；未来新 prior 单独阶段/tag/消融 | residual 可能无法精确分离；Supporting 候选不得回填 C1 | strict-boundary GPU gradient oracle、`gP_clean` SH/exposure=0、N=0/1、ALR on/off、禁止近似 |
| 11 | new `GradientRouter.route_c1/c2/c3`; `train.py` step | 无 per-Gaussian coarse executor；baseline 两次 backward 各自注入隐式 Ray-Color | C1 gate、C2 no-Abstain、C3 Abstain；`N=1` 仍须复现 baseline total gradient | 以原语义收集 `g_baseline_total`，另用 `ray_reg=0` prior graph 得到 `gP_clean`，令 `gB_residual=g_baseline_total-gP_clean`；路由后统一赋 grad；C1–C5 screen proxy 冻结为 baseline total | 至少额外 prior render/backward 的成本、graph retention、residual 数值误差；无法容差等价即停 | 每参数组/屏幕 proxy/step 后参数 oracle、coarse table、Abstain geometry zero/SH base、一次 optimizer step、densify candidate parity |
| 12 | `GaussianModel.training_setup` groups；new `GradientRouter.route_c4` | 参数组存在但无路由；`knn_f` 未使用 | pos/scale/opa/SH 表；prior clip；exposure base only | 映射 xyz+rotation→pos，DC/rest→SH；组内 per-Gaussian clip/weights；未规范 `knn_f` 保持 base/None | 不同 shape 拼接、None/zero grad、median empty | C4 全表、clip bound、empty median、prior SH/exposure zero、finite checks |
| 13 | new `GradientRouter.project_conflicts/route_c5` | 无 projection | 仅 retained dual paths 且 dot<0；dominant 按状态/可靠性 | 先 clip 后 projection，再 mask；按 Gaussian/组计算 | 符号/epsilon/零 dominant、误投影单路状态 | 负/正/零 dot、post-dot>=0、dominant unchanged、C4 off 等价 |
| 14 | `GaussianModel::{densify_and_clone,densify_and_split,prune_points,densify_and_prune}` | 候选和动作耦合；Parameter 被替换；只迁 Adam；无 mapping | D0 起 EMA/state 对齐；C6 候选/两阶段提交 | 抽 candidate masks 与 apply；返回 `new_to_old`（新点=-1）；legacy wrapper 保留相同随机调用次序 | feature-off 数值改变、split/clone 顺序、旧 grad 丢失 | legacy topology parity、clone/split/prune mapping、Adam shapes/values、state length |
| 15 | `train.py` densify/trim/step 顺序；new `LifecycleManager` | topology 在 optimizer step 前；trim 独立 prune | step old params 后 lifecycle；Normal/Confirmed/Repair/Protected/Quarantine/Probation；z/c/d；Probation 只 `g_B` | Core-active C6 路径重排为 step→candidate→manager→apply→migrate；C1–C5 不 gate topology；保存 `born_iteration` | 动量/grad 应用对象错位、保护失效、状态抖动、Probation 提前释放 | event streak reset、cooldown、protected prune veto、quarantine 条件、499/500/下一刷新、step-before-replace spy、resume |
| 16 | design `§6/§8/§8.1` lifecycle gate；new `LifecycleManager.update_from_arbitration` | 代码无对应；原规格把 `s_i` 与 `ell_i` 混用 | 已批准 A：`s` 仅稳定仲裁、`ell` 仅生命周期；映射和 `d` 仅证据刷新；所有 gate 用 `ell` | 以独立 enum 和显式 update 接口实现；新点 `ell=Probation,d=0,born_iteration=current`；满 500 后下一刷新映射 | enum 错域导致 gate 全错；iteration 级更新造成提前释放/持续计数膨胀 | 五状态映射、非刷新不映射、gate 用 ell、Probation 499/500/下一刷新、d 只刷新、checkpoint/resume |
| 17 | `GaussianModel.capture/restore`, `train.py` checkpoint | 固定 legacy tuple；无 Core state | checkpoint 恢复所有 EMA/history/state/counters/optimizer | feature-off 继续写 legacy tuple；Core-active 写 versioned dict；load 同时接受两种 | 旧 checkpoint 失效；resume 与 uninterrupted 不等价 | legacy load、Core roundtrip、resume one-step parity、无 GT fields |
| 18 | `metrics.py`, `mesh_extract/*`, `scripts/eval_*`; old Tool `train.log/extract_general.log` | 有通用 TSDF、外观、DTU/TnT；无 ScanNet++ entry；已恢复 Tool r2/r4 全训练命令和 mesh 参数，旧协议无 split/eval，但无 GT metric/Utility command | Tool/Utility 同输入/分辨率/mesh/GT 协议、manifest 可复现 | 以恢复命令建立 C0；冻结 r4-smoke/r2-primary 角色；新增只读 ScanNet++ evaluator wrapper/manifest，不改 GT 或旧结果 | 评价漂移、不同 resolution 误比较、结果覆盖、GT 进训练 | identical input list/config/checksum、output path uniqueness、read-only canonical/view guard、metric golden fixture |

### Baseline audit 风险一致性结论

`docs/research/baseline-audit.md` 的 11 项代码事实全部与当前 HEAD 源码一致：动态 SH 缺口、`out_observe` 语义、trim 扫描、detached appearance SSIM、Ray-Color→means 泄漏、ALR 重复 backward、切片 `requires_grad_` 无效、全局 truncation、step 前 topology/状态迁移、DA3 双重参与、最短轴法线与视角翻转均已定位到上表证据。此结论是源码一致性审计，不是服务器运行验证；其中 ALR 属性赋值是否立即报错、CUDA 数值与显存行为仍需 GPU smoke。

## Decisions

| Date | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-09-01 | 用户批准选择 A：Core C1–C6 沿用 baseline DA3 depth `0.1` + Dual-End ALR `0.1` 及原时序/有效区域；C1 只以 `N_i` 门控 `g_P_clean` | 用户正式实施澄清；`train.py:378-445`; design §7.1 | 新 `L_Pd/L_Pn` 不进入 Core ladder，列为 Core 后 Supporting 候选；必须通过 strict-boundary GPU residual oracle，失败/需近似则禁止进入 C2 |
| 2026-09-02 | 用户批准 K joint-validity 方案 A：`V_pg=1[Z_pg>tau_Z]`，`tau_Z=1e-4`；invalid 不更新/消费 K | 用户 architectural specification clarification；design §3.2/§5.4/§6/§11.1/§11.2 | D0/C3–C6 joint-invalid→Abstain，C2→`g_B`；Bypass/单方可靠不受影响；新增 EMA transition、feature-off 和 coverage 诊断测试 |
| 2026-09-02 | 用户批准 lifecycle 方案 A：`s_i` 固定为稳定五状态仲裁，`ell_i` 固定为生命周期；映射/`d_i` 只在 evidence refresh，所有 topology gate 用 `ell_i`；新点 Probation 至少 500 iteration 后等下一刷新 | 用户 architectural specification clarification；design §6/§8/§8.1/§8.2 | 修正枚举域冲突；C6 增加 mapping、499/500/refresh、gate-domain、duration 与 resume 测试；不授权 truncation 改动 |
| 2026-09-02 | 用户批准 Probation truncation 方案 A：Core 仅沿用 baseline 全局 `trunc_sigma/disable_trunc`，默认 `2.0` 不是生命周期动作 | 用户 architectural specification clarification；`arguments/__init__.py:64-67`; design §8.2/§9.4 | C1–C6 禁止 per-Gaussian truncation Tensor、状态迁移或 renderer/CUDA 接口改动；§9.4 保留为 G0–G2 后独立 Supporting 候选 |
| 2026-09-03 | 用户批准 Git 基线锁定：仅创建 `c0-baseline`、创建并切换 `research/core-routing`、提交和推送四份规划/设计文档 | C0 training+mesh completion audit；用户明确授权边界 | `main`/`c0-baseline` 保持 baseline SHA `d6f15c8891a53800d5e3100f95817a7dd7f98e2f`；Core 文档提交进入累计分支；本次不修改方法源码或启动实验 |
| 2026-09-03 | 用户批准完整 Core 实施计划，并授权当前仅实施 E0 | 用户明确回复“好的”；Git 已在 clean `research/core-routing@59ffca971782b44439c19f7bc18ea3490b1d452c` | 可以用 TDD 修改 E0 配置/入口/元数据/测试；仍禁止 D0/C1、renderer/CUDA、tag、依赖安装或服务器实验，E0 smoke 需另行批准 |
| 2026-09-03 | 用户批准首次 E0 paired 500-iteration experiment | E0 component gate 23/23 PASS、8/8 CLI flags present、AutoDL post-test clean；用户明确回复“批准” | 仅 Tool Room `-r 2`/seed0/500，baseline→E0 all-off 串行、独立 private views/输出、只读 comparator；不授权 GT/mesh/tag/D0/C1/8k |
