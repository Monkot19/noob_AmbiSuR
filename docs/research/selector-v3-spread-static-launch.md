# selector_v3_spread：固定版本与静态检查交接

2026-10-30：本地实现及独立审查完成，实际 GPU 运行尚未执行。D 盘主目录使用 `codex/selector-v3-spread`，固定生产提交为 `8b1def920b2d4722e9ff238ed2116a477cd4846c`，起点为 `43e2ea0`。旧 baseline、V1、V2、early、静态检查分支和输出均保留。

## 实际接线与边界

- 新公共模块 `utils/depth_spread.py`：移动前模型渲染深度作锚点，第一支持视角的 RGB NCC 与前后投影误差生成置信度，DA3 法线引导传播；5×5 邻域、10 步、置信阈值 .6、法线余弦 .99、传播掩码阈值 .01。沿用宿主整数像素 K⁻¹ 射线，排除无效传递。
- `candidate_depth_source` 默认 `da3`，显式 `spread` 时才启用。`proposal_depth/proposal_valid` 仅进入原候选位置接口；原始 DA3 depth/normal 继续进入原深度损失和 ALR。现有元数据自动保存该选项。
- 当前高斯与候选的共同支持、两支持视角联合位置/法线 NCC、标签 A∪E、原位置执行器均复用。此轮没有固定法线 NCC、传播深度损失、梯度融合或学习率调整。
- 无有效传播位置的候选在新增几何/NCC 比较前退出，不产生 E 或新增重定位；原 SH 标签 A 和基线监督照常。没有 raw DA3 位置回退，也不存在另一套“SH 匹配”。
- 训练采用调用内源视角深度缓存：移动前无梯度参考渲染 → 传播与选择 → 已有重定位 → 原主渲染。早期调度不变：未来显式 cutoff=15000/interval=100 对应 7100…14900 共79次移动；15K 后仍用传播候选补选 ALR，停止的是重定位。
- DP-GS 原动作是传播深度损失，本实现借其候选生成机制接入现有比较与重定位，并非完整原样移植。未来替换原深度 loss 需单独实验。

## 本地验证与结论范围

既有两个测试入口共56项：48项通过，8项原生测试因本地缺渲染扩展跳过，退出码0。包括实际传播 producer → selector → literal world target → observer、真实 RGB/灰度 NCC、不同 NCC 尺寸、稀疏锚点解析平面、无锚点/法线边界/无效射线、训练 helper 的缓存/渲染顺序/原 DA3 消费接口。原灰度匹配与冻结实现逐元素一致；原 ALR、离线评分核心 AST，以及 renderer/Gaussian 执行器/CUDA 均未改变。Python 3.10 语法检查通过。

独立审查固定提交相对43e2ea0，重读供体实际源码并复跑新增9项CPU测试，全通过，未发现 Critical/Important 问题。此证据仅支持接口与局部算法实现；真实 GPU 显存、耗时和候选/重建收益未知。

## 下一批：复用原生7K模型，仅采集17视角

复用已保存的原五问题视角＋12覆盖视角、seed0、预算16384、probe7100；只采集 native7k，不重采三份30K终态、不训练、不补 checkpoint。原 raw DA3 的 native7k 记录已在 `static_candidate_audit/static_v2`，之后在同快照、视角和已有分组协议上比较；各方法可比较/接受域不同，分别报告，不把接受行数当独立高斯数。前向 capture 不读取 GT。

以下仅为交接命令，尚未执行。用户已开启服务器、确认没有其他任务使用该 checkout 后执行。保留历史失败目录；若新输出已存在，先检查它，勿直接覆盖或换名重复启动。网络失败发生在拉取阶段，不代表 checkpoint 或候选计算失败。

```bash
bash <<'SH'
set -euo pipefail
trap 'rc=$?; printf "传播候选检查停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
spread_commit=8b1def920b2d4722e9ff238ed2116a477cd4846c
if ! git cat-file -e "$spread_commit^{commit}" 2>/dev/null; then
  git fetch origin codex/selector-v3-spread
fi
git switch --detach "$spread_commit"
git rev-parse HEAD
git diff --exit-code

spread_python=/root/miniconda3/envs/ambisur/bin/python
spread_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
spread_prefix=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
spread_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
spread_output=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_spread/static_native7k_v1
test -s "$spread_prefix/chkpnt7000.pth"
test -d "$spread_ext/diff_plane_rasterization_ambisur"
test ! -e "$spread_output"
test ! -e "$spread_output.log"
test ! -e "$spread_output.exitcode"
mkdir -p "$(dirname "$spread_output")"
export PYTHONPATH="$spread_ext${PYTHONPATH:+:$PYTHONPATH}"

# 复用同一个测试入口中的新原生集成用例；不是另建冒烟框架。
"$spread_python" -B -m unittest \
  tests.test_candidate_selector.CandidateIntegrationTests.test_spread_real_render_to_alr_backward_and_step -v
# 上条应实际运行并 OK；如显示 skipped，不把它记为 GPU 已通过。

nohup bash -c '
  "$1" -u -B scripts/evaluate_candidate_actions.py capture \
    --config experiments/candidate_actions/toolroom_spread_native7k_v1.json \
    --source "$2" \
    --snapshot "native7k=$3/chkpnt7000.pth" \
    --output "$4"
  rc=$?
  printf "%s\n" "$rc" > "$4.exitcode"
  exit "$rc"
' _ "$spread_python" "$spread_source" "$spread_prefix" "$spread_output" \
  > "$spread_output.log" 2>&1 < /dev/null &
printf '静态检查PID：%s\n日志：%s.log\n' "$!" "$spread_output"
SH
```

只有17次选择前向和一次模型加载；不会执行重定位或优化器更新。传播多一次移动前参考渲染和稠密邻域运算，耗时/显存须以本次服务器执行为准，不能由CPU测试估计。进展沿用原日志，不新增诊断程序：

```bash
spread_output=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_spread/static_native7k_v1
tail -n 20 "$spread_output.log"
if [ -f "$spread_output.exitcode" ]; then cat "$spread_output.exitcode"; fi
```

完成时应有退出码0、`capture_manifest.json`及17份NPZ。请下载完整目录、同级 `.log` 和 `.exitcode` 到本地独立目录，例如 `D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/analysis/static_candidate_capture_spread_native7k_v1`。不删除原始 capture。

下载后继续复用 `scripts/evaluate_candidate_actions.py score` 和原GT/分割/单位；沿用已有评分合同，在新输出目录执行一次。比较传播候选与旧 native7k 的位置改善率/均值、法线和位置联合分组、可比较数量与接受数量，再讨论是否值得30K；本轮不预先宣布有效或启动训练。
