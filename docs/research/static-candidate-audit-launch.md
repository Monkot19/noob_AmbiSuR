# 静态候选检查：统一执行入口

用户已批准；服务器当前关闭，本次没有开机、租卡或真实场景capture。脚本只调用真实selector前向并观察返回局部变量，不执行重定位或optimizer.step。原训练/renderer/selector代码未改。

## 检查范围与为何扩展

- 7K原生模型：检查额外动作开始前的候选质量。
- C0 30K：检查成熟基线是否已经优于外部候选。
- 完整V2与early的30K：检查两种终态仍会接受什么候选；不是重建原训练历史，也不能据此确定15K阶段因果。
- 没有15K快照，不为补快照重训。
- 原五个问题视角加12个按图像名称等间隔选择的视角，共17个；两组分开汇总。扩展视角不依赖GT误差，但不声称空间均匀、随机样本或独立验证集。
- 所有快照seed0、预算16384、保留原SH/NCC/共同支持/两支持视角规则；7K probe_iteration7100、30K probe_iteration30000均为静态反事实调用，不真正移动。
- 相机UID来自真实COLMAP reader global_id，不使用cameras.json枚举ID；支持顺序复用已归档multi_view.json。

扩展12视角：DSC06274、DSC06311、DSC06348、DSC06384、DSC06421、DSC06458、DSC06495、DSC06532、DSC06569、DSC06605、DSC06642、DSC06681。

## 下次服务器已开启后：capture

固定生产代码`c5e624307de97f91898e461e5e789a4d8ee0eae3`，不用分支浮动版本替代。只启动一次前向检查，无训练。原生扩展继续使用用户已验证的独立C0构建，不覆盖conda包。

```bash
bash <<'SH'
set -euo pipefail
trap 'rc=$?; printf "静态检查停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
git fetch origin codex/selector-static-audit
git switch --detach c5e624307de97f91898e461e5e789a4d8ee0eae3
git rev-parse HEAD
git diff --exit-code

audit_python=/root/miniconda3/envs/ambisur/bin/python
audit_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
audit_runs=/root/autodl-tmp/ambisur_runs/Tool_Room
audit_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
audit_output="$audit_runs/static_candidate_audit/static_v1"
test ! -e "$audit_output"
test ! -e "$audit_output.log"
test -d "$audit_ext/diff_plane_rasterization_ambisur"
mkdir -p "$(dirname "$audit_output")"
export PYTHONPATH="$audit_ext${PYTHONPATH:+:$PYTHONPATH}"
nohup bash -c '
  "$1" -u -B scripts/evaluate_candidate_actions.py capture \
    --config experiments/candidate_actions/toolroom_static_v1.json \
    --source "$2" \
    --snapshot "native7k=$3/native_c0_ckpt7k/d6f15c88_seed0_30k_v1/chkpnt7000.pth" \
    --snapshot "c0_30k=$3/native_c0_ckpt7k/d6f15c88_seed0_30k_v1" \
    --snapshot "v2_30k=$3/selector_v2/a114ed5b_seed0_b16384_30k_v1" \
    --snapshot "early_30k=$3/selector_v2_early/a52ab904_seed0_b16384_until15000_30k_v1" \
    --output "$4"
  rc=$?
  printf "%s\n" "$rc" > "$4.exitcode"
  exit "$rc"
' _ "$audit_python" "$audit_source" "$audit_runs" "$audit_output" \
  > "$audit_output.log" 2>&1 < /dev/null &
printf '静态检查PID：%s\n日志：%s.log\n' "$!" "$audit_output"
SH
```

完成标记为 `capture_manifest.json`；退出码在同级 `static_v1.exitcode`，预期0。目录只含候选记录/元数据，不含GT。不依赖中期checkpoint，也不修改已有训练输出。按4×17共68次选择前向执行，加载快照和原生渲染实际耗时待测；不按30K训练成本安排。

## 下载capture目录后：本地score

将完整capture目录下载到以下约定位置；命令路径可显式修改，不写入代码。

```powershell
& 'D:/research_Space/ambisur-eval-runtime/Scripts/python.exe' -B `
  'D:/research_Space/noob_AmbiSuR/scripts/evaluate_candidate_actions.py' score `
  --capture 'D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/analysis/static_candidate_capture_v1' `
  --gt 'D:/dataset/ScanNet++/data/data/d415cc449b_Tool_Room/scans/mesh_aligned_0.05.ply' `
  --segments 'D:/dataset/ScanNet++/data/data/d415cc449b_Tool_Room/scans/segments.json' `
  --annotations 'D:/dataset/ScanNet++/data/data/d415cc449b_Tool_Room/scans/segments_anno.json' `
  --output 'D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/analysis/static_candidate_score_v1'
```

输出results.json/metrics.csv；正gain表示候选更接近GT。分别检查接受/拒绝可比较候选、问题/覆盖视角、快照及对象。nearest GT表面距离与同参考第一命中z误差分别报告；法线用生产GT-depth normal计算，有效邻域另计数。不能把最近表面更近等同正确可见归属。多视角可能重复同一Gaussian，计数是候选—视角观测，不是独立Gaussian数量；不据此计算训练显著性。

目前本地证据：统一selector测试37项，30CPU通过、7native因本地无扩展跳过；真实CPU selector→NPZ→真实GT/BVH/normal→CSV/JSON链路通过。独立终审发现全批/子批FP32矩阵乘法舍入差异导致逐位核对误中止，新增16行混合接受回归已RED→GREEN；接受行直接保存实际生产返回目标，未改候选规则。复审无进一步阻断问题，独立复跑37项结果一致。四份本地模型、配置和17视角双支持已核对；尚未跑真实capture，不能宣布已得到接受候选的几何收益比例。
