# selector_v3_spread：已批准原生7K续训至30K

2026-11-02用户批准：静态候选结果正向后，原样运行一轮重建实验。固定生产8b1def920b2d4722e9ff238ed2116a477cd4846c，复用已经发布的codex/selector-v3-spread，不需要新增代码或推送文档才能启动。当前本地Git元数据权限阻止文档提交，此指南是待提交文件，生产提交已在服务器完成17视角capture。

相对原v2_early启动唯一算法选项变化为 `--candidate_depth_source spread`。同一原生7K/App、seed0、r2、预算16384、总30000、重定位interval100/until15000、原DA3深度/法线损失及LR/TSDF/评价不变。传播仍用于15000以后的候选补选ALR；停止的是重定位。原生checkpoint未包含RNG/相机队列，保持此前共同恢复协议，不声称等同连续baseline轨迹。没有新baseline、额外smoke、GT训练输入或诊断框架。

## AutoDL启动

在没有其他训练使用该checkout时完整执行；使用独立bash，不会给交互shell设置set-e。新view只链接只读图像/原COLMAP/DA3深度与置信，复制可写aligned先验；所有旧输出和数据保留。显式OMP_NUM_THREADS=1与此前运行库对无效0的回退值一致，仅消除该环境警告，写入launch记录。

```bash
bash <<'SH' || printf '\nselector_v3_spread启动未成功，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "selector_v3_spread启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
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
spread_name="${spread_commit:0:8}_seed0_b16384_until15000_30k_v1"
spread_view="/root/autodl-tmp/ambisur_work/data_views/selector_v3_spread/$spread_name/colmap_undistorted"
spread_out="/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_spread/$spread_name"

test "$(cat "$spread_prefix/train.exitcode")" = 0
test -s "$spread_prefix/chkpnt7000.pth"
test -s "$spread_prefix/app_model/iteration_7000/app.pth"
test -d "$spread_ext/diff_plane_rasterization_ambisur"
test -f "$spread_source/sparse_da3_aligned/0/points3D.bin"
test -f "$spread_source/sparse_da3_aligned/0/trans.json"
test ! -e "$spread_view"
test ! -e "$spread_out"
mkdir -p "$spread_view" "$spread_out"
for name in images sparse estimated_depths estimated_confs; do
  test -e "$spread_source/$name"
  ln -s "$spread_source/$name" "$spread_view/$name"
done
cp -aL "$spread_source/sparse_da3_aligned" "$spread_view/sparse_da3_aligned"
if [ -f "$spread_source/split.json" ]; then
  cp -aL "$spread_source/split.json" "$spread_view/split.json"
fi

export PYTHONPATH="$spread_ext${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
printf 'method=selector_v3_spread\ncommit=%s\nseed=0\niterations=30000\nresume_iteration=7000\nbudget=16384\ncandidate_depth_source=spread\nrelocation_interval=100\nrelocation_until_iter=15000\nOMP_NUM_THREADS=1\ncheckpoint=%s\nrasterizer_target=%s\nsource_view=%s\n' \
  "$(git rev-parse HEAD)" "$spread_prefix/chkpnt7000.pth" "$spread_ext" "$spread_view" > "$spread_out/launch.txt"
git status --short > "$spread_out/git.status"
git diff --exit-code > "$spread_out/git.diff"

nohup bash -c '
  set +e
  "$1" -u -B train.py --source_path "$2" --model_path "$3" \
    -r 2 --iterations 30000 --seed 0 \
    --enable_candidate_selector --candidate_selector_budget 16384 \
    --candidate_depth_source spread \
    --enable_candidate_relocation --candidate_relocation_interval 100 \
    --candidate_relocation_until_iter 15000 --start_checkpoint "$4"
  spread_rc=$?
  printf "%s\n" "$spread_rc" > "$3/train.exitcode"
  exit "$spread_rc"
' _ "$spread_python" "$spread_view" "$spread_out" "$spread_prefix/chkpnt7000.pth" \
  > "$spread_out/train.log" 2>&1 < /dev/null &
spread_pid=$!
printf '%s\n' "$spread_pid" > "$spread_out/train.pid"
printf 'selector_v3_spread训练PID：%s\n输出：%s\n日志：%s/train.log\n' "$spread_pid" "$spread_out" "$spread_out"
SH
```

## 看进展

```bash
spread_out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_spread/8b1def92_seed0_b16384_until15000_30k_v1
tail -n 25 "$spread_out/train.log"
if [ -f "$spread_out/train.exitcode" ]; then
  printf '\n退出码：'
  cat "$spread_out/train.exitcode"
fi
```

23000/23000对应全局7001—30000。首个重定位7100、最后14900；15000后仍有带anchors/spread_pixels/proposal_usable的selector日志，不应有relocation日志。moved可为0，不补更新。真实GPU训练耗时未知，不沿用旧early约1小时作保证。完成后只提取此模型，仍用5m/.005m/trunc4/cluster2与原正式评价；不覆盖或重提取baseline，不自动追加新训练。

## 训练完成回执与单模型提取（2026-11-02）

用户回传23000/23000、全局ITER30000、训练81:17、退出0、Gaussian/App保存及Trainingcomplete15:34:11。N1279368，trainL10.017056512646377087、PSNR30.861851501464844；相对原early+0.01130257dB，相对原生C0-0.21042862dB。仅训练视角统计，不能代替新视角或mesh几何结论。末段只有selector符合截止后行为，但完整79次序列/文件非空尚待下方检查。

以下沿用原提取流程，不fetch/改源码/重训，仅验证本模型并提取。原有mesh存在时退出，不覆盖。

```bash
bash <<'SH' || printf '\nmesh提取未启动成功，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "mesh提取准备停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_spread/8b1def92_seed0_b16384_until15000_30k_v1
test "$(git rev-parse HEAD)" = 8b1def920b2d4722e9ff238ed2116a477cd4846c
test -z "$(git status --porcelain)"
git rev-parse HEAD
git diff --exit-code
test "$(cat "$out/train.exitcode")" = 0
test -s "$out/point_cloud/iteration_30000/point_cloud.ply"
test -s "$out/app_model/iteration_30000/app.pth"
/root/miniconda3/envs/ambisur/bin/python -c 'import re,sys; s=open(sys.argv[1]).read(); a=list(map(int,re.findall(r"\[relocation (\d+)\]",s))); assert a==list(range(7100,15000,100)),a; print("重定位序列核对通过：79次，7100至14900")' "$out/train.log"
test ! -e "$out/mesh"
test ! -e "$out/extract_general.log"
test ! -e "$out/extract.exitcode"
export PYTHONPATH="/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
printf 'commit=%s\niteration=30000\nresolution=2\nmax_depth=5.0\nvoxel_size=0.005\nsdf_trunc_scale=4.0\nnum_cluster=2\nuse_depth_filter=false\n' "$(git rev-parse HEAD)" > "$out/extract.launch.txt"
nohup bash -c '
  "$1" -u -B mesh_extract/extract_general.py -m "$2" -r 2 \
    --iteration 30000 --skip_test --max_depth 5.0 \
    --voxel_size 0.005 --sdf_trunc_scale 4.0 --num_cluster 2
  rc=$?
  if [ "$rc" -eq 0 ]; then
    test -s "$2/mesh/tsdf_fusion.ply" && test -s "$2/mesh/tsdf_fusion_post.ply"
    rc=$?
  fi
  printf "%s\n" "$rc" > "$2/extract.exitcode"
  exit "$rc"
' _ /root/miniconda3/envs/ambisur/bin/python "$out" > "$out/extract_general.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$out/extract.pid"
printf '提取PID：%s\n日志：%s/extract_general.log\n' "$(cat "$out/extract.pid")" "$out"
SH
```

```bash
out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_spread/8b1def92_seed0_b16384_until15000_30k_v1
tail -n 20 "$out/extract_general.log"
if [ -f "$out/extract.exitcode" ]; then
  printf '\n提取退出码：'
  cat "$out/extract.exitcode"
  ls -lh "$out/mesh/tsdf_fusion.ply" "$out/mesh/tsdf_fusion_post.ply"
fi
```

提取成功后用户下载此训练目录到独立本地selector_v3_spread目录；既有baseline/early等本地归档直接复用。再运行原正式evaluator评价Accuracy/Completeness/SurfaceChamfer与1/2/5cm PRF，不能预先把静态gain或PSNR当最终几何收益。尚未收到实际提取回执。
