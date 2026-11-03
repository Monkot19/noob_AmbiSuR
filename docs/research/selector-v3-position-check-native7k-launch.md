# V3有效域抽样＋固定当前法线位置检查：原生7K直接至30K

2026-11-03用户批准，D分支`codex/selector-v3-position-check`。固定生产提交`6172bc455184c2b77b7b199c7b66194319bd55fd`；旧V3/early/baseline分支及output保留。正式方法目录名`selector_v3_position_check`。

仅开启`candidate_selector_valid_pool`和`candidate_relocation_position_check`：传播有效域内最多抽16384；联合E继续扩展ALR，硬位置动作仅执行当前法线位置检查通过的子集；支持取G与move各自有效域交集，无新增位置NCC绝对门槛。其余source=spread/interval100/until15000、原LR/DA3 loss/ALR/传播/renderer不变。原生7K恢复不含完整RNG/相机队列，仍是既定共同恢复协议。

本地完整65项57通过、8原生依赖缺失跳过；独立终审发现并修复一个支持mask问题，9新增CPU测试独立通过。未运行本版GPU训练/显存/几何评价，不能声称已有收益。有效域抽样会提高实际NCC工作量，GPU耗时未知。

## 直接启动

确保没有其他训练正在使用该checkout。以下只创建新view/output；图像、原位姿、DA3深度/置信只读链接，可写aligned先验复制到新view。独立bash避免set-e关闭交互终端。无需新baseline、静态采集或GPU smoke作前置。

```bash
bash <<'SH' || printf '\n位置检查版启动未成功，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "位置检查版启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
position_commit=6172bc455184c2b77b7b199c7b66194319bd55fd
if ! git cat-file -e "$position_commit^{commit}" 2>/dev/null; then
  git fetch origin codex/selector-v3-position-check
fi
git switch --detach "$position_commit"
git rev-parse HEAD
git diff --exit-code

position_python=/root/miniconda3/envs/ambisur/bin/python
position_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
position_prefix=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
position_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
position_name="${position_commit:0:8}_seed0_b16384_until15000_30k_v1"
position_view="/root/autodl-tmp/ambisur_work/data_views/selector_v3_position_check/$position_name/colmap_undistorted"
position_out="/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_position_check/$position_name"

test "$(cat "$position_prefix/train.exitcode")" = 0
test -s "$position_prefix/chkpnt7000.pth"
test -s "$position_prefix/app_model/iteration_7000/app.pth"
test -d "$position_ext/diff_plane_rasterization_ambisur"
test -f "$position_source/sparse_da3_aligned/0/points3D.bin"
test -f "$position_source/sparse_da3_aligned/0/trans.json"
test ! -e "$position_view"
test ! -e "$position_out"
mkdir -p "$position_view" "$position_out"
for name in images sparse estimated_depths estimated_confs; do
  test -e "$position_source/$name"
  ln -s "$position_source/$name" "$position_view/$name"
done
cp -aL "$position_source/sparse_da3_aligned" "$position_view/sparse_da3_aligned"
if [ -f "$position_source/split.json" ]; then
  cp -aL "$position_source/split.json" "$position_view/split.json"
fi

export PYTHONPATH="$position_ext${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
printf 'method=selector_v3_position_check\ncommit=%s\nseed=0\niterations=30000\nresume_iteration=7000\nbudget=16384\ncandidate_depth_source=spread\ncandidate_selector_valid_pool=true\ncandidate_relocation_position_check=true\nrelocation_interval=100\nrelocation_until_iter=15000\nOMP_NUM_THREADS=1\ncheckpoint=%s\nrasterizer_target=%s\nsource_view=%s\n' \
  "$(git rev-parse HEAD)" "$position_prefix/chkpnt7000.pth" "$position_ext" "$position_view" > "$position_out/launch.txt"
git status --short > "$position_out/git.status"
git diff --exit-code > "$position_out/git.diff"

nohup bash -c '
  set +e
  "$1" -u -B train.py --source_path "$2" --model_path "$3" \
    -r 2 --iterations 30000 --seed 0 \
    --enable_candidate_selector --candidate_selector_budget 16384 \
    --candidate_depth_source spread --candidate_selector_valid_pool \
    --enable_candidate_relocation --candidate_relocation_position_check \
    --candidate_relocation_interval 100 --candidate_relocation_until_iter 15000 \
    --start_checkpoint "$4"
  position_rc=$?
  printf "%s\n" "$position_rc" > "$3/train.exitcode"
  exit "$position_rc"
' _ "$position_python" "$position_view" "$position_out" "$position_prefix/chkpnt7000.pth" \
  > "$position_out/train.log" 2>&1 < /dev/null &
position_pid=$!
printf '%s\n' "$position_pid" > "$position_out/train.pid"
printf '位置检查版训练PID：%s\n输出：%s\n日志：%s/train.log\n' "$position_pid" "$position_out" "$position_out"
SH
```

若GitHub连接失败，先单独执行`source /etc/network_turbo`，再重试；不覆盖已创建的view/output，不在服务器临时修改源码。

## 查看日志

```bash
position_out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_position_check/6172bc45_seed0_b16384_until15000_30k_v1
tail -n 25 "$position_out/train.log"
if [ -f "$position_out/train.exitcode" ]; then
  printf '\n退出码：'
  cat "$position_out/train.exitcode"
fi
```

事件日志示例`[relocation 7100] moved=... position_checked=... position_passed=...`：checked为原联合E数量，passed应等于moved；0允许、不补移动。selector的added仍是ALR联合E，不是移动数。7001..30000进度条23000步；7100..14900共79事件，15K后继续ALR/有效域抽样但无重定位或位置检查。

完成后沿用原提取5m/.005m/trunc4/cluster2，只提取新模型；正式比较固定baseline、early、原V3和本版的Accuracy/Completeness/CD及1/2/5cmPRF，5cm主列。既有输出只读复用。此指南不自动启动服务器，也不新增评估或可视化框架。

## 2026-11-03训练完成回执与单模型提取

用户返回全局30000/23000续训步完成、75:20、退出0、N1277504、trainL10.017082376033067705、PSNR30.850026321411136及Gaussian/App保存日志。较early -0.000522614dB、旧spread -0.011825180dB、C0 -0.222253799dB，均为原训练视角统计，不是新视角或几何收益。末段22条25800..30000 sampled==proposal_usable，均值sampled12526.77/comparable10735/added4779.32；只说明末段抽样利用表现，不是移动数或全程统计。完整79事件/checked-pass-moved关系、实际HEAD/clean和模型文件大小待下方核对，不重新训练。

```bash
bash <<'SH' || printf '\nmesh提取未启动成功，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "提取准备停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_position_check/6172bc45_seed0_b16384_until15000_30k_v1
test "$(git rev-parse HEAD)" = 6172bc455184c2b77b7b199c7b66194319bd55fd
test -z "$(git status --porcelain)"
git rev-parse HEAD
git diff --exit-code
test "$(cat "$out/train.exitcode")" = 0
test -s "$out/point_cloud/iteration_30000/point_cloud.ply"
test -s "$out/app_model/iteration_30000/app.pth"
/root/miniconda3/envs/ambisur/bin/python - "$out/train.log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding='utf-8').read()
indices = list(map(int, re.findall(r'\[relocation (\d+)\]', text)))
assert indices == list(range(7100, 15000, 100)), indices
events = [tuple(map(int, row)) for row in re.findall(
    r'\[relocation (\d+)\] moved=(\d+) position_checked=(\d+) position_passed=(\d+)', text)]
assert [row[0] for row in events] == indices, 'missing position counters'
assert all(moved == passed <= checked for _, moved, checked, passed in events), events
checked = sum(row[2] for row in events)
moved = sum(row[1] for row in events)
print('79次事件及checked/pass/moved对应通过；累计检查=', checked, '累计移动=', moved)
PY
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
' _ /root/miniconda3/envs/ambisur/bin/python "$out" \
  > "$out/extract_general.log" 2>&1 < /dev/null &
extract_pid=$!
printf '%s\n' "$extract_pid" > "$out/extract.pid"
printf '提取PID：%s\n日志：%s/extract_general.log\n' "$extract_pid" "$out"
SH
```

```bash
out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v3_position_check/6172bc45_seed0_b16384_until15000_30k_v1
tail -n 20 "$out/extract_general.log"
if [ -f "$out/extract.exitcode" ]; then
  printf '\n提取退出码：'
  cat "$out/extract.exitcode"
  ls -lh "$out/mesh/tsdf_fusion.ply" "$out/mesh/tsdf_fusion_post.ply"
fi
```
