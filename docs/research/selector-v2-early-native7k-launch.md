# v2_early：15K后停止重定位，继续至30K

2026-10-28用户已批准。固定代码`a52ab904aa09f67e98bc392c6e0edddb15f5a391`，分支`codex/second-run-selector-v2-early`。只增加可选截止值；本实验显式15000，原V2默认0行为保留。原生7K→30K/r2/seed0/预算16384/间隔100/原LR与loss，重定位7100..14900共79次，15000及以后仍执行V1候选、ALR及原优化。本地及独立终审复跑均31项25CPU通过/6native缺扩展跳过；终审无Critical/Important/Minor。外层/内层Bash语法及实际train.py CLI解析通过，与已完成V2 cfg_opts逐字段相同，仅新增截止值。不能视为GPU或收益验证；无新baseline/smoke/重编译。固定文档提交ad8ecc7357db21ea595a43971df7e471bd5a886a中的启动块相同，可用于短命令提取。

## 启动

在AutoDL的ambisur环境且无其他训练运行时执行。使用独立bash，不向交互shell设置set-e；既有输出不重用或覆盖。联网只用于取得固定提交。新data view复制先验可写目录并链接只读输入，沿用此前做法。

```bash
bash <<'SH' || printf '\nv2_early启动未成功，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "v2_early启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
early_commit=a52ab904aa09f67e98bc392c6e0edddb15f5a391
if ! git cat-file -e "$early_commit^{commit}" 2>/dev/null; then
  git fetch origin codex/second-run-selector-v2-early
fi
git switch --detach "$early_commit"
git rev-parse HEAD
git diff --exit-code

early_python=/root/miniconda3/envs/ambisur/bin/python
early_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
early_prefix=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
early_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
early_name="${early_commit:0:8}_seed0_b16384_until15000_30k_v1"
early_view="/root/autodl-tmp/ambisur_work/data_views/selector_v2_early/$early_name/colmap_undistorted"
early_out="/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v2_early/$early_name"

test "$(cat "$early_prefix/train.exitcode")" = 0
test -s "$early_prefix/chkpnt7000.pth"
test -s "$early_prefix/app_model/iteration_7000/app.pth"
test -d "$early_ext/diff_plane_rasterization_ambisur"
test -f "$early_source/sparse_da3_aligned/0/points3D.bin"
test -f "$early_source/sparse_da3_aligned/0/trans.json"
test ! -e "$early_view"
test ! -e "$early_out"
mkdir -p "$early_view" "$early_out"
for name in images sparse estimated_depths estimated_confs; do
  test -e "$early_source/$name"
  ln -s "$early_source/$name" "$early_view/$name"
done
cp -aL "$early_source/sparse_da3_aligned" "$early_view/sparse_da3_aligned"
if [ -f "$early_source/split.json" ]; then
  cp -aL "$early_source/split.json" "$early_view/split.json"
fi

export PYTHONPATH="$early_ext${PYTHONPATH:+:$PYTHONPATH}"
printf 'method=v2_early\ncommit=%s\nseed=0\niterations=30000\nresume_iteration=7000\nbudget=16384\nrelocation_interval=100\nrelocation_until_iter=15000\ncheckpoint=%s\nrasterizer_target=%s\nsource_view=%s\n' \
  "$(git rev-parse HEAD)" "$early_prefix/chkpnt7000.pth" "$early_ext" "$early_view" > "$early_out/launch.txt"
git status --short > "$early_out/git.status"
git diff --exit-code > "$early_out/git.diff"

nohup bash -c '
  set +e
  "$1" -u -B train.py --source_path "$2" --model_path "$3" \
    -r 2 --iterations 30000 --seed 0 \
    --enable_candidate_selector --candidate_selector_budget 16384 \
    --enable_candidate_relocation --candidate_relocation_interval 100 \
    --candidate_relocation_until_iter 15000 --start_checkpoint "$4"
  early_rc=$?
  printf "%s\n" "$early_rc" > "$3/train.exitcode"
  exit "$early_rc"
' _ "$early_python" "$early_view" "$early_out" "$early_prefix/chkpnt7000.pth" \
  > "$early_out/train.log" 2>&1 < /dev/null &
early_pid=$!
printf '%s\n' "$early_pid" > "$early_out/train.pid"
printf 'v2_early训练PID：%s\n输出：%s\n日志：%s/train.log\n' "$early_pid" "$early_out" "$early_out"
SH
```

## 进度与后续

```bash
early_out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v2_early/a52ab904_seed0_b16384_until15000_30k_v1
tail -n 25 "$early_out/train.log"
if [ -f "$early_out/train.exitcode" ]; then
  cat "$early_out/train.exitcode"
  ls -lh "$early_out/point_cloud/iteration_30000/point_cloud.ply" "$early_out/app_model/iteration_30000/app.pth"
fi
```

进度23000/23000对应全局7001..30000。首个重定位7100、最后14900；15000后仍会有selector日志，但没有relocation日志。moved可以为0，不额外补更新。预计成本参考V2续训53分钟，实际耗时/收益待回执。

完成后沿用5m/0.005m/trunc4/cluster2提取、同一scannetpp_surface_v1评价，对照原生C0/V1-16384/完整V2；不改评价域、筛掉区域或把GT区域用于训练。本实验检验调度适配，不证明15000是最优值。不要自动清理旧实验、补baseline或改变其他变量。

## early训练完成后的单模型提取（2026-10-28）

用户已返回30K完成日志、trainPSNR30.850548934936526。以下先核对退出码、模型非空及精确79次重定位序列，再只提取early模型；原提取配置不变，Bash语法核对通过。尚无实际提取回执。无需fetch或切换版本。

```bash
bash <<'SH'
set -euo pipefail
trap 'printf "提取准备停止：第%s行\n" "$LINENO" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v2_early/a52ab904_seed0_b16384_until15000_30k_v1
test "$(git rev-parse HEAD)" = a52ab904aa09f67e98bc392c6e0edddb15f5a391
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
printf 'commit=%s\niteration=30000\nresolution=2\nmax_depth=5.0\nvoxel_size=0.005\nsdf_trunc_scale=4.0\nnum_cluster=2\nuse_depth_filter=false\n' "$(git rev-parse HEAD)" > "$out/extract.launch.txt"
nohup bash -c '
  "$1" -u -B mesh_extract/extract_general.py -m "$2" -r 2 --iteration 30000 --skip_test --max_depth 5.0 --voxel_size 0.005 --sdf_trunc_scale 4.0 --num_cluster 2
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
