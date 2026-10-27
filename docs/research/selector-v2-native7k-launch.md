# selector_V2：复用原生7K，直接续训至30K

2026-10-27。用户已批准V2=V1+重定位、预算16384、原LR/loss、间隔100，并授权实施及常规修复。**固定代码`a114ed5b5ee1c265eea5ececc98268934eca6b2e`**；独立整体审查无Critical/Important/Minor发现。本地统一30项中24CPU通过、6原生集成因缺扩展跳过；启动外层/内层及进度块bash语法检查、真实CLI参数组解析通过。**不代表服务器已执行、GPU已验证或已有重建收益**。

原生baseline仍d6f15c8，复用已有chkpnt7000.pth与其7K App，不重训/覆盖baseline。参考V1继续使用当前两支持源、NCC/回退和ALR；V2仅在7100、7200、…、29900轮对E行重定位。新输出在selector_v2下，不复用V1输出。数据盘上次用户回执余14G，本文不执行清理。

## 启动（AutoDL ambisur环境）

先确保服务器上一训练队列已结束。整段包含独立bash，失败不会使交互终端因set-e退出；GitHub需要联网仅发生在固定commit不在本地时，网络失败修复后可重试，不重复启动已有输出。

```bash
bash <<'SH' || printf '\nV2启动未成功，请保留上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "V2启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR

cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
v2_commit=a114ed5b5ee1c265eea5ececc98268934eca6b2e
if ! git cat-file -e "$v2_commit^{commit}" 2>/dev/null; then
  git fetch origin codex/second-run-selector-v2
fi
git cat-file -e "$v2_commit^{commit}"
git switch --detach "$v2_commit"
git rev-parse HEAD
git diff --exit-code

v2_python=/root/miniconda3/envs/ambisur/bin/python
v2_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
v2_prefix=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
v2_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
v2_name="${v2_commit:0:8}_seed0_b16384_30k_v1"
v2_view="/root/autodl-tmp/ambisur_work/data_views/selector_v2/$v2_name/colmap_undistorted"
v2_out="/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v2/$v2_name"

test "$(cat "$v2_prefix/train.exitcode")" = 0
test -s "$v2_prefix/chkpnt7000.pth"
test -s "$v2_prefix/app_model/iteration_7000/app.pth"
test -d "$v2_ext/diff_plane_rasterization_ambisur"
test -f "$v2_source/sparse_da3_aligned/0/points3D.bin"
test -f "$v2_source/sparse_da3_aligned/0/trans.json"
test ! -e "$v2_view"
test ! -e "$v2_out"
mkdir -p "$v2_view" "$v2_out"
for name in images sparse estimated_depths estimated_confs; do
  test -e "$v2_source/$name"
  ln -s "$v2_source/$name" "$v2_view/$name"
done
cp -aL "$v2_source/sparse_da3_aligned" "$v2_view/sparse_da3_aligned"
if [ -f "$v2_source/split.json" ]; then
  cp -aL "$v2_source/split.json" "$v2_view/split.json"
fi

export PYTHONPATH="$v2_ext${PYTHONPATH:+:$PYTHONPATH}"
printf 'method=selector_V2\ncommit=%s\nseed=0\niterations=30000\nresume_iteration=7000\nbudget=16384\nrelocation_interval=100\ncheckpoint=%s\nrasterizer_target=%s\nsource_view=%s\n' \
  "$(git rev-parse HEAD)" "$v2_prefix/chkpnt7000.pth" "$v2_ext" "$v2_view" > "$v2_out/launch.txt"
git status --short > "$v2_out/git.status"
git diff --exit-code > "$v2_out/git.diff"

nohup bash -c '
  set +e
  "$1" -u -B train.py --source_path "$2" --model_path "$3" \
    -r 2 --iterations 30000 --seed 0 \
    --enable_candidate_selector --candidate_selector_budget 16384 \
    --enable_candidate_relocation --candidate_relocation_interval 100 \
    --start_checkpoint "$4"
  v2_rc=$?
  printf "%s\n" "$v2_rc" > "$3/train.exitcode"
  exit "$v2_rc"
' _ "$v2_python" "$v2_view" "$v2_out" "$v2_prefix/chkpnt7000.pth" \
  > "$v2_out/train.log" 2>&1 < /dev/null &

v2_pid=$!
printf '%s\n' "$v2_pid" > "$v2_out/train.pid"
printf 'V2训练PID：%s\n输出：%s\n日志：%s/train.log\n' "$v2_pid" "$v2_out" "$v2_out"
SH
```

不重复全量数据哈希、不新装环境、不重编译renderer、不先开独立GPU smoke或selector-off对照。启动的真实GPU/重定位是否成功，要看此任务日志；出现异常保留输出与退出码，修复先本地提交再换固定版本，不在服务器临时改训练代码。

## 看进度和完成状态

把下方commit前八位与启动块保持一致；首个重定位日志应在7100轮，moved计数可为0（正常无新增候选），非零只说明动作发生，不代表收益。

```bash
v2_out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v2/a114ed5b_seed0_b16384_30k_v1
tail -n 25 "$v2_out/train.log"
if [ -f "$v2_out/train.exitcode" ]; then
  cat "$v2_out/train.exitcode"
  ls -lh "$v2_out/point_cloud/iteration_30000/point_cloud.ply" \
         "$v2_out/app_model/iteration_30000/app.pth"
fi
```

预计成本参考现有16K续训约1小时，重定位实际耗时待日志验证；进度23000/23000对应全局7001—30000，不是仅训练到23000。结束后沿用已有5m/0.005m/trunc4/cluster2提取配置与正式mesh evaluator，对照新native C0及V1-16384；不更换评价域、阈值或后处理规则。下载完整新V2 output后在D盘归档，再讨论任何服务器清理。
