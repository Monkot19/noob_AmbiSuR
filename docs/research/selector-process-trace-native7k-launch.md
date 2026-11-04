# 既有7K→30K过程采集：定位首次前鼓及后期纠正失败

用户已确认直接到30K。方法仍为6172bc45位置检查版，预算16384、spread、valid-pool、固定当前法线位置检查、原LR/loss、重定位7100..14900不变。只新增独立采集启动器，普通train.py不加载采集代码。生产提交：`cd6475526b8cccc50e3ce797bcea4a2c66bea664`，分支`codex/selector-process-trace`。

本地统一入口63项：55通过、8原生渲染依赖缺失跳过；6项新增过程采集测试通过。CPU测试贯通真实传播/选择/训练helper与共享record，renderer以CPU边界替代，不能宣称GPU通过。原train/arguments/selector/spread/Gaussian/renderer/submodules相对6172bc45无差异。独立审查状态见交接文档。

额外记录包含96个实际参考事件（7001、7100..14900每100、15000..30000每1000）及24个固定DSC06642探针。前者保存真实锚点/传播/联合E/实际move及移动后深度、法线、绝对未归一化rendered_unc；后者为A-only只读渲染，不代表固定视角的联合ALR。15K后实际move为空，继续观察后期。无GT、无额外反传，不增加7K保存或新baseline。记录与只读前向会增加耗时和空间，不保证GPU轨迹逐位重现；原7K也没有保存完整RNG/相机队列。

## 直接复制启动

确保当前checkout没有其他训练作业。独立bash中的set-e不会关闭外层交互终端。只写新view/output，旧输入及输出不改动；aligned先验复制供原训练使用。本场景876×584，96/24事件未压缩数组约5GB；为trace额外预留约6GB，另需正常训练output空间，压缩量不保证；不以额外smoke为前置。

```bash
bash <<'SH' || printf '\n过程采集启动未成功，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
df -h /root/autodl-tmp
test -z "$(git status --porcelain)"
trace_commit=cd6475526b8cccc50e3ce797bcea4a2c66bea664
if ! git cat-file -e "$trace_commit^{commit}" 2>/dev/null; then
  git fetch origin codex/selector-process-trace
fi
git switch --detach "$trace_commit"
git rev-parse HEAD
git diff --exit-code

trace_python=/root/miniconda3/envs/ambisur/bin/python
trace_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
trace_prefix=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
trace_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
trace_name="${trace_commit:0:8}_seed0_b16384_until15000_30k_v1"
trace_view="/root/autodl-tmp/ambisur_work/data_views/selector_process_trace/$trace_name/colmap_undistorted"
trace_out="/root/autodl-tmp/ambisur_runs/Tool_Room/selector_process_trace/$trace_name"
test "$(cat "$trace_prefix/train.exitcode")" = 0
test -s "$trace_prefix/chkpnt7000.pth"
test -s "$trace_prefix/app_model/iteration_7000/app.pth"
test -d "$trace_ext/diff_plane_rasterization_ambisur"
test -f "$trace_source/sparse_da3_aligned/0/points3D.bin"
test -f "$trace_source/sparse_da3_aligned/0/trans.json"
test ! -e "$trace_view"
test ! -e "$trace_out"
mkdir -p "$trace_view" "$trace_out"
for name in images sparse estimated_depths estimated_confs; do
  test -e "$trace_source/$name"
  ln -s "$trace_source/$name" "$trace_view/$name"
done
cp -aL "$trace_source/sparse_da3_aligned" "$trace_view/sparse_da3_aligned"
if [ -f "$trace_source/split.json" ]; then
  cp -aL "$trace_source/split.json" "$trace_view/split.json"
fi
export PYTHONPATH="$trace_ext${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
printf 'method=unchanged_selector_v3_position_check\nentry=scripts/trace_candidate_training.py\ncommit=%s\nseed=0\niterations=30000\nresume_iteration=7000\nbudget=16384\nsource=spread\nvalid_pool=true\nposition_check=true\nrelocation_interval=100\nrelocation_until_iter=15000\nprobe_view=DSC06642\ncheckpoint=%s\nrasterizer_target=%s\nsource_view=%s\n' \
  "$(git rev-parse HEAD)" "$trace_prefix/chkpnt7000.pth" "$trace_ext" "$trace_view" > "$trace_out/launch.txt"
git status --short > "$trace_out/git.status"
git diff --exit-code > "$trace_out/git.diff"

nohup bash -c '
  set +e
  "$1" -u -B scripts/trace_candidate_training.py \
    --trace-output "$3/trace" --probe-view DSC06642 -- \
    --source_path "$2" --model_path "$3" -r 2 --iterations 30000 --seed 0 \
    --enable_candidate_selector --candidate_selector_budget 16384 \
    --candidate_depth_source spread --candidate_selector_valid_pool \
    --enable_candidate_relocation --candidate_relocation_position_check \
    --candidate_relocation_interval 100 --candidate_relocation_until_iter 15000 \
    --start_checkpoint "$4"
  trace_rc=$?
  printf "%s\n" "$trace_rc" > "$3/train.exitcode"
  exit "$trace_rc"
' _ "$trace_python" "$trace_view" "$trace_out" "$trace_prefix/chkpnt7000.pth" \
  > "$trace_out/train.log" 2>&1 < /dev/null &
trace_pid=$!
printf '%s\n' "$trace_pid" > "$trace_out/train.pid"
printf '采集训练PID：%s\n输出：%s\n日志：%s/train.log\n' "$trace_pid" "$trace_out" "$trace_out"
SH
```

若fetch出现TLS错误，先单独执行`source /etc/network_turbo`后重试；不在服务器修改源码或复用已创建的输出路径。无需重新编译已隔离C0 renderer。

## 查看与下载

```bash
trace_out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_process_trace/cd647552_seed0_b16384_until15000_30k_v1
tail -n 25 "$trace_out/train.log"
if [ -f "$trace_out/train.exitcode" ]; then
  printf '\n训练退出码：'
  cat "$trace_out/train.exitcode"
  python - "$trace_out/trace/trace_manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); m = json.loads(p.read_text())
print('采集成功标记:', m['successful'])
print('实际参考事件:', len(m['events']), '固定探针:', len(m['probes']))
assert m['successful'] and len(m['events']) == 96 and len(m['probes']) == 24
assert all(e['reference_complete'] for e in m['events'])
for e in m['events']:
    for key in ('spread_file', 'candidate_file', 'selection_file', 'reference_file'):
        assert (p.parent/e[key]).is_file(), (e['iteration'], key)
for e in m['probes']:
    assert (p.parent/e['file']).is_file(), e
print('事件完整性检查通过；尚未进行GT/几何收益评价。')
PY
fi
```

下载完整新output，尤其保留`trace/`、train.log、元数据、最终point_cloud/App和原debug；不只下载mesh。过程NPZ可复用现有schema1 scorer离线分析；auxiliary里的move_accepted/target_ids才是实际动作，不能用candidates.accepted代替。已保存数值深度和未经JET归一化的mask，后续无需从JPEG逆推绝对值。

此次目标是查因，不宣称新增方法收益。训练完成后仍按原5m/.005m/trunc4/cluster2提取新模型；旧baseline和early不重新训练或提取。

## 2026-11-04训练回执与检查后提取

用户日志回执：30000/23000续训步完成，74:32、退出0、N1279429、train L1=0.01710001826286316、PSNR=30.836247634887698，12:20:20打印Training complete及App保存。整千轮4.3–4.8秒的峰值与既定采集时点一致，但仅凭tail不能确认全清单/文件。PSNR为原训练抽样统计，不是几何或完整新视角评价。下方先检查96/24事件、79次位置动作及最终文件，再原参数只提取本次模型；不重训或更改方法。

```bash
bash <<'SH' || printf '\n检查或提取启动中止，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_process_trace/cd647552_seed0_b16384_until15000_30k_v1
test "$(git rev-parse HEAD)" = cd6475526b8cccc50e3ce797bcea4a2c66bea664
test -z "$(git status --porcelain)"
git rev-parse HEAD
git diff --exit-code
test "$(cat "$out/train.exitcode")" = 0
test -s "$out/point_cloud/iteration_30000/point_cloud.ply"
test -s "$out/app_model/iteration_30000/app.pth"
/root/miniconda3/envs/ambisur/bin/python - "$out" <<'PY'
import json, re, sys
from pathlib import Path
p = Path(sys.argv[1]); t = p / 'trace'
m = json.loads((t / 'trace_manifest.json').read_text())
expected = [7001] + list(range(7100, 15000, 100)) + list(range(15000, 30001, 1000))
assert m['successful'], m.get('error')
assert [e['iteration'] for e in m['events']] == expected, 'reference schedule mismatch'
assert [e['iteration'] for e in m['probes']] == [7001] + list(range(8000, 30001, 1000)), 'probe schedule mismatch'
for e in m['events']:
    assert e['reference_complete'], e['iteration']
    for key in ('spread_file', 'candidate_file', 'selection_file', 'reference_file'):
        assert (t / e[key]).stat().st_size > 0, (e['iteration'], key)
for e in m['probes']:
    assert e['camera'] == 'DSC06642', e
    assert (t / e['file']).stat().st_size > 0, e
text = (p / 'train.log').read_text()
events = [tuple(map(int, row)) for row in re.findall(
    r'\[relocation (\d+)\] moved=(\d+) position_checked=(\d+) position_passed=(\d+)', text)]
assert [e[0] for e in events] == list(range(7100, 15000, 100)), 'relocation schedule mismatch'
assert all(moved == passed <= checked for _, moved, checked, passed in events)
print('96个参考事件、24个固定探针、79次动作及非空文件检查通过。')
print('累计检查:', sum(e[2] for e in events), '累计移动:', sum(e[1] for e in events))
print('尚未验证NPZ内容或分析几何收益。')
PY
test ! -e "$out/mesh"
test ! -e "$out/extract_general.log"
test ! -e "$out/extract.exitcode"
export PYTHONPATH="/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
printf 'commit=%s\niteration=30000\nresolution=2\nmax_depth=5.0\nvoxel_size=0.005\nsdf_trunc_scale=4.0\nnum_cluster=2\nuse_depth_filter=false\n' "$(git rev-parse HEAD)" > "$out/extract.launch.txt"
nohup bash -c '
  set +e
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
out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_process_trace/cd647552_seed0_b16384_until15000_30k_v1
tail -n 20 "$out/extract_general.log"
if [ -f "$out/extract.exitcode" ]; then
  printf '\n提取退出码：'
  cat "$out/extract.exitcode"
  ls -lh "$out/mesh/tsdf_fusion.ply" "$out/mesh/tsdf_fusion_post.ply"
fi
```

提取退出0后下载本次完整output，尤其trace目录；不清理这些历史过程记录。服务器尚未回传上述检查/提取结果，不写成全部采集已验证。
