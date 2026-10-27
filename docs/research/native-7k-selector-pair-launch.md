# 原生7K checkpoint：4096与16384配对续训

2026-10-26。用户回执确认新原生基线退出码为0，7K checkpoint为1074511797字节，7K外观权重为14349字节，30K点云为320315108字节。候选固定版本为 `c3510b029247e783d3163b2c8e7ce5d1f182039d`；恢复修复包含指定7K外观权重，以及显式weights_only=False读取自行生成、含NumPy标量的原生checkpoint；基线分支仍完整指向C0。

此前f41任务PID14168在首次checkpoint读取处失败，未进入续训；本次使用独立v2目录，保留原失败记录。

两组按顺序运行，分别从完成7000轮后的同一checkpoint续训7001—30000轮，各23000步。两组使用相同Gaussian/Adam状态、7K外观权重、重启seed0、30K调度及r2/DA3输入；唯一方法参数差别为候选预算4096与16384。原生checkpoint未保存RNG、相机队列及外观优化器状态，因此不能声称逐步复现基线连续训练轨迹。

直接运行原train.py，不新增测试、编译、安装或训练框架。各组独立复制对齐目录，以承接原loader对points3D.ply的写入；图像和DA3数组仅链接。原基线输入与输出只读。第一组失败后保留产物并停止队列，不自动启动第二组。请完整复制以下代码块执行，包括开头的bash行和末尾的SH。独立Bash承接set -e，错误会显示行号，避免退出当前交互终端。

```bash
bash <<'SH' || printf '\n启动未成功，请贴回上方错误。\n'
  set -euo pipefail
  trap 'rc=$?; printf "启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
  cd /root/autodl-tmp/noob_AmbiSuR
  test -z "$(git status --porcelain)"
  git fetch origin codex/second-run-selector
  git switch --detach c3510b029247e783d3163b2c8e7ce5d1f182039d
  git rev-parse HEAD
  git diff --exit-code

  pair_python=/root/miniconda3/envs/ambisur/bin/python
  pair_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
  pair_prefix=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
  pair_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
  pair_views=/root/autodl-tmp/ambisur_work/data_views/selector_from_native7k/c3510b02_seed0_30k_pair_v2
  pair_out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_from_native7k/c3510b02_seed0_30k_pair_v2
  pair_failed=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_from_native7k/f41dac0b_seed0_30k_pair_v1

  test -s "$pair_failed/pair.exitcode"
  test "$(cat "$pair_failed/pair.exitcode")" != 0

  test "$(cat "$pair_prefix/train.exitcode")" = 0
  test -s "$pair_prefix/chkpnt7000.pth"
  test -s "$pair_prefix/app_model/iteration_7000/app.pth"
  test -f "$pair_source/sparse_da3_aligned/0/points3D.bin"
  test -f "$pair_source/sparse_da3_aligned/0/trans.json"
  test -d "$pair_ext/diff_plane_rasterization_ambisur"
  test ! -e "$pair_views"
  test ! -e "$pair_out"
  mkdir -p "$pair_views" "$pair_out"

  for budget in 4096 16384; do
    view="$pair_views/b$budget/colmap_undistorted"
    out="$pair_out/b$budget"
    mkdir -p "$view" "$out"
    for name in images sparse estimated_depths estimated_confs; do
      ln -s "$pair_source/$name" "$view/$name"
    done
    cp -aL "$pair_source/sparse_da3_aligned" "$view/sparse_da3_aligned"
    if [ -f "$pair_source/split.json" ]; then
      cp -aL "$pair_source/split.json" "$view/split.json"
    fi
    printf 'commit=%s\nseed=0\niterations=30000\nresume_iteration=7000\nbudget=%s\ncheckpoint=%s\nrasterizer_target=%s\n' \
      "$(git rev-parse HEAD)" "$budget" "$pair_prefix/chkpnt7000.pth" "$pair_ext" > "$out/launch.txt"
  done
  sha256sum "$pair_prefix/chkpnt7000.pth" > "$pair_out/prefix.sha256"
  git status --short > "$pair_out/git.status"
  git diff --exit-code > "$pair_out/git.diff"
  export PYTHONPATH="$pair_ext${PYTHONPATH:+:$PYTHONPATH}"

  nohup bash -c '
    for budget in 4096 16384; do
      out="$3/b$budget"
      printf "budget=%s start\n" "$budget"
      "$1" -u -B train.py \
        --source_path "$2/b$budget/colmap_undistorted" --model_path "$out" \
        -r 2 --iterations 30000 --seed 0 --enable_candidate_selector \
        --candidate_selector_budget "$budget" --start_checkpoint "$4" \
        > "$out/train.log" 2>&1 < /dev/null &
      train_pid=$!
      printf "%s\n" "$train_pid" > "$out/train.pid"
      wait "$train_pid"
      train_rc=$?
      printf "%s\n" "$train_rc" > "$out/train.exitcode"
      printf "budget=%s exitcode=%s\n" "$budget" "$train_rc"
      if [ "$train_rc" -ne 0 ]; then
        printf "%s\n" "$train_rc" > "$3/pair.exitcode"
        exit "$train_rc"
      fi
    done
    printf "0\n" > "$3/pair.exitcode"
  ' _ "$pair_python" "$pair_views" "$pair_out" "$pair_prefix/chkpnt7000.pth" \
    > "$pair_out/pair.log" 2>&1 < /dev/null &

  pair_pid=$!
  printf '%s\n' "$pair_pid" > "$pair_out/pair.pid"
  printf '配对训练PID：%s\n总日志：%s/pair.log\n4096日志：%s/b4096/train.log\n16384日志：%s/b16384/train.log\n' \
    "$pair_pid" "$pair_out" "$pair_out" "$pair_out"
SH
```

查看总日志pair.log确定当前预算，再查看对应train.log。Scene构造可能先打印原始点云初始化信息，恢复后的实际循环从7001开始，进度条总步数为23000。运行期间不要切换服务器代码或并行启动另一项GPU训练。

本命令不自动提取mesh或访问GT；训练结束后复用已经确认的提取设置与正式评价程序，对新基线和两组候选进行比较。未收到服务器启动回执前，项目状态只记录为命令已准备。

## 三组30K模型顺序提取（2026-10-27）

16384组最新日志已完成23K续训，PSNR30.840198135375978，L10.01708100289106369，N1275897，耗时58:42。4096已确认exitcode0/30KPLY302MB/App15KB；16384的最终退出码与文件由下方命令在启动前检查。三份都取这次新原生baseline及共享7K候选，不使用旧C0mesh作为本次替代。

保持原提取参数5m/0.005m/4/2；不启用深度过滤、不访问GT。baseline使用原始C0代码提取，两组候选使用c351；实际extract_general.py在两个commit逐字节相同。真实CLI和Bash语法核对通过，不增加GPU测试。所有版本已在服务器本地，无需fetch。队列仅在训练全部成功、模型非空、目标mesh/日志尚不存在时启动；一次顺序提取，失败保留文件并停止，不覆盖旧产物。

```bash
bash <<'SH' || printf '\n提取未启动，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "提取准备停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
git cat-file -e d6f15c8891a53800d5e3100f95817a7dd7f98e2f^{commit}
git cat-file -e c3510b029247e783d3163b2c8e7ce5d1f182039d^{commit}

mesh_python=/root/miniconda3/envs/ambisur/bin/python
mesh_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
mesh_base=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
mesh_pair=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_from_native7k/c3510b02_seed0_30k_pair_v2
mesh_queue="$mesh_pair/mesh_extraction_v1"

test "$(cat "$mesh_pair/pair.exitcode")" = 0
for out in "$mesh_base" "$mesh_pair/b4096" "$mesh_pair/b16384"; do
  test "$(cat "$out/train.exitcode")" = 0
  test -s "$out/point_cloud/iteration_30000/point_cloud.ply"
  test -s "$out/app_model/iteration_30000/app.pth"
  test ! -e "$out/mesh"
  test ! -e "$out/extract_general.log"
  test ! -e "$out/extract.exitcode"
done
test ! -e "$mesh_queue"
mkdir -p "$mesh_queue"
export PYTHONPATH="$mesh_ext${PYTHONPATH:+:$PYTHONPATH}"

nohup bash -c '
  set -u
  trap '\''rc=$?; printf "%s\n" "$rc" > "$4/queue.exitcode"'\'' EXIT
  for label in baseline b4096 b16384; do
    if [ "$label" = baseline ]; then
      out="$2"
      commit=d6f15c8891a53800d5e3100f95817a7dd7f98e2f
    else
      out="$3/$label"
      commit=c3510b029247e783d3163b2c8e7ce5d1f182039d
    fi
    git switch --detach "$commit" || exit "$?"
    git diff --exit-code || exit "$?"
    printf "commit=%s\niteration=30000\nresolution=2\nmax_depth=5.0\nvoxel_size=0.005\nsdf_trunc_scale=4.0\nnum_cluster=2\nuse_depth_filter=false\n" "$commit" > "$out/extract.launch.txt" || exit "$?"
    printf "%s start\n" "$label"
    "$1" -u -B mesh_extract/extract_general.py \
      -m "$out" -r 2 --iteration 30000 --skip_test \
      --max_depth 5.0 --voxel_size 0.005 \
      --sdf_trunc_scale 4.0 --num_cluster 2 \
      > "$out/extract_general.log" 2>&1 < /dev/null
    rc=$?
    if [ "$rc" -eq 0 ] && { [ ! -s "$out/mesh/tsdf_fusion.ply" ] || [ ! -s "$out/mesh/tsdf_fusion_post.ply" ]; }; then
      rc=1
    fi
    printf "%s\n" "$rc" > "$out/extract.exitcode"
    printf "%s exitcode=%s\n" "$label" "$rc"
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
  done
' _ "$mesh_python" "$mesh_base" "$mesh_pair" "$mesh_queue" \
  > "$mesh_queue/queue.log" 2>&1 < /dev/null &

mesh_pid=$!
printf '%s\n' "$mesh_pid" > "$mesh_queue/queue.pid"
printf '提取队列PID：%s\n总日志：%s/queue.log\n' "$mesh_pid" "$mesh_queue"
SH
```

总日志为pair目录下mesh_extraction_v1/queue.log，完成退出码为queue.exitcode；逐模型extract_general.log、extract.exitcode、extract.launch.txt和mesh/tsdf_fusion{,_post}.ply均位于各自output。提取期间不要切换服务器代码。baseline原训练checkpoint/模型不改，loader只写其获准private输入视图。

全部成功后下载新native_c0_ckpt7k/d6f15c88_seed0_30k_v1与selector_from_native7k/c3510b02_seed0_30k_pair_v2两个完整目录（后者含两预算），保存point_cloud、app_model、7Kcheckpoint、mesh、渲染、cfg、日志/退出码与身份记录。用户下载后在D盘复用现有正式mesh evaluator比较三个模型；不要只下载mesh而丢弃复现材料。
