# 原生7K checkpoint：4096与16384配对续训

2026-10-26。用户回执确认新原生基线退出码为0，7K checkpoint为1074511797字节，7K外观权重为14349字节，30K点云为320315108字节。候选固定版本为 `f41dac0b47f5d00397e346ff6cedad0c107471ff`；恢复修复只调整外观权重的来源目录与迭代号，基线分支仍完整指向C0。

两组按顺序运行，分别从完成7000轮后的同一checkpoint续训7001—30000轮，各23000步。两组使用相同Gaussian/Adam状态、7K外观权重、重启seed0、30K调度及r2/DA3输入；唯一方法参数差别为候选预算4096与16384。原生checkpoint未保存RNG、相机队列及外观优化器状态，因此不能声称逐步复现基线连续训练轨迹。

直接运行原train.py，不新增测试、编译、安装或训练框架。各组独立复制对齐目录，以承接原loader对points3D.ply的写入；图像和DA3数组仅链接。原基线输入与输出只读。第一组失败后保留产物并停止队列，不自动启动第二组。请完整复制以下代码块执行。

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/noob_AmbiSuR
  test -z "$(git status --porcelain)"
  git fetch origin codex/second-run-selector
  git switch --detach f41dac0b47f5d00397e346ff6cedad0c107471ff
  git rev-parse HEAD
  git diff --exit-code

  pair_python=/root/miniconda3/envs/ambisur/bin/python
  pair_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
  pair_prefix=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
  pair_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
  pair_views=/root/autodl-tmp/ambisur_work/data_views/selector_from_native7k/f41dac0b_seed0_30k_pair_v1
  pair_out=/root/autodl-tmp/ambisur_runs/Tool_Room/selector_from_native7k/f41dac0b_seed0_30k_pair_v1

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
)
```

查看总日志pair.log确定当前预算，再查看对应train.log。Scene构造可能先打印原始点云初始化信息，恢复后的实际循环从7001开始，进度条总步数为23000。运行期间不要切换服务器代码或并行启动另一项GPU训练。

本命令不自动提取mesh或访问GT；训练结束后复用已经确认的提取设置与正式评价程序，对新基线和两组候选进行比较。未收到服务器启动回执前，项目状态只记录为命令已准备。
