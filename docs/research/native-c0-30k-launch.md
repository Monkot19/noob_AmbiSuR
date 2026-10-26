# 原生C0：30K/seed0并保存7K checkpoint

2026-10-26用户批准。基线分支`codex/baseline-30k-checkpoint`与C0整树相同，固定提交`d6f15c8891a53800d5e3100f95817a7dd7f98e2f`。不新增基线模块/包装器/钩子；原seed0硬编码，不传二周目的--seed。复用既有C0扩展和输入，不重装/重编译/跑smoke。需有GPU开机。

以下普通shell只准备独立数据视图/输出、记录身份/退出码，Python进程直接运行原train.py，不注入代码。aligned目录必须独立，因为原loader会写points3D.ply。旧C0/4096与源数据均不覆盖。

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/noob_AmbiSuR
  test -z "$(git status --porcelain)"
  git fetch origin codex/baseline-30k-checkpoint
  git switch --detach d6f15c8891a53800d5e3100f95817a7dd7f98e2f
  git rev-parse HEAD
  git diff --exit-code

  native_python=/root/miniconda3/envs/ambisur/bin/python
  native_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
  native_parent=/root/autodl-tmp/ambisur_work/data_views/second_run_selector/toolroom_r2_seed0_30k_v1/colmap_undistorted
  native_view=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
  native_out=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1

  test -d "$native_ext/diff_plane_rasterization_ambisur"
  test -f "$native_parent/sparse_da3_aligned/0/points3D.bin"
  test -f "$native_parent/sparse_da3_aligned/0/trans.json"
  test ! -e "$native_view"
  test ! -e "$native_out"
  mkdir -p "$native_view" "$native_out"
  for name in images sparse estimated_depths estimated_confs; do
    ln -s "$native_parent/$name" "$native_view/$name"
  done
  cp -aL "$native_parent/sparse_da3_aligned" "$native_view/sparse_da3_aligned"
  if [ -f "$native_parent/split.json" ]; then
    cp -aL "$native_parent/split.json" "$native_view/split.json"
  fi

  export PYTHONPATH="$native_ext${PYTHONPATH:+:$PYTHONPATH}"
  printf 'commit=%s\nseed=0\niterations=30000\nresolution=2\ncheckpoint_iteration=7000\nrasterizer_target=%s\nsource_view=%s\n' \
    "$(git rev-parse HEAD)" "$native_ext" "$native_view" > "$native_out/launch.txt"
  git status --short > "$native_out/git.status"
  git diff --exit-code > "$native_out/git.diff"

  nohup bash -c '
    "$1" -u -B train.py --source_path "$2" --model_path "$3" \
      -r 2 --iterations 30000 --checkpoint_iterations 7000
    native_rc=$?
    printf "%s\n" "$native_rc" > "$3/train.exitcode"
    exit "$native_rc"
  ' _ "$native_python" "$native_view" "$native_out" \
    > "$native_out/train.log" 2>&1 < /dev/null &

  native_pid=$!
  printf '%s\n' "$native_pid" > "$native_out/train.pid"
  printf '训练PID：%s\n日志：%s/train.log\n' "$native_pid" "$native_out"
)
```

查看进度（不要在7000轮停止，基线继续完成30K）：

```bash
native_out=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
tail -n 8 "$native_out/train.log"
```

7K应产生`chkpnt7000.pth`和`app_model/iteration_7000/app.pth`；完成后应有train.exitcode=0、30KGaussian和最终app模型。checkpoint含Gaussian/Adam/增密统计，不含RNG/相机队列/Appoptimizer；不能声称完整随机轨迹恢复。旧7KPLY不作恢复源。

后续候选恢复实现仅在二周目分支，不在运行中的baseline代码上编辑。baseline运行期间也不要切换服务器代码版本；等基线退出后再切换候选代码。候选4096/16384共用此7Kcheckpoint，原调度到30000，最终复用既有mesh提取/正式评价。
