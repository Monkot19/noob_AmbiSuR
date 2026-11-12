# 四来源深度选择：29对离线检查

2026-11-12用户已返回服务器c4546a3、退出0、最终manifest约15K和74M目录的成功回执。当前无需再执行启动命令；请直接按“下载与本地评价”归档新目录及.log/.exitcode。实际manifest／NPZ完整性和GT精度待本地检查，不能把采集完成等同选择有效。

用户已确认[首批规格](../superpowers/specs/2026-11-10-four-source-depth-routing-design.md)。D主目录分支`codex/four-source-depth-routing`，固定实现`c4546a390004c8c9bbf41b12e21b0a0a2d919cac`。只采集RGB对应／离线几何，不是30K训练，不接受GT参数，不改数据、baseline、原NCC／ALR／renderer。

## 复用与成本

- 四个原参考、65536个原像素，current／DA3／MVS／spread直接复制原成功8对产物中的数组；不重采集7K、不重跑MVS或传播。
- 原生C0邻居列表29对，8对原NPZ直接复用／复制到新产物，**实际只新增21对RoMa precise前向**。新目录最终包含4个参考NPZ＋29对NPZ，manifest记录原文件SHA和reused标记。
- 复用`observed-correspondence/env`、原RoMa权重与DINO源码缓存；不建环境、不安装或升级任何包。新输出为NPZ级，预计几十至几百MiB，实际大小由用户查看；不承诺其他GPU峰值。原8对4090运行reserved峰值约4.48GiB仅作参考。
- 此批无额外GPU smoke；第一个新配对就是正式21对之一。旧8对完整性与全部相机／相关照片的预检在模型加载前完成。失败保留新目录与日志，不自动续跑或重算GT。

## 服务器直接复制

如GitHub不稳定，先把交付包中的`four-source-depth-routing.bundle`上传到`/root/autodl-tmp/`。代码已存在时无需联网；bundle以服务器已有7c698a3为前提，不含模型、数据或GT。下面不会重建环境或重下载主权重；缺缓存时直接停在检查处。

```bash
bash <<'SH' || printf '\n四来源检查启动停止，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"

fixed=c4546a390004c8c9bbf41b12e21b0a0a2d919cac
if ! git cat-file -e "$fixed^{commit}" 2>/dev/null; then
  bundle=/root/autodl-tmp/four-source-depth-routing.bundle
  if [ -f "$bundle" ]; then
    git bundle verify "$bundle"
    git fetch "$bundle" refs/heads/codex/four-source-depth-routing
  else
    if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
    git fetch origin codex/four-source-depth-routing
  fi
fi
git switch --detach "$fixed"
git rev-parse HEAD
git diff --exit-code

runtime=/root/autodl-tmp/ambisur_runtime/observed-correspondence
python="$runtime/env/bin/python"
source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
baseline=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
existing=/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_views_offline_cache_v3
out=/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_source_routing_v1

test -x "$python"
test -s "$existing/correspondence_manifest.json"
test -s "$existing/input_receipt.json"
test -s "$baseline/cameras.json"
test -s "$baseline/multi_view.json"
test -d "$source/images"
test -s "$runtime/cache/torch/hub/checkpoints/romav2.0.1.pt"
test -s "$runtime/cache/torch/hub/facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa/hubconf.py"
test ! -e "$out"
test ! -e "$out.log"
test ! -e "$out.exitcode"
df -h / /root/autodl-tmp
export TMPDIR="$runtime/tmp" TORCH_HOME="$runtime/cache/torch" HF_HOME="$runtime/cache/huggingface"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD TORCH_FORCE_WEIGHTS_ONLY_LOAD
mkdir -p "$TMPDIR" "$(dirname "$out")"

observer=$(cat <<'PY'
import faulthandler,runpy,sys
faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
entry='scripts/evaluate_depth_sources.py';sys.argv=[entry,*sys.argv[1:]]
try:runpy.run_path(entry,run_name='__main__')
finally:faulthandler.cancel_dump_traceback_later()
PY
)
nohup bash -c '
  "$1" -u -B -c "$2" routing-match \
    --config experiments/depth_sources/toolroom_four_source_routing_v1.json \
    --source "$3" --baseline "$4" --existing "$5" --vendor "$6" --output "$7"
  rc=$?
  printf "%s\n" "$rc" > "$7.exitcode"
  exit "$rc"
' _ "$python" "$observer" "$source" "$baseline" "$existing" "$runtime/RoMaV2" "$out" \
  > "$out.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$out.pid"
printf '对应检查PID：%s\n日志：%s.log\n' "$!" "$out"
SH
```

预期日志先显示`29 pairs / 8 reused / 21 new`；逐对输出REUSE或START／DONE，四个参考分别FROZEN。退出0且`correspondence_manifest.json`存在才表明采集完成，不等于深度选择有效。

```bash
out=/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_source_routing_v1
tail -n 30 "$out.log"
if [ -f "$out.exitcode" ]; then
  printf '\n退出码：'; cat "$out.exitcode"
  if [ -f "$out/correspondence_manifest.json" ]; then
    ls -lh "$out/correspondence_manifest.json"
    du -sh "$out"
  fi
fi
```

## 下载与本地评价（采集成功后）

下载新`four_source_routing_v1`整个目录及同级`.log/.exitcode`即可。目录内已复制原8对，无需再上传／下载权重或完整训练output。先本地核对manifest／原65536像素／四来源数组与旧产物一致，然后用同一入口`routing-score --matches ... --gt-cache ... --output ...`；GT参数只在本地评分出现，复用已有4个GT网格，不查询mesh。

输出维持`results.json/metrics.csv`：两种支持协议的完整冻结域／各自可比较域／交集域分开，包含最终选中目标、DA3回退、配额随机及逐来源统计。current胜出只评价静态位置，不是自监督教师。三来源与spread有效的四来源结果不混合；GT后有效数量不冒称等预算或统计独立。

尚无本批GPU前向、GT质量、训练或表面收益；本地通过仅说明接线与已读输入合同成立。新训练mask／分区loss／权重／支持池扩展及13参考后续批次不由本指南自动启动。

## 已保存诊断的读取方式

每个参考NPZ保留`all_candidate_costs_px[N,4,S]`、投影有效性、全部pair几何和两套选中支持索引。对协议`old_first2/new_best2`及来源数`common3/common4`：

- `*_means`保存全部来源均值；只在`*_domain=True`的行对均值作稳定升序排序，即得到完整来源排序。精确并列保留并列信息，不把稳定索引顺序当额外证据；不可比较行没有排名。
- `*_winner`为-1表示弃权，`*_margin_px`为次小减最小；`*_support_agreement_count`是两个支持中单独唯一胜者与最终胜者一致的数量。
- 若要逐支持布尔标记，按`{protocol}_support_indices`取出每个支持的各来源代价；只在共同域且最终winner>=0时，检查该支持的最小代价是否唯一且来源等于最终winner。原代价均已保存，无需重推理／读GT。

独立终审指出完整排序／逐支持同意未另存重复数组，上述读取方式补清其接口；本版保持计算和NPZ底层证据不变。旧／新组合在相同两支持时的真实8对适配结果完全一致；这不是新增21对质量验证。
