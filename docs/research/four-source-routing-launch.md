# 全像素四来源混合深度：29对采集与评价

2026-11-12用户已批准全像素／可变来源／单支持与“来源数优先、支持数其次”。固定生产提交`53600fd83e4bc488ee3ad99df801ba7fe9f3855a`，分支`codex/four-source-dense-routing`。已完成本地接口验证与独立审查；**实际服务器GPU、整图来源质量尚未验证**。不训练、不读GT、不新建环境、不安装或下载权重。

## 本次运行与资源

- 4参考×584×876=2,046,336参考像素，29对重新match，每对一次；65,536仅为查询分块大小，不是像素预算。总14,835,936对查询。
- 原整图current/DA3/spread、MVS原预测及regrid全部复用；旧29对仅有抽样查询结果，不能当全像素缓存直接补齐。
- 原RoMa/DINO缓存与`observed-correspondence/env`复用。若缓存缺失或身份不同，离线检查停止，不自动下载。当前不用DA3环境或renderer扩展。
- 每对逐个GPU前向，CPU缓存／记录随全像素增大。按字段规模粗估未压缩约3–4GiB，压缩后实际大小待本次运行记录；启动时要求数据盘至少空6GiB。旧GPU reserved约4.48GiB仅作参考，尚无本次真实峰值或总时长承诺。
- 输出新目录`four_source_dense_v2`，旧采集、评价、baseline和数据全部保留。无额外smoke或测试前置。

## 服务器复制这一整段

如GitHub不稳定，可先上传`D:/research_Space/handoff_bundle/four-source-dense-routing_20261112/four-source-dense-routing.bundle`到`/root/autodl-tmp/`；代码已在本地时无需联网。该bundle以服务器已有c4546a3为前提，不包含模型／照片／GT。

```bash
bash <<'SH' || printf '\n全像素采集启动停止，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
fixed=53600fd83e4bc488ee3ad99df801ba7fe9f3855a
if ! git cat-file -e "$fixed^{commit}" 2>/dev/null; then
  bundle=/root/autodl-tmp/four-source-dense-routing.bundle
  if [ -f "$bundle" ]; then
    git bundle verify "$bundle"
    git fetch "$bundle" refs/heads/codex/four-source-dense-routing
  else
    if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
    git fetch origin codex/four-source-dense-routing
  fi
fi
git switch --detach "$fixed"
git rev-parse HEAD
git diff --exit-code
runtime=/root/autodl-tmp/ambisur_runtime/observed-correspondence
python="$runtime/env/bin/python"
source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
baseline=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
audit=/root/autodl-tmp/ambisur_runs/Tool_Room/depth_source_check/20261109_v1
existing=/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_source_routing_v1
out=/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_source_dense_v2
test -x "$python"
test -d "$source/images"
test -s "$existing/correspondence_manifest.json"
test -s "$audit/capture/capture_manifest.json"
test -s "$audit/mvs_support_cap_v4/inference_manifest.json"
test -s "$runtime/cache/torch/hub/checkpoints/romav2.0.1.pt"
test -s "$runtime/cache/torch/hub/facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa/hubconf.py"
test ! -e "$out"
test ! -e "$out.log"
test ! -e "$out.exitcode"
df -h / /root/autodl-tmp
"$python" -c 'import shutil; f=shutil.disk_usage("/root/autodl-tmp").free; print("Data disk free: %.2f GiB"%(f/2**30)); assert f>=6*2**30,"Reserve at least 6 GiB before capture"'
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
    --config experiments/depth_sources/toolroom_four_source_dense_v2.json \
    --source "$3" --baseline "$4" --existing "$5" \
    --capture "$6/capture" --mvs "$6/mvs_support_cap_v4" \
    --vendor "$7/RoMaV2" --output "$8"
  rc=$?
  printf "%s\n" "$rc" > "$8.exitcode"
  exit "$rc"
' _ "$python" "$observer" "$source" "$baseline" "$existing" "$audit" "$runtime" "$out" \
  > "$out.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$out.pid"
printf '采集PID：%s\n日志：%s.log\n' "$!" "$out"
SH
```

## 查看进度与完成条件

```bash
out=/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_source_dense_v2
tail -n 30 "$out.log"
if [ -f "$out.exitcode" ]; then
  printf '\n退出码：'; cat "$out.exitcode"
  ls -lh "$out/correspondence_manifest.json"
  du -sh "$out"
fi
```

每对DONE只说明匹配查询完成；每参考FROZEN表示该图的选择／地图完成。必须总退出码0且最终manifest存在，再下载完整目录及同级.log/.exitcode/.pid。失败保留目录与日志，不覆盖、自动恢复或换规则。

每图主NPZ含全部像素及成本／来源／支持／回退证据；`*_maps.npz`含H×W混合浮点深度与原始标签，`*_sources.png`图例为黑=回退DA3、蓝=current、绿=明确DA3、橙红=MVS、紫=spread；`*_supports.png`灰度0/100/200对应0/1/2个实际共享支持。参考原照片观察，PNG颜色不是误差图或GT标签。

## 下载后的本地正式评价

建议本地新目录`D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/observed_correspondence_full_image_four_source_20261112/four_source_dense_v2`。确认下载完整后再执行原routing-score，复用4份原GT缓存，不启动新mesh查询；以下路径按实际下载位置调整，新output必须不存在。

```powershell
& 'D:/research_Space/ambisur-eval-runtime/Scripts/python.exe' -u -B scripts/evaluate_depth_sources.py routing-score `
  --matches D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/observed_correspondence_full_image_four_source_20261112/four_source_dense_v2 `
  --gt-cache D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/analysis/depth_source_failure_evidence_20261110_v1/scored `
  --output D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/analysis/four_source_dense_20261112_v1
```

主DA3表的`full_image`包含回退区域，不能只挑接受区域；各参照X的count/MAE/gain均来自GT、X、混合目标共同有效域，不被无关current或DA3缺失过滤。单／双支持及2/3/4来源单列。离线混合目标好坏不等于削弱loss有效或30K表面收益，GT后失败不自动重跑。旧抽样模式保持可回读。

---

# 历史：29对抽样离线检查

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
