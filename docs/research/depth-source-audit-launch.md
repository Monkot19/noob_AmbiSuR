# 原生7K位置来源检查：只读采集与MVSAnywhere推理

2026-11-09用户已批准。D分支`codex/depth-source-audit`。本批没有训练、优化器step、重定位、baseline重训或renderer改动。服务器只生成17视角的同7K全图深度和MVS深度，GT仅在随后本地评分使用。

固定执行提交：`AUDIT_FIXED_COMMIT`。配置`experiments/depth_sources/toolroom_native7k_v1.json`。

## 本批实际检查范围

- 原17视角（5个示例＋12个系统覆盖）来自同一原生7K；旧Gaussian候选记录不能冒充渲染表面深度。
- `capture`复用原采集器，新增默认关闭的全图保存选项；记录current/DA3/spread及mask、相机和两个原支持视角，不写原数据／模型。
- `infer`固定官方MVSAnywhere `5bd49bba`，hero权重、480×640、7个原邻居、1次refinement。采用独立venv，继承原torch/torchvision并用约束文件禁止替换其版本；依赖、权重和缓存均置于数据盘。供体构造器的重复权重下载改为指定缓存文件，forward/preprocess继承原函数。额外DINO等依赖可能下载。
- 新MVS深度按两份K对应的同一射线重采样；不是简单对齐图像角点。原DA3／spread记录保持原生产协议。
- 本地固定法线NCC只是准入**对照探针**：无A排除，同两源共同有效集取均值；Q须严格优于current与DA3且代价<0.5。不以当前渲染深度的往返误差阈值否决Q，但保留原函数的有限／正深度等检查。这不是已实现的真实观测对应准入，也不是获准训练mask。
- 来源统计分native及common4；NCC统计分全部可比较／获准／拒绝／同数量随机。抽样域hybrid收益与“只替换这些实际获准抽样像素”的全图分母收益分开；均不是训练或mesh收益。示例视角、系统覆盖和合并统计分别保存，不把重复像素当独立样本。

## 服务器一键运行

完整复制下块。外层是独立Bash，失败不会关闭交互终端。`audit.log`是阶段总日志；每阶段另有log/exitcode。源目录和checkpoint不变，只写新audit/runtime/vendor/weight目录。若已有同名output，命令会停止，不能覆盖旧记录。

```bash
bash <<'SH' || printf '\n检查启动失败，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
fixed=AUDIT_FIXED_COMMIT
if ! git cat-file -e "$fixed^{commit}" 2>/dev/null; then
  if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
  git fetch origin codex/depth-source-audit
fi
git switch --detach "$fixed"
git rev-parse HEAD
git diff --exit-code

audit_python=/root/miniconda3/envs/ambisur/bin/python
audit_ext=/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python
audit_prefix=/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1
audit_source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
audit_out=/root/autodl-tmp/ambisur_runs/Tool_Room/depth_source_check/20261109_v1
test -s "$audit_prefix/chkpnt7000.pth"
test -d "$audit_ext/diff_plane_rasterization_ambisur"
test -d "$audit_source/images"
test ! -e "$audit_out"
test ! -e "$audit_out.log"
mkdir -p "$(dirname "$audit_out")"
df -h /root/autodl-tmp

nohup "$audit_python" -u -B scripts/evaluate_depth_sources.py server \
  --config experiments/depth_sources/toolroom_native7k_v1.json \
  --source "$audit_source" --baseline "$audit_prefix" \
  --rasterizer "$audit_ext" --output "$audit_out" \
  --runtime /root/autodl-tmp/ambisur_runtime/mvsanywhere-env \
  --vendor /root/autodl-tmp/ambisur_runtime/mvsanywhere-vendor \
  --weights /root/autodl-tmp/ambisur_models/mvsanywhere/mvsanywhere_hero.ckpt \
  > "$audit_out.log" 2>&1 < /dev/null &
printf '检查PID：%s\n总日志：%s.log\n' "$!" "$audit_out"
SH
```

查看进度：

```bash
audit_out=/root/autodl-tmp/ambisur_runs/Tool_Room/depth_source_check/20261109_v1
tail -n 15 "$audit_out.log"
for stage in capture dependencies inference; do
  if [ -f "$audit_out/$stage.log" ]; then
    printf '\n%s：\n' "$stage"
    tail -n 5 "$audit_out/$stage.log"
  fi
done
if [ -f "$audit_out/audit.exitcode" ]; then
  printf '\n总退出码：'
  cat "$audit_out/audit.exitcode"
fi
```

预期成功：capture/inference/audit均exit0，`capture/capture_manifest.json`含17项depth_file，`mvs/inference_manifest.json`含17视角。首次会下载官方模型及DINO／matching encoder权重；不能用“暂无inference_manifest”单独判断卡死。服务端进程不会运行GT评分。供体可信官方checkpoint加载进程设置`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`兼容宿主Torch2.7；原训练环境及训练命令不变。

完成后仅下载新`20261109_v1`目录及同级`.log`；无需下载独立runtime、vendor或权重，也无需重复下载baseline／数据集。保留失败记录和退出码；接口失败与科学负结果分开，不覆盖失败目录重跑。

## 本地评分

下载后由同一`evaluate_depth_sources.py score`入口处理，参数为config/capture/mvs/GT/output及可选GT语义文件。会先验证实际生产输出、相机／支持metadata身份、完成NCC并保存非GT探针，再初始化原GTQueries；GT只产出评价汇总，不产训练mask。当前还没有MVS质量、准入收益或新mesh结论。评分前需核验输出和路径，不提前启动训练。
