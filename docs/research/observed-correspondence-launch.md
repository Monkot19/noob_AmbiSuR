# 四参考／八对真实对应离线检查

批准规格：`docs/superpowers/specs/2026-11-10-observed-correspondence-probe-design.md`。
分支：`codex/observed-correspondence-audit`。固定实现提交：`7c698a3f11e5df50eaa5d70d3947639e2c39ed18`；不会启动训练，不改DA3、MVS、renderer或baseline。

## 这批执行什么

复用原生7K capture及已有MVS结果。原4份probe冻结65536个位置；RGB-only RoMaV2 precise查询8对对应，同时记录反向查询、overlap/precision，随后固定相机线性三角化。服务器不接受GT参数；完成manifest和NPZ冻结后，本地另用既有4份GT网格评价，不重查询mesh。

DSC06348→DSC06347/349；DSC06669→DSC06559/560；DSC06274→DSC06275/276；DSC06289→DSC06288/290。两个失败与两个成功对照不足以声明泛化，匹配分数/重投影/循环一致都不是已校准正确率。

## 一次准备

将交付目录中的两个bundle和`observed-correspondence-probes.zip`上传至`/root/autodl-tmp/`。代码bundle以前次服务器已有的b769fd55为前提；RoMa source bundle为完整源历史，不含模型权重。主权重会由官方构造路径缓存，校验1095883548bytes与官方SHA256。

旧ambisur是Torch2.7.1/vision0.22.1，不满足供体vision>=0.23。此处新建**不继承site-packages**的venv，固定Torch2.8.0+cu128/vision0.23.0+cu128及fused-local-corr0.3.28；后者PyPI声明torch==2.8.0。不能无约束安装最新版fused，它当前要求Torch2.11。依据：[PyTorch官方历史版本](https://pytorch.org/get-started/previous-versions/)、[作者fused wheel元数据](https://pypi.org/project/fused-local-corr/0.3.28/)。

额外成本包含独立Torch/CUDA运行库、约1.02GiB模型、DINOv3代码缓存与临时下载；不只是1GB模型。以下先要求数据盘空闲至少20GiB，缓存全部在数据盘；这不是峰值保证。8对顺序运行，第一对属于正式批次，无额外smoke。实际显存峰值和耗时逐对保存；OOM/安装失败保留原记录，不自动降低分辨率或更换后端。fused官方wheel无需编译AmbiSuR扩展。

```bash
bash <<'SH' || printf '\n准备停止，请贴回错误；不要改动原ambisur环境。\n'
set -euo pipefail
trap 'rc=$?; printf "准备停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
git bundle verify /root/autodl-tmp/observed-correspondence-audit.bundle
git fetch /root/autodl-tmp/observed-correspondence-audit.bundle refs/heads/codex/observed-correspondence-audit
git switch --detach 7c698a3f11e5df50eaa5d70d3947639e2c39ed18
git rev-parse HEAD
git diff --exit-code

runtime=/root/autodl-tmp/ambisur_runtime/observed-correspondence
mkdir -p "$runtime/tmp" "$runtime/cache/torch" "$runtime/cache/huggingface"
export TMPDIR="$runtime/tmp" TORCH_HOME="$runtime/cache/torch" HF_HOME="$runtime/cache/huggingface"
export PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
df -h / /root/autodl-tmp
/root/miniconda3/envs/ambisur/bin/python - <<'PY'
import shutil
free=shutil.disk_usage('/root/autodl-tmp').free
print('Data disk free GiB:',round(free/2**30,2))
if free<20*2**30:raise SystemExit('Stop: prepare at least 20 GiB data-disk space first.')
PY

if [ ! -d "$runtime/RoMaV2/.git" ]; then
  git init "$runtime/RoMaV2"
  git -C "$runtime/RoMaV2" fetch /root/autodl-tmp/romav2-pinned-source.bundle HEAD
  git -C "$runtime/RoMaV2" switch --detach FETCH_HEAD
fi
test "$(git -C "$runtime/RoMaV2" rev-parse HEAD)" = 95c9968145c8906b7b59383258e9f73b02853d89
test -z "$(git -C "$runtime/RoMaV2" status --porcelain)"
if [ ! -x "$runtime/env/bin/python" ]; then
  /root/miniconda3/envs/ambisur/bin/python -m venv "$runtime/env"
fi
python="$runtime/env/bin/python"
"$python" -m pip --version
"$python" -m pip install --no-cache-dir --only-binary=:all: --upgrade 'pip==25.3' \
  --index-url https://pypi.org/simple
"$python" -m pip install --no-cache-dir --only-binary=:all: \
  -r experiments/depth_sources/correspondence-requirements.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128
"$python" -m pip check
"$python" -m pip freeze > "$runtime/environment.freeze.txt"
"$python" - <<'PY'
import json,zipfile
from pathlib import Path
from scripts.evaluate_depth_sources import file_sha
root=Path('/root/autodl-tmp/ambisur_runtime/observed-correspondence/probes')
config=json.loads(Path('experiments/depth_sources/toolroom_correspondence_v1.json').read_text())
with zipfile.ZipFile('/root/autodl-tmp/observed-correspondence-probes.zip') as z:
    for view in config['views']:
        name=view['reference']+'_probe.npz';content=z.read(name)
        import hashlib
        assert hashlib.sha256(content).hexdigest()==view['probe_sha256'],name
        root.mkdir(parents=True,exist_ok=True);p=root/name
        if p.exists():assert file_sha(p)==view['probe_sha256'],name
        else:p.write_bytes(content)
print('Four original probes ready. No GT or model loaded.')
PY
printf '\n独立环境与原probe已准备；下一段启动8对正式离线检查。\n'
SH
```

## 已建环境、但依赖安装因flit_core失败时恢复

2026-11-10用户实际日志：Git与供体源码准备成功，pip拒绝Jinja2/typing_extensions的wheel名称后退回源码包，构建子进程继承唯一PyTorch索引，找不到flit_core，准备第35行退出1。未进入RoMa加载/匹配，日志不是空间不足。尚未读取服务器pip版本，因此旧pip是待验证的上游原因；恢复先打印版本，仅升级本独立venv的pip25.3，再用PyPI＋官方cu128索引及固定二进制包恢复。依据：[pip安装来源与解析阶段](https://pip.pypa.io/en/stable/cli/pip_install/)。原7c698a3科学代码、环境之外的Conda/DA3/MVS均不变，不必重传bundle或重新clone。

已经执行过前半段的服务器，直接运行这一段；不重复新建环境，也不以已经开始安装后的20GiB预检阻挡恢复。旧失败记录保留，不声称已在服务器修好。安装结果和剩余空间由本命令实际输出确认。

```bash
bash <<'SH' || printf '\n依赖恢复停止，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "恢复停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test "$(git rev-parse HEAD)" = 7c698a3f11e5df50eaa5d70d3947639e2c39ed18
test -z "$(git status --porcelain)"
runtime=/root/autodl-tmp/ambisur_runtime/observed-correspondence
python="$runtime/env/bin/python"
test -x "$python"
mkdir -p "$runtime/tmp" "$runtime/cache/torch" "$runtime/cache/huggingface"
export TMPDIR="$runtime/tmp" TORCH_HOME="$runtime/cache/torch" HF_HOME="$runtime/cache/huggingface"
export PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
"$python" -m pip --version
"$python" -m pip install --no-cache-dir --only-binary=:all: --upgrade 'pip==25.3' \
  --index-url https://pypi.org/simple
"$python" -m pip install --no-cache-dir --only-binary=:all: \
  -r experiments/depth_sources/correspondence-requirements.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128
"$python" -m pip check
"$python" -m pip freeze > "$runtime/environment.freeze.txt"
"$python" - <<'PY'
import hashlib,json,zipfile
from pathlib import Path
from scripts.evaluate_depth_sources import file_sha
root=Path('/root/autodl-tmp/ambisur_runtime/observed-correspondence/probes')
config=json.loads(Path('experiments/depth_sources/toolroom_correspondence_v1.json').read_text())
with zipfile.ZipFile('/root/autodl-tmp/observed-correspondence-probes.zip') as z:
    for view in config['views']:
        name=view['reference']+'_probe.npz';content=z.read(name)
        assert hashlib.sha256(content).hexdigest()==view['probe_sha256'],name
        root.mkdir(parents=True,exist_ok=True);p=root/name
        if p.exists():assert file_sha(p)==view['probe_sha256'],name
        else:p.write_bytes(content)
print('Four original probes ready. No GT or model loaded.')
PY
df -h / /root/autodl-tmp
du -sh "$runtime/env" "$runtime/cache" "$runtime/tmp"
printf '\n依赖和原probe准备完成；可执行指南中的八对检查启动段。\n'
SH
```

## 权重下载83%后hash失败的恢复

2026-11-10用户原four_views_v1日志：870MiB/1.02GiB、下载66分钟后invalid hash，退出1，位置在下载校验，尚未加载RoMa/匹配。重新读取GitHub官方release确认1095883548bytes、SHA2561557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7；不改预期哈希。官方Torch2.8下载器finally会删除未通过校验的临时包，不能承诺复用之前83%。原日志/目录保留，恢复只处理同一官方缓存，不改7c698a3科学代码、参数或匹配器。

先独立下载，保留`.part`以便连接中断后继续；只在大小/SHA256都通过时转为正式缓存。以下下载无需GPU；若官方链路仍慢，可在本地下载同一[官方权重](https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt)，上传为这里的`.part`文件，再执行此段校验。没有下载或运行本地模型。

```bash
bash <<'SH' || printf '\n权重准备停止；保留part文件，请贴回错误。\n'
set -euo pipefail
cd /root/autodl-tmp/noob_AmbiSuR
runtime=/root/autodl-tmp/ambisur_runtime/observed-correspondence
python="$runtime/env/bin/python"
weights="$runtime/cache/torch/hub/checkpoints/romav2.0.1.pt"
part="$weights.part"
mkdir -p "$(dirname "$weights")"
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
if [ ! -f "$weights" ]; then
  size=0
  if [ -f "$part" ]; then size=$(stat -c %s "$part"); fi
  if [ "$size" -lt 1095883548 ]; then
    curl --fail --location --continue-at - --connect-timeout 20 --max-time 900 \
      --output "$part" \
      https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt
  fi
fi
"$python" - "$weights" <<'PY'
import hashlib,os,sys
from pathlib import Path
target=Path(sys.argv[1]);source=target if target.is_file() else Path(str(target)+'.part')
expected='1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7'
size=source.stat().st_size
print('Downloaded bytes:',size,flush=True)
assert size==1095883548,'Incomplete/incorrect size; keep the part file.'
h=hashlib.sha256()
with source.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
print('SHA256:',h.hexdigest(),flush=True)
assert h.hexdigest()==expected,'Hash differs; keep the file, do not load it.'
if source!=target:os.replace(source,target)
print('Official RoMaV2 checkpoint verified and cached.')
PY
SH
```

校验成功后执行下方相同启动段，**仅将out改为**`/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_views_weight_retry_v2`。原four_views_v1目录/log/exitcode全部保留，不重复安装环境，不覆盖旧输出。固定7c698a3会校验并复用正式缓存；后续DINOv3源码下载仍可能联网，原60秒self-stack继续保留。服务器恢复由用户执行，尚无实际匹配结果。

## 启动正式八对检查

不需要重新capture、运行MVS或训练。选择新输出目录；已存在则停止，不覆盖旧记录。`match`会先校验实际图片header、原probe散列、7K/支持/相机身份，再加载模型。每对开头/结尾打印进度；启用60秒自身调用栈，网络缓存等待也可定位。

```bash
bash <<'SH' || printf '\n对应检查启动停止，请贴回上方错误。\n'
set -euo pipefail
trap 'rc=$?; printf "启动停止：第%s行，退出码%s\n" "$LINENO" "$rc" >&2' ERR
cd /root/autodl-tmp/noob_AmbiSuR
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = 7c698a3f11e5df50eaa5d70d3947639e2c39ed18
git rev-parse HEAD
git diff --exit-code
runtime=/root/autodl-tmp/ambisur_runtime/observed-correspondence
audit=/root/autodl-tmp/ambisur_runs/Tool_Room/depth_source_check/20261109_v1
source=/root/autodl-tmp/ambisur_work/data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted
out=/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_views_v1
test -x "$runtime/env/bin/python"
test -s "$audit/capture/capture_manifest.json"
test -s "$audit/mvs_support_cap_v4/inference_manifest.json"
test ! -e "$out"
test ! -e "$out.log"
mkdir -p "$(dirname "$out")"
export TMPDIR="$runtime/tmp" TORCH_HOME="$runtime/cache/torch" HF_HOME="$runtime/cache/huggingface"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
# Deliberately do not inherit MVS's TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD setting.
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD TORCH_FORCE_WEIGHTS_ONLY_LOAD

observer=$(cat <<'PY'
import faulthandler,runpy,sys
faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
entry='scripts/evaluate_depth_sources.py';sys.argv=[entry,*sys.argv[1:]]
try:runpy.run_path(entry,run_name='__main__')
finally:faulthandler.cancel_dump_traceback_later()
PY
)
nohup bash -c '
  "$1" -u -B -c "$2" match \
    --config experiments/depth_sources/toolroom_correspondence_v1.json \
    --source "$3" --capture "$4/capture" --mvs "$4/mvs_support_cap_v4" \
    --probes "$5/probes" --vendor "$5/RoMaV2" --output "$6"
  rc=$?
  printf "%s\n" "$rc" > "$6.exitcode"
  exit "$rc"
' _ "$runtime/env/bin/python" "$observer" "$source" "$audit" "$runtime" "$out" \
  > "$out.log" 2>&1 < /dev/null &
printf '对应检查PID：%s\n日志：%s.log\n' "$!" "$out"
SH
```

```bash
out=/root/autodl-tmp/ambisur_runs/Tool_Room/observed_correspondence/four_views_v1
tail -n 30 "$out.log"
if [ -f "$out.exitcode" ]; then cat "$out.exitcode"; ls -lh "$out/correspondence_manifest.json"; fi
```

退出码0且manifest包含4参考/8对后，下载`four_views_v1`整个目录、同级log/exitcode及独立环境`environment.freeze.txt`。不要下载约1GB权重/整个环境。不清理旧模型或数据。

## 本地单独评分

下载完成再执行既有本地评价Python，`--matches`指向新下载目录，GT-cache指向已存在的`analysis/depth_source_failure_evidence_20261110_v1/scored`，输出选新路径：

```powershell
& D:/research_Space/ambisur-eval-runtime/Scripts/python.exe -B scripts/evaluate_depth_sources.py correspondence-score `
  --matches <下载目录/four_views_v1> `
  --gt-cache D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/analysis/depth_source_failure_evidence_20261110_v1/scored `
  --output <新分析目录>
```

输出按每参考/原区域/common3与common4/单源与双源报告；原NCC与新对应在相同冻结像素域对比。随机对照掩码在服务器非GT证据阶段已冻结。三角化保存pair0/pair1/joint、camera-z目标、原3D点与原参考射线重投影、夹角、深度分歧和±1原生像素敏感度。无效反向插值不是零循环误差，无效扰动不是零敏感度。全体扰动有效才给最大绝对z变化，不把这些数值当训练准入概率。

## 许可证／算子来源

`utils/observed_correspondence.py`的归一化SVD与多视角特征值小算子适配COLMAP3.11.1，额外仅记录有限/数值退化/正深度及规定敏感度；不导入COLMAP BA/SfM。COLMAP原通知如下：

Copyright (c) 2023, ETH Zurich and UNC Chapel Hill. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of ETH Zurich and UNC Chapel Hill nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

RoMaV2源bundle保留作者完整Git历史与LICENSE；网络权重使用作者发布链接，不纳入项目Git。
