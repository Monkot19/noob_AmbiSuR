# 共享7K实验：下载归档核对与服务器清理范围

日期：2026-10-27。用户授权核对本地下载，提出之后清理服务器已归档产物。本次只读检查原始归档，报告/清单另存独立analysis目录；未修改训练代码、模型、原始日志或数据集，未访问GT、运行几何评价或删除文件。

## 结论

本地结构、二进制有效性与可读取性检查通过，具备后续几何评价和分析所需材料。共4749文件、7373232294字节，约7.37GB/6.87GiB。服务器两棵目录尚未完整逐文件对照，因此本次不执行清理。共享7K checkpoint已与启动时留存的服务器SHA256匹配。

| 归档 | 文件数 | 字节 | 训练/提取退出码 | 30K高斯数 | raw/post mesh顶点数 |
|---|---:|---:|---|---:|---|
| 新原生C0 | 1578 | 3421450805 | 0 / 0 | 1291587 | 13166585 / 11684066 |
| 4096 | 1581 | 1982660340 | 0 / 0 | 1276690 | 13248949 / 11704284 |
| 16384 | 1581 | 1969120439 | 0 / 0 | 1275897 | 13067281 / 11612408 |

候选父目录另有9个队列/身份文件，共710字节；pair与提取queue退出码均0。原生C0无run_identity.json/resolved_config.json是其原生输出约定，用原launch、cfg及干净Git记录核对，不当作下载缺失。候选没有独立7K checkpoint/7K PLY是本次共享前缀续训安排，并非丢文件。

## 已执行检查

- 三组406相机、邻居记录及初始化点云内容相同；每组406张RGB、406张深度可视化、406张法线可视化，图像名覆盖全部相机。全部4683张JPEG完整解码，无错误。
- 所有13个PLY文件长度与二进制头部严格吻合，顶点坐标有限；Gaussian浮点属性有限；6个mesh的三角面个数及全部索引范围有效。没有建立BVH或计算GT距离。
- 全部5个PyTorch ZIP归档逐成员CRC通过；基线7K checkpoint与4份App权重实际在CPU加载成功。原生checkpoint迭代7000、16字段、1421298高斯、Adam有6项状态/7参数组。
- 本地checkpoint为1074511797字节，SHA256=`c0264cf6765cec8bf771311f5e6a9e88d995fe5903dbcecd31c71fa7da8b645b`，与候选父目录prefix.sha256所记原服务器文件相同。
- 两候选cfg_opts仅candidate_selector_budget不同；seed0/r2/总30K/共享7K路径/干净c351身份吻合。三份提取均使用5m、0.005m、截断系数4、主要连通簇2，不启用额外深度过滤。
- 为全部4749文件生成相对路径、长度、SHA256清单。空git.status/git.diff是干净状态记录，不是截断文件。其他文件均非空。

证据位于：

```text
D:\research_Space\output\AmbiSuR_second_run\ScanNetpp\Tool_Room\analysis\shared_native7k_archive_20261027_v1
  verification.json
  baseline_manifest.json
  pair_manifest.json
```

本地模型路径记录在verification.json；完整归档保持用户提供的目录结构。清单摘要仅对相对路径、字节数和文件内容散列生成，不依赖Windows/Linux时间戳。

## 清理前的服务器只读对照

无需GPU，可无卡开机执行。读取约7.37GB文件一次，只打印摘要，不修改源目录。两行均MATCH后才进入具体删除范围确认。此项只影响服务器清理，不阻塞本地几何评价。

```bash
python -B <<'PY'
from pathlib import Path
import hashlib, json
roots = {'baseline': '/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1', 'pair': '/root/autodl-tmp/ambisur_runs/Tool_Room/selector_from_native7k/c3510b02_seed0_30k_pair_v2'}
expected = {'baseline': (1578, 3421450805, 'b6e9ea2dc7ae5f88de2f09d04cf3cba54dc80dad9b85a01ae5212a9969b8fbdb'), 'pair': (3171, 3951781489, '9bc3808cbae49079ac32796eb27b5a72f13a28cd8cbc43a14af55bace69aa8f5')}
for label, directory in roots.items():
    root = Path(directory)
    if not root.is_dir():
        print(label, "MISSING", directory, flush=True)
        continue
    rows = []
    files = sorted((p for p in root.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(root).as_posix())
    for index, path in enumerate(files, 1):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        rows.append([path.relative_to(root).as_posix(), path.stat().st_size,
                     digest.hexdigest()])
        if index % 1000 == 0:
            print(label, "checked", index, "/", len(files), flush=True)
    encoded = (json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    actual = (len(rows), sum(row[1] for row in rows), hashlib.sha256(encoded).hexdigest())
    print(label, "MATCH" if actual == expected[label] else "MISMATCH", actual, flush=True)
PY
```

## 建议分两步清理（待用户确认，不是删除授权）

只针对上述新native baseline与两个候选run，不包含旧固定baseline、一周目其他目录或任何数据/模型权重环境。

1. 三组各自的`mesh/tsdf_fusion.ply`、`train/ours_30000`、`debug`、`pg_view`：本地均已完整归档，按本地文件逻辑长度估算3133210288字节（约3.13GB/2.92GiB）。实际释放以服务器du/df为准。
2. 若仍需空间，三组`mesh/tsdf_fusion_post.ply`另1823147794字节（约1.82GB/1.70GiB），本地正式几何评价用本地副本即可。两步合计约4.96GB。

服务器先保留共享chkpnt7000.pth、基线7K App、各组30K Gaussian/App、基线7K PLY、cfg/相机/邻居/日志/身份/退出码与小型队列记录；数据集、DA3先验/权重、Conda环境、C0扩展和代码仓库均不碰。这样后续实验可以继续使用共同前缀，必要时可重提取渲染/mesh。

一周目其他服务器输出需要先按当前du清单逐目录识别并核对本地归档，不能仅因名称旧或声称下载过就批量删除。实际删除命令只在摘要匹配并确认具体范围后提供。
