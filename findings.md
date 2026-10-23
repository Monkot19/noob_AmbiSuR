# AmbiSuR 新项目 Findings

本文件只记录新项目通过代码、测试和实验得到的事实。设计假设不得写成已证实结论。

## Confirmed Inputs

- Spec：`docs/research/ambisur-reliability-routing-design.md`
- Fork：`https://github.com/Monkot19/noob_AmbiSuR`
- GPU：RTX 4090 24GB
- 主场景：Tool Room、Utility Room
- 开发场景：Tool Room seed 0

## Evidence Table

| Claim | Status | Evidence/Path | Next check |
|---|---|---|---|
| feature-off 与 baseline 等价 | untested | | E0 |
| N 比 A 或 1-S 更能定位几何风险 | hypothesis | | D0/G1 |
| 双可靠性能校准风险 | hypothesis | | coverage-risk |
| Abstain 能减少错误选边 | hypothesis | | C2 vs C3 |
| 参数路由改善几何 | hypothesis | | C3–C5 |
| 生命周期改善稳定性与几何 | hypothesis | | C5 vs C6 |

## Code Findings

每项记录：文件与行号、可复现证据、是否已测试、是否与设计冲突。不要只写“可能有问题”。

## Experiment Findings

每项必须引用对应 manifest、commit、scene 和 seed。不得把单次运行写成稳定结论。

## Decisions

| Date | Decision | Evidence | Consequence |
|---|---|---|---|



## 2026-10-22 Clean second-run migration
User approves c0-baseline branch plus selector-only port and necessary seed/metadata reuse. Old records above are historical. The original dirty checkout stays unchanged. No baseline/GT/storage operation.
- Server disk evidence supplied by user: overlay30G used25G/free5.3G, persistent50G used36G/free15G; no inode issue. pip cache nearly empty, conda tarballs106.7MB. Three environments (base,ambisur,da3-v1) and VSCode CLI3.8G explain major visible use; da3-v1 is an environment, not proof of duplicate DA3 weights. Overlay du43G versus df25G means visible directory totals cannot be treated as reclaimable writable-layer bytes.
- Do not relocate/delete Conda or DA3 envs, model weights or old results. Next no-GPU inventory can focus on VSCode versions/active executable paths and installed rasterizer provenance. No remote cleanup executed.

## 2026-10-23 Server cache and installed extension evidence
- User-supplied no-GPU receipt shows six Stable VSCode Server versions under /root/.vscode-server/cli/servers totaling3.8G; no matching executable links were printed. This alone does not prove no other client references a version.
- Fresh read-only local code --version returns1.131.0, commit e4c7e7b1d6d060162f4aa7f8225271b67ce1df75, matching the server671M Stable directory; retain that version and propose only the five explicitly listed older directories, after closing all remote VSCode clients and using the web terminal. Approximate candidate directory total3.1GiB, actual recovery requires df.
- User's active ambisur imports rasterizer wrapper and binary from /root/miniconda3/envs/ambisur/lib/python3.10/site-packages and exposes rasterize_gaussians_with_evidence=True. Thus branch/source migration alone has not selected a baseline-only installed extension; API presence alone does not prove numerical interference.
- c0-baseline ext.cpp exports only rasterize_gaussians/rasterize_gaussians_backward/mark_visible (corrected after server receipt: markVisible is the C++ symbol and Python wrapper method, not the extension export); setup.py uses CUDAExtension/BuildExtension. Propose building pinned baseline extension into a separate small data-disk target and using per-process PYTHONPATH for the second run, preserving the first-run package/environment. Before no-GPU build, verify nvcc/toolkit and target architecture; explicit TORCH_CUDA_ARCH_LIST is needed without visible GPUs. No installation, code changes, deletion or remote execution this step.

### 2026-10-23 Build receipt and assertion correction
- User successfully removed the five explicitly named old VSCode versions: system df now22G used/8.3G available/73%; CLI672M. Actual recovery is approximately3G, no Conda/model/results cleanup needed now.
- Pinned c0 wheel built and independently installed at /root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python; wheel SHA256 56a2e694c96145e81081d15486679e8bb3fb2eb3b95d2d3ba6e1538a0de37346. Import and wrapper/binary target assertions passed before the erroneous markVisible assertion; absence of old evidence export was not yet checked because execution stopped first.
- Root cause verified with git show c0-baseline:.../ext.cpp: m.def("mark_visible", &markVisible). Baseline wrapper markVisible calls _C.mark_visible. Agent's handoff command used the wrong extension name; no evidence of a build failure. Correct only the one-shot validation command and reuse the same target; native GPU tests/training remain pending.
- Subsequent user receipt confirms corrected check passed with wrapper/binary in the independent target, all baseline exports callable and old evidence export absent. This establishes import/export selection only; it does not establish GPU rendering, selector integration or reconstruction benefit.

### 2026-10-23 AutoDL native acceptance and next run configuration
- User-pasted full output confirms exact3d85faf95951b4806eee18498841e66551b93320 on RTX4090, baseline extension selected from U365Ym/python, all23 unified tests pass in2.875s with no skips. Native producer/renderer/ALR backward+optimizer, normal-coordinate, feature-off/e0 and callback checks actually executed. Synthetic active helper27.8098ms/17607168peak allocated bytes is not a real-scene time/memory estimate. Tensorboard absence did not prevent existing console/file metadata tests.
- Read-only frozen Tool_Room_r2 cfg_args/cfg_opts and train.log confirm406views,r2,eval=False,seed0 baseline defaults,30k,DA3,unc_from_iter7000,original ray/trunc/depth settings. No baseline checkpoint/resume required; candidate first activates at7001. Dataset source path in old cfg is historical and cannot be assumed still present.
- Original project task plan requires canonical uploaded data to stay read-only: baseline loader can write sparse_da3_aligned/0/points3D.ply. Reuse a qualified writable Tool Room working view (or discuss a minimal private copy); do not launch directly against canonical data. Need current server path, not another scientific audit or new smoke framework.
- Directory receipt resolves an existing c0 r2 working-view parent. To preserve old input state, use a new thin view; image/DA3 array links do not duplicate bulk data, and an independent aligned sparse copy absorbs loader's unconditional PLY rewrite. Fresh actual-parser comparison to frozen cfg_args/cfg_opts passes, with only intended path and new selector/seed fields differing. Directory existence alone is not proof of training success; launch command checks required input files before starting.
