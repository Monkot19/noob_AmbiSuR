# AmbiSuR 二周目交接入口

> **静态检查格式修复（2026-10-29）：** 用户已租服务器，c5e6243正式capture在读取sparse/0/images.bin时失败，未进入候选计算。审计入口遗漏了基线已有binary→text回退；本地实际source只有text，完整406相机用真实reader读取后位姿/内参与C0归档吻合。固定修复代码`9603ef679fc3253c49fbeb73e9c9c38ec3fe09f7`仅改审计输入薄适配，原训练/selector/COLMAP reader不变，相机顺序保留；记录pose_source/pose_format。真实text/binary回退测试RED→GREEN，两既有测试入口共46项39CPU通过/7native跳过，窄终审无阻断。保留失败static_v1，按[更新指南](static-candidate-audit-launch.md)在新static_v2采集，不修改服务器源码或重训。新原生执行/候选收益仍待回执。

> **静态候选检查已批准、本地实现（2026-10-28）：** 用户授权离开期间自行解决常规问题，随后明确服务器已关闭；不得自动开机/租卡。D目录新隔离分支`codex/selector-static-audit`已发布（首次13e1287远端与本地一致），保留既有训练分支；先前未提交分析文档已保存为2c2fb1f。固定代码`c5e624307de97f91898e461e5e789a4d8ee0eae3`，新增单入口`evaluate_candidate_actions.py capture/score`，调用实际selector并观察返回数据，不改训练/选择/重定位/渲染器。范围为原生7K、C0/完整V2/early30K，原五视角加12个名称等间隔覆盖视角，分组报告；无15K快照，不补训，终态探针不是历史决策重放。真实CPU selector→NPZ→GT/BVH/法线→CSV/JSON链路通过；37项统一测试30CPU通过/7native缺扩展跳过。独立终审发现FP32批量舍入误触发坐标逐位核对，已RED→GREEN修复并导出真实返回目标；修复复审无进一步阻断问题，独立复跑37项结果一致。真实capture未执行，尚无接受候选的几何收益比例。操作见[统一静态检查指南](static-candidate-audit-launch.md)；下次用户已开启服务器后仅运行这一次前向检查，不训练、不自动选择新方法。

> **DA3与动作关系核对（2026-10-28）：** 用户要求先理解差区域、数据集与位置/法线先验关系。本地11份先验/变换文件SHA匹配训练manifest；五个固定视角151832可见网格点离线GT对照发现metric位置偏差可大而法线相对可用，最高置信也常不优于C0。不是历史已接受候选的因果精度，不能据此断言损伤均由DA3造成。实际ALR已覆盖A∪E、重定位事件移动E，E限定~A；“重定位选中也加ALR”已存在，独立R需要明确与A交集及选择/移动频率。ScanNet++可继续作压力测试，不替代官方DTU/TnT可比性。下一步建议仅讨论一次真实接受候选离线核查，再决定目标修正/ALR质量；没有批准新训练/方法修改。详见[DA3误差与动作选择](2026-10-28-da3-failure-regions-and-action-selection.md)。


> **early 区域与论文主张核查（2026-10-28）：** 复用同一离线程序，三模型六方向均值及双向CD分解精确复现；early对C0完整性改善部分-3.922mm、损伤+3.112mm，约79.34%抵消。wall69双向贡献-0.589879mm，其余区域净-0.363854mm，收益比之前分散；wall89等仍损伤。没有方法达到理论上限的证据，也不足以凭单场景2.34%宣布论文主创新；NCC候选比较/位置替换/阶段门已见Sparse2DGS实际源码，新增适配需要独立收益和泛化证据。已读ConfidenceMattersGS固定官方train实际加权代码；未改训练/方法或开新实验。详见[区域、上限与论文主张](2026-10-28-selector-v2-early-regions-and-claims.md)。下一步范围待讨论，勿自动补跑或调参。


> **early首次正式评价完成（2026-10-28）：** 下载关键产物及train/extract退出码0、精确79次重定位序列已核对。原evaluator一次四模型评价退出码0/132.27秒，旧3模型指标/hash和共同GT/协议精确复现，JSON/CSV12行一致。early Accuracy34.679897mm、Completeness44.809032mm、CD39.744465mm、F1@5cm0.787739、trainPSNR30.850549；CD相对C0改善2.34343%、相对V1改善2.22385%、相对完整V2改善0.87855%，PSNR恢复到V1附近。本次四组中几何均值最好但增益有限。额外核对发现前79次moved计数与完整V2不逐条相同（323721vs325258合计），尽管实际优化参数/相机/记录环境一致，不能当完全相同前15K轨迹或严格因果证明。详见[early结果报告](2026-10-28-selector-v2-early-results.md)，产物evaluation/selector_v2_early_native7k_seed0_30k_v1。尚未做early区域分解，不能声称旧负区域已修复；下一步范围待讨论，不自动加训练/改算法。

> **v2_early训练完成日志（2026-10-28）：** 用户返回a52ab904对应输出：全局30000、23000/23000、58:27、Training complete12:34:24及Gaussian/App保存；N1226364、train L1=0.017084373719990255、PSNR=30.850548934936526。比完整V2高0.158683dB、比V1-16384高0.010351dB、比连续C0低0.221731dB，均为既有训练视角统计，非新视角/几何结论。末段26800..30000只有selector日志，符合截止后继续V1路径的预期；完整79次重定位序列、退出码/文件非空仍待服务器检查。下一步仅提取此early模型（5m/.005m/trunc4/cluster2），不重提取旧模型、不改方法；尚无earlymesh或几何评价结果。

> **v2_early分支已发布（2026-10-28）：** 新origin/codex/second-run-selector-v2-early首次发布HEAD78312ce4ed8dc38cd31cf3f8958d4046a88316b9，git ls-remote与本地精确一致；之后仅更新文档收据。固定生产a52ab904、固定启动文档ad8ecc7357db21ea595a43971df7e471bd5a886a均已在远端。D主目录保持early分支，工作树清洁。用户可直接执行early指南首个bash块或提取固定文档执行，无需再确认已批准范围；尚未收到early服务器启动/GPU/训练/收益回执。

> **v2_early实现与终审完成（2026-10-28）：** 用户同意只改重定位窗口。D分支codex/second-run-selector-v2-early，固定生产a52ab904aa09f67e98bc392c6e0edddb15f5a391：可选candidate_relocation_until_iter默认0保持完整V2，显式15000对应7100..14900共79次；截止后仍执行V1/ALR/原优化。仅3个代码文件，真实helper边界/默认与旧配置兼容/元数据测试RED→GREEN，31项25CPU通过、6native缺扩展跳过；原selector其余函数AST及train/optimizer/renderer未改。一次独立终审无Critical/Important/Minor，独立复跑31项25pass/6skip；启动Bash/实际CLI核对通过，逐字段与原V2 cfg_opts相同，仅新增截止值。启动命令见[early指南](selector-v2-early-native7k-launch.md)，可用固定文档ad8ecc7357db21ea595a43971df7e471bd5a886a提取首个bash块执行。尚无early服务器/GPU/收益回执；原生7K/30K及V1/完整V2继续复用，不自动加实验。

> **V2区域定位完成（2026-10-28）：** 用户批准利用现有输出定位收益/损失。真实GT分割与顶点坐标顺序核对后，复用正式采样/距离生产函数，完整性和准确性六个均值复现。69号墙贡献全局CD变化-0.609137mm，超过整场景V2-V1的-0.551690mm，其余区域合计+0.057447mm；不是其他区域全部变差。原V1最好/最差10% GT点在V2中完整性分别均值退化1.693mm/改善7.177mm，属描述性分层，非因果证明。门旁墙71、门68及部分管道为负贡献；天花板/储物架虽完整性略差，双向贡献净改善，不能笼统称其整体退化。详见[区域分析](2026-10-28-selector-v2-regional-analysis.md)，图片及CSV在analysis/selector_v2_regions_20261028_detail。当前资料不能区分候选误选/重定位力度/随后优化拓扑后处理；下一轮仅讨论最小变量，不自动增加训练或可靠性框架。

> **V2正式几何结果（2026-10-28）：** 本地下载关键文件齐全、train/extract退出码0；229次重定位时序精确、227次非零。复用原正式evaluator三模型评价退出码0/76.15秒，C0/V1旧指标和mesh SHA、共同GT样本/协议精确复现。V2 Accuracy34.813288mm、Completeness45.380178mm、CD40.096733mm、F1@5cm0.787254；相对C0 CD降低1.47787%、F1增加0.729875个百分点，相对V1－16384 CD降低1.35722%。两向距离与各阈值PRF均改善，属于幅度有限的单场景单seed正向结果；trainPSNR30.691866继续低于两对照，不能称整体显著突破。报告[2026-10-28-selector-v2-geometry-results.md](2026-10-28-selector-v2-geometry-results.md)，产物在output的evaluation/selector_v2_native7k_seed0_30k_v1。没有更改算法/评价/训练或清理数据；下一步与用户讨论针对已有输出的分析及后续范围。

> **V2训练完成日志回执（2026-10-28）：** 用户返回a114ed5对应output末段：全局ITER30000、23000/23000、耗时53:07、Training complete（10:11:37），Gaussian/App保存日志；N1225318，train L1=0.017557380348443986、PSNR=30.691865921020508。较V1-16384低0.14833221435547017dB，较原生连续C0低0.3804141998291044dB；均为训练视角统计，不作新视角/几何结论。末段每100轮重定位非零，29000/29200/29400/29600/29800的moved与对应added一致，29900后30000无重定位，符合末轮排除规格；尚未核对全229次事件。退出码与模型非空待服务器检查，尚无V2 mesh或几何指标。下一步仅提取V2，沿用5m/0.005m/trunc4/cluster2及现有正式evaluator；旧baseline/V1本地归档复用，不重训、不重复提取旧模型、不调整算法。

> **当前入口（2026-10-27）：** 主工作目录为 `D:\research_Space\noob_AmbiSuR`，当前分支 `codex/second-run-selector-v2`。原V1分支 `codex/second-run-selector`保留在ef333bb；C盘临时worktree保留detached3775135。完整归档与换机步骤见[主目录与恢复指南](second-run-workspace-and-recovery.md)。下方一周目记录仅供查阅，不作为当前派工。

## 当前方法与代码

> **V2已发布（2026-10-27）：** 新GitHub分支`codex/second-run-selector-v2`已推送，首次发布远端HEAD0afa978与本地一致；固定生产代码a114ed5已包含于该分支，随后仅文档收据更新。D主目录仍V2，V1ef333bb、baseline原生d6f15c8及C detached3775135均保留。用户回服务器可直接执行[V2启动指南](selector-v2-native7k-launch.md)完整bash块，无需再审批已批准的实现。尚无V2服务器启动/GPU/重建评价回执。

> **V2本地实现与审查完成（2026-10-27）：** 用户明确授权开始实现、离开期间自行解决常规问题并事后报告。代码固定`a114ed5b5ee1c265eea5ececc98268934eca6b2e`：同一V1选择核心额外输出乱序ID配对的世界目标；每100轮7100..29900在主render前只移动E行并清理局部xyz动量/stat，原ALR/LR/loss/拓扑继续。统一30项24CPU通过、6native因本地缺扩展跳过；Python3.10语法与原核心AST核对通过，独立整体审查无需修复问题。唯一中间问题是测试夹具缺实际标定接口，接入真实Camera方法解决，未改算法。AutoDL直接7K→30K命令见[V2启动指南](selector-v2-native7k-launch.md)，bash与真实CLI组检查通过；尚无服务器启动/GPU/几何收益回执。baseline分支仍原生d6f15c8，所有旧输出保留；不追加selector-off/GPU smoke作为前置任务。

> **V2实际改法待审阅（2026-10-27）：** 用户基本同意最小规格，但要求从供体源码出发汇报具体函数改法。新增[源码到代码适配清单](2026-10-27-selector-v2-code-adaptation.md)：重新取固定官方train/model，字节匹配；逐项映射选择目标输出、当前主render前插槽、局部Adam/stat原地更新及原测试入口。明确供体SGU替代ADC而V2保留ADC，不能宣称整模块照搬；五个文件预计修改，renderer/CUDA/ALR/metadata/evaluator复用。生产代码未改，尚未创建V2分支或启动训练。

> **selector_V2路线已批准（2026-10-27）：** 用户确认名称`selector_V2`、定义`V2=V1+重定位`，选择保留V1并给补选E增加位置更新，再接原ALR。已写[最小规格](../superpowers/specs/2026-10-27-selector-v2-relocation-design.md)；每100轮、7100..29900窗口、主渲染前执行及局部状态处理是待确认的具体提议。生产代码仍c351，未开新训练；继续复用现有7K及30K baseline，不补selector-off对照作为前置任务。规格确认后再从D盘创建V2隔离分支，不静默改变V1或baseline。

> **用户最新优先级（2026-10-27）：** 暂缓同恢复selector-off对照，先讨论能产生实际几何收益的新动作；不要自动启动对照或继续扩大诊断体系。用户提出固定中心比较先验法线，或保留selector_v1并对补选高斯加入Sparse2DGS式位置替换，同时考虑学习率和SH实际数量。本轮只读执行原SH函数：原生7K checkpoint选中213195/1421298、30K模型193737/1291587，均约15%；这不是错误几何计数。供体SGU目标来自模型渲染表面而非DA3；整张量Parameter替换也会影响优化器状态，适配方案须讨论。核对记录追加在[失败机制分析](2026-10-27-shared7k-failure-analysis.md)第7节。生产代码仍c351；尚未批准新方法或新训练。

> **当前分析（2026-10-27）：** 三批清理完成，服务器余14G/74%；用户授权解释效果不佳。新增只读离线分析显示16K的GT点误差有明显局部改善和退化抵消，非完全没动；raw TSDF小收益被post进一步削弱。完整平面NCC与仅新增ALR法线动作有未证明连接，实际平行平面反例已核对；相对连续baseline仍有原生恢复流程混杂，未锁定单一因果。见[失败机制分析](2026-10-27-shared7k-failure-analysis.md)。同恢复selector-off对照/固定中心法线候选仅为待讨论方案，未改代码或开新训练。

> **第二批清理完成（2026-10-27）：** 三份postmesh已由用户清除，df余10G/81%。下一批仅清理已确认本地归档的旧C0与旧3d85selector的mesh/渲染/debug/pg_view，约3.27GB，保留所有模型、App、元数据、旧正式评价和D0-GT审计记录。旧selector本地目录现为selector_v1/selector_v1_4096_without_baseline_checkpoint_2026_10_23。D0/G0/Utility大模型备份尚未确认，不按旧目录名批量删。

> **存储清理回执（2026-10-27）：** 第一批三组rawmesh/render/debug/pg_view清理已由用户执行，df为50G/用42G/余8.3G/84%。用户要求继续，已核对本地三份postmesh及正式评价仍在；第二批只删对应三份服务器postmesh（约1.82GB），保留模型/shared7K/App/元数据/数据环境，附只读du用于识别下一批旧output。尚未收到第二批执行回执，不自动批量清理旧一周目目录。

> **当前结果（2026-10-27）：** 用户撤销服务器全量哈希对照要求，授权尽快清理并分析。已交付仅rawmesh/render/debug/pg_view的第一批约3.13GB清理脚本，保留模型/shared7K/postmesh/数据环境，尚待用户执行回执。本地原正式evaluator已完成三组比较（clean28e/exit0）：CD基线40.698198mm、4096为41.104636、16384为40.648423；16384相对基线仅-0.1223%，完整性仍差，尚无明确整体收益。完整数值与限制见[共享前缀预算结果](2026-10-27-shared7k-budget-results.md)。不继续等待服务器摘要，不自动追加训练/诊断体系。

> **下载核对完成（2026-10-27）：** 用户提供的新baseline与两候选本地归档通过只读检查，共4749文件/约7.37GB；13PLY、5PTH归档CRC及CPU加载、4683图片解码、406视角覆盖全部通过。共享7K checkpoint与原服务器记录SHA相同。完整核对报告和后续只读服务器对照命令见[下载核对与清理范围](2026-10-27-shared7k-download-verification.md)。原件未改、未跑GT评价或删除服务器文件；先对照服务器完整目录摘要，再确认具体清理范围，保留共享前缀、模型和环境。已有本地材料可进入正式几何评价，清理核验不作为其前置门槛。

> **三组提取派工（2026-10-27）：** 16384亦已返回Trainingcomplete（11:33:05），23K耗时58:42，N1275897，trainPSNR30.840198135375978，较新C0低0.23208198547363423dB，较4096高0.018185806274413352dB。4096退出码0及最终PLY302MB/App15KB已确认，队列自动启动16384有真实日志。用户授权提取这次新基线与两候选mesh，随后自行下载三份完整output做本地分析。沿用5m/0.005m/4/2，baseline在C0、候选在c351，串行提取不覆盖已有mesh；[现有指南](native-7k-selector-pair-launch.md)末节提供命令，会先确认全部训练退出码与文件。未收到实际提取回执，几何成绩未知。

> **最新训练回执（2026-10-27）：** c351配对v2的4096组已打印23000/23000、全局ITER30000、Gaussian/App保存及Trainingcomplete（10:33:50），续训耗时57:49，N1276690，trainL1=0.017112244665622712、PSNR=30.822012329101565。相对新原生C0的31.072280120849612下降0.2502677917480476dB；这是5个训练视角指标，不是新视角或几何结论。仍待4096退出码/文件列表，以及16384自动接续状态；不要重复启动队列或在下一组运行时切换代码/提取mesh。两组完成后复用原提取/正式几何评价流程，不增加测试或诊断体系。以下启动失败记录保留为历史。

**当前可执行修复版本：`c3510b029247e783d3163b2c8e7ce5d1f182039d`。** 最新[配对续训指南](native-7k-selector-pair-launch.md)使用`c3510b02_seed0_30k_pair_v2`，4096和16384各7001—30000轮；参数、checkpoint、基线输入不变。启动前确认旧pair已以非零退出码结束；不删除旧目录。代码/指南发布不等于服务器已经重启，尚待用户执行。

> **最新服务器状态（2026-10-26）：** 用户已启动f41配对任务PID14168，但4096在读取原生7K checkpoint时失败，尚未进入续训。原因是原生NumPy标量不被PyTorch默认weights_only加载器接受；不是checkpoint丢失或候选算法负结果。用户授权自行修复。仅二周目读取处显式weights_only=False，原有测试改用真实归一化函数的NumPy返回值，RED→GREEN，19CPU通过/5native跳过，独立审查通过。基线分支整树不变。f41启动指南已失效；待下方固定修复版本的v2新目录命令，不重用或删除失败目录。

> **2026-10-26最新已批准派工：** 用户要求baseline零源码改动并同意原生checkpoint限制。新建`codex/baseline-30k-checkpoint`仅指向C0/d6f15c8；运行原train.py、r2、30000轮、内置seed0、`--checkpoint_iterations 7000`，不加保存模块/钩子，不做smoke。新output为`/root/autodl-tmp/ambisur_runs/Tool_Room/native_c0_ckpt7k/d6f15c88_seed0_30k_v1`；独立privateview位于`data_views/native_c0_ckpt7k/toolroom_r2_seed0_30k_v1/colmap_undistorted`。后续4096/16384读取相同原生7K模型/Adam状态；不承诺RNG/相机队列恢复。下方53K/单次16384/完整状态早停提议均被本次[批准规格](../superpowers/specs/2026-10-26-shared-7000-prefix.md)取代。先交付原生baseline启动命令，尚待服务器回执。

直接运行命令见[原生C0启动指南](native-c0-30k-launch.md)。基线branch已在本地创建并验证整树相同；不在其上添加文档commit。D盘规则/规格/命令记录只提交到二周目分支。不要在基线进程运行期间切换服务器代码版本。

> **原生baseline完成日志已收到：** d6f15c8/PID11962，30000/30000、50:21、Trainingcomplete，trainPSNR31.072280120849612、L10.016401121392846107、N1291587。尚待exitcode及7Kcheckpoint/App/30KPLY列表，不冒称产物已核验。二周目已作唯一生产行修复：从checkpoint所属run、按first_iter加载Appweights，而非新output或latest30K；Gaussian/Adam原恢复逻辑不变。真实方法CPU集成用例RED→GREEN，24项19pass/5native跳过，独立review通过。未改baseline、未新开GPU测试/训练。

**后续产物回执已确认：** exitcode0；chkpnt7000.pth=1074511797字节，7Kapp.pth=14349字节，30KPLY=320315108字节。原生基线训练及关键产物完成，尚无新baseline几何成绩。现可交付二周目固定修复版本的4096/16384原生checkpoint续训命令。

二周目恢复修复固定`f41dac0b47f5d00397e346ff6cedad0c107471ff`，顺序运行4096/16384各7001..30000命令见[配对续训指南](native-7k-selector-pair-launch.md)。新output父目录`/root/autodl-tmp/ambisur_runs/Tool_Room/selector_from_native7k/f41dac0b_seed0_30k_pair_v1`，各有b4096/b16384。只是在本地准备/发布启动命令，尚无服务器候选续训启动回执；不把完成的baseline和未启动的候选混写。

> **2026-10-26当前派工：** 用户已批准候选预算16384、其余不变、直接30K，不增加smoke/安全框架。训练仍固定3d85faf；新输出`/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/3d85faf_seed0_b16384_30k_v1`，新private data view含`toolroom_r2_seed0_b16384_30k_v1`。CLI参数实验无生产代码变化；尚待用户启动回执，详见首轮结果报告末节。输入1219/0/0与源码审计已完成，不重复派工。

> **启动前最新转向：** 用户提出共同7K起点及4096可比性问题，尚未发出启动命令。现有两run只有7K PLY，没有完整7K checkpoint；不能单独将16384从PLY续训当公平预算对照。当前待讨论单次探索性30K或共享7K前缀/两个23K分支（总53K步），不自动实现恢复或启动训练。

- 从 `c0-baseline`（`d6f15c8891a53800d5e3100f95817a7dd7f98e2f`）仅移植已批准候选模块：保留原 SH 双端标签，从未命中池抽样4096，比较当前平面与 DA3 平面的跨视角 NCC，补选后复用原标签渲染与 ALR。
- 当前规格：`docs/superpowers/specs/2026-10-21-candidate-selector-design.md`；移植记录：`docs/superpowers/plans/2026-10-22-clean-selector-migration.md`。
- 方法提交 `d979c07`；服务器验证／训练固定 `3d85faf95951b4806eee18498841e66551b93320`。本地接管起点 `3775135690b2ea4edfdc50e3b19fba467d02d117` 与它仅有文档差异；本次也只更新规则和文档，实际当前 HEAD 用 `git rev-parse HEAD` 读取。
- 旧 D 未提交内容保存为归档 `7b3154b7daa7e03ca7dad5b0e5cf1260f4e1099d`，不是当前二周目实现；不将旧 Core、五状态、梯度路由或 Utility 恢复自动合并／恢复。

## 已验证与首轮30K完成日志

- AutoDL RTX4090／`3d85faf` 统一23项测试全部通过、零跳过，包含真实 renderer／ALR backward／optimizer step。
- D 接管后同一入口23项中18项通过、5项原生集成因本地缺少 AmbiSuR 扩展而跳过；8个代码文件语法检查通过。1697个已跟踪文件齐全，192个忽略文件未改变。
- Tool Room 首轮30K已由用户实际启动：406张训练图像、r2、seed0、`eval=False`、DA3、预算4096，保留原调度，候选／ALR从7001轮启用。
- 用户最新回传为30000/30000、保存Gaussian及app_model并打印 `Training complete`；日志总时长1:14:56，train PSNR31.102751541137696／L1 0.0163764713332057，点数1295772。末轮sampled4096／comparable3143／added1463，只证明补选分支活动，尚无几何收益结论。
- 服务器输出：`/root/autodl-tmp/ambisur_runs/Tool_Room/selector_v1/3d85faf_seed0_30k_v1`。
- 输入工作视图：`/root/autodl-tmp/ambisur_work/data_views/second_run_selector/toolroom_r2_seed0_30k_v1/colmap_undistorted`。
- 独立基线扩展：`/root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python`，通过进程级 `PYTHONPATH` 选择；旧一周目扩展保留。
- 已确认服务器环境为Python3.10／Torch2.7.1+cu128／CUDA toolkit12.8／GCC11.4／RTX4090，Python入口 `/root/miniconda3/envs/ambisur/bin/python`。下方旧环境字段不是当前环境。
- 本地主目录整理没有干预服务器作业。用户已回传 `train.exitcode=0`、30K点云307M和app.pth15K，训练完成及非空模型产物已确认；不重训或恢复checkpoint。

## 正式mesh评价入口（2026-10-23）

- 用户回传406视角渲染/TSDF/后处理结束，新mesh raw12962510、post11470565顶点；后续下载核验两run提取退出码均0，mesh身份匹配。顶点数量不代表几何收益。
- 用户授权可复用正式评价，并确认PRF阈值1/2/5cm、5cm主列。评价固定提交 `4010fd01e4b5bbe651ac4e221e00532bf99f4ff9`（已推送），统一入口 `scripts/evaluate_mesh_geometry.py`，协议 `experiments/mesh_geometry/scannetpp_surface_v1.json`，完整用法/既有资产命令见 [正式mesh评价指南](mesh-geometry-evaluation.md)。
- Accuracy/Completeness/Surface Chamfer + Precision/Recall/F1；1M面积样本、seed0、共同GT、点到三角面、毫米/0–1，不ICP/裁剪/截断。训练代码和DTU/TnT原脚本未修改。ScanNet++适配协议不冒称官方排行榜或DTU/TnT完全同口径。
- 外部独立本地环境Open3D0.19.0真实8项测试全过/零跳过，含原TnT函数数值对照、真实PLY/CLI/metadata/JSON/CSV和输入只读；审查seed偏差已RED→GREEN修复。Python3.10语法检查通过。后续服务器8项测试全过/零跳过（10.591s）及Completed/完整CSV已收到。用户自行下载原件后，agent进行了获准离线GT空间诊断，未连接服务器。

## 首次正式几何结果（2026-10-23）

- [完整结果与解释](2026-10-23-toolroom-selector-v1-results.md)：baseline→selector的Accuracy35.1244→37.2194mm，Completeness46.3068→54.8356mm，Surface Chamfer40.7156→46.0275mm（+13.0463%，变差）。
- F1@1/2/5cm均略升，5cm为0.77933825→0.78048925（+0.1151个百分点）；5cm Recall下降0.2820个百分点。结论为混合结果、尚无整体几何收益，不隐去正负任一方向。
- PID12484消失由正式Completed日志解释为正常结束；不重启评价。新run、完整评价及实际C0已由用户下载到D盘，双方postmesh/本地GT哈希与报告完全匹配，JSON/CSV一致；不由agent SSH连接。
- 训练、mesh与评价原件保全；阈值、评价域与方法不变。先归档，下一批定位/改动须讨论，不自动叠加模块或新训练。

## 最新本地分析（2026-10-23）

> **2026-10-26更新：** 用户已完成1219文件输入哈希比对，0missing/0mismatched。随后要求逐行确认基线来源；[完整源码审计](2026-10-26-selector-baseline-source-audit.md)确认d979c07直接父提交为C0，未继承旧Core；C0与官方88d6405的8个Python差异文件AST相同、4个native差异文件非注释token相同。4个既有生产文件加2个新模块逐行核对，1630受保护文件与C0逐字节相同。本轮18CPU测试通过/5native跳过；不是全7K轨迹等价证明。当前不再等待输入核对，不自动启动训练或归因数值随机性。

- 新归档位于 `D:/research_Space/output/AmbiSuR_second_run/ScanNetpp/Tool_Room/selector_v1_2026_10_23`；实际C0在相邻`baseline/c0-candidate-r2-30k/d6f15c88/seed_0/attempt_20260902T082615Z`。原本地original不是正式C0，不混用成绩。正确C0 train PSNR31.07406044→31.10275154（+0.0286911dB），仅为5个训练视角。
- 最终快照三个既有debug视角的A阴性候选池约129416/741704/164701，4096覆盖3.165%/0.552%/2.487%。固定预算在不同视角改变新增ALR剂量，给预算不足提供具体机制线索，但尚无因果结论；总掩码图不能分离extra。
- 实际C0到齐后复用原评价核心/相同GT样本离线诊断：z<4m平均Completeness31.7489→29.9042mm，z>=4m为84.1396→119.6272mm；整体退化主要集中上部。完整性差距88.03%在raw已存在；最坏1%点84.67%满足至少一相机GT first-hit 1cm可见容差。不据此裁剪正式评价或生成训练mask。
- 关键混杂：上部局部包围盒Gaussian中心数7000轮已为C0 29904/new14205，selector从7001启动；日志600轮首次点数差异。中心数不等于7K表面误差，但最终差距不能全归因于补选。相机/初始点云/邻居哈希一致；全部图像/深度/置信/对齐scale待核验，不先认定数值随机性或4096是根因。
- 独立分析在新归档`analysis/toolroom_pair_seed0_30k_diagnosis_v1`。原均值复算最大差4.62e-13mm；正式原件、方法/训练代码均未改。下一批建议只读比对C0已有manifest与新工作视图1219文件，不用GPU/GT、不重训baseline；完整命令与边界见结果报告末节，待用户讨论。

## 接手后的下一步

1. 在 D 盘检查分支、HEAD、clean状态，阅读当前候选规格与根目录三个进展记录。
2. 用户已同意提取新模型mesh并与基线比较。先沿用baseline的 `max_depth=5.0, voxel_size=0.005, sdf_trunc_scale=4.0, num_cluster=2`，以服务器原3d85faf和独立扩展提取iteration30000；已有baseline mesh保持只读。已收到提取后处理日志及三个mesh路径；固定适配协议的服务器8项真实CPU测试和首轮正式评价均已完成；当前转向原产物归档及待讨论的退化定位，不重训或重提取baseline。
3. baseline `D:\research_Space\output\AmbiSuR_original` 与数据集保持只读；不重训、不覆盖、不移动。GT只用于获准离线评价。
4. DTU沿用CD及双向距离，TnT沿用Precision／Recall／F1，Mip360沿用PSNR／SSIM／LPIPS；当前Tool Room属于ScanNet++，不能套用DTU/TnT专用评价域。`eval=False` 的训练视角PSNR不冒称新视角指标。
5. C 清理、旧 Utility 恢复及新方法扩展均另行讨论；当前优先获得实际重建比较结果。

## 以下为历史交接记录

保留旧目标、环境和阶段约定供追溯；冲突时以上方当前入口、AGENTS和已批准候选规格为准。

## 1. 项目目标

以 AmbiSuR 官方实现为 baseline，在 ScanNet++ 紧凑、高反射室内场景上研究：观测校准的歧义需求、外部先验与内部几何双可靠性、拒绝式仲裁、参数级梯度路由和可靠性驱动的 Gaussian 生命周期。

优先目标是会议论文；保留期刊扩展与硕士毕业论文兜底。当前主张限定为“改善高反射室内场景的整体表面重建”，不声称已经对人工标注反光区域做独立定量证明。

## 2. 代码来源

- 官方实现：`https://github.com/Fictionarry/AmbiSuR`
- 用户 fork：`https://github.com/Monkot19/noob_AmbiSuR`
- 新项目开始时必须在这里补写实际基线 commit：`BASELINE_COMMIT=<运行 git rev-parse HEAD 后填写>`
- `main` 只保存可复现 baseline；开发从 `research/core-routing` 开始。

## 3. 环境与算力

- 平台：AutoDL
- GPU：单张 RTX 4090 24GB
- PyTorch：2.8.0
- Python：3.12
- 系统：Ubuntu 22.04
- CUDA：12.8
- Core 峰值显存建议低于 22GB，训练时间不超过 baseline 约 2 倍。

## 4. 数据

- ScanNet++ Tool Room：`d415cc449b_Tool_Room`
- ScanNet++ Utility Room：`0a5c013435_Utility_Room`
- 本地数据根目录：`D:\dataset\ScanNet++\data\data`
- 本地 baseline 结果：`D:\research_Space\output\AmbiSuR_original`，只读
- AutoDL 数据根目录：首次配置时填写到实验 manifest，不得猜测或写死在代码中
- 后续泛化：AmbiSuR 官方 DTU 15 scenes、Tanks and Temples 6 scenes

Tool Room seed 0 是开发与 D0 标定场景；Utility Room 和其他 seeds 用于确认。论文必须披露 Tool Room 的开发用途。

## 5. 已确认的方法边界

Core 顺序：

1. 连续高端 SH 歧义量 A。
2. 有效视图数与方向离散度得到观测充分度 S。
3. 得到外部帮助需求 N。
4. 分别计算外部 `(T^P,V^P)` 与内部 `(T^G,V^G)`。
5. 计算可靠性、一致性和可靠性差。
6. 输出 Bypass、Consensus、Prior-led、Geometry-led、Abstain。
7. C2 开始执行粗粒度动作；C3 增加拒绝；C4 参数组路由；C5 冲突投影；C6 生命周期。

Supporting 仅在 Core 通过后实施：

- CoMe 式解耦曝光；
- 可靠前表面局部 Ray-Color；
- 可靠前表面局部 Ray-Normal；
- 生命周期状态驱动的逐 Gaussian 截断。

## 6. 实验门槛

- G0：关闭新功能时复现 baseline；无 NaN/Inf、显存增长和重复 backward。
- G1：N 的高几何误差 AUROC 大于 0.60，且比 A 或 1-S 中较优者至少提高 0.03。
- G2：两个反光场景平均几何改善约 3%，单场景系统退化不超过约 1%；PSNR/LPIPS 分别不恶化超过 0.3dB/0.01。
- G3：Supporting 每个模块最多进行默认配置、一次有依据调整、一次跨场景确认。
- G4：DTU/TnT 多数场景不得系统退化。

## 7. 本地—GitHub—AutoDL 闭环

本地 Agent 只在隔离分支/worktree 修改代码，完成单元测试、代码审查和短步检查后提交并推送。AutoDL checkout manifest 中记录的明确 tag/commit，运行短步 GPU smoke 或正式实验。实验输出连同 manifest、配置、日志、指标和环境快照下载回本地。分析脚本读取结果但不修改原始输出；新结论写入 `findings.md`，阶段状态写入 `progress.md`。

禁止行为：服务器临时改代码、只记录分支名不记录 commit、用脏工作树跑正式实验、结果出来后移动 tag、一次提交混入多个消融模块。
