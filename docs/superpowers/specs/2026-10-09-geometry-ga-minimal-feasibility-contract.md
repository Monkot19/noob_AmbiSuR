# Geometry G-A 最小可行性优先：合并审批合同

**状态：待审书面提案，不是执行confirmation。** 2026-10-09。用户选择缩小范围、拓扑干预后置；只原则认可数学方向和单案例检查。本文件一次交付精确数学修订、最小检查合同及阶段计划；审批它也不自动授权实验。保留[可行性复核](2026-10-09-geometry-ga-feasibility-review.md)与[原待批附件](2026-10-09-geometry-ga-execution-annex-proposal.md)作为历史对照，不静默改写已批准完整子规格/TDD。

**下一项交付：** 经本合同审阅、另获最小脚本编写/检查执行授权后，提交一个小案例的“独立导数参照+完整J成本”资格receipt。通过只能推进到**另写、另批固定拓扑科学预注册**；失败停在工程资格阶段。现在不建128例、不做拓扑干预、不读取真实数据/GT、不产生科学阴性或geometry release。Utility三seed合格但GT防火墙不变；五状态、生产路由、C1仍不授权。

## A. 精确数学修订及旧公式对照

### A1. 固定测量对象

唯一action仍是32个原始成员的共享米制translation `δ∈R³`，其余Gaussian参数固定；无batch cluster。1A的16例仅支持不足拒绝对照，不是有效观测病态几何验证。2A半径0.07m、32行、guard0.10m明确改变干预空间尺度；本检查不检验guard/far outcome。

源相机全场景贡献 `s_vr=Σ_{i∈C}sg(T_ivr α_ivr)`；component j为(source,target或same-view,pixel,family)。有效且 `w_j=s_source,pixel>0` 才入inventory；同一个像素在不同component出现时分别计权、计n。depth及无符号normal/dn均按生产reprojection合同；`τ_d=0.05,τ_n=τ_dn=0.10`。无效观测不作为零误差或风险最大值。

对 `B=P,H`，在δ=0冻结全部I_B、w、masks、分母及P/H；不能抽样、挑梯度或在δ变化后删除component。定义

\[
\epsilon=10^{-8},\quad W_B=\sum_{j\in I_B}w_j,\quad
R_{B,j}(\delta)=\sqrt{\frac{w_j}{W_B+\epsilon}}\frac{e_j(\delta)}{\tau_{family(j)}}.
\]

`W_B>10⁻⁴`且有限；n_B=|I_B|。虚拟δ时原component变为非有限、零normal、非正depth或出frame，整次测量unknown；不重分区/填零。重新遮挡不删除已冻结component，只记录mask变化。只保留渲染几何/投影/采样值的导数；weight及pre-action判定不求导。

**研究测量图的额外待批澄清：** 生产renderer返回的 `depth_normal=render_normal(camera,plane_depth)*alpha.detach()` 保持不动；该stop-gradient的局部反向不等于重新渲染前向值的完整导数。本研究adapter唯一提议是从同一次native输出的plane_depth/alpha重建 `N_geom(δ)=render_normal(camera,plane_depth(δ))*alpha(δ)`，不detach alpha；再按原reprojection/normal normalization组成R。对δ=0及全部差分扰动，重建乘积必须与生产depth_normal前向值按B2前向容差一致。P-action、H-virtual使用同一值函数/测量图；固定weight/mask仍stop-gradient，不能混用两种图。此项改变研究J的解释，需单独明确批准；不说明生产detach是bug，不修复r_g，不升级Evidence或影响生产训练。

令 `r=R_P(0),J=∂R_P/∂δ|₀,G=JᵀJ,c=Jᵀr`，J明确针对上述研究值函数的完整导数，不针对生产depth_normal的stop-gradient代理。G与现有证据A不同名。

| 项 | 原待批4A与原主公式的组合 | 本次唯一待批修订 |
|---|---|---|
| 目标 | `F=||R_P||²/(2n_P)` | `F=||R_P||²/2` |
| per-view目标/梯度 | `F_v=Σ_source=v R_j²/(2n_P)` | `F_v=Σ_source=v R_j²/2; g_v=∇δF_v(0)` |
| solve | `H=G/n_P+λI,b=c/n_P` | `H=G+λI,b=c` |
| 实际raw step | `−(G+n_PλI)⁻¹c` | `−(G+λI)⁻¹c` |
| conditioning | `a_min(H)/(a_max(H)+ε)` | 数据G的`a_min/(a_max+ε)`，不是H |

固定 `λ=10⁻³`，求解后仍

\[
\delta_{raw}=-\operatorname{solve}(G+\lambda I,c),\qquad
\delta_C=\delta_{raw}\min\left(1,\frac{0.01}{\|\delta_{raw}\|+\epsilon}\right).
\]

不重设阻尼/步长、不另加候选；R归一化一次后λ相对于同一加权目标固定。旧版对n的尺度效应以及秩亏 `diag(1,0.1,0)` 在n=256时被阻尼抬高条件比的手算证明，见前述复核§2。这不是从实验结果选公式。

### A2. 精确validity与分数

- `V_cluster,V_views`保持完整主规格身份/支持要求：至少8支持相机，P/H各≥4、各≥4合法有向edge，同一相机不能跨集合。`V_support`保持combined/P/H各自有限、非零support及原公式；拒绝对照不能进有效预测样本域。
- `V_action`要求P≥256component且三family均出现；r/J/G/c/H/raw step全部有限，`λ_min(H)>10⁻⁸`且精确solve成功。零raw step仍有效，但不能预设有benefit。H门只表示数值求解可用，**不证明信息充分**。
- `V_dir`仍要求≥4个有限且严格非零的g_v，`u_v=−g_v/(||g_v||+ε)`，`q_dir=||ΣZ_Cv u_v||/ΣZ_Cv`；不加梯度幅度阈值，不用1/n制造小梯度。
- `V_virt`仍要求H≥256component/三family，`L_H(0)>10⁻⁴`且两个RMS有限；`L_H(δ)=||R_H(δ)||₂`，`q_virt=clip(((L_H(0)−L_H(δ_C))/(L_H(0)+ε))/0.10,0,1)`。
- **新V_cond的明确改动：** 对同一float64 G用对称eigvalsh得a_min/a_max；两者有限且a_max>0。G本应半正定；仅允许 `−ε*a_max≤a_min<0` 的数值舍入区间，将其视为0计算q；小于 `−ε*a_max` 则unknown/数值资格失败。该guard复用ε，不是新科学门，也不将rank-deficient输入剔除。rank-deficient且a_max>0得到有效q_cond=0；全零G的condition未定义，V_cond=False。此新增数值规则须审批，不能假称与旧validity等价。
- 因而唯一新分数为 `q_cond=clip(max(a_min,0)/(a_max+ε)/0.05,0,1)`（只在上述V_cond内）。它不证明绝对信息强度、外部正确性或路由安全。`V_mag`仍要求有限raw norm，`q_mag=exp(−max(0,||δ_raw||/0.01−1))`。
- `V_Q`仍为上述各validity的AND；有效时 `Q_g=(q_dir*q_virt*q_support*q_cond*q_mag)^(1/5)`，否则null/unknown。不得把unknown写成0。完整科学阶段的授权tail仍 `V_Q & Q_g≥0.50`，本工程检查**不计算其PASS**。

### A3. 数字不变不等于阈值含义不变

| 保留的量 | 含义及变化披露 |
|---|---|
| λ=10⁻³ | 数字不变；由旧等效nλ变成归一化目标上的固定λ，实际更新通常更强，不是等价重排 |
| 0.01m cap、q_mag尺度 | 米制上限不变；raw step改变会改变q_mag及cap命中率 |
| q_cond的0.05 | 改为**数据曲率**特征值比达到0.05时饱和；不再是含阻尼H的条件比，旧分数不可直接比较 |
| q_virt的0.10与L_H门 | 相同RMS/相对改善含义不变，虚拟action改变会改变其值 |
| Q≥0.50及原benefit/safety/性能门 | 数字保留为以后科学阶段的预注册要求，不证明校准等价；本检查没有Y、AUROC或科学PASS |
| coverage/validity | condition对象及数值guard改变可能改变coverage；必须报告，不能删失败case保证覆盖 |

未来固定拓扑科学合同需另写、另批所有域/样本及以上门；不沿用旧128例PASS，也不根据本检查调科学阈值。完整3A的horizon/合成观测设计仍未批准。

## B. 单案例关键导数/成本合同（尚未获执行授权）

### B1. 一个固定工程fixture，不是预测suite

只用合成96行，row=12l+k，k=0..11,l=0..7；xyz=`(0.02(k−5.5),0.02(l−3.5),0.003((k mod3)−1))`m，normal=`normalize(0.15sin(πk/4),0.15cos(πl/4),1)`。anchor row41；按未扰动xyz距anchor、row tie选择0.07m内最近32行，固定其身份，再统一加0.002m*anchor normal；其余64行不变。重核半径及身份，失败停止，不换成员。这是导数fixture，不是完整科学surface/benefit构造。

沿用原待批附件§3.3的physical scales0.015、opacity0.8、activeSH0/maxSH3零padding、RGB0.6、knn_f0、normal为rotation第一列的确定性构造；其§3.3渲染开关固定（含ray_reg=-1、trunc_sigma=2、opt=None），**只配置研究fixture**。不得改生产collector/evidence no_grad或任何生产参数默认值。

8个相机`cam_00..07`，radius1m、az=45j°、elevation偶数+15°/奇数−15°，看原点；right/down/forward及near0.01/far10按旧附件§4。512² K=(fx=fy500,cx=cy255.5)；64²同FoV K=(62.5,62.5,31.5,31.5)，黑背景。两分辨率是同一病例，不能换scene。fixture_id=`ga_backend_minimal_v1`，cluster_key为32升序row逗号串；按UTF-8 `fixture_id|cluster_key|camera_name` SHA排序、交替P/H，P先，支持/edge失败不补相机。不加入DA3或真实场景参考。

inventory按source name、target name（same-view排序在cross-view后）、pixel行列、depth/mv-normal/dn排序；所有合法正权标量都保留，包含P内部12有向edges/H内部12edges及各源same-view dn；不能跨P/H。512²必须满足A2的P/H库存、视角/edge门，否则检查输入不合格，不以减少component补救。

64²三个固定probe：P哈希序第1相机为source、第2为target，像素(row31,col31)的depth/mv-normal，以及该source同像素dn；对象都是加权R_j，不是raw e、不随结果选择像素。各probe不合法、处于不稳定分段边界，或独立差分向量范数≤ε时，返回`FIXTURE_REFERENCE_UNINFORMATIVE`并停止；不能据此说后端无导数或Q无效。fixture定义、完整生成manifest及probe IDs在任何渲染前由唯一脚本确定并SHA冻结；只记录，不筛选或先试跑再换。

### B2. 独立参照、差分与容差

1. **闭式装配参照**：toy三个等权component，`R_toy=( (δx−0.002)/0.05, (1−cos(π/4+δy/0.10))/0.10, (δx+δz−0.001)/0.05 )/sqrt(3+ε)`。其δ=0的J各行为`(20,0,0),(0,100sin(π/4),0),(20,0,20)`再同除sqrt(3+ε)。独立float64闭式参照r/J、G/c与相同阻尼solve，elementwise `|x−x_ref|≤10⁻¹⁰+10⁻⁸|x_ref|`；不调用同一autograd结果当参照。toy只查数学装配，3行不冒充≥256科学validity。
2. **64²权重/前向参照**：全场景indicator RGB（成员1、其他0）核验Σ_C(Tα)；CPU从fixture3D参数和K自行算投影covariance/footprint、depth顺序、alpha与透射乘积，不读native中间贡献。现有源码字面常数：camera-z≤0.2剔除；projected x/z,y/z按±1.3*tanFoV截；screen covariance对角加0.3 pixel²；16×16 tiles；power>0跳过，power<−2乘100（trunc_sigma=2）；alpha=min(0.99,opacity*exp(power))，alpha<1/255跳过；T初值1，若T*(1−alpha)<10⁻⁴则结束且该Gaussian不计入，否则计alpha*T再更新T。footprint/radius与tile边界按固定forward.cu/auxiliary.h的投影方程独立计算；正式脚本交付将所有上述字面常数及source hash列入manifest，缺项不得执行。前向elementwise `atol=5×10⁻⁵,rtol=5×10⁻⁴`；不能用alpha-only/cluster-only/out_observe替代。
3. **64²native导数参照**：共享translation三轴，独立forward重算中心差分 `D_h,k=[R_j(+h e_k)−R_j(−h e_k)]/(2h)`，三档h固定 `10⁻³,5×10⁻⁴,2.5×10⁻⁴`m；两侧用δ=0的冻结w/W/masks。compare的是R对δ的3-vector，normal normalization、reprojection及sampling也必须随δ变化，不能detach它们。按A1重建可微N_geom：alpha随扰动重算且参与autograd，只有贡献weight冻结；先核验重建乘积与native depth_normal的前向一致性。原生depth_normal的局部stop-gradient反向不是独立正确性oracle，不能把它与完整差分的预期差异误报为CUDA导数错误，也不能以放宽容差掩盖差异。
4. 对最后两档D，**各自**与autograd、以及两档互相比较，均要求：elementwise `atol=5×10⁻³,rtol=2×10⁻²` **且** `||x−x_ref||₂/max(||x_ref||₂,ε)≤0.02`；差分互比以最小h为ref，autograd以对应差分为ref。另要求ref非零方向的autograd向量非零、点积>0，禁止小权重让绝对atol吞掉全零梯度。不能选最好h/平均差分/放宽容差。此native核验容差替代旧未批附件gradient通用容差，不改变科学门。
5. 如sorting/culling/abs/clamp分段、visibility或合法component身份变化导致参照不稳定，报告具体原因并停止。差分**仅作核验**，绝不作为action/J，不能用其输出产生候选分数。

### B3. 完整三列J的唯一计算方式

**冻结为exact reverse-mode、32-row batched VJP分块**，不是算法候选集。对固定R∈Rⁿ和δ∈R³，第b块构造精确基向量矩阵E_b∈R^(m×n)，m=min(32,n−b)，其每行是对应component的one-hot；使用Torch `autograd.grad(R,δ,grad_outputs=E_b,is_grads_batched=True,retain_graph=未到末块,create_graph=False,allow_unused=False)`。返回m×3正是这些行的J；按inventory顺序拼成n×3。最后不足32行不padding。完整J形成后，float64累计G/c并solve，不用随机方向、低秩投影或gradient-of-summed-loss替代。

对64²固定probe，普通单scalar reverse仅作独立分块一致性核验（同float64 toy/native分层容差）；**不得在成本阶段回退逐行unbatched、另换JVP、差分或新CUDA路线**。实现使用native float32 forward/backward，残差/J累加、G/c/eigen/solve为float64，明确这是混合精度，不宣称全链float64。dense E_b逐块分配、使用、释放，不同时保留全部E_b；分配成本和完整J都进入资源/成本预算。

现有native wrapper只证明有custom backward，不证明上述batched VJP能力。脚本固定开启Torch的`_debug_only_display_vmap_fallback_warnings(True)`并捕获batching fallback警告；开关缺失也停止。若报不支持batched tensor/vmap、materialization失败或框架发出批处理fallback警告，立即`BACKEND_STRATEGY_UNSUPPORTED`停止；导数错误则按DERIVATIVE_MISMATCH停止。不抑制警告、不改block size或自动选择另一算法。该结论只对本冻结实现路线有效，不是所有可能后端均不可行，更不是科学阴性。native能力风险正是最小检查要先回答的问题，不应提前建设完整fixture/干预框架。

### B4. 精确计时、显存和失败边界

| 门 | 冻结定义 |
|---|---|
| 总时限15分钟 | 单次授权attempt的外部单调wall clock：watchdog启动worker之前t₀，到所有小receipt/manifest持久化完成t₁，`t₁−t₀≤900s`；包括冷import、CUDA初始化、fixture/SHA、CPU参照、所有分辨率渲染、差分、warmup、计时、最终核验和写盘。预先审批/下载依赖不在本检查做。超时终止worker并记录未完成，不延时 |
| 导数块30秒 | 指**一次完整512² P-inventory的三列J计算阶段**，不是每32行30秒。CUDA synchronize→wall起表，重新渲染全场景8view并冻结mask/weight/完整P/H inventory，重建N_geom并组R、逐块分配/使用/释放E_b、全部native backward、拼J、累计G/c、solve→synchronize→wall停表，含CPU调度/传输/allocation。不得从已有图/已生成J开始，不能只计kernel或单小chunk |
| 次数 | 一次完整同条件warmup+3次fresh-graph计时；warmup也≤30s、计入900s及显存。每次从相同固定fixture重新开始，不缓存图、weight/inventory/J；逐次审门，不合格就不执行后续重复；计时取3次最慢，所有n/行ID及SHA必须一致 |
| 显存6GiB | 6×2³⁰=6442450944字节。GPU须空闲、仅worker使用；CUDA初始化后、fixture分配前仅reset一次峰值（不清cache或改变allocator）；全attempt包括warmup的Torch peak allocated与reserved均≤6GiB，另用100ms间隔nvidia-smi whole-device used监测（含context，报告采样峰值，不能称连续精确峰值），三项均不得超6GiB。监测不可用/有其他GPU任务则输入不合格，不能忽略该门 |
| 监控执行 | 外部watchdog每≤100ms检查900s、当前阶段30s截止及显存；到期立刻要求终止worker，最多5秒停机宽限不算成功预算、不可继续计算，停不下来则记录强制终止失败；这不是扩大预算。复用已有操作级receipt/进程终止模式，不新建研究调度系统 |
| 其他资源 | worker RSS≤4GiB，输出总≤20MiB；无预测suite、训练checkpoint/渲染画廊、八步optimizer或拓扑事件 |
| 停止/输出 | 任一身份、参考信息、前向/导数、finite、算法支持或预算门失败，立刻停止，保留一个小FAIL receipt（完成到哪里、原因/计时/n/peak/source/fixture hashes）。无自动重试、清理、换算法/fixture/阈值或扩大预算 |

每次计时前fixture不变，测量后不能利用残差或耗时删行。Q的完整支持/virtual评分成本**没有**被这个只到solve的30秒门覆盖；receipt必须这样写，不能宣称全Q或完整paired horizon合格。正常完成小receipt只是工程报告，不产生任何release文件。

### B5. 工程结论枚举

- `BACKEND_CHECK_QUALIFIED`：所有冻结检查与资源门完成；仅说明**该小fixture、该batched-VJP实现路线的关键导数与完整J→solve计算可行**。不能外推96→1681+行/8→12views成本，也不证明Q预测能力、horizon benefit或生产安全。
- `BACKEND_CHECK_NOT_QUALIFIED`：失败；附固定reason（如FIXTURE_REFERENCE_UNINFORMATIVE、BACKEND_STRATEGY_UNSUPPORTED、DERIVATIVE_MISMATCH、BUDGET_EXCEEDED）。未完成的检查不得记PASS。没有科学阴性、候选淘汰的GT依据或Geometry终止release。

## C. 缩小后的阶段计划与审批点（不是执行授权）

| 阶段/下一交付 | 通过后能推进什么 | 失败/未批时停在哪里 |
|---|---|---|
| MF-0：本合并书面合同+新commit | 用户一次审阅精确数学、数值validity、检查策略/容差/资源及授权地图；通过仅允许随后申请最小脚本阶段 | 继续改书面，不写方法代码/执行；不把规格问题叫科学失败 |
| MF-1：另获最小脚本编写授权后，单案例脚本及预执行fixture/source manifest | 只做本合同所需最少toy/身份/监控测试，复用现有renderer/collector前向语义；脚本与字面常数冻结，再请求一次backend执行授权，不重开Tool Room探针或跑无关全套 | 脚本/参照/策略不成立则停工程阶段，无自动新backend开发 |
| MF-2：另获执行授权后，一次backend资格receipt | 仅可开始撰写固定拓扑科学预注册；真实规模资格、完整Q成本和科学目标均留在该后续合同审批中 | 任一门失败，停止并报告；需要用户决定归档、修订合同或是否再投入；不自动重试/释放GT |
| MF-3：另写、另批固定拓扑科学合同 | 明确真实规模fixture、唯一horizon/Y、coverage与原科学门的适用域，再单独申请实现/执行 | 未冻结不做paired干预；不沿用128例或旧PASS，不看Utility GT |

MF为本次最小可行性阶段，**不是**生产C0–C6。后置拓扑是本轮范围选择，不等于批准96例科学阶段：若未来采用6family×16=96预测+16支持拒绝对照，总112及6family bootstrap，必须在MF-3独立预注册审批。候选仍唯一，已有科学门数字不因工程检查结果重选。旧clone/split映射测试保留为回归，不声称已覆盖真实拓扑后的outcome/Adam/渲染耦合。

**本次审批请求合并为一个包：** A唯一精确数学/validity修订、B单案例核验策略与预算、C分阶段停止/授权地图。审批仅批准设计；最小脚本编写、GPU执行、固定拓扑科学预注册及以后实验各自遵循上表授权。工程不合格、缩小或归档均不构成科学否定，不解除Utility GT防火墙；geometry release、GT评价、五状态、生产路由、C1仍需独立合法合同和授权。
