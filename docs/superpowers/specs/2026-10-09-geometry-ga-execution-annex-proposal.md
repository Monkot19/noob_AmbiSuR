# Geometry G-A Gate 0：具体执行附件与审批包

**状态：待审提案，未生效；不是 execution confirmation。** 日期：2026-10-09。

主规格：`2026-09-30-geometry-stage-ga-exact-subspecification.md`。四项原规则/理由/备选：`2026-10-08-geometry-ga-gate0-amendment-proposal.md`。本文补齐该提案第3项，不改写主规格或已阻塞的TDD计划。凡本文新给出的值都是需要人类批准的建议，不是实现者可自行采用的默认值。

## 1. 本轮需要决定的四项

| 项目 | 原规则与阻塞 | 本轮推荐 | 实验含义及需要批准的内容 |
|---|---|---|---|
| 1 | 12相机相隔30°；20° sector最多1相机；却要求8支持视角和每family覆盖≥0.60 | **1A**：128登记例中，16个ill-conditioned为必须拒绝的合同对照；其余112为预测库存 | 批准预测总体/coverage分母112、7family bootstrap、最多1792步；不是偷偷删除失败例。1B另设计相机布局，不推荐 |
| 2 | 单层0.02m网格在0.05m半径内只有21点，无法32-row | **2A**：仅建议R_C=0.07m；32行、网格0.02m、guard0.10m、action cap0.01m不变 | 批准干预空间尺度扩大；2B改密度至0.015m另有信息密度影响，不推荐 |
| 3 | surface/anchor/corruption、优化器、8步loss、step4事件未唯一规定 | **3A**：批准本文§§3–10的具体研究合同 | 所有细项共同定义tau_C(8)，不是无关工程细节；可逐节要求修改，不接受“沿用默认”。§11列出对原文的解释/补充影响 |
| 4 | detached生产collector不能直接提供J；权重分母/g_v/dtype未闭合 | **4A**：沿用前提案的partition-global归一化、明确g_v、research-only可微adapter及native float32 renderer→float64测量 | 批准唯一测量图和dtype例外；正确导数及贡献权重可获得性仍须以后资格验证。不可获得就INCONCLUSIVE，不改CUDA/用有限差分替代 |

四项通过只允许随后同步正式addendum和修订TDD计划交审；**不直接授权代码、backend smoke、128登记例执行、Tool Room干预、Geometry终止确认、Utility GT或C1**。Utility三seed已经合格不改变此边界。

## 2. 保持冻结、不请求重新选择的量

一个cluster/branch pair，无batching；32原始行；guard距实际成员≤0.10m；共享xyz translation，原Gauss–Newton解法和lambda=1e-3、delta_max=0.01m；原五个q组件、等权几何均值、Q≥0.50；H=8；原Y_C、0.001m/5% benefit、八项safety/spillover门；epsilon=1e-8，tau_d=0.05、tau_n=tau_dn=0.10。Evidence保持v4，r_g/1-r_g仅归档telemetry且不作候选。

本文只研究固定合成训练系统下的cluster级差异；不称逐Gaussian tau_i、不证明生产更新安全，不直接授权五状态路由。预测门、阈值和公式不能在执行后重新选。

## 3. 坐标、网格、三角面及初始化的建议值（3A-geometry）

### 3.1 三种surface

单位米，右手世界坐标；局部canonical中心为(0,0,0)。所有integer区间含两端。先生成未扰动模型和独立参考三角面，再确定成员，然后扰动；禁止根据分支outcome重建成员。

| surface | 行生成规则（外循环在前） | anchor | 三角化 |
|---|---|---|---|
| plane | k,l=-20..20；x=(0.02k,0.02l,0)，1681行 | k=l=0，row840 | 每格角点a=(k,l),b=(k+1,l),c=(k+1,l+1),d=(k,l+1)；(a,b,c),(a,c,d) |
| cylinder | k=-39..39,l=-20..20；theta=0.08k，x=(0.25cos(theta),0.02l,0.25sin(theta))，3239行 | k=l=0，row1619 | 相邻(k,l)矩形沿上述对角线三角化；角端不wrap、不加cap |
| corner | 先plane的1681行；再k,l=-20..20,k≠0，x=(0,0.02l,0.02k)，1640行，共3321 | 第一片k=10,l=0，row1250 | 第二片k=0共边复用第一片row；其余同格对角线规则；两片法向符号不影响无符号距离 |

平面normal=(0,0,1)，圆柱normal=(cos(theta),0,sin(theta))，corner第一片normal=(0,0,1)、第二片=(1,0,0)。共边归第一片。以上normal仅用于构造初始模型旋转/扰动；Q/action接口只接初始化后的模型，不接参考surface或解析normal。

圆柱沿曲面弧长间距恰为0.02m，欧氏弦长约0.019995m；不宣称其是平面Cartesian lattice。所有原始row按表中生成顺序编号，不因扰动重新排序。三角面与Gaussian是两个独立容器，禁止共享可写storage。

### 3.2 成员/region及variant

- 在尚未扰动的canonical模型上按主规格的distance/row tie规则选32行；再扰动这些行。在扰动后pre-action状态重核R_C≤0.07m，不替换成员。此“先选后扰动”的身份固定顺序需批准，避免扰动把干预对象偷偷换掉。
- guard/far使用扰动后pre-action实际成员位置和主规格0.10m距离规则，随后冻结。二者必须均非空；空区域是输入合同不可计算，INCONCLUSIVE，不能当零spillover。
- variant v=0..15，整个场景/所有相机/参考surface/扰动方向同乘绕世界z轴R_z(22.5v°)。行号不变；无额外随机几何扰动。这16个旋转是约定的算法压力变体，不是16个独立场景；bootstrap不解释为跨场景置信区间。
- fixture_id为ASCII `ga_<family>_v<两位十进制v>`；cluster_key为32个升序十进制row以英文逗号连接，无空格。哈希键严格UTF-8 `fixture_id|cluster_key|camera_name`。

### 3.3 Gaussian tensors

每行physical scale=(0.015,0.015,0.015)，内部log scale；opacity=0.8，内部logit；**active SH degree=0，存储max degree=3**；f_dc=(RGB-0.5)/C0，C0=0.28209479177387814，f_rest形状[N,15,3]且全零/永久不训练；knn_f=zeros[N,6]。RGB=(0.6,0.6,0.6)，不做曝光、纹理拟合或appearance模型。

此storage例外明确用于复用现有renderer：生产`compute_weighted_sh_norm`会无条件reshape为[N,15,3]，即使opt=None；N×0×3会报错。零padding保持degree-0有效颜色，不启用高阶SH、不把该辅助值输入Q，不修改生产方法。若不接受此显式兼容布局，不能通过临时改生产函数继续；须退回该条。

render调用固定app_model=None、opt=None、return_plane=True、return_depth_normal=True、ray_reg=-1、scaling_modifier=1；pipe.compute_cov3D_python=False、convert_SHs_python=False、debug=False；pc.use_app=False、disable_trunc=False、trunc_sigma=2.0。上述开关只配置研究harness，不改生产默认值或实现。world_view_transform为§4 world-to-camera的transpose，projection采用同一K/near/far的生产camera构造，禁止另换坐标约定。

等尺度的最短轴存在tie；明确使用生产`get_scaling.min`返回的第0轴。旋转矩阵第一列固定为上述局部normal n；取t为世界y在n正交平面的单位投影，第三列n×t；矩阵列(n,t,n×t)。四元数用该旋转矩阵的单位四元数，标量优先且首个非零分量为正，旋转后再规范符号。不得通过改成各向异性尺度来消除tie。

## 4. 相机、内参和view inventory（3A-cameras）

12个PINHOLE，512×512，fx=fy=500，cx=cy=255.5；near=0.01m、far=10m，黑背景(0,0,0)。camera_name=`cam_00`..`cam_11`。j=0..11，az=30j°，elevation=+15°(j偶数)/−15°(j奇数)，中心c=(cos(e)cos(az),cos(e)sin(az),sin(e))，距原点1m。

camera local +z指向原点f=−c/||c||；right=normalize(f×(0,0,1))，down=f×right。world-to-camera前三行依次right/down/f，平移=−R c；像素u向右、v向下。若cross归一化失败返回输入错误，不换up轴。variant共同旋转相机中心及basis。

Directed edge候选为所有有序不同camera对，共132；按source j再target j排序。使用生产reprojection的边界、正深度、normal非零、有限值、foreground occlusion projected_depth≤1.05*target_depth语义；same-view depth-normal另要求alpha≥0.5。不用最近邻数默认值、不引入DA3。

每源像素的cluster贡献s_vr=sum_{i∈C}sg(T_ivr*alpha_ivr)，按全场景原深度排序，不是只渲染cluster的alpha。mask对源像素采用至少一个有效pair的并集，Z_Cv=sum s_vr*m_vr，支持Z>1e-4。edge有效要求它的cluster加权有效支持>1e-4。SHA排序保留最多16，交替P/H，主规格≥8、各≥4及各≥4内部edge不变；dn仅用分区自己的源相机。

不假定现有CUDA已公开每像素s_vr；正确贡献权重无法在不改生产/CUDA的边界内取得，backend资格就是INCONCLUSIVE。禁止用out_observe、alpha-only或cluster-alone render替代T*alpha。

## 5. 唯一128登记库存及实际扰动（3A-families）

family顺序固定为下表；每family16variant，surface固定，不随结果挑选。n_a为canonical anchor的模型normal。除表明的额外保护层外，无额外Gaussian。成员构造和region顺序见§3.2。

| family | surface | 初始扰动及观测算子 | step4事件 |
|---|---|---|---|
| coherent-offset | plane | 32成员xyz+=0.015*n_a；普通当前模型渲染 | 无 |
| small-correctable | cylinder | 32成员xyz+=0.002*n_a；普通当前模型渲染 | 无 |
| self-consistent-wrong | plane | 32成员xyz+=0.015*n_a；使用下面的同移context算子 | 无 |
| contradictory-views | plane | 32成员xyz+=0.015*n_a；P内第b个源相机的projected target-camera z加(−1)^b*0.015m，然后按普通公式形成relative depth error；target采样深度不加偏置 | 无 |
| ill-conditioned | plane | xyz不扰动；只保留相对旋转后方位[−10°,10°]相机，含端点 | 无；16例均为validity拒绝对照 |
| occlusion-harm | cylinder | 32成员xyz+=0.015*n_a；P观测使用下面的foreground corruption | 无 |
| guard-spillover | plane | 32成员xyz+=0.015*n_a；附加保护平面z=−0.03m（同41×41grid，第二surface row1681..3361），颜色(0.2,0.2,0.2)、其余初始化相同；参考surface包括两层 | 无 |
| topology-lineage | corner | 32成员xyz+=0.015*n_a；普通渲染；v/4向下取整依次survivor、clone、split、prune-pressure | §8 |

普通算子：对delta虚拟变化重新渲染全场景的源和目标depth/primitive/depth-normal，只有32成员共享平移，其余参数固定。P/H无交叉edge。

凡依赖P的corruption（contradictory/occlusion）均采用同一次顺序：ordinary当前模型渲染→ordinary support/edge及SHA次序→冻结P/H→corruption→在既定P/H内重算最终support/edge/residual inventory。若最终不满足原支持门则invalid，不重新分区/补相机。contradictory的b从0起按冻结P的SHA顺序计数；偏置后的projected z同时用于projection的分母与像素坐标、正深度、foreground occlusion以及relative depth error，不能仅给loss添加一个常数。H无此偏置。最终inventory冻结后delta只按同一算子重算值。

**同移context算子具体定义：** source为当前普通模型；target由独立只读context模型渲染，其所有原始xyz均从未扰动模型整体平移0.015*n_a得到；虚拟delta同时加到source的32行和context的所有行。source same-view dn仍普通计算。context不进horizon、reference或branch outcome；它是显式合成“内部观测与错误一起移动”的机制，不能使用解析surface/GT depth代替。并不预设其Q、validity或Y标签；若它导致该预测family覆盖不足，依原门判负，不能换定义。

**foreground corruption具体定义：** 仅在已经分好的P使用；目标/源plane-depth像素在u,v∈[224,287]闭区间且原depth>0时，替换为max(0.01,depth−0.03m)。同一patch内primitive和depth-normal各绕相机+z旋转30°，旋转后转回world frame再做符号不变normal比较。P/H划分和原始support先用普通渲染冻结，再对corrupted maps重算并冻结最终residual inventory；P/H不再重选。delta时只重算相同算子，patch不移动。H不腐化。patch是合成sensor-corruption，非人工真实场景mask。

上述0.015m为原contradictory family给定的米制扰动，明确落到**projected depth在relative归一化之前**；不是直接给无量纲error加0.015。normal仍由普通模型产生，不能把预期反方向写入score。

对guard层：成员只从主平面候选域选；保护层仅为全场景非成员，参与guard/far及渲染/优化。此专门排除需作为fixture身份设计批准，不推广为按opacity/GT过滤的cluster规则。其他families成员候选为全部原始模型行。

主规格要求的“beneficial/high Q”与负family“不授权”是**待检验门**，不是fixture生成时的赋值。不得依据family硬编码beneficial_safe或去调整扰动使门通过；实测与预期冲突就保留实测并按冻结门停止。

## 6. Residual、权重、梯度和dtype（4A）

depth e=abs(z_projected−d_target)/(z_projected+d_target+1e-8)；normal及dn e=1−clamp(abs(dot(unit normals)),0,1)，严格先abs再clamp，与生产符号不变语义一致。插值bilinear/align_corners=True/zeros padding，u,v归一化用width−1,height−1。world-frame normal先按生产camera-to-world约定转换；freeze inventory在delta=0且corruption已应用后。

component为(source camera,target camera或same-view,pixel行列,family)的一个标量；顺序source、target（dn标为12）、pixel行、pixel列、depth/normal/dn。每个有效component w_j=s_source,pixel，无效不入vector；有效但w=0不计component。所有有效正权component全部保留，无裁剪/抽样/挑高梯度pixel。

对B=P/H：W_B=sum_{j∈I_B}w_j，严格W_B>1e-4且有限；R_B,j=sqrt(w_j/(W_B+epsilon))*e_j/tau_family。P中的n=|I_P|同时用于主规格H/b和F_v=sum_{source=v}R_P,j²/(2n)，g_v=grad_delta F_v(0)。H：L_H=sqrt(sum w_j*(e_j/tau)²/(W_H+epsilon))，不再除component数。三family/256component、g_v≥4个有限非零、L_H(0)>1e-4等原validity门不变。

pre-action mask、weight、inventory、component normalization stop-gradient；source/target投影和depth/normal值保留gradient。虚拟delta若原component变成非有限、normal为零、depth非正或投影出frame，则unknown，不删component/填零。重新遮挡判定不删既定component；仅按原冻结mask评价，另记录变化。absence与真实数值0不得混淆。

renderer native float32 fwd/bwd；输出cast float64后残差/J/H/b/solve/Q用float64。J是该混合精度autograd图的导数，不宣称全链float64。abs/clamp使用Torch当前算子导数（abs在0导数0）；不另加平滑项。所有production collector/evidence no_grad保持不动。

exact Jacobian可按component行分块求reverse-mode autograd，每块32行、最后一块不足32不padding；保留完整J或精确累加JtJ/Jtr，结果须与未分块手算toy相同容差。不得用finite differences、低秩随机投影或拟合J代替。正确非零depth/normal gradient和固定贡献权重是以后backend资格门；缺能力即INCONCLUSIVE，不开新CUDA项目。

**数值比对建议值（也需批准）：** float64 hand-toy残差/J/H/b/solve/score采用elementwise atol=1e-10、rtol=1e-8；native renderer forward/gradient parity测试采用atol=5e-5、rtol=5e-4，另必须验证预期非零项不被detach及方向正确。bool masks、行号/partition/mapping、component数量及直接写入隔离采用exact，不用浮点容差吞掉身份错误。以上仅为以后资格测试的比较容差，不加到benefit/safety/Q任何门上。

## 7. 八步horizon的建议目标与optimizer（3A-horizon）

### 7.1 冻结目标

参考RGB由独立三角面ray-cast一次产生：像素中心(u,v)按K反投影，nearest positive triangle intersection、two-sided、无光照，surface颜色见§3/5，黑背景；禁止调用Gaussian renderer生成reference RGB。mesh depth/normal仅留在reference/outcome对象，不入Q/action。参考RGB是horizon合成训练观测，不是Utility/Tool Room GT。

每步loss=全512×512×3 linear-RGB的mean absolute error，权重1.0；不含SSIM、DA3、几何depth/normal、Ray-Color、ALR、exposure或正则项。**只有全场景xyz有梯度**，包括成员与非成员；其余参数fixed/grad=None。全场景耦合和非成员间接移动是被测spillover，不禁止其发生。

这是“几何translation后继续8步RGB-only xyz优化”的具体tau，不是生产多参数长训练tau；若这个狭窄科学含义不合意应在审批时拒绝，不能在实施时偷偷启用其他loss或groups。

### 7.2 Adam与执行顺序

7个生产命名groups均保留以复用topology tensor/state迁移：xyz lr=1e-4m/step，其他knn_f/f_dc/f_rest/opacity/scaling/rotation lr=0且grad=None。Adam betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad=False、foreach=False、fused=False、maximize=False；无scheduler/gradient clipping/AMP/accumulation。

source所有optimizer state开始为空（生产Adam lazy state）；xyz在第1次step初始化step/exp_avg/exp_avg_sq，初始moments零；其余不制造step counter。branch restoration深拷贝，模型/RNG/optimizer/source/reference不共享可写storage。

每步：zero_grad(set_to_none=True)→全场景RGB render→loss→一次backward→采集branch-private xyz gradient统计→一次Adam step。第4步step后按§8唯一一次topology事件，再zero_grad；第8步step完成后再测outcome/safety。不因末步省略optimizer，不提前保存当作H=8。

held-out reference RGB L1在horizon结束用H全部相机取等权全图均值；不择best view；H不参与horizon梯度。reference mesh只在outcome边界读取，所有八项safety沿主规格原数值。

## 8. Topology四子类的具体事件（3A-topology）

仅topology-lineage family在step4后发生事件；v=0..3 survivor，4..7 clone，8..11 split，12..15 prune-pressure；每种四旋转variant。其他family无topology。该显式子类是原文已规定的variant旋转以外的唯一类别差异。

- 每branch在steps1..4收集各行xyz gradient的L2 norm，G_i=四次的算术均值；非成员的事件候选统计置0。该候选限制只属研究harness的注册topology政策，不进入生产路由。
- survivor：调用生产`prune_points`全False mask，return_topology_change=True，identity应成立。
- clone：调用生产`densify_and_clone`，grads形状[N,1]值G_i、threshold=1e-6、scene_extent=1.0、percent_dense=0.02。max_all_points=4096。生产scale门/gradient门照常，0候选合法，不强造clone。
- split：调用生产`densify_and_split`，grads=G_i、threshold=1e-6；grads_abs=0、abs_threshold=1e30、max_radii2D=0、abs_split_radii2D_threshold=1e30、max_abs_split_points=0；scene_extent=1.0、percent_dense=0.01、N=2、max_all_points=4096。原生产append/prune及compose mapping照常，不自己重造children。
- prune-pressure：对初始32成员的当前survivor，在canonical未旋转坐标下的z>0.012m时mask=True；其他False。调用生产`prune_points`；同一阈值在两branch应用。该坐标阈值不读reference距离或Y，不强制treated才删。控制或两支都删合法，实际disappearance计入原safety门。
- clone/split随机draw直接使用生产torch.normal，不额外拒绝采样、排序或改scale；child顺序/parent mapping按生产repeat规则。事件不改变Adam本来lazy/迁移语义，不补global iteration counter。

每个branch恢复同一初始RNG；在事件前重设专用事件seed（§9），确保共享draw stream，候选数可因合法分支梯度不同而不同；记录实际draw数和mapping。**不要求topology后RNG结束态/row数相等**；要求同一机制、同一初始stream及可解释的消费差异。任何人为按treated身份选mask、mapping无父系或非候选参数直接写入是INCONCLUSIVE。

## 9. 随机数、camera replay与canonical记录（3A-replay）

- seed基数20261009；family序号f=0..7，variant v=0..15；pair_seed=20261009+100*f+v，event_seed=pair_seed+10000。Python、NumPy PCG64、Torch CPU及所有CUDA generator分别以pair_seed初始化并保存状态；事件只重设Torch CPU/CUDA至event_seed。不使用Python hash派生seed。
- 八步相机为proposal按SHA排序后的P[t mod |P|]，t=0..7；H从不训练。branch必须相同schedule，缺P或V_Q=False不执行action/horizon，记录invalid，不能换camera。黑背景，无随机augmentation、曝光或其他采样。
- 关闭TF32/cudnn benchmark；同一GPU/runtime、sequential branch，原生renderer非确定原子操作不冒充bitwise确定结果。无action的双restore qualification需验证固定容差重复性；若观察到backend噪声足以跨原benefit/safety门，停止INCONCLUSIVE，不调宽门或多跑挑一对。
- 该no-action重复性资格仅在以后获单独许可的backend toy上做一次固定双restore，不生成128套件outcome。两次均H=8，相机/RNG相同；要求row/mapping/step和有限性exact，xyz max绝对差≤1e-6m、全图RGB max绝对差≤1e-5、RGB L1差≤1e-5、祖先error/Y max绝对差≤1e-6m；任一超限为backend INCONCLUSIVE。它不证明所有future paired门在数值噪声下都稳定，也不授权增加实测门容差。
- manifest按family序、variant升序；execution前的fixture manifest包含生成式版本、family/surface、所有原始model tensor bytes SHA、参考surface/RGB SHA、camera/K/R/t SHA、32-row key、guard/far、corruption算子、全部seed、camera schedule生成规则、topology政策、主规格/修订/本文SHA。canonical JSON UTF-8、sort_keys=True、compact separators、末尾LF、禁止NaN，浮点tensor另存明确dtype/shape的bytes。fixture生成及内容SHA在以后单独授权的confirmation阶段产生，本文**不伪造尚不存在的fixture digest**。
- P/H、实际8camera schedule、edge/component inventory及其SHA在计时的pre-action measurement阶段生成；写入parent manifest SHA绑定的pair measurement receipt，再执行Q/action。它们不冒称是提前已知的fixture SHA，也不在confirmation前偷偷跑residual/J。分区由已冻结算法唯一决定，非法/不足不换分区。branch outcome记录引用同一measurement receipt，禁止执行后反向改mask/partition。

## 10. 指标、停止与预算（1A及原门）

登记128=112预测+16invalidity controls；controls应全部V_Q=False/Q=null/unauthorized，不执行未知action、不伪造Y/tau。预测invalid仍留112和family16分母，不因结果删除。原positive门保留，但“每family coverage”仅指7预测family；ill-conditioned改为16/16拒绝且0授权。bootstrap仅7预测family，各抽16，2000次PCG64 seed20260930、percentile2.5/97.5；无refit/no redraw。

预测valid域至少两类且所有既定指标可计算；AUROC≥0.80且lower>0.70；AUPRC≥prevalence+0.20；Q≥0.50 tail precision≥0.90且lower≥0.80、recall≥0.50、unsafe≤0.05且upper≤0.10；overall coverage≥0.80、各预测family≥0.60；其余原family授权/禁止授权门全部保留。全体无授权为valid-negative，不以undefined precision bootstrap制造vacuous PASS；其余classless/nonfinite replicate仍INCONCLUSIVE。

最多112预测pairs×2×8=1792步；一个pair sequential。每fixture/运行时Gaussian≤4096、512² render、retained view≤16、peak allocated≤12GiB、每pair≤30s、完整登记套件≤60min（除一次import/compile）、输出≤5GiB。pair计时从support/corruption/residual/action/J开始，包含两branch及outcome，不把昂贵Q计算藏在预处理。controls检查时间计入总时长。

inputs损坏/错误模型identity/源写入/无法拿到权重或导数/分支不可对齐/非有限/测量缺类是INCONCLUSIVE；否则完整可计算结果未过性能、覆盖、semantic gate或有效执行预算则NO_ACTION_SPECIFIC_SIGNAL。不得用主规格矛盾、未批准附件或backend缺能力产生负结论/Utility release。任何首次valid完整结果单用，不因负结果改seed/fixture/阈值重跑。

## 11. 明确的解释影响与仍未被证明的可行性

本文不能宣称“参数补齐就必定可运行/通过”。审批应显式接受以下含义，否则退回具体条款：

1. 成员先在clean model确定、扰动后核验而不重选；guard层禁止成为成员。这比笼统“全部finite中心选成员”更具体，guard层的候选限制是研究fixture特例，不是生产过滤政策。
2. 具体surface分配、context同移模型、corruption patch/rotation、RGB-only xyz horizon、lr/Adam及拓扑阈值此前未定义；它们确定了实际合成estimand，需批准，不宣称只是文字整理。
3. 单独corruption的normal变换及context机制不保证预期Q/Y。严禁根据future outcome修改机制或把family名字当标签。冻结以后出乎预期就是结果。
4. production topology被复用，但统计是xyz-gradient norm而非生产viewspace densification proxy；prune是合成位置压力而非baseline opacity政策。因此只验证该注册合成过程的lineage与spillover，不证明生产densification在长训练的作用。
5. native float32、s_vr可获得性、normal/depth完整导数、全residual Jacobian计算预算都尚未backend验证。可以停为INCONCLUSIVE；不能为了赶进度在资格阶段偷偷缩pixel域/换weights/改J。
6. 三surface的32-row及非空guard/far需要未来label-free合同测试确认；相机支持/三family数量需要以后backend资格确认，不能把表中算式误称已生成fixture或已通过GPU验证。
7. 16共旋转variant存在强结构相关；family bootstrap是这个有限压力套件内的稳定性摘要，不是跨几何分布、跨真实场景或路由收益置信区间。

## 12. 审批后的正确顺序与当前交付

本轮仅写Markdown/规划记录并做文本一致性审阅；不生成tensor/fixture、J、Q或Y，不调用Torch/CUDA/服务器、GT、DA3或生产训练。

请分别决定：**1A或1B；2A或2B；3A的§§3–9和§11具体影响是否接受（可逐节修改）；4A归一化/dtype/不可计算停止是否接受。** 如不接受3A中的context或horizon，不由实现者替换，只按意见重新提交书面定义。

四项及附件获明确批准后，才同步正式addendum与阻塞的TDD计划；实施、backend资格、fixture confirmation、synthetic执行及结果审计仍分别请求授权。任何未来正结果也只允许请求G-B规格；任何真实valid-negative终止及Utility GT防火墙release仍需其独立审批，绝不随文档批准产生。
