# AmbiSuR 可靠性仲裁与 Gaussian 生命周期研究设计

> 状态：第二次审核重写稿；公式闭包、可读性与实现约束已补齐 日期：2026-09-01 目标：以高反射室内场景为压力测试，在 AmbiSuR 上形成可发表、可消融、可由单张 RTX 4090 验证的方法与实验计划。

## 1. 研究目标与主张边界

### 1.1 核心问题

AmbiSuR 使用外观歧义寻找需要几何先验帮助的区域，但仍存在三个连续决策没有被显式解决：

1. 当前异常究竟来自观测不足，还是在充分观测下仍然存在的真实歧义；

2. 外部 DA3 先验与当前 Gaussian 几何谁更可信，双方冲突时是否应该拒绝决策；

3. 决策结果如何真正作用于不同 Gaussian 参数及其增密、保护和剪枝过程。

本文拟建立一条完整控制链：

$$
\text{观测充分度校准的歧义需求}
\rightarrow
\text{外部先验/内部几何双可靠性}
\rightarrow
\text{冲突仲裁}
\rightarrow
\text{参数级梯度路由}
\rightarrow
\text{Gaussian 生命周期}.
$$

### 1.2 论文级贡献候选

1. **观测校准的先验需求**：不再把低或高 SH 指标直接解释为几何好坏，而是结合有效观察数量与方向覆盖估计外部几何帮助的需求。

2. **双可靠性拒绝式仲裁与参数路由**：分别估计外部先验和内部几何的可靠性，在一致、单方占优、冲突不明时采取不同动作，并把动作落实到参数组。

3. **可靠性驱动的 Gaussian 生命周期**：同一仲裁状态同时控制连续参数优化和离散的复制、分裂、保护、隔离与剪枝。

曝光解耦、局部 Ray-Color/Ray-Normal、自适应截断和训练稳定性属于支撑机制，不分别包装为独立主创新。

### 1.3 可以与不能声称的结论

若实验成功，可以声称：

* 方法在两个被近期工作描述为紧凑、高反射室内环境的 ScanNet++ 场景上改善整体表面重建；

* 仲裁机制减少不可靠外部先验干预，并改善可靠性—覆盖率权衡；

* 参数路由与生命周期将可靠性判断转化为可验证的优化行为。

不能仅凭当前实验声称：

* 定量证明了“反光区域本身”的几何改善；当前计划不制作人工反光 mask；

* 解决了完整的反射分解、BRDF 或光照传输；

* DA3 与 Gaussian 两条证据统计独立；DA3 已参与初始化与监督；

* 对所有室内或室外数据都具有普遍提升；该结论需要 DTU/TnT 泛化证据支持。

## 2. 资源、基线与约束

* 基线：AmbiSuR 官方实现；开发 fork 为 `https://github.com/Monkot19/noob_AmbiSuR`。

* 训练环境：AutoDL，RTX 4090 24 GB，PyTorch 2.8，Python 3.12，CUDA 12.8。

* 主要数据：ScanNet++ `d415cc449b_Tool_Room` 与 `0a5c013435_Utility_Room`。

* 后续泛化：AmbiSuR 官方 DTU 15 场景与 Tanks and Temples 6 场景子集；正式实施前确认数据位置。

* 本地结果仅作只读分析；代码修改在 GitHub fork 的独立开发分支完成，再同步至服务器。

* 首版不引入学习式置信网络、完整反射分解或动态双源 TSDF，以控制 4090 显存和工程风险。

## 3. 符号、已知量与计算规则

### 3.1 量的来源

| 类别          | 符号                                                                                                | 来源                                   |
| ----------- | ------------------------------------------------------------------------------------------------- | ------------------------------------ |
| 已知训练数据      | $I_v^{gt}(r),D_v^P(r),C_v^P(r),\mathcal V,\mathcal P,K_v,R_v,t_v$                                 | GT 图像、DA3 深度与 confidence、训练相机及相邻视图对  |
| Gaussian 参数 | $\mu_i,\boldsymbol\sigma_i,q_i,\rho_i,\Theta_i$                                                   | xyz、正尺度向量、旋转四元数、opacity logit、SH 系数  |
| 曝光参数        | $a_v,b_v$                                                                                         | 每个训练视图学习的仿射曝光参数                      |
| renderer 输出 | $\alpha_{ivr},T_{ivr},w_{ivr},\widehat D_v,\widehat n_v^{prim},\widehat n_v^{depth},O_{iv}$&#x20; | alpha、透射率、合成权重、渲染深度/法线、`out_observe` |
| 派生统计        | $A_i,S_i,N_i,T_i^P,V_i^P,T_i^G,V_i^G,r_i^P,r_i^G,Z_i^{PG},V_i^{PG},K_i,\Delta_i$                    | 本文后续公式依次计算                           |
| 离线量         | GT mesh 与其点到面误差                                                                                   | 只用于 D0/最终评价，严禁进入训练与仲裁                |

像素记为 $r$，相机记为 $v,u$，Gaussian 记为 $i$。$\pi_v(X)$ 和 $\pi_v^{-1}(r,d)$ 分别是由已知相机内外参确定的投影和反投影。$\operatorname{sg}(x)$ 表示 stop-gradient；$\mathbf1[\cdot]$ 为条件成立取 1、否则取 0 的指示函数；$\operatorname{clip}(x,l,h)=\min(\max(x,l),h)$。除非另行说明，$\epsilon=10^{-8}$。

按深度从近到远光栅化时：

$$
w_{ivr}=T_{ivr}\alpha_{ivr},\qquad
T_{ivr}=\prod_{j:\,z_{jvr}<z_{ivr}}(1-\alpha_{jvr}).
$$

这里 $z_{ivr}$ 是 Gaussian $i$ 在相机 $v$ 中沿射线 $r$ 的深度。后续证据汇聚只把 $\operatorname{sg}(w_{ivr})$ 当权重，可靠性分支不能反向操纵 Gaussian。

对像素量 $x_{vr}$、有效 mask $m_{vr}$ 定义 Gaussian 加权汇聚：

$$
Z_i(x,m)=\sum_{v\in\mathcal V}\sum_{r\in\Omega_v}
\operatorname{sg}(w_{ivr})m_{vr},
$$

$$
\operatorname{Pool}_i(x;m)=
\frac{\sum_{v,r}\operatorname{sg}(w_{ivr})m_{vr}x_{vr}}
{Z_i(x,m)+\epsilon}.
$$

输出是 Gaussian 级标量；当分母不足时不把结果当低置信，而由后面的 $V$ 置零。

### 3.2 EMA 规则

需要平滑的原始统计 $x_i^{(t)}\in\{A_i,S_i,T_i^P,T_i^G,K_i\}$ 使用：

$$
\bar x_i^{(t)}=
\begin{cases}
x_i^{(t)},&\text{该 Gaussian 首次获得有效统计},\\
\beta_x\bar x_i^{(t-1)}+(1-\beta_x)x_i^{(t)},&\text{否则},
\end{cases}
$$

其中固定初值 $\beta_x=0.9$，D0 最多允许调整一次并随后冻结。新 Gaussian 不继承父节点的 EMA。下文为避免符号臃肿，仲裁中的 $A,S,T,K$ 均指其 stop-gradient EMA；需求度 $N$ 由平滑后的 $A,S$ 重新计算，不再对 $N$ 二次 EMA。

对 $K_i$ 另加 joint-validity 合同：只有当前刷新 $V_i^{PG}=1$ 时，$K_i^{raw}$ 才是有效观测并按上式初始化或更新 EMA。$V_i^{PG}=0$ 时不计算或不使用零分母产生的 $K_i^{raw}$，不更新 EMA；可以保留历史 $K_i$ EMA 供日志观察，但当前刷新严禁绕过 $V_i^{PG}$ 参与仲裁。新 Gaussian 在首次获得有效 joint support 前，K EMA 标记为未初始化，不能初始化为 1。

## 4. 观测校准的歧义需求

### 4.1 高端 SH 连续歧义量 A_i

设当前 SH 阶数为 $L$，非 DC 系数为 $\theta_{i,\ell m}\in\mathbb R^3$：

$$
u_i=\sqrt{\sum_{\ell=1}^{L}\sum_{m=-\ell}^{\ell}
\|\theta_{i,\ell m}\|_2^2}.
$$

在当前活动 Gaussian 集合 $\mathcal G$ 上计算：

$$
q_L=Q_{0.10}(\{u_j:j\in\mathcal G\}),\qquad
q_H=Q_{0.95}(\{u_j:j\in\mathcal G\}),
$$

$$
A_i=\operatorname{clip}\!\left(
\frac{u_i-q_L}{q_H-q_L+\epsilon},0,1\right).
$$

它只表达高 SH/视角相关外观复杂度。低 SH 不再自动判为歧义；低 SH 但观测不足的风险由 $S_i$ 负责。实现必须按 $(L+1)^2-1$ 动态取得非 DC 系数，不能沿用当前代码写死的 degree-3 共 15 个系数。

### 4.2 观测充分度 S_i

每隔固定 $\Delta_{obs}=1000$ 轮，在 `no_grad` 下遍历全部训练相机。现有 `out_observe` 是单视图前景区域的像素命中数(一个长度等于 Gaussian 数量的整型张量：

`out_observe[i]`表示在当前相机视图`v`中，`Gaussian (i)` 被多少个像素“较靠前地有效使用”)，因此先二值化：

$$
o_{iv}=\mathbf1[O_{iv}>0],\qquad
M_i=\sum_{v\in\mathcal V}o_{iv}.
$$

视图数量充分度为：

$$
S_i^{count}=\operatorname{clip}\left(\frac{M_i}{K_c},0,1\right),
\qquad K_c=5.
$$

相机 $v$ 相对 Gaussian 的单位观察方向为：

$$
d_{iv}=\frac{c_v-\mu_i}
{\max(\|c_v-\mu_i\|_2,\epsilon)},\qquad
q_i^{view}=\sum_{v\in\mathcal V}o_{iv}d_{iv},
$$

其中 $c_v$ 是已知相机中心。方向离散度为：

$$
D_i=
\begin{cases}
0,&M_i<2,\\
\operatorname{clip}\!\left(
\dfrac{M_i^2-\|q_i^{view}\|_2^2}{2M_i(M_i-1)},0,1\right),&M_i\ge2.
\end{cases}
$$

令固定参考角度 $\theta_c=30^\circ$，则：

$$
D_c=\frac{1-\cos\theta_c}{2},\qquad
S_i^{angle}=\operatorname{clip}\left(\frac{D_i}{D_c},0,1\right),
$$

$$
S_i=\sqrt{S_i^{count}S_i^{angle}}.
$$

解释：很多相机从近似同一方向观察，或者只有少量相机从分散方向观察，都不能得到高 $S_i$。原始 $O_{iv}$ 不能直接参与 $S^{count}$，否则大 footprint Gaussian 会因覆盖像素多而虚假显得“观测充分”。

### 4.3 最终需求度

$$
N_i=1-S_i(1-A_i).
$$

因此 $S_i=0$ 时必有 $N_i=1$；$S_i\approx1,A_i\approx0$ 时需求接近 0；充分观测下仍有高 SH 时需求仍高。$N_i$ 是“需要外部几何帮助的程度”，不是反射概率，也不能替代反光区域 mask。

## 5. 外部先验与内部几何的双可靠性

### 5.1 共用跨视图误差

DA3 法线 $n_v^P$ 由已知深度 $D_v^P$ 反投影相邻像素、叉乘并单位化得到；内部法线 $n_v^G=\widehat n_v^{depth}$。对相邻训练视图对 $(v,u)\in\mathcal P$，给定有效深度图 $D_v^X,D_u^X$（$X=P$ 表示 DA3，$X=G$ 表示当前渲染）：

$$
X_{vr}^X=\pi_v^{-1}(r,D_v^X(r)),\quad
(r',z_{u\leftarrow v}^X)=\pi_u(X_{vr}^X).
$$

有效投影 mask 为：

$$
m_{vur}^X=\mathbf1[r'\in\Omega_u]\mathbf1[z_{u\leftarrow v}^X>0]
\mathbf1[D_v^X(r),D_u^X(r')\text{有限且}>0]
\mathbf1[z_{u\leftarrow v}^X\le(1+\tau_{occ})D_u^X(r')],
$$

其中固定 $\tau_{occ}=0.05$ 只排除被目标视图前景遮挡的投影。相对深度误差与法线误差为：

$$
e_{d,vur}^X=
\frac{|z_{u\leftarrow v}^X-D_u^X(r')|}
{z_{u\leftarrow v}^X+D_u^X(r')+\epsilon},
$$

$$
e_{n,vur}^X=1-
\left|\langle n_v^X(r),n_u^X(r')\rangle\right|.
$$

法线先转换到同一世界坐标系并单位化；绝对内积避免法线符号翻转被误判为几何冲突。单个跨视图一致性分数为：

$$
k_{vur}^X=\exp\!\left[-\frac12\left(
\frac{e_{d,vur}^X}{\tau_d^X}+\frac{e_{n,vur}^X}{\tau_n^X}
\right)\right],
$$

初值为 $\tau_d^P=\tau_d^G=0.05$、$\tau_n^P=\tau_n^G=0.10$。NCC 不进入可靠性主通道，因为反射表面可能几何正确但光度不一致。

### 5.2 外部 DA3 可靠性

先把每个视图的 DA3 confidence 稳健归一化：

$$
c_{05,v}=Q_{0.05}(C_v^P),\quad c_{95,v}=Q_{0.95}(C_v^P),
$$

$$
\widehat C_v^P(r)=\operatorname{clip}\left(
\frac{C_v^P(r)-c_{05,v}}{c_{95,v}-c_{05,v}+\epsilon},0,1\right).
$$

外部 confidence 与跨视图一致性分别汇聚为：

$$
E_i^{P,conf}=\operatorname{Pool}_i(\widehat C_v^P;\,m_v^P),
$$

$$
E_i^{P,mv}=
\frac{\sum_{(v,u)\in\mathcal P}\sum_r
\operatorname{sg}(w_{ivr})m_{vur}^P k_{vur}^P}
{\sum_{(v,u),r}\operatorname{sg}(w_{ivr})m_{vur}^P+\epsilon},
$$

其中 $m_v^P=\mathbf1[D_v^P,C_v^P\text{有限且深度}>0]$。对每个源视图定义有效支持：

$$
b_{iv}^P=\mathbf1\!\left[
\sum_{u:(v,u)\in\mathcal P}\sum_r
\operatorname{sg}(w_{ivr})m_{vur}^P>\tau_Z\right],
$$

$$
M_i^P=\sum_v b_{iv}^P,\qquad
V_i^P=\mathbf1[M_i^P\ge K_P],
$$

其中 $\tau_Z=10^{-4}$、$K_P=2$ 为固定初值。最终：

$$
T_i^P=\sqrt{E_i^{P,conf}E_i^{P,mv}},\qquad
r_i^P=V_i^PT_i^P.
$$

$T_i^P$ 表示有证据时先验有多可信；$V_i^P$ 表示证据是否足以使用。两者不得合并成一个含义不清的 confidence。

### 5.3 内部 Gaussian 几何可靠性

令 $D_v^G=\widehat D_v$，$n_v^G=\widehat n_v^{depth}$。跨视图分数按同一公式得到：

$$
E_i^{G,mv}=
\frac{\sum_{(v,u),r}\operatorname{sg}(w_{ivr})m_{vur}^G k_{vur}^G}
{\sum_{(v,u),r}\operatorname{sg}(w_{ivr})m_{vur}^G+\epsilon}.
$$

同一视图中 primitive 混合法线与深度法线的一致性为：

$$
k_{vr}^{dn}=\exp\!\left[-
\frac{1-|\langle\widehat n_v^{prim}(r),\widehat n_v^{depth}(r)\rangle|}
{\tau_{dn}}\right],\qquad \tau_{dn}=0.10,
$$

$$
E_i^{G,dn}=\operatorname{Pool}_i(k_{vr}^{dn};\,m_{vr}^{dn}),
$$

其中 $m_{vr}^{dn}=1$ 当两种法线有限、单位化且渲染 alpha 不低于固定 $\tau_\alpha=0.5$，否则为 0。

历史稳定性使用相邻两次证据刷新时保存的中心与 Gaussian 法线。Gaussian 法线由最短尺度轴得到：

$$
k_i^*=\arg\min_{k\in\{1,2,3\}}\sigma_{ik},\qquad
n_i=R(q_i)e_{k_i^*}.
$$

设上次刷新量为 $\mu_i^- ,n_i^-$，尺度参考 $\sigma_i^{ref}=\max_k\sigma_{ik}$：

$$
e_i^{move}=\frac{\|\mu_i-\mu_i^-\|_2}{\sigma_i^{ref}+\epsilon},\qquad
e_i^{rot}=\frac{\arccos(\operatorname{clip}(|\langle n_i,n_i^-\rangle|,-1,1))}{\pi},
$$

$$
E_i^{G,stab}=\exp\!\left(-\frac{e_i^{move}}{\tau_{move}}
-\frac{e_i^{rot}}{\tau_{rot}}\right),
$$

初值 $\tau_{move}=0.25$、$\tau_{rot}=15^\circ/180^\circ$。首次刷新没有历史，因此历史有效标记 $v_i^{hist}=0$；以后为 1。

内部跨视图支持明确定义为：

$$
b_{iv}^G=\mathbf1\!\left[
\sum_{u:(v,u)\in\mathcal P}\sum_r
\operatorname{sg}(w_{ivr})m_{vur}^G>\tau_Z\right],\qquad
M_i^G=\sum_vb_{iv}^G.
$$

然后定义：

$$
V_i^G=\mathbf1[M_i^G\ge K_G]\,v_i^{hist},\qquad K_G=2,
$$

$$
T_i^G=(E_i^{G,mv}E_i^{G,dn}E_i^{G,stab})^{1/3},\qquad
r_i^G=V_i^GT_i^G.
$$

### 5.4 外部—内部一致性 K_i

在同一相机像素上比较 DA3 与当前渲染：

$$
e_{d,vr}^{PG}=\frac{|D_v^P(r)-\widehat D_v(r)|}
{D_v^P(r)+\widehat D_v(r)+\epsilon},
$$

$$
e_{n,vr}^{PG}=1-|\langle n_v^P(r),\widehat n_v^{depth}(r)\rangle|.
$$

令 $m_{vr}^{PG}$ 表示两路深度和法线都有限且渲染 alpha $\ge\tau_\alpha$，则：

$$
Z_i^{PG}=\sum_{v,r}\operatorname{sg}(w_{ivr})m_{vr}^{PG},
$$

$$
V_i^{PG}=\mathbf1[Z_i^{PG}>\tau_Z],\qquad \tau_Z=10^{-4}.
$$

这里复用第 5.1 节已有的 $\tau_Z$，不新增超参数。只有 $V_i^{PG}=1$ 时才计算当前刷新的 pooled error 与原始一致性：

$$
\bar e_{d,i}^{PG}=\operatorname{Pool}_i(e_{d,vr}^{PG};m_{vr}^{PG}),\quad
\bar e_{n,i}^{PG}=\operatorname{Pool}_i(e_{n,vr}^{PG};m_{vr}^{PG}),
$$

$$
K_i^{raw}=\exp\!\left[-\frac12\left(
\frac{\bar e_{d,i}^{PG}}{\tau_d^{PG}}+
\frac{\bar e_{n,i}^{PG}}{\tau_n^{PG}}\right)\right],
$$

其中 $\tau_d^{PG}=0.05,\tau_n^{PG}=0.10$。有效的 $K_i^{raw}$ 按第 3.2 节更新 K EMA；仲裁中的 $K_i$ 指该 EMA，但其本轮可用性的必要条件仍是 $V_i^{PG}=1$。$V_i^{PG}=0$ 时，不把零分母对应的 $K_i^{raw}\approx1$ 当作一致，不更新 K EMA；历史 EMA 只能记录，不能参与当前仲裁。可靠性差仍定义为：

$$
\Delta_i=r_i^P-r_i^G.
$$

## 6. 五状态拒绝式仲裁

先定义“达到可主导水平”的布尔量：

$$
H_i^P=\mathbf1[V_i^P=1]\mathbf1[T_i^P\ge\tau_P],\qquad
H_i^G=\mathbf1[V_i^G=1]\mathbf1[T_i^G\ge\tau_G],
$$

固定初值为 $\tau_N=0.5,\tau_P=\tau_G=0.5,\tau_K=0.6,\delta=0.1$。候选状态 $\widehat s_i$ 按优先级完整定义为：

$$
\widehat s_i=
\begin{cases}
\text{Bypass},&N_i\le\tau_N,\\
\text{Consensus},&H_i^P=H_i^G=1\ \land\ V_i^{PG}=1\ \land\ K_i\ge\tau_K,\\
\text{Prior-led},&H_i^P=1\ \land\ H_i^G=0,\\
\text{Geometry-led},&H_i^P=0\ \land\ H_i^G=1,\\
\text{Prior-led},&H_i^P=H_i^G=1\ \land\ V_i^{PG}=1\ \land\ K_i<\tau_K\ \land\ \Delta_i>\delta,\\
\text{Geometry-led},&H_i^P=H_i^G=1\ \land\ V_i^{PG}=1\ \land\ K_i<\tau_K\ \land\ \Delta_i<-\delta,\\
\text{Abstain},&\text{其余情况}.
\end{cases}
$$

Bypass 保持最高优先级，不要求 $V_i^{PG}$。单方达到主导条件时也不依赖 K：$H_i^P=1,H_i^G=0$ 仍为 Prior-led，$H_i^P=0,H_i^G=1$ 仍为 Geometry-led。只有 $H_i^P=H_i^G=1$、需要判断双方一致或冲突时才要求 $V_i^{PG}=1$。若同时满足 $N_i>\tau_N$、$H_i^P=H_i^G=1$、$V_i^{PG}=0$，候选状态为 Abstain；$\Delta_i$ 只记录诊断，不得选边，也不得把 joint-invalid 解释为 Consensus 或双方冲突。

这覆盖了低需求、双方一致、单方可信、双方冲突且一方明显占优、joint-invalid、冲突难判以及双方证据不足，不存在未定义分支。

为防止状态抖动，候选状态连续计数为：

$$
h_i^{(t)}=
\begin{cases}
h_i^{(t-1)}+1,&\widehat s_i^{(t)}=\widehat s_i^{(t-1)},\\
1,&\widehat s_i^{(t)}\ne\widehat s_i^{(t-1)},
\end{cases}
$$

$$
s_i^{(t)}=
\begin{cases}
\widehat s_i^{(t)},&h_i^{(t)}\ge H_{enter},\\
s_i^{(t-1)},&h_i^{(t)}<H_{enter},
\end{cases}
\qquad H_{enter}=3.
$$

新 Gaussian 在 Probation 期间只接受基础训练，不直接进入强状态。所有阈值只在 Tool Room seed 0 的 D0 中允许一次有记录的调整，随后冻结并原样用于 Utility Room、其他种子和泛化场景；Tool Room 应明确标作开发场景，避免把调参结果包装成完全独立测试。

## 7. 从状态到参数梯度的执行器

### 7.1 两条梯度路径

基础损失和先验损失先分别形成标量：

$$
\mathcal L_B=\mathcal L_{photo}
+\lambda_{scale}\mathcal L_{scale}
+\lambda_{sv}\mathcal L_{sv}
+\lambda_{mv}\mathcal L_{mv}
+\lambda_{RC}\mathcal L_{RC}
+\lambda_{RN}\mathcal L_{RN},
$$

$$
\mathcal L_P=\lambda_{Pd}\mathcal L_{Pd}
+\lambda_{Pn}\mathcal L_{Pn}.
$$

令 $R_{scene}>0$ 为 AmbiSuR 已知的相机场景尺度，先验像素权重 $\omega_{vr}^P=m_v^P\widehat C_v^P(r)$。两项先验损失完整定义为：

$$
\mathcal L_{Pd}=\frac{
\sum_{v,r}\omega_{vr}^P|D_v^P(r)-\widehat D_v(r)|}
{R_{scene}(\sum_{v,r}\omega_{vr}^P+\epsilon)},
$$

$$
\mathcal L_{Pn}=\frac{
\sum_{v,r}\omega_{vr}^P
\left[1-|\langle n_v^P(r),\widehat n_v^{depth}(r)\rangle|\right]}
{\sum_{v,r}\omega_{vr}^P+\epsilon}.
$$

其中 $\mathcal L_{photo}$ 是第 9.1 节曝光解耦图像损失；$\mathcal L_{scale},\mathcal L_{sv},\mathcal L_{mv}$ 分别是 AmbiSuR 原有尺度、单视图法线和多视图几何损失；Ray 损失见第 9.2–9.3 节。所有 $\lambda$ 都是配置中的固定超参数：Baseline 已有项保持官方默认，新项初值写在对应小节。

Gaussian 参数分组为：

$$
\theta_{i,pos}=(\mu_i,q_i),\quad
\theta_{i,scale}=\boldsymbol\sigma_i,\quad
\theta_{i,opa}=\rho_i,\quad
\theta_{i,SH}=\Theta_i.
$$

两条梯度由各自损失明确得到：

$$
g_{B,i,q}=\frac{\partial\mathcal L_B}{\partial\theta_{i,q}},\qquad
g_{P,i,q}=\frac{\partial\mathcal L_P}{\partial\theta_{i,q}}.
$$

外部先验不允许更新 SH、曝光，因此强制 $g_{P,i,SH}=0$，且 $\partial\mathcal L_P/\partial(a_v,b_v)=0$。曝光参数始终只接受 $\partial\mathcal L_B/\partial(a_v,b_v)$，不做逐 Gaussian 状态 mask。核心路由验证时先令 $\lambda_{RC}=\lambda_{RN}=0$，避免把 Ray 模块收益混入 Core。

先验梯度在路由前按参数组限幅。令：

$$
b_q=Q_{0.50}\left(\{\|g_{B,j,q}\|_2:\|g_{B,j,q}\|_2>0\}\right),
$$

若集合为空则取 $b_q=0$。固定 $\gamma_q=2$，并定义：

$$
g_{P,i,q}^{clip}=g_{P,i,q}
\min\left(1,\frac{\gamma_qb_q}{\|g_{P,i,q}\|_2+\epsilon}\right).
$$

后文简写 $g_P$ 时均指 $g_P^{clip}$。这防止外部先验仅因梯度量纲大而赢得优化主导权。

### 7.2 参数组

$q\in\{pos,scale,opa,SH\}$。曝光单独按上一节处理。

### 7.3 C4 的原始加权路由

$$
g_{i,q}^{C4}=m_{s_i,q}^{B}g_{B,i,q}+m_{s_i,q}^{P}g_{P,i,q}.
$$

推荐动作表如下：

| 状态           | xyz/rotation  | scaling           | opacity     | SH/exposure |
| ------------ | ------------- | ----------------- | ----------- | ----------- |
| Bypass       | $g_B$&#x20;   | $g_B$&#x20;       | $g_B$       | $g_B$       |
| Consensus    | $g_B+0.5g_P$  | $g_B+0.1g_P$      | $g_B$&#x20; | $g_B$       |
| Prior-led    | $g_P+0.25g_B$ | $0.25g_P+0.10g_B$ | 0           | $g_B$       |
| Geometry-led | $g_B$&#x20;   | $g_B$&#x20;       | $g_B$       | $g_B$       |
| Abstain      | 0             | 0                 | 0           | $g_B$       |

表中数值是 D0 初值，只允许一次有记录调整。C4 故意使用未投影加权和；因此 C5 才是首次加入冲突投影，二者保持严格单变量增量。

### 7.4 C5 的梯度冲突投影

仅当同一参数组同时保留两路梯度且内积为负时，保留主导梯度 $g_d$，删除辅助梯度 $g_a$ 的对抗分量：

$$
\widetilde g_a=g_a-
\frac{\min(0,\langle g_a,g_d\rangle)}
{\|g_d\|_2^2+\epsilon}g_d.
$$

* Prior-led：$g_d=g_P, g_a=g_B$；

* Consensus：若 $r_i^P\ge r_i^G$，取 $g_d=g_P,g_a=g_B$；否则反过来；

* Bypass、Geometry-led、Abstain 中没有双路混合，不执行投影。

梯度执行顺序固定为：

$$
\text{有限性检查}
\rightarrow
\text{先验梯度限幅}
\rightarrow
\text{冲突投影}
\rightarrow
\text{状态 mask}
\rightarrow
\text{一次 optimizer step}.
$$

所有损失先分别用 `torch.autograd.grad` 或等价显式接口取得两路梯度，再执行上述有限性检查、限幅、投影和 mask；最终只允许一次参数更新。不得沿用 ALR 的布尔切片 `requires_grad_` 或多次 `backward(retain_graph=True)` 来模拟逐 Gaussian 路由。

## 8. 可靠性驱动的 Gaussian 生命周期

状态映射：

| 仲裁状态         | 生命周期状态     | 主要行为                  |
| ------------ | ---------- | --------------------- |
| Bypass       | Normal     | 使用基线更新和普通增密规则         |
| Consensus    | Confirmed  | 允许可靠增密，保护已确认表面        |
| Prior-led    | Repair     | 先修复位置/旋转，持续稳定后才允许分裂   |
| Geometry-led | Protected  | 拒绝外部先验；不因高 RGB 残差盲目复制 |
| Abstain      | Quarantine | 暂停强几何更新与增密，延迟决定是否剪枝   |
| 新生成 Gaussian | Probation  | 从中性统计开始，设冷却期，禁止连续分裂   |

**生命周期状态语义（2026-09-02 已批准澄清 A）：** 第 6 节得到的 $s_i$ 始终表示经过 $H_{enter}$ 迟滞后的稳定五状态仲裁结果；$\ell_i$ 始终表示供连续动作保护与离散 topology 门控消费的生命周期状态。记上表的确定性映射为 $\mathcal M$。除新点冷却外，只在证据刷新 $t$ 执行

$$
\ell_i^{(t)}=\mathcal M\!\left(s_i^{(t)}\right),
$$

两次证据刷新之间保持 $\ell_i$ 不变，不允许按每个训练 iteration 重新解释 $s_i$。

若 Gaussian $i$ 在 optimizer iteration $b_i$ 的 topology 提交后新生成，则其 $\ell_i$ 强制初始化为 Probation，并至少保持 $C_{prob}=500$ 个 optimizer iteration。Probation 期间仍可计算和记录 $s_i$，但 $s_i$ 不得覆盖 $\ell_i$，也不得绕过 Probation 触发先验强路由、clone、split 或普通 opacity prune。令 $t^*$ 是满足当前 iteration $k_{t^*}\ge b_i+C_{prob}$ 的第一次证据刷新；只在 $t^*$ 才用当时已经稳定的 $s_i^{(t^*)}$ 执行 $\ell_i^{(t^*)}=\mathcal M(s_i^{(t^*)})$。因此达到第 500 轮本身不构成刷新，也不在刷新间隔中途解除 Probation。

### 8.1 两阶段提交

令 $a\in\{clone,split,prune\}$，$B_{i,a}^{(t)}\in\{0,1\}$ 是 AmbiSuR 原始增密/剪枝规则在第 $t$ 轮给出的候选事件。生命周期只增加门控，不暗中改写 baseline 判据。状态门控为：

$$
G_{i,clone}=G_{i,split}=
\mathbf1[\ell_i\in\{\text{Normal},\text{Confirmed}\}],
$$

$$
G_{i,prune}=
\mathbf1[\ell_i\ne\text{Confirmed}]
\mathbf1[\ell_i\ne\text{Protected}].
$$

动作条件为 $E_{i,a}^{(t)}=B_{i,a}^{(t)}G_{i,a}^{(t)}$。所有 clone、split、prune 候选只消费 $\ell_i$，不得直接把五状态枚举 $s_i$ 与生命周期枚举比较。正确的持续计数递推是：

$$
z_{i,a}^{(t)}=
\begin{cases}
\min(z_{i,a}^{(t-1)}+1,H_a),&E_{i,a}^{(t)}=1,\\
0,&E_{i,a}^{(t)}=0.
\end{cases}
$$

只有 $z_{i,a}^{(t)}=H_a$ 且冷却计数 $c_i^{(t)}=0$ 才提交动作。初值 $H_{clone}=H_{split}=3,H_{prune}=5$。冷却递推为：

$$
c_i^{(t+1)}=
\begin{cases}
C_a,&\text{第 }t\text{ 轮执行动作 }a,\\
\max(c_i^{(t)}-1,0),&\text{否则},
\end{cases}
$$

其中 $C_{clone}=C_{split}=500,C_{prune}=0$ 轮。该分段递推替换旧稿中无法实际计数的指示函数写法。

生命周期状态持续次数另记为：

$$
d_i^{(t)}=
\begin{cases}
d_i^{(t-1)}+1,&\ell_i^{(t)}=\ell_i^{(t-1)},\\
1,&\ell_i^{(t)}\ne\ell_i^{(t-1)},
\end{cases}
$$

这里 $t$ 表示证据刷新序号。$d_i$ 只在证据刷新时更新：若该次刷新后的 $\ell_i$ 与上次刷新相同则加一，发生映射变化则重置为 1；普通训练 iteration 和仅到达 Probation 的 500 轮边界均不得改变 $d_i$。$\ell_i\in\{Normal,Confirmed,Repair,Protected,Quarantine,Probation\}$ 由第 8 节映射表和 Probation 覆盖规则得到；第 8.2 节“状态持续至少 $H_Q$ 次刷新”即 $d_i\ge H_Q$。

### 8.2 具体约束

* Repair：$G_{clone}=G_{split}=0$，先路由位置/旋转；只有仲裁持续转为 Consensus 后才映射到 Confirmed。

* Protected：允许基础参数与外观学习，但 $G_{clone}=G_{split}=0$，避免高光 RGB 残差复制错误几何。

* Quarantine：禁止 clone/split；仅当状态持续至少 $H_Q=5$ 次证据刷新、opacity $\operatorname{sigmoid}(\rho_i)<\tau_o=0.01$ 且 $M_i<K_{prune}=2$ 时，才额外允许 prune 候选。

* Probation：新 Gaussian 的 $A,S,T,V,K$ 历史清零；Core C6 不为其设置或迁移逐 Gaussian truncation 状态，而是原样沿用 baseline 的全局 `--trunc_sigma/--disable_trunc` 配置（当前默认 `trunc_sigma=2.0`）。从生成后开始至少 $C_{prob}=500$ 个 optimizer iteration 只接受 $g_B$，禁止先验强路由、再次 split/clone 和普通 opacity prune；满 500 轮后仍须等到下一次证据刷新，才按当时稳定的 $s_i$ 映射到非 Probation 生命周期状态。

核心原则是“先修复再复制、先隔离后剪枝”。

**Core 边界澄清（2026-09-02 已批准方案 A）：** 上述 `2.0` 只是 baseline 全局 renderer 配置的当前默认值，不是 Probation 的逐 Gaussian 生命周期动作，也不构成 C6 新变量。Core C1–C6 不新增 per-Gaussian `trunc_sigma` Tensor、不按 $\ell_i$ 改写全局截断参数，也不修改现有 Python/C++/CUDA truncation 接口。第 9.4 节的状态驱动逐 Gaussian 截断完整保留为 Supporting 候选；只有 Core 通过 G0–G2 后，才可在独立阶段、tag 和消融中另行评估。

## 9. 配套增强

### 9.1 CoMe 式解耦曝光损失

保留 AmbiSuR 的每视角仿射曝光模型：

$$
I_v^{app}(r)=\exp(a_v)I_v^{raw}(r)+b_v.
$$

$a_v,b_v\in\mathbb R^3$ 是每视图、每颜色通道学习参数，初始化为 0；训练损失前不 clamp，只在显示/评价时 clamp 到 $[0,1]$。对任一局部 SSIM 窗口和两幅图 $X,Y$，令高斯窗加权均值、方差与协方差为：

$$
\mu_X=G_\sigma*X,\quad
\sigma_X^2=G_\sigma*(X^2)-\mu_X^2,\quad
\sigma_{XY}=G_\sigma*(XY)-\mu_X\mu_Y,
$$

$$
l(X,Y)=\frac{2\mu_X\mu_Y+C_1}{\mu_X^2+\mu_Y^2+C_1},
$$

$$
c(X,Y)=\frac{2\sigma_X\sigma_Y+C_2}{\sigma_X^2+\sigma_Y^2+C_2},\qquad
s(X,Y)=\frac{\sigma_{XY}+C_3}{\sigma_X\sigma_Y+C_3}.
$$

采用标准 $C_1=(0.01)^2,C_2=(0.03)^2,C_3=C_2/2$，各通道和有效像素取平均。于是：

训练损失为：

$$
\mathcal L_{photo}
=(1-\lambda_{ssim})\operatorname{mean}_{v,r}
\|I_v^{app}(r)-I_v^{gt}(r)\|_1
+\lambda_{ssim}
\operatorname{mean}_{v,r}\left[1-
l(I_v^{gt},I_v^{app})
c(I_v^{gt},I_v^{raw})
s(I_v^{gt},I_v^{raw})
\right].
$$

初值沿用 baseline 的 $\lambda_{ssim}$。L1 与亮度项向 Gaussian 外观和 $a_v,b_v$ 传梯度；对比度、结构项只看原始渲染，不允许曝光模块掩盖结构错误。当前代码的 `app_image_detach` 不能用于亮度项，否则曝光得不到 SSIM 亮度梯度。

### 9.2 可靠前表面局部 Ray-Color

对当前训练视图的一条射线 $r$，按深度排列参与者并记 $w_{ri}=T_{ri}\alpha_{ri}$、总 alpha $A_r=\sum_iw_{ri}$、累计权重 $C_{rk}=\sum_{j\le k}w_{rj}$。仅对 $A_r\ge\tau_A$ 的射线继续计算，否则直接令 $h_r=0$。令固定 $\rho_f=0.5$，前表面索引和深度为：

$$
k_r=\min\{k:C_{rk}\ge\rho_fA_r\},\qquad z_r^f=z_{rk_r}.
$$

使用 $\tau_w=0.01,\tau_f=0.02$ 定义局部集合：

$$
\mathcal F_r=\{i:w_{ri}\ge\tau_wA_r,\ |z_{ri}-z_r^f|\le\tau_fz_r^f\}.
$$

局部权重、均值和相对标准差为：

$$
W_r^F=\sum_{i\in\mathcal F_r}w_{ri},\quad
\widetilde w_{ri}=\frac{w_{ri}}{W_r^F+\epsilon},\quad
\bar z_r^F=\sum_{i\in\mathcal F_r}\widetilde w_{ri}z_{ri},
$$

$$
\nu_r^F=\frac{
\sqrt{\sum_{i\in\mathcal F_r}\widetilde w_{ri}(z_{ri}-\bar z_r^F)^2}}
{\bar z_r^F+\epsilon}.
$$

可靠单前表面门控为：

$$
h_r=\mathbf1[A_r\ge\tau_A]
\mathbf1[W_r^F/(A_r+\epsilon)\ge\tau_F]
\mathbf1[\nu_r^F\le\tau_\nu],
$$

初值 $\tau_A=0.8,\tau_F=0.8,\tau_\nu=0.01$。集合、门控和权重全部 stop-gradient。

令 $d_{ri}=\operatorname{sg}((c_v-\mu_i)/\max(\|c_v-\mu_i\|,\epsilon))$，Gaussian 颜色 $c_{ri}=\operatorname{SH}(\Theta_i,d_{ri})$。CoMe v2 一致的平方 L2 颜色方差为：

$$
\mathcal L_{RC}
=\frac{\sum_r h_r\sum_{i\in\mathcal F_r}
\operatorname{sg}(\widetilde w_{ri})
\|c_{ri}-I_r^{gt}\|_2^2}
{\sum_rh_r+\epsilon}.
$$

初值 $\lambda_{RC}=0.5$，7k 后启用。由于方向和权重均 stop-gradient，主版本只更新 SH；这显式切断当前 CUDA Ray-Color 经视角方向泄漏到 xyz 的路径。无条件全前景版本仅作消融。

### 9.3 可靠前表面局部 Ray-Normal

由第 5.3 节得到无符号 Gaussian 法线 $n_i$。令观察方向 $d_{ri}$ 指向相机，并用 stop-gradient 符号面向相机：

$$
n_{ri}=\xi_{ri}n_i,\qquad
\xi_{ri}=\begin{cases}1,&\langle n_i,d_{ri}\rangle\ge0,\\-1,&\text{否则}.\end{cases}
$$

忠实采用 CoMe v2 的 alpha 混合法线，而不是旧稿擅自改写的单位均值余弦损失：

$$
N_r=\sum_{i\in\mathcal F_r}\operatorname{sg}(w_{ri})n_{ri},
$$

$$
\mathcal L_{RN}=\frac{\sum_rh_r\sum_{i\in\mathcal F_r}
\operatorname{sg}(w_{ri})\|n_{ri}-N_r\|_2^2}
{\sum_rh_r+\epsilon}.
$$

初值 $\lambda_{RN}=0.005$，15k 后启用。最短尺度轴索引、符号、集合和权重都 stop-gradient，因此该项主要更新 rotation，不通过权重更新 xyz/scale/opacity。它与 rendered depth-normal 一致性分别约束 primitive 层与渲染表面层。

### 9.4 状态驱动 Gaussian 截断

每个 Gaussian 的离散截断半径为：

| 生命周期状态                | `trunc_sigma` |
| --------------------- | ------------: |
| Confirmed / Protected |           2.2 |
| Normal / Probation    |           2.0 |
| Repair                |           1.7 |
| Quarantine            |           1.5 |

截断状态延迟启用并通过渐变或有限步长更新。半径缩小时限制 opacity 的补偿性增大；新 Gaussian 从 2.0 开始。只有自适应硬截断表现稳定后，才增加平滑边界消融。

具体地，令上表给出的目标为 $\gamma_i^*=\Gamma(\text{lifecycle}_i)$，7k 前 $\gamma_i=2.0$，之后每次证据刷新：

$$
\gamma_i^{(t)}=\gamma_i^{(t-1)}+
\operatorname{clip}(\gamma_i^*-\gamma_i^{(t-1)},-\eta_\gamma,\eta_\gamma),
\qquad\eta_\gamma=0.1.
$$

若 $\gamma_i^*<\gamma_i^{(t-1)}$，对 opacity logit 梯度执行 $g_{\rho_i}\leftarrow\max(g_{\rho_i},0)$。因为优化器做 $\rho\leftarrow\rho-\eta g_\rho$，负梯度才会增大 opacity；这一符号关系必须写对。逐 Gaussian $\gamma_i$ 必须作为 CUDA 张量进入 forward 与 backward 的同一截断判断，不能只改 Python 表或仍使用全局标量。

## 10. 30k 训练时序

|      迭代 | 阶段      | 开启内容                         | 禁止内容                   |
| ------: | ------- | ---------------------------- | ---------------------- |
|    0–1k | 基础初始化   | AmbiSuR 基础训练与统计缓冲区初始化        | 强路由、生命周期动作             |
|   1k–5k | 曝光/证据预热 | 解耦曝光；积累 $A,S,T,V,K$ 的 EMA    | 强路由、不可逆动作              |
|   5k–7k | 影子决策    | 计算五状态和诊断指标                   | 状态不影响梯度或生命周期           |
|  7k–15k | 主动修复    | 参数路由、局部 Ray-Color、状态截断、生命周期  | 过早 Ray-Normal；无持续证据的剪枝 |
| 15k–30k | 固定拓扑精修  | 停止增密；局部 Ray-Normal；先验衰减；截断冻结 | 新 clone/split；频繁状态结构变化 |

15k 后的先验衰减系数为：

$$
\lambda_P(t)=\lambda_P^{15k}\operatorname{clip}\left(
\frac{30000-t}{15000},0,1\right),\qquad t\ge15000.
$$

它只乘 $\mathcal L_P$。Ray 辅助项必须显式输出可记录的 loss map，并与其他损失合成后只执行一次参数更新。每轮顺序固定为：前向与统计 → 分别取得 $g_B,g_P$ → 路由/投影 → `optimizer.step()` 更新旧参数 → 无梯度执行到期的 lifecycle 动作 → 迁移/初始化 Adam 状态及 EMA/生命周期数组。不得在 optimizer step 前替换 Parameter 后仍假设本轮旧梯度和动量自然有效。

## 11. 诊断、消融与主实验

### 11.1 D0：影子仲裁诊断

D0 只计算 $S,A,N,T,V,K,\Delta$ 和五状态，不改变训练。joint-invalid 时只把候选状态记录为 Abstain，不执行任何训练动作。报告：

* $N$、$A$、$1-S$ 对高几何误差的 AUROC、AUPRC 或分位风险；

* 两路可靠性的 selective-risk / coverage-risk；

* 状态占比、转移矩阵、平均持续时间和抖动率；

* 冲突子集的选边正确率、Abstain coverage 与 oracle regret；

* 各状态对应的 GT 几何误差分布。

* $Z^{PG}$、$V^{PG}$、$P(V^{PG}=1)$ 与 $P(V^{PG}=1\mid H^P=H^G=1)$；

* joint-invalid 导致的 Abstain 比例，以及按场景、训练阶段和 N 分位区间分组的 joint coverage；

* joint-valid 与 joint-invalid Gaussian 的 GT 离线几何误差。GT 仍只在离线诊断读取。

若采用该 gate 后几乎全部进入 Abstain，只报告 coverage 问题并停止晋级；不得自动改成基于 $\Delta$ 的回退、降低 $\tau_Z$ 或绕过 $V^{PG}$。

D0 不进入 C0–C6 的重建性能表。

### 11.2 严格嵌套的 C0–C6

| 阶段 | 唯一新增能力           | 执行动作                                                 |
| -- | ---------------- | ---------------------------------------------------- |
| C0 | AmbiSuR baseline | 原始训练                                                 |
| C1 | 需求门控             | $\lambda_{P,i}=\lambda_{P,0}N_i$，尚无双可靠性选边            |
| C2 | 粗粒度强制仲裁          | 对 xyz+rotation+scaling+opacity 整组执行选边/融合，不允许 Abstain |
| C3 | 粗粒度拒绝仲裁          | 在 C2 上加入 Abstain，几何可冻结、外观继续训练                        |
| C4 | 参数组级路由           | 按第 7.3 节分别路由四个参数组，使用未投影梯度                            |
| C5 | 梯度冲突投影           | 仅在 C4 上加入第 7.4 节投影                                   |
| C6 | 生命周期             | 在 C5 上加入 split/clone/protect/quarantine/prune 控制     |

其中 C1 的需求门控为：

$$
\lambda_{P,i}^{C1}=\lambda_{P,0}N_i,\qquad
g_i^{geo,C1}=g_{B,i}^{geo}+\lambda_{P,i}^{C1}g_{P,i}^{geo},
$$

其中 $\lambda_{P,0}$ 沿用 baseline 先验权重，$geo=\{pos,scale,opa\}$。C2 把这些参数作为一个整体，执行无 Abstain 的粗粒度仲裁：

$$
g_i^{geo,C2}=
\begin{cases}
g_{B,i}^{geo}, & N_i\le\tau_N,\\
g_{B,i}^{geo}+0.5g_{P,i}^{geo},
& H_i^P=H_i^G=1,\ V_i^{PG}=1,\ K_i\ge\tau_K,\\
g_{P,i}^{geo},&H_i^P=1,\ H_i^G=0,\\
g_{B,i}^{geo},&H_i^P=0,\ H_i^G=1,\\
g_{P,i}^{geo},&H_i^P=H_i^G=1,\ V_i^{PG}=1,\ K_i<\tau_K,\ \Delta_i>0,\\
g_{B,i}^{geo},&\text{otherwise}.
\end{cases}
$$

C2 无拒绝能力；打平、两路均不可信，或双方均可靠但 $V_i^{PG}=0$ 时都落入 `otherwise`，回退基础梯度。joint-invalid 时不得用 $\Delta_i$ 选边。C3 首次加入 Abstain，其完整公式为：

$$
g_i^{geo,C3}=
\begin{cases}
g_{B,i}^{geo},&N_i\le\tau_N,\\
g_{B,i}^{geo}+0.5g_{P,i}^{geo},&H_i^P=H_i^G=1,\ V_i^{PG}=1,\ K_i\ge\tau_K,\\
g_{P,i}^{geo},&H_i^P=1,\ H_i^G=0,\\
g_{B,i}^{geo},&H_i^P=0,\ H_i^G=1,\\
g_{P,i}^{geo},&H_i^P=H_i^G=1,\ V_i^{PG}=1,\ K_i<\tau_K,\ \Delta_i>\delta,\\
g_{B,i}^{geo},&H_i^P=H_i^G=1,\ V_i^{PG}=1,\ K_i<\tau_K,\ \Delta_i<-\delta,\\
0,&\text{otherwise}.
\end{cases}
$$

因此双方均可靠但 $V_i^{PG}=0$ 时，C3 落入最后的 Abstain。C4–C6 继承 C3 已产生的状态，不得自行绕过 $V_i^{PG}$ 重新解释历史 K 或 $\Delta_i$。所有 C1–C3 都始终允许 $g_i^{SH}=g_{B,i}^{SH}$ 及曝光基础梯度。由此 D0 只观察；C2 已有整组执行器；C3 增加拒绝；C4 细化参数组；C5 才投影；C6 才改变拓扑。

### 11.3 支撑模块消融

以 C6 作为 Core，逐项增加且逐项留汰：

1. Core + decoupled exposure；

2. 上一步 + local Ray-Color；

3. 上一步 + local Ray-Normal；

4. 上一步 + state-adaptive truncation，形成 Full。

每个模块最多允许“默认配置、一次有依据调整、一次跨场景确认”三次验证。没有稳定信号就删除，不为了系统完整而保留。

### 11.4 E0–E4 证据漏斗

| 层级 | 目的       | 数据与重复                                                     | 晋级结果              |
| -- | -------- | --------------------------------------------------------- | ----------------- |
| E0 | 工程等价性    | Tool Room，小步/固定 seed                                      | 关闭新功能时复现 baseline |
| E1 | 机制诊断     | Tool Room seed 0，D0                                       | 确认需求与可靠性具有预测价值    |
| E2 | Core MVP | Tool Room seed 0，C0/C1/C3/C5/C6 快速筛选                      | Core 有几何信号且训练稳定   |
| E3 | 反光场景主实验  | Tool + Utility；Baseline/Core/Full 各 3 seeds；完整 C0–C6 关键消融 | 支撑核心论文结论          |
| E4 | 泛化与外部比较  | DTU/TnT 场景级运行；可比方法                                        | 证明不只适配两个反光场景      |

随机种子是同一场景×方法的优化重复；场景才是泛化比较的主要独立单位。像素、Gaussian 或 mesh 顶点不能被当成独立样本扩大 $n$。

## 12. 数据、指标与可视化

### 12.1 ScanNet++

* 场景：Tool Room、Utility Room；采用相同输入视图、相同 mesh 提取和相同官方评价协议。

* 几何：优先报告官方可复现的 accuracy/completeness/Chamfer/F-score 或项目实际提供的等价指标。

* 外观：PSNR、SSIM、LPIPS 作为守门指标，而不是唯一优化目标。

* 效率：训练时间、峰值显存、最终 Gaussian 数量和 mesh 提取时间。

* 机制：状态覆盖、可靠性风险曲线、梯度冲突率、生命周期动作数。

不做人工高光 mask。反光/细结构部位只使用预先固定的相机视角和 crop，展示 mesh、normal、depth 与误差图的定性对比；禁止结果出来后挑选最有利视角。

### 12.2 DTU 与 Tanks and Temples

* DTU 使用 AmbiSuR 官方 15 个 scan 和官方 Chamfer、data-to-surface、surface-to-data 统计。

* TnT 使用官方 6 场景配置和 F-score、precision、recall；保留各场景专用配置。

* 先运行 Baseline/Core/Full；只有 Core 通过反光场景门槛后才投入完整泛化成本。

### 12.3 外部比较

主要 mesh 可比对象优先为 2DGS、PGSR、AmbiSuR、CoMe 和 Ours。DirectFisheye-GS/3DGUT 等以 NVS 为主的方法可用于数据选择或相关工作讨论，但不强行进入主要 mesh 表。

## 13. 成功标准与止损门槛

### G0：工程等价性

* 所有新功能关闭时，损失、Gaussian 数量、训练曲线和最终指标与原 baseline 在合理随机波动内一致；

* 不出现 NaN/Inf、显存持续增长或额外 backward 重复注入梯度。

#### 非确定 baseline 的数值等价合同（2026-09-04 已批准方案 A）

本澄清将上述“合理随机波动”落实为可审计的工程验收规则，不改变训练算法、先验合同或 C0–C6 的唯一变量。现有 500 轮三方结果是选择规则时已看过的**探索性标定证据**，不能作为预注册的确认性检验，也不能单独宣布 G0 通过。

在同一验证时长 $h$，记 $B_h^{(1)},B_h^{(2)}$ 为两次独立启动的 exact-baseline 运行，$E_h$ 为一次独立启动的 E0 feature-off 运行。三次必须使用相同 GPU/runtime、canonical data snapshot、语义 seed、分辨率和训练配置；仅允许事先声明的 commit、输出/private-view 路径及 E0 元数据参数差异。

**严格不变量不使用噪声容差：** 输入与初始化 prior hash、共同有效配置、feature-off legacy dispatch、seed/RNG 与相机采样轨迹合同、checkpoint schema/字段/dtype/shape、Gaussian 数量、optimizer groups/hyperparameters/state keys/step counters，以及该验证时长内未激活或未更新的字段，均须严格一致。每个运行的 commit 必须分别匹配其批准的 exact SHA，而非要求 baseline/E0 commit 相同。缺失必需证据不能算通过。500 轮的 SH rest/app 等未激活字段 exact，不意味着它们在 8k 激活后仍归入未激活字段。

对其余有限且同形状的已更新参数张量、densification proxy 和 optimizer moments，按相同字段及存储顺序分别计算：

$$
D_{\mathrm{RMSE}}(x,y)=\sqrt{\frac{1}{n}\sum_{j=1}^n(x_j-y_j)^2},
\qquad
D_{\mathrm{MAE}}(x,y)=\frac{1}{n}\sum_{j=1}^n|x_j-y_j|.
$$

对每个字段 $x$ 和每一种距离 $D$，定义：

$$
d_B=D\!\left(x(B_h^{(1)}),x(B_h^{(2)})\right),
\qquad
d_E=\min_{k\in\{1,2\}}D\!\left(x(E_h),x(B_h^{(k)})\right).
$$

固定接受条件为：

$$
\begin{cases}
d_E=0\ \text{且字段 exact},&d_B=0,\\
d_E\le 2d_B,&d_B>0.
\end{cases}
$$

RMSE 与 MAE 必须分别通过；两种距离各自取最近 baseline，不以跨字段平均掩盖失败。对预先指定的每个标量 loss/评价指标，使用绝对差作为 $D$，沿用同一规则。保留三组 pairwise 距离和最近参照身份，以便审计。`max_abs`、mismatch count 和已学习结果文件的 SHA 仅作诊断，不单独决定数值等价；输入/prior 的 SHA 仍属于严格不变量。

在后续独立 **8k 三次运行（baseline、baseline repeat、E0 all-off）** 开始前冻结本规则。8k 使用其自身同 horizon 的 baseline self-distance，不沿用 500 轮的绝对误差数值；必须覆盖 densify 首次 600、multi-view trim 首次 1000、Ray-Color 首次 5001、ALR 首次 7001 的实际 baseline 分支。启动命令、比较字段/评价点、日志与资源证据清单须在运行前固定。

只有严格不变量、所有数值门与安全门均通过，才可接受该确认性验证。若 Gaussian 数量/shape 不一致、某字段超界、$d_B=0$ 却不 exact、出现 NaN/Inf、额外重复 backward 或显存持续增长，立即停止并报告；不得删字段、截断/补齐张量、事后选参照组合或提高系数来通过。该 2 倍规则是工程验收合同，不是统计置信区间，也不证明不同环境/场景下的普遍等价。

本合同只处理独立训练轨迹的 G0 验收，**不替代或放宽 C1 的同一状态 GPU 单步 residual decomposition gradient oracle**。本次批准仅同步规格与规划文档；8k 启动、D0/C1、方法源码和 tag 仍须遵守各自授权关口。

未通过 G0 不得解释任何方法收益。

### G1：诊断有效性

* $N$ 对高几何误差的 AUROC > 0.60；

* 相比 $A$ 或 $1-S$ 中较好的单一组成至少提高 0.03；

* 双可靠性与 coverage-risk 呈合理单调关系；状态不是被 Bypass 或 Abstain 单一状态完全占据。

若 $N$ 无增益，停止扩展完整生命周期，先回到观测充分度定义。

### G2：Core 几何收益

* Tool Room 与 Utility Room 两场景平均核心几何指标改善约 3% 或以上；

* 任一场景相对 baseline 的系统性退化不超过约 1%；

* PSNR 下降不超过 0.3 dB，LPIPS 恶化不超过 0.01；

* Core 训练时间不超过 baseline 约 2 倍，峰值显存建议低于 22 GB。

若仅 Tool Room 有效而 Utility Room 无效，论文主张降级为场景特定发现，不继续堆叠配套模块掩盖问题。

### G3：支撑模块留汰

每个增强必须在至少一个几何或稳定性指标上带来可重复收益，同时不破坏 G2 的外观和效率守门条件。三次验证后仍无稳定信号即删除。

### G4：泛化

Core/Full 在 DTU/TnT 上不得出现跨多数场景的系统性退化。论文报告逐场景配对效应、均值/标准差或置信区间；预先指定核心比较，其余多模块比较标为探索性。

## 14. 失败模式与应对

| 失败模式                   | 诊断                        | 预定应对                               |
| ---------------------- | ------------------------- | ---------------------------------- |
| 状态快速抖动                 | 转移矩阵、持续时间                 | 提高 EMA/迟滞，不增加新证据源                  |
| 几乎全部 Bypass            | $N$ 分布                    | 检查观测覆盖归一化与 $\tau_N$ 标定             |
| 几乎全部 Abstain           | $V,K,\Delta$ 分布           | 检查证据有效条件是否过严；不直接降低所有阈值             |
| 先验压平细结构                | Prior-led 区域、scale/normal | 降低先验 scale 路由，增强 Protected/Abstain |
| opacity 补偿小截断          | opacity 与 trunc 联合曲线      | 对缩半径状态抑制 opacity-logit 负梯度         |
| Ray-Color 改坏几何         | xyz 梯度审计                  | 切断 SH 视角方向到 means3D 的梯度泄漏          |
| 生命周期复制错误               | split 前后状态追踪              | 延长 Repair/Probation 冷却；坚持两阶段提交     |
| Full 优于 Core 但 Core 无效 | C0–C6 结果                  | 不把配套增强包装成仲裁贡献，重新评估主线               |

### 14.1 已核对的实现级踩坑

1. `compute_weighted_sh_norm` 当前写死 15 个非 DC 系数，只适配 SH degree 3；必须按当前阶数动态读取。

2. `out_observe` 是 $T>0.5$ 时的单视图像素命中数，不是视图数；只能先用 `>0` 二值化。

3. 当前曝光 SSIM 使用 `app_image_detach`；新版亮度项不得 detach `I_app`，否则曝光收不到 SSIM 梯度。

4. 当前 Ray-Color 在 CUDA backward 隐式注入，并可能因 ALR 额外 backward 重复；新版必须显式 loss、一次参数更新。

5. SH 视角方向依赖 `means3D`，Ray-Color 可能泄漏到 xyz；必须 stop-gradient 方向与合成权重。

6. 布尔切片操作 `requires_grad_` 不能实现逐 Gaussian 冻结；必须直接 mask 已分离的参数梯度。

7. 当前 `trunc_sigma` 是 CUDA 全局标量；必须改为逐 Gaussian Tensor，forward/backward 使用同一 $\gamma_i$。

8. 截断缩小时应抑制 opacity logit 的负梯度，因为梯度下降中负梯度才会增大 opacity。

9. densify/prune 会替换 Parameter；必须迁移 Adam `exp_avg`/`exp_avg_sq`、EMA、状态、持续计数和截断半径；新点按 Probation 初始化。

10. DA3 已参与初始化/监督，两路只能称“双证据通道”，不能声称统计独立。

11. GT mesh 只允许 D0 标签生成和最终评价读取；训练、可靠性缓存与 checkpoint 禁止包含其派生量。

## 15. 实施边界与版本控制流程

正式实现应按独立功能开关组织，使所有实验能从同一代码基线产生：

* `enable_observation_calibration`

* `enable_dual_reliability`

* `enable_abstention`

* `enable_parameter_routing`

* `enable_gradient_projection`

* `enable_reliability_lifecycle`

* `enable_decoupled_exposure`

* `enable_local_ray_color`

* `enable_local_ray_normal`

* `enable_state_truncation`

每个开关关闭时必须回退到上一消融阶段，不能留下隐式状态影响。服务器运行配置、Git commit、数据场景、seed、命令、环境和输出目录写入实验 manifest；结果回传本地后只做只读统计。

实现上拆成 `EvidenceAccumulator`、`GradientRouter` 和 `LifecycleManager` 三个边界清晰的组件。densify/prune 后由 LifecycleManager 返回旧到新索引映射，再统一迁移其他状态，避免统计数组与 Gaussian 参数错位。

## 16. 第二次审核结果

本稿已针对以下问题重写并检查：

* **公式依赖闭包**：训练量均来自第 3 节已知输入、model/renderer 输出、前式派生量或已声明固定超参数；GT mesh 是离线量。

* **公式可读性**：Ray-Normal 已拆成 Gaussian 法线、面向相机符号、前表面集合、alpha 混合法线和平方 L2 损失。

* **CoMe 一致性**：Ray-Color 已从错误 L1 改回平方 L2；Ray-Normal 已从单位均值余弦式改回 alpha 混合法线平方 L2；本文新增点是局部集合、可靠门控和梯度去向。

* **曝光完整性**：已展开 SSIM 的 $l,c,s$ 与均值、方差、协方差，并明确曝光梯度边界。

* **生命周期递推**：错误计数式已替换为条件成立加一、失败清零的分段式，并加入冷却。

* **状态有无执行器**：D0 明确只诊断；C2 起已经实际改变几何梯度。

* **C4/C5 是否混淆**：C4 只直接加权，C5 才投影对抗分量。

* **无证据是否被当作低置信**：通过 $T,V$ 分离解决。

* **需求、一致性与可靠性是否遗漏**：仲裁显式使用 $N,V,K,\Delta$。

* **不可逆动作是否过早**：生命周期使用持续证据与两阶段提交。

* **反光区域主张是否越界**：无人工 mask，不做反光区域定量因果主张。

* **配套模块是否抢占创新**：曝光、Ray 方差、截断均单列为支撑机制并设置止损。

* **算力是否超出单卡范围**：首版仅两条梯度路径，不引入学习式置信网络和多卡依赖。

### 16.1 用普通中文说明各计算环节

| 环节      | 已知输入                                       | 本环节做什么                                                   | 得到什么                                              | 后续用途                     |
| ------- | ------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------- | ------------------------ |
| 外观歧义    | 当前 Gaussian 的非 DC 球谐系数                     | 计算非 DC 球谐能量，再用全部活动 Gaussian 的第 10% 和第 95% 分位归一化          | 外观歧义强度 $A_i$                                      | 判断充分观测后是否仍存在明显视角相关外观     |
| 观测充分度   | `out_observe`、相机中心、Gaussian 中心&#x20;       | 统计看见该 Gaussian 的相机数量，并检查这些相机的观察方向是否足够分散                  | 观测充分度 $S_i$                                       | 区分“真正歧义”和“只是还没看清”        |
| 先验需求    | 外观歧义强度 $A_i$、观测充分度 $S_i$                   | 按第 4.3 节合并二者                                             | 先验需求度 $N_i$                                       | 决定该 Gaussian 是否需要进入强先验仲裁 |
| 外部可靠性   | DA3 深度、DA3 confidence、多视图重投影、Gaussian 合成权重 | 评价 DA3 自身置信度和跨视图一致性，并检查有效支持视图是否足够                        | 外部可信度 $T_i^P$、有效标记 $V_i^P$、可靠性 $r_i^P$            | 判断 DA3 是否有资格主导几何更新       |
| 内部可靠性   | 当前渲染深度、两种渲染法线、多视图重投影、历史位置和法线               | 评价当前 Gaussian 几何的跨视图一致性、法线一致性和历史稳定性                      | 内部可信度 $T_i^G$、有效标记 $V_i^G$、可靠性 $r_i^G$            | 判断当前重建是否有资格拒绝外部先验        |
| 双方一致性   | DA3 深度/法线、当前渲染深度/法线                        | 先用 $Z_i^{PG},V_i^{PG}$ 验证共同支持，再比较同像素深度和法线差异                    | joint validity 与一致性 $K_i$                         | 区分双方一致、双方冲突与 joint-invalid |
| 可靠性优势   | 外部可靠性 $r_i^P$、内部可靠性 $r_i^G$                | 用外部可靠性减去内部可靠性                                            | 可靠性差 $\Delta_i$                                   | 冲突时判断哪一方有明显优势            |
| 五状态仲裁   | $N_i$、两路可信度与有效标记、$V_i^{PG}$、$K_i$、$\Delta_i$ | 按第 6 节的完整优先级判断                                           | Bypass、Consensus、Prior-led、Geometry-led 或 Abstain | 决定每个 Gaussian 应采取哪种训练动作  |
| 参数执行    | 当前状态、基础梯度、先验梯度                             | 对位置/旋转、尺度、opacity、SH 分别保留、冻结、加权或投影梯度                     | 最终参数梯度                                            | 执行一次 optimizer 更新        |
| 生命周期与截断 | 稳定后的仲裁状态、持续计数、冷却计数                         | 控制 clone、split、protect、quarantine、prune 和逐 Gaussian 截断半径 | Gaussian 的连续更新与离散拓扑动作                             | 完成从“证据判断”到“实际训练行为”的闭环    |

换句话说，整个方法按以下顺序工作：先判断“这个 Gaussian 是否真的需要帮助”，再分别判断“外部先验是否可信”和“当前几何是否可信”，随后决定相信谁或暂时拒绝，最后才改变梯度、增密、剪枝和截断。

所有超参数初值写在对应公式附近，并统一遵守“D0 最多一次有依据调整、随后跨场景冻结”。尚需在实施计划中落到代码文件、Tensor 接口、单元测试、服务器命令和预计运行数量；在用户终审前不修改其本地结果或 AmbiSuR 工作副本。
