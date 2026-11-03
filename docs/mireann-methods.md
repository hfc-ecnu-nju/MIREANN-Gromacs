# MIREANN：能量、力、电荷与周期边界

本文件说明本分支实际采用的计算定义，而不是把它称为通用 QM-PME。
接口中的 QM 表示模型原子区，不意味着运行时启动 DFT。

## 能量

对于同时开启 REANN 气相 PES 和七输入 MIREANN 的单端点：

\[
U(R)=U_{\rm FF}^{\rm kept}(R,q(R))+E_{\rm PES}(R_Q)
     +E_{\rm EE}(R_Q,R_M,Q_M)-E_{\rm ref}(R,q(R))+U_{\rm bias}.
\]

`FF kept` 是经过 NNP 拓扑处理后保留的经典项；完全 QM 内部的被替换项不能再算一次。
`E_EE` 是七输入模型能量，不含独立气相 PES。
接口在 NN-Potential 项加入 EE 减参考项，PES 由原生 NNP provider 累加。
限制势由原生 GROMACS 计算，不重复加到模型能量。

经典整体静电使用更新后的物理电荷及 GROMACS PME。局部参考只作用于所选单份
image 中的 QM–MM 对，**不是删除这些原子所有周期镜像的完整 Ewald 相互作用**。
不被拓扑排除的普通对采用：

\[
E_{\rm ref}^{\rm ordinary}=C\sum_{i\in Q,j\in\Omega}
w_j q_i Q_j K(r_{ij}),\qquad
K(r)=\begin{cases}r^{-1}-c_{\rm shift},&r\le r_c,\\
\operatorname{erf}(\alpha r)/r,&r>r_c.
\end{cases}
\]

这里距离是**模型输入那份 image**的距离；`r_c` 是经典 Coulomb cutoff，
不是选区球半径。`C`、`alpha`、`c_shift` 来自本次实际 PME 参数。
`coulomb-modifier=None` 时 `c_shift=0`，Potential-shift 时使用匹配的实空间位移。
模型范围之外及其余周期贡献继续由整体经典静电保留。
该分段在 cutoff 有被截去的 erfc 小尾项，选区/image 变更也需要单独考虑连续性。

外段力包含高斯导数：

\[
F_i=Cq_iQ_j\left[\operatorname{erf}(\alpha r)
-\frac{2\alpha r}{\sqrt\pi}e^{-\alpha^2r^2}\right]
\frac{R_i-R_j}{r^3}.
\]

跨区 1–2/1–3 排除不在普通参考中扣除；普通跨区 1–4 静电另按
`C*fudgeQQ*qi*Qj/r` 扣除，LJ-14 保留。显式固定电荷 pair 函数涉及 QM 时拒绝。
具体边界键合规则见 [赝键说明](mireann-pseudo-bond.md)。

## 电荷提交和剩余响应

初始化一次对**全部非 QM 原子**补偿：

\[
\Delta Q=\frac{\sum_{i\in Q}q_i^{\rm TOP}-Q_{\rm model}}{N_{\rm MM}},
\qquad Q_j=Q_j^{\rm TOP}+\Delta Q.
\]

包括球外与 zero-mask；后续不重复补偿。已补偿 MM 电荷按全局编号保存，在原生
mdatoms 重建后恢复同一份数值，防止退回原 TOP 值。每步把模型 QM 电荷提交给 chargeA/B，
刷新电荷和、平方和、短程打包电荷和 PME 输入。模型总电荷固定且检查有限性、容差。
zero-mask 仅从**模型背景列表**排除，物理 FF 电荷及参考扣除继续使用补偿后的值。
双哈密顿 MM 端例外：它完全使用原 TOP 电荷，不执行这份 ML/MM 补偿。

模型自身能量的自动微分已包含其内部电荷响应。模型之外的残余项另需：

\[
\phi_i=\left.\frac{\partial(U_{\rm FF}^{\rm kept}-E_{\rm ref})}{\partial q_i}\right|_R,
\qquad
F_a^{\rm response}=-\sum_{i\in Q}\phi_i\frac{\partial q_i}{\partial R_a}.
\]

`phi` 包括保留下来的整体周期贡献及对应自能、净电荷背景、排除/1–4 修正，
不是只有球内电势。PME 网格电势复用 GROMACS 本步已经求解的网格，额外核插值 QM 电势。
VJP 将本步 `phi` 当常量，对 `scale*E_EE + sum(q*phi)` 一次反向传播，
替代模型原返回梯度，避免把同一模型能量梯度加两次。
辅助标量 `sum(q*phi)` **不计入物理能量**，也不对 phi 再求坐标导数。

这里 R 是 QM 及已输入模型的 MM 坐标。周期镜像共用同一物理原子的电荷；其余 image
通过 phi 影响响应系数，不要求模型看到所有 image。固定 image 分支上平移的雅可比为单位阵，
中心平移的反作用力也须保留。未输入模型的 MM 坐标对模型电荷没有该导数，
但仍有使用新 QM 电荷计算的普通经典静电力。

`mireann-charge-response=no` 仅用于与省略该残余响应的旧实现对照；它不改变模型能量、
预测电荷或模型自身梯度，不是当前能量定义下完整保守力的模式。

## 残基选区、image 与可选平滑

指定中心是 QM 内一个**全局 1-based 原子**。任一 MM 原子进入外球，整残基全部纳入；
以最近入球原子为锚，完整残基进行 PBC image。边缘成员可以超出球半径，每个物理原子只取一份。
使用真实拓扑残基实例，不依赖 GRO 的循环显示编号。每步重新判断最终选区和 image。
30 Å 候选表不等于将 30 Å 环境全部传入模型；位移/盒变化可提前重建，20 步不是绝对保证有效期。

默认 `switch-width=0` 为硬选区。可选整残基平滑使用每成员的 C2 径向函数：

\[
s_a=\begin{cases}1&r_a\le R_{out}-d,\\
10u_a^3-15u_a^4+6u_a^5& R_{out}-d<r_a<R_{out},\\
0&r_a\ge R_{out},\end{cases}
\quad u_a=(R_{out}-r_a)/d,
\quad w_g=1-\prod_{a\in g}(1-s_a).
\]

同一残基共用权重，避免最近原子改变时直接对 min 距离求导引入折点。
模型背景为 `w_g*Qj`，匹配参考也乘 `w_g`；**不是把整个模型输出能量乘一次 w**。
物理 FF 电荷不乘这个权重。补充链式力：

\[
D_g=\sum_{j\in g}\left[Q_j\frac{\partial(E_{EE}+q^T\phi)}{\partial(w_gQ_j)}
-E_{{ref},j}^{\rm unweighted}\right],\qquad
\Delta F_a=-D_g\nabla_a w_g.
\]

能量单位缩放在代码中统一为 kJ/mol；所有成员梯度及中心的反作用力均累加。
跨区 1–4 原子要求位于单位权重内核。平滑可改变有效模型输入分布，是实验选项；
此前 50 ps MECL 测试仍有漂移，不能称为已解决长时间守恒。

## 单位与精度

GROMACS：nm、ps、kJ/mol、kJ/(mol nm)、e。
模型：Å、e；能量单位由用户显式声明。kcal/mol 模型能量乘 4.184，坐标梯度转力乘 −41.84。
当前 MIREANN 后端要求 float64 参数和输入。GPU 混合精度 GROMACS 的物理电荷、
经典坐标/力缓冲仍可为 float32；提交到经典计算时会取其可表示值，模型本身不因此转为 float32。
不同模型的 cell/image 约定和训练分布必须核对；接口不能保证任意 .pt 可互换。
