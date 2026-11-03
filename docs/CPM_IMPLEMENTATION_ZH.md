# 论文版动态电荷如何实现

对照材料：本地 `Gromacs_CPM-main/Gromacs_CPM-main/gromacs` 和 [论文 v1](https://arxiv.org/html/2610.15733v1)。本地用户指南明确写其基于 GROMACS 2020。

## 求的不是 ML 电荷，而是恒电势电极电荷

每个电极原子是幅值 q_i 可变的高斯电荷，其宽度由 eta 控制；电解液保持点电荷。电极上的电势分为电极自身和电解液两部分：

`phi_i = sum_j A_ij q_j + phi_sol,i`。

固定电势要求每个电极原子满足目标电势，所以逐步求：

`A q = U_set + lambda - phi_sol`。

lambda 用于总电荷约束。A 包含高斯短程核、Ewald 倒空间、自能、可选 slab 3DC 修正，还可加 Thomas–Fermi 对角项。固定总电荷模式中，电极电势成为待求的拉格朗日乘子；并非给每个原子均分总电荷。

## 一步内的工作

1. 使用 SIMD/CUDA 非键 kernel 累加电解液对每个电极原子的短程电势，同时加入高斯–点电荷短程修正。
2. 将电极电荷暂置零，运行电解液-only PME，在电极位置 gather 倒空间电势，组成 phi_sol 和方程右端。
3. 用矩阵逆乘、CG 或扩展拉格朗日得到电荷；固定电极的矩阵可预计算，可移动电极通常用 matrix-free CG。
4. 用一致电荷计算正常短程力和 PME 倒空间力。GPU 路径有额外 PME pass；matrix-free CG 每次 A*v 还需要电极-only PME。

**冻结电极与移动电极时序不同。** 论文的冻结电极路径使用预测电荷计算本轮力，新求解电荷进入下一步外推：

`q_pred(n+1) = 2 q_solved(n) - q_solved(n-1) + 0.5 delta_q(n)`，

其中 delta_q 是当步求解值与预测值之差。源码中可见 `gamma` 误差反馈及 `q_Nm1`、`cpm_qPredicted`。同一步短程和倒空间必须使用相同 q，代码会暂存/恢复 `qUsedBySr` 与 `qSolved`。论文的移动电极 CUDA 路径则先解电荷，再将本次解用于两个力路径，避免冻结电极外推方案直接用于移动电极。

扩展拉格朗日方案给电荷引入虚拟质量和动力学，不是每次严格求逆；不能与 BO/CG 求解或外推混为一谈。

## 源码入口（相对论文版 gromacs/）

| 文件 | 作用 |
|---|---|
| `src/gromacs/mdlib/force.cpp` | 组装电势、求解/预测调度、保证 SR/LR 电荷一致；约 1253–1347 行为外推与误差反馈 |
| `src/gromacs/mdlib/sim_util.cpp` | 约 1244–1258 行刷新 NBNxM 电荷，邻居表重建与步间分别走全量/电极区域更新 |
| `src/gromacs/nbnxm/nbnxm.cpp` | `updateAtomCharges` / `updateElectrodeCharges` |
| `src/gromacs/nbnxm/atomdata.cpp` | 实际写入打包的短程电荷数据 |
| `src/gromacs/ewald/cpm_matrix.cpp` | 电极矩阵和相关求解逻辑 |
| `src/gromacs/ewald/cpm_gpu_cg.cu` | GPU matrix-free CG |
| `src/gromacs/ewald/cpm_gpu_solve.cu` | GPU 稠密求解操作 |
| `src/gromacs/ewald/pme.cpp`、`pme_gpu.cpp` | CPM PME 电势/电荷 pass 的扩展 |
| `src/gromacs/ewald/cpm_potential_control.cpp` | 电势/电荷控制协议 |

本地指南规定电极是最前面的 `userint1` 个原子，两电极默认各占一半；`userint2=0/1/2` 分别选逆矩阵/稠密CG/matrix-free CG，`userreal1` 为高斯宽度，`userreal2` 为电势差；具体组合参见原指南。它不是任意索引区域的 ML 接口。

## 对本次修改的启发

可以借鉴的是“逐步生成电荷 → 正确传播到全部力路径”的机制。不能直接把该版本整套移植后把求解器换成 ML，就认为高斯核、恒电势约束和 ML/MM 的物理模型一致。

本次在 2026.4 中保留标准 MM 点电荷核，让 `predictCharges()` 负责输出指定区域电荷；其余数组同步由引擎修改承担。ML 势能/力和电荷响应导数仍需后续接入。

GPU 扩展复用了论文版两条传播机制：短程先更新主机打包电荷，再走 `gpu_copy_xq_to_gpu()`；PME 适配 `pme_gpu_cpm_copy_charges_impl()` 的锁页双缓冲与异步完成事件。新接口叫 `gmx_pme_update_dynamic_charges()` / `pme_gpu_update_dynamic_charges()`，在 2026.4 中使用 `d_coefficients[0]` 和 `pmeStream_`。没有搬入电极高斯核、两次 PME 求电势流程或恒电势求解器，因为新电荷由外部 ML 接口提供。

后续性能优化进一步采用了论文版 `updateElectrodeCharges` 的区域更新思路：2026.4 的 `updateRegionCharges()` / `nbnxn_atomdata_update_region_charges()` 使用 `GridSet::cells()` 的逆映射，只覆盖指定全局选区所对应的本地打包电荷，并检查逆映射一致性；不在每一步重设 LJ 参数、原子类型或能量组。这是针对 2026.4 数据结构重新实现的适配，不是直接复制旧版函数体。
