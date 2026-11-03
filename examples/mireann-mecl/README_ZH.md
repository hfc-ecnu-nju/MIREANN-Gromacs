# MECL 溶液的 MIREANN 接口示例

来自用户的 MECL.prmtop/MECL.rst，使用七输入 MIREANN 电荷/静电模型，
气相 PES 使用所提供 MIREANN 仓库 MECL 示例中的 REANN_PES_DOUBLE.pt。
QM 是全局 1–6 号原子（Cl Cl C H H H），中心 1，球半径 25 Å。
GMX_MIREANN RESIDUE：默认按残基选区，任一 MM 原子入球则整个 MM 残基纳入；
保持残基周期镜像完整，外围原子可超过 25 Å。每次 force 调用重选，硬切换仍可能引入能量跳变。
模型坐标 Å、电荷 e、能量 kcal/mol；MDP 中转换系数 4.184。

在本目录执行：
```bash
gmx grompp -f smoke.mdp -c start.gro -p system.top -n qm.ndx -o smoke.tpr
gmx mdrun -deffnm smoke -ntmpi 1 -ntomp 4 -nb cpu -pme cpu -update cpu -notunepme
```

CUDA 构建可使用 `-nb gpu -pme gpu -update cpu`。GPU 推理需要 CUDA 版 LibTorch，
同时把 `mireann-device` 和 `nnpot-reann-device` 改为实际设备，例如 `cuda:0`。
建议保留 `-notunepme`，固定这套分段扣除使用的经典 cutoff 和 Ewald 参数。
没有全 GPU 驻留：选区、扣除和主机力累加保留 CPU 路径。单 rank 是当前限制。
这是短接口测试，不是已完成平衡的生产模拟方案；请自行设置后续平衡及生产参数。
