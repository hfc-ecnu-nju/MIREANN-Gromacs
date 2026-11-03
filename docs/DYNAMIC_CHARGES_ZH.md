# GROMACS 2026.4 指定区域逐步电荷接口

> 当前 QEq 模型的集成构建和 MDP 配置见源码顶层 `QEQ_REANN_README_ZH.md`。下文环境变量接口为保留的早期接口及确定性回归测试方式，不是新用户的必需步骤。

这是在用户提供的 2026.4 上直接修改的本地扩展，不是移植整个 GROMACS 2020 CPM。

## 源码改动

- `src/gromacs/mdlib/dynamicchargeprovider.h`：区域读取、预测接口、输出校验，以及每个模拟独立持有的运行缓存。
- `src/gromacs/mdlib/sim_util.cpp`：在 `do_force()` 的力计算开始前调用预测；写入 `chargeA/chargeB`，更新 `qsum/q2sum`，刷新 NBNxM 打包电荷。CPU PME 直接使用更新后的数组；偶极矩计算在更新之后执行。
- `src/gromacs/mdlib/forcerec.cpp`：启用时为所有原子保留 HasCharge 标记，避免初始零电荷原子的库仑 cluster pair 被优化删除。该标记必须在初始 DD 网格构建前设置。
- `src/gromacs/mdtypes/forcerec.h`：由当前模拟的 `t_forcerec` 拥有缓存，避免多个模拟共享进程级静态状态。
- `src/gromacs/nbnxm/atomdata.*`、`nbnxm.*`：在非邻居表重建步，通过当前网格逆映射只刷新选区电荷；保留已有 LJ、类型、能量组和填充电荷。重建步仍走原有完整设置流程。
- `src/gromacs/ewald/pme_gpu_internal.cpp` 和相应头文件：适配论文版 `pme_gpu_cpm_copy_charges_impl` 的锁页双缓冲、异步 PME-stream 上传与完成事件；新增只刷新电荷系数的入口，不逐步重初始化 PME。
- `src/gromacs/taskassignment/decidesimulationworkload.cpp`：动态电荷自动关闭 GPU x/f buffer ops，使标准 `gpu_copy_xq_to_gpu()` 每步携带新电荷；显式拒绝 GPU update 和直接 GPU 通信。

GROMACS 2026 即使单 rank 也可能按 DD 顺序重排原子；这里用 `globalAtomIndices` 将本地数据还原成拓扑全局顺序，再将区域预测映射回本地。区域无需连续，也无需位于坐标文件开头。

运行缓存复用 Frame 的分配空间，在邻居表重建、DD 分区计数变化或本地原子数变化时重建编号映射。每次调用仍用当前坐标和电荷填充整个 Frame；大体系使用 OpenMP 并行填充，不减少预测频率或丢弃全体系输入。选区外电荷固定，其 qsum/q2sum 贡献只在重建时重算；随后每次把选区的新电荷贡献重新相加，而不是累计每步差值。

优化版大体系复测和速度范围见 [性能优化与复测](DYNAMIC_CHARGES_OPTIMIZATION_ZH.md)。当前未计入真实 ML 推理的占位接口中位速度为 103.608 ns/day（41,472 原子）和 32.982 ns/day（139,968 原子）；选区均为 300 原子，动态路径仍使用 CPU 积分。

## 调用顺序与单位

常规 leap-frog：上一轮积分完成 → 当前坐标 x_n → `predictCharges(frame, ...)` → q_n → 本轮所有力 → 下一轮积分。

首次力计算也调用接口，为初始坐标准备电荷。它并非“先用旧电荷算本轮力、算完再覆盖数组”；预测发生在相邻 MD 积分之间，下一次力计算立即使用新电荷。接口按力计算调用，不按输出频率或邻居表频率调用。

`Frame` 包含 step、timePs、boxNm、全体系 positionsNm、chargesE、atomTypes，以及 region。atomTypes 是力场类型索引，不是元素原子序数；模型需要预先配置类型/全局编号与元素的对应表。内部 region 为 **0-based 全局拓扑编号**；位置和盒子为 nm，时间 ps，电荷单位 e。位置可能是不同周期镜像；模型需要用盒子正确处理 PBC。没有假设坐标已经展开。

替换 `predictCharges()` 中默认 identity 分支，即可接 C++/LibTorch/ONNX 等推理。返回值长度必须等于区域原子数，顺序与 region 一致。默认 identity 只用于接口连通检查，不是训练好的预测模型，不会产生新的电荷。

## 启用与运行

创建纯文本 `region.txt`，写 **1-based 全局拓扑原子编号**，可空格/换行分隔，支持 # 注释；不直接读取 `.ndx` 组名。例如：

```text
1 8 23 24
```

Linux/WSL，所有路径使用 Linux 路径：

```bash
export GMX_DYNAMIC_CHARGE_REGION=/absolute/path/region.txt
export GMX_DYNAMIC_CHARGE_LOG=/absolute/path/charges.log
export GMX_DISABLE_MODULAR_SIMULATOR=1
gmx mdrun -s run.tpr -deffnm run -ntmpi 1 -nb cpu -pme cpu -bonded cpu -update cpu
```

使用 CUDA 构建时，GPU 力计算的命令为：

```bash
gmx mdrun -s run.tpr -deffnm run -ntmpi 1 -nb gpu -pme gpu -bonded cpu -update cpu
```

也支持 `-nb gpu -pme cpu`；GPU PME 的 FFT 可通过 `-pmefft cpu` 放到 CPU。2026.4 原生调度要求 GPU PME 同时使用 GPU 非键，因此不支持 `-nb cpu -pme gpu`。预测函数目前在 CPU 上调用；CPU 积分产生坐标，短程/PME 力可在 GPU 上计算。动态电荷会自动关闭 GPU buffer ops，不需要额外设置 `GMX_GPU_DISABLE_BUFFER_OPS`。这避免 GPU 坐标打包保留旧的 xq.w 电荷。PME 新系数的 H2D 与 spread 位于同一 stream，按提交顺序执行；锁页 staging 槽位复用前等待其 H2D 完成事件。

mdp 使用 `integrator = md`，支持 `coulombtype = PME` 或 `Cut-off`。日志逐条记录 `step time_ps atom_id_1based charge_e`，以追加方式写入。XTC/TRR 不保存逐帧电荷，须保留日志。QEq-REANN C++ 模型接入、单位及乙烷验证见 `QEQ_REANN_ETHANE_ZH.md`。

无 `GMX_DYNAMIC_CHARGE_REGION` 时扩展关闭。环境变量必须在进程启动前设置。

区域文件和预测表目录、日志路径在本次模拟首次更新时读取并缓存；运行期间编辑区域文件不会改变选区。改变选区或这些路径后应启动新的模拟进程。日志流保持打开，每次更新后刷新；日志格式不变。

调试真实逐步变化时可设置：

```bash
export GMX_DYNAMIC_CHARGE_TABLE_DIR=/absolute/path/predictions
```

在此目录预先放置 `0.charges`、`1.charges` 等文件，文件名使用实际绝对 MD step，文件内容按 region 顺序给出全部区域电荷。续跑使用续跑 step；初始和最终力计算可能需要对应文件。缺文件或数目不符立即终止；这只是回放测试接口，没有启动外部 Python 服务或等待文件生成。

## 当前实现边界

当前实现：单 rank，CPU 或 GPU 非键/PME，CPU 更新，标准 md，无 FEP/MTS。普通 `[pairs]` 的 LJ14 使用 `chargeA`，随之更新；含显式 qi/qj 的 LennardJonesCoulomb14Q 和 `[pairs_nb]` 暂不支持，运行时拒绝。只更新选定原子的电荷，LJ 参数和拓扑排除关系仍按原拓扑。GPU 验证状态见验证记录；实现未宣称其他 GPU 后端经过测试。

要求区域总电荷保持不变（1e-5 e 容差），不自动给模型结果做投影。模型若需电荷守恒，应自行实现受约束输出；不要靠逐步容差容许的漂移改变总电荷。NaN、Inf、越界/重复原子编号均拒绝。

GPU 版本已增加 device 短程电荷与 PME 系数同步；当前不支持 GPU 积分、GPU graphs、多 rank 和直接 GPU 通信。多 rank 扩展仍需区域原子收集、halo 电荷通信及统计量归约。

模型最好是当前几何的确定性函数。默认接口不保存模型状态到 checkpoint；依赖历史/随机数的模型需要额外设计 checkpoint。DD 重分区可能从拓扑恢复旧电荷，所以不要把输入 chargesE 当作可靠的历史状态。

## 物理含义

本扩展实现的是可变点电荷的 MM 静电计算，不会自动添加 ML 能量或 ML 力。如果 q=q(R) 来自普通机器学习输出，GROMACS 现有力是固定 q 的偏导，缺少一般情况下的链式项 `-sum_i (dU/dq_i)(dq_i/dR)`。因此不能仅凭接口工作就声称得到严格保守的 ML/MM 势或 NVE 能量守恒。需要由一致的能量模型导数提供附加力，或者证明电荷是相应受约束能量的变分驻点。

## 构建

```bash
cmake -S . -B build -DGMX_GPU=OFF -DGMX_MPI=OFF -DGMX_FFT_LIBRARY=fftpack \
  -DBUILD_TESTING=OFF -DGMX_BUILD_UNITTESTS=OFF -DGMXAPI=OFF -DGMX_INSTALL_NBLIB_API=OFF
cmake --build build --parallel 4
```

FFTPACK 用于无额外依赖的功能验证，正式计算可换成本机配置的 FFTW 和合适 SIMD。验证状态见同目录 `DYNAMIC_CHARGES_VALIDATION.md`。

CUDA 构建示例（编译器路径和 SM 应匹配本机）：

```bash
cmake -S . -B build-cuda -DGMX_GPU=CUDA -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DGMX_CUDA_TARGET_SM=75 -DGMX_CUDA_TARGET_COMPUTE=75 \
  -DGMX_FFT_LIBRARY=fftpack -DGMX_MPI=OFF -DBUILD_TESTING=OFF \
  -DGMX_BUILD_UNITTESTS=OFF -DGMXAPI=OFF -DGMX_INSTALL_NBLIB_API=OFF
cmake --build build-cuda --parallel 8
```
