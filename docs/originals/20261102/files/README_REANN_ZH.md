# GROMACS 2026.4：REANN 势与 QEq 动态电荷

## MIREANN 七输入静电嵌入（GMX_MIREANN）

本接口接收用户提供的七输入 TorchScript 模型：PBC、QM 坐标、MM 坐标、MM 电荷、
盒子、QM 元素和质量；模型返回静电能、QM/MM 能量梯度及新 QM 电荷。
五输入的气相 QEq 模型不能用于本接口。气相 REANN PES 通过独立 NNP 模块启用。

- `mireann-center-atom` 是体系全局编号，从 1 开始，必须属于 QM。
- GMX_MIREANN RESIDUE：默认 `mireann-mm-selection=residue`。每次力评估用
  GROMACS 原生邻域搜索检查 MM 原子到指定中心的周期距离；任一 MM 原子在半径内，
  就纳入其所属拓扑残基的全部 MM 原子，边缘原子可以超过该半径。
  按真实拓扑残基分组，区分重复水分子的不同实例；不使用会循环的 GRO 残基号。
  每个残基以球内最近原子为周期 image 锚点，其余原子相对该锚点 image，保持水完整。
  每个物理 MM 原子只纳入一份，输入和回传力均按全局原子编号对应。
  `mireann-mm-selection=atom` 保留先前逐原子球选区，用于复现旧测试。
  新残基模式须重新运行 grompp，将残基元数据写入 TPR。
- 模型坐标为 Å，电荷为 e；GROMACS 距离 nm，能量 kJ/mol，力 kJ/(mol nm)。
  kcal/mol 模型设置 `mireann-energy-scale=4.184`；力转换包含额外的 10 倍。
- QM 选区采用完整、无跨区共价键的分子。不支持赝键、link atom 或 zeromask 边界。
  初始拓扑 QM 总电荷必须与模型总电荷一致；不隐式修改 MM 电荷或投影模型电荷。

总能量约定（模型 EE 不含气相 PES）：

```
U = U_GROMACS(q_new) + U_MIREANN_EE - U_subtraction(q_new)
K(r) = 1/r                         r <= rcoulomb
K(r) = erf(alpha*r)/r              r >  rcoulomb
U_subtraction = sum(epsfac * q_QM_new * q_MM * K(r))
F_outer_on_QM = epsfac*q_QM_new*q_MM *
  [erf(alpha*r) - 2*alpha*r/sqrt(pi)*exp(-(alpha*r)^2)] * (R_QM-R_MM)/r^3
```

扣除作用于模型输入的 QM/MM image 对；25 Å 是选区半径，`rcoulomb` 是经典 PME
实空间截断，两者不同。alpha 与 epsfac 使用本次 GROMACS PME 的实际参数。
建议使用 `mdrun -notunepme` 固定经典截断和 Ewald 参数，避免 PME 调优改变分段核。
该扣除沿用用户指定的 Amber 分段能量约定，并补上其缺失的高斯导数项；它不等价于
对这些原子做完整周期 Ewald 求和。分支在 rcoulomb 处有被截去的 erfc 小尾项，
硬球选区边缘也可能不连续；有限差分及能量守恒评估必须区分是否跨越这些边界。
默认 `mireann-switch-width=0` 保持原始 MM q(e)。残基模式目前要求该值为 0；
整残基入/出球仍有硬切换，不代表解决了能量漂移。可选平滑壳层仅适用于显式 atom 模式，
会改变模型输入电荷，并显式计入权重导数及中心原子的反作用力。

模型本身的电荷响应包含在其能量梯度中，只计算一次。残余项
`U_GROMACS(q_new)-U_subtraction(q_new)` 的响应另用 VJP 补齐；不把 VJP 辅助标量加到能量。
直接从本步已解出的 PME 电势网格插值，再加入实空间、排除、自能和净电荷背景项。
CUDA PME 通过新 CUDA gather 核只返回 N_QM 个电势，不下载整个网格、不重新运行 PME。
模型推理/自动微分支持 `cpu` 或 `cuda:N`。选区、经典扣除和力累加目前在 CPU；
坐标、电荷和最终力通过现有主机接口传递，因此不是全程 GPU 驻留实现。

限制：3D 周期、导体边界 PME、单 rank、`integrator=md`、`-update cpu`；
不支持 FEP/MTS/GPU update/GPU graph/跨 QM-MM 拓扑排除及显式电荷 pair 类型。
NN Potential 能量项包含本接口净修正，并可与原生 NNP 的气相 PES 相加。
`mireann-charge-log`、`mireann-energy-log`、`mireann-environment-log` 默认关闭，
用于逐步核对新电荷、能量分解、image 坐标及原始 MM 编号；长程生产模拟避免逐步写大量环境日志。
`mireann-profile` 排除前 20 次预热调用；短于预热长度时 measured_calls=0，均值不代表实际耗时。

MDP 示例（下面的 QM 编号和元素仅适用于所附 MECL 示例）：

```ini
mireann-active = yes
mireann-model = REANN_CHA_DOUBLE.pt
mireann-atoms = 1 2 3 4 5 6
mireann-elements = Cl Cl C H H H
mireann-center-atom = 1
mireann-mm-cutoff = 25
mireann-mm-selection = residue
mireann-switch-width = 0
mireann-energy-scale = 4.184
mireann-device = cpu
mireann-threads = 1
mireann-charge-log = charges.log
mireann-energy-log = embedding.csv
; 气相 PES 可另外使用原生 NNP；输入组必须是相同 QM 原子。
; nnpot-active = yes
; nnpot-model-type = reann
; nnpot-modelfile = REANN_PES_DOUBLE.pt
; nnpot-input-group = QM
; nnpot-embedding = mechanical
; nnpot-reann-device = cpu
; nnpot-reann-energy-scale = 4.184
```

## MIREANN 本机验证范围

已完成完整双精度 CPU 构建与混合精度 CUDA 构建。验证环境为 WSL Ubuntu 24.04、
GCC 12、CUDA 12.1、现有 CPU 版 LibTorch 2.6；服务器环境仍需重新编译验证。

- C++ 后端与原模型的能量、电荷及 QM/MM 力最大差 8.6e-14；空 MM 选区和重复调用通过。
- C++ 残余电荷响应 VJP 的坐标有限差分最大差 1.4e-6 kJ/(mol nm)。
- 分段库仑/erf 扣除核的力有限差分最大差 7.0e-9 kJ/(mol nm)，包括显著高斯项的低 alpha 测试。
- 双精度 GROMACS 总能分解差 2.7e-11 kJ/mol；检查的 QM/MM 总力分量与总能有限差分
  最大差 1.2e-6 kJ/(mol nm)。单精度网格/能量的消减误差不适合用很小差分步长作严格审计。
- CUDA 短程 + PME + 新电势 gather 的小体系最大力差 0.0025 kJ/(mol nm)，参照双精度 CPU。
- 先前逐原子模式测试：用户 MECL 溶液 46,801 原子，中心原子 1，半径 25 Å，初始选入 6,699 个 MM；
  气相 REANN PES + MIREANN 嵌入分别在 CPU 双精度及 CUDA 静电路径跑通 5 步，电荷逐步更新。
- 本机 LibTorch 不含 CUDA，以上 CUDA 测试的模型推理在 CPU；尚未验证 CUDA LibTorch 推理，
  也没有完成生产时长、能量守恒或稳态速度基准。5 步含 JIT 预热，不能据此评价生产速度。

独立检查实际使用的扣除核（在源码根目录执行，无需 LibTorch）：

```bash
c++ -std=c++17 -O2 -I src/gromacs/mdlib tests/dynamiccharge/mireann_subtraction_check.cpp -o /tmp/mireann-check
/tmp/mireann-check
```

## Linux 编译

需要 CMake、兼容的 C/C++ 编译器、CUDA Toolkit（GPU 构建）、Python 开发头文件和
Linux cxx11-ABI LibTorch。PES 与 QEq 使用同一套 LibTorch；无需另建 Python 环境，
MD 循环通过 C++ 调用 TorchScript。`nvidia-smi` 的 CUDA Version 不等于已安装的
Toolkit 版本，请用 `nvcc --version` 核对。

在源码根目录执行，把 `/absolute/path/to/libtorch` 改为实际解压目录：

```bash
cmake -S . -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/gromacs-reann" \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch \
  -DGMX_NNPOT=TORCH -DGMX_QEQ_REANN=ON \
  -DGMX_GPU=CUDA -DGMX_MPI=OFF -DGMX_THREAD_MPI=ON \
  -DGMXAPI=OFF -DGMX_INSTALL_NBLIB_API=OFF -DGMX_INSTALL_LEGACY_API=OFF \
  -DGMX_USE_COLVARS=NONE -DGMX_USE_PLUMED=OFF \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DGMX_BUILD_OWN_FFTW=ON
cmake --build build --parallel 4
cmake --install build
source "$HOME/gromacs-reann/bin/GMXRC"
```

`GMX_BUILD_OWN_FFTW=ON` 会下载并编译 FFTW。离线且已安装 FFTW 时，改为
`-DGMX_BUILD_OWN_FFTW=OFF -DGMX_FFT_LIBRARY=fftw3`；默认混合精度需要 `libfftw3f`。
仅构建 CPU 版时改为 `-DGMX_GPU=OFF`，模型设备也选 CPU。
不要混用其他 Conda/Python 环境里的 Torch 动态库。

## 最小运行示例

QEq 动态电荷，保留经典力场：

```bash
cd examples/qeq-reann-ethane
gmx grompp -f md.mdp -c start.gro -p eth.top -o run.tpr
gmx mdrun -deffnm run -ntmpi 1 -ntomp 1 -nb gpu -update cpu
```

REANN 能量与力，使用原生 NNP 模块：

```bash
cd examples/reann-ethane
gmx grompp -f reann-vacuum.mdp -c start.gro -p eth.top -n qm.ndx -o run.tpr
gmx mdrun -deffnm run -ntmpi 1 -ntomp 1 -nb gpu -update cpu
```

上面的 `cd` 都从源码根目录开始。CPU 测试将 `-nb gpu` 改为 `-nb cpu`，并将 PES
示例中的 `nnpot-reann-device` 改为 `cpu`。两个示例保留原始模型、结构、拓扑与 MDP。
模型文件路径相对于运行目录；权重不嵌入 TPR，上传服务器时也要带上模型。

## MDP 接口与单位

PES 模型输入为坐标/盒矩阵（Å）、元素序号和质量（u），输出能量与力。输入坐标
从 GROMACS nm 转为 Å；此模型训练能量单位为 kcal/mol，所以设置：

```ini
nnpot-active = yes
nnpot-model-type = reann
nnpot-modelfile = REANN_PES_DOUBLE.pt
nnpot-input-group = QM
nnpot-embedding = mechanical
nnpot-reann-device = gpu
nnpot-reann-energy-scale = 4.184
```

能量乘 4.184 转为 kJ/mol，力乘 41.84 转为 kJ/(mol nm)。`qm.ndx` 必须含所选 QM
组以及覆盖整个体系的 System 组。完整 QM 分子内部的普通经典项由原生 NNP
预处理替换，QM-MM 与 MM-MM 相互作用保留；不要另外约束由 PES 描述的 QM 内部键。
当前只支持完整分子、机械嵌入，模型 PBC 为 `0 0 0`。

QEq 可以单独启用，也可以和 PES 联用：

```ini
qeq-reann-active = yes
qeq-reann-model = REANN_CHA_DOUBLE.pt
qeq-reann-atoms = 1 2 3 4 5 6 7 8
qeq-reann-elements = C H H H C H H H
qeq-reann-device = cpu
qeq-reann-threads = 1
qeq-reann-pbc = 0 0 0
; 可选；逐步日志会影响速度
qeq-reann-charge-log = charges.log
```

原子编号从 1 开始，元素按编号顺序对应；质量取自拓扑。模型输出直接是 q(e)，
不使用 Amber 电荷缩放。乙烷模型的前 8 项为原子电荷，第 9 项为拉格朗日乘子。
`qeq-reann-device` 可用 `cpu` 或 `cuda:0`，与 GROMACS 的 `-nb cpu/gpu` 分别设置。
选区初始总电荷必须与模型总电荷一致；示例乙烷为中性。水相只将乙烷传给孤立
分子模型，完整体系的周期性由 GROMACS 的 `pbc` 和 PME 设置控制。

## 实现与当前范围

- `GMX_QEQ_REANN` 标记动态电荷改动，`GMX_REANN_PES` 标记原生 PES 改动。
  原文件中的替换范围用 CHANGE BEGIN/END 标记，原语句以 `ORIGINAL |` 注释保留。
- 每次力计算开始预测电荷，更新 chargeA/B、总电荷/平方和、短程打包电荷和 PME
  电荷系数后再计算力。QEq 本身不额外增加模型能量。
- 当前动态电荷支持单 rank、普通 `integrator=md`、CPU 积分、CPU/GPU 短程和 PME；
  不支持 FEP、MTS、多 rank 或 GROMACS GPU graphs。PES 的能量与力通过原生主机
  接口累加；当前并非全程 GPU 驻留或 GPU 电荷直连。
- **独立五输入 qeq-reann 模式**联用 q(R) 与经典跨区静电时，尚未补齐该静电能对 q(R) 的链式力。
  固定构型单点等价不代表已验证完整守恒的水相动力学。

七输入 MIREANN 的剩余电荷响应与赝键边界另见 `docs/mireann-pseudo-bond.md`。
2026-10-25 的 `GMX_MIREANN_GPU_AUDIT` 增加 CUDA 逐对电势/参考扣除，复用 PME
采样缓冲区，并省去 MIREANN 后端重复反向传播；支持
`mireann-electrostatics-device=auto/cpu/cuda:N`。积分仍走 CPU，不能称为全 GPU 驻留。
- 本地已完成 CPU 编译、乙烷短程 MD，以及 12 组动态/手动电荷单点对照：
  能量分项和力差值均为 0；GPU 完整构建、数值与性能仍需在服务器验证。
