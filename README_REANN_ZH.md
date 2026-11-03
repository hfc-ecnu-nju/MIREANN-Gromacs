# GROMACS 2026.4 的 REANN / MIREANN 扩展

本项目是**非官方研究分支**，在保留 GROMACS 原生源代码及许可证的基础上提供：

| 功能 | 作用与当前范围 |
|---|---|
| REANN PES | 经原生 NNP 模块调用 TorchScript，计算 QM 区气相能量和力 |
| MIREANN 静电嵌入 | QM 与局部 MM 背景七输入推理、EE 能量、QM/MM 力和动态 QM 电荷 |
| 周期经典静电 | 原生 PME 使用本步更新电荷；扣除被模型替代的指定局部参考，保留其余周期贡献 |
| 电荷响应 | 对经典剩余项补充 VJP；不重复计算模型自身能量的梯度 |
| 赝键边界 | 使用真实原子与专用训练标签，无人工 link H；保留跨边界键、角、二面角及 LJ-14 |
| GPU 与缓冲 | 单端点模型、短程、PME、部分键合与积分可用 GPU；使用主机中转与必要同步 |
| 候选选区 | 可配置内球、外候选球和重建周期；实际模型选区与 image 每步更新 |
| 最小化 | 单端点 MIREANN 支持原生 `integrator=steep` |
| 平滑选区 | 可选整残基 C2 权重及链式力，默认关闭，仍属实验功能 |
| 双哈密顿 | 固定 s 的完整 MM ↔ 原 ML/MM 混合，两套独立电荷/PME，一个 CPU 积分器 |

这不是运行时 DFT 接口，也不能自动把任意 .pt 变成适用的赝键势。
已有单点/端点一致性测试不等于长期动力学完全正确。**此前 MECL 平滑选区 50 ps NVE
仍有系统性漂移，尚未解决；不应据此宣称生产 NVE 守恒。**

本轮另修复了一个明确复现的问题：原生邻居/最小化重建会覆盖已补偿的 MM 电荷。
现已保存并恢复固定 MM 电荷快照，详见 [电荷保持修复](docs/mm-charge-persistence-fix-20261102.md)。
旧单点/端点对照不能代替此跨步状态检查；修复后仍须重新评估长期动力学。

## 目录和说明

- [实际能量、力、电荷与周期处理公式](docs/mireann-methods.md)
- [赝键及跨区拓扑处理](docs/mireann-pseudo-bond.md)
- [固定 s 双哈密顿用法](docs/mireann-dual-hamiltonian.md)
- [2026-11-02 复核记录与验证边界](docs/release-validation-20261102.md)
- [变更记录](CHANGELOG_MIREANN.md)
- `src/gromacs/mdlib/`：选区、动态电荷、参考扣除、剩余响应及双哈密顿桥接。
- `extensions/qeq-reann/`：隔离的 LibTorch C ABI 后端；无需在 MD 循环启动 Python。
- `extensions/dual-hamiltonian/`：两个本机端点的输入准备及启动器。
- `tests/mireann/`：不需要模型的公式、缓存、权重和双端点核心检查。

原始 `README`、`COPYING`、`INSTALL`、`AUTHORS`、`CITATION.cff` 均保留。
新增/修改代码有 `GMX_*` 标记；被替换语句保留在 `ORIGINAL` 注释或原分支中。
本轮更新另外保存文件级更新前快照，索引见 `docs/originals/20261102/manifest.json`。

## Linux 编译

源码要求 CMake **至少 3.28**；3.31.8 是可用版本，不是唯一允许的版本。
需要兼容的 C/C++ 编译器、Linux LibTorch、FFT 库；GPU 构建另需兼容的 CUDA Toolkit。
原有 CUDA 基线曾使用 LibTorch 2.7.1+cu128；本轮合并使用已有 CPU LibTorch 本地复核，
不能把它描述为本轮 CUDA 重新资格验证。请区分 `nvidia-smi` 的驱动支持版本与 `nvcc --version`。

在**本源码根目录**执行，先设置自己的依赖路径：

```bash
export LIBTORCH_DIR="$HOME/software/libtorch"
cmake -S . -B build-cu128 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/software/gromacs-mireann" \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
  -DGMX_NNPOT=TORCH -DGMX_QEQ_REANN=ON \
  -DGMX_DOUBLE=OFF -DGMX_GPU=CUDA \
  -DGMX_MPI=OFF -DGMX_THREAD_MPI=ON \
  -DGMXAPI=OFF -DGMX_INSTALL_NBLIB_API=OFF -DGMX_INSTALL_LEGACY_API=OFF \
  -DGMX_USE_COLVARS=NONE -DGMX_USE_PLUMED=OFF \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DGMX_BUILD_OWN_FFTW=ON
cmake --build build-cu128 --parallel 8
cmake --install build-cu128
source "$HOME/software/gromacs-mireann/bin/GMXRC"
```

`GMX_BUILD_OWN_FFTW=ON` 需要下载 FFTW。离线时使用已安装的匹配精度 FFTW 并设
`-DGMX_BUILD_OWN_FFTW=OFF -DGMX_FFT_LIBRARY=fftw3`；小型本地诊断也可显式使用 `fftpack`。
不要混用不同 Python 环境中的 Torch 动态库。训练模型不嵌入 TPR，运行时仍须提供模型文件。

CPU 双精度对照使用**另一个构建目录**，同一配置中改为：

```bash
# 替换上面配置中的构建目录、安装目录及两个精度/设备选项；不要复用 CUDA 的 CMakeCache。
# -B build-double
# -DCMAKE_INSTALL_PREFIX="$HOME/software/gromacs-mireann-double"
# -DGMX_DOUBLE=ON -DGMX_GPU=OFF
```

默认双精度程序名 `gmx_d`。GROMACS 混合精度不等于模型改成单精度：
当前 MIREANN 要求 float64 模型及 Å 输入；经典 mixed-precision 缓冲仍可能是 float32。

## 单端点最小输入示例

下面的六原子编号/标签仅示范 MECL 接口，不能照搬到 COX。QM 组必须在 `qm.ndx`
定义，NNP 组与 `mireann-atoms` 是同一组真实原子，模型元素按该输入顺序对应。
GROMACS 输入编号从 **1** 开始；VMD `index` 从 **0** 开始，显示时各减 1。

```ini
integrator = md
dt = 0.001
nsteps = 100
; 正常经典力场/PME、温控、约束及输出参数由体系输入提供

nnpot-active = yes
nnpot-model-type = reann
nnpot-modelfile = REANN_PES_DOUBLE.pt
nnpot-input-group = QM
nnpot-embedding = mechanical
nnpot-reann-device = gpu
nnpot-reann-energy-scale = 4.184

mireann-active = yes
mireann-model = REANN_CHA_DOUBLE.pt
mireann-atoms = 1 2 3 4 5 6
mireann-elements = Cl Cl C H H H
mireann-center-atom = 1
mireann-mm-selection = residue
mireann-mm-cutoff = 25
mireann-mm-list-cutoff = 30
mireann-mm-list-nst = 0
mireann-switch-width = 0
mireann-charge-response = yes
mireann-energy-scale = 4.184
mireann-device = cuda:0
mireann-electrostatics-device = auto
mireann-threads = 1
```

`mm-cutoff/list-cutoff/switch-width` 为 **Å**，不同于原生 `rcoulomb/rvdw` 的 **nm**。
上述 25/30 是用户可改参数；`mm-list-nst=0` 随原生邻居更新，正整数（如 20）给出最大步龄。
位移越过 skin 或盒改变会提前重建；外表缓存不允许漏掉内球新原子。
默认以中心原子为球心，整残基 image 后传入；不是用原始未处理的全盒坐标。
zero-mask 位置不输入模型，但仍保留在物理经典体系和参考扣除中。

```bash
gmx grompp -f md.mdp -c start.gro -p system.top -n qm.ndx -o md.tpr
CUDA_VISIBLE_DEVICES=0 gmx mdrun -s md.tpr -deffnm md \
  -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu -update gpu -notunepme
```

GPU 版要求支持该体系的 CUDA 路径；不支持的原生键合项可使用 `-bonded auto/cpu`。
PES 的 `nnpot-reann-device` 使用 `auto/cpu/gpu`，MIREANN 使用 `cpu/cuda:N`；两项语法不同。
`-notunepme` 用于固定参考分段所依赖的 cutoff/alpha。模型 `cuda:0` 指可见设备编号，
物理 GPU 1 可用 `CUDA_VISIBLE_DEVICES=1` 映射到模型的 `cuda:0`。
CPU 对照将三项模型/修正设备改为 `cpu`，并用 `-nb cpu -pme cpu -bonded cpu -update cpu`。
改变 MDP 或拓扑后重新 grompp；不要把本分支选项交给官方未修改的 gmx。

### 最小化、赝键、平滑和日志

- MIREANN `integrator=steep` 走同一能量/力路径；建议 CPU 对照，不能仅取消检查后跳过模型。
- 截断化学键使用 `nnpot-boundary=pseudo-bond` 与 `mireann-boundary=pseudo-bond`，
  需要赝键专用训练模型及真实元素/质量。不要自动生成 link H，也不要把物理 C 质量改成模型标签 F 的质量。
- `mireann-zero-atoms` 是全局 MM 编号列表，从模型背景排除；不是令全部经典电荷为零。
- 可选 `mireann-switch-width=2` 是外半径 25 Å 下 23–25 Å 的平滑层，需核对训练定义和长期稳定性。
- `mireann-charge-log`、`mireann-energy-log`、`mireann-environment-log`、`mireann-profile` 默认不大量输出；
  诊断时可指定文件。逐步环境日志可能很大，测速须排除预热与密集 I/O。

## 固定 s 双哈密顿

\[
U_s=(1-s)U_{MM}+sU_{ML/MM},\qquad
F_s=(1-s)F_{MM}+sF_{ML/MM},\qquad
\partial U_s/\partial s=U_{ML/MM}-U_{MM}.
\]

MM 端使用原始完整 TOP 的固定电荷及全部经典项；ML/MM 端保留上述动态电荷及补偿/扣除。
两套电荷分开保存，每步各求完整端点/PME，**不是电荷插值后只做一次 PME**。
共同 US/NMR 势必须在两端一致，加权后仍是一份。s 在一次 MD 中固定。

```bash
python3 extensions/dual-hamiltonian/prepare_inputs.py \
  --gmx /absolute/path/to/build-double/bin/gmx_d \
  --mdp /absolute/path/to/original/md.mdp \
  --top /absolute/path/to/original/system.top \
  --coordinates /absolute/path/to/start.gro --index /absolute/path/to/qm.ndx \
  --s 0.5 --out inputs-s05
python3 extensions/dual-hamiltonian/run_dual.py \
  --config inputs-s05/config.json --run-dir run-s05 --ntomp 4
```

必须使用**未经 NNP 删除 QM 项的原始 TOP**。输入/运行目录要求新建，避免覆盖。
当前双端点模式仅 Linux/WSL、单 rank、CPU 积分、固定盒子 NVE/NVT、`nstcalcenergy=1`；
不支持动态 s、FEP/MTS、虚拟位点、NPT、GPU graph，GPU/MPI/续跑未通过资格验证。
`--ntomp 4` 是每端 4 线程，最多合计 8。输出 `dual-energy.csv` 含两端势能、混合势能、dU/ds 和两套电荷统计。

## 测试和使用边界

```bash
python3 tests/mireann/run_checks.py
```

这组检查不需要模型，覆盖参考核及 shift、权重导数、缓冲列表、双电荷隔离、
端点混合和本机 IPC。完整模型/拓扑对照及本轮编译检查见验证记录。
独立五输入 `qeq-reann-*` 是另一模式，不含七输入 MIREANN 的完整残余能量响应，
不能把它的经典变电荷 MD 自动当成保守哈密顿量模拟。

当前普通 MIREANN 路径限定单 rank、3D 导体边界 PME；不支持 FEP/MTS/设备端 graph，
涉及 QM 的约束、虚拟位点及不支持的显式电荷 pair 会拒绝。纯 MM 水的 SETTLE 可保留。
本分支不保证与旧 Amber 所有公式完全相同：高斯导数和默认剩余电荷响应是明确差异。
不保证任意模型、稀有高能构型或任意长程嵌入都物理可靠。

## 上传 GitHub

以本 `gromacs-2026.4` 源码目录为仓库根，而不是它的项目父目录。
保留 `COPYING` 与上游作者/引用声明，明确标注非官方分支；论文引用请同时遵守 GROMACS 和所用模型要求。
`.gitignore` 排除模型、轨迹、构建和访问密钥，但它不能清理已经跟踪的文件，也不授予模型/第三方数据发布许可。
先执行 `git status --short` 和 `git diff --cached --stat` 审核待提交内容；不要盲目上传父目录中的 Amber 源码、服务器脚本或训练数据。
本次仅更新本地源码，未自动创建远程仓库、提交或推送。
