# 固定 s 的 MM ↔ ML/MM 双哈密顿

用户定义：`s` 在一次MD中保持固定，0≤s≤1。两个端点使用同一帧坐标和盒子：

\[
U_s=(1-s)U_{\mathrm{MM}}+sU_{\mathrm{ML/MM}},\qquad
\mathbf F_s=(1-s)\mathbf F_{\mathrm{MM}}+s\mathbf F_{\mathrm{ML/MM}}.
\]

`dU/ds = U_ML/MM − U_MM` 随每次力计算写入 `dual-energy.csv`，可用于后续TI分析。固定s没有空间权重导数力。本功能没有把两套电荷先插值后送入一遍PME：那一般不是线性哈密顿混合。

## 两套电荷和两个原生端点

- **MM**：原始、未经ML拓扑预处理的TOP；原TOP固定电荷；完整原始键、角、二面角、1–4、LJ、经典静电、PME及原输入的色散修正。不做ML/MM初始化补偿。
- **ML/MM**：沿用之前的NNPot/PES与MIREANN、动态QM电荷、一次性全MM补偿、zero-mask、残基/image选区、普通与跨区1–4静电扣除、高斯导数、剩余电荷响应等原流程。没有为本功能重写或简化它们。
- 主端内存中单独保存 `qMM` 和 `qMLMM`；qMM按全局编号逐原子检查固定，qMLMM每步接收更新。两套电荷总和、电荷平方和独立记录，不互相覆盖，也不建立所谓“混合电荷”。原生MM和ML/MM各有独立的PME、拓扑与模型状态。
- 每步两个原生GROMACS进程各计算一次完整端点，因此各执行自己的PME能量/力计算。ML/MM的剩余响应沿用其原PME网格电势读取，不由本功能增加第三次PME求解。

这一版采用**两个本机进程、一个积分器**：MM主进程执行GROMACS原生积分，ML/MM进程通过原生rerun力循环返回同一帧能量、力、维里和电荷，不执行独立MD。二者用权限0600的本机Unix socket通信，父目录0700；不开放TCP服务、不访问远程主机。主端在全部原生力/能量/维里归并后、积分前加权。普通非双哈密顿运行不建立socket，也不改变原有ML/MM路径。

使用独立计算状态是为了保留两个端点不同的拓扑排除、键合项和电荷；只替换一套chargeA后乘模型力系数无法完成这些恢复。两进程求力与数据往返有开销，这个版本不宣称比原单端点快。

## 当前支持范围

本地已建立独立CPU双精度构建。初版运行要求Linux/WSL、单rank、`integrator=md`、`-update cpu`、`nstcalcenergy=1`、固定盒子NVE/NVT；禁止原生FEP/MTS、虚拟位点、GPU graph和设备端buffer ops。编译CUDA后的模型/短程/PME路径仍需独立验证；**本次不能宣称CUDA版本或GPU积分已通过**。NPT、外部MPI和动态s不在本版已验证范围。

两个端点必须具有相同原始原子顺序、原始TOP电荷、质量、盒子、约束和共同限制势。运行时检查原始电荷/质量/全局编号以及请求步号与时间。原生重排序后重新建立全局↔局部映射，避免混淆原子；不能用当前位置行号直接对应两端。

共同US/NMR等限制势应在两个端点都包含且参数相同，这样加权后仍是一份原限制势。不要在混合后再手动重复加入。准备脚本使用同一份原输入和原TOP，保留共同项。

## 准备输入

**必须提供原始完整system.top，不能使用已删除QM经典项的processed.top作为MM端点。** 原文件不修改，脚本在新目录产生两份MDP、两个TPR和config.json。现有模型文件路径转成绝对路径，权重不改。输入保留原物理参数，MM端仅关闭ML模块以恢复完整原始力场；ML/MM端保留原模块设置。

```bash
python prepare_inputs.py \
  --gmx /path/to/new-build/bin/gmx_d \
  --mdp /path/to/original/nvt.mdp \
  --top /path/to/original/system.top \
  --coordinates /path/to/start.gro \
  --index /path/to/qm.ndx \
  --s 0.5 --out inputs-s0p5

python run_dual.py --config inputs-s0p5/config.json \
  --run-dir run-s0p5 --ntomp 4
```

`--ntomp 4`为每个端点4个OpenMP线程，合计最多8个；不是两个端点合计4个。运行目录必须不存在，避免覆盖。启动器只管理自己创建的两个子进程，不附加或干扰其他MD。

原生MM输入新增参数为：

```ini
dual-hamiltonian-active = yes
dual-hamiltonian-s = 0.5
dual-hamiltonian-socket = /tmp/gmx-dual-unique/endpoint.sock
dual-hamiltonian-energy-log = dual-energy.csv
nstcalcenergy = 1
```

每个s使用单独的新目录和socket；ML/MM工作端的`dual-hamiltonian-active=no`，由启动器设置工作端环境变量并进入原生rerun。现有checkpoint的续跑尚未做新功能资格验证，不能直接拿旧ML/MM checkpoint当作新的混合哈密顿初态。

## 文件和构建

本仓库包含完整 GROMACS 源码；两个辅助脚本位于 `extensions/dual-hamiltonian/`。
在源码根目录先按主 README 的 CPU 双精度方式构建，再将示例中的 gmx 路径替换为实际程序。
`python3 tests/mireann/run_checks.py` 可运行不依赖模型的核心检查。
端点及短步积分检查的数值结果见 `docs/release-validation-20261102.md`。
本功能当前没有通过 GPU、MPI、checkpoint 续跑或长期 NVE 资格验证。
