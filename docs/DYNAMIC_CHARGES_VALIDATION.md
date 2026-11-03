# 验证记录

日期：2026-10-21。

- WSL Ubuntu 24.04，GCC：独立 C++ 接口测试通过，覆盖区域解析、非连续编号、返回顺序、零变非零、电荷守恒、错误长度、NaN/Inf、重复/非法/空区域与缺失逐步预测文件。
- GROMACS 2026.4 全部源码已编译及链接成功，生成 `build-dynamiccharge/bin/gmx`（Linux/WSL 可执行程序）。配置：CPU/FFTPACK、单精度 real、无外部 MPI、无 GPU、SIMD=None；未构建项目完整单元测试套件。
- 引擎对照测试 `tests/dynamiccharge_engine_smoke.py` 通过。四原子体系，区域为全局原子 1 和 3，区域外电荷保持 0.3/-0.3 e。动态区域从拓扑零电荷更新到 0.2/-0.2 e，与独立在拓扑中设置 0.2/-0.2 e 的固定电荷参考体系比较。
- 运行 12 步，检查 0–12 共 13 个电荷输出步，覆盖 nstlist=10 的邻居表复用及重建；X/V/F 与势能在工具输出精度下最大绝对差为 **0.0**。
- 另一轮以 q(step)=0.01*step、-q(step) 逐步改变区域电荷；逐步日志与预期一致，并产生与恒定电荷不同的轨迹/力。
- 测试输入采用 comm-mode=Linear、nstcomm=1；grompp 不使用 -maxwarn 放行警告。

复现：

```bash
python3 tests/dynamiccharge_engine_smoke.py build-dynamiccharge/bin/gmx
```

测试生成数据在 `tests/engine-smoke/`，总结果在 `engine-smoke-dynamiccharge.log`。这验证了小体系的 CPU 电荷传播和动力学一致性，不等于真实 ML 模型、所有拓扑或长期统计性质的验证。

桌面原始目录的写回权限请求被拒绝，因此桌面源文件尚未修改。工作区包含完整修改版和独立补丁。

## CUDA 扩展验证

GPU 版已经完成全量编译及真实 GPU 运行验证。环境：WSL Ubuntu 24.04，GCC/G++ 12.4，CUDA Toolkit 12.1.105，驱动 566.07（WSL 显示 565.65），GeForce GTX 1660 系列 6 GB，SM 75，cuFFT。可执行程序：`build-dynamiccharge-cuda/bin/gmx`。

| 路径 | 固定电荷参考对照 | 逐步变化电荷与 CPU 对照 |
|---|---|---|
| CPU 非键 + CPU PME | 通过 | CPU 基准 |
| GPU 非键 + CPU PME | 通过 | 通过 |
| GPU 非键 + GPU PME（cuFFT） | 通过 | 通过 |
| GPU 非键 + GPU PME（CPU FFT） | 通过 | 通过 |

每个测试覆盖 0–12 步、非连续区域 1/3、初始零电荷变非零、每步电荷变化、邻居表复用和第 10 步的列表重建。对照读取 TRR 的 X/V/F 和 EDR 势能。GPU 与 CPU 允许混合精度路径的数值差异；详细诊断在对应测试日志中。GPU/固定参考的数值最大差不超过 8e-6；逐步变化的 CPU/GPU 对照数值最大差不超过 1.98e-4（脚本汇总 X/V/F/势能字段，仅是数值诊断，各字段量纲不同）。

运行日志明确确认 `PP tasks ... on the GPU`、`PME tasks ... on the GPU` 和 CPU update；不是静默回退到 CPU。GPU buffer ops 没有手工设置禁用变量，测试验证了动态电荷对它的自动处理。

```bash
python3 tests/dynamiccharge_engine_smoke.py build-dynamiccharge-cuda/bin/gmx gpu
python3 tests/dynamiccharge_engine_smoke.py build-dynamiccharge-cuda/bin/gmx nb-gpu
python3 tests/dynamiccharge_engine_smoke.py build-dynamiccharge-cuda/bin/gmx gpu-mixed
```

相应日志：`engine-smoke-dynamiccharge-gpu.log`、`engine-smoke-dynamiccharge-nb-gpu.log`、`engine-smoke-dynamiccharge-gpu-mixed.log`。生成的输入、TRR、EDR 和电荷日志在 `tests/engine-smoke-{模式}/` 中。

仍未验证且显式不支持：多 rank、GPU 积分、GPU graphs、FEP/MTS、含显式 qi/qj 的特殊 pair 类型；其他 GPU 后端未测试。默认预测仍是保留原电荷，测试使用逐步电荷表回放，尚未包含真实机器学习模型。上述小体系测试不替代真实体系、长期稳定性或 ML/MM 能量一致性验证。

## 2026-10-21 独立复核

重新执行接口单元测试和上述四条 CPU/CUDA 路径，全部通过。静态检查核对了预测发生在力计算之前、非邻居表更新步重新打包短程电荷、GPU PME 上传与 spread 使用同一 stream、双缓冲复用前等待事件，以及 qsum/q2sum 的更新。

新增 `tests/dynamiccharge_snapshot_oracle.py`：从逐步变化电荷的实际 TRR 轨迹抽取 0–12 步的原始二进制坐标，每帧生成对应电荷的固定拓扑，取消所有动态电荷环境变量，以 CPU rerun 独立计算该帧力和势能。参考计算仍使用本项目的 GROMACS 二进制，但不经过动态电荷更新分支。四条路径共 52 帧全部通过。这比两条共用预测接口的路径互相比对更能检查共同的更新时序错误。

| 被检查路径 | 最大力分量绝对差 kJ/(mol·nm) | 最大势能绝对差 kJ/mol |
|---|---:|---:|
| CPU 非键 + CPU PME | 0 | 0.000004 |
| GPU 非键 + GPU PME | 0.000100 | 0.000198 |
| GPU 非键 + CPU PME | 0.000010 | 0.000004 |
| GPU 非键 + GPU PME、CPU FFT | 0.000100 | 0.000004 |

数值由 GROMACS dump/energy 文本输出解析，受其输出精度限制；不是二进制逐位一致性声明。原始结果保存于工作区 `tests/snapshot-oracle/results.json`。

另外新增 `bonded` 测试选项，包含三条键、成键排除和普通 function-1 的 1–4 pair。CPU 与 GPU 的固定/动态电荷对照均通过，GPU 最大混合字段差 0.000004；变化电荷 CPU/GPU 对照最大混合字段差 0.000114。GPU bonded 核本身未测试，这里仍使用 `-bonded cpu`。

```bash
python3 tests/dynamiccharge_snapshot_oracle.py build-dynamiccharge-cuda/bin/gmx
python3 tests/dynamiccharge_engine_smoke.py build-dynamiccharge/bin/gmx cpu bonded
python3 tests/dynamiccharge_engine_smoke.py build-dynamiccharge-cuda/bin/gmx gpu bonded
```

本轮修复了测试脚本合并 stdout/stderr 导致诊断文本偶尔插入数值行的问题；引擎源码未改动。新增成键用例最初因排除原子间距超过截断半径被 grompp 正确拒绝，随后把测试截断半径设为 1.3 nm、列表半径设为 1.4 nm，没有用 maxwarn 放行。

结论范围：这些测试支持当前单 rank、CPU 积分配置下的电荷传递、短程/PME/普通 1–4 力和势能一致性。没有完成大体系、长时间、checkpoint 续跑、温压耦合及所有拓扑组合验证。守恒检查比较的是输入帧区域总电荷与本次预测，容差 1e-5 e；它不是相对于固定初始总电荷的累计漂移约束。真实 ML 模型仍必须自行保证电荷约束及能量/力导数一致性。

## 后续大体系补充

上述“小体系复核”完成后，进一步完成了 41,472 和 139,968 原子的 SPC 水体系检查，以及四种运行路径各三次测速。大体系采用 300 原子的更新区域，GPU 短程 + GPU PME，动态版使用 CPU 积分。两种规模的固定拓扑/动态电荷对照通过；139,968 原子的 200 步连续变电荷检查及六帧固定拓扑 CPU 独立力参考也通过。测试包含约束、300 K v-rescale 和周期边界。

动态接口占位预测速度为 36.525、16.722 ns/day，相同条件固定电荷 CPU 积分参考为 101.539、34.200 ns/day；真实 ML 推理耗时未计入。当前性能开销明显，不能据此称为高效 GPU 驻留实现。

完整条件、原始数据路径、误差和速度范围见 [大体系正确性与性能实测](DYNAMIC_CHARGES_LARGE_BENCHMARK_ZH.md)。长时间稳定性、checkpoint 续跑、压力耦合及所有拓扑组合仍未覆盖。本次只新增测试与报告，未修改引擎源码。

## 后续源码优化复测

随后对运行缓存、Frame 填充、电荷统计量和选区 NBNxM 刷新进行了源码优化。优化版四条 CPU/GPU 路径、普通 1–4、52 帧独立参考及两种大体系对照通过；139,968 原子的连续变电荷测试及六帧固定拓扑 CPU 独立参考也通过，力相对 RMS 差约 1.2e-6。

优化后 identity 占位预测路径三次测速中位数为 **103.608、32.982 ns/day**，优化前为 36.525、16.722；另测含文件读取成本的每步变化电荷回放为 **42.543、24.018 ns/day**。真实 ML 推理时间未计入。完整改动、范围、对照条件和限制见 [性能优化与复测](DYNAMIC_CHARGES_OPTIMIZATION_ZH.md)。
