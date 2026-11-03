# 动态电荷接口性能优化与复测

日期：2026-10-21。本轮完成源码优化和大体系复测。选区为 300 原子，短程力和 PME 使用 GPU，动态电荷路径仍使用 CPU 积分；未接入真实 ML 模型。

## 实际修改

- `dynamicchargeprovider.h` 新增 `RuntimeState`，由当前模拟的 `t_forcerec` 独立拥有。区域文件只读取一次，预测目录和日志路径只解析一次；Frame 的内存复用，不使用进程级静态缓存。
- `sim_util.cpp` 在邻居表重建、DD 分区计数变化或本地原子数变化时重建映射。普通步复用映射和类型，大体系用 OpenMP 刷新全体系当前坐标和电荷。预测仍每次力计算调用，输入仍包含完整体系。
- 选区外电荷固定，重建时计算其 qsum/q2sum 贡献；每步用本次选区电荷重新求和得到总量。没有累计每步差值，避免增量统计自身的长期舍入累积。
- 新增 `nonbonded_verlet_t::updateRegionCharges()` 和 `nbnxn_atomdata_update_region_charges()`，在非重建步按 `GridSet::cells()` 的逆映射覆盖选区打包电荷，并验证逆映射一致性。无需重设类型、LJ 组合参数和能量组，填充原子的零电荷保留。重建步仍走原生完整设置。
- PME 锁页双缓冲上传和同 stream 排序保留，每步仍上传全部物理原子的 PME 系数。没有跳过 identity 预测的写入、校验、短程刷新或 PME 上传，也没有降低电荷更新频率。

区域文件不再支持运行中编辑后热更新；改变区域或配置路径后应启动新的模拟进程。当前扩展仍要求选区外电荷固定、无 FEP/MTS，CPU 积分和单 rank。

## 测速结果

与首次大体系基准使用相同硬件、SPC 水体系、步长和输出设置：i7-10700、GTX 1660 SUPER 6 GB、WSL、CUDA/cuFFT；8 个 OpenMP 线程。每种模式运行 3 次，每次 2,200 步，200 步后清零计时。Release、SIMD=None，绝对速度不代表平台最佳调优结果。

下表为 ns/day 中位数，越大越快。

| 路径 | 41,472 原子 | 139,968 原子 |
|---|---:|---:|
| 优化前动态接口，identity 占位预测 | 36.525 | 16.722 |
| **优化后动态接口，identity 占位预测** | **103.608** | **32.982** |
| 本轮固定电荷，CPU 积分 | 103.980 | 37.894 |
| 本轮固定电荷，CPU 积分，关闭 buffer ops | 105.105 | 35.968 |
| 本轮固定电荷，GPU 积分 | 164.859 | 53.648 |
| 优化后动态接口，每步变化的文件回放 | 42.543 | 24.018 |

占位预测路径相对优化前实测吞吐量分别提高 **2.84 倍、1.97 倍**。本轮相对同样 CPU 积分的固定电荷参考，吞吐量差约为 **0.4%、13.0%**；4.1 万原子的差异小于本轮运行波动。优化前同样的固定电荷参考是 101.539、34.200 ns/day，说明两轮测量存在机器状态/运行波动，因此不能把全部跨轮速度差严格归因于一个代码片段。

三次运行的范围：

| 路径 | 41,472 原子范围 | 139,968 原子范围 |
|---|---:|---:|
| 优化后 identity | 92.783–107.668 | 32.881–33.311 |
| 本轮固定电荷 CPU 积分 | 101.027–104.864 | 36.291–38.171 |
| 本轮固定电荷关闭 buffer ops | 97.172–110.839 | 35.770–36.526 |
| 本轮固定电荷 GPU 积分 | 153.139–169.180 | 52.984–54.010 |
| 每步变化的文件回放 | 40.634–44.856 | 21.291–24.111 |

identity 用于测量没有真实模型时的接口成本，不会产生新电荷，但执行所有更新工作。文件回放确实每步改变区域电荷，包含每步打开文件、读取和解析 300 个数值的成本，速度不能冒充内存内模型推理速度。两者都不包含真实 ML 推理。测速关闭电荷审计和轨迹输出；与单独正确性测试的输出条件不同。

GPU 积分仍未接入动态路径，优化后仍慢于本轮固定电荷 GPU 积分参考。当前仍保留完整 Frame 的 O(N) 刷新、全体系 CPU 坐标打包/上传、PME 电荷系数上传和 CPU 积分约束；这些是进一步优化的边界。

## 正确性复测

- 独立接口测试通过，新增填充原子、DD 重排、缓存空间复用、两个模拟独立持有缓存及无效映射拒绝检查。
- CUDA 构建中的 CPU/CPU、GPU/CPU、GPU/GPU、GPU PME + CPU FFT 四条路径全部通过；普通 1–4 成键对照通过。
- CPU-only 构建也完成编译链接，CPU 普通及成键体系回归均通过，固定/动态电荷对照的混合字段最大差为 0。
- 52 帧固定拓扑独立参考通过，误差水平与优化前相当。
- 两种大体系的 40 步固定拓扑/动态电荷对照通过，覆盖约束、周期镜像、温度耦合和邻居表重建。
- 优化版 139,968 原子连续变电荷运行 200 步，60,300 条电荷记录全部符合预测。选区 O/H/H 电荷按 `q_O=-0.82-0.02*sin(2*pi*step/160)`、`q_H=-q_O/2` 改变。
- 在 0、40、80、120、160、200 步分别取相同二进制坐标，用对应固定电荷拓扑关闭扩展，执行 CPU 非键 + CPU PME 独立重算。力相对 RMS 差为 **1.18e-6–1.20e-6**，最大力分量差不超过 **0.008545 kJ/(mol·nm)**，全部通过。未发现重建后缓存失效或旧电荷滞留的问题。

这些测试没有验证长期统计性质、checkpoint 续跑、压力耦合、其他 GPU 后端或真实 ML 能量/力导数一致性。

## 文件和复现

主要实现：

- `src/gromacs/mdlib/dynamicchargeprovider.h`：缓存与映射；
- `src/gromacs/mdlib/sim_util.cpp`：每步数据填充、统计量和预测调度；
- `src/gromacs/mdtypes/forcerec.h`、`src/gromacs/mdlib/forcerec.cpp`：缓存所有权；
- `src/gromacs/nbnxm/atomdata.cpp`、`atomdata.h`、`nbnxm.cpp`、`nbnxm.h`：选区打包电荷更新。

工作区原始结果：

- 优化前：`tests/large-benchmark/results.json` 及各次日志；
- 优化后：`tests/large-benchmark-optimized/results.json` 及各次日志；
- 连续变电荷独立参考：`tests/large-benchmark-optimized/139968/varying/results.json`；
- CUDA 构建：`optimization-build-cuda.log`；
- CPU-only 构建/回归日志：`optimization-build-cpu.log`、`optimization-smoke-cpu.log`。

在工作区运行：

```bash
GMX_DYNAMIC_CHARGE_BENCHMARK_TAG=optimized python3 tests/dynamiccharge_large_benchmark.py build-dynamiccharge-cuda/bin/gmx
GMX_DYNAMIC_CHARGE_BENCHMARK_TAG=optimized python3 tests/dynamiccharge_large_varying.py build-dynamiccharge-cuda/bin/gmx
```

`GMX_DYNAMIC_CHARGE_BENCHMARK_TAG` 仅由测试脚本使用，用于独立保存结果及启用变化电荷测速，不是引擎运行开关。历史补丁保留为工作区 `dynamic-charge-2026.4-before-optimization.patch`，当前补丁为 `dynamic-charge-2026.4.patch`。本次修改位于工作区；桌面原始源码未写回。
