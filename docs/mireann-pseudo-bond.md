# MIREANN 真实原子赝键边界（2026-10-25）

修改标记：`GMX_MIREANN_PSEUDOBOND`。这是本地扩展，不是官方 GROMACS 功能。

## 输入

在原有 REANN PES + MIREANN 的 MDP 中增加：

```ini
nnpot-boundary = pseudo-bond
; 按模型输入组顺序给出训练标签。COX 第一个真实 C 使用模型标签 F。
; 完整列表见 COX 测试的 single.mdp
nnpot-reann-elements = F C H H O H O C C H H H O C C C O O C H C H C H C H
mireann-boundary = pseudo-bond
; 以下仅为编号示例，必须替换成实际 MM 背景置零原子编号
mireann-zero-atoms = 7 8
mireann-switch-width = 0 ; 默认硬选区；可选平滑见 mireann-methods.md
; 默认 yes：保留完整总能量对动态电荷的残余响应，用于能量梯度一致性
mireann-charge-response = yes
```

`mireann-atoms` 和 NNP 的输入组仍必须包含同一组真实 QM 原子。
`mireann-zero-atoms` 使用全局、从 1 开始的 MM 原子编号，可为空。
零背景原子必须位于选中的 MM 环境内，不能属于 QM。

**zero-mask 原子从模型背景列表中排除**，不是保留坐标后仅将输入电荷设成零。
这些原子仍保留在经典参考扣除列表中；其物理电荷沿用下述一次性全 MM 补偿规则；
它们与 QM 的非排除经典近区静电仍进入扣除项。没有偷偷修改总体系电荷，
没有把这些 MM 原子的质量、元素或积分坐标改成虚构的 H。
模型背景排除不等于物理零电荷。即使 TOP 电荷写成零，以下均摊规则也可能将它改变；
若要求严格物理零电荷，必须另外核对初始化约定，不能把 zero-mask 当作此功能。
模型电荷总和必须等于模型目标值。初始化一次性在所有非 QM 原子上均摊
`(原 TOP 的 QM 总电荷 − 模型目标 QM 总电荷)/N_MM`，包括 zero-mask 和球外 MM，
以保留原体系总电荷；此后每步只提交模型 QM 电荷，不重复均摊。

## 边界含义

赝键的势能面来自**针对该赝键边界训练的模型**。本接口不凭空构造一套通用赝势，
不会把完整分子模型自动转换成赝键模型。需要在输入拓扑中给出训练约定的边界原子类型，
模型元素顺序和质量也必须与训练接口一致；不根据被截断键自动猜测元素或替换参数。
`nnpot-reann-elements` 只覆盖 PES 的模型标签，必须按 QM 输入组顺序逐个列出；
`mireann-elements` 同样给出 MIREANN 的训练标签。未设置 PES 标签时沿用拓扑元素。
两处标签都不能通过修改物理碳的质量或原子序数来代替。

所有模型输入使用真实原子位置，单位仍为 Å，电荷单位仍为 e。
模型力和电荷响应力直接映射到相应真实原子，不使用 link atom 的坐标投影或力重分配。
PES、MIREANN 的 CPU/CUDA 选择和原有设备参数保持一致。

## A–B–C–D 示例

若 A、B 在 QM 区，C、D 在 MM 区：

| 项 | 处理 |
|---|---|
| AB 键 | 用模型替换经典势能；保留无能量 ConnectBonds 连通性 |
| BC、CD 键 | 保留经典项 |
| ABC、BCD 角 | 保留经典项 |
| ABCD 二面角 | 保留经典项 |
| 完全位于 QM 内的角和二面角 | 移除经典项 |
| QM–QM 非键项 | 沿用 NNP 的排除机制 |
| QM–MM 原有 1–2、1–3 排除 | 保留，不从模型局部参考中再扣一次完整库仑项 |
| 跨 QM/MM 普通 1–4 静电 | 单独扣除 C × fudgeQQ × qi × qj / r，含能量、两端力及响应电势 |
| 跨区 1–4 的 LJ 项 | 保留经典项 |

原有整体 PME、模型能量、动态电荷和 Gaussian 导数继续使用。
跨边界排除项参与 PME 电势修正；跨区 1–4 的经典电势与参考扣除电势相互抵消，
不会在额外 VJP 中重新带回已经移除的直接 1–4 响应。

### 与 7luck 对照的响应开关

`mireann-charge-response=yes` 是默认值：对 U_FF(q(R))−U_reference(q(R)) 计算额外响应，
用于所列总能量的完整坐标梯度。`no` 只关闭这份额外残余响应，模型自身的自动微分力仍保留，
用于与 7luck 的力定义对照。`no` 模式的力不包含所列总能量的完整电荷响应，
不应把它的有限差分/长期 NVE 性质与 `yes` 混为一谈。
二者不会改变预测电荷、能量或模型权重。修改后必须重新 grompp 生成 TPR。
Gaussian 导数在两种模式中都保留，这是用户明确要求的区别。

修改标记 `GMX_MIREANN_CROSS14`：
`src/gromacs/mdlib/mirelectrostatics.cpp`、`dynamicchargeoptions.h`、`mireannoptions.cpp`、`mireann.cpp`。
原有代码在分支或 ORIGINAL 注释中保留。

该版不支持涉及 QM 的约束、SETTLE、虚拟位点和 CMAP 等超过四中心的键合项，遇到后明确报错。
纯 MM 水的 SETTLE 不受影响。对重复分子先使用上游 splitEmbeddedBlocks 隔离类型，避免改到其他分子实例。

## 参考与区别

参考 `sander-qmcut6.tar.gz` 内：

- `qchem_set.f` 的 `qchem_setbon/setang/setdih`：只移除完全在 QM 内的键合项。
- `make_qm_mm_list`：跳过拓扑非键排除与 zero-mask 原子。
- `QM_MM_Real_Space`：对 zero-mask 原子另行扣除经典局部静电，并跳过排除对。
- `qchem_setup.f`：用户给出 `qmmask/zeromask`；未发现创建几何补氢的赝键代码。

这个压缩包的预处理 `_qchem_set.f` 与普通 `.f` 中模型调用内容有差异，不能当作
当前七输入 MIREANN 的完整实现。本次借鉴其边界拓扑和 zero-mask 规则，
七输入推理仍使用现有 GROMACS MIREANN 后端。

GROMACS 的默认 NNP LinkAtom 流程保留在 `else` 分支。赝键模式复用其分子拆分、
QM–QM 排除和两中心项处理，不调用 `buildLinkFrontier`。

## 验证边界

已使用本地 COX 示例及其真实训练模型做单点、边界力有限差分与五步短测。
合成小体系另外检查全部 30 个坐标分量，包括跨区 1–4 和电荷响应。
这些历史单点检查不等于长期守恒验证。GPU 基线及本次合并的验证边界见 release-validation-20261102.md。

COX 首帧修正后：两种响应模式的总势能都是 −705489.3772882075 kcal/mol，
与用户 Amber 输出相差约 −0.137 kcal/mol；预测电荷和 QM 模型能量保持原先结果。
`charge-response=no` 时，78 个 QM 力分量相对 Amber 的最大差为 0.0095056075 kcal/(mol Å)，
剩余差主要对应用户要求保留的 Gaussian 导数；当前实现没有提供关闭 Gaussian 导数的开关。
`yes` 时最大差为 2.5810884 kcal/(mol Å)，其完整梯度通过边界有限差分，
不能将此模式说成与 Amber 力定义相同。

仅支持使用动态拓扑电荷的普通 pairs；涉及 QM 的显式固定电荷 pair 函数会明确拒绝。
跨区 1–4 的 MM 端必须处于模型环境内，否则报错要求增大 mm-cutoff。

本次未更改任何服务器文件或服务器运行任务。
