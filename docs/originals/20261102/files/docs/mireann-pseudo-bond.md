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
mireann-switch-width = 0
; 默认 yes：保留完整总能量对动态电荷的残余响应，用于能量梯度一致性
mireann-charge-response = yes
```

`mireann-atoms` 和 NNP 的输入组仍必须包含同一组真实 QM 原子。
`mireann-zero-atoms` 使用全局、从 1 开始的 MM 原子编号，可为空。
零背景原子必须位于选中的 MM 环境内，不能属于 QM。

<!-- ORIGINAL: 这里的零电荷指模型背景输入中的电荷为零（保留坐标）。 -->
**GMX_MIREANN_PHYSICS_AUDIT：zero-mask 原子从模型背景列表直接排除。**
模型返回的 MM 力按压缩列表映射回真实原子；排除原子没有模型背景力。
这些原子的 GROMACS 拓扑电荷保持不变；
它们与 QM 的非排除经典近区静电仍进入扣除项。没有偷偷修改总体系电荷，
没有把这些 MM 原子的质量、元素或积分坐标改成虚构的 H。
如果训练定义要求某个 MM 原子的物理电荷也为零，应在输入拓扑中明确设置，
并自行满足初始 QM 总电荷与模型总电荷一致的原有检查。

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
这些结果不代表已验证长时间动力学或 GPU 模型推理。

COX 首帧修正后：两种响应模式的总势能都是 −705489.3772882075 kcal/mol，
与用户 Amber 输出相差约 −0.137 kcal/mol；预测电荷和 QM 模型能量保持原先结果。
`charge-response=no` 时，78 个 QM 力分量相对 Amber 的最大差为 0.0095056075 kcal/(mol Å)，
剩余差主要对应用户要求保留的 Gaussian 导数；当前实现没有提供关闭 Gaussian 导数的开关。
`yes` 时最大差为 2.5810884 kcal/(mol Å)，其完整梯度通过边界有限差分，
不能将此模式说成与 Amber 力定义相同。

仅支持使用动态拓扑电荷的普通 pairs；涉及 QM 的显式固定电荷 pair 函数会明确拒绝。
跨区 1–4 的 MM 端必须处于模型环境内，否则报错要求增大 mm-cutoff。

本次未更改任何服务器文件或服务器运行任务。

## 2026-10-25 物理核对与 CUDA 加速

标记为 `GMX_MIREANN_PHYSICS_AUDIT`、`GMX_MIREANN_GPU_AUDIT`。

```ini
; auto：经典电势/参考扣除跟随 mireann-device。
; 也可显式指定 cpu 或 cuda:N，与模型设备分别选择。
mireann-electrostatics-device = auto
```

当模型使用 `mireann-device=cuda:0` 且采用正交盒时，`auto` 自动使用新的 CUDA
逐对电势/参考扣除核；也支持 CPU 模型 + `electrostatics-device=cuda:0`。
非正交盒继续使用原有 CPU/native-PBC 公式。显式 CUDA 经典计算要求编译时开启
`GMX_GPU=CUDA`。跨区普通 1–4 的小型拓扑列表仍在 CPU 上处理。
GPU 核使用双精度，包含 Gaussian 导数和相同的势平移；不减少 PME 镜像项。

后端保留原模型文件及公开 forward，另外建立仅返回能量/电荷的内部方法，
省去随后 VJP 前的那一次无用模型梯度。最终的模型力和剩余电荷响应仍共同求导一次。
环境日志增加 `in_model,model_q_e`，能量日志增加 `model_mm_atoms`，便于核对 zero-mask。

`Potential-shift` 模式的参考项同步扣掉实空间 PME 平移常数，修复截断处的小能量跳变；
`None` 的原公式保持不变。MIREANN 自动关闭 PME 负载调优，避免运行中改变参考项的
alpha/cutoff。普通 GROMACS 与独立 QEq 模式不受此设置影响。

**物理适用范围：** 在固定选区和周期镜像的分支内，`charge-response=yes` 才包含所列
能量的完整电荷链式力。整残基硬选区在残基进出边界时仍可能有能量跳变，不能据此宣称
任意长时间 NVE 严格守恒。当前球内模型也不会自动变成全周期自洽极化模型。
QM 必须能用一份紧凑周期镜像同时送给模型和参考项；不一致时现在明确报错。
本次没有恢复已取消的 COX 生产模拟，也没有把短步测试当作 NVE 漂移验证。
