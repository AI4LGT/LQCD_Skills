# `pyquda_enhanced_interpolator` 用户说明书

## 1. 适用范围与证据边界

本 skill 提供 boosted-hadron kinematic-enhancement 的数组级 building blocks：Euclidean $\gamma_\pm$、quark-component kernel、meson/nucleon source-sink spin kernels、局域 Wick contraction，以及对 checkerboard field 的空间动量投影。它不构造 propagator，不执行 Dirac inversion，不生成 global Fourier phase，也不负责配置 I/O、拟合或物理外推。

本文区分三类结论：

- **直接支持**：可由 bundled code、`reference/API.md` 或列出的原始论文逐项核对。
- **代数推导**：在明确的 Euclidean Clifford algebra 下由代码公式推出。
- **待验证**：需要真实 propagator、interacting ensemble、CUDA-aware MPI 或统计分析才能证明。

本地已有 NumPy/stub 的代数、shape、source-conjugation 和 reduction contract 测试；没有证明 enhanced operator 在目标 ensemble 上一定改善 overlap 或 signal-to-noise。

完整 gamma-basis、source/sink 与“enhanced”声明 gate 见 `reference/PHYSICS_CONTRACT.md`。
当前 release label、可运行测试、E2E manifest 与证据升级流程见 `VALIDATION.md`。

## 2. 如何用自然语言调用

可以对 Codex 说：

> 使用 `$pyquda_enhanced_interpolator`，为沿 $z$ 方向 boost 的 nucleon 构造 `projector_kind='plus'`、`diquark_kind='plus'` 的 source/sink kernels。先声明 Euclidean gamma convention，再解释为什么 source diquark 使用 complex conjugate 而不是 Hermitian adjoint。

> 用 `$pyquda_enhanced_interpolator` 审查这个 `uud` contraction。输入末四轴是 `(sink_spin,source_spin,sink_color,source_color)`；请用 explicit-loop oracle 检查 direct-minus-exchange 的 spin/color indices。

> 使用 `$pyquda_enhanced_interpolator` 对局域 contraction 做 momentum projection。我的 phase 已按 global coordinates 构造；请检查 checkerboard `(e,t,z,y,xh)`、same-timeslice communicator 和 CUDA-aware MPI gate。

不应让本 skill 单独完成 smearing、inversion、effective-mass fit，或仅凭函数名断言 boosted-state overlap 已改善。

## 3. Euclidean gamma 约定

调用前应验证所给 matrices 满足选定基底，例如常用 Hermitian Euclidean convention：

实现首先要求每个 public Dirac-matrix input 都是同一 NumPy/CuPy backend
上的 numerical、finite、exact `(4,4)` matrix；`NaN`、`Inf`、object/string
matrix 与错误 shape 会在 gamma algebra 或 contraction 前 fail closed。这项
数值 gate 不替代 Clifford algebra、charge conjugation 或目标 propagator basis 核对。

$$
\gamma_\mu^\dagger=\gamma_\mu,
\qquad
\{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu}I.
$$

若 boost 方向为 $\parallel$，代码定义

$$
\gamma_+=\frac{\gamma_t+i\gamma_\parallel}{\sqrt2},
\qquad
\gamma_-=\frac{\gamma_t-i\gamma_\parallel}{\sqrt2}.
$$

这对应 nucleon paper 在 Euclidean continuation 下使用的 light-cone-like components。方向 $\parallel$ 必须与 hadron boost、momentum phase 和调用者的 gamma-index ordering 一致。

在上述 Clifford algebra 下，

$$
\gamma_+^2=\gamma_-^2=0.
$$

因此 `gamma_plus` 本身不是 projector。历史 API `plus_quark_projector` 返回

$$
Q_+=\frac{\gamma_-\gamma_+}{\sqrt2},
$$

并保留论文采用的 normalization。按这一 normalization，通常有 $Q_+^2=\sqrt2 Q_+$，而不是 $Q_+^2=Q_+$；不要因函数名静默假定 idempotence。

## 4. Euclidean conjugation 与 meson kernel

### 4.1 两种不同的 source operation

对普通 bilinear matrix $M$，`euclidean_adjoint` 实现

$$
\overline M=\gamma_t M^\dagger\gamma_t.
$$

对 nucleon diquark source kernel $D$，`euclidean_source_conjugate` 则实现

$$
\overline D_{\rm src}=\gamma_t D^*\gamma_t.
$$

第二式没有 $D^T$。这是 source diquark spin-index contraction 的具体约定；把它替换为 $\gamma_tD^\dagger\gamma_t$ 会转置两个 spin indices，通常改变 baryon contraction。

### 4.2 Enhanced meson bilinear

给定原始 bilinear $\Gamma$、right-leg kernel $Q_R$ 和 antiquark kernel $Q_L$，代码构造

$$
\Gamma_{\rm enh}
=\left(\gamma_tQ_L^\dagger\gamma_t\right)\Gamma Q_R.
$$

若未给 `antiquark_projector`，则 $Q_L=Q_R$。函数只做 spin algebra；它不会根据 pion、vector meson 或其他 quantum numbers 自动选择 $\Gamma$。

这里 `enhanced_meson_kernel` 的 `bilinear_gamma` 明确定义为
$\bar q\Gamma q$ 中的 Dirac-bar matrix，而论文中的 projected-pion
identity 从 $u_+^\dagger\gamma_5d_+$ 写起。因为
$\bar u=u^\dagger\gamma_t$，canonical pion 调用必须使用

$$
\Gamma_{\bar q q}=\gamma_t\gamma_5,
$$

而不是直接传 `gamma5`。在本文的 Clifford convention 下，独立代数
oracle 给出

$$
K(\gamma_5)=0,
\qquad
K(\gamma_t\gamma_5)=\sqrt2\,\gamma_+\gamma_5.
$$

因此公开的安全入口是 `enhanced_pion_kernel(gamma5,Qp,gamma_t)`；generic
`enhanced_meson_kernel` 保留给调用者已明确证明 operator identity 的情形。

列出的论文直接支持其中特定的 meson operator choices；公开参数 `bilinear_gamma` 接受任意满足调用者 gamma-basis contract 的 $(4,4)$ matrix，这一 generic algebra extension 不是论文对任意 bilinear 的 whitelist 或物理增益保证。

局域 connected meson contraction 使用 backward antiquark line

$$
S_{\bar q}^{\rm back}=\gamma_5S_{\bar q}^*\gamma_5
$$

及代码约定下的整体负号：

$$
C_{\rm loc}
=-\operatorname{Tr}_{s,c}
\left[
\Gamma_{\rm sink}S_q
\overline\Gamma_{\rm src}S_{\bar q}^{\rm back}
\right].
$$

这个负号和 source-bar convention 必须与项目的 interpolator definition 一起固定，不能从本 helper 单独推广为所有 meson convention。

## 5. Enhanced nucleon kernels 与 Wick contraction

代码对应的 nucleon operator class 可写为

$$
N^\Gamma_\alpha(x)
=\epsilon_{abc}
\left[u_a^T(x)\,C\gamma_5\Gamma\,d_b(x)\right]
u_{c,\alpha}(x).
$$

`enhanced_baryon_kernels` 构造

$$
D_{\rm sink}=C\gamma_5\Gamma,
\qquad
D_{\rm source\_bar}=\gamma_tD_{\rm sink}^*\gamma_t,
$$

其中 `diquark_kind` 可选 `identity`、`time`、`plus`，分别令

$$
\Gamma\in\{I,\gamma_t,\gamma_+\}.
$$

外部 spin kernel `projector_kind` 可选：

$$
T\in\left\{
\frac{I+\gamma_t}{2},\gamma_t,\gamma_+
\right\}
$$

，对应 `parity`、`time`、`plus`。`parity` 是标准 positive-parity kernel；`plus` 仍不应因名称被当作数学上的 idempotent projector。

`baryon_contraction` 对两条 $u$ lines 和一条 $d$ line 实现颜色双 epsilon 的 direct-minus-exchange：

$$
C_N=C_{\rm direct}-C_{\rm exchange}.
$$

exchange 项交换两条 identical-$u$ line 的 source spin/color ownership。修改 einsum 时应使用小数组 explicit spin/color loop 作为 oracle，而不能只检查输出 shape。

## 6. 数据布局与 backend

所有 propagator 的末四轴固定为

```text
(sink_spin, source_spin, sink_color, source_color) = (4,4,3,3)
```

前导轴可以是 batch 或 lattice axes。输入 arrays 必须全部使用同一 NumPy/CuPy backend；spin matrices 也应位于同一 backend。

`meson_contraction` 和 `baryon_contraction` 返回全部前导轴。例如输入是

```text
(e,t,z,y,xh,4,4,3,3)
```

时，局域输出应为

```text
(e,t,z,y,xh)
```

## 7. Momentum projection 与 MPI ownership

`momentum_project` 要求局域 contraction 和 phase 都恰为

```text
(e,t,z,y,xh),  e=2
```

并计算

$$
C(t,\mathbf p)
=\sum_{e,z,y,x_h}
C_{\rm loc}(e,t,z,y,x_h)\,\phi_{\mathbf p}(e,t,z,y,x_h).
$$

函数不生成 $\phi_{\mathbf p}$。若物理 convention 是

$$
\phi_{\mathbf p}(x)=e^{-i\mathbf p\cdot(\mathbf x-\mathbf x_0)},
$$

调用者必须用 global coordinates 自行构造，并确认 checkerboard parity mapping。多 rank 时：

1. `spatial_comm` 只能连接拥有同一组 global time slices 的 ranks；
2. collective 使用 mpi4py-style buffer `Allreduce(send,recv)`；
3. CuPy buffer 只有在目标 MPI stack 已证明 CUDA-aware 后才设置 `cuda_aware_mpi=True`；该值必须是 Python/NumPy Boolean，整数、字符串和任意 truthy object 均在 contraction/early return 前拒绝；
4. 本 helper 不提供 host fallback，也不构造 spatial subcommunicator。

## 8. API 速查

| API | 作用 | 关键约定 |
|---|---|---|
| `EnhancedKernels` | baryon spin-kernel immutable record | 保存外部 kernel 与 source/sink diquarks |
| `euclidean_adjoint` | $\gamma_tM^\dagger\gamma_t$ | bilinear source-bar |
| `euclidean_source_conjugate` | $\gamma_tM^*\gamma_t$ | diquark；不转置 |
| `lightcone_gammas` | 构造 $\gamma_\pm$ | boost direction 必须明确 |
| `plus_quark_projector` | 返回 $Q_+=\gamma_-\gamma_+/\sqrt2$ | 不保证 idempotent |
| `enhanced_meson_kernel` | 两条 meson legs 的 Dirac-bar spin algebra | generic extension；pion 不可直接传 `gamma5` |
| `enhanced_pion_kernel` | canonical $u_+^\dagger\gamma_5d_+$ wrapper | 内部传 $\gamma_t\gamma_5$；exact normalization oracle |
| `enhanced_baryon_kernels` | 返回 projector、sink/source diquarks | source 与 sink kernels 不同 |
| `meson_contraction` | connected local meson contraction | 含 $\gamma_5$ backward line |
| `baryon_contraction` | local $uud$ direct-minus-exchange | 末四轴固定 |
| `momentum_project` | checkerboard spatial sum/reduction | phase 由调用者提供 |

最小数组级调用骨架为：

```python
from skills.pyquda_enhanced_interpolator.scripts.Def_enhanced_interpolator import (
    baryon_contraction,
    enhanced_baryon_kernels,
    momentum_project,
)

kernels = enhanced_baryon_kernels(
    charge_conjugation, gamma5, gamma_t, gamma_parallel,
    projector_kind="plus", diquark_kind="plus",
)
local = baryon_contraction(
    Su_1, Su_2, Sd,
    kernels.projector, kernels.diquark_sink, kernels.diquark_source_bar,
)
projected = momentum_project(
    local, global_coordinate_phase, spatial_comm=spatial_comm,
)
```

实际调用必须显式给出 `projector_kind`、`diquark_kind`、backend 和 `cuda_aware_mpi`，并先核对函数签名；这里省略的对象不是默认物理约定。

## 9. 推荐工作流

1. 记录 gamma matrices 的显式数值或 basis ID，并验证 Clifford algebra、$\gamma_5$ 和 charge-conjugation relation。
2. 固定 boost direction、hadron momentum sign、source/sink phase convention。
3. 用 `lightcone_gammas` 和 high-level kernel helper 构造 spin matrices，保存 `projector_kind/diquark_kind`。
4. 从已验证的 propagators 生成局域 contraction；先用 free/tree-level 或小数组 explicit loop 验证 indices。
5. 独立构造 global-coordinate Fourier phase，再调用 `momentum_project`。
6. 对 standard 与 enhanced operators 使用相同 propagators、smearing、statistics 和 fit windows 比较 overlap；报告 covariance 与失败样本。

## 10. 常见错误检查表

- [ ] $\gamma_t$ 与 $\gamma_\parallel$ 是否满足同一 Euclidean convention？
- [ ] boost direction 是否与 momentum/phase direction 一致？
- [ ] 是否把 `gamma_plus` 或 `plus_quark_projector` 错称为 idempotent projector？
- [ ] source diquark 是否错误用了 $D^\dagger$ 而不是 $D^*$？
- [ ] propagator 的 spin/color source-sink axes 是否混序？
- [ ] exchange term 是否真正交换 identical-$u$ ownership？
- [ ] phase 是否使用 global coordinates？
- [ ] spatial communicator 是否混入不同 global time slices？
- [ ] 是否把 shape/stub test 误写成 enhanced overlap 的 physics proof？

## 11. 已验证与未验证

**本地直接验证：**

- NumPy 下 meson/baryon contraction 的 shape；
- source diquark 使用 complex conjugation without transpose；
- checkerboard `e=2` validation 和 buffer `Allreduce` contract；
- distinct fake-CuPy type 下 strict-Boolean `cuda_aware_mpi` gate；这不是 CUDA-aware MPI 运行证据；
- public API、YAML 与静态编译。

**仍未验证：**

- 目标 gamma basis 与 production propagator 的端到端一致性；
- 真实 CuPy、CUDA-aware MPI 和 multi-rank result；
- interacting ensemble 上 overlap、effective-mass plateau 或 signal-to-noise 改善；
- 配置 I/O、smearing、inversion、renormalization 和 fit pipeline。

## 12. 参考文献

1. Daniel Reitinger, Tobias Sizmann, Andreas Schäfer, Rui Zhang, and Yong Zhao, “Kinematic enhancement for nucleon interpolators,” [arXiv:2606.02447v2](https://arxiv.org/abs/2606.02447v2). Eq. (2.6) 给出 nucleon source/source-bar operator，Eqs. (2.9)-(2.11) 给出一般 $\mathcal T$ kernel 与 positivity 条件；$\gamma_\pm$ 与 projected-quark component 位于 Section 2.1、Eq. (2.2) 之前。
2. Rui Zhang, Anthony V. Grebe, Daniel C. Hackett, Michael L. Wagman, and Yong Zhao, “Kinematically-enhanced interpolating operators for boosted hadrons,” [arXiv:2501.00729v2](https://arxiv.org/abs/2501.00729v2). Page 2 的 “Kinematic enhancement: theory” 段落给出 unnumbered identity $u_+^\dagger\gamma_5d_+=\sqrt2\,\bar u\gamma_+\gamma_5d$；Eq. (12) 给出相对 pseudoscalar 的 SNR scaling。Generic two-leg $Q$ algebra不是该论文对任意 $\Gamma$ 的 whitelist。
3. 本项目 `scripts/Def_enhanced_interpolator.py` 是 exact array-index implementation 的直接依据，但不是目标 ensemble 上物理增益的证据。

## 逐函数物理动机与证据卡

这里把 paper-derived identities、implementation-level algebra 与未验证 physics claim 分开。所有 spin constructors 现在都要求 exact `(4,4)` Dirac space；这项 shape gate 不能替代 Clifford-algebra 或 charge-conjugation 检查。

| Public symbol | 物理动机与公式 | 输入输出、轴与适用域 | 证据状态与 oracle |
|---|---|---|---|
| `EnhancedKernels` | 将 $\gamma_\pm$、$Q_+$、外部 $\mathcal T$、sink/source diquark 放进不可变 provenance record，防止 source conjugation 在调用链中丢失。 | fields 均为同 backend `(4,4)` matrices；`diquark` 仅是 `diquark_sink` compatibility alias。 | 容器为实现约定；field identity 与 alias 有 API contract，物理正确性取决于构造函数输入。 |
| `euclidean_adjoint` | 对 $\bar qMq$ 计算 $\overline M=\gamma_tM^\dagger\gamma_t$，用于普通 bilinear 的 Euclidean source-bar。 | 两个输入均为同 backend `(4,4)`；返回 `(4,4)`，不触及 propagator axes。 | 稳定 Clifford algebra；非对角 complex matrix 顺序由直接 matrix oracle 支持。 |
| `euclidean_source_conjugate` | 对 baryon diquark source 使用 $\gamma_tM^*\gamma_t$；不做 transpose，保留两个 source-spin index 的论文 contraction ownership。 | exact `(4,4)`；与 `euclidean_adjoint` 有意不同。 | arXiv:2606.02447v2 Eq. (2.6) operator chain与实现 index oracle支持；不能泛化到任意 baryon convention。 |
| `lightcone_gammas` | 从 boost direction 构造 $\gamma_\pm=(\gamma_t\pm i\gamma_\parallel)/\sqrt2$，隔离大动量下的 enhanced spin component。 | `gamma_t` 与 `gamma_parallel` 必须同 basis/backend、exact `(4,4)`；方向必须与 momentum label一致。 | 论文直接：arXiv:2606.02447v2 Section 2.1 before Eq. (2.2)；Euclidean continuation sign 需 caller 核对。 |
| `plus_quark_projector` | 返回论文 normalization 的 $Q_+=\gamma_-\gamma_+/\sqrt2$；它提取 projected quark component但一般不满足 $Q_+^2=Q_+$。 | 两个 `(4,4)` matrices；输出 `(4,4)`。 | 论文直接：Section 2.1 before Eq. (2.2)；normalization 与非 idempotence 有 independent Clifford oracle。 |
| `enhanced_meson_kernel` | 计算 Dirac-bar algebra $K=(\gamma_tQ_L^\dagger\gamma_t)\Gamma Q_R$；用于已明确证明的 bilinear choice。 | `bilinear_gamma` 是 $\bar q\Gamma q$ 的 $\Gamma$，四个 matrices 均 `(4,4)`；不推断 flavor/$J^{PC}$。 | 实现约定：generic $\Gamma$ extension；论文仅支持特定 operators。$K(\gamma_5)=0$ exact oracle防止误用。 |
| `enhanced_pion_kernel` | 将 paper identity $u_+^\dagger\gamma_5d_+$ 映射为 Dirac-bar input $\Gamma=\gamma_t\gamma_5$，得到 $K=\sqrt2\gamma_+\gamma_5$。 | 输入 `gamma5,Qp,gamma_t` 均 `(4,4)`；返回 canonical pion kernel。 | 论文直接 operator identity加独立 finite-dimensional oracle；尚未验证 interacting pion SNR。 |
| `enhanced_baryon_kernels` | 为 $N^\Gamma=\epsilon_{abc}(u_a^TC\gamma_5\Gamma d_b)u_c$ 构造 $D_{sink}=C\gamma_5\Gamma$、$D_{src}=\gamma_tD^*\gamma_t$ 与 $\mathcal T$。 | 四个 `(4,4)` inputs；choices 受 whitelist 限制，返回 `EnhancedKernels`。 | 论文直接：arXiv:2606.02447v2 Eqs. (2.6),(2.9)-(2.11)；exact C/gamma basis仍由 caller验证。 |
| `meson_contraction` | 计算 $-\operatorname{Tr}_{s,c}(\Gamma_sS_q\bar\Gamma_r\gamma_5S_{\bar q}^*\gamma_5)$，形成 local connected meson density。 | propagators 末四轴固定 `(4,4,3,3)`，前导 batch/lattice axes 原样返回；所有 arrays 同 backend。 | explicit spin-color loop oracle直接支持 array algebra；整体符号与 flavor convention 是实现 contract。 |
| `baryon_contraction` | 计算双颜色 epsilon 和两条 identical-$u$ 的 direct-minus-exchange，落实 Fermi antisymmetry。 | 三 propagators 同 shape/backend，末四轴 `(4,4,3,3)`；三个 spin kernels `(4,4)`；输出前导 axes。 | Eq. (2.6) operator class与完整 explicit spin/color nested-loop oracle支持；normalization/overlap未验证。 |
| `momentum_project` | 将 local checkerboard density 与 global-coordinate phase相乘，$C(t,\mathbf p)=\sum_{e,z,y,x_h}\phi_\mathbf pC_{loc}$。 | 两数组 exact `(2,t,z,y,xh)`；返回 `(t,)`，optional same-timeslice buffer `Allreduce`；CuPy 的 `cuda_aware_mpi` 为 strict Boolean opt-in。 | Fourier algebra、CPU collective stub和 fake-CuPy错误 opt-in gate支持；phase producer、真实 MPI 与 enhanced-state physics未验证。 |
