# `pyquda_blending` 用户说明书

## 1. 适用范围与边界

本 skill 提供 blending all-to-all 方法的数组级 building blocks：blending weights、meson/baryon elementals、projected perambulator callback orchestration，以及两点函数 contraction。它可以辅助生成 PyQUDA production driver，但不会替调用者构造随机高模向量、正交补投影、dilution、Dirac solver、MPI communicator 或文件格式。

历史代码中的 optional `momentum_smear_internal` 是 blending 内部的独立 phased-link helper；通用 momentum-smearing workflow 应使用 `$pyquda_momentum_smear`，避免两个 skill 因共享 “momentum” 关键词被同时隐式路由。

证据边界：权重、shape、global-coordinate Fourier phase 和 CPU buffer-`Allreduce` 已有本地测试；真实 stochastic unbiasedness、CUDA-aware MPI、QUDA solves 和生产 correlator 尚未验证。

逐式 estimator、dilution、极限与 promotion gate 见 `reference/PHYSICS_CONTRACT.md`。
当前 release label、可运行测试、E2E manifest 与证据升级流程见 `VALIDATION.md`。

## 2. 如何用自然语言调用

> 使用 `$pyquda_blending`，根据 `N_ev=100, N_st=200` 生成 pion blending elementals 和 projected perambulator driver。我的 PyQUDA `grid_size=(2,2,1,1)`，global lattice extents 是 `(GLx,GLy,GLz,GLt)`；请明确每个 rank 的 `(z0,y0,x0)`、global `3V_3` 和 same-timeslice communicator。

> 用 `$pyquda_blending` 审查 baryon elemental 的 $\Omega_{ijk}^{(3)}$。检查重复 stochastic label 是否只计一次，并用论文 Eq. (37) 逐类对照。

> 使用 `$pyquda_blending` 只生成 callback contract 和输出 shape，不要假定我的 noise ensemble、dilution 或 MRHS 实现；把这些缺失项列为 production gate。

不适合的请求包括 generic smearing、RI renormalization、effective-mass fit 或 ensemble statistics。

## 3. 物理空间与 blending basis

在固定时间片上，颜色-空间向量空间记为 $\mathcal L$：

$$
\dim\mathcal L=N_cV_3=3L_x^{\rm global}L_y^{\rm global}L_z^{\rm global}.
$$

低模子空间 $\mathcal L_1$ 由 $N_{\rm ev}$ 个三维 gauge-covariant Laplacian eigenvectors 张成；高模补空间为

$$
\mathcal L_2=\mathcal L/\mathcal L_1,
\qquad
D_2\equiv\dim\mathcal L_2=3V_3-N_{\rm ev}.
$$

blending basis 把低模 $v_i$ 与从 $\mathcal L_2$ 采样、且已经投影到低模正交补的 stochastic vectors $\eta_r$ 合并：

$$
\phi_i=\begin{cases}
v_i,&0\le i<N_{\rm ev},\\
\eta_{i-N_{\rm ev}},&N_{\rm ev}\le i<N_{\rm ev}+N_{\rm st}.
\end{cases}
$$

本模块接收已经构造好的 `eigvecs`/basis vectors；它不检查 $\eta_r$ 是否真的满足

$$
P_{\mathcal L_1}\eta_r=0,
$$

也不产生 noise expectation。因此，一份 production record 必须说明 noise ensemble、normalization、dilution 和 low-mode orthogonalization tolerance。

## 4. Reweighting 公式与推导

论文定义 successive factors

$$
\omega_n=\frac{D_2-n}{N_{\rm st}-n},
\qquad 0\le n<N_{\rm st}.
$$

对一个 mode tuple $I=(i_1,\ldots,i_r)$，令 $d(I)$ 为 tuple 中**不同的** stochastic high-mode labels 数量。代码统一实现

$$
W(I)=\prod_{n=0}^{d(I)-1}\omega_n,
$$

空乘积定义为 $1$。这给出：

- 两个 indices 都是低模：$W_{ij}=1$；
- 两个不同 high labels：$W_{ij}=\omega_0\omega_1$；
- 两个相同 high labels，或一个低模一个高模：$W_{ij}=\omega_0$；
- 三个不同 high labels：$W_{ijk}=\omega_0\omega_1\omega_2$；
- 三元组中只有两个不同 high labels：$W_{ijk}=\omega_0\omega_1$；
- 只有一个不同 high label：$W_{ijk}=\omega_0$。

blending paper 的 Eqs. (3) 与 (37) 直接支持 successive weights 与 meson/baryon tuple 分类；`blending_tuple_weight` 是把这些分类压缩成“不同 stochastic labels 数量”的实现。重复 high label 只计一次是避免对同一 stochastic contribution 重复 reweight 的关键。

当 `n_st=0` 时，代码只允许 $i<N_{\rm ev}$，并令 $W=1$；这是 standard distillation limit。约束

$$
N_{\rm st}\le D_2=3V_3-N_{\rm ev}
$$

使用的是**global** $V_3$，不能用每个 rank 的 local volume。

## 5. Elementals、Fourier phase 与数据布局

### 5.1 输入布局

`eigvecs` 的布局固定为

```text
(mode, t, z, y, x, color) = (N, Lt, Lz, Ly, Lx, 3)
```

这里是 lexicographic spatial storage，不是 PyQUDA 的 checkerboard `(e,t,z,y,xh)`。公开 momentum tuple 是 `(p_x,p_y,p_z)`，但 shape 和 local offset 是 `(z,y,x)`。

### 5.2 Global-coordinate phase

rank-local block 上的 Fourier phase 必须使用 global coordinates：

$$
\phi_{\mathbf p}(z,y,x)=
\exp\left[-2\pi i\left(
\frac{p_xx_{\rm global}}{L_x^{\rm global}}+
\frac{p_yy_{\rm global}}{L_y^{\rm global}}+
\frac{p_zz_{\rm global}}{L_z^{\rm global}}
\right)\right].
$$

其中

$$
(z_{\rm global},y_{\rm global},x_{\rm global})
=(z_0+z,y_0+y,x_0+x).
$$

如果 spatial communicator 包含多个 ranks，`global_spatial_shape` 和本 rank 的 `local_spatial_offset` 都是 mandatory；否则不同 ranks 会重复从原点构造 phase。

### 5.3 Meson elemental

代码实现

$$
\Phi_{ij}(t,\mathbf p)=W_{ij}
\sum_{\mathbf x,c}
\phi_i^*(\mathbf x,t,c)
e^{-i\mathbf p\cdot\mathbf x}
\phi_j(\mathbf x,t,c).
$$

返回 shape 为

```text
(t, sink_mode, source_mode) = (Lt, N, N)
```

并在同一 global time slices 的 spatial communicator 上做 buffer `Allreduce`。

### 5.4 Baryon elemental

颜色反对称 elemental 为

$$
B_{ijk}(t,\mathbf p)=W_{ijk}
\sum_{\mathbf x}\epsilon_{abc}
\phi_i^a(\mathbf x,t)
\phi_j^b(\mathbf x,t)
\phi_k^c(\mathbf x,t)
e^{-i\mathbf p\cdot\mathbf x}.
$$

返回 `(t,N,N,N)`。代码把 color extent 强制为 $N_c=3$；这不是一个 generic $SU(N_c)$ kernel。

## 6. Optional momentum smearing

`phase_spatial_links_internal` 使用

$$
U_j(x)\longrightarrow e^{+2\pi i k_j/G_j}U_j(x),
$$

然后 `momentum_smear_internal` 调用 PyQUDA `source.gaussianSmear`。该 helper 是为了让 blending skill 自包含，不从另一个 skill import；它不包含完整的 S-to-S、sequential-source 或 phase-sign audit。需要这些功能时应显式改用 `$pyquda_momentum_smear`。

这里必须区分两类 artifact：`$pyquda_momentum_smear` 产生的 quark
source/propagator line，不是 `meson_elemental` 所需的
`(N,t,z,y,x,c)` low-mode/stochastic basis。不能把前者直接作为后者的
`eigvecs`，也不能因为工作流按“smear → blending”书写就假设存在直接
field handoff。二者只能在显式定义的 source callback、Dirac solve 或
observable assembly 层汇合，并分别记录 artifact ID、算法 ownership 和
是否已 smearing。若确实对某个 basis vector 调用本 skill 的
`momentum_smear_internal`，该变换属于 blending basis construction，必须
记录变换前后的 basis provenance，而不是继承 generic source-smearing
输出。

该 private helper 仍执行完整参数域检查：`rho` 必须是 finite positive real，`n_steps` 必须是 non-Boolean positive integer，并满足

$$
\rho^2<\frac{2N}{3}
$$

以保证 PyQUDA Wuppertal parameter $\alpha>0$。`NaN`、Boolean 或 fractional `n_steps` 均 fail closed。

## 7. Perambulator callback contract

投影后的 perambulator 形式为

$$
\tau_{ij}^{\alpha\beta}(t,t_0)
=\langle\phi_i(t)|D^{-1}_{\alpha\beta}(t,t_0)|\phi_j(t_0)\rangle.
$$

`generate_perambulator` 把 source construction、solve 和 sink projection 留给三个 callbacks：

```python
tau = generate_perambulator(
    n_modes=N,
    n_spin=4,
    make_source=make_source,
    solve=solve,
    project_sink=project_sink,
)
```

`project_sink` 必须返回 `(t,sink_spin,sink_mode)`；最终 `tau` 为

```text
(t, sink_spin, source_spin, sink_mode, source_mode)
```

该函数逐 column 串行编排 callbacks，不是 MRHS implementation。production 规模下应由 callback 内部实现 batching/MRHS，并记录 setup、solve、projection、total time 和 true residual。

## 8. 两点函数 contraction 约定

### 8.1 Meson

对 Wilson-like propagator，backward line 使用 $\gamma_5$ Hermiticity：

$$
\tau_{\rm back}(t)=
(\gamma_5\otimes I_N)\,
\tau^\dagger(t)\,
(\gamma_5\otimes I_N).
$$

因此不能把它替换为不带 spin kernels 的裸 `tau.conj().T`。`meson_two_point` 返回 backend `(t,)` connected correlator；整体负号、source-bar kernel 和 Fourier convention 必须与项目的 interpolator 定义一起固定。

### 8.2 Baryon

`baryon_two_point` 收缩 $uud$ 的 direct-minus-exchange 结构：

$$
C_N(t)=C_{\rm direct}(t)-C_{\rm exchange}(t).
$$

输入的 `spin_projector`、`diquark_sink` 和 `diquark_source_bar` 都是显式 $(N_s,N_s)$ matrices。不要假设 source diquark 是 sink diquark 的普通 dagger；其确切共轭由所选 baryon interpolator 决定。

## 9. MPI 与 GPU contract

多 spatial-rank 运行必须同时满足：

1. communicator 只连接拥有同一组 global time slices 的 ranks；若 time decomposed，不能直接使用 world communicator。
2. phase 使用 global shape 与每 rank offset。
3. $D_2$ 使用 global $3V_3$。
4. `Allreduce(send,recv)` 是 mpi4py-style buffer collective。
5. CuPy input 只有在目标 MPI stack 已验证 CUDA-aware 时才允许 `cuda_aware_mpi=True`；该 opt-in 必须是 Python/NumPy Boolean，整数、字符串和任意 truthy object 都在 communicator early return 前拒绝。代码没有隐藏的全数组 host fallback。

当前测试中的 two-rank communicator 是 stub，只验证 buffer/API 与 global-coordinate arithmetic，不证明真实 MPI transport。

## 10. 完整 public API 索引

| API | 作用 | 关键边界 |
|---|---|---|
| `omega` | 单个 successive blending factor | 使用 global $D_2$ |
| `blending_tuple_weight` | 按不同 high-mode labels 组合权重 | `n_st=0` 时拒绝 high index |
| `phase_spatial_links_internal` | 内部 spatial-link phase helper | 通用流程优先用 momentum-smearing skill |
| `momentum_smear_internal` | 内部 phased-link Gaussian smearing | 不含完整 source/sink workflow |
| `spatial_fourier_phase` | rank-local global-coordinate phase | momentum 为 $(p_x,p_y,p_z)$ |
| `meson_elemental` | meson blending elemental | 返回 $(t,N,N)$ |
| `baryon_elemental` | color-antisymmetric baryon elemental | 返回 $(t,N,N,N)$，$N_c=3$ |
| `generate_perambulator` | callback-based projected perambulator | 串行 orchestration，不是 MRHS 本身 |
| `meson_two_point` | 指定 kernels 下的 meson contraction | 完整 Wick/sign convention 属于本实现 |
| `baryon_two_point` | $uud$ direct-minus-exchange contraction | source/sink diquark convention 必须显式给出 |

最小调用骨架为：

```python
from skills.pyquda_blending.scripts.Def_blending import (
    blending_tuple_weight,
    meson_elemental,
    spatial_fourier_phase,
)

weight = blending_tuple_weight(indices, global_spatial_dimension, n_ev, n_st)
phase = spatial_fourier_phase(
    local_shape_zyx, momentum_xyz,
    global_shape=global_shape_zyx, local_offset=local_offset_zyx,
)
phi = meson_elemental(
    eigvecs, momentum_xyz, n_ev, n_st,
    global_spatial_shape=global_shape_zyx,
    local_spatial_offset=local_offset_zyx,
    spatial_comm=spatial_comm,
)
```

示例中的 communicator、offset、callbacks 与 backend 参数必须按实际 decomposition 补齐；它不是可直接提交到集群的 production driver。

## 11. 推荐生产工作流

1. 固定 gauge revision、LapH operator、smearing links、eigensolver tolerance 和 eigenvector normalization。
2. 构造 $N_{\rm ev}$ low modes；检查 orthonormality。
3. 在 $\mathcal L_2$ 生成 stochastic vectors；保存 seed、noise ensemble、projection residual 和 dilution labels。
4. 为每 rank 计算 `(GLz,GLy,GLx)` 与 `(z0,y0,x0)`，并与 PyQUDA `grid_size/grid_coord` 分开记录，建立 same-timeslice spatial communicator。
5. 用 `meson_elemental`/`baryon_elemental` 生成 elementals，保存 `N_ev/N_st` 与 weight convention。
6. 用 MRHS-aware callbacks 生成 perambulator，记录 solver evidence。
7. 调用 meson/baryon contraction；用小 $N$ explicit-loop oracle 检查 einsum 和 direct/exchange sign。
8. 先在 `N_st=0` 对照 standard distillation，再增加 stochastic samples 检查 ensemble expectation，而不是用一个 noise sample 声称 unbiased。

## 12. 常见错误检查表

- [ ] 是否把 local $3V_3$ 传给了 `omega`？
- [ ] repeated high label 是否错误地重复乘 $\omega$？
- [ ] phase 是否在每个 rank 都从零开始？
- [ ] momentum `(px,py,pz)` 与 offset `(z0,y0,x0)` 是否混序？
- [ ] color extent 是否恰为 3？
- [ ] spatial communicator 是否混入不同 global time slices？
- [ ] CuPy `Allreduce` 是否有真实 CUDA-aware MPI 证据？
- [ ] stochastic complement 是否真的与 low modes 正交？
- [ ] callback loop 是否被误称为 MRHS？
- [ ] 单个随机样本是否被误称为无偏性的数值证明？

## 13. 验证证据与未验证边界

**已验证：**

- Eqs. (3)/(37) 支持的 reweighting 分类，以及 Eqs. (31)/(32) 支持的 elemental 结构；
- `meson_two_point`/`baryon_two_point` 的完整 Wick、kernel 与整体符号是 bundled implementation convention，不由上述论文公式 blanket 验证；
- `n_st=0` distillation limit；
- global phase reconstruction、global $3V_3$ 和 `N_c=3` validation；
- NumPy buffer `Allreduce` contract；
- distinct fake-CuPy type 下的 strict-Boolean CUDA-aware opt-in gate；这只证明 dispatch fail closed，不证明真实 CUDA-aware transport；
- perambulator axis order；
- meson $\gamma_5$-Hermiticity array identity；
- baryon direct-minus-exchange 的独立 mode/spin explicit-loop oracle；
- CPU/stub compile 与 shape tests。
- scheduler-free E2E contract 的 LapH/perambulator/exact-reference ID、
  SHA-256、轴顺序、residual、seed/dilution provenance 交叉检查；这仍是
  metadata-only evidence，不是生产运行或无偏性证明。

**未验证：**

- stochastic complement generator、dilution 和 ensemble-unbiasedness；
- real CUDA/CuPy、CUDA-aware MPI、multi-rank/multi-GPU；
- QUDA solver convergence、MRHS speedup 和 production memory；
- interacting correlator、statistical efficiency 或论文数值复现；
- project-specific I/O/schema/checkpointing。

## 14. 参考文献

1. Z.-C. Hu, J.-H. Wang, X. Jiang, L. Liu, S.-H. Su, P. Sun, and Y.-B. Yang, “Realization of all-to-all fermion propagator for the first principle high accuracy strong interaction prediction,” [arXiv:2505.01719v2](https://arxiv.org/abs/2505.01719). Eq. (1) 后的 unnumbered definition 给出 $\omega_n$；Eqs. (3), (31), (32), and (37) 支持 tuple reweighting 与 elemental 分类。
2. C. Egerer, R. G. Edwards, K. Orginos, and D. G. Richards, “Distillation at High-Momentum,” *Phys. Rev. D* **103**, 034502 (2021), [arXiv:2009.10691v1](https://arxiv.org/abs/2009.10691v1), DOI: 10.1103/PhysRevD.103.034502. Section II.A, Eq. (4) 仅用于 optional momentum-smeared distillation 背景。
3. M. Peardon et al., “A novel quark-field creation operator construction for hadronic physics in lattice QCD,” *Phys. Rev. D* **80**, 054506 (2009), [arXiv:0905.2160](https://arxiv.org/abs/0905.2160), DOI: 10.1103/PhysRevD.80.054506. 作为 standard distillation/LapH 背景；本地代码没有复现其完整 eigensolver 或 distillation pipeline。

## 逐函数物理动机与证据卡

下表为 public API 的逐项 contract。每一行同时给出物理动机/公式、数据与并行约定、以及证据等级；“实现约定”表示代码代数已检查，但不是论文对完整 production observable 的背书。

| Public symbol | 物理动机与公式 | 输入输出、轴与适用域 | 证据状态与 oracle |
|---|---|---|---|
| `omega` | 为 stochastic high-mode label $n$ 构造 inclusion reweighting，$\omega_n=(3V_3-N_{\rm ev}-n)/(N_{\rm st}-n)$，避免有限 stochastic subset 对 mode tuple 的偏置。 | 输入是 non-Boolean integers；$0\le n<N_{\rm st}$。输出 host scalar，不涉及 backend/MPI。 | 论文直接：arXiv:2505.01719v2 Eq. (1) 后的 unnumbered definition；边界与 Boolean rejection 为实现 contract，已有标量 oracle。 |
| `blending_tuple_weight` | 对 meson/baryon mode tuple 中不同 stochastic labels 乘 successive $\omega_n$；$N_{\rm st}=0$ 严格退化为 distillation weight 1。 | indices 必须位于 $0,\ldots,N_{\rm ev}+N_{\rm st}-1$；重复 high label 只计一次。 | 论文直接：meson Eq. (3)、baryon Eq. (37)；distillation limit 与 Boolean index 已回归。 |
| `phase_spatial_links_internal` | 实现 link-phased Wuppertal step，$U_j(x)\to e^{2\pi i k_j/L_j}U_j(x)$，而不是在完成的 field 上乘 plane wave。 | `k_mode=(k_x,k_y,k_z)` 可为 finite fractional modes；built-in/`UserList`/custom non-string `Sequence`/object-array mixed Boolean在 float conversion与gauge copy前拒绝；只改空间 links。 | 方法直接：arXiv:2009.10691v1 Section II.A, Eq. (4)；generic-Sequence provenance regression是实现证据，真实 device field 未验证。 |
| `momentum_smear_internal` | 以 phased links 反复施加 Gaussian/Wuppertal kernel，改善给定 boost 附近的 source overlap。 | 输入 PyQUDA field/gauge；要求 $\rho>0$、integer $N>0$ 且 $\rho^2<2N/3$，返回同类 backend field。 | 论文支持 operator class；参数到 PyQUDA `gaussianSmear` 的映射为上游实现约定；未做 QUDA/interacting test。 |
| `spatial_fourier_phase` | 为 elemental 构造 $e^{-2\pi i(n_xx/G_x+n_yy/G_y+n_zz/G_z)}$，把 exact hadron momentum 与 smearing mode 分离。 | 输出 local `(z,y,x)` array；generic non-string Sequence/object-array mixed Boolean在 conversion前拒绝；multi-rank必须给 global shape/offset。 | Fourier 定义直接；global-coordinate与 generic-Sequence Boolean CPU oracles支持，checkerboard不在本函数内。 |
| `meson_elemental` | 构造 spatial/color factor $\Phi_{ab}(t)=\sum_{\mathbf x,c}v_a^\dagger e^{-i\mathbf p\cdot\mathbf x}v_b$ 并乘 blending weights。完整 hadron operator仍需 spin/flavor 因子。 | `eigvecs=(N,t,z,y,x,3)`；`cuda_aware_mpi` strict Boolean gate在任何 shape access/phase/contraction前执行；返回 `(t,a,b)`。 | 论文直接：Eqs. (31),(32) spatial/color 部分；early-gate/fake-CuPy只证明拒绝路径，不证明完整 pion operator或真实 MPI。 |
| `baryon_elemental` | 构造 $\epsilon_{abc}v_i^av_j^bv_k^c e^{-i\mathbf p\cdot\mathbf x}$ 和三-index blending weight，为 baryon Wick contraction提供颜色/空间因子。 | 同一 `(N,t,z,y,x,3)` 输入；strict Boolean opt-in同样在大 contraction前执行；返回 `(t,i,j,k)`，固定 $N_c=3$。 | 论文直接：Eq. (37) reweighting；early-gate regression是实现证据，spin/flavor/Wick与真实 MPI仍是 caller contract。 |
| `generate_perambulator` | 实现投影 all-to-all line $\tau=V^\dagger D^{-1}V$ 的 mode-spin columns；solve 由 callback 注入，避免 helper 隐式决定 action/solver。 | callbacks 逐 `(source_mode,source_spin)`；projected column 必须 `(t,sink_spin,sink_mode)`，返回 `(t,s_s,s_r,n_s,n_r)`。 | 论文直接：arXiv:2505.01719v2 Eq. (4) 的对象；串行 callback orchestration 是实现约定，MRHS/QUDA 性能未验证。 |
| `meson_two_point` | 组合 source/sink elementals、perambulator 与 $\gamma_5$-Hermitian backward line，形成 connected meson two-point Wick trace。 | 所有 arrays 同 backend；perambulator 轴为 `(t,sink_spin,source_spin,sink_mode,source_mode)`；`t_source`为 range内 non-Boolean integer并在 array access前验证；输出 `(t,)`。 | 代数实现有非对称 complex matrix与 Boolean-time pre-array oracles；完整 flavor、disconnected term 与 ensemble normalization 未实现。 |
| `baryon_two_point` | 对三条 perambulator lines 计算颜色/模式 elementals 与 spin kernels 的 direct-minus-exchange，落实 identical-quark antisymmetry。 | source time为 strict integer；三条 $\tau$、两个 elementals 与 spin matrices 必须形状/backend 一致，输出 `(t,)`。 | explicit mode-spin nested-loop与 pre-array time oracles直接支持数组代数；论文只支持 operator class，完整 spectroscopy 未验证。 |
