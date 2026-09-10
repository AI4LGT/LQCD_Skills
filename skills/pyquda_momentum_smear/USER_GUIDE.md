# `pyquda_momentum_smear` 用户说明书

## 1. 适用范围与证据等级

本 skill 封装的是 PyQUDA 上的 link-phased Wuppertal momentum smearing：对 source、sink propagator 和 meson sequential source 使用带动量相位的空间 links，并把 `mrhs`、`restart` 等求解器参数显式传给 PyQUDA。它适合生成或审查每个 gauge configuration 上的 smearing/inversion driver，不负责 gauge I/O、配置循环、收缩后的统计拟合或物理结果解释。

本文采用三种证据标签：

- **直接支持**：可由 bundled code、当前 `/Users/zhaodianjun/PyQUDA` API 或列出的原始论文直接核对。
- **实现推导**：由代码中的符号与离散算子推出，必须与调用者的 propagator/source 约定一起检查。
- **待验证**：需要真实 CUDA/QUDA/MPI、目标 gauge ensemble 或 correlator 数据才能证明。

本地已有的证据是 CPU/stub、shape、phase-order 和静态 API 测试；没有运行真实 CUDA/QUDA inversion。因此，本文不声称已验证 solver convergence、GPU performance、最佳 smearing 参数或生产 correlator。

完整 source/sink/Fourier/sequential-dagger 符号链见 `reference/PHYSICS_CONTRACT.md`。
当前 release label、可运行测试、interacting E2E manifest 与证据升级流程见 `VALIDATION.md`。

## 2. 如何用自然语言调用

可以直接对 Codex 说：

> 使用 `$pyquda_momentum_smear`，为 pion 的 source-to-sink momentum smearing 写一个 PyQUDA driver。source mode 是 `(0,0,1.5)`，sink mode 是 `(0,0,2.0)`，物理末态动量是 `P_f=(0,0,4)`；请分别标出 smearing mode、sink Fourier phase 和 insertion transfer momentum。

> 用 `$pyquda_momentum_smear` 审查这个 sequential-source 脚本。重点检查 `MomentumPhase.getPhase` 的正负号、active leg 是否重复 sink smearing，以及 `mrhs/restart` 是否实际传给 inversion。

> 使用 `$pyquda_momentum_smear` 生成 degenerate nucleon 的 S-to-S source/sink smearing，但不要替我猜最佳 `k/P`；输出需要记录 gauge revision、边界条件、precision、true residual、setup/solve/total time。

不应这样调用：

> 用 `$pyquda_momentum_smear` 完成有效质量拟合并给出物理质量。

后者应交给 correlator/fit workflow；本 skill 只覆盖 smearing 与相邻的 inversion/sequential-source glue code。

## 3. 坐标、轴顺序与符号约定

### 3.1 PyQUDA 数据约定

- `x_src` 使用 PyQUDA 坐标顺序 `(x,y,z,t)`。
- `LatticeFermion`/`LatticePropagator.data` 的 lattice 前缀是 checkerboard `(e,t,z,y,xh,...)`，其中 `e=2`。
- `k_mode=(k_x,k_y,k_z)` 是可为 fractional 的无量纲 mode number；相位角为

  $$
  \kappa_j = \frac{2\pi k_j}{L_j^{\rm global}},
  \qquad L_j^{\rm global}\in\{GL_x,GL_y,GL_z\}.
  $$

- `k_mode` 是 quark-smearing kernel 的中心，不等同于 hadron 的精确 Fourier momentum。
- `P_i`、`P_f` 表示初末态 hadron momentum；三点函数 insertion transfer 定义为

  $$
  q=P_f-P_i.
  $$

### 3.2 Link phase

代码只修改三个空间方向，保持 time links 不变：

$$
U_j^{(k)}(x)=e^{+i\kappa_j}U_j(x),\qquad j=x,y,z,
$$

$$
U_t^{(k)}(x)=U_t(x).
$$

这是 Bali et al. momentum-smearing Eq. (24) 的直接实现。`phase_spatial_links` 总是先复制 gauge，避免静默覆盖输入 links。

## 4. Wuppertal kernel 与参数推导

忽略整体归一化，带相位的一步 Wuppertal operator 可写成

$$
\begin{aligned}
(\Phi^{(k)}q)(x)=q(x)+\alpha\sum_{j=1}^{3}\big[&
U_j(x)e^{+i\kappa_j}q(x+\hat{j})\\
&+U_j^\dagger(x-\hat{j})e^{-i\kappa_j}q(x-\hat{j})\big].
\end{aligned}
$$

重复 $N$ 次得到 $(\Phi^{(k)})^Nq$。因此，momentum smearing 必须在每一步 hopping 中使用 phased links；先做普通 smearing 再把最终 field 乘一个 plane wave 不是同一算子。

当前 PyQUDA `source.gaussianSmear(field,gauge,rho,n_steps)` 使用

$$
\alpha=\frac{1}{4N/\rho^2-6}.
$$

为了保证 $\alpha>0$，wrapper 要求

$$
\rho^2<\frac{2N}{3},\qquad \rho>0,\quad N\ge 1.
$$

实现进一步要求 $\rho$ 为 finite real、$N$ 为 non-Boolean integer；`NaN`、Boolean 或 fractional `n_steps` 不会穿过 positive-$\alpha$ gate。

这只是 PyQUDA 参数化的定义域检查，不证明某组 $(\rho,N)$ 对给定 hadron/ensemble 最优。

### 4.1 Free-field momentum center

在 $U_j=1$ 时，plane wave $q(x)\propto e^{i\mathbf{p}\cdot\mathbf{x}}$ 的 hopping 部分给出

$$
\lambda(\mathbf{p})\propto 2\sum_{j=1}^{3}\cos(p_j+\kappa_j).
$$

因此，这一代码符号下 kernel 的峰值位于 $p_j\simeq-\kappa_j$。这是**实现推导**；把 `k_mode` 转换为某条 quark line 或整个 hadron 的首选动量时，必须同时追踪 propagator 的共轭、source/sink 定义和最终 Fourier phase。

## 5. Meson、baryon 与非前向约定

### 5.1 Meson source/sink pair

`momentum_smear_meson` 构造两条 source-smeared lines：

- spectator 使用 `+k_i`；
- active line 使用 `-k_i`；
- S-to-S 时，sink 分别使用 `+k_f` 与 `-k_f`；
- `active_sink_k=-k_f` 被记录下来，供 sequential source 对 active endpoint 恰好再 smearing 一次。
- S-to-P 没有 sink-smearing endpoint，因此必须令 `sink_k_mode=None`；若仍传入非空 $k_f$，函数会在 import、point source 和 inversion 前拒绝，避免 provenance 被静默丢弃。

在本实现的共轭路径下，free-field heuristic 为

$$
P_{\rm meson}\sim (+k)-(-k)=2k,
$$

所以常见起点是 $k\sim P/2$，但 interacting ensemble 上的最优比例属于**待验证假设**，不能由本 API 自动推断。

### 5.2 Degenerate baryon

`momentum_smear_baryon_degenerate` 返回一条可复用的 equal-mode propagator。由同一 link-phase 约定得到的 heuristic 是

$$
P_{\rm baryon}\sim-(k_1+k_2+k_3),
$$

equal sharing 时常从 $k\sim-P/3$ 开始扫描。若 flavors/masses 不等，应分别构造并记录 $k_1,k_2,k_3$；degenerate helper 不代表 unequal-mass baryon。

### 5.3 Smearing momentum 不替代 Fourier projection

两点函数的精确 sink momentum 仍由

$$
C_2(t,\mathbf{P})=\sum_{\mathbf{x}}
e^{-i\mathbf{P}\cdot(\mathbf{x}-\mathbf{x}_0)}C_2(t,\mathbf{x})
$$

决定。改变 `k_mode` 主要改变 overlap；若最终 Fourier phase 未同步，不能把输出解释为另一个精确 lattice momentum。

## 6. Sequential-source phase 的完整共轭链

当前 PyQUDA `MomentumPhase.getPhase(P,x0)` 返回

$$
e^{+i\mathbf{P}\cdot(\mathbf{x}-\mathbf{x}_0)}.
$$

`fourier_phase_pair` 因而返回一对 algebraic candidates：

$$
\phi_{\rm sink}(P_f)=e^{-i\mathbf{P}_f\cdot(\mathbf{x}-\mathbf{x}_0)},
$$

$$
\phi_{+}(P_f)=e^{+i\mathbf{P}_f\cdot(\mathbf{x}-\mathbf{x}_0)}
=\phi_{\rm sink}^*(P_f).
$$

只有当项目的下游 contraction **明确**对这条 sequential line 做 dagger/conjugate 时，$\phi_+$ 才是与物理 $+P_f$ sink 对应的 sequential-source phase；经过该共轭后才恢复 $\phi_{\rm sink}$。`fourier_phase_pair`、`build_sequential_source` 与当前 PyQUDA `source.sequential12` 本身都不执行这个 dagger；`sequential12` 只提取 sink time slice。因此不能从 helper 名称静默推断正相位必然正确，必须追踪最终 contraction 的完整 index/conjugation chain。

若下游确有该 dagger，则在 sequential source 和 final sink contraction 都使用 $e^{-iP_f x}$ 会给出相反的有效符号；若下游没有 dagger，则应重新从目标三点函数定义推导 source phase，而不是套用本段结论。

对非前向三点函数：

- `fourier_phase_pair(...,P_f,...)` 只处理末态 sink/sequential pair；
- insertion 上的 $e^{\pm iq\cdot x}$ 由 contraction layer 负责；
- 代码生成时必须明确采用的 $q=P_f-P_i$ 还是相反定义。

## 7. API 与推荐工作流

| API | 作用 | 关键输入/输出 |
|---|---|---|
| `MomentumSmearResult` | primary meson result record | source/sink lines 与 `active_sink_k` |
| `MesonMomentumSmearResult` | `MomentumSmearResult` compatibility alias | 不产生新 storage |
| `BaryonMomentumSmearResult` | degenerate-baryon result record | one reusable line |
| `SingleMomentumSmearResult` | generic one-line result record | raw/sink propagators |
| `negate_mode` | validate 并反号三分量 mode | three-float tuple |
| `phase_spatial_links` | 复制 gauge 并相位化 spatial links | `k_mode=(kx,ky,kz)` |
| `momentum_smear_kernel` | 调用 phased-link `gaussianSmear` | 同类 PyQUDA field |
| `apply_momentum_smear_source` | source smearing 后 `invertPropagator` | 显式 `mrhs,restart` |
| `apply_momentum_smear_sink` | propagator sink endpoint smearing | 不做 inversion |
| `apply_momentum_smear_seqprop` | one-smear contract 后 sequential inversion | `already_smeared` mandatory |
| `momentum_smear_propagator` | 单条 line 的 source 与可选 sink wrapper | `SingleMomentumSmearResult` |
| `momentum_smear_meson` | `+k/-k` meson lines | `MomentumSmearResult` |
| `momentum_smear_baryon_degenerate` | equal-mode degenerate baryon line | `BaryonMomentumSmearResult` |
| `momentum_smear_s_to_p`, `momentum_smear_meson_s_to_p` | meson S-to-P compatibility aliases | fixed `sink_mode='s2p'` |
| `momentum_smear_s_to_s`, `momentum_smear_meson_s_to_s` | meson S-to-S compatibility aliases | fixed `sink_mode='s2s'` |
| `momentum_smear_baryon_s_to_p` | baryon S-to-P compatibility alias | fixed `sink_mode='s2p'` |
| `momentum_smear_baryon_s_to_s` | baryon S-to-S compatibility alias | fixed `sink_mode='s2s'` |
| `fourier_phase_pair` | sink phase 与 conditional positive candidate | 必须审计下游 dagger |
| `build_sequential_source`, `build_meson_sequential_source` | primary/alias fixed-sink source builder | 必要时 active-leg smearing 一次 |

`build_sequential_source` 先把 caller-supplied phase 逐点乘到完整 checkerboard spectator block，再由 `source.sequential12` 选择 sink time slice。它要求 exact `(e,t,z,y,xh,4,4,3,3)` layout、`e=2`、phase/propagator/gamma 同 backend，以及 gauge/field/lattice decomposition 一致。返回后 required active-end operation 已完成；随后应这样避免第二次 smearing：

```python
result = momentum_smear_meson(
    latt_info, gauge, dirac, x_src, k_i, rho, n_steps,
    sink_mode="s2s", sink_k_mode=k_f, mrhs=mrhs, restart=restart,
)
sink_phase, positive_candidate = fourier_phase_pair(latt_info, P_f, x_src)
seq_source = build_sequential_source(
    result, latt_info, t_sink, positive_candidate,
    gamma_sink_bar, gamma_source_bar, gauge, rho, n_steps,
)
seqprop = apply_momentum_smear_seqprop(
    dirac, seq_source, t_sink, gauge, None, rho, n_steps,
    mrhs=mrhs, restart=restart, already_smeared=True,
)
```

上例只有在下游 contraction 确认 dagger/conjugate 时才可把 `positive_candidate` 解释为物理 sequential phase。若输入是另行构造且尚未做 required sink smearing 的 source，则改用 `already_smeared=False` 并显式传 `sink_k_mode`。

推荐每个 configuration 的顺序：

1. 读取并校验 gauge；记录 ensemble/configuration ID、gauge checksum/revision、boundary condition 和 precision。
2. 建立 `latt_info` 和 Dirac operator，确认 `base_gauge.latt_info`、source field 与 solver decomposition 完全一致。
3. 选择 source `k_i`、可选 sink `k_f`、physical `P_i/P_f`；把四者分开记录。
4. 调用 high-level meson/baryon API；不要手写另一份 link-phase kernel。
5. 用 `fourier_phase_pair(P_f)` 构造候选 phases；先审计下游 dagger，再选择 sequential phase；在 insertion layer 独立实现并测试 $q$ phase。
6. 对每次 inversion 保存 solver type、tolerance、reported/true residual、iterations、setup time、solve time 与 total time。
7. 先用小格点/单 rank 做 phase-sign oracle，再上多 rank/GPU production。

## 8. 常见错误检查表

- [ ] 是否把 `(x,y,z,t)` source coordinate 写成 `(t,z,y,x)`？
- [ ] 是否把 `k_mode` 当成了精确 hadron momentum？
- [ ] 是否误把普通 smeared field 乘 plane wave 当作 momentum smearing？
- [ ] S-to-S active endpoint 是否被 smearing 了零次或两次？
- [ ] 是否确认 `sequential12` 只选 time slice，且真正的 downstream dagger/conjugate 在哪里发生？
- [ ] `apply_momentum_smear_seqprop` 是否显式设置 `already_smeared`，并避免 required smearing 为零次或两次？
- [ ] insertion transfer 是否明确为 $q=P_f-P_i$？
- [ ] `rho,n_steps` 是否满足 positive-$\alpha$ 条件？
- [ ] 是否把 momentum-smeared quark line 误当成 blending
  `(N,t,z,y,x,c)` basis，并直接传给 elemental？两类 artifact 必须独立，
  只在 source callback、solve 或 observable assembly 层显式汇合。
- [ ] 比较性能时 `mrhs,restart`、precision、hardware、tolerance 和 true residual 是否相同？
- [ ] 是否把 stub/compile 结果写成真实 GPU/MPI 证明？

## 9. 已验证与未验证边界

**本地直接验证：**

- 三个 spatial links 的相位和 time link 不变；
- positive-$\alpha$ 参数 gate；
- `mrhs/restart` 传递；
- high-level API 与 compatibility aliases；
- PyQUDA `MomentumPhase` 的 $e^{+iPx}$ 定义、phase pair algebra，以及 `sequential12` 不做 dagger 的边界；
- sequential-source one-smear fail-closed contract、global-time/mode/source bounds 与 lattice-signature gates；
- CPU/stub 下的调用顺序、shape 和静态编译。

**仍未验证：**

- 真实 CUDA/CuPy kernel 行为和 QUDA inversion；
- 多 rank/multi-GPU、halo exchange 与 CUDA-aware MPI；
- interacting ensemble 上的 phase-sign correlator oracle；
- S-to-P/S-to-S 数值等价性、最优 $k/P$、统计增益和生产性能；
- gauge boundary condition、solver residual 与输出 schema 的具体项目集成。

## 10. 参考文献

1. G. S. Bali, B. Lang, B. U. Musch, and A. Schäfer, “Novel quark smearing for hadrons with high momenta in lattice QCD,” *Phys. Rev. D* **93**, 094515 (2016), [arXiv:1602.05525v3](https://arxiv.org/abs/1602.05525), DOI: 10.1103/PhysRevD.93.094515. Eq. (24) 是 link-phased Wuppertal kernel 的主要依据。
2. C. Egerer, R. G. Edwards, K. Orginos, and D. G. Richards, “Distillation at High-Momentum,” *Phys. Rev. D* **103**, 034502 (2021), [arXiv:2009.10691](https://arxiv.org/abs/2009.10691), DOI: 10.1103/PhysRevD.103.034502. 仅作为 momentum-smearing 与 distillation 结合的补充背景。
3. 当前项目的 PyQUDA API 依据是 `pyquda_utils/source.py`、`pyquda_utils/core.py` 和 `pyquda_utils/phase_v2.py`；它们是 runtime API 证据，不是物理有效性文献。

## 逐函数物理动机与证据卡

下表沿完整 data flow 区分 source-smearing mode $k_i$、sink-smearing mode $k_f$、physical Fourier momentum $P_f$ 与 insertion transfer $q=P_f-P_i$。任何 helper 都不能用 $k$ 代替 exact hadron Fourier projection。

| Public symbol | 物理动机与公式 | 输入输出、phase 与 decomposition contract | 证据状态与 oracle |
|---|---|---|---|
| `MomentumSmearResult` | 保存 meson spectator/active 两条 lines及 source/sink $k$ provenance，避免 contraction阶段只剩未标记 propagator。 | record含 source `k_mode`、S-to-P/S-to-S mode、两端 fields和 `active_sink_k`；不含 $P_f$。 | 容器为实现约定；field provenance有 wrapper tests，不能证明物理 momentum。 |
| `MesonMomentumSmearResult` | 为旧调用名保留 meson result语义，不复制数据。 | `MomentumSmearResult` 的 exact alias；序列化时应按 canonical class fields解释。 | 兼容别名；无独立 physics content。 |
| `BaryonMomentumSmearResult` | 保存 degenerate baryon reusable line与 source/sink $k$，供三条 quark lines共享或显式复制。 | record含 `mode,k_mode,propagator,propagator_at_sink,active_sink_k`。 | 容器实现约定；非退化 flavor assignment由 caller负责。 |
| `SingleMomentumSmearResult` | 为 generic single-line helper保留 source与optional sink-smearing provenance。 | legacy三字段位置构造仍有效；末字段 `sink_k_mode` default `None`，新返回会写入 canonical tuple。 | compatibility contract与 provenance regression直接支持。 |
| `negate_mode` | 构造 active/spectator 相反 link phases，$k\to-k$。 | finite three-vector，允许 fractional modes；返回 float tuple。 | 纯代数实现；正负物理含义仍取决于 source/sink convention。 |
| `phase_spatial_links` | 实现 Bali et al. Eq. (24)：$U_j(x)\to e^{2\pi ik_j/L_j}U_j(x)$，$j=x,y,z$。 | copy gauge，仅空间 links改变；global extents为 positive non-Boolean integers，time link不变。 | 论文直接加 link-phase oracle；真实 gauge field/CUDA未验证。 |
| `momentum_smear_kernel` | 用 phased links调用 PyQUDA Wuppertal/Gaussian iteration；$\alpha=(4N/\rho^2-6)^{-1}>0$。 | radius/steps、list/tuple/`UserList`/custom `Sequence`/object-array mixed-Boolean mode与 field/gauge decomposition 全部在 `pyquda_utils` import、gauge copy、smearing前验证；返回 same-kind field。 | validation-order sentinel与上游 `gaussianSmear` mapping直接；QUDA/interacting result未验证。 |
| `momentum_smear_meson` | 生成 source $+k/-k$ spectator/active pair，并可在 sink使用独立 $k_f$；为 boosted meson 2pt/3pt准备 lines。 | `x_src=(x,y,z,t)` global；source/sink modes、domain、solver与decomposition在 point source/import/inversion前完成；S-to-S用 opposite sink modes，S-to-P禁止非空 `sink_k_mode`。 | validation-order、S-to-P provenance gate与 data-flow stub tests直接；$k/P$ 最优比例与 correlator phase不是本函数证明。 |
| `momentum_smear_baryon_degenerate` | 对 degenerate baryon生成一条可复用 source-smeared line及optional sink endpoint。 | global source、independent S-to-S sink mode、MRHS/restart显式且全部 pre-inversion validation；S-to-P禁止非空 sink mode；返回 baryon record。 | validation-order与 provenance regression为实现约定；三-quark momentum sharing与 flavor masses由 caller定义。 |
| `momentum_smear_s_to_p` | 固定 `sink_mode='s2p'`，只做 source endpoint smearing。 | 不允许 caller重复传 `sink_mode`，也不允许 `sink_k_mode`；返回 meson record，sink field等于 unsmeared propagator。 | thin alias及 pre-inversion rejection回归；无额外物理假设。 |
| `momentum_smear_s_to_s` | 固定 `sink_mode='s2s'`，source与sink均施加 link-phased kernel。 | duplicate `sink_mode` fail closed；sink mode默认source mode但可独立给出。 | thin alias；S-to-S增益未由本地 tests证明。 |
| `momentum_smear_meson_s_to_p` | 保留旧 meson-specific S-to-P名称。 | exact alias of `momentum_smear_s_to_p`；同样拒绝非空 `sink_k_mode`。 | 兼容别名及 rejection回归；无独立 physics content。 |
| `momentum_smear_meson_s_to_s` | 保留旧 meson-specific S-to-S名称。 | exact alias of `momentum_smear_s_to_s`。 | 兼容别名；无独立 physics content。 |
| `momentum_smear_baryon_s_to_p` | 固定 baryon wrapper为 source-to-point endpoint。 | duplicate `sink_mode`及非空 `sink_k_mode`均拒绝；返回 `BaryonMomentumSmearResult`。 | thin alias及 pre-inversion gate；无额外 physics validation。 |
| `momentum_smear_baryon_s_to_s` | 固定 baryon wrapper为 source-to-smeared endpoint。 | sink mode默认source $k$或显式独立 $k_f$。 | thin alias；无额外 physics validation。 |
| `apply_momentum_smear_source` | 先 smearing source再调用 $D^{-1}$，产生 source-smeared propagator。 | mode/domain/decomposition与 `mrhs>=1,restart>=0` 在 `core` import、smearing、`invertPropagator`前验证。 | import/inversion sentinel及 forwarding stub直接支持；真实 residual、precision与performance未验证。 |
| `apply_momentum_smear_sink` | 在线性 propagator sink index上施加相同 spatial kernel，构造 S-to-S endpoint。 | propagator与gauge decomposition一致；返回 smeared field，不做 inversion。 | 上游 Gaussian API mapping；sink-index orientation需真实 PyQUDA layout确认。 |
| `apply_momentum_smear_seqprop` | 对 sequential source确保 smearing恰好一次，然后调用 `invertSequential`。 | `already_smeared`显式 Boolean；time/mode/domain/solver/decomposition均在 `core` import与 inversion前验证。 | one-smear、validation-order与 forwarding已有 stub oracle；真实 3pt未验证。 |
| `momentum_smear_propagator` | generic single-line source inversion加optional sink smearing，供非 meson/baryon workflow复用。 | source与optional sink mode先 canonicalize；非法 sink不会先消耗 source solve；返回 record并保存 mode。 | pre-inversion sentinel与 wrapper behavior已回归；不自动选择 hadron Fourier momentum。 |
| `fourier_phase_pair` | 从 PyQUDA $e^{+ip\cdot(x-x_0)}$ 同时返回 physical sink $e^{-iP_f\cdot(x-x_0)}$ 与 later-dagger sequential $e^{+iP_f\cdot(x-x_0)}$。 | momentum为 integer `(Px,Py,Pz)` modes，source为 global `(x,y,z,t)`；返回两个 checkerboard arrays。 | 只读上游 `MomentumPhase`与 conjugate-pair oracle直接支持；适用前提是下游确实 dagger/conjugate。 |
| `build_sequential_source` | 先作 $\Gamma_{sink}\,S_{spectator}\,\bar\Gamma_{src}$，乘 sequential $+P_f$ phase，选 $t_{seq}$，再按 record的 active sink $k$ smearing。 | record/time/layout/backend/gamma/phase/domain/mode在 `opt_einsum`/PyQUDA import前验证；返回 sequential field。 | import sentinel、非交换 gamma explicit-loop与 phase/layout oracles已覆盖；真实 current insertion未验证。 |
| `build_meson_sequential_source` | 保留旧的 meson-specific builder名称，防止现有 scripts断裂。 | exact alias of `build_sequential_source`；同一 phase/dagger/one-smear contract。 | 兼容别名；无独立 physics content。 |
