# `pyquda_gluon_renorm` 用户说明书

## 1. 先澄清名称：这里主要是 bare observables

本 skill 的历史名称含 `renorm`，但 bundled code 主要构造 bare pure-gauge observables：plaquette、link-centered $A_\mu$、clover $F_{\mu\nu}$、$F W F$、Fourier modes、gluonic EMT candidates，以及 gauge-dependent $\mathbf E\times\mathbf A$ 和 Chern-Simons-current candidates。

除非调用者另行提供 operator mixing、scheme、scale、matching/normalization 和 continuum prescription，否则输出不能称为“已重整化”。本地证据只覆盖静态 API、stubbed PyQUDA shifts、checkerboard conversion、single-rank FFT phase 和 compact CPU reductions；没有真实 CUDA/QUDA、多 GPU 或 production physics validation。

bare lattice units、gauge dependence 与 renormalization promotion gate 见 `reference/PHYSICS_CONTRACT.md`。
当前 release label、可运行测试、E2E manifest 与证据升级流程见 `VALIDATION.md`。

## 2. 如何用自然语言调用

> 使用 `$pyquda_gluon_renorm`，从 Wilson gauge action 的 `beta=6.0` 构造 bare clover $F_{xt}$ 和 $\widetilde F_{xt}$。请明确 `0,1,2,3=x,y,z,t`、dual 的 $1/2$，并把 operator normalization 标记为 bare。

> 用 `$pyquda_gluon_renorm` 审查 $F(0)W(0,z)F(z)W^\dagger(z,0)$。检查 Wilson-line orientation、PyQUDA shift、`Gt=1` gate，以及只把 time series 传回 host。

> 使用 `$pyquda_gluon_renorm` 计算 Coulomb-gauge $E\times A$ 或 $K_\mu$ candidate；先要求 gauge-fixing functional、tolerance、residual gauge convention 和 Gribov-copy treatment，不要把 bare result 叫作 gluon helicity。

> 用 `$pyquda_gluon_renorm` 生成一个 momentum mask，但先估算 global dense `(Gt,Gz,Gy,Gx)` Boolean arrays 的 host memory。

raw RI quark vertices、propagator inversion 和 quark $\overline{\rm MS}$ conversion 应分别路由到其他 skills。

## 3. 方向、checkerboard 与颜色约定

- Lorentz/lattice directions 固定为

  ```text
  0=x, 1=y, 2=z, 3=t
  ```

- PyQUDA gauge field storage 是

  ```text
  (direction,e,t,z,y,xh,sink_color,source_color)
  ```

  ，其中 checkerboard parity extent 必须为 `e=2`。
- 当前代码绑定 $N_c=3$；traceless projection 使用 $\operatorname{Tr}/3$。
- `LatticeInfo.size/grid_size/grid_coord` 顺序为 `(x,y,z,t)`，而 array lattice axes 是 `(e,t,z,y,xh)`。

## 4. Bare coupling 与 gauge-action normalization

`g_0` 只接受明确命名的 gauge normalization。

Wilson convention：

$$
\beta=\frac{6}{g_0^2},
\qquad
g_0=\sqrt{\frac6\beta}.
$$

legacy tadpole convention：

$$
\beta=\frac{10}{g_0^2u_0^4},
\qquad
g_0=\sqrt{\frac{10}{\beta u_0^4}}.
$$

后者必须同时显式写 `u_0=...` 与
`normalization='legacy_tadpole_10'`。Wilson 分支必须省略 `u_0`；因此
历史式 `g_0(beta,u_0)` 会 fail closed，不会把第二个参数静默丢弃。
DWF、clover、HISQ 等 fermion-action labels 不能决定 gauge-action
normalization；在不知道生成 ensemble 的 gauge action 时不得猜 $g_0$。

## 5. $A_\mu$ 与 $F_{\mu\nu}$

### 5.1 Link-centered gauge potential

对 `half_flag=1`，代码先取 anti-Hermitian link part 并去迹：

$$
A_\mu\!\left(x+\frac{\hat\mu}{2}\right)
=\frac1{g_0}
\left[
\frac{U_\mu(x)-U_\mu^\dagger(x)}{2i}
\right]_{\rm traceless}.
$$

对 `half_flag=0`，它平均 $x$ 与 $x-\hat\mu$ 两条 links：

$$
A_\mu(x)=\frac1{g_0}
\left[
\frac{U_\mu(x)-U_\mu^\dagger(x)
+U_\mu(x-\hat\mu)-U_\mu^\dagger(x-\hat\mu)}{4i}
\right]_{\rm traceless}.
$$

格距 $a$ 的幂和 generator normalization 没有由 API 参数化；与 continuum $A_\mu^{\rm phys}$ 比较时必须从项目 convention 补齐。

### 5.2 Clover field strength

`F_munu` 以四个 oriented plaquette leaves 的 clover average 构造 anti-Hermitian difference。结构上可写为

$$
F_{\mu\nu}^{\rm clover}(x)
\propto
\frac{Q_{\mu\nu}(x)-Q_{\mu\nu}^\dagger(x)}{2ig_0},
$$

其中 $Q_{\mu\nu}$ 是四叶平均；exact overall signs 由 bundled loop paths 决定。`save_flag=0` 返回 full antisymmetric tensor

```text
(mu,nu,e,t,z,y,xh,3,3)
```

并满足代码构造的 $F_{\nu\mu}=-F_{\mu\nu}$。`save_flag=1` 返回两个 triplets；最安全的解释是按 pair mapping 读取：

$$
E_i\equiv F_{3i},
$$

$$
B_{\rm pair}=(F_{01},F_{02},F_{12}).
$$

在本文件的 $0,1,2,3=x,y,z,t$ convention 下，空间 pair 与 magnetic components 的关系是

$$
(F_{xy},F_{xz},F_{yz})=(B_z,-B_y,B_x).
$$

因此不要只凭 `B` 的数组下标猜物理 $(B_x,B_y,B_z)$ signs。

### 5.3 Dual tensor

代码采用

$$
\widetilde F_{\rho\sigma}
=\frac12\epsilon_{\rho\sigma\mu\nu}F_{\mu\nu}.
$$

$1/2$ 用于避免对 antisymmetric $(\mu,\nu)$ 与 $(\nu,\mu)$ 重复计数。Euclidean $\epsilon_{0123}$ sign、Minkowski continuation 和 topological-charge normalization 必须由调用者明确。

## 6. Wilson-line field-strength products

`FW_FW` 与 `FW_tildeFW` 的核心结构为

$$
\mathcal O(t;\ell,\hat d)
=-\frac12\sum_{\mathbf x}
\operatorname{Tr}
\left[
F_1(x)W_d(x,x+\ell\hat d)
F_2(x+\ell\hat d)W_d^\dagger(x,x+\ell\hat d)
\right].
$$

Wilson line 由 PyQUDA `LatticeLink.shift(step,direction)` 在 device field 上逐 link 累乘。反向 path 使用 shifted backward links 的 dagger。`link_length=0` 给出 local product。

若 `FW_FW` 与 `FF` 接收相同 fields、orientation、communicator 和 reduction convention，则 bundled definitions 在零长度满足

$$
\operatorname{FW\_FW}(\ell=0)=-\frac12\operatorname{FF}.
$$

这个关系是实现级 normalization oracle，不应脱离输入 orientation 推广成任意文献 convention。

时间分解目前被拒绝：

$$
G_t=1
$$

是 `FW_FW`、`FW_tildeFW`、`FF` 及若干 time-resolved observables 的必要条件，因为直接 world reduction 会混合不同 global time slices。大体积 matrix contraction 在 device 上完成，随后只把 local $L_t$ complex/real series 传到 host 做 MPI reduction；这不是全场 CPU fallback。

## 7. Fourier transform 约定

### 7.1 Single momentum

`FT_Phase` 接受 integer modes

```text
(px,py,pz,pt)
```

并使用 global centered coordinates 构造 checkerboard phase：

$$
\phi_p(x)=
\exp\left[-2\pi i\left(
\frac{p_xx}{G_xL_x}+\frac{p_yy}{G_yL_y}
+\frac{p_zz}{G_zL_z}+\frac{p_tt}{G_tL_t}
\right)\right].
$$

integer labels 先规范化到与 `np.fft.fftfreq` 一致的 fundamental bins。对 even extent，Nyquist representative 固定为 $-L_\mu/2$；因此传入 $+L_\mu/2$ 也会得到同一个 canonical site/half-link convention。

`FT_Gauge_1mom`/`FT_Prop_1mom` 做 global sum 后在 root 除以

$$
V_{\rm global}=G_xL_xG_yL_yG_zL_zG_tL_t.
$$

root 返回 normalized host result，non-root 的 MPI reduce result 为 `None`。

当 `FT_Gauge_1mom(...,half_flag=1)` 的输入是 link-centered 四方向 gauge field 时，component $\mu$ 还会独立乘

$$
e^{-ip_\mu/2}.
$$

因此 single-momentum 与 all-momentum APIs 都使用 direction-specific half-link correction；`half_flag=0` 只适用于明确 site-centered 的输入。

### 7.2 All-momentum FFT

尽管名字含 `_MPI`，`FFT_Gauge_Allmom_MPI` 当前严格要求 one GPU/rank：

$$
(G_x,G_y,G_z,G_t)=(1,1,1,1),
\qquad \texttt{comm.size}=1.
$$

它先在 GPU 上把 checkerboard 转为 lexicographic `(t,z,y,x)`，再执行 CuPy FFT。对 link-centered $A_\mu(x+\hat\mu/2)$，每个方向独立乘

$$
e^{-ip_\mu/2}.
$$

不能用一个依赖 $p_x+p_y+p_z+p_t$ 的 common phase 替代方向特异 correction。

## 8. EMT、$\mathbf E\times\mathbf A$ 与 $K_\mu$

### 8.1 Gluonic EMT candidates

continuum-like structure为

$$
T_{\mu\nu}^{g}
\sim2\operatorname{Tr}
\left[
F_{\mu\rho}F_{\nu\rho}
-\frac14\delta_{\mu\nu}F_{\alpha\beta}F_{\alpha\beta}
\right].
$$

`gaugeEMT_munu` 和 `gaugeEMT_mumu(...,def_type=1/2)` 实现 legacy clover contractions；`def_type=0` 使用 plaquette expression。它们是 bare lattice candidates。exact normalization、vacuum subtraction、trace anomaly、operator mixing 和 renormalization matrix 均未由本 skill 确立。

`gaugeEMT_mumu_tzyx` 保留四个 site fields。单 rank 返回 local CuPy tuple；多 rank 显式走 full-field host gather，其中 rank 0 返回四个 global NumPy arrays，所有 non-root ranks 返回 `(None,None,None,None)`。应在 production 前估算 root memory 并优先考虑 local contraction。

### 8.2 Gauge-dependent quantities

$A_\mu$、$\mathbf E\times\mathbf A$ 和 Chern-Simons current 都依赖 gauge。其结构可概括为

$$
K_\mu\sim\epsilon_{\mu\nu\rho\sigma}
\operatorname{Tr}
\left[A_\nu F_{\rho\sigma}
-\frac{2ig_0}{3}A_\nu A_\rho A_\sigma\right],
$$

但 bundled implementation 的 exact component factors 和 normalization 仍需要逐式 provenance。

调用这些 APIs 前至少记录：gauge-fixing functional、stopping tolerance、measured residual、residual global gauge convention、Gribov-copy selection 和 boundary condition。

特别地，`Topological_current_Kmu_t` 只返回 selected $K_t/K_z$ decomposition terms，不是完整四分量 $K_\mu(t)$。完整 component/site ownership 必须按 exact API name 核对。

## 9. Momentum mask

`generate_pselect_mask` 使用

$$
(ap)_\mu=\frac{2\pi n_\mu}{L_\mu^{\rm global}}
$$

以及 cylinder-like hypercubic cut

$$
\frac{p^{[4]}}{(p^2)^2}
=\frac{\sum_\mu p_\mu^4}{\left(\sum_\mu p_\mu^2\right)^2}
\le c.
$$

它在 host 上为每个 mode/cut 组合分配 global dense four-dimensional Boolean mask，setup memory 约为

$$
N_{\rm masks}\,V_{\rm global}\ \text{bytes}
$$

（NumPy Boolean 通常每元素一 byte，未含临时 momentum grids）。大格点应先估算临时 arrays 的数倍内存。

## 10. API 速查

| API | backend / shape / ownership | 关键 gate |
|---|---|---|
| `g_0` | host scalar | gauge-action normalization |
| `plaq_munu` | PyQUDA/CuPy lattice matrix field | directions distinct |
| `A_mu` | CuPy `(4,e,t,z,y,xh,3,3)` | link/site centering 与 gauge fixing |
| `F_munu` | CuPy full tensor 或 two triplets | direction/sign、$N_c=3$ |
| `F_and_tildeF` | CuPy $(F,\widetilde F)$ components | epsilon convention |
| `FW_tildeFW`, `FW_FW`, `FF` | root host $(L_t,)$；non-root `None` | `Gt=1`、orientation |
| `FT_Phase` | CuPy `(e,t,z,y,xh)` | global integer modes |
| `FT_Gauge_1mom` | root host `(4,3,3)`；non-root `None` | global volume、half-link centering |
| `FT_Prop_1mom` | root host `(4,4,3,3)`；non-root `None` | global volume |
| `FFT_Gauge_Allmom_MPI` | CuPy `(4,Lt,Lz,Ly,Lx,3,3)` | one rank/GPU only |
| `gaugeEMT_munu` | root host `(4,4)` | bare candidate |
| `gaugeEMT_mumu` | root host scalar/component result | `def_type` provenance |
| `gaugeEMT_mumu_tzyx` | one-rank 四个 CuPy；multi-rank root 四个 NumPy；non-root 四个 `None` | full-field root memory |
| `ExA_t` | root host time series | gauge fixing mandatory |
| `Topological_current_Kmu` | root host component result | exact normalization unmapped |
| `Topological_current_Kmu_t` | root host selected $K_t/K_z$ terms | 非完整 $K_\mu(t)$ |
| `Topological_current_Kmu_tzyx` | 与 EMT site fields 相同的 one-rank/root/non-root 四元组 contract | gauge fixing、root ownership |
| `generate_pselect_mask` | global host Boolean masks | $O(V_{\rm global})$ setup memory |

最小 one-momentum 骨架为：

```python
from skills.pyquda_gluon_renorm.scripts.Def_gluon_renorm import (
    A_mu,
    FT_Gauge_1mom,
    g_0,
)

bare_g0 = g_0(beta, normalization="wilson")
gauge_potential = A_mu(gauge, bare_g0, half_flag=1)
root_mode = FT_Gauge_1mom(
    gauge_potential, (px, py, pz, pt), latt_info, half_flag=1,
)
```

`root_mode` 仅在 root 有值；示例未执行 gauge fixing、I/O 或 renormalization。

## 11. 常见错误检查表

- [ ] 是否用 fermion action 猜了 $g_0$ normalization？
- [ ] `0,1,2,3` 是否确认为 `x,y,z,t`？
- [ ] checkerboard 与 lexicographic axes 是否混用？
- [ ] dual tensor 是否漏掉 $1/2$？
- [ ] $B$ triplet 是否按 pair mapping 而非名称猜 sign？
- [ ] time-resolved world reduction 是否在 `Gt>1` 混合 global times？
- [ ] link-centered FFT 是否用了 direction-specific $e^{-ip_\mu/2}$？
- [ ] 是否把 single-rank FFT 名字中的 `_MPI` 当作 distributed FFT proof？
- [ ] gauge-dependent observable 是否缺少 gauge-fixing metadata？
- [ ] bare EMT/$K_\mu$ 是否被误称为 renormalized physical observable？

## 12. 已验证与未验证

**本地直接验证：**

- `g_0` 两种显式 normalization、ambiguous positional `u_0` 与
  invalid-label rejection；
- PyQUDA field-shift call signature；
- device-side checkerboard-to-lexicographic mapping；
- Wilson-line kernel 的小数组 stub；
- public `FW_FW(\ell=0)=-\tfrac12 FF` normalization oracle；
- single-rank FFT 的 direction-specific half-link phase；
- single-momentum 与 all-momentum transforms 的逐 mode 一致性，包括 even-extent Nyquist canonicalization；
- anisotropic lattice 上 direction-specific momentum mask，以及 Hermitian/commutator contraction CPU oracle；
- `beta/u_0/g_0/cut` 的 Boolean/complex/nonfinite rejection，以及 Lorentz/link/`half_flag`/`save_flag`/`forward_flag`/`def_type` selectors 的 strict integer gates；
- `FW_tildeFW`、`FW_FW`、`FF` 对 explicit 与 gauge-owned global/local extents、process grid 和 rank coordinate 的 pre-field-strength equality gate；
- `gaugeEMT_munu` 在 device allocation 与 `F_munu` 前验证 coupling；
- 无 legacy full-volume `.get()`/Python site loops；
- static compile、YAML 与 public API。

**仍未验证：**

- real CUDA/CuPy/PyQUDA link shifts 和 halo exchange；
- multi-rank reductions、root ownership 和 performance；
- clover sign/normalization 对目标 ensemble 的 independent physics oracle；
- EMT/current equation-level provenance、operator mixing 和 renormalization；
- gauge-fixing quality、Gribov effects 和 physical matrix elements。

## 13. 参考文献

1. S. O. Bilson-Thompson, D. B. Leinweber, and A. G. Williams, “Highly-improved lattice field-strength tensor,” [arXiv:hep-lat/0203008](https://arxiv.org/abs/hep-lat/0203008), *Annals Phys.* **304**, 1 (2003), DOI: 10.1016/S0003-4916(03)00009-5. 支持 clover/improved field-strength method class；本代码并不因此自动等同于论文的全部 improved-loop combination。
2. Y. Hatta, X. Ji, and Y. Zhao, “Gluon Helicity $\Delta G$ from a Universality Class of Operators on a Lattice,” [arXiv:1310.4263](https://arxiv.org/abs/1310.4263), *Phys. Rev. D* **89**, 085030 (2014), DOI: 10.1103/PhysRevD.89.085030. 用于 gauge-dependent gluon-spin/Coulomb-gauge context，不是 bundled $K_\mu$ normalization 的 blanket validation。
3. 本项目 `scripts/Def_gluon_renorm.py` 是 exact paths、array layout、host-transfer boundary 和 API return ownership 的直接依据。

## 逐函数物理动机与证据卡

这些 functions 返回 bare lattice objects。表中“production-blocked”表示 literal code formula可以审计，但尚无逐式 paper/version/equation、Euclidean sign、normalization 与 renormalization closure，不能把结果称为 renormalized observable。

| Public symbol | 物理动机与 literal 公式 | 数据布局、gauge 与 ownership | 证据状态与 oracle |
|---|---|---|---|
| `g_0` | 把 simulation gauge-action parameter 映射为 bare coupling；Wilson convention 为 $g_0=\sqrt{6/\beta}$，legacy tadpole branch 为 $\sqrt{10/(\beta u_0^4)}$。 | `beta` 及 legacy branch 的 `u_0` 为 finite positive non-Boolean real host scalars；Wilson 必须省略 `u_0`，legacy 必须显式给出；返回 0-D CuPy scalar。 | 定义、return type、normalization oracle、ambiguous positional rejection与 NaN/Inf/Boolean rejection直接；不推断 ensemble 的 action convention。 |
| `plaq_munu` | 构造 $P_{\mu\nu}(x)=U_\mu U_\nu(x+\hat\mu)U_\mu^\dagger(x+\hat\nu)U_\nu^\dagger$，作为 clover/EMT building block。 | 返回 `(4,4,e,t,z,y,xh,Nc,Nc)`；方向序 `x,y,z,t`；依赖 PyQUDA halo-aware `shift(step,direction)`。 | 标准 plaquette 定义直接；fake-link shift oracle只证明调用签名，不证明真实 halo。 |
| `A_mu` | 从 anti-Hermitian link part 提取 traceless bare potential：half-link 为 $A_\mu=(U_\mu-U_\mu^\dagger)/(2ig_0)$；site-centered branch再平均 backward link。 | `(4,e,t,z,y,xh,Nc,Nc)`；$g_0$ finite positive，`half_flag`为 non-Boolean 0/1；gauge dependent且要求 gauge-fixed input。 | literal implementation与 selector/domain regression已审计；非线性 corrections、$a$ factors和 gauge-fixing quality未验证。 |
| `F_munu` | 由四叶 clover anti-Hermitian loop构造 $F_{\mu\nu}$；triplets满足 $(F_{01},F_{02},F_{12})=(B_z,-B_y,B_x)$。 | full return `(4,4,e,t,z,y,xh,Nc,Nc)` 或 `(E_i,B_{pair})`；`save_flag` strict 0/1，方向 `0,1,2,3=x,y,z,t`。 | Bilson-Thompson et al. 支持 method class；pair-order、antisymmetry及 invalid flag有 CPU oracle，目标 normalization未 physics-validated。 |
| `F_and_tildeF` | 计算 $\widetilde F_{\alpha\beta}=\tfrac12\epsilon_{\alpha\beta\mu\nu}F_{\mu\nu}$，返回指定 $F$ 与 dual component。 | Lorentz indices均为 non-Boolean `0..3`；返回两个 checkerboard color fields。 | dualization algebra直接；epsilon orientation与 Euclidean topological normalization仍须 project convention核对。 |
| `FW_tildeFW` | 构造 nonlocal density $-\tfrac12\operatorname{Tr}[F(x)W\widetilde F(x+n)W^\dagger]$，用于 gluon-helicity/Wilson-line operator研究。 | link/Lorentz/coupling、explicit/gauge global-local-grid-coordinate signature与$G_t$在 `F_munu/gauge.loop`前验证；输出 root-owned `(t,)`，non-root `None`。 | pre-loop sentinel、metadata-mismatch、Wilson transport与 zero-length class支持实现；论文不能 blanket 验证 component normalization。 |
| `FW_FW` | 构造 $-\tfrac12\operatorname{Tr}[F(x)WF(x+n)W^\dagger]$，为 gluonic bilocal correlator提供 bare kernel。 | Lorentz/link indices有界，length nonnegative，orientation strict 0/1，lattice signatures必须一致，且全部在 field construction前拒绝；device先约化。 | pre-loop invalid-control/mismatch与 $n=0$ 对 `FF` 的 $-1/2$ oracle直接；真实 link shift/MPI未验证。 |
| `FF` | 计算 local $\operatorname{Tr}[F_{\mu\nu}F_{\rho\sigma}]$ 的 time series，作为 bilocal $n=0$ 对照。 | 要求 matching explicit/gauge lattice signatures与$G_t=1$；coupling/metadata先验证，再 device sum checkerboard/spatial axes与 host MPI reduce，root `(t,)`。 | literal contraction和 pre-field mismatch gate直接；未含 renormalization、vacuum subtraction或 $a$ dimensions。 |
| `FT_Phase` | 构造 $e^{-ip\cdot x}$ checkerboard phase，integer labels canonicalize到 NumPy FFT fundamental bins。 | 返回 `(2,Lt,Lz,Ly,Lx/2)` CuPy array；使用 global `grid_coord`，local $L_x$ 必须偶数。 | single/all-FFT、anisotropic extent与 even-Nyquist oracles支持；真实 multi-rank phase未验证。 |
| `FT_Gauge_1mom` | 对一个 mode计算 $A_\mu(p)=V^{-1}\sum_xe^{-ipx}A_\mu(x)$；link-centered field额外乘 direction-specific $e^{-ip_\mu/2}$。 | exact `(4,2,Lt,Lz,Ly,Lx/2,Nc,Nc)`；MPI reduce后仅 root返回 host `(4,Nc,Nc)`。 | 与 all-momentum FFT及 Nyquist label有 numerical oracle；输入必须已是目标 gauge potential convention。 |
| `FT_Prop_1mom` | 对 spin-color propagator作单 momentum Fourier projection，为 NPR/gluon-propagator诊断提供 compact block。 | exact `(2,Lt,Lz,Ly,Lx/2,4,4,3,3)`；root返回 `(4,4,3,3)`，non-root `None`。 | layout gate与 Fourier algebra为实现证据；未验证 interacting propagator、gauge fixing或 MPI。 |
| `FFT_Gauge_Allmom_MPI` | 一次计算全部 modes并保留 link-origin phase，便于 propagator/vertex scans。 | `half_flag`为 non-Boolean 0/1；目前 fail closed 要求 one GPU、`grid_size=(1,1,1,1)`；输出 device `(4,Lt,Lz,Ly,Lx,Nc,Nc)`。 | invalid flag、single/all path与 per-direction half-link phase有 CPU-stub oracle；不支持 distributed FFT。 |
| `gaugeEMT_munu` | literal code实现 $T_{\mu\nu}=2\sum_{x,\rho}\operatorname{Tr}(F_{\mu\rho}F_{\nu\rho})-\delta_{\mu\nu}\tfrac12\sum_{x,\alpha,\beta}\operatorname{Tr}(F_{\alpha\beta}F_{\alpha\beta})$。 | finite positive non-Boolean $g_0$ 在 device allocation/field construction前验证；返回 MPI-root host `(4,4)`，non-root `None`；再检查 imaginary residual。 | validation-order与 literal algebra可审计；trace、Euclidean sign、$g_0$/$a$ normalization与 mixing无逐式 provenance，production-blocked。 |
| `gaugeEMT_mumu` | 提供三种 diagonal EMT discretization：`def_type=2` full tensor、`1` E/B decomposition、`0` plaquette expression。 | `def_type`为 non-Boolean `{0,1,2}`；返回 root host `(4,)`，三分支不是自动可互换 estimators。 | selector gate与分支 literal formulas直接；缺 continuum-normalized cross-check/equation mapping，production-blocked。 |
| `gaugeEMT_mumu_tzyx` | 保留每 site 的四个 diagonal densities，便于后续 Fourier projection或局域诊断。 | `def_type`为 non-Boolean `{1,2}`；one rank返回四个 device fields，多 rank root返回四个 NumPy globals，non-root四个 `None`。 | selector/ownership由 source/stub支持；物理 normalization继承 `gaugeEMT_mumu` blocker。 |
| `ExA_t` | 计算 code-defined $\sum_{\mathbf x}\epsilon_{ijk}\operatorname{Tr}[E_jA_k]$ time series，作为 canonical gluon-spin候选 density。 | 需要 gauge-fixed input和 $G_t=1$；root host `(3,t)`，non-root `None`；显著 imaginary part fail closed。 | Hatta-Ji-Zhao支持 operator context，不直接证明该 lattice normalization；production-blocked。 |
| `Topological_current_Kmu` | 按 code component表组合 schematic $K_\mu\sim\epsilon_{\mu\nu\rho\sigma}\operatorname{Tr}(A_\nu F_{\rho\sigma}-\tfrac{2ig_0}{3}A_\nu A_\rho A_\sigma)$。 | gauge dependent；返回 root四 scalar tuple `(Kx,Ky,Kz,Kt)`，non-root四个 `None`。 | AF/$A^3$ literal terms可追踪；无 exact paper equation、Euclidean conversion、$a$ factors与 residual-gauge contract，production-blocked。 |
| `Topological_current_Kmu_t` | 暴露 selected decomposition `(Kt_AF,Kt_3A,Kz_AtB,Kz_ExA,Kz_3A)`，用于审计 AF 与 cubic contributions。 | 不是完整四分量 $K_\mu(t)$；要求 $G_t=1$，root返回五个 `(t,)` host arrays。 | tuple order由 source直接支持；函数名的历史含义已限定，physics provenance仍 blocked。 |
| `Topological_current_Kmu_tzyx` | 返回完整 code-defined $K_x,K_y,K_z,K_t$ site densities，供 caller自行 Fourier/average。 | one rank四个 device fields；multi-rank root四个 global NumPy fields，non-root四个 `None`。 | spatial component algebra未有独立 noncommuting-color production oracle；normalization/gauge/renormalization均未关闭。 |
| `generate_pselect_mask` | 按 $p^{[4]}/(p^2)^2$ cylinder-like cut和 4-bit active-direction pattern选择 FFT modes。 | bounds/modes为 non-Boolean integers，cut为 finite real且 $0<cut\le1$；invalid input在 dense mask allocation前拒绝。返回 `(GtLt,GzLz,GyLy,GxLx)` masks。 | implementation utility；NaN/Inf/Boolean cut、anisotropic、Nyquist与 wrapped-label regressions已覆盖，不是唯一物理 cut。 |
