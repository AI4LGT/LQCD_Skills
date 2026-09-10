# `pyquda_hisq_recover` 用户说明书

## 1. 适用范围与术语

本 skill 把 staggered/HISQ propagator 的 one-component spin structure 提升为 naive four-spin form，并存入 Wilson-like spin-color axes。它做的是 Kawamoto-Smit spin reconstruction：

$$
G_{\rm naive}(x,x_0)
=\Omega(x)G_\chi(x,x_0)\Omega^\dagger(x_0).
$$

“Wilson-like”仅描述输出 storage；它不把 HISQ action 变成 Wilson/clover action，也不执行 Dirac inversion、taste projection、correlator contraction 或 continuum extrapolation。

输入 $G_\chi$ 的 producer contract 是：它必须是同一个 source $x_0$、同一 gauge/background、同一 boundary condition 下得到的 one-component staggered/HISQ color propagator，并且 producer 对 staggered phases、lattice origin 与 source normalization 的处理必须已知。本 helper 只乘 $\Omega(x)$ 与 $\Omega^\dagger(x_0)$；它不会补做或撤销额外的 $\eta_\mu$、taste phase、anti-periodic wrap sign 或 source phase。

函数名中的 `evengrid` 是历史名称。输入是完整 lexicographic `(t,z,y,x)` grid，不是仅 even sites、不是 checkerboard half-volume，也不执行 even/odd projection。

本地验证覆盖 NumPy shape、full source coordinate、dtype、vectorized parity 和 stubbed gather contract；真实 CuPy/PyQUDA/MPI transport 尚未运行。

production adapter 必填 metadata 与 fail-closed 条件见 `reference/PHYSICS_CONTRACT.md`。
当前 release label、可运行测试、producer-specific E2E manifest 与证据升级流程见 `VALIDATION.md`。

## 2. 如何用自然语言调用

> 使用 `$pyquda_hisq_recover`，把 shape 为 `(Nt,Nz,Ny,Nx,3,3)` 的 HISQ propagator 从全局 source `(t0,z0,y0,x0)` 恢复成 naive four-spin field。请明确 $\Omega(x)$ 的 gamma 次序和输出 axes。

> 用 `$pyquda_hisq_recover` 审查 MPI-local reconstruction。我的 PyQUDA `grid_coord` 是 `(gx,gy,gz,gt)`；请检查 global parity offset，并保持 `gather=False` 在 GPU 上做后续 contraction。

> 使用 `$pyquda_hisq_recover` 比较自定义 Euclidean gamma basis 与 producer 的 convention；不要把输出称作 Wilson propagator。

不适合的请求包括 HISQ inversion、taste-splitting analysis、Wilson-action conversion 或 hadron fit。

## 3. Spin diagonalization 公式

### 3.1 代码采用的 $\Omega(x)$

坐标顺序为

$$
x=(t,z,y,x).
$$

由于 gamma powers 只依赖坐标奇偶性，代码构造

$$
\Omega(x)
=\gamma_x^{x\bmod2}
 \gamma_y^{y\bmod2}
 \gamma_z^{z\bmod2}
 \gamma_t^{t\bmod2}.
$$

矩阵乘法不交换；必须让 producer 和 consumer 使用同一 gamma order。`gamma_ops` 的 keys 恰为

```text
t, z, y, x, unit
```

且每个 matrix 是 `(4,4)`。

当前只读核对的 PyQUDA checkout 中，默认 `gamma.gamma(n)` 返回 `DeGrandRossiMatrix.matrix(n)`；本模块据此使用 `n=(8,4,2,1,0)` 对应 `(t,z,y,x,unit)`。这是当前实现 contract，不应推广为所有 HISQ file producers 的默认 basis。若 producer 使用另一 gamma representation，应显式传 custom `gamma_ops`。

同一 spin diagonalization convention 通常要求检查

$$
\Omega^\dagger(x)\gamma_\mu\Omega(x+\hat\mu)
=\eta_\mu(x)I,
$$

其中在 $x,y,z,t$ ordering 下常用

$$
\eta_\mu(x)=(-1)^{\sum_{\nu<\mu}x_\nu}.
$$

该 identity 的 gamma order、坐标原点和跨 boundary wrap 必须与 $G_\chi$ producer 一起验证；origin shift 或不同 staggered-phase convention 会改变 signs。

### 3.2 Propagator reconstruction

staggered propagator $G_\chi$ 只带 color source/sink indices。spin factor 为

$$
\mathcal S_{\alpha\beta}(x,x_0)
=\left[\Omega(x)\Omega^\dagger(x_0)\right]_{\alpha\beta}.
$$

因此代码输出

$$
G_{{\rm naive},\alpha\beta}^{ab}(x,x_0)
=\mathcal S_{\alpha\beta}(x,x_0)
G_\chi^{ab}(x,x_0).
$$

这是 spin lift；taste 与 operator construction 的后续约定仍由调用者负责。

## 4. 数据布局与坐标映射

### 4.1 Global-array API

`hisq_to_wilson_evengrid` 输入：

```text
(GLt, GLz, GLy, GLx, sink_color, source_color)
= (GLt,GLz,GLy,GLx,3,3)
```

输出：

```text
(GLt,GLz,GLy,GLx,sink_spin,source_spin,sink_color,source_color)
= (...,4,4,3,3)
```

输入必须是 `complex64` 或 `complex128`；输出保持同一 dtype 与 NumPy/CuPy backend。

### 4.2 Source coordinate

推荐显式传 full global coordinate：

```python
x0_tzyx = (t0, z0, y0, x0)
naive = hisq_to_wilson_evengrid(info, Gchi, x0_tzyx)
```

legacy scalar `tsource=t0` 仍被接受，但严格表示

$$
x_0=(t_0,0,0,0).
$$

对 displaced spatial source 若只传 scalar，会得到错误的 $\Omega^\dagger(x_0)$。

### 4.3 PyQUDA metadata reorder

本 API 的 field/source coordinates 使用 `(t,z,y,x)`；PyQUDA 的 `size`、`grid_size`、`grid_coord` 使用 `(x,y,z,t)`。MPI helper 显式重排：

$$
(G_t,G_z,G_y,G_x)
=(L_tg_t,L_zg_z,L_yg_y,L_xg_x)
$$

，并在每个 local block 上生成 global coordinate ranges。代码 review 时必须逐处标记 reorder，不能凭变量名字猜轴序。

## 5. MPI-local reconstruction

`hisq_to_wilson_evengrid_MPI` 的 local input 是

```text
(Lt,Lz,Ly,Lx,3,3)
```

rank coordinate `(gx,gy,gz,gt)` 对应 global offsets：

$$
(t_{\rm off},z_{\rm off},y_{\rm off},x_{\rm off})
=(g_tL_t,g_zL_z,g_yL_y,g_xL_x).
$$

`gather` 必须是 Python `bool` 或 `np.bool_`。字符串 `"false"`、整数 `0/1` 和任意 truthy object 会在读取 lattice metadata 或执行 local reconstruction 前拒绝，不能依赖 Python truthiness 控制 full-field gather。

`gather=False` 返回 local reconstructed field，并保持 backend；这是 production contraction 的首选，因为可以先在 GPU/local ranks 上收缩，再只 reduce 小 observables。

`gather=True` 是显式全场 gather：

- CuPy local field 先完整传到 host；
- `core.gatherLattice(...,[0,1,2,3])` 组装 global field；
- root ownership 由 active communicator 的 `core.getMPIRank()` 判定；pinned
  `LatticeInfo` 虽也保存 `mpi_rank` snapshot，本 helper 不把可能陈旧的
  geometry-object metadata 当作 return ownership 的唯一来源；
- 只有 rank 0 返回 global NumPy array，其他 ranks 返回 `None`。

该路径的内存和通信成本是 $O(V_{\rm global}\times4^2\times3^2)$ complex numbers，可能远大于直接收缩后的 compact reduction。

## 6. Gamma-basis contract

custom `gamma_ops` 必须满足：

1. keys 恰为 `{'t','z','y','x','unit'}`；
2. 每个 matrix 是 numerical、finite、shape `(4,4)`；`NaN`、`Inf`、
   object/string matrix 会在 reconstruction 前拒绝；
3. `unit` 是与该 basis 一致的 identity；
4. Euclidean Clifford algebra、$\gamma_5$ 和 producer convention 已核对；
5. matrices 会 cast 到 `hisq_prop.dtype`，因此 complex64 输入不会静默提升为 complex128。

建议在单点 color identity input 上建立解析 oracle。例如 $x=(0,0,0,0)$、$x_0=(0,0,0,1)$ 时，spin factor 应为 $\gamma_x^\dagger$。

## 7. API 速查

| API | 作用 | 返回 ownership |
|---|---|---|
| `gamma_ops` | 默认 PyQUDA Euclidean gamma mapping | 配置字典，不是 field |
| `hisq_to_wilson_evengrid` | global resident reconstruction | same backend global field |
| `hisq_to_wilson_evengrid_MPI(...,gather=False)` | local reconstruction | 每 rank local field |
| `hisq_to_wilson_evengrid_MPI(...,gather=True)` | host gather | root global NumPy；非 root `None` |

## 8. 推荐验证流程

1. 从 propagator producer 记录 lattice order、source coordinate、boundary condition、gamma basis 和 color-axis ownership。
2. 用 $1^4$ 或 $2^4$ 小格点、color identity field，逐点对照 $\Omega(x)\Omega^\dagger(x_0)$。
3. 对 displaced source 测试所有四个坐标的 parity；不要只测 $t_0$。
4. 单 rank 比较 global API 与 MPI API `gather=False`。
5. 多 rank 重组 local blocks，检查跨 rank global parity；再决定是否允许 full gather。
6. 在 physical contraction 前固定 staggered phase、taste/operator 和 source/sink convention。

## 9. 常见错误检查表

- [ ] 是否把 `(t,z,y,x)` 与 PyQUDA `(x,y,z,t)` 混用？
- [ ] displaced source 是否误传成 scalar `tsource`？
- [ ] $\Omega$ 的 gamma multiplication order 是否与 producer 相同？
- [ ] 是否把 reconstructed naive field 称为 Wilson-action propagator？
- [ ] 是否遗漏 taste/operator convention？
- [ ] producer 的 $\eta_\mu$、lattice origin 与 boundary wrap signs 是否与 $\Omega$ contract 一致？
- [ ] 是否把历史名 `evengrid` 错当成 even-site/checkerboard input？
- [ ] MPI local coordinates 是否加了 global rank offsets？
- [ ] 非 root 是否错误使用了 `gather=True` 的 `None` 返回值？
- [ ] 是否不必要地把完整 CuPy field gather 到 host？

## 10. 已验证与未验证

**本地直接验证：**

- global reconstruction shape 和 input validation；
- full source coordinate 对 $\Omega^\dagger(x_0)$ 的影响；
- custom gamma 下 pure-NumPy import/runtime，以及当前 PyQUDA DeGrand--Rossi default 的只读 API mapping；
- 非平凡 $\gamma_z\gamma_t$ sink order 与 source reversed-dagger oracle；
- complex64 dtype preservation；
- vectorized coordinate construction，无 Python lattice-site loop；
- 静态 API 与 stub import。

**仍未验证：**

- 真实 CuPy execution 与 PyQUDA `core.gatherLattice`；
- 多 rank parity、MPI performance 和 root memory；
- 与目标 HISQ solver/output file 的 gamma/taste convention；
- producer 的 $\eta_\mu$、origin、boundary-condition 与 source-phase compatibility；
- interacting correlator 或 physical observable。

## 11. 参考文献

1. N. Kawamoto and J. Smit, “Effective Lagrangian and Dynamical Symmetry Breaking in Strongly Coupled Lattice QCD,” *Nucl. Phys. B* **192**, 100–124 (1981), DOI: 10.1016/0550-3213(81)90196-6. 用于 staggered spin diagonalization/Kawamoto-Smit transformation 的方法背景。
2. E. Follana et al., “Highly Improved Staggered Quarks on the Lattice, with Applications to Charm Physics,” [arXiv:hep-lat/0610092](https://arxiv.org/abs/hep-lat/0610092), *Phys. Rev. D* **75**, 054502 (2007), DOI: 10.1103/PhysRevD.75.054502. 用于 HISQ action 背景；不是本地 MPI runtime 的证据。
3. 本项目 `scripts/Def_hisq_recover.py` 是具体 gamma order、axis layout 和 gather ownership 的直接依据。

## 逐函数物理动机与证据卡

恢复公式本身不能辨认 producer 的 taste basis、$\eta_\mu$ phases、coordinate origin 或 temporal boundary condition；这些是进入函数前必须记录的外部 contract。

| Public symbol | 物理动机与公式 | 输入输出、坐标与 ownership | 证据状态与 oracle |
|---|---|---|---|
| `gamma_ops` | 提供当前 PyQUDA DeGrand-Rossi basis中的 $\gamma_x,\gamma_y,\gamma_z,\gamma_t,I$，用于构造 Kawamoto-Smit $\Omega(x)$。 | 小型 `(4,4)` matrices；仅在 `pyquda_utils.gamma` 可导入时存在，mixed NumPy/CuPy时只允许显式小矩阵转换。 | 只读上游 gamma mapping直接支持；producer是否使用同一 basis仍需外部 metadata。 |
| `hisq_to_wilson_evengrid` | 实现 $G_{naive}(x,x_0)=\Omega(x)G_\chi(x,x_0)\Omega^\dagger(x_0)$，其中 $\Omega=\gamma_x^x\gamma_y^y\gamma_z^z\gamma_t^t$，固定不可交换次序。 | local input `(t,z,y,x,3,3)`，source为 global `(t,z,y,x)`；输出 `(t,z,y,x,4,4,3,3)`，保留 complex64/128。 | Kawamoto-Smit方法直接；two-odd-coordinate、displaced-source、dtype与 global-coordinate oracles已覆盖。 |
| `hisq_to_wilson_evengrid_MPI` | 在各 rank先按 global offset恢复 local block，再通过 PyQUDA gather形成 global Wilson-like storage，便于 root-side I/O/analysis。 | `gather`仅接受 Python/NumPy Boolean并在 metadata/reconstruction前验证；False返回 local，True时 root返回 gathered NumPy、non-root返回 `None`。 | strict-gate、`np.bool_`与 stub ownership oracles直接；真实 MPI/parity/large-volume gather未验证，producer checklist仍 mandatory。 |
