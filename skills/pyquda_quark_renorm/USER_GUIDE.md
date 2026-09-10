# `pyquda_quark_renorm` 用户说明书

## 1. 适用范围与模块分层

本 skill 是一个 mixed legacy postprocessing module，包含三类功能：

1. NumPy/CuPy spin-color inversion、$\gamma_5$ adjoint 和 connected/disconnected amputation；
2. CPU perturbative coupling、RI/RI-prime/RI-SMOM 到 $\overline{\rm MS}$ conversion、running 与 Padé tables；
3. pandas/gvar/lsqfit host I/O 和 fits。

raw gauge-fixed RI projectors、explicit `gamma_mu/qslash` scheme 和 $Z_q$ 应优先使用 `$pyquda_ri_renorm`。本 skill 不做 gauge loading、gauge fixing、momentum-source solve 或 MPI reduction。

最重要的证据边界是：spin-color array identities 有 CPU oracle；legacy perturbative tables 尚未全部映射到 original equations。tensor RI-prime/MOM 的高阶 normalization 已由 Gracey Eq. (4.11) 闭合并修正为 `/64`，但这不能外推到相邻 SMOM、general-$\xi$、Padé 或 field/tensor running tables。因此不能把“函数可运行”或一个已映射函数当作“所有 coefficients 已验证”。

已定位与未定位的 coefficient groups、RG/Padé 与 promotion gate 见 `reference/PHYSICS_CONTRACT.md`。
当前 release label、explicit-only 条件、reference-reproduction manifest 与证据升级流程见 `VALIDATION.md`。

## 2. 如何用自然语言调用

> 使用 `$pyquda_quark_renorm` 审查这个 propagator 的 $12\times12$ inversion 和 $\gamma_5S^\dagger\gamma_5$。输入末四轴为 `(sink_spin,source_spin,sink_color,source_color)`，请用非 color-diagonal oracle 验证 dagger。

> 用 `$pyquda_quark_renorm` 对已给出的 RI/SMOM scalar $Z_S$ 做 $\overline{\rm MS}$ conversion audit。先列出 exact scheme、paper、equation、$a_s$ normalization 和 loop order；缺任何一项就不要给 production number。

> 使用 `$pyquda_quark_renorm` 审查 `Lambda_O_dis`。明确 raw loop 是否已做 vacuum subtraction，并保持 configuration correlations through jackknife。

> 用 `$pyquda_quark_renorm` 复核 chiral/$a^2p^2$ fit，但要输出 fit window、priors、covariance assumption、$ZA$、momentum definition 和 systematic flag。

## 3. Coupling 与 expansion variable

`strong_coupling_constant(scale,Lambda,nf)` 使用

$$
a(\mu)=\frac{\alpha_s(\mu)}{\pi}
$$

的 asymptotic five-loop expansion，并返回五个 successive truncations：

```text
(a_1loop,a_2loop,a_3loop,a_4loop,a_5loop)
```

函数要求

$$
\frac{\mu}{\Lambda}>3.
$$

`alpha_s(scale,nf)` 则返回 fifth-order truncation 乘 $\pi$：

$$
\alpha_s(\mu)=\pi a_{5\,loop}(\mu),
$$

并固定 legacy $\Lambda=0.332$。不同 conversion functions 中 coefficients 可能按 $a$、$a/4$ 或显式 decimal normalization 写入；必须以 exact source/equation 为准，不能仅按邻近函数模式统一重写。

## 4. Conversion factors

函数名采用方向性命名，例如

```text
scalar_conversion_ms_bar_over_rismom
```

应解释为一个 legacy candidate

$$
C_O^{\overline{\rm MS}\leftarrow RI}(\mu)
=\frac{Z_O^{\overline{\rm MS}}(\mu)}{Z_O^{RI}(\mu)},
$$

从而在 convention 已核实后使用

$$
Z_O^{\overline{\rm MS}}=C_O^{\overline{\rm MS}\leftarrow RI}Z_O^{RI}.
$$

公开 series 分为：

- quark field：RI/MOM、RI-prime/MOM 及两者 ratio；
- quark mass/scalar：RI-prime/MOM 与两种 legacy RI/SMOM families；
- tensor：对应的 RI-prime/MOM 与两种 RI/SMOM families；
- 带 gauge parameter `xi` 的 generalized legacy `*2` variants；
- scalar/tensor Padé approximants。

代码中 scalar conversions 通过 mass conversion 的 reciprocal 构造：

$$
C_S=\frac1{C_m}.
$$

这符合 $Z_mZ_S=1$ 的常见 continuum relation，但仍需确认 scheme、normalization 与 truncation prescription。

### 4.1 Equation-level provenance gate

通用参考文献只说明 method class，不能为整个 coefficient table 背书。每个 production channel 都应建立：

```text
function -> operator -> input scheme -> output scheme -> gauge -> nf
         -> expansion variable -> loop order -> paper equation -> regression test
```

特别是 `tensor_conversion_ms_bar_over_rimom_prime` 的 high-order denominator/normalization
不能凭 neighboring functions 推断。该项现已由 Gracey, arXiv:hep-ph/0304113v1,
Eq. (4.11), PDF p. 23 逐式确认：paper variable 为
$a=\alpha_s/(4\pi)$，故 fixed-order inverse 的三阶项为 $a_s^3/64$；其余
SMOM、general-$\xi$、Padé 与 quark-field/tensor running tables 仍是
high-risk unresolved items。

## 5. Running factor

`anomalous_dimension(gamma,nf)` 把三到五阶 anomalous-dimension coefficients 转成代码的 integrated-series coefficients `dim`。`scale_running(scale,scale0,dim,Lambda,nf)` 构造

$$
c(a)=a^{d_0}
\left(1+d_1a+d_2a^2+d_3a^3+d_4a^4+\cdots\right)
$$

并返回逐 truncation ratio

$$
U(\mu\to\mu_0)
=\frac{c(a(\mu_0))}{c(a(\mu))}.
$$

public wrappers 提供 quark mass、quark field、scalar 和 tensor 的 legacy anomalous-dimension tables。scalar running 是 mass running 的 reciprocal。所有 tables 都受同一 equation-level provenance gate；数值 finite 不证明 operator convention 正确。

`gamma` 与 `dim` 的 public input contract 是 finite real one-dimensional arrays，长度为 3–5。实现会在 NumPy dtype inference 前递归检查 list/tuple、`UserList`、custom non-string `Sequence` 与 object array 中仍可识别的 Boolean，再拒绝 complex dtype，最后才转换为 `float`。因此 `[True,2.0,3.0]` 不会静默变成 `[1.0,2.0,3.0]`；若 caller 已提前生成纯 float ndarray，原始 provenance 已不可恢复。

## 6. Systematics 与 Padé

`matching_systematic_error` 返回一个 legacy seven-channel factor array，并通过 `MOM_flag` 与 `error_flag` 切换 nominal、truncation、running-order 和 $\Lambda_{\rm QCD}$ variations。由于历史 branches 没有一致转发 parameters，当前 API 只允许

$$
\mu_0=2.0\ \mathrm{GeV},
\qquad
\Lambda=0.332\ \mathrm{GeV},
\qquad
n_f=3.
$$

其他值会显式拒绝，而不是静默忽略。seven channels 的物理 labels 必须从项目原 schema 核对；不要仅按 array position猜 observable。

Padé helpers 返回 legacy rational approximants。Padé estimate 应作为 model/truncation sensitivity，而不是伪装成 independently calculated higher-loop coefficient。`pade_matching_factor` 的公开 legacy return 只有三个 scalar channels；`scale0` 是 deprecated、仍作 finite-positive validation 的 compatibility no-op，因为返回值不含 running。实现不再计算并丢弃 tensor/running branches，避免无关系数或 singularity 阻断 scalar-only return。

## 7. Spin-color algebra

### 7.1 Layout 与 inversion

输入末四轴固定为

```text
(sink_spin,source_spin,sink_color,source_color)=(4,4,3,3)
```

compound matrix indices 为

$$
r=(s_{\rm sink},c_{\rm sink}),
\qquad
c=(s_{\rm source},c_{\rm source}).
$$

`inverse_propagator` 在输入 NumPy/CuPy backend 上执行 batched

$$
S^{-1}=\operatorname{inv}_{12\times12}S.
$$

legacy `on_device` 只作为一致性检查；若提供，必须是 Python/NumPy Boolean，整数、字符串与任意 truthy object 会在 inversion 前拒绝。若 Boolean 值与 inferred backend 不同会报错，不会触发 hidden host transfer。

### 7.2 $\gamma_5$ adjoint

`adj` 实现

$$
\overline S
=(\gamma_5\otimes I_c)S^\dagger(\gamma_5\otimes I_c).
$$

Hermitian dagger 同时交换 source/sink spin 和 color axes。只在 color-diagonal input 上测试会漏掉 color transpose bug；本地 oracle 使用 general non-color-diagonal arrays。

### 7.3 Connected amputation

`Lambda_O_con` 可先沿 configuration axis 做 leave-one-out jackknife：

$$
X^{(i)}=\frac1{N-1}\sum_{j\ne i}X_j.
$$

随后构造 legacy amputated spin vertex：

$$
\Lambda_O^{(i)}
=\sum_{c_1c_2c_3}
S^{-1,(i)}G_O^{(i)}\overline{S^{-1,(i)}}.
$$

输出保留 configuration/resample axis 和两个 spin axes，color 已收缩。这个 helper 使用一条 $S_q$ 及其 $\gamma_5$ adjoint；一般 nonexceptional two-leg kinematics 应使用 `$pyquda_ri_renorm` 的 independent-leg API。

### 7.4 Disconnected amputation

`Lambda_O_dis` 先形成

$$
G_O^{\rm disc}=S_q\,L_O,
$$

其中 `current` 是 raw scalar loop per configuration，然后按同一方式 jackknife/amputate。函数不做

$$
L_O\longrightarrow L_O-\langle L_O\rangle
$$

的 vacuum subtraction。调用者必须说明 subtraction、contact terms、flavor normalization 和 configuration pairing。

## 8. Host I/O 与 fits

heavy dependencies 只在使用时 lazy import。不要把 full CuPy lattice fields 传给这些 routines；应先在 GPU 上 projection/reduction，再只传 compact host tables。

### 8.1 Mass/chiral fit

`ma_fit` 对每个 momentum 独立拟合

$$
f(m)=\frac{A}{m^2}+B+Cm,
$$

并返回 intercept-like parameter $B$。每个 momentum 的 input means/errors 被重新构造成 diagonal gvars，未接收 cross-mass 或 cross-momentum covariance；因此这不是 correlated fit。

### 8.2 $a^2p^2$ fit

`a2p2_fit` 使用

$$
f(x)=C_0+C_1x+C_2x^2+C_3x^3,
\qquad x=a^2p^2,
$$

并输出 $C_0Z_A$。fit 前会按 `matching_systematic_error` rescale data。必须记录 `lattice_flag`、$Z_A$、prior、window、momentum definition、systematic flag 和 covariance assumption。

`am1` 是 inverse lattice spacing array，必须 finite、real、positive；Boolean/complex/NaN/Inf 在导入 gvar/lsqfit 前 fail closed。这里的检查只保护数值域，不替调用者确认单位和 ensemble 对应关系。

`read_data` 的 momentum window 精确定义为

$$
\texttt{fitmin}<a^2p^2\le\texttt{upper\_limit}.
$$

下界严格排除、上界包含；输入 momentum order 也必须与 legacy table schema 一致。

只有 `pade_flag=1` 时才构造 `pade_matching_factor`。`pade_flag=0` 的 bare-data path 对该函数的调用次数严格为零，避免 disabled option 仍触发 legacy coefficient/domain failure。

### 8.3 Ratio fit

`ratio_fit` 先把一个 ensemble 插值到另一个 momentum grid，再对 ratio 使用同一 polynomial form。bracketing interpolation 和最终 `gvar_output * ZS_small` 不会重新包装已有 gvar；若调用者确实传入共享底层 gvar sources，interpolation/product 会保留其 covariance。反之，从 plain means/errors 构造的数据仍只有 diagonal uncertainty，函数也不会凭空恢复跨 ensemble、跨 momentum 或共同 $Z_A$ 的 correlations。production workflow 优先传 resample-level 或 correlated-gvar inputs。

两套 inverse spacing `am1`、`am1_0` 分别执行同一 pre-conversion real-domain gate；不能用 complex scale array 借 NumPy conversion 绕过 ensemble-scale consistency。

## 9. 完整 public API 索引

- Companion resampling：`jackknife_resampling`。
- Coupling：`beta_coupling_constant`、`strong_coupling_constant`、`alpha_s`。
- Quark-field/vector conversions：`vector_conversion_ms_bar_over_rimom_prime`、`quark_field_conversion_rimom_prime_over_rimom`、`quark_field_conversion_ms_bar_over_rimom`、`quark_field_conversion_ms_bar_over_rimom_prime`、`quark_field_conversion_rimom_prime_over_rimom2`。
- Quark-mass conversions：`quark_mass_conversion_ms_bar_over_rismom`、`quark_mass_conversion_ms_bar_over_rismom_mu`、`quark_mass_conversion_ms_bar_over_rimom_prime`、`quark_mass_conversion_ms_bar_over_rimom_prime2`。
- Scalar conversions：`scalar_conversion_ms_bar_over_rismom`、`scalar_conversion_ms_bar_over_rismom_mu`、`scalar_conversion_ms_bar_over_rimom_prime`、`scalar_conversion_ms_bar_over_rimom_prime2`。
- Tensor conversions：`tensor_conversion_ms_bar_over_rismom`、`tensor_conversion_ms_bar_over_rismom_mu`、`tensor_conversion_ms_bar_over_rimom_prime`、`tensor_conversion_ms_bar_over_rimom_prime2`。
- Running：`anomalous_dimension`、`scale_running`、`quark_mass_anomalous_dimension_under_ms_bar`、`quark_field_anomalous_dimension_under_ms_bar`、`scalar_anomalous_dimension_under_ms_bar`、`tensor_anomalous_dimension_under_ms_bar`。
- Systematics/Padé：`matching_systematic_error`、`scalar_conversion_ms_bar_over_rimom_pade_3loop`、`scalar_conversion_ms_bar_over_rimom_pade_4loop`、`tensor_conversion_ms_bar_over_rimom_pade_3loop`、`tensor_conversion_ms_bar_over_rimom_pade_4loop`、`pade_matching_factor`。
- Host data/fits：`read_data`、`ma_fit`、`a2p2_fit`、`ratio_fit`。这些 routines 的 legacy mean/error construction 是 diagonal-error；只有已传入的 correlated gvars 才能在后续 algebra 中保留 covariance。
- Spin-color algebra：`inverse_propagator`、`adj`。两者支持任意数量的 leading batch/lattice axes，只固定末四轴。
- Amputation：`Lambda_O_con`、`Lambda_O_dis`。

最小 pure-array 审查示例为：

```python
from skills.pyquda_quark_renorm.scripts.Def_qcd_analysis import (
    jackknife_resampling,
)
from skills.pyquda_quark_renorm.scripts.Def_quark_renorm import (
    adj,
    inverse_propagator,
)

propagator_inverse = inverse_propagator(propagator)
propagator_bar = adj(propagator, gamma5)
jackknife_samples = jackknife_resampling(per_configuration_values)
```

conversion/running APIs 未列入示例，是因为每一个 production call 都必须先通过 equation-level provenance gate。

## 10. 推荐工作流

1. raw NPR 阶段先用 `$pyquda_ri_renorm` 固定 gauge、momenta、projector 和 $Z_q$ definition。
2. 保留 per-configuration/resample projected vertices，不要只保存 means/errors。
3. 为目标 conversion function 建 equation map 和 independent benchmark；未通过时停止 numerical production。
4. 同时记录 conversion scale、running target scale、$n_f$、$\Lambda$、loop truncation 和 Padé flag。
5. fits 使用完整 covariance 或 resample-level fit；若 legacy API 只能用 diagonal errors，明确标记 approximation。
6. 报告 statistical、matching truncation、running、$\Lambda$、window 和 lattice-artifact systematics，避免重复计数相关误差。

## 11. 常见错误检查表

- [ ] raw projector 是否误路由到 conversion skill？
- [ ] $a_s=\alpha_s/\pi$ 与 $\alpha_s/(4\pi)$ 是否混淆？
- [ ] function name 指定的 input/output scheme 是否已核对？
- [ ] 是否用通用论文为整张 legacy coefficient table 背书？
- [ ] tensor high-order normalization 是否在无 equation 情况下被“修正”？
- [ ] running ratio 的方向 $\mu\to\mu_0$ 是否反了？
- [ ] `gamma/dim/am1/am1_0` 是否在调用前已明确为 real physical inputs，而不是被静默取实部？
- [ ] spin/color dagger 是否交换了两组 source/sink axes？
- [ ] disconnected loop 是否已做 vacuum subtraction？
- [ ] jackknife/covariance 是否在 fit/interpolation 中丢失？
- [ ] Padé estimate 是否被称为 explicit loop result？

## 12. 已验证与未验证

**本地直接验证：**

- coupling domain 和 finite legacy series smoke tests；
- undefined-symbol removal 与 lazy optional imports；
- NumPy/CuPy backend inference contract；
- 在无 CuPy module 环境下的 pure-NumPy inversion/adjoint path；
- $12\times12$ identity inversion；
- general non-color-diagonal $\gamma_5S^\dagger\gamma_5$ oracle；
- `adj` 在 0/1/2 个 leading axes 下的 generalized batch oracle；
- companion leave-one-out jackknife shape；
- sorted bracketing、duplicate/out-of-range rejection、fit upper-bound inclusion；
- correlated-gvar interpolation 与 shared-source product covariance；
- `gamma/dim/am1/am1_0` 与 fit/read momentum arrays 的 complex、mixed-Boolean、nonfinite pre-conversion rejection；
- `read_data(pade_flag=0)` 不调用 Padé factor；
- static compile、YAML 与 public `__all__`。

**仍未验证：**

- 每个 conversion/anomalous-dimension coefficient 的 equation-level provenance；
- interacting NPR、real CuPy/GPU、QUDA 或 MPI；
- fit quality、full covariance、window/model stability 和 continuum extrapolation；
- production $\overline{\rm MS}$ renormalization constants。

## 13. 参考文献

1. K. G. Chetyrkin and A. Rétey, “Renormalization and Running of Quark Mass and Field in the Regularization Invariant and $\overline{\mathrm{MS}}$ Schemes at Three and Four Loops,” [arXiv:hep-ph/9910332v2](https://arxiv.org/abs/hep-ph/9910332), *Nucl. Phys. B* **583**, 3 (2000), DOI: 10.1016/S0550-3213(00)00331-X. Eqs. (34), (36), (37), (41) map the selected RI/RI-prime field and mass series；不是所有 copied decimals 的 blanket validation。
2. J. A. Gracey, “Three loop anomalous dimension of non-singlet quark currents in the RI' scheme,” [arXiv:hep-ph/0304113v1](https://arxiv.org/abs/hep-ph/0304113), *Nucl. Phys. B* **662**, 247 (2003), DOI: 10.1016/S0550-3213(03)00335-3. Eq. (4.11) fixes the mapped Landau-gauge tensor series and its $a_s^3/64$ conversion。
3. P. A. Baikov, K. G. Chetyrkin, and J. H. Kühn, “Quark Mass and Field Anomalous Dimensions to $O(\alpha_s^5)$,” [arXiv:1402.6611v1](https://arxiv.org/abs/1402.6611), *JHEP* **10**, 076 (2014), DOI: 10.1007/JHEP10(2014)076. Eqs. (3.1)–(3.4), (4.7)–(4.12) map the selected mass-running algebra。
4. P. A. Baikov, K. G. Chetyrkin, and J. H. Kühn, “Five-Loop Running of the QCD coupling constant,” [arXiv:1606.08659v2](https://arxiv.org/abs/1606.08659), *Phys. Rev. Lett.* **118**, 082002 (2017), DOI: 10.1103/PhysRevLett.118.082002. Eqs. (1)–(5) support the code's $a_s=\alpha_s/\pi$ beta coefficients。
5. C. Sturm et al., “Renormalization of quark bilinear operators in a momentum-subtraction scheme with a nonexceptional subtraction point,” [arXiv:0901.2599v2](https://arxiv.org/abs/0901.2599), *Phys. Rev. D* **80**, 014501 (2009), DOI: 10.1103/PhysRevD.80.014501. RI/SMOM primary context；不能 blanket-validate local decimal tables。
6. 本项目 `scripts/Def_quark_renorm.py` 与 companion `Def_qcd_analysis.py` 是 exact implementation evidence；它们不是 coefficient provenance 的替代品。逐式审计记录见 `reference/SOURCE_VERIFICATION.md`。

## 逐函数物理动机与证据卡

本 skill保持 explicit-only。以下“literal series”只说明代码实际计算什么；凡未给 exact paper version/equation、operator、input/output scheme、gauge、$N_f$ 与 expansion-variable mapping 的 coefficient均为 production-blocked。特别禁止根据相邻函数的 denominator pattern修补数值。

### Coupling 与 conversion functions

| Public symbol | 物理动机与 literal series | 输入输出与 scheme contract | 证据状态与 oracle |
|---|---|---|---|
| `QuarkRenormProfile` | 不可变的 formula-level provenance record，记录 operator、scheme、gauge、basis 与 expansion variable。 | 字段均为显式字符串或 `None`；`complete_provenance` 只在 mapped status 且字段齐全时为 true。 | registry structural oracle 直接检查；不把 record 当作 interacting NPR 证据。 |
| `QuarkRenormProfileError` | provenance-gated API 的基类异常。 | 暴露 `symbol/status/detail`，用于 fail-closed 的结构化错误。 | registry structural/rejection oracle 直接检查；不提升为物理结果。 |
| `UnverifiedLegacyError` | 拒绝把未逐式核验的 legacy decimal table 当作 physical formula。 | 由 `physical=True` 的 unmapped 请求触发，保留 `UNVERIFIED_LEGACY` status。 | fail-closed profile oracle 直接检查；未提供 coefficient provenance。 |
| `UnknownQuarkRenormProfileError` | 拒绝从未知函数名推断 operator 或 scheme provenance。 | 查询缺失 symbol 时抛出，禁止静默回退到 legacy literal。 | unknown-profile rejection oracle 直接检查；不产生数值或物理 claim。 |
| `QuarkRenormLiteral` | 显式包装一个 legacy callable，标记其只能作为 literal replay。 | `physical=False` 且保留 `profile` 与 `function`；调用不改变 provenance status。 | registry structural oracle 直接检查；literal replay 不等于 physical validation。 |
| `QUARK_RENORM_PROFILE_REGISTRY` | 汇总所有 quark-renorm symbol 到其 provenance profile 的只读 registry。 | `MappingProxyType`；包含 mapped 与 `UNVERIFIED_LEGACY` entries，禁止 caller mutation。 | registry inventory 与 immutability boundary 有直接 oracle。 |
| `QUARK_RENORM_PHYSICAL_PROFILES` | 仅筛选 complete、equation-mapped profiles，作为 physical gate 的 allowlist。 | 只包含通过 completeness gate 的 mapped records；legacy entries 不得出现。 | structural inventory oracle 直接检查；不替代 raw NPR 或 continuum evidence。 |
| `get_quark_renorm_profile` | 查询一个 symbol 的 machine-readable provenance。 | `physical=True` 要求 complete mapped profile；未知或不完整记录 fail closed。 | mapped/unknown profile rejection oracle 直接检查。 |
| `invoke_quark_renorm_profile` | 以 provenance gate 包装 legacy callable，阻止未核验公式进入 physical path。 | `physical=True` 只允许 complete profile；`physical=False` 才允许 literal invocation。 | mapped invocation 与 legacy rejection oracle 直接检查；不声称 production physics。 |
| `get_quark_renorm_legacy_literal` | 返回带有 explicit-only 标记的 legacy callable wrapper。 | `physical=False` 才能取得 `QuarkRenormLiteral`；physical 请求保持 fail closed。 | wrapper identity/status/rejection oracle 直接检查。 |
| `beta_coupling_constant` | 为 $a_s=\alpha_s/\pi$ 构造五阶 beta coefficients及 $\beta_i/\beta_0$ ratios。 | `nf` non-Boolean integer `0..16`；返回 length-5 host array。 | BCK2016 Eqs. (1)–(5) 逐项映射；这不提供 threshold matching 或 coupling-fit uncertainty。 |
| `strong_coupling_constant` | 计算 $a_s(\mu)$ 的 1–5 loop asymptotic partial sums，$L=2\log(\mu/\Lambda)$。 | finite positive `scale,Lambda`且 ratio $>3$；`Lambda=None`只在 `nf=3`映射0.332 GeV。 | BCK2016 fixes the beta convention；finite/domain/flavor gate有回归，默认 Lambda 与 threshold treatment 仍是项目约定。 |
| `alpha_s` | 返回 five-loop entry乘 $\pi$，即 code convention的 $\alpha_s(\mu)$。 | 允许显式 `Lambda`；`nf\ne3`若不提供 Lambda则 fail closed。 | wrapper行为已测试；不等同 global fit或阈值匹配。 |
| `vector_conversion_ms_bar_over_rimom_prime` | 历史名称；实际 series 是 CR2000 $C_2^{RI'}/C_2^{RI}$ 通过 $O(a_s^3)$ 的 fixed-order ratio，不是已验证的 vector-current conversion。 | 返回五个 coupling-truncation entries；external-field/$Z_q$ direction 由 caller convention 决定。 | 由 CR2000 Eqs. (36)/(34) 独立展开并回归；historical name 不得提升为 vector-current physics claim。 |
| `quark_mass_conversion_ms_bar_over_rismom` | literal $C_m=1-0.1613797a_s-0.66044182a_s^2$，面向 RI/SMOM mass conversion。 | host length-5；默认仅 $N_f=3,\Lambda=0.332$。 | Sturm et al.支持 RI/SMOM method class；decimals无逐式 map，production-blocked。 |
| `quark_mass_conversion_ms_bar_over_rismom_mu` | literal $C_m=1-0.494713025a_s-(55.03243483-6.161687618N_f)a_s^2/16$，代表代码命名的 SMOM-$\gamma_\mu$ branch。 | scale/Lambda/nf显式 contract；输出 length-5。 | scheme name来自 legacy source；exact projector/equation未认证，production-blocked。 |
| `quark_field_conversion_rimom_prime_over_rimom` | CR2000 $C_2^{RI'}/C_2^{RI}$ 的 fixed-order ratio wrapper；不是由名称单独定义的 $Z_q$ observable。 | exact wrapper，返回与 historical vector-named factor相同 array；方向需结合 CR2000 Eqs. (9)–(10) 与 caller $Z_q$ convention。 | Eqs. (34)/(36) 的独立 series division regression 已通过；interacting NPR 未运行。 |
| `quark_field_conversion_ms_bar_over_rimom` | CR2000 Eq. (34) 的 $C_2^{RI}=1+c_2a_s^2/16+c_3a_s^3/64$ literal series。 | SU(3)、Landau gauge；host small array，不接收 raw vertex。 | primary locator 与 expansion-variable conversion 已闭合；physical $Z_q$ direction 仍由 Eqs. (9)–(10)/caller convention 决定。 |
| `quark_field_conversion_ms_bar_over_rimom_prime` | CR2000 Eq. (36) 的 $C_2^{RI'}=1+c_2a_s^2/16+c_3a_s^3/64$ literal series。 | SU(3)、Landau gauge；scheme label必须与 upstream $Z_q$ definition一致，输出 length-5。 | primary locator 与逐项 algebra 已闭合；interacting gauge-fixed NPR 未运行。 |
| `quark_mass_conversion_ms_bar_over_rimom_prime` | CR2000 Eq. (37) 的 $C_m^{RI'}=1+c_1a_s/4+c_2a_s^2/16+c_3a_s^3/64$。 | SU(3)、Landau gauge RI-prime/MOM mass factor；scalar inverse只能在同一 convention下使用。 | Eq. (41) 的 $N_f=4,a_s=0.1$ reference point 与逐式 regression 已通过；continuum fit/uncertainty 未验证。 |
| `scalar_conversion_ms_bar_over_rismom` | 利用 $Z_S=1/Z_m$ 逐 truncation entry取 reciprocal。 | 输入/输出 scheme继承 `quark_mass_conversion_ms_bar_over_rismom`；zero factor会失败。 | inverse relation是 Ward-identity context；underlying decimals仍 blocked。 |
| `scalar_conversion_ms_bar_over_rismom_mu` | 对 SMOM-$\gamma_\mu$ mass factor逐 entry取 reciprocal。 | host array，不传播 coefficient covariance。 | implementation derivation；underlying scheme/equation未闭合。 |
| `scalar_conversion_ms_bar_over_rimom_prime` | 对 source-mapped RI-prime/MOM mass conversion逐 entry取 reciprocal。 | 只有在 scalar/mass conventions成对且 Ward-identity assumptions适用时有物理意义。 | underlying CR2000 mass coefficients已映射；reciprocal是本地 derivation，尚无 interacting scalar NPR/continuum proof。 |
| `tensor_conversion_ms_bar_over_rismom` | literal $C_T=1-0.05379324a_s-1.94213215a_s^2$。 | RI/SMOM tensor candidate，host length-5。 | method class有文献；decimal provenance未定位。 |
| `tensor_conversion_ms_bar_over_rismom_mu` | literal $C_T=1+0.279540095a_s-(8.607630493-1.955130440N_f)a_s^2/16$。 | `Lambda`为显式 required参数；命名为 SMOM-$\gamma_\mu$ branch。 | source literal可审计；exact equation未认证。 |
| `tensor_conversion_ms_bar_over_rimom_prime` | G2003 Eq. (4.11) 在 $\xi=0$ 下给出 $C_T^{RI'/\overline{MS}}=1+c_2a^2+c_3a^3$，$a=\alpha_s/(4\pi)$；code 取其 fixed-order inverse：$1-c_2a_s^2/16-c_3a_s^3/64$。 | 仅显式调用；已做 source-algebra/reference regression，但不构成 production RI-prime/MOM tensor NPR。 | source、方向与 `/64` 已闭合；相邻 SMOM/general-$\xi$/Padé tables 仍不可由此类推。 |
| `quark_mass_conversion_ms_bar_over_rimom_prime2` | 展开一般 covariant-gauge parameter $\xi$ 的高阶 mass conversion coefficients。 | finite real `xi`、explicit scale/Lambda/nf；返回 cumulative series array。 | code包含 literal analytic expressions，但缺 paper/equation mapping，production-blocked。 |
| `quark_field_conversion_rimom_prime_over_rimom2` | 计算一般 $\xi$ 下 RI-prime/RI field-factor高阶 series。 | `xi`不能 NaN/complex；输出 truncation array。 | literal source evidence only；operator/gauge normalization需逐式核验。 |
| `scalar_conversion_ms_bar_over_rimom_prime2` | 对 general-$\xi$ mass factor取 reciprocal得到 scalar candidate。 | 继承 mass2的 scheme/gauge/nf；host array。 | implementation relation；underlying coefficients blocked。 |
| `tensor_conversion_ms_bar_over_rimom_prime2` | 计算 general-$\xi$ tensor conversion series。 | finite `xi`；output cumulative truncation entries。 | literal source可审计；无 exact source equation/reference oracle。 |

### RG、systematics 与 Padé functions

| Public symbol | 物理动机与 literal series | 输入输出与 scheme contract | 证据状态与 oracle |
|---|---|---|---|
| `anomalous_dimension` | 将 $\gamma(a_s)/\beta(a_s)$ 重新展开为 evolution coefficients `dim`，供 $c(a_s)$ ratio使用。 | finite real 1D length `3..5`与 explicit `nf`；built-in/`UserList`/custom non-string `Sequence`/object-array Boolean及 complex在 float conversion 前拒绝，返回同长度。 | 代数递推与 generic-Sequence silent-narrowing regression可审计；caller预制 float ndarray的历史 provenance不可恢复；各 operator $\gamma_n$ provenance另行要求。 |
| `scale_running` | 计算 $U(\mu,\mu_0)=c(a_s(\mu_0))/c(a_s(\mu))$ 的逐阶近似。 | positive real scales、finite real `dim`、explicit Lambda/nf；Boolean/complex `dim` fail closed，返回 length `3..5`。 | 实现公式与数值域 regression直接；threshold matching与 uncertainty未实现。 |
| `quark_mass_anomalous_dimension_under_ms_bar` | 用 BCK2014 的 five-loop $\overline{MS}$ mass anomalous dimensions构造 code orientation $c(a_s(\mu_0))/c(a_s(\mu))$。 | defaults `scale0=2 GeV`与 nf3 Lambda convention；host array；该方向是 BCK2014 Eq. (4.7) mass ratio 的 inverse。 | Eqs. (3.1)–(3.4), (4.7)–(4.12) 的 coefficient/c-function algebra 已映射，equal-scale/composition regression通过；threshold与 uncertainty 未实现。 |
| `quark_field_anomalous_dimension_under_ms_bar` | 用 code中的 field anomalous dimensions构造 quark-field running。 | gauge dependence必须与 conversion factor一致；最后未知阶在 source中置0。 | 置0不是已知 coefficient；必须计入 truncation，production-blocked。 |
| `scalar_anomalous_dimension_under_ms_bar` | 利用 scalar/mass inverse relation逐 entry取 $1/U_m$。 | 继承 mapped mass-running scale/nf/Lambda；flavor/Ward-identity convention由 caller 保证。 | underlying BCK2014 mass coefficients已映射；scalar wrapper是本地 reciprocal algebra，不是独立 coefficient table或 observable proof。 |
| `tensor_anomalous_dimension_under_ms_bar` | 由 code tensor anomalous-dimension array生成 tensor running。 | host small array；source最后一项置0。 | coefficients与0-padding无 equation-level provenance，production-blocked。 |
| `matching_systematic_error` | 生成七个 legacy channels在 statistical/truncation/running/$\Lambda_{QCD}$ variants下的 multiplicative factors。 | 仅 `scale0=2, Lambda=0.332, nf=3`；`MOM_flag`和`error_flag`拒绝 Boolean。 | 七 channels的物理名称与 decimals未逐式定位；保持 explicit-only，不能称 complete uncertainty model。 |
| `scalar_conversion_ms_bar_over_rimom_pade_3loop` | 对 scalar RI/MOM conversion构造 code-defined 3-loop Padé variant。 | host array；Padé只估计 perturbative next-order structure。 | 数值算法实现可审计；不替代 exact coefficient或 truncation posterior。 |
| `scalar_conversion_ms_bar_over_rimom_pade_4loop` | 构造 scalar conversion的 code-defined 4-loop Padé variant。 | 同上；singular/unstable cases须由 caller比较 fixed order。 | legacy estimate，无独立 reference-point validation。 |
| `tensor_conversion_ms_bar_over_rimom_pade_3loop` | 构造 tensor conversion 3-loop Padé variant。 | scheme必须与 tensor base factor一致；host array。 | implementation estimate；physical systematic需多方案比较。 |
| `tensor_conversion_ms_bar_over_rimom_pade_4loop` | 构造 tensor conversion 的 code-defined 4-loop Padé variant，用于与 fixed-order series 比较 truncation sensitivity。 | 输入 finite positive scale、flavor-compatible Lambda/nf；返回 host truncation array，scheme 必须与 tensor base factor一致。 | implementation estimate；无 exact literature coefficient claim，也不替代 coefficient-level provenance。 |
| `pade_matching_factor` | 返回 scalar base、3-loop Padé 与 4-loop Padé 三个 matching variants；不把未返回的 tensor/running channels当作工作量或物理输出。 | 返回 `(scalar_base,scalar_pade3,scalar_pade4)`；`scale0` 为 deprecated finite-positive compatibility no-op；discarded tensor/running helpers调用次数严格为零。 | preserved numerical oracle与 zero-call regression直接；Padé decimals仍缺逐式 provenance，production-blocked。 |

### I/O、fits、spin-color 与 companion function

| Public symbol | 物理动机与 literal algebra | 输入输出、shape 与 statistical contract | 证据状态与 oracle |
|---|---|---|---|
| `read_data` | 按五段 momentum table layout读取 bare/converted data，选择 $fitmin<a^2p^2\le upper$ window并可应用 Padé ratio。 | 验证 non-Boolean counts/flags、required columns、finite values与 errors；`pade_flag=0` 时不计算任何 Padé factor；返回 masses、means/errors、$a^2p^2,p$。 | disabled-Padé zero-call regression直接；schema为 project-specific实现约定，不通用于任意 NPR table，production I/O样本未验。 |
| `ma_fit` | 对每个 momentum拟合 $A/(am)^2+B+C(am)$，取 chiral-limit/intercept-like $B$。 | means/errors equal 2D shape，至少3 masses，$am\ne0$；gvar/lsqfit host path，prior由 caller提供。 | deterministic schema已加 gate；fit model、correlations与质量未 physics-validated。 |
| `a2p2_fit` | 对 mass-extrapolated data拟合 $C_0+C_1a^2p^2+C_2(a^2p^2)^2+C_3(a^2p^2)^3$并取 $C_0Z_A$。 | equal finite 1D arrays、positive momenta；`am1` finite real positive且 complex/Boolean在 optional imports前拒绝；lattice flag显式。 | numerical-domain regression直接；model是 legacy ansatz，full covariance/window/continuum systematic未闭合。 |
| `ratio_fit` | 将 small-ensemble data插到 reference momenta，拟合 ensemble ratio的 $a^2p^2\to0$ intercept。 | 两 momentum grids需 bracket、denominator nonzero；`am1/am1_0`分别 finite real positive；gvar objects保留共享 covariance，$Z_A$显式。 | interpolation covariance与 silent-complex-scale rejection有 oracle；ensemble matching assumptions与 fit stability未 production验证。 |
| `inverse_propagator` | 将末四轴映射为 compound $(12,12)$ 并逐 batch求 $S^{-1}$。 | input `(...,4,4,3,3)`；NumPy/CuPy backend自动推断，`on_device`仅作 strict Boolean一致性检查，truthy非 Boolean在 inverse前拒绝。 | arbitrary leading-axis、pure-NumPy与 pre-inversion flag oracles支持；真实 CuPy未验证。 |
| `adj` | 计算 $\gamma_5S^\dagger\gamma_5$，dagger同时交换 source/sink spin与color。 | `S=(...,4,4,3,3)`、`gamma5=(4,4)`；小 gamma可显式移到 volume backend。 | non-color-diagonal 12x12 oracle直接关闭旧 color-axis bug。 |
| `Lambda_O_con` | 对 connected Green function执行 $\Lambda_O=S^{-1}G_OS^{-\dagger}_{\gamma_5}$ 的 code contraction。 | exact `(Ncfg,4,4,3,3)` for `Sq,GreenB`，禁止 batch broadcast；`is_jack`为0/1 non-Boolean。 | shape/backend gate直接；compound contraction尚需目标 projector/normalization，非 raw RI constant。 |
| `Lambda_O_dis` | 构造 disconnected $G_O=S\,J$ 后作同样 amputation。 | `current=(Ncfg,)`必须与 `Sq`配置轴一致；exact layout与flag gate。 | implementation algebra；disconnected subtraction/noise/covariance由 caller负责。 |
| `jackknife_resampling` | companion helper生成 leave-one-out samples $\bar x_{(i)}=(\sum_jx_j-x_i)/(N-1)$，供 amputation/fits传播配置波动。 | 第一轴为 configurations且 $N>1$；返回同 shape NumPy array。 | 标准 jackknife代数与 shape oracle支持；不自动生成 covariance或 bias correction。 |
