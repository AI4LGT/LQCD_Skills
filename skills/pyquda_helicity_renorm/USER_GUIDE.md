# `pyquda_helicity_renorm` 用户说明书

## 1. 适用范围与最高优先级警告

本 skill 是一个小型 CPU/NumPy/SciPy legacy module，用于审查或谨慎复用 $2\times2$ helicity/EMT matching matrices、$n_f=3$ RG evolution、momentum interpolation 和 outer-leg subtraction。

代码中 beta-function context 可核查，但 helicity matching 与 anomalous-dimension decimal tables 尚未逐项映射到原始论文、operator basis、scheme 和 equation。因此：

- 可以做 array algebra、domain、loop placement 和 sensitivity audit；
- 可以接受用户提供并已核实的 coefficient matrices；
- 在 provenance gate 未通过前，不能把 bundled decimals 当作 production QCD coefficients。

这也是 `allow_implicit_invocation: false` 的原因：仅出现 “helicity” 一词不应自动调用 legacy numerical tables。

逐项 basis、matrix orientation、Padé/ODE 与 provenance blocker 见 `reference/PHYSICS_CONTRACT.md`。
当前 release label、explicit-only 条件、reference-reproduction manifest 与证据升级流程见 `VALIDATION.md`。

## 2. 如何用自然语言调用

> 使用 `$pyquda_helicity_renorm` 审查这个 $2\times2$ matching matrix。operator basis 依次是我给出的 $(O_1,O_2)$，scheme 和 gauge 见论文 Eq. (...)；请核对 `R12/R21` 的 row-column placement，不要改任何 coefficient。

> 用 `$pyquda_helicity_renorm` 比较 `loop=3` 与 `loop=4,is_pade=1`。明确后者只是 $r_4\approx r_3^2/r_2$ 的 Padé estimate，并报告对 scale variation 的敏感性。

> 使用 `$pyquda_helicity_renorm` 解 $n_f=3$ 的 coupled RG equation，但先把 $a_s=\alpha_s/\pi$、$d/d\ln\mu^2$、matrix orientation 和初值写清楚。

> 用 `$pyquda_helicity_renorm` 只做 provenance audit：列出缺失的 paper、operator basis、scheme、gauge、loop normalization 和 equation mapping；不要输出 production number。

## 3. Coupling convention

`beta(Nf)` 使用

$$
a(\mu)=\frac{\alpha_s(\mu)}{\pi}
$$

的 convention，并把 beta function 写为

$$
\frac{da}{d\ln\mu^2}
=-\beta_0a^2
\left(1+b_1a+b_2a^2+b_3a^3+b_4a^4+\cdots\right).
$$

函数返回 $\beta_0$ 和 ratios $b_i=\beta_i/\beta_0$。`alpha_s(mu_scale,Nf,Lambda)` 通过 asymptotic expansion 给出从 one-loop 到 five-loop 的五个 successive truncations：

```text
alpha_orders[0:5]
```

返回值是 $\alpha_s$，不是 $a$。代码要求

$$
\frac{\mu}{\Lambda}>2,
$$

这是实现 domain gate，不代表靠近边界的 perturbative truncation 已可靠。

`Lambda` 省略时，默认 $\Lambda=0.332\,\mathrm{GeV}$ **只对 $n_f=3$ 定义**。若 `Nf != 3`，调用者必须显式传入与 flavor scheme 匹配的 `Lambda`；代码会 fail closed，而不是沿用 $n_f=3$ 数值。

matching functions 随后定义

$$
\bar a=\frac{\alpha_s^{(5\,loop)}(\mu)}{4\pi}.
$$

审查 coefficients 时必须区分 $a=\alpha_s/\pi$ 与 $\bar a=\alpha_s/(4\pi)$，否则每阶会产生 $4^n$ 的 normalization error。

## 4. $2\times2$ matching matrix

API 固定返回

$$
R=
\begin{pmatrix}
R_{11}&R_{12}\\
R_{21}&R_{22}
\end{pmatrix},
$$

flattened tuple 顺序是

```text
(R11,R12,R21,R22)
```

但代码本身没有记录 $(O_1,O_2)$ 的物理 basis。production 前必须明确：

$$
\begin{pmatrix}O_1^{\rm out}\\O_2^{\rm out}\end{pmatrix}
=R
\begin{pmatrix}O_1^{\rm in}\\O_2^{\rm in}\end{pmatrix}
$$

还是采用 row-vector/right multiplication；二者不能凭 tuple order互换。

对每个 entry，结构为

$$
R_{ij}=R_{ij}^{(0)}
+\bar a\,r_{ij}^{(1)}
+\bar a^2r_{ij}^{(2)}
+\bar a^3r_{ij}^{(3)}
+\bar a^4r_{ij}^{(4)}.
$$

在 $n_f=3$、$\mu=\mu_R$ 的 code algebra 下，one-loop off-diagonal coefficients 给出

$$
r_{12}^{(1)}=8,
\qquad
r_{21}^{(1)}=-6.
$$

本地测试只证明这两个 numbers 被放在 `R12/R21` 的一致 positions；它不证明 numbers 对目标 operator basis/scheme 正确。

`Helicity_MatchingCoeff_tmp` 与 primary `Helicity_MatchingCoeff` 的 $R_{11}^{(0)}$ convention 不同：前者从 $1$ 开始，后者包含一个 legacy rational prefactor。两者不能只因其余 entries 类似而混用。

## 5. Loop 与 Padé 语义

- `loop=1,2,3`：使用代码中显式的一、二、三阶 terms。
- `loop=4,is_pade=0`：四阶 coefficient 设为零，因此数值上仍是 explicit three-loop truncation。
- `loop=4,is_pade=1`：逐 matrix entry 使用 legacy estimate

  $$
  r_{ij}^{(4)}\approx\frac{\left(r_{ij}^{(3)}\right)^2}{r_{ij}^{(2)}}.
  $$

这不是已知 four-loop coefficient。若 denominator 很小，estimate 可能不稳定；应报告 denominator、scale variation 和与 three-loop result 的差异。

`Helicity_MatchingCoeff_array` 返回

```text
(4,n_momenta) = (R11,R12,R21,R22; momentum)
```

其 scale modes 为：

- `is_fixing_order=1`：所有 $\mu_R$ 使用同一个 positive `muscale`；
- `is_fixing_order=0`：使用 $\mu=\mu_R\times\texttt{muscaleOVmuR}$。

历史参数名不够直观，production manifest 应同时保存实际 `muR_array` 和逐点 `mu_scale_array`。

## 6. Coupled RG evolution

`Helicity_RunningFactorCalculator` 当前只允许

```text
nf=3, ope_type='ghelicity'
```

并用 $a=\alpha_s/\pi$ 解

$$
\frac{da}{d\ln\mu^2}=\beta(a),
$$

$$
\frac{dR}{d\ln\mu^2}=\Gamma(a)R,
\qquad
\Gamma(a)=\sum_{n\ge0}\Gamma_n a^{n+1}.
$$

SciPy `solve_ivp` 使用 `RK45`、`rtol=10^{-8}`、`atol=10^{-10}`。若输入初值为 $R(\mu_1)$，返回 evolution matrix

$$
U(\mu_2,\mu_1)
=R(\mu_2)R^{-1}(\mu_1),
$$

满足

$$
R(\mu_2)=U(\mu_2,\mu_1)R(\mu_1).
$$

公开方法是 `run` 与 `run_array`；`calculate_*` 保留 compatibility。初始矩阵 `R_at_mu1/R_init` 必须是 finite real `(2,2)`；Boolean 与 complex arrays 在 float conversion 前 fail closed。decimal $\Gamma_n$ table 目前缺少 equation-level provenance，所以 ODE numerical accuracy 不等于 anomalous dimensions 的 physics correctness。

## 7. EMT helper、interpolation 与 subtraction

### 7.1 `EMT_MatchingCoeff`

legacy scalar helper 使用 $n_f=N_c=3$ 和

$$
R_{\rm MS}=1
+\frac{g_0^2n_f}{16\pi^2}
\left[\frac23\ln\frac{\mu^2}{\mu_R^2}+\frac{10}{9}\right]
-\frac{g_0^2N_c}{16\pi^2}\frac5{12}.
$$

由于原 source 没有记录 exact paper/operator equation，在补齐 provenance 前应视为 audit-only helper。

### 7.2 `momentum_interpolate`

函数先按 momentum 排序 reference points，拒绝 duplicates 与 range 外 targets；对每个非精确 target 使用左右两个 **bracketing** points，并对 means 线性插值：

$$
\bar y=(1-w)y_0+wy_1,
\qquad
w=\frac{x-x_0}{x_1-x_0}.
$$

在没有 covariance input 时，代码明确采用 endpoint independence assumption：

$$
\sigma_y^2=(1-w)^2\sigma_0^2+w^2\sigma_1^2.
$$

target/reference momenta 必须是 finite real one-dimensional arrays。代码先检查 raw dtype，再转成 `float`；complex 或 Boolean arrays 会直接报错，因此虚部和 truth value 不会被 NumPy narrowing 静默丢弃。

如果两点来自同一 gauge ensemble，这一 independence 通常并不成立；应传递 resamples 或完整 covariance 重新实现。

### 7.3 `AA_subtract`

函数在指定 $a^2p^2$ window 内拟合

$$
f(x)=\sum_{k=0}^{n_{\rm poly}}c_kx^k
$$

并返回

$$
\text{subtract\_fac}(x)
=\frac{c_0}{\langle A/A\rangle(x)}.
$$

它先拒绝显著 complex input；调用者必须先选择并记录物理 real projection。随后使用 configuration-axis mean/std、diagonal gvar errors 和宽 Gaussian priors，但不接收 full covariance matrix。返回值只使用 posterior $c_0$ 的 mean，再除以 full-range sample mean；posterior fit uncertainty 没有传播到 `subtract_fac`。因此它应标成 exploratory diagonal-error fit，fit window、polynomial order、prior sensitivity、full-covariance/resample comparison 与被丢弃的 fit uncertainty 都应报告。

## 8. API 速查

| API | 作用 | 关键边界 |
|---|---|---|
| `beta`, `alpha_s` | five-loop coupling context | $\alpha_s/\pi$ convention |
| `EMT_MatchingCoeff` | legacy scalar EMT factor | exact provenance missing |
| `Helicity_MatchingCoeff[_tmp]` | one momentum 的 $2\times2$ matching | basis/scheme mandatory |
| `Helicity_MatchingCoeff_array` | momentum array evaluation | actual scales must be recorded |
| `Helicity_RunningFactorCalculator` | $n_f=3$ coupled RG | decimal $\Gamma$ unmapped |
| `Get_Helicity_Running` | $n_f=3$ wrapper | returns real part by legacy contract |
| `momentum_interpolate` | two-point interpolation | independence assumption |
| `AA_subtract` | host polynomial subtraction | covariance not accepted |

最小 host-side审查示例为：

```python
from skills.pyquda_helicity_renorm.scripts.Def_helicity_renorm import (
    Helicity_MatchingCoeff,
    alpha_s,
)

alpha_orders = alpha_s(mu_scale=3.0, Nf=3)
matching = Helicity_MatchingCoeff(muR=2.0, mu_scale=3.0, loop=3, is_pade=0)
```

这个示例只演示 API/domain；在 coefficient equation map 完成前，`matching` 不能标为 production QCD result。

## 9. Mandatory provenance gate

在任何 production numerical use 前，逐项提供并核对：

1. 原始 paper、version、section/equation；
2. operator basis 的顺序和 normalization；
3. row/column evolution convention；
4. RI/intermediate scheme、gauge 和 external kinematics；
5. coupling expansion variable $\alpha_s/\pi$ 或 $\alpha_s/(4\pi)$；
6. $n_f$、$N_c$、loop order 和 scale-log definition；
7. matching 与 RG 是左乘还是右乘；
8. Padé 是 uncertainty model 还是 nominal central value。

任一项缺失时，skill 应输出审查报告或使用 caller-supplied matrices，不应输出带“已验证 QCD matching”标签的结果。

## 10. 常见错误检查表

- [ ] 是否混淆了 $\alpha_s/\pi$ 与 $\alpha_s/(4\pi)$？
- [ ] `(O1,O2)` basis 与 `R11,R12,R21,R22` 是否明确？
- [ ] matrix 是左乘还是右乘？
- [ ] `tmp` 与 primary $R_{11}$ convention 是否混用？
- [ ] `loop=4,is_pade=0` 是否被误称为 explicit four-loop？
- [ ] Padé denominator 是否接近零？
- [ ] $n_f\ne3$ 是否被静默代入 legacy anomalous dimensions？
- [ ] ODE solver success 是否被误写成 coefficient provenance proof？
- [ ] correlated momentum points 是否被按独立误差传播？
- [ ] `AA_subtract` 的 complex projection、full covariance 与 posterior fit uncertainty 是否有独立处理？

## 11. 已验证与未验证

**本地直接验证：**

- coupling/matching input domain；
- $n_f=3$-only default $\Lambda$、其他 flavors 的 explicit-`Lambda` gate；
- one-loop `R12/R21` placement；
- `loop`/Padé branches 的 finite array behavior；
- `nf=3` 和 operator-type gates；
- lazy host fit imports、静态编译与 public API。
- interpolation sorting/bracketing、duplicate/out-of-range rejection 与 independent-error propagation；
- matching/RG/Get/interpolation 的 complex/Boolean pre-conversion rejection，
  包括 list/tuple、`UserList`、custom non-string `Sequence` 与 object arrays；
- `AA_subtract` 对显著 complex input 的 fail-closed gate。

**仍未验证：**

- helicity/EMT matching decimals 的 original-equation mapping；
- anomalous-dimension table 的 basis、scheme 和 flavor provenance；
- production scale evolution、truncation uncertainty 和 covariance；
- 任何 GPU lattice-volume work（本 skill 也不应承担该工作）。

### Physical-profile keyword gate

`Helicity_MatchingCoeff`、`Helicity_MatchingCoeff_tmp`、
`Helicity_MatchingCoeff_array`、`Helicity_RunningFactorCalculator` 和
`Get_Helicity_Running` 都有 keyword-only
`physical=False, physical_profile=None`。默认及显式 `physical=False` 只返回
legacy literal replay；`physical=True` 在任何 numerical evaluation 前审计
ordered basis、scheme/gauge、`N_f`/color/normalization、coupling/derivative/log
conventions、逐 entry locator、truncation、Padé/scale domain 及 uncertainty
record。当前没有注册 calculator profile：缺字段得到
`INCOMPLETE_PHYSICAL_PROFILE`，Zhao v2 在 matching API 上得到
`PRIMARY_SOURCE_CONTRADICTION`，running API 的未闭合 RG map 得到
`UNVERIFIED_PHYSICAL_PROFILE`，完整但未注册 metadata 得到
`UNVERIFIED_NO_MAPPED_PHYSICAL_PROFILE`。这些是 local refusal dispositions，
不验证 physical matching/running、NPR 或 lattice observable。

## 12. 参考文献

1. P. A. Baikov, K. G. Chetyrkin, and J. H. Kühn, “Five-Loop Running of the QCD coupling constant,” [arXiv:1606.08659](https://arxiv.org/abs/1606.08659), *Phys. Rev. Lett.* **118**, 082002 (2017), DOI: 10.1103/PhysRevLett.118.082002. 支持 five-loop beta-function context；不验证 helicity matching/anomalous-dimension tables。
2. bundled `scripts/Def_helicity_renorm.py` 是当前 matrix placement、Padé branch 和 ODE orientation 的直接依据。
3. helicity matching coefficient 的 primary literature/equation 目前仍是 production blocker；在确认前不附会一个通用引用为整张 decimal table 背书。

## 逐函数物理动机与证据卡

本 skill 保持 explicit-only。表中写出的 series 是 literal code contract，不代表每个 legacy coefficient 已获文献认证；没有 exact operator basis、scheme、gauge、$N_f$、expansion variable 与 equation locator 的条目继续 fail closed。

| Public symbol | 物理动机与 literal 公式 | 输入输出与数值域 | 证据状态与 oracle |
|---|---|---|---|
| `beta` | 给出 $a_s=\alpha_s/\pi$ convention 下五阶 $\beta$ coefficients，并返回 $(\beta_0,\beta_1/\beta_0,\ldots,\beta_4/\beta_0)$ 供 asymptotic solution 使用。 | `Nf` 为 non-Boolean integer `0..16`；host tuple，不接收 lattice arrays。 | Baikov-Chetyrkin-Kühn支持 five-loop context；normalization由代码注释与 analytical expression直接审计。 |
| `alpha_s` | 计算五个 truncation orders的 $\alpha_s(\mu)$，基本变量 $L=2\log(\mu/\Lambda)$，最后乘 $\pi$。 | $\mu,\Lambda$ finite positive且 $\mu/\Lambda>2$；默认 $\Lambda=0.332$ GeV仅允许 $N_f=3$。 | perturbative running method直接；默认 Lambda是项目约定，不是普适 world average。NaN/Inf/domain已有回归。 |
| `EMT_MatchingCoeff` | 实现 code-defined one-loop scalar factor $R_{MS}=1+g_0^2N_f(2\log(\mu^2/\mu_R^2)/3+10/9)/(16\pi^2)-5g_0^2N_c/(192\pi^2)$。 | finite positive $g_0,\mu_R,\mu$；固定 $N_f=N_c=3$，返回 host scalar。 | literal formula可审计，但缺 operator/scheme/equation locator，production-blocked。 |
| `Helicity_MatchingCoeff_tmp` | 构造 $R=I+\sum_{n=1}^3a^nr_n$，其中 $a=\alpha_s/(4\pi)$；四阶可用 $r_4=r_3^2/r_2$ Padé estimate。 | 返回 `(R11,R12,R21,R22)`；loop `1..4`、Padé flag `0/1` 均拒绝 Boolean；singular denominator fail closed。 | matrix placement与 one-loop off-diagonal oracle已检查；整套 coefficients无逐式 provenance，experimental/production-blocked。 |
| `Helicity_MatchingCoeff` | 与 tmp branch同为 $2\times2$ quark/gluon mixing matrix，但 `R11` 含 code-defined resummed-like $r_{0,11}$ term。 | scale/domain同上；输出顺序固定 `11,12,21,22`，不得按 row-major assumption之外重排。 | 实现差异直接来自 source；legacy decimals未定位，必须显式调用且不得发表 production number。 |
| `Helicity_MatchingCoeff_array` | 在多个 $\mu_R$ 上评估 matching matrix，可用 fixed $\mu$ 或比例 $\mu/\mu_R$，便于 momentum scan。 | 输入 finite positive 1D array；built-in/`UserList`/custom non-string `Sequence`/object-array 中的 Boolean 在 NumPy dtype inference 前拒绝；返回 `(4,N)`；fixed/proportional branches互斥。 | vector wrapper有 shape/finite/generic-Sequence Boolean regressions；每一点继承 underlying coefficient provenance blocker。 |
| `Helicity_RunningFactorCalculator` | 解 $dR/d\ln\mu^2=\Gamma(a_s)R$ 与 $da_s/d\ln\mu^2=\beta(a_s)$，返回 evolution $U=R(\mu_2)R^{-1}(\mu_1)$。 | 仅 `nf=3, ope_type='ghelicity'`；$\mu_1,\mu_2>0$，`R_init` 为 finite real `(2,2)`；generic-Sequence/object-array Boolean与 complex在 SciPy import前拒绝。 | ODE orientation、负终点及 silent-coercion rejection已测试；$\Gamma$ decimals无 equation-level provenance，production-blocked。 |
| `Get_Helicity_Running` | 把 input scales乘 `muscaleOVmuR` 后调用 coupled RG calculator，并返回 code-assumed real evolution。 | scalar或finite positive real 1D `mu_from`；mixed Boolean在 array inference与乘比例前拒绝；输出 `(2,2)` 或 `(N,2,2)` NumPy。 | convenience implementation；raw-Boolean回归已闭合，caller预先生成的 float ndarray已无 provenance，`.real` 仍是 explicit legacy output contract。 |
| `momentum_interpolate` | 将 reference momentum table的 mean/error slots线性插到 target grid；误差按独立 endpoints作 quadrature。 | momentum arrays为 finite real 1D；mixed Boolean/object arrays与 complex在 float conversion 前拒绝；data形状 `(Ncase,slots,Np)`，slots 8/9为 mean/sdev；禁止重复和外推。 | 数值插值与 silent-narrowing regressions直接；不保留 cross-momentum covariance，`datasets`只提供 target schema而非数值。 |
| `AA_subtract` | 在 window内拟合 $A_0+A_1a^2p^2+\cdots+A_n(a^2p^2)^n$，以 $A_0/\langle A/A\rangle$ 构造 outer-leg subtraction factor。 | `AovA=(samples,Np)` real finite；window至少 $n+1$ points，zero denominator拒绝；gvar/lsqfit lazy host path。 | deterministic validation已覆盖；sdev是 pointwise sample std且未建 covariance，这一统计假设必须报告。 |
