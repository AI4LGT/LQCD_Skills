# `pyquda_ri_renorm` 用户说明书

## 1. 适用范围与边界

本 skill 从已经 gauge-fixed、Fourier-transformed 的 off-shell quark propagators 和 Green functions 出发，提供 RI-prime/MOM 与 nonexceptional NPR kinematics 的数组级 inversion、amputation、projector，以及使用 propagator-$Z_q$ 的 raw bilinear factors。nonexceptional 输出被显式标成 custom/unmatched，不能仅凭 $\omega$ 和 projector family 当作标准 RI/SMOM scheme。

它不负责 gauge fixing、momentum-source construction、QUDA inversion、MPI reduction、orbit averaging、window fit、$a^2p^2$ extrapolation 或 $\overline{\mathrm{MS}}$ conversion。后者应交给 `pyquda_quark_renorm`，且必须通过 coefficient provenance gate。

本地验证是 tree-level NumPy oracle 和 shape/backend contract；没有 interacting gauge-fixed data、真实 CuPy 或 production NPR 结果。

完整 kinematics/projector/scheme identity 与 Ward-identity 条件见 `reference/PHYSICS_CONTRACT.md`。
当前 release label、可运行测试、gauge-fixed E2E manifest 与证据升级流程见 `VALIDATION.md`。

所有 gamma/projector/tree matrices 必须是同一 NumPy/CuPy backend 上的
numerical、finite、exact `(4,4)` matrices；`NaN`、`Inf`、object/string
matrix 会在 projector 或 tree normalization 前 fail closed。该 gate 只证明
数值输入域，不赋予 custom projector 一个标准 continuum scheme identity。

## 2. 如何用自然语言调用

> 使用 `$pyquda_ri_renorm`，验证我的 nonexceptional momenta 是否满足 $p_{\rm in}^2=p_{\rm out}^2=q^2=\mu^2$，并分别生成 `gamma_mu` 与 `qslash` projectors。momentum definition 是 `kinetic_sin_ap`。

> 用 `$pyquda_ri_renorm` 审查 bilinear amputation。`S_out` 和 `S_in` 是独立测得的 propagators；请检查 `(sink_spin,source_spin,sink_color,source_color)` 到 $12\times12$ 的映射。

> 使用 `$pyquda_ri_renorm` 计算 raw `Zq,ZA,ZV,ZS,ZP,ZT`。显式采用 `zq_definition='incoming'`，不要替我做 $\overline{\mathrm{MS}}$ conversion 或 continuum fit。

不应只说“做 RI/SMOM”而不给 projector family、gauge、momentum definition、gamma basis 和 incoming/outgoing sign convention。

## 3. Kinematics 与 scheme label

定义

$$
q=p_{\rm out}-p_{\rm in}.
$$

`validate_kinematics` 要求

$$
p_{\rm in}^2=p_{\rm out}^2=\mu^2>0,
\qquad
q^2=\omega\mu^2,
\qquad
\omega\ge0.
$$

- $\omega=0$：exceptional propagator-$Z_q$ setup，label 为 `RIprime/MOM`，并要求 `projector_scheme='gamma_mu'`。
- $\omega=1$：symmetric nonexceptional point，kinematic label 为 `symmetric_nonexceptional_gamma_mu` 或 `symmetric_nonexceptional_qslash`。
- 其他 $\omega>0$：label 为 `generalized_nonexceptional_*`。

这些只是 kinematics/projector-family labels。`compute_ri_constants` 会进一步返回例如

```text
symmetric_nonexceptional_gamma_mu__Zq_propagator_incoming__custom_unmatched
```

的 `scheme_identity`，并设置 `continuum_matching_status='unmapped_fail_closed'`。在实现并逐式核对 vector-vertex $Z_q$ 或 exact continuum scheme map 前，不得自动选择标准 RI/SMOM matching factor。

`momentum_definition` 是 mandatory metadata，例如：

$$
(ap)_\mu=\frac{2\pi n_\mu}{L_\mu},
\qquad
\hat p_\mu=\sin(ap_\mu),
\qquad
\tilde p_\mu=2\sin\frac{ap_\mu}{2}.
$$

本 helper 不把 integer mode 转换为这些 momenta；调用者必须选定并在 propagator、kinematics 和 projectors 中一致使用。

正常入口应使用 `validate_kinematics` 创建 immutable record。即使调用者手工构造 `RIKinematics`，`compute_ri_constants` 仍会从 `p_in/p_out/omega/projector_scheme/momentum_definition` 重建 kinematics，并在任何 field/projector calculation 前重新检查 stored `q`、`mu2` 与 `scheme`。complex/Boolean $q$ 或 complex/Boolean/nonfinite $\mu^2$ 不会经 `dtype=float` 静默窄化。

## 4. Spin-color layout 与 amputation

输入末四轴固定为

```text
(sink_spin,source_spin,sink_color,source_color)=(4,4,3,3)
```

映射到 $12\times12$ matrix 时，row/column compound indices 为

$$
r=(s_{\rm sink},c_{\rm sink}),
\qquad
c=(s_{\rm source},c_{\rm source}).
$$

`inverse_propagator` 对每组前导 batch axes 执行

$$
S^{-1}(p)=\operatorname{inv}_{12\times12}S(p).
$$

对一般 nonexceptional bilinear，`amputate_vertex` 使用独立 external legs：

$$
\Lambda_O(p_{\rm out},p_{\rm in})
=S^{-1}(p_{\rm out})
G_O(p_{\rm out},p_{\rm in})
S^{-1}(p_{\rm in}).
$$

不能只靠 $\gamma_5$ Hermiticity 从一条 propagator 伪造一般的第二条 external leg；source construction、momentum signs 和 Green-function normalization 必须来自 production workflow。

## 5. Quark-field projector

给定 Euclidean gamma matrices，

$$
\slashed p=\sum_{\mu=0}^{3}p_\mu\gamma_\mu.
$$

`project_quark_field` 实现 propagator-based projection

$$
Z_q^{\rm prop}(p)
=-\frac{i}{12p^2}
\operatorname{Tr}_{s,c}
\left[(\slashed p\otimes I_c)S^{-1}(p)\right].
$$

`compute_ri_constants` 可对 incoming/outgoing estimates 选择：

$$
Z_q^{\rm in},\qquad Z_q^{\rm out},
$$

$$
Z_q^{\rm arith}=\frac{Z_q^{\rm in}+Z_q^{\rm out}}{2},
$$

$$
Z_q^{\rm geom}=\sqrt{Z_q^{\rm in}Z_q^{\rm out}}.
$$

geometric mean 使用 NumPy/CuPy principal square-root branch。复杂统计 fluctuation 可能跨越 branch cut；函数不会静默取 `real`、`abs` 或改 branch。legacy `'propagator'` alias 等于 arithmetic mean；新代码应使用具名选项。

这里没有实现 vector-vertex Ward-identity $Z_q$。propagator-based $Z_q$ 与 vector-vertex $Z_q$ 是不同定义，不得混称。

## 6. Vertex projectors 与 tree normalization

### 6.1 通用单算符 projector

给定 projector $P_O$ 和 tree vertex $\Lambda_O^{(0)}$，代码计算

$$
\lambda_O
=\frac{\operatorname{Tr}\left[P_O^\dagger\Lambda_O\right]}
{\operatorname{Tr}\left[P_O^\dagger\Lambda_O^{(0)}\right]}.
$$

color identity 自动以 Kronecker product 加入。`project_multiplet` 对 aligned components 先分别求 numerator/denominator，再做统一比值，而不是简单平均已归一化的 components。

为使 zero-denominator gate fail closed，CuPy 路径会显式把 `isclose` 的一个 Boolean scalar 同步并传回 host；这是一次有记录的 scalar validation boundary，不是把 vertex 或 volume array 静默搬回 CPU。当前本地只用 fake backend 检查了这一 dispatch contract，真实 CuPy synchronization cost 尚未测量。

### 6.2 $\gamma_\mu$ 与 $q\!\!/$ families

vector/axial tree vertices 为

$$
\Lambda_{V,\mu}^{(0)}=\gamma_\mu,
\qquad
\Lambda_{A,\mu}^{(0)}=\gamma_\mu\gamma_5.
$$

`gamma_mu` family 直接以这组 tree matrices 作为 projectors。

`qslash` family 的 exact code kernels 为

$$
P_{V,\mu}=q_\mu\slashed q,
\qquad
P_{A,\mu}=q_\mu\slashed q\gamma_5.
$$

总体 normalization 由 tree-level denominator 自动确定。continuum matching 前仍须核对目标论文对 $\gamma_5$ order、Euclidean factors 和 Ward identity 的定义。

scalar、pseudoscalar 和 tensor tree kernels 为

$$
\Lambda_S^{(0)}=I,
\qquad
\Lambda_P^{(0)}=\gamma_5,
$$

$$
\sigma_{\mu\nu}
=\frac12[\gamma_\mu,\gamma_\nu],
\qquad \mu<\nu,
$$

其中 tensor multiplet 必须包含六个 components，顺序与调用者的 vertex list 一致。

## 7. Raw renormalization factors

`vertices` mapping 必须恰含：

```text
S: one vertex
P: one vertex
V: four vertices
A: four vertices
T: six vertices ordered by mu < nu
```

投影后代码返回

$$
Z_O=\frac{Z_q}{\lambda_O},
\qquad
O\in\{A,V,S,P,T\}.
$$

这些是所选 lattice/projector convention 下的 raw tree-normalized factors。要得到最终 $Z_O^{\overline{\rm MS}}(\mu_0)$，还需要 scheme-specific continuum conversion、running、window/cut、hypercubic artifact treatment 和 continuum extrapolation；这些不在本 skill 内。

## 8. API 速查

| API | 作用 | 关键约定 |
|---|---|---|
| `RIKinematics` | validated kinematics record | 不是完整 continuum scheme |
| `RIConstants` | raw factors 与 provenance record | custom/unmatched、fail closed |
| `validate_kinematics` | 检查 invariants 并记录 kinematic family | 非完整 scheme identity |
| `inverse_propagator` | batched $12\times12$ inverse | 末四轴固定 |
| `amputate_vertex` | $S_{out}^{-1}GS_{in}^{-1}$ | 两条独立 legs |
| `slash` | 构造 supplied four-vector 的 $\slashed p$ | 不转换 integer modes |
| `project_quark_field` | propagator $Z_q$ projector | 非 vector-vertex definition |
| `project_vertex` | 单 vertex tree normalization | same backend |
| `project_multiplet` | V/A/T aligned component projection | denominator 联合归一化 |
| `compute_ri_constants` | 返回 raw factors 与 structured provenance | custom/unmatched、fail closed |

最小 kinematics/amputation 骨架为：

```python
from skills.pyquda_ri_renorm.scripts.Def_ri_renorm import (
    amputate_vertex,
    compute_ri_constants,
    validate_kinematics,
)

kinematics = validate_kinematics(
    p_in, p_out, omega=1.0,
    projector_scheme="qslash",
    momentum_definition="kinetic_sin_ap",
)
amputated = amputate_vertex(S_out, green_function, S_in)
raw = compute_ri_constants(
    kinematics, S_in_inverse, S_out_inverse,
    vertices, gammas, gamma5, zq_definition="incoming",
)
assert raw.continuum_matching_status == "unmapped_fail_closed"
```

`vertices` 必须使用由 independent legs 一致构造的完整 `S/P/V/A/T` mapping；示例不代表 gauge fixing、solve 或 continuum matching 已完成。

## 9. 推荐 production workflow

1. 固定 gauge-fixing functional、tolerance、Gribov-copy treatment 和 gauge revision。
2. 记录 source type、boundary condition、gamma basis、momentum modes 与 physical/lattice momentum definition。
3. 独立生成 $S(p_{\rm in})$、$S(p_{\rm out})$ 和 $G_O$；保存 solver true residual、iterations 和 failed solves。
4. 用 `validate_kinematics` 建立可序列化 `RIKinematics` record。
5. amputation 后在 tree/free-field oracle 上验证所有 channels 为 $1$。
6. 对 configurations 做保留相关性的 jackknife/bootstrap，再进行 orbit average、window/cut 和 artifact fit。
7. 仅在 exact scheme/equation 已映射后调用 continuum conversion/running。

## 10. 常见错误检查表

- [ ] $q$ 是否定义为 $p_{out}-p_{in}$？
- [ ] 是否手工构造过 `RIKinematics`，且 stored $q/\mu^2/scheme$ 已通过 defensive revalidation？
- [ ] integer modes、$ap$、$\sin(ap)$ 是否混用？
- [ ] 是否把 $\omega=1$ 和 projector family 错当成完整标准 RI/SMOM scheme？
- [ ] 下游是否保留 `scheme_identity` 和 `unmapped_fail_closed` status？
- [ ] source/sink spin-color axes 是否映射成正确的 $12\times12$ indices？
- [ ] 是否错误复用一条 propagator 作为一般 nonexceptional second leg？
- [ ] `qslash` axial projector 的 $\gamma_5$ order 是否与目标 scheme 一致？
- [ ] tensor 六 components 的顺序是否一致？
- [ ] propagator $Z_q$ 是否被误称为 vector-vertex $Z_q$？
- [ ] complex geometric mean 是否被静默实数化？
- [ ] raw RI factors 是否被误称为最终 $\overline{\rm MS}$ results？

## 11. 已验证与未验证

**本地直接验证：**

- exceptional、symmetric 和 generalized invariant checks；
- `gamma_mu`/`qslash` tree-level normalization；
- 所有 $Z$ channels 的 tree-level value $1$；
- incoming/outgoing/arithmetic/geometric $Z_q$ choices，包括 complex principal branch；
- NumPy shape/backend validation 与静态编译。
- 手工构造 record 的 complex $q/\mu^2$ 在 projector/field computation 前 fail closed。

**仍未验证：**

- interacting gauge-fixed ensemble；
- real CuPy/GPU、MPI 和 momentum-source solves；
- gauge-fixing systematics、hypercubic artifacts、fit windows 和 covariance；
- exact continuum matching/evolution 到 $\overline{\rm MS}$。

## 12. 参考文献

1. C. Sturm et al., “Renormalization of quark bilinear operators in a momentum-subtraction scheme with a nonexceptional subtraction point,” [arXiv:0901.2599](https://arxiv.org/abs/0901.2599), *Phys. Rev. D* **80**, 014501 (2009), DOI: 10.1103/PhysRevD.80.014501. RI/SMOM 的 primary scheme reference。
2. Zhi-Cheng Hu, Bo-Lun Hu, Ji-Hao Wang, Ming Gong, Liuming Liu, Peng Sun, Wei Sun, Wei Wang, Yi-Bo Yang, and Dian-Jun Zhao, “Quark masses and low energy constants in the continuum from the tadpole improved clover ensembles,” [arXiv:2310.00814v2](https://arxiv.org/abs/2310.00814v2), Appendix B.1 “Vector normalization and $Z_q$,” PDF pp. 13–14；论文内部将其交叉引用为 Sec. IV.B.1。Eq. (32) 直接支持 propagator-$Z_q$；Eq. (34) 的同一个编号同时覆盖 independent-leg amputation 与 $G_O$ 定义；Eqs. (35)–(42) 给出该论文采用的 vertex/projector factors。它不 blanket validate 本 skill 的 custom `qslash` family、axial ordering 或 symmetric-leg averaging。
3. 本项目 `scripts/Def_ri_renorm.py` 是 shape、compound-index mapping、$Z_q$ choices 和 exact projector algebra 的直接依据。

## 逐函数物理动机与证据卡

下表严格区分三类证据：paper-backed amputation/propagator-$Z_q$、implementation-defined external-leg combinations，以及 custom/unmatched projector family。每一行都是 public API 的独立 contract；相同 kinematics 不意味着相同 continuum scheme。

| Public symbol | 物理动机与公式 | 输入输出、轴与适用域 | 证据状态与 oracle |
|---|---|---|---|
| `RIKinematics` | 保存 $q=p_{out}-p_{in}$、$p_{in}^2=p_{out}^2=\mu^2$ 与 $q^2=\omega\mu^2$，使 momentum definition 不在调用链中丢失。 | immutable host record；四向量为 finite real length-4，`scheme` 只是 kinematic/projector-family label；manual records在使用时再验证。 | 容器是 implementation provenance contract；complex stored $q/\mu^2$ fail-closed regression已闭合，record 本身不是标准 scheme 证明。 |
| `RIConstants` | 将 $Z_q$、$Z_O=Z_q/\lambda_O$、projected vertices 和 matching status绑定为一个结果，防止 raw factors 被误称为 $\overline{\mathrm{MS}}$。 | immutable record；fields 保留 backend scalars/batches、canonical `zq_definition` 与 `unmapped_fail_closed` status。 | 容器为实现约定；tree-level value 与 provenance fields 有回归，interacting renormalization和 continuum matching 未验证。 |
| `validate_kinematics` | 检查 $p_{in}^2=p_{out}^2>0$ 及 $q^2=\omega\mu^2$，区分 exceptional、symmetric 与 generalized nonexceptional points。 | two finite real four-vectors在 dtype inference前拒绝 list/tuple、`UserList`、custom non-string `Sequence` 与 object-array Boolean entries；另需 $\omega\ge0$ 与 mandatory momentum definition。 | invariant/generic-Sequence Boolean checks有回归；arXiv:0901.2599支持 symmetric method class，custom labels/tolerance是 implementation contract。 |
| `inverse_propagator` | 将 spin-color field 映射为 compound matrix $S_{(s,c),(s',c')}$ 并计算 batched $S^{-1}$，为 external-leg amputation服务。 | 输入/输出 exact `(...,4,4,3,3)`、同 NumPy/CuPy backend；前导 batch axes保持不变，singular blocks由 backend inverse报错。 | arXiv:2310.00814v2 的 amputation setup需要 $S^{-1}$；独立手工 12×12 compound-index oracle支持数组映射，真实 CuPy未验证。 |
| `amputate_vertex` | 实现 $\Lambda_O(p_{out},p_{in})=S^{-1}(p_{out})G_O(p_{out},p_{in})S^{-1}(p_{in})$，保留两条独立 external legs。 | 三个输入必须 exact equal `(...,4,4,3,3)` shapes 与同 backend；默认禁止 batch broadcasting，返回同 shape。 | 论文直接：arXiv:2310.00814v2 Appendix B.1, Eq. (34)；独立 12×12 oracle支持实现次序。 |
| `slash` | 构造 $\slashed p=\sum_{\mu=0}^3p_\mu\gamma_\mu$，供 propagator和 longitudinal projectors使用。 | momentum为 finite real length-4，generic non-string Sequence/object-array mixed Boolean在 conversion前拒绝；四个 gamma同 backend且 exact `(4,4)`。 | Clifford contraction定义直接；shape/nonfinite/generic-Sequence Boolean gates有回归，函数不证明 gamma basis 与 production propagator一致。 |
| `project_quark_field` | 计算 $Z_q^{prop}=-i\operatorname{Tr}_{s,c}[(\slashed p\otimes I_c)S^{-1}(p)]/(12p^2)$。 | inverse propagator 为 `(...,4,4,3,3)`，$p^2>0$，gammas exact `(4,4)`且同 backend；返回 batch scalar array。 | 论文直接：arXiv:2310.00814v2 Eq. (32)；tree-level oracle支持 normalization，但 chiral limit和 gauge-fixed ensemble未运行。 |
| `project_vertex` | 以 $\lambda_O=\operatorname{Tr}(P_O^\dagger\Lambda_O)/\operatorname{Tr}(P_O^\dagger\Lambda_O^{(0)})$ 做 single-channel tree normalization。 | vertex 末四轴 `(4,4,3,3)`；projector/tree exact `(4,4)`且同 backend；zero denominator fail closed，CuPy仅同步一个 Boolean scalar。 | generic normalization为 implementation algebra；论文 Eqs. (35)–(42) 只支持其指定 projector，不 blanket validate任意 supplied matrix。 |
| `project_multiplet` | 对 V/A/T aligned components先求总 numerator与 denominator，再形成 $\lambda_{multi}=\sum_iN_i/\sum_iD_i$，避免错误平均已归一化 components。 | 三个 sequences 等长且非空；每个 vertex shape与 projector/tree `(4,4)` contract一致，返回 backend batch scalar。 | implementation-defined combination有 tree-level oracle；component order与 custom multiplet normalization不是 arXiv:2310.00814v2 的 blanket result。 |
| `compute_ri_constants` | 组合 $Z_O=Z_q/\lambda_O$，并显式选择 incoming、outgoing、arithmetic 或 principal-branch geometric $Z_q$。 | 先重建并核对 record 的 $q/\mu^2/scheme$，再要求 propagators与全部 `S/P/V[4]/A[4]/T[6]` vertices exact equal shapes/backend；输出 matching status fail closed。 | Eq. (32)与 amputation部分 paper-backed；manual-record adversarial gate与 tree oracle是实现证据，不等于 interacting production NPR。 |
