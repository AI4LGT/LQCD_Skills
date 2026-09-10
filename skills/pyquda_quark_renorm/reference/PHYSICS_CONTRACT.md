# Quark conversion/running physics contract

## Status and evidence ceiling

- TODO: `SPEC-QRK-01`
- Status: `PARTIAL_BLOCKED_BY_PROVENANCE`
- Highest supported evidence: E1 for spin-color array algebra and selected
  source-mapped series; E0 literal contract for the remaining perturbative
  tables.
- Invocation policy: explicit-only until every production coefficient and
  scheme identity is closed.

The module combines independent array kernels, perturbative tables and legacy
fits. A result from one layer must not promote the others. In particular,
passing a NumPy test does not validate a conversion coefficient, and matching a
published coefficient does not validate gauge-fixed interacting NPR data.

The coefficient-by-coefficient source pass is recorded in
[`SOURCE_VERIFICATION.md`](SOURCE_VERIFICATION.md). It is authoritative for
which rows are source-mapped, which are candidate literature searches, and the
exact expansion-variable algebra.

## Expansion variable and mapped conversion factors

`strong_coupling_constant` returns successive truncations of

$$
a_s(\mu)=\frac{\alpha_s(\mu)}{\pi}.
$$

Several local expressions divide powers by $4^n$ so that their actual series
variable is $\alpha_s/(4\pi)$. The following subset is mapped directly to
Chetyrkin and Retey, arXiv:hep-ph/9910332, Sec. 4.1, SU(3), Landau gauge:

| Public function | Primary equation | Local direction |
|---|---|---|
| `quark_field_conversion_ms_bar_over_rimom` | Eq. (34), $C_2^{RI}$ | candidate $Z_q^{\overline{MS}}/Z_q^{RI}$ |
| `quark_field_conversion_ms_bar_over_rimom_prime` | Eq. (36), $C_2^{RI'}$ | candidate $Z_q^{\overline{MS}}/Z_q^{RI'}$ |
| `quark_mass_conversion_ms_bar_over_rimom_prime` | Eq. (37), $C_m^{RI'}$ | $m^{\overline{MS}}/m^{RI'}$ |

For example, the last row is

$$
C_m^{RI'}=1-\frac{16}{3}\frac{\alpha_s}{4\pi}
+c_2\left(\frac{\alpha_s}{4\pi}\right)^2
+c_3\left(\frac{\alpha_s}{4\pi}\right)^3,
$$

with $c_2,c_3$ exactly transcribed from Eq. (37). The paper's $N_f=4$,
$\alpha_s/\pi=0.1$ reference decomposition in Eq. (41) is the required
independent numeric checkpoint. This partial map does not validate neighboring
functions by similarity.

arXiv:0901.2599 defines RI/SMOM and RI/SMOM-$\gamma_\mu$ method families, but
the local decimal SMOM mass/tensor factors have not yet been mapped
coefficient-by-coefficient to exact equations and conventions. They remain
`UNVERIFIED_LEGACY`.

## Scalar, tensor and Padé boundaries

The nonsinglet scalar relation

$$
Z_S=Z_m^{-1}
$$

is used by the scalar wrappers. It applies only when mass and scalar factors
refer to the same scheme, gauge, flavor content and truncation convention.
Reciprocating a truncated polynomial is a nonlinear operation: the current
code takes the numerical reciprocal of each returned truncation; it does not
re-expand the inverse to a declared order. This implementation convention must
be recorded when comparing references.

The source-mapped Landau-gauge `tensor_conversion_ms_bar_over_rimom_prime`
uses the fixed-order inverse of Gracey, arXiv:hep-ph/0304113v1, Eq. (4.11),
PDF p. 23. With the module variable $a_s=\alpha_s/\pi$ and paper variable
$a=\alpha_s/(4\pi)$, it contains

$$
C_T=1-c_2\frac{a_s^2}{16}-c_3\frac{a_s^3}{64}+O(a_s^4).
$$

The source function is $Z_T^{RI'}/Z_T^{\overline{MS}}$; this API returns its
inverse through the displayed order, so the signs are reversed and the third
order denominator is $4^3=64$. The former `/6` was neither the paper's
expansion nor a variable-conversion factor. The source, algebraic reduction and
fixed-coupling regression appear in [`SOURCE_VERIFICATION.md`](SOURCE_VERIFICATION.md).

The scalar/tensor Padé functions contain copied decimal rational functions.
Their Padé order, construction equations, singular domains and coefficient
provenance are unmapped. `pade_matching_factor` returns only three scalar
channels; its `scale0` argument is a validated compatibility no-op and must not
be interpreted as RG evolution.

## Running and composition

The generic implementation constructs $c(a_s)$ from an anomalous-dimension
series and returns

$$
U(\mu,\mu_0)=\frac{c(a_s(\mu_0))}{c(a_s(\mu))}.
$$

For a completely identified operator, the following are required checks:

$$
U(\mu,\mu)=1,
\qquad
U(\mu_2,\mu_0)=U(\mu_2,\mu_1)U(\mu_1,\mu_0)
+O(a_s^{n+1}),
$$

where $n$ is the declared truncation order. Exact composition must not be
demanded of independently truncated series. The five-loop beta-function
context is supported by arXiv:1606.08659. The MS mass table and generic
$c(a_s)$ construction are additionally mapped to BCK2014
(arXiv:1402.6611, Eqs. (3.1)--(3.4), (4.7)--(4.12)); the returned code
orientation is $c(a_s(\mu_0))/c(a_s(\mu))$. Field and tensor anomalous-
dimension arrays still require coefficient-level source maps.

The source pass now isolates two narrower results without promoting either
public running factor. CR2000 `hep-ph/9910332v2`, p. 4 Eq. (7) and p. 16
Eqs. (50)--(52), exactly supplies the local Landau-gauge MS quark-field
coefficient subset `r0..r3` before this implementation doubles it. Gracey
`hep-ph/0304113v1`, p. 21 Eq. (4.7), exactly supplies the MS tensor-current
subset `r0..r2` after the explicit $a=\alpha_s/(4\pi)=a_s/4$ conversion.
They are `PRIMARY_COEFFICIENT_SUBSET_NONPROMOTING`, recorded in
`SOURCE_VERIFICATION.md`, rather than physical profiles: the code has not
identified whether its field output denotes $\psi$, $Z_2$, lattice $Z_q$, or an
inverse; it doubles the field vector and appends a source-free zero. Tensor
`r3` and its final zero are likewise not source-mapped, and the generic ratio
$c(a_s(\mu_0))/c(a_s(\mu))$ has no operator-object direction. Thus field and
tensor *full outputs* remain explicit-only; no entry is added to
`QUARK_RENORM_PROFILE_REGISTRY`.

Matching and running must be staged and labeled:

```text
raw gauge-fixed RI factor
  -> chiral/orbit/a2p2 treatment with covariance
  -> exactly identified conversion at mu
  -> exactly identified running from mu to target scale
```

Do not reorder these steps or combine statistical, fit, scale, coupling and
perturbative-truncation uncertainties into one unlabeled error.

## Spin-color and fit contracts

Array inputs end in
`(sink_spin,source_spin,sink_color,source_color)=(4,4,3,3)` and map to a
compound $(s,c)$ $12\times12$ matrix. `adj` applies
$\gamma_5S^\dagger\gamma_5$ and swaps both spin and color source/sink axes.
These kernels have local E1 evidence but are not substitutes for the
independent-leg amputation in `pyquda_ri_renorm`.

Legacy `ma_fit`, `a2p2_fit`, `ratio_fit` and `read_data` are not covariance-
complete production analyses by default. A result record must state sample
axis, jackknife/bootstrap definition, covariance matrix, priors, momentum
definition, fit window and every systematic flag.

## Promotion gate

Implicit invocation may be reconsidered only after every public conversion,
running and Padé coefficient has a paper version/page/equation, exact scheme
and gauge, independent reference point, and reviewer approval. Until then,
unmapped functions must return only literal/audit results with an explicit
`UNVERIFIED_LEGACY` label.

## Sources

- arXiv:hep-ph/9910332, especially Eqs. (34), (36), (37), and reference
  values (38)-(41).
- arXiv:0901.2599: RI/SMOM method and scheme definitions; no blanket decimal
  validation.
- arXiv:1606.08659: five-loop beta-function context.
- arXiv:1402.6611: MS mass anomalous dimensions and $c(a_s)$ algebra.
- arXiv:hep-ph/0304113: RI-prime tensor conversion Eq. (4.11).
- arXiv:hep-ph/9910332v2, p. 4 Eq. (7), p. 16 Eqs. (50)--(52):
  non-promoting MS quark-field anomalous-dimension coefficient subset.
- arXiv:hep-ph/0304113v1, p. 21 Eq. (4.7): non-promoting MS tensor-current
  anomalous-dimension coefficient subset.
- `scripts/Def_quark_renorm.py`: checksum-frozen local implementation.
- [`SOURCE_VERIFICATION.md`](SOURCE_VERIFICATION.md): primary-source ledger,
  algebraic reductions, and the fail-closed legacy register.
