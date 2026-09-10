# Helicity/EMT source-verification ledger

Status date: 2026-08-26: the built-in helicity/EMT API remains
`UNVERIFIED_LEGACY`/explicit-only as a whole. A newly located primary paper
maps several intended terms but contradicts the returned off-diagonal matching
matrix. This is a fail-closed evidence record, not a reconstruction of a
physical operator basis from numerical similarities.

## Directly checked beta-function source

P. A. Baikov, K. G. Chetyrkin and J. H. Kühn, *Five-Loop Running of the QCD
coupling constant*, Phys. Rev. Lett. 118 (2017) 082002,
[arXiv:1606.08659v2](https://arxiv.org/abs/1606.08659),
[DOI:10.1103/PhysRevLett.118.082002](https://doi.org/10.1103/PhysRevLett.118.082002),
PDF p. 1, Eqs. (1)--(5), defines

$$
a_s=\frac{\alpha_s}{\pi},\qquad
\frac{d a_s}{d\ln\mu^2}=-\sum_{i=0}^4\beta_i a_s^{i+2}.
$$

It directly supports only `beta` and `alpha_s`: their SU(3) coefficients are
the printed five-loop beta coefficients, with the asymptotic helper storing
`beta_i/beta_0` for `i>0`. It does **not** identify `EMT_MatchingCoeff`, either
`Helicity_MatchingCoeff` branch, the ordered two-operator vector, or a helicity
anomalous-dimension entry.

## Primary helicity candidate and contradiction

D.-J. Zhao *et al.*, *Total Gluon Helicity Contribution to the Proton Spin
from Lattice QCD*, [arXiv:2512.24315v2](https://arxiv.org/abs/2512.24315),
official version updated 2026-01-16, has an exact relevant scope. Supplemental
pp. 8--10, Eqs. (16), (18), and (20)--(25) define the ordered physical vector
$(\Delta G,\Delta\Sigma)^T$ associated with $(K^\mu,J_5^\mu)$, the action
$(\Delta G,\Delta\Sigma)^T_{\overline{\rm MS}}=
R(\Delta G,\Delta\Sigma)^T_{\rm RI}$, RI/MOM source and modified-
$\overline{\rm MS}$ (Larin) target, Landau gauge for the printed finite
matching expressions, $N_c=3$, color-trace normalization $1/2$,
$a_s=\alpha_s/(4\pi)$, and $L_\mu=\ln(\mu^2/\mu_{\rm RI}^2)$.

This is not a full mapping to the local functions. In Eqs. (22)--(23) the
official v2 text prints

$$
R_{12}=1+a_s C_F(3L_\mu+6)+\cdots,\qquad
R_{21}=1-2n_f a_s+\cdots .
$$

`Helicity_MatchingCoeff_tmp` omits both displayed tree-level ones. Its local
equal-scale values are instead $R_{12}=8a+\cdots$ and
$R_{21}=-6a+\cdots$ for $a=\alpha_s/(4\pi)$ and $n_f=N_c=3$. The difference
from the printed source is exactly $-1$ in both entries. No correction to those
source terms was found in the inspected official arXiv v2 record. Hence a
similarity of the $a_s$, $a_s^2$, or $a_s^3$ terms is not a license to delete
the conflict.

Supplemental Eqs. (20)--(21) do term-match the local `tmp` $R_{11}$ and
$R_{22}$ fixed-order pieces. The `Helicity_MatchingCoeff` rational branch does
not: its `r0_11` and $r_{1,11}=49/4$ have no map to Eq. (20). Eq. (25) is a
large-$\beta_0$ observation only; it is not the code's entrywise
$r_4=r_3^2/r_2$ Padé rule.

For the running table, Eq. (24) supplies the left action and relations
$\gamma_{11}=-\beta$, $\gamma_{21}=0$, and
$\gamma_{22}=-2a_sn_f\gamma_{12}$. The $\gamma_{12}$ numerators intended by the
local table are independently consistent with L. Chen and M. Czakon,
[arXiv:2201.01797v3](https://arxiv.org/abs/2201.01797), Eq. (10), after the
factor-$-4$ $K^\mu$ normalization noted below Eq. (24) of arXiv:2512.24315v2
and conversion to $\alpha_s/\pi$: $4$, $76$, and
$316.679353\ldots$. This remains non-promoting because the local API never
declares that basis/normalization, rounds the latter to `316.679`, and omits
the related quartic $\gamma_{22}$ contribution while retaining the cubic
$\gamma_{12}$ term. Likewise `643.833` rounds the exact $3863/6$ beta
numerator.

## Required helicity provenance map: result of the source pass

| Code object | Required original information | Current evidence and disposition |
|---|---|---|
| `EMT_MatchingCoeff` | named EMT operator, source/target scheme, gauge, continuum/lattice normalization, and finite-term equation | No original equation or source identity is stored. `UNVERIFIED_LEGACY`; return only a literal code replay. |
| `Helicity_MatchingCoeff_tmp` | ordered vector `(O1,O2)`, matrix action (`O_target=R O_source`, transpose or inverse), source/target scheme, gauge, `N_f`, color factors, log convention, and an equation for every `r_n^{ij}` | arXiv:2512.24315v2 supplies all of these for its $K^\mu/J_5^\mu$ problem and term-matches $R_{11}/R_{22}$ plus the non-identity parts of $R_{12}/R_{21}`. Its printed off-diagonal tree terms disagree with the local return values, while the code does not declare this basis. `PRIMARY_SOURCE_CONTRADICTION`. |
| `Helicity_MatchingCoeff` | all preceding fields plus a source for its rational `R11` term | The rational `R11` contribution has no named resummation or primary equation; its fixed $r_{1,11}=49/4$ is not arXiv:2512.24315v2 Eq. (20). `UNVERIFIED_LEGACY`. |
| entrywise Padé `r4=(r3**2)/r2` | Padé order/construction, intended coefficients, scale domain and truncation uncertainty | Eq. (25) is not an entrywise Padé prescription. Zero/nearly-zero denominator rejection is a local safety condition, not a four-loop calculation. |
| `Helicity_RunningFactorCalculator._ghelicity_gamma_coefficients` | ordered operator basis, scheme, gauge, direction and an equation for all `gamma11`, `gamma12`, `gamma21`, `gamma22` entries | arXiv:2512.24315v2 Eq. (24) maps the matrix structure, and arXiv:2201.01797v3 Eq. (10) gives an intended coefficient chain, but the local API lacks the basis binding, rounds `643.833`/`316.679`, and truncates the required `gamma22` relation inconsistently at the next order. `PRIMARY_RELATION_PARTIAL_TRUNCATION_CONFLICT`. |
| `Get_Helicity_Running` | preceding gamma map plus proof that a real output is legitimate in that basis | Its API does not bind `mu_from*muscaleOVmuR` and `mu_to` to the source's $(\mu,\mu_{\rm RI})$ roles; it returns `.real` from a real-only numerical path. This cannot establish physical real evolution for an undeclared/inconsistent matrix. |

## Algebra that is local rather than physical evidence

The implementation unambiguously applies

$$
\frac{dR}{d\ln\mu^2}=\Gamma(a_s)R,\qquad
U(\mu_2,\mu_1)=R(\mu_2)R(\mu_1)^{-1}.
$$

Identity at equal scale, left-action orientation and composition are therefore
implementation checks for caller-supplied verified matrices. They do not
validate the built-in `Gamma`, a physical matrix orientation, an operator basis
or a gauge. The primary candidate adds a source-specific ODE structure but does
not repair the contradicting matching matrix.

## Physical-profile API guard (implementation evidence, not physics promotion)

The module now exposes `validate_helicity_physical_profile` and keyword-only
`physical=False`/`physical_profile=None` controls on the two matching entry
points, the matching-array wrapper, the running calculator and
`Get_Helicity_Running`. The old call signatures remain literal API access: an
ordinary call, or `physical=False`, returns the same `UNVERIFIED_LEGACY` tuple
as before. A caller requesting `physical=True` must provide a profile with all
of the following evidence fields:

1. the ordered `("K^mu", "J5^mu")` operator basis;
2. source scheme, target scheme, gauge, matrix action, and matrix direction;
3. explicit `R12`/`R21` tree terms;
4. version, page, and equation locators for `R11`, `R12`, `R21`, `R22`, and
   `Gamma`; and
5. matching and running truncation records marked closed.

The result object is serializable (`.as_dict()`) and has a typed `status`; a
failing `physical=True` request raises `HelicityPhysicalProfileError` carrying
that result. There is intentionally no registered physical calculator profile
in this checkout: complete-looking arbitrary metadata receives
`UNVERIFIED_NO_MAPPED_PHYSICAL_PROFILE`, rather than a physical pass.

`ZHAO_V2_K_J5_SOURCE_PROFILE` is supplied solely as a structured audit record
for arXiv:2512.24315v2. Binding it to either legacy matching function returns
`PRIMARY_SOURCE_CONTRADICTION` with the two tree-level differences: the source
requires $R_{12}^{(0)}=R_{21}^{(0)}=1$, whereas both legacy tuples have zero in
those positions. Its running truncation is also recorded as not closed. The
API guard was covered by local regression tests; this verifies refusal and
backwards-compatible literal access only. It does **not** reconcile the
source, map the rational/Padé branch, complete $\Gamma_{22}$, reproduce a
physical reference point, or establish an interacting observable.

## Promotion gate

This skill remains explicit-only. Before a matching or running number can be
physical, reconcile the primary off-diagonal tree-level contradiction, then
provide a code-to-paper basis binding, an equation for the rational/Padé
branches, exact (not rounded) running coefficients at a consistent truncation,
source/target scheme and scale-role mapping, plus independent reference-point,
equal-scale and composition tests. Until then a call remains a literal audit
result marked `UNVERIFIED_LEGACY` or `PRIMARY_SOURCE_CONTRADICTION`, and it
must not be implicitly routed.
