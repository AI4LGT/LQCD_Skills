# Helicity matching/running physics contract

## Status and evidence ceiling

- TODO: `SPEC-HEL-01`
- Status: `BLOCKED_BY_PROVENANCE`
- Highest supported evidence: E0 for the literal coefficient contract; selected
  matrix placement and ODE algebra have local E1 tests.
- Invocation policy: explicit-only. This contract does not authorize implicit
  use of the bundled numbers.

The local source is reproducible. A relevant primary source now identifies one
$(K^\mu,J_5^\mu)$ RI/MOM-to-modified-$\overline{\rm MS}$ (Larin), Landau-gauge
problem, but its printed $R_{12}$/$R_{21}$ tree terms conflict with the local
return values and the code has no basis binding to that paper. Consequently, no
tuple returned by a matching/running routine is a production helicity or EMT
renormalization factor unless the caller supplies an independently verified
coefficient set and full scheme identity.

The source-verification search and its explicit negative result for the
matching/running tables are recorded in
[`SOURCE_VERIFICATION.md`](SOURCE_VERIFICATION.md). It isolates the mapped
beta-function source from the unmapped matrix coefficients.

## Literal matching contract

`Helicity_MatchingCoeff_tmp` and `Helicity_MatchingCoeff` return

$$
(R_{11},R_{12},R_{21},R_{22})
\longleftrightarrow
R=\begin{pmatrix}R_{11}&R_{12}\\R_{21}&R_{22}\end{pmatrix}.
$$

This fixes array placement only. It does not identify row/column 1 or 2 with a
quark, gluon, helicity or EMT operator. The code fixes $N_f=N_c=3$, uses

$$
a=\frac{\alpha_s(\mu)}{4\pi},\qquad
L=\ln\frac{\mu^2}{\mu_R^2},
$$

and forms a fixed-order series through $a^3$. At equal scales the independently
checked one-loop off-diagonal literal is

$$
R_{12}=8a+O(a^2),\qquad R_{21}=-6a+O(a^2).
$$

This check detects row/column swaps; it does not validate the physical basis.
`Helicity_MatchingCoeff` additionally inserts a code-defined rational
$R_{11}$ contribution. Its relation to a named resummation prescription is
unverified. The nearby primary formula does not supply this branch, and its
off-diagonal entries have an additional printed tree-level one that `tmp` and
this branch omit; neither function may be promoted by term similarity.

For `loop=4,is_pade=1`, every matrix entry uses the local estimate

$$
r_4^{ij}=\frac{(r_3^{ij})^2}{r_2^{ij}}.
$$

It is an entrywise estimate, not a four-loop calculation. A zero or numerically
singular denominator fails closed. `loop=4,is_pade=0` adds no fourth-order
coefficient and is therefore the explicit three-loop result under a legacy
loop label.

`EMT_MatchingCoeff` is likewise a literal one-loop scalar expression. Until its
operator and scheme are mapped to an original equation it remains
`UNVERIFIED_LEGACY`, not a renormalized EMT observable.

## Coupling and running contract

`beta` and `alpha_s` use

$$
a_s=\frac{\alpha_s}{\pi},\qquad
\frac{da_s}{d\ln\mu^2}=-\sum_{n=0}^4\beta_n a_s^{n+2}.
$$

The five-loop beta-function context is supported by arXiv:1606.08659, but that
paper does not establish the helicity matrix coefficients. The default
$\Lambda=0.332\,\mathrm{GeV}$ is allowed only for $N_f=3$, and the asymptotic
helper requires $\mu/\Lambda>2$.

`Helicity_RunningFactorCalculator` fixes $N_f=3$ and integrates

$$
\frac{dR}{d\ln\mu^2}=\Gamma(a_s)R,
$$

so the anomalous-dimension matrix acts from the left. The returned evolution is

$$
U(\mu_2,\mu_1)=R(\mu_2)R(\mu_1)^{-1}.
$$

arXiv:2512.24315v2 Supplemental Eq. (24) maps a nearby matrix structure and
the relations $\gamma_{11}=-\beta$, $\gamma_{21}=0$, and
$\gamma_{22}=-2a_sn_f\gamma_{12}$; arXiv:2201.01797v3 Eq. (10) identifies an
intended coefficient chain. The code nevertheless rounds `643.833` and
`316.679`, omits the next related $\gamma_{22}$ term, and never declares the
paper's $(K^\mu,J_5^\mu)$ basis. `Get_Helicity_Running` also returns
`.real` from a real-only numerical path without binding its scale arguments to
the paper. Production use remains blocked until those discrepancies are
resolved in a source-identified basis.

## Required provenance record

Before returning a physical number, require all fields below:

1. ordered operator vector $O=(O_1,O_2)^T$ and whether matching acts as
   $O^{\rm target}=RO^{\rm source}$ or with the transpose/inverse;
2. source and target schemes, gauge, $N_f$, color factors and operator
   normalization;
3. coupling variable, derivative variable, loop order and logarithm sign;
4. paper version plus page/equation for every entry of $r_n$ and $\Gamma_n$;
5. Padé definition, allowed scale domain and truncation prescription;
6. input covariance and separate coupling/scale/truncation uncertainties.

If any field is absent, the only allowed outputs are a source audit, literal
code replay, or a calculation with caller-supplied verified matrices. Do not
rename the built-in tuple as a physical quark/gluon basis by inference.

## Verification and blockers

Locally verified: input-domain gates, tuple placement, equal-scale one-loop
$R_{12}/R_{21}$, fixed-order versus Padé branching, and the orientation of the
ODE implementation. Not verified: coefficient provenance, physical matrix
orientation, reference-point reproduction, truncation uncertainty, or any
interacting observable. Those blockers must remain visible in every result
record and release label.

## Sources

- arXiv:1606.08659: five-loop QCD beta-function context only.
- arXiv:2512.24315v2 Supplemental pp. 8--10, Eqs. (16), (18), (20)--(25):
  primary candidate whose precise off-diagonal tree-level conflict is recorded
  in `SOURCE_VERIFICATION.md`.
- arXiv:2201.01797v3 Eq. (10): source for the related singlet-axial anomalous
  dimension chain, not by itself a code-basis binding.
- `scripts/Def_helicity_renorm.py`: checksum-frozen literal implementation.
- [`SOURCE_VERIFICATION.md`](SOURCE_VERIFICATION.md): exact beta-source map
  and the fail-closed matching/running provenance register.
- `docs/pyquda_skill_provenance.yaml`: source and blocker ledger.
