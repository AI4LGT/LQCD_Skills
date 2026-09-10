# Bare gluon-observable physics contract

## Status and evidence ceiling

- TODO: `SPEC-GLU-01`
- Status: `PASS_BARE_CONTRACT_WITH_RENORMALIZATION_BLOCKERS`
- Highest supported evidence: local E1 array algebra for selected functions.
- Runtime/observable ceiling: E0 in the current environment.

The historical skill name contains `renorm`, but the bundled functions end at
bare lattice fields or gauge-fixed candidates. Nothing in this module closes
operator mixing, continuum matching or an $\overline{\mathrm{MS}}$ scheme.

## Axes, epsilon and lattice units

Directions are fixed as

```text
0=x, 1=y, 2=z, 3=t; epsilon[x,y,z,t] = epsilon[0,1,2,3] = +1.
```

Gauge storage is
`(direction,e,t,z,y,xh,sink_color,source_color)` with $e=2$ and $N_c=3$.
The trace is the unnormalized fundamental-color trace unless a function states
otherwise.

Links are dimensionless, $U_\mu(x)\sim\exp[iag_0A_\mu(x+a\hat\mu/2)]$. Since
the code inserts no explicit powers of $a$,

$$
A_\mu^{\rm code}\sim aA_\mu^{\rm continuum},\qquad
F_{\mu\nu}^{\rm code}\sim a^2F_{\mu\nu}^{\rm continuum}.
$$

Consequently, bilinears such as `FF`, `FW_FW` and `FW_tildeFW` carry an
implicit $a^4$ before sums/averages. `ExA_t` and Chern-Simons candidates carry
the corresponding powers inherited from $A^{\rm code}$ and $F^{\rm code}$.
A physical-dimension conversion requires a separately reviewed $a$ factor,
volume normalization and operator normalization.

`g_0` is determined by the gauge action, never by the fermion action:

$$
g_0=\sqrt{6/\beta}\quad\text{for `wilson`},
\qquad
g_0=\sqrt{10/(\beta u_0^4)}\quad\text{for `legacy_tadpole_10`}.
$$

No other action is inferred from a positional $u_0$.

## Operator map

### Gauge potential

For half-link storage,

$$
A_\mu^{\rm code}(x+\tfrac12\hat\mu)=
\left[\frac{U_\mu-U_\mu^\dagger}{2ig_0}\right]_{\rm traceless}.
$$

The site-centered branch averages forward and backward anti-Hermitian link
parts. `A_mu` is gauge dependent. Physical interpretation requires the exact
gauge-fixing functional, stopping residual, boundary/residual-gauge treatment
and a policy for Gribov copies.

### Clover field strength and dual

`F_munu` uses four oriented $1\times1$ loops and the anti-Hermitian clover
part. The full tensor obeys

$$
F_{\mu\nu}=-F_{\nu\mu},\qquad
\widetilde F_{\alpha\beta}=\frac12
\epsilon_{\alpha\beta\mu\nu}F_{\mu\nu}.
$$

The compact magnetic tuple is

$$
(F_{xy},F_{xz},F_{yz})=(B_z,-B_y,B_x),
$$

not $(B_x,B_y,B_z)$. arXiv:hep-lat/0203008 supports the clover/improved
field-strength method class and illustrates the required $a,g_0$ accounting;
it does not validate every normalization in this legacy implementation.

### Wilson-line bilinears

The code-defined nonlocal kernels are

$$
-\frac12\operatorname{Tr}
[F(x)W(x,x+n\hat d)F(x+n\hat d)W^\dagger],
$$

and the corresponding $F\widetilde F$ expression. Gauge covariance of the
endpoints makes the closed color trace gauge invariant for the stated path,
but this does not supply the continuum operator's mixing, power-divergence,
line self-energy or matching prescription. Link direction, signed orientation,
length and endpoint convention belong in result provenance.

### EMT candidates

`gaugeEMT_munu` implements the literal Euclidean array combination

$$
T_{\mu\nu}^{\rm code}=
2\sum_{x,\rho}\operatorname{Tr}[F_{\mu\rho}F_{\nu\rho}]
-\delta_{\mu\nu}\frac12
\sum_{x,\alpha,\beta}\operatorname{Tr}[F_{\alpha\beta}F_{\alpha\beta}].
$$

The `gaugeEMT_mumu*` alternatives are separate discretizations, not proven
interchangeable renormalized EMT definitions. They remain `bare_emt_candidate`
until normalization, hypercubic mixing, quark/gluon mixing and continuum
conversion are supplied.

### Gauge-dependent spin/current candidates

`ExA_t` computes the literal spatial sum

$$
(\mathbf E\times\mathbf A)_i^{\rm code}
=\sum_{\mathbf x}\epsilon_{ijk}
\operatorname{Tr}[E_jA_k].
$$

The `Topological_current_Kmu*` routines assemble a code-defined Euclidean
version of

$$
K_\mu\sim\epsilon_{\mu\nu\rho\sigma}\operatorname{Tr}
\left(A_\nu F_{\rho\sigma}-\frac{2ig_0}{3}A_\nu A_\rho A_\sigma\right).
$$

arXiv:1310.4263 directly supports the gauge-dependence and large-momentum
matching context; it also warns that nonperturbative gauge dependence remains.
Without a declared gauge and exact Euclidean continuation, these arrays are
diagnostic candidates, not unique gluon spin or helicity observables.

## MPI and ownership

Compact quantities may be reduced after all local spatial axes are contracted.
Time-resolved functions require `Gt=1` because a world reduction over
time-decomposed ranks would mix different global times. Site-resolved fields
remain local unless explicitly assembled. A single-rank or fake-communicator
test is not multi-rank evidence.

## Promotion gate

A bare candidate may be promoted only when a record provides: gauge action and
$g_0$ convention; lattice spacing and volume normalization; Euclidean-to-
Minkowski map; trace normalization; gauge prescription where needed;
discretization; operator mixing basis; renormalization/matching scheme and
scale; and interacting gauge/transformation controls. Until then use names
`bare_*`, `gauge_fixed_candidate_*` or `diagnostic_*`.

## Sources

- arXiv:hep-lat/0203008: clover/improved $F_{\mu\nu}$ method class.
- arXiv:1310.4263: gauge-dependent gluon-spin and Chern-Simons context.
- `scripts/Def_gluon_renorm.py`: literal implementation and axis ordering.

