# Quark conversion and running source-verification ledger

Status date: 2026-08-28. This ledger separates primary-PDF checks from
candidate searches and local algebra. `MAPPED` means that the cited paper, its
displayed expansion variable, and the source-to-code algebra were checked. It
does not establish a gauge-fixed lattice NPR measurement, continuum
extrapolation, or a complete uncertainty budget.

## Primary records checked

| Key | Exact primary record | Stable locations | Directly checked locator |
|---|---|---|---|
| CR2000 | K. G. Chetyrkin and A. Retey, *Renormalization and Running of Quark Mass and Field in the Regularization Invariant and MS-bar Schemes at Three and Four Loops*, Nucl. Phys. B583 (2000) 3-34 | [arXiv:hep-ph/9910332v2](https://arxiv.org/abs/hep-ph/9910332), [DOI:10.1016/S0550-3213(00)00331-X](https://doi.org/10.1016/S0550-3213(00)00331-X) | PDF p. 11, Eqs. (34), (36), (37); p. 12, Eqs. (38)-(41); Eqs. (9)-(12) fix the conversion-function notation. |
| G2003 | J. A. Gracey, *Three loop anomalous dimension of non-singlet quark currents in the RI' scheme*, Nucl. Phys. B662 (2003) 247-278 | [arXiv:hep-ph/0304113v1](https://arxiv.org/abs/hep-ph/0304113), [DOI:10.1016/S0550-3213(03)00335-3](https://doi.org/10.1016/S0550-3213(03)00335-3) | PDF p. 23, Eq. (4.11), tensor conversion function in MS variables and a general covariant gauge. |
| BCK2014 | P. A. Baikov, K. G. Chetyrkin and J. H. Kuehn, *Quark Mass and Field Anomalous Dimensions to O(alpha_s^5)*, JHEP 10 (2014) 076 | [arXiv:1402.6611v1](https://arxiv.org/abs/1402.6611), [DOI:10.1007/JHEP10(2014)076](https://doi.org/10.1007/JHEP10(2014)076) | PDF pp. 1, 3 and 5, Eqs. (1.1), (3.1)-(3.4), (4.7)-(4.12). |
| BCK2016 | P. A. Baikov, K. G. Chetyrkin and J. H. Kuehn, *Five-Loop Running of the QCD coupling constant*, Phys. Rev. Lett. 118 (2017) 082002 | [arXiv:1606.08659v2](https://arxiv.org/abs/1606.08659), [DOI:10.1103/PhysRevLett.118.082002](https://doi.org/10.1103/PhysRevLett.118.082002) | PDF p. 1, Eqs. (1)-(5), with `a_s=alpha_s/pi`. |
| SJ2009 | C. Sturm, Y. Aoki, N. H. Christ, T. Izubuchi, C. T. C. Sachrajda and A. Soni, *Renormalization of quark bilinear operators in a momentum-subtraction scheme with a nonexceptional subtraction point*, Phys. Rev. D80 (2009) 014501 | [arXiv:0901.2599v2](https://arxiv.org/abs/0901.2599), [DOI:10.1103/PhysRevD.80.014501](https://doi.org/10.1103/PhysRevD.80.014501) | Scheme definitions and nonexceptional kinematics; it is not a blanket source for local decimal tables. |

## Mapped conversion coefficients

All CR2000 formulae here are for SU(3), Landau gauge and use
`a = alpha_s/(4*pi) = a_s/4`, where this module's `a_s` is `alpha_s/pi`.
The code accepts symbolic `nf`; the cited reference point uses `nf=4`.

| Public symbol | Status and physical direction | Primary locator and algebraic recheck |
|---|---|---|
| `quark_field_conversion_ms_bar_over_rimom` | `MAPPED`; the literal polynomial is CR2000 `C_2^RI` in the paper's field-conversion notation. Do not replace the explicit API direction with a `Z_q` convention without also applying Eqs. (9)-(10). | Eq. (34), p. 11. The code evaluates `A2*(a_s**2/16) + A3*(a_s**3/64)`, exactly the paper's `A2*a**2 + A3*a**3`. |
| `quark_field_conversion_ms_bar_over_rimom_prime` | `MAPPED`; the literal polynomial is CR2000 `C_2^{RI'}` under the same notation caveat. | Eq. (36), p. 11. Its absent one-loop term and its `a_s**2/16`, `a_s**3/64` factors agree term by term with the printed series. |
| `quark_mass_conversion_ms_bar_over_rimom_prime` | `MAPPED`; `C_m^{RI'}` in CR2000's `m = C_m m'` convention. | Eq. (37), p. 11. The code is `1-(16/3)*a_s/4 + A2*a_s**2/16 + A3*a_s**3/64`. With `nf=4` and `a_s=0.1`, Eq. (41), p. 12 gives `1.0-0.1333-0.0701-0.0458=0.7508`. |
| `vector_conversion_ms_bar_over_rimom_prime`; `quark_field_conversion_rimom_prime_over_rimom` | `MAPPED_DERIVED`, but the first name is historical and must not be read as a verified vector-current conversion. | Divide Eq. (36) by Eq. (34) through `O(a**3)`. As both series have zero `O(a)`, `B2-A2=-67/6+2*nf/3=(8*nf-134)/12` and `B3-A3=(-52321/18+607*zeta(3)+(8944/27-32*zeta(3))*nf-208*nf**2/27)/4`, exactly the code coefficients multiplying `a_s**2/16` and `a_s**3/64`. The result is algebraically `C_2^{RI'}/C_2^{RI}`; external field or `Z_q` direction follows the caller convention, not the legacy name. |
| `tensor_conversion_ms_bar_over_rimom_prime` | `MAPPED`; source tensor convention is `C_T=Z_T^{RI'}/Z_T^{MS}` in MS variables. The API returns the inverse fixed-order direction through this order. | G2003 Eq. (4.11), p. 23. At `xi=0`, `C_T=1+k2*a**2+k3*a**3+O(a**4)` with `k2=3847/54-313*nf/81-184*zeta(3)/9` and `k3` equal to code `a3`. Thus `(C_T)**(-1)=1-k2*(a_s**2/16)-k3*(a_s**3/64)+O(a_s**4)`. The former `/6` was neither this expansion nor a variable conversion and is corrected to `/64`. |

## Mapped beta and mass-running algebra

`beta_coupling_constant` and `strong_coupling_constant` use BCK2016 Eq. (1),
`beta(a_s)=d a_s/d ln(mu^2)=-sum_i beta_i a_s^(i+2)`, `a_s=alpha_s/pi`.
Its SU(3) Eqs. (2)-(5) reproduce the five code entries before the code stores
`beta_i/beta_0` for `i>0`.

`quark_mass_anomalous_dimension_under_ms_bar` uses the BCK2014 MS mass
coefficients in Eqs. (3.1)-(3.4), with
`gamma_m=-sum_i gamma_i a_s^(i+1)`. The generic helpers implement the same
c-function algebra as BCK2014 Eqs. (4.7)-(4.12): internally
`d2_code=2*d2_paper` and `d3_code=3*d3_paper`, yielding
`d1**2/2+d2_paper` and `d1**3/6+d1*d2_paper+d3_paper` at successive orders.
Their returned orientation is `c(a_s(mu0))/c(a_s(mu))`, the inverse of the
mass ratio as written in Eq. (4.7). Equal-scale identity and composition are
therefore algebraic test targets, not new physical evidence.

`scalar_anomalous_dimension_under_ms_bar` is only the local reciprocal wrapper
around that mass-running output. It is not a new independent coefficient table,
and scalar Ward-identity, flavor, gauge and truncation conditions remain caller
requirements.

## Primary coefficient subsets that must not promote a running output

Two additional coefficient identities have been checked directly against the
official PDFs. They are deliberately recorded outside
`QUARK_RENORM_PROFILE_REGISTRY`: an anomalous-dimension coefficient is not yet
an identified running *object* (for example, a quark field, `Z_2`, lattice
`Z_q`, or an inverse factor). These records therefore have status
`PRIMARY_COEFFICIENT_SUBSET_NONPROMOTING`, not `MAPPED`.

| Local symbol and exact subset | Candidate primary source, version, page and equation | Ordered definition, scheme and direction actually fixed by the source | Why the public running result remains blocked; minimal repair status |
|---|---|---|---|
| `quark_field_anomalous_dimension_under_ms_bar` at `Def_quark_renorm.py:1520-1530`: `r1`, `r2`, `r3` before the local factor of two | CR2000 `hep-ph/9910332v2`, PDF p. 4 Eq. (7) defines the anomalous dimension of the renormalized quark field `psi`; PDF p. 16 Eqs. (50)--(52) give the SU(3), Landau-gauge MS coefficients `gamma_2^(1,2,3)` in powers of `alpha_s/pi`. | The source fixes the MS, Landau-gauge quark-field `psi` anomalous dimension and its sign convention in Eq. (7). With the module variable `a_s=alpha_s/pi`, the local `r1,r2,r3` expressions are term-identical to Eqs. (50)--(52); `r0=0` is the source's Landau-gauge statement. | The code passes `gamma=[2*r0,2*r1,2*r2,2*r3,0]` to `anomalous_dimension` and then returns `c(a_s(mu0))/c(a_s(mu))`. Neither the factor two nor that ratio is bound to `psi`, `Z_2`, lattice `Z_q`, or an inverse convention; the final zero is padding, not a source coefficient. **No numerical code repair is justified.** A future promotion needs an API-object/direction declaration, a source-backed truncation including the final slot, and an independent running reference point. |
| `tensor_anomalous_dimension_under_ms_bar` at `Def_quark_renorm.py:1540-1550`: `r0`, `r1`, `r2` only | Gracey 2003 `hep-ph/0304113v1`, PDF p. 21 Eq. (4.7), the MS anomalous dimension of the flavour-nonsinglet tensor current `bar(psi) sigma^(mu nu) psi`. | Eq. (4.7) fixes the MS operator and gives its three-loop series in `a=alpha_s/(4*pi)`. Substituting `C_A=3`, `C_F=4/3`, `T_F=1/2` and `a=a_s/4` gives the local `r0,r1,r2` (the local denominators are `4`, `16`, `64`). | Local `r3` has no equation locator in the checked source, and `gamma[4]=0` is source-free padding. The generic `scale_running` ratio has no declared `Z_T`/matrix-element/reciprocal interpretation. **No numerical code repair is justified.** A physical profile needs a four-loop policy or shorter source-closed vector, an object/direction binding, and a reference evolution check. |

The first row is not a claim that CR2000 supplies a lattice NPR quark-field
projector; the second is not a claim that Gracey supplies the local public
output's orientation. In particular, neither subset changes an accepted
`physical=True` profile, the status of the corresponding full API, or the four
selected E2E reference gates.

## Machine-readable physical-profile boundary

`Def_quark_renorm.py` exposes `QUARK_RENORM_PROFILE_REGISTRY` and makes
`get_quark_renorm_profile(..., physical=True)` / `invoke_quark_renorm_profile`
fail closed. An accepted row must be `MAPPED` or `MAPPED_DERIVED` **and** carry
the paper version, page, equation, source/target scheme, ordered
operator/definition basis, gauge, direction, expansion variable, normalization,
and source check below. `projector_or_definition` deliberately records the
paper's continuum conversion-function definition; it does not assert that a
caller supplied a matching lattice NPR vertex projector.

| Profile symbols | Status | Frozen binding retained in the registry |
|---|---|---|
| `quark_field_conversion_ms_bar_over_rimom` | `MAPPED` | CR2000 `hep-ph/9910332v2`, p. 11 Eq. (34), `C_2^RI` with Eqs. (9)-(12) notation; SU(3), Landau gauge; paper `a=alpha_s/(4*pi)=a_s/4`; API retains the literal `C_2` direction and leaves external `Z_q` direction to the caller. |
| `quark_field_conversion_ms_bar_over_rimom_prime` | `MAPPED` | CR2000 `hep-ph/9910332v2`, p. 11 Eq. (36), `C_2^{RI'}` under the same definition, gauge and expansion-variable binding. |
| `quark_mass_conversion_ms_bar_over_rimom_prime` | `MAPPED` | CR2000 `hep-ph/9910332v2`, pp. 11-12 Eqs. (37), (41); `m_MS-bar=C_m^{RI'}m_RI-prime`, SU(3), Landau gauge, `a=a_s/4`; Eq. (41) is the `N_f=4`, `a_s=0.1` source checkpoint. |
| `vector_conversion_ms_bar_over_rimom_prime`; `quark_field_conversion_rimom_prime_over_rimom` | `MAPPED_DERIVED` | Fixed-order `C_2^{RI'}/C_2^{RI}` ratio of the two CR2000 rows above through `O(a^3)`; neither name binds a vector-current NPR observable. |
| `tensor_conversion_ms_bar_over_rimom_prime` | `MAPPED` | G2003 `hep-ph/0304113v1`, p. 23 Eq. (4.11); non-singlet tensor `C_T=Z_T^{RI'}/Z_T^MS`, specialized to `xi=0`; API returns its fixed-order inverse with `a_s^2/16`, `a_s^3/64`. |
| `beta_coupling_constant`; `strong_coupling_constant` | `MAPPED`; `MAPPED_DERIVED` | BCK2016 `1606.08659v2`, p. 1 Eqs. (1)-(5); continuum MS-bar coupling, `a_s=alpha_s/pi`, no lattice projector or gauge-fixed NPR input. |
| `quark_mass_anomalous_dimension_under_ms_bar`; `scalar_anomalous_dimension_under_ms_bar` | `MAPPED`; `MAPPED_DERIVED` | BCK2014 `1402.6611v1`, pp. 3, 5 Eqs. (3.1)-(3.4), (4.7)-(4.12); MS-bar mass c-function and its explicit scalar reciprocal. |

The registry's `physical=True` is a **formula-provenance** assertion only. It
does not mean a raw vertex has the documented projector/gauge convention, and
it cannot establish gauge-fixed NPR, chiral/orbit or `a^2p^2` treatment,
continuum extrapolation, threshold matching, covariance propagation, or a
complete physical uncertainty budget.

## Search record and retained fail-closed scope

The following primary searches were performed, but the local code lacks enough
stored scheme/projector/gauge/normalization metadata to promote the tables.
They remain `UNVERIFIED_LEGACY`; bibliographic similarity or a matching decimal
is not a map.

| Local table(s) | Primary literature located | Why it remains unmapped/fail-closed |
|---|---|---|
| `quark_mass_conversion_ms_bar_over_rismom`, `_rismom_mu`, and scalar reciprocals | M. Gorbahn and S. Jaeger, *Precise MS-bar light-quark masses from lattice QCD in the RI/SMOM scheme*, Phys. Rev. D82 (2010) 114001, [arXiv:1004.3997v2](https://arxiv.org/abs/1004.3997), [DOI:10.1103/PhysRevD.82.114001](https://doi.org/10.1103/PhysRevD.82.114001), pp. 2-5, Eqs. (29)-(32). | The paper gives analytic general-covariant-gauge conversion functions, while the decimal paths do not record exact RI/SMOM versus RI/SMOM-gamma_mu projector, `xi`, matching-scale or special-function reduction. No coefficient is promoted from a decimal coincidence. |
| `tensor_conversion_ms_bar_over_rismom`, `_rismom_mu` | J. A. Gracey, *RI'/SMOM scheme amplitudes for quark currents at two loops*, Eur. Phys. J. C71 (2011) 1567, [arXiv:1101.5266v1](https://arxiv.org/abs/1101.5266), [DOI:10.1140/epjc/s10052-011-1567-8](https://doi.org/10.1140/epjc/s10052-011-1567-8), pp. 16-18, Eqs. (6.1)-(6.8). | `RI'/SMOM` is not silently interchangeable with each local `rismom` name; tensor-projector basis choices are material. No local equation-to-decimal reduction is frozen. |
| `quark_mass_conversion_ms_bar_over_rimom_prime2`, `quark_field_conversion_rimom_prime_over_rimom2`, `scalar_conversion_ms_bar_over_rimom_prime2`, `tensor_conversion_ms_bar_over_rimom_prime2` | CR2000 `hep-ph/9910332v2`, Appendix B pp. 26--30, equations numbered (1)--(12) within that Appendix, is a candidate for the general-gauge field/mass conversion classes. Gracey 2003 `hep-ph/0304113v1`, p. 23 Eq. (4.11), is a candidate for the general-gauge tensor conversion class. | The local names do not declare whether `rimom_prime2` means RI, RI-prime, RI/SMOM, or a particular projector; they also do not bind `xi` to the source gauge-parameter convention or fix the conversion direction. | Keep `UNVERIFIED_LEGACY`. A decimal/formula resemblance cannot choose an RI-family projector, gauge parameter sign, source/target direction, or truncation. A minimal repair requires a coefficient-by-coefficient symbolic reduction plus a frozen public scheme/basis contract, not a number substitution. |
| `quark_field_anomalous_dimension_under_ms_bar`; `tensor_anomalous_dimension_under_ms_bar` full outputs | CR2000 `hep-ph/9910332v2`, p. 4 Eq. (7), p. 16 Eqs. (50)--(52); Gracey 2003 `hep-ph/0304113v1`, p. 21 Eq. (4.7). | The two checked subsets are recorded above as `PRIMARY_COEFFICIENT_SUBSET_NONPROMOTING`. | Their local factor/padding and generic running orientation remain unbound. They are not `UNVERIFIED` coefficients, but their public physical outputs remain `UNVERIFIED_LEGACY` and explicit-only. |
| `matching_systematic_error` | No primary equation is identified for the seven channel slots, `trunZ*` decimals, `error_flag` transformations, or their uncertainty combination. | No ordered channel/operator basis, covariance, or perturbative-truncation prescription is stored. | `UNVERIFIED_LEGACY`; source discovery alone cannot repair the seven-channel systematic table. |
| all scalar/tensor Padé helpers and `pade_matching_factor` | No primary derivation, Padé order, fit input, pole domain or truncation prescription was found in the local source or cited records. | `UNVERIFIED_LEGACY`. These are literal replays only; they have no reference-point or physical singularity test and must not estimate a missing loop coefficient. |

## Explicit promotion gate

The routing configuration outside this directory already lists this skill in
`requires_explicit_invocation`. Any function not in a `MAPPED` or
`MAPPED_DERIVED` row remains an explicit-only literal or audit path. Such a
row can be replayed only through `get_quark_renorm_legacy_literal(...,
physical=False)`, whose wrapper permanently exposes `physical=False` and
`UNVERIFIED_LEGACY`. A `physical=True` request raises before numerical
evaluation. A future promotion requires a frozen paper version, page and
equation, ordered operator/projector basis, source/target direction, scheme,
gauge, color/flavor content, expansion variable, normalization, an independent
numerical reference point, and reviewer approval.
