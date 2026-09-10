# pyquda_helicity_renorm API

These are small host-side legacy perturbative matrices. Matching/anomalous-
dimension coefficient provenance is incomplete; see the gate below.
All public scales and couplings must be finite, real, and positive where
physically required; flavor counts and selector flags reject Booleans.
`reference/SOURCE_VERIFICATION.md` records the primary beta-function map and
why all matching/running matrix entries remain `UNVERIFIED_LEGACY`.

## Contents

- Coupling and EMT helper
- Helicity matching matrix
- Legacy nf=3 RG evolution
- Interpolation and outer-leg subtraction

## pyquda_helicity_renorm.beta

`beta(Nf)` returns five coefficients for the module's `a_s=alpha_s/pi`
convention.

- Returns: five Python/NumPy scalars.
- Example: `b = beta(3)`.
- Reference context: arXiv:1606.08659 for the five-loop beta function.

## pyquda_helicity_renorm.alpha_s

`alpha_s(mu_scale,Nf,Lambda=None)` returns successive one- through five-loop
truncations and rejects `mu/Lambda<=2`. Omitting `Lambda` selects
`0.332 GeV` only for `Nf=3`; every other flavor count requires an explicit,
flavor-appropriate value.

- Returns: NumPy length-five array.
- Example: `orders = alpha_s(3.0,3)`.

## pyquda_helicity_renorm.EMT_MatchingCoeff

`EMT_MatchingCoeff(g_0,muR,mu_scale)` is the legacy scalar EMT helper.

- Returns: scalar matching factor.
- Example: `r = EMT_MatchingCoeff(g0,2.0,3.0)`.
- Provenance: no exact equation was recorded in the source; block production
  use until supplied.

## pyquda_helicity_renorm.Helicity_MatchingCoeff

`Helicity_MatchingCoeff(muR,mu_scale,loop,is_pade,*,physical=False,physical_profile=None)` returns the literal tuple
`(R11,R12,R21,R22)`. Its fixed tuple/array placement is not an identified
physical operator basis. `loop=4,is_pade=0` is the explicit three-loop result;
`is_pade=1` adds the legacy estimate.

- Returns: four scalars.
- Example: `R = Helicity_MatchingCoeff(2.0,3.0,3,0)`.
- Provenance: `UNVERIFIED_LEGACY`; an operator basis and coefficient equations
  must be supplied before production.
- `physical=True`: fails closed unless a registered calculator profile passes
  the complete provenance audit. The default and `physical=False` are literal
  replay only, not a physical matching claim.

## pyquda_helicity_renorm.Helicity_MatchingCoeff_tmp

Same signature (including `*,physical=False,physical_profile=None`) with the alternate legacy `R11` convention used by the `tmp`
branch. It is not interchangeable with the primary function.

- Returns: four scalars.
- Example: `Rtmp = Helicity_MatchingCoeff_tmp(2.0,3.0,3,0)`.

## pyquda_helicity_renorm.Helicity_MatchingCoeff_array

`Helicity_MatchingCoeff_array(muR_array,loop,is_pade,is_fixing_order=1,
muscale=None,muscaleOVmuR=None,tmp=0,*,physical=False,physical_profile=None)` evaluates either matching function.
`muR_array` must be a nonempty finite positive real 1D array; complex values
and Boolean entries, including Booleans inside built-in, `UserList`, custom
non-string `Sequence`, and object-array inputs, are rejected before any float
conversion. Exactly one of
the fixed-scale or proportional-scale contracts is selected by
`is_fixing_order`.

- Returns: NumPy `(4,n_momenta)`.
- Example: `R = Helicity_MatchingCoeff_array(mus,3,0,muscale=3.0)`.
- `physical=True`: audits before evaluating the array and forwards the exact
  profile to every scalar call; a legacy literal is never silently substituted.

## pyquda_helicity_renorm.Helicity_RunningFactorCalculator

`Helicity_RunningFactorCalculator(nf=3,ope_type='ghelicity',*,physical=False,physical_profile=None)` accepts only
`nf=3`. Public methods are `run(mu1,mu2,R_at_mu1=None)` and
`run_array(mu_from,mu_to,R_init=None)`; the `calculate_*` names remain
compatibility methods.
The original `mu1`, `mu2`, `mu_from`, and `mu_to` are validated as positive
finite real scales before any squaring or logarithm, so a negative endpoint
cannot be accepted through `mu**2`. `R_at_mu1`/`R_init` must be a finite real
`(2,2)` matrix; Boolean entries in pure or mixed built-in/generic non-string
sequences and object arrays, plus complex arrays, are rejected before float
conversion rather than silently coercing them.

- Returns: NumPy `(2,2)` or batched matrices.
- Example: `U = Helicity_RunningFactorCalculator().run(2.0,3.0)`.
- Provenance: decimal anomalous dimensions remain unmapped.
- `physical=True`: rejects an incomplete, contradicted, or unregistered profile
  before allocating the calculator.

## pyquda_helicity_renorm.Get_Helicity_Running

`Get_Helicity_Running(mu_from,muscaleOVmuR,mu_to,*,physical=False,physical_profile=None)` is an nf=3 wrapper. The raw
scalar or one-dimensional `mu_from` is validated as finite, positive, real,
and non-Boolean before multiplication by `muscaleOVmuR`; built-in, `UserList`,
custom non-string `Sequence`, and object-array raw inputs cannot turn `True`
into `1.0` through NumPy dtype inference.

- Returns: real scalar/batched running matrices.
- Example: `U = Get_Helicity_Running([2.0,2.5],1.0,3.0)`.
- `physical=True`: retains the profile while delegating to the calculator; it
  must not be read as a physical scale-evolution result unless a registered map
  is added in a separately reviewed change.

## pyquda_helicity_renorm.momentum_interpolate

`momentum_interpolate(...)` sorts unique reference momenta, rejects
extrapolation, selects the two bracketing points, linearly interpolates means,
and propagates endpoint standard deviations assuming independence. An exact
target copies the matching reference point.
Target and reference arrays must be finite, real, one-dimensional, and
non-Boolean. Complex values and Boolean provenance in built-in/generic
non-string sequences and object arrays are rejected before conversion to
`float`, so no imaginary component or truth value can be silently narrowed. A
caller-created float ndarray has already
discarded its original element provenance and cannot be diagnosed later.

- Returns: `(means,sdevs)` arrays.
- Example: `mean,sd = momentum_interpolate(p,pref,data,ref,ncase)`.

## pyquda_helicity_renorm.AA_subtract

`AA_subtract(a2p2,AovA,a2p2_low,a2p2_high,n_poly,
is_show_fitting_result=0)` rejects non-negligible complex input, lazily imports
gvar/lsqfit, and performs an exploratory diagonal-error host fit. It does not
accept full covariance and does not propagate the posterior fit uncertainty
into the returned factor.
Before optional imports it also validates exact `(samples,len(a2p2))` shape,
finite nonempty arrays, an ordered window containing at least `n_poly+1`
points, non-Boolean polynomial/display selectors, and a nonsingular sample
mean denominator. Nonfinite output fails closed.

- Returns: real subtraction-factor array.
- Example: `fac = AA_subtract(a2p2,AovA,3,8,2)`.

Do not treat this API as validated numerical physics without an exact paper,
scheme, operator-basis, and equation map for every coefficient.

## Executable example registry

The quality contract maps every public symbol below to one real AST-direct
local test. CPU/stub/structural/rejection examples support at most E1; they
do not establish GPU, MPI, QUDA, runtime, interacting-physics, or production
evidence.

| Public symbol | Example ID | Test ID | Mode |
|---|---|---|---|
| `AA_subtract` | `API-EX-031` | `test_helicity_aa_subtract_rejects_complex_before_fit_imports` | `rejection-only-e1` |
| `EMT_MatchingCoeff` | `API-EX-033` | `test_helicity_public_api_invalid_domains_fail_closed` | `rejection-only-e1` |
| `Get_Helicity_Running` | `API-EX-034` | `test_helicity_rejects_nonfinite_scales_flags_and_negative_running_endpoint` | `rejection-only-e1` |
| `Helicity_MatchingCoeff` | `API-EX-002` | `test_all_eight_skills_match_independent_synthetic_golden` | `cpu-synthetic` |
| `Helicity_MatchingCoeff_array` | `API-EX-018` | `test_generic_sequences_preserve_boolean_provenance_before_array_coercion` | `cpu-synthetic` |
| `Helicity_MatchingCoeff_tmp` | `API-EX-033` | `test_helicity_public_api_invalid_domains_fail_closed` | `rejection-only-e1` |
| `Helicity_RunningFactorCalculator` | `API-EX-009` | `test_compatibility_aliases_delegate_to_primary_public_apis` | `cpu-synthetic` |
| `alpha_s` | `API-EX-034` | `test_helicity_rejects_nonfinite_scales_flags_and_negative_running_endpoint` | `rejection-only-e1` |
| `beta` | `API-EX-033` | `test_helicity_public_api_invalid_domains_fail_closed` | `rejection-only-e1` |
| `momentum_interpolate` | `API-EX-032` | `test_helicity_interpolation_brackets_unsorted_references` | `cpu-synthetic` |
