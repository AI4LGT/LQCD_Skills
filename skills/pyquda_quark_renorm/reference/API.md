# pyquda_quark_renorm API

This mixed legacy module contains small CPU perturbative tables, optional host
fit/I/O routines, and NumPy/CuPy spin-color kernels. Public status is defined
by its `__all__`; internal formatting helpers are not routed.

## Contents

- Coupling and legacy conversion series
- Anomalous-dimension running and systematic tables
- Optional host fits
- Companion jackknife resampling
- Spin-color inversion, adjoint, and amputation
- Provenance boundary

## pyquda_quark_renorm.coupling

- `beta_coupling_constant(nf)` returns the module's five beta coefficients.
- `strong_coupling_constant(scale,Lambda=None,nf=3)` returns five successive
  `alpha_s/pi`-style truncations and requires `scale/Lambda>3`.
- `alpha_s(scale,nf=3,Lambda=None)` returns only the scalar fifth-order
  `alpha_s`, not the five-entry truncation array.

All scales and Lambdas must be finite positive reals and `nf` must be a
non-Boolean integer in `0..16`. `Lambda=None` maps to `0.332 GeV` only for
`nf=3`; every other flavor count must pass an explicit flavor-appropriate
Lambda.

- Returns: `beta_coupling_constant` and `strong_coupling_constant` return
  NumPy length-five arrays; `alpha_s` returns one scalar.
- Example: `orders = strong_coupling_constant(3.0,0.332,3)`.
- General source context: arXiv:1606.08659; verify normalization against the
  downstream conversion series.

## pyquda_quark_renorm.conversion_series

Every function below returns a length-five perturbative series at `scale`.
The exact function name fixes the operator and intermediate scheme:
Functions whose signature has `Lambda=None` use the same `nf=3`-only default
gate as the coupling API; `tensor_conversion_ms_bar_over_rismom_mu` retains an
explicit required `Lambda` argument. General-gauge branches require finite
real `xi`.

| Function | Intended legacy series |
|---|---|
| `vector_conversion_ms_bar_over_rimom_prime` | historical name; derived `C_2^{RI'}/C_2^{RI}` field ratio through `O(a^3)`, not a vector-current map |
| `quark_mass_conversion_ms_bar_over_rismom` | mass, RI/SMOM gamma-mu family |
| `quark_mass_conversion_ms_bar_over_rismom_mu` | mass, alternate RI/SMOM family |
| `quark_field_conversion_rimom_prime_over_rimom` | RI-prime/MOM over RI/MOM field conversion |
| `quark_field_conversion_ms_bar_over_rimom` | field, RI/MOM to MS-bar |
| `quark_field_conversion_ms_bar_over_rimom_prime` | field, RI-prime/MOM to MS-bar |
| `quark_mass_conversion_ms_bar_over_rimom_prime` | mass, RI-prime/MOM to MS-bar |
| `scalar_conversion_ms_bar_over_rismom` | scalar, RI/SMOM gamma-mu family |
| `scalar_conversion_ms_bar_over_rismom_mu` | scalar, alternate RI/SMOM family |
| `scalar_conversion_ms_bar_over_rimom_prime` | scalar, RI-prime/MOM to MS-bar |
| `tensor_conversion_ms_bar_over_rismom` | tensor, RI/SMOM gamma-mu family |
| `tensor_conversion_ms_bar_over_rismom_mu` | tensor, alternate RI/SMOM family |
| `tensor_conversion_ms_bar_over_rimom_prime` | tensor, RI-prime/MOM to MS-bar |
| `quark_mass_conversion_ms_bar_over_rimom_prime2` | generalized-gauge mass series with `xi` |
| `quark_field_conversion_rimom_prime_over_rimom2` | generalized-gauge field ratio with `xi` |
| `scalar_conversion_ms_bar_over_rimom_prime2` | generalized-gauge scalar series with `xi` |
| `tensor_conversion_ms_bar_over_rimom_prime2` | generalized-gauge tensor series with `xi` |

- Returns: NumPy length-five array for each function.
- Example: `ct = tensor_conversion_ms_bar_over_rismom(3.0,0.332,3)`.
- References: arXiv:hep-ph/9910332 for RI/RI-prime conversion context and
  arXiv:0901.2599 for RI/SMOM. The copied source does not yet map every table
  to an exact equation, so these citations are not blanket validation.
- Blocker: `tensor_conversion_ms_bar_over_rimom_prime` contains a high-risk
  third-order normalization. It is now directly mapped to Gracey,
  arXiv:hep-ph/0304113v1, Eq. (4.11), PDF p. 23: the paper variable is
  `alpha_s/(4*pi)`, so the fixed-order inverse uses `a_s**3/64`, not `/6`.
  See `reference/SOURCE_VERIFICATION.md`; remaining conversion tables are not
  blanket-validated.

## pyquda_quark_renorm.running

- `anomalous_dimension(gamma,nf)` maps a five-entry coefficient series to the
  module's running convention.
- `scale_running(scale,scale0,dim,Lambda,nf)` integrates the perturbative ratio.
- `quark_mass_anomalous_dimension_under_ms_bar`,
  `quark_field_anomalous_dimension_under_ms_bar`,
  `scalar_anomalous_dimension_under_ms_bar`, and
  `tensor_anomalous_dimension_under_ms_bar` supply legacy operator tables.

- Returns: NumPy truncation arrays.
- Example: `u = tensor_anomalous_dimension_under_ms_bar(3.0,2.0,0.332,3)`.
- Domain: `gamma` and `dim` are finite real one-dimensional arrays of length
  three to five. Boolean provenance in built-in, `UserList`, custom non-string
  `Sequence`, and object-array inputs, plus complex arrays, is rejected before
  float conversion; scales and `Lambda` are finite positive reals.
- Provenance: the MS mass table and generic c-function algebra are mapped to
  arXiv:1402.6611 Eqs. (3.1)-(3.4), (4.7)-(4.12), but code returns the inverse
  c-ratio `c(a_s(scale0))/c(a_s(scale))`. Field/tensor running tables remain
  subject to the equation-level gate; see `reference/SOURCE_VERIFICATION.md`.

## pyquda_quark_renorm.matching_systematic_error

`matching_systematic_error(mu,MOM_flag,error_flag,scale0=2.0,Lambda=None,nf=3)`
returns the legacy seven-channel table. It explicitly rejects nondefault
`scale0/Lambda/nf` because the historical branches do not consistently
propagate them.

- Returns: NumPy length-seven array.
- Example: `fac = matching_systematic_error(3.0,1,1)`.

## pyquda_quark_renorm.pade_helpers

`scalar_conversion_ms_bar_over_rimom_pade_3loop`,
`scalar_conversion_ms_bar_over_rimom_pade_4loop`,
`tensor_conversion_ms_bar_over_rimom_pade_3loop`, and
`tensor_conversion_ms_bar_over_rimom_pade_4loop` return legacy rational
approximants. `pade_matching_factor(mu,scale0,Lambda,nf)` returns its historical
three scalar channels. `scale0` is a deprecated, validated compatibility no-op:
no running factor is returned, so tensor and running branches are not evaluated.

- Returns: NumPy perturbative arrays.
- Example: `pade = pade_matching_factor(3.0,2.0,0.332,3)`.
- Provenance: fitted Padé decimals are unmapped and must not be presented as an
  independently known loop coefficient.

## pyquda_quark_renorm.host_analysis

- `read_data(...)` lazily imports pandas and parses the legacy table schema.
- `ma_fit(...)`, `a2p2_fit(...)`, and `ratio_fit(...)` lazily import
  gvar/lsqfit and operate on reduced host data.
- `read_data` selects the exact window `fitmin < a2p2 <= upper_limit`; its
  aligned `a2p2` input must already be finite and nondecreasing so one
  contiguous slice is unambiguous.
- Before optional imports, these entrypoints validate non-Boolean selectors,
  table/count contracts, exact aligned shapes, finite real inputs, fit-window
  support, positive scales, and nonsingular interpolation denominators.
- In particular, `a2p2_fit` validates the selected inverse-spacing array `am1`,
  while `ratio_fit` independently validates `am1` and `am1_0`; Boolean,
  complex, NaN, or infinite scale arrays fail before importing gvar/lsqfit.
- With `pade_flag=1`, `pade34` is a required non-Boolean index in `0..2`.
- With `pade_flag=0`, `read_data` does not evaluate `pade_matching_factor` at
  all; disabled legacy matching therefore cannot fail or add hidden cost.
- The legacy mean/error constructors are diagonal-error fits. Bracketing
  interpolation and later products preserve covariance only when the caller
  has already supplied correlated gvar sources; they cannot reconstruct
  covariance from plain means/errors.

- Returns: host arrays and fit objects defined by each signature.
- Example: `fit = ma_fit(values,errors,mqa,prior,0)`.
- Contract: expose fit window, priors, covariance, `ZA`, momentum definition,
  and configuration correlation. Never pass a full CuPy lattice field here.

## pyquda_quark_renorm.jackknife_resampling

`jackknife_resampling(data)` is exported by the companion
`scripts/Def_qcd_analysis.py` and forms leave-one-out means along the leading
configuration axis.

- Returns: NumPy jackknife array with the same leading sample count.
- Example: `samples = jackknife_resampling(per_configuration_values)`.
- Boundary: downstream fits must retain shared-configuration correlations;
  this helper alone does not construct a full covariance model.

## pyquda_quark_renorm.inverse_propagator

`inverse_propagator(prop,on_device=None)` treats
`(...,sink_spin,source_spin,sink_color,source_color)` as a 12x12 matrix.
`on_device`, when present, must be an actual Python/NumPy Boolean and is only a
consistency check against the inferred backend; integers and truthy objects are
rejected before inversion.

- Returns: same-shape/backend batched inverse.
- Example: `Sinv = inverse_propagator(Sp)`.

## pyquda_quark_renorm.adj

`adj(prop,Gm5)` computes `gamma5 prop^dagger gamma5`, swapping both spin and
color source/sink axes. Any number of leading batch/lattice axes is supported.

- Returns: same-shape/backend array.
- Example: `Sbar = adj(Sinv,g5)`.

## pyquda_quark_renorm.Lambda_O_con

`Lambda_O_con(Sq,GreenB,is_jack=1)` optionally jackknifes and amputates a
connected Green function.
Both arrays must have exact equal `(Ncfg,4,4,3,3)` layout and one backend;
implicit broadcasting of the configuration axis is forbidden. `is_jack` is a
non-Boolean selector in `{0,1}`.

- Returns: backend `(n,4,4)` amputated spin vertex under the legacy contraction.
- Example: `Lambda = Lambda_O_con(Sq,G,is_jack=1)`.
- Boundary: input is already gauge-fixed and momentum transformed; general
  nonexceptional external legs require an explicit two-propagator treatment.

## pyquda_quark_renorm.Lambda_O_dis

`Lambda_O_dis(Sq,current,is_jack=1)` builds and amputates the legacy
disconnected product.
`Sq` has exact `(Ncfg,4,4,3,3)` layout and `current` exact `(Ncfg,)`; both use
one backend and configuration-axis broadcasting is forbidden.

- Returns: backend `(n,4,4)` vertex.
- Example: `Lambda_disc = Lambda_O_dis(Sq,loop,is_jack=1)`.
- Boundary: state and test the vacuum-subtraction convention before use.

The array identities are covered by CPU oracles, including non-color-diagonal
daggers. No interacting NPR, CuPy kernel, QUDA solve, MPI, fit-quality, or
coefficient-level physics validation was performed in the local audit.

## Executable example registry

The quality contract maps every public symbol below to one real AST-direct
local test. CPU/stub/structural/rejection examples support at most E1; they
do not establish GPU, MPI, QUDA, runtime, interacting-physics, or production
evidence.

| Public symbol | Example ID | Test ID | Mode |
|---|---|---|---|
| `QUARK_RENORM_PHYSICAL_PROFILES` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `QUARK_RENORM_PROFILE_REGISTRY` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `QuarkRenormLiteral` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `QuarkRenormProfile` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `QuarkRenormProfileError` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `UnknownQuarkRenormProfileError` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `UnverifiedLegacyError` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `get_quark_renorm_legacy_literal` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `get_quark_renorm_profile` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `invoke_quark_renorm_profile` | `API-EX-059` | `test_public_profile_api_registry_is_explicit_and_fail_closed` | `cpu-structural` |
| `Lambda_O_con` | `API-EX-046` | `test_quark_amputation_entrypoints_reject_batch_broadcast_and_bad_flags` | `rejection-only-e1` |
| `Lambda_O_dis` | `API-EX-046` | `test_quark_amputation_entrypoints_reject_batch_broadcast_and_bad_flags` | `rejection-only-e1` |
| `a2p2_fit` | `API-EX-001` | `test_adversarial_public_inputs_fail_closed_before_numerical_kernels` | `rejection-only-e1` |
| `adj` | `API-EX-011` | `test_critical_physics_mutants_are_distinguishable` | `cpu-synthetic` |
| `alpha_s` | `API-EX-052` | `test_quark_scale_flavor_contract_is_finite_and_explicit` | `cpu-synthetic` |
| `anomalous_dimension` | `API-EX-048` | `test_quark_conversion_has_no_undefined_zeta_dependency` | `cpu-synthetic` |
| `beta_coupling_constant` | `API-EX-052` | `test_quark_scale_flavor_contract_is_finite_and_explicit` | `cpu-synthetic` |
| `inverse_propagator` | `API-EX-010` | `test_cpu_differential_implementations_match_independent_references` | `cpu-synthetic` |
| `jackknife_resampling` | `API-EX-047` | `test_quark_companion_jackknife` | `cpu-synthetic` |
| `ma_fit` | `API-EX-038` | `test_mixed_boolean_arrays_fail_before_optional_numerical_imports` | `rejection-only-e1` |
| `matching_systematic_error` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `pade_matching_factor` | `API-EX-049` | `test_quark_pade_factor_skips_discarded_running_and_tensor_branches` | `cpu-synthetic` |
| `quark_field_anomalous_dimension_under_ms_bar` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `quark_field_conversion_ms_bar_over_rimom` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `quark_field_conversion_ms_bar_over_rimom_prime` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `quark_field_conversion_rimom_prime_over_rimom` | `API-EX-008` | `test_chetyrkin_retey_field_ratio_is_eq36_over_eq34` | `cpu-synthetic` |
| `quark_field_conversion_rimom_prime_over_rimom2` | `API-EX-048` | `test_quark_conversion_has_no_undefined_zeta_dependency` | `cpu-synthetic` |
| `quark_mass_anomalous_dimension_under_ms_bar` | `API-EX-037` | `test_mapped_ms_mass_running_has_equal_scale_and_composition` | `cpu-synthetic` |
| `quark_mass_conversion_ms_bar_over_rimom_prime` | `API-EX-002` | `test_all_eight_skills_match_independent_synthetic_golden` | `cpu-synthetic` |
| `quark_mass_conversion_ms_bar_over_rimom_prime2` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `quark_mass_conversion_ms_bar_over_rismom` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `quark_mass_conversion_ms_bar_over_rismom_mu` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `ratio_fit` | `API-EX-001` | `test_adversarial_public_inputs_fail_closed_before_numerical_kernels` | `rejection-only-e1` |
| `read_data` | `API-EX-051` | `test_quark_read_data_skips_pade_factors_when_disabled` | `cpu-synthetic` |
| `scalar_anomalous_dimension_under_ms_bar` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `scalar_conversion_ms_bar_over_rimom_pade_3loop` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `scalar_conversion_ms_bar_over_rimom_pade_4loop` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `scalar_conversion_ms_bar_over_rimom_prime` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `scalar_conversion_ms_bar_over_rimom_prime2` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `scalar_conversion_ms_bar_over_rismom` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `scalar_conversion_ms_bar_over_rismom_mu` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `scale_running` | `API-EX-045` | `test_property_equal_scale_running_is_literal_identity_only` | `cpu-synthetic` |
| `strong_coupling_constant` | `API-EX-048` | `test_quark_conversion_has_no_undefined_zeta_dependency` | `cpu-synthetic` |
| `tensor_anomalous_dimension_under_ms_bar` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `tensor_conversion_ms_bar_over_rimom_pade_3loop` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `tensor_conversion_ms_bar_over_rimom_pade_4loop` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `tensor_conversion_ms_bar_over_rimom_prime` | `API-EX-030` | `test_gracey_tensor_eq411_uses_a_s_cubed_over_64` | `cpu-synthetic` |
| `tensor_conversion_ms_bar_over_rimom_prime2` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `tensor_conversion_ms_bar_over_rismom` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `tensor_conversion_ms_bar_over_rismom_mu` | `API-EX-050` | `test_quark_public_conversion_and_running_invalid_domains_fail_closed` | `rejection-only-e1` |
| `vector_conversion_ms_bar_over_rimom_prime` | `API-EX-048` | `test_quark_conversion_has_no_undefined_zeta_dependency` | `cpu-synthetic` |
