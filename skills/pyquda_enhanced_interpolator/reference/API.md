# pyquda_enhanced_interpolator API

Propagators end in `(sink_spin,source_spin,sink_color,source_color)`. Momentum
projection consumes exact checkerboard `(e,t,z,y,xh)` arrays.

## Contents

- Euclidean gamma operations
- Meson and nucleon enhanced kernels
- Wick contractions
- Spatial momentum reduction

Every public Dirac-matrix input must be numerical, finite, and exact `(4,4)`
on one NumPy/CuPy backend. `NaN`, `Inf`, object/string matrices, and wrong
shapes fail before gamma algebra or contractions.

## pyquda_enhanced_interpolator.euclidean_adjoint

`euclidean_adjoint(M, gamma_t)` computes `gamma_t M^dagger gamma_t` for a
bilinear kernel.

- Returns: same-backend spin matrix.
- Example: `Gbar = euclidean_adjoint(G,gamma_t)`.

## pyquda_enhanced_interpolator.euclidean_source_conjugate

`euclidean_source_conjugate(D, gamma_t)` computes `gamma_t D* gamma_t` for the
nucleon source diquark. It deliberately does not transpose `D`.

- Returns: source spin kernel.
- Example: `Dbar = euclidean_source_conjugate(D,gamma_t)`.

## pyquda_enhanced_interpolator.lightcone_gammas

`lightcone_gammas(gamma_t, gamma_parallel)` returns
`(gamma_t+i gamma_parallel, gamma_t-i gamma_parallel)/sqrt(2)`.

- Returns: `(gamma_plus,gamma_minus)`.
- Example: `gp,gm = lightcone_gammas(g4,gz)`.
- Reference: arXiv:2606.02447v2, Section 2.1 before Eq. (2.2).

## pyquda_enhanced_interpolator.plus_quark_projector

`plus_quark_projector(gamma_plus, gamma_minus)` returns the enhanced
quark-component kernel `gamma_minus gamma_plus/sqrt(2)`.

- Returns: spin matrix; idempotence is not assumed by the API.
- Example: `Qp = plus_quark_projector(gp,gm)`.

## pyquda_enhanced_interpolator.enhanced_meson_kernel

`enhanced_meson_kernel(bilinear_gamma, quark_projector, gamma_t,
antiquark_projector=None)` enhances both bilinear legs. `bilinear_gamma` is
the matrix in `bar(q) Gamma q`, not the matrix in `q^dagger Gamma q`.

- Returns: projected meson spin kernel.
- Example: `G = enhanced_meson_kernel(g4 @ g5,Qp,g4)` for the projected-pion
  Dirac-bar representation. Passing `g5` itself gives the exact zero kernel
  for the documented `Qp` convention.
- Provenance: generic two-leg algebra is an implementation extension; it is
  not a paper whitelist for arbitrary bilinears.

## pyquda_enhanced_interpolator.enhanced_pion_kernel

`enhanced_pion_kernel(gamma5, quark_projector, gamma_t)` is the fail-safe
canonical pion wrapper. It maps the paper's `u_+^dagger gamma5 d_+` identity to
the Dirac-bar input `Gamma=gamma_t gamma5` required by this module.

- Returns: projected pion spin kernel; under the documented Clifford
  convention it equals `sqrt(2) gamma_plus gamma5`.
- Example: `Gpi = enhanced_pion_kernel(g5,Qp,g4)`.
- Reference: Rui Zhang et al., arXiv:2501.00729v2, page 2,
  “Kinematic enhancement: theory” paragraph, unnumbered projected-pion
  identity.

## pyquda_enhanced_interpolator.enhanced_baryon_kernels

`enhanced_baryon_kernels(C, gamma5, gamma_t, gamma_parallel,
projector_kind='plus', diquark_kind='plus')` supports the documented
`parity/time/plus` and `identity/time/plus` choices.

- Returns: `EnhancedKernels` containing `projector`, `diquark_sink`, and
  `diquark_source_bar`; `.diquark` is a sink-only compatibility alias.
- Example: `K = enhanced_baryon_kernels(C,g5,g4,gz)`.
- Reference: arXiv:2606.02447v2 Eqs. (2.6), (2.9)-(2.11).

## pyquda_enhanced_interpolator.meson_contraction

`meson_contraction(Sq, Sa, gamma_sink, gamma_source_bar, gamma5)` performs the
gamma5-Hermitian connected local meson contraction.

- Returns: leading lattice/batch axes.
- Example: `local = meson_contraction(Su,Sd,Gsink,Gsrcbar,g5)`.

## pyquda_enhanced_interpolator.baryon_contraction

`baryon_contraction(Su1,Su2,Sd,P,D_sink,D_source_bar)` evaluates the explicit
direct-minus-exchange `uud` structure.

- Returns: leading lattice/batch axes.
- Example: `local = baryon_contraction(Su,Su2,Sd,K.projector,K.diquark_sink,K.diquark_source_bar)`.

## pyquda_enhanced_interpolator.momentum_project

`momentum_project(local_contraction, phase, *, spatial_comm=None,
cuda_aware_mpi=False)` requires exact parity extent `e=2` and sums `e,z,y,xh`;
the phase must already contain global coordinates. Multi-rank reduction uses
mpi4py-style buffer `Allreduce`; set `cuda_aware_mpi=True` for CuPy only after
verifying the target MPI stack.
`cuda_aware_mpi` is a strict Python/NumPy Boolean opt-in. Integers, strings,
and arbitrary truthy objects fail before contraction or communicator early
returns, preventing an accidental CUDA-aware claim.

- Returns: local or spatially all-reduced `(t,)` array.
- Example: `ct = momentum_project(local,phase,spatial_comm=space_comm)`.

No function builds propagators, phases, MPI subcommunicators, or files. CPU
algebra tests do not prove the quoted interpolator improves overlap.

## Executable example registry

The quality contract maps every public symbol below to one real AST-direct
local test. CPU/stub/structural/rejection examples support at most E1; they
do not establish GPU, MPI, QUDA, runtime, interacting-physics, or production
evidence.

| Public symbol | Example ID | Test ID | Mode |
|---|---|---|---|
| `EnhancedKernels` | `API-EX-016` | `test_enhanced_public_api_invalid_domains_fail_closed` | `cpu-structural` |
| `baryon_contraction` | `API-EX-013` | `test_enhanced_contractions_match_explicit_spin_color_loops` | `cpu-synthetic` |
| `enhanced_baryon_kernels` | `API-EX-012` | `test_enhanced_baryon_source_uses_complex_conjugate_without_transpose` | `cpu-synthetic` |
| `enhanced_meson_kernel` | `API-EX-015` | `test_enhanced_pion_dirac_bar_contract_and_exact_normalization` | `cpu-synthetic` |
| `enhanced_pion_kernel` | `API-EX-015` | `test_enhanced_pion_dirac_bar_contract_and_exact_normalization` | `cpu-synthetic` |
| `euclidean_adjoint` | `API-EX-011` | `test_critical_physics_mutants_are_distinguishable` | `cpu-synthetic` |
| `euclidean_source_conjugate` | `API-EX-011` | `test_critical_physics_mutants_are_distinguishable` | `cpu-synthetic` |
| `lightcone_gammas` | `API-EX-002` | `test_all_eight_skills_match_independent_synthetic_golden` | `cpu-synthetic` |
| `meson_contraction` | `API-EX-013` | `test_enhanced_contractions_match_explicit_spin_color_loops` | `cpu-synthetic` |
| `momentum_project` | `API-EX-014` | `test_enhanced_momentum_projection_has_exact_layout_and_spatial_allreduce` | `cpu-synthetic` |
| `plus_quark_projector` | `API-EX-002` | `test_all_eight_skills_match_independent_synthetic_golden` | `cpu-synthetic` |
