# pyquda_momentum_smear API

All volume work stays in PyQUDA/QUDA. Mode tuples are `(kx,ky,kz)` and may be
fractional. Raw/mixed list, tuple, `UserList`, custom non-string `Sequence`, and
object-array modes reject Python/NumPy Booleans before numeric coercion. Source
coordinates are PyQUDA `(x,y,z,t)` lists where required.
Global/local/grid extents and solver counts reject Booleans; a multi-rank
lattice signature must include `grid_coord`, not only equal extents.

## Contents

- Result containers and mode helpers
- Link phasing and Wuppertal smearing
- Source/sink and MRHS inversion
- Meson/baryon primary APIs and compatibility aliases
- Fourier and sequential-source phases
- Cross-skill artifact ownership

## Cross-skill artifact ownership

The source, propagator, sink-propagator, and sequential-line objects returned
by this module are not blending basis arrays. In a workflow that also uses
`pyquda_blending`, do not pass a completed momentum-smeared line to
`meson_elemental` or `baryon_elemental`. Preserve separate artifact IDs and
combine them only through an explicit basis-source callback, Dirac solve, or
observable assembly. A separately requested transformation of a blending
basis vector belongs to the blending basis-construction stage.

## pyquda_momentum_smear.result_containers

`MomentumSmearResult`, `BaryonMomentumSmearResult`, and
`SingleMomentumSmearResult` record source mode, raw propagators, optional sink
propagators, and the active sequential sink mode. `MesonMomentumSmearResult`
is a compatibility alias of `MomentumSmearResult`.
`SingleMomentumSmearResult.sink_k_mode` stores the canonical independent sink
mode. It is defaulted to `None` at the end of the dataclass, preserving legacy
three-positional-field construction.

- Returns: immutable dataclass instances from high-level calls.
- Example: `result = momentum_smear_meson(...)`.

## pyquda_momentum_smear.negate_mode

`negate_mode(mode)` validates and flips all three components.

- Returns: three-float tuple.
- Example: `km = negate_mode((0.5,0,0))`.

## pyquda_momentum_smear.phase_spatial_links

`phase_spatial_links(base_gauge, k_mode)` copies the gauge and multiplies
direction `j=x,y,z` by `exp(+2*pi*i*k_j/L_j)`; time links remain unchanged.

- Returns: phased gauge copy.
- Example: `gk = phase_spatial_links(gauge,(0.5,0,0))`.
- Reference: arXiv:1602.05525 Eq. (24).

## pyquda_momentum_smear.momentum_smear_kernel

`momentum_smear_kernel(field, base_gauge, latt_info, k_mode, radius, n_steps)`
checks lattice decomposition and positive PyQUDA Wuppertal alpha.
Compatibility includes global/local/grid extents and rank `grid_coord`; on a
nontrivial grid, missing rank-coordinate metadata fails closed.
All mode/domain/decomposition checks finish before importing
`pyquda_utils.source`, copying the gauge, or calling `gaussianSmear`.

- Returns: smeared same-kind PyQUDA field.
- Example: `src_k = momentum_smear_kernel(src,gauge,info,k,4.0,40)`.
- Domain: `radius` must be a finite positive real value; `n_steps` must be a
  non-Boolean positive integer; `radius**2 < 2*n_steps/3` is mandatory.

## pyquda_momentum_smear.apply_momentum_smear_source

`apply_momentum_smear_source(dirac, source_field, base_gauge, k_mode, rho,
n_steps, mrhs=1, restart=0)` smears then calls `core.invertPropagator`.
`mrhs` is a non-Boolean positive integer and `restart` is a non-Boolean
nonnegative integer. Solver, smearing, mode, and lattice metadata are validated
before importing `core`, smearing, or entering an inversion.

- Returns: source-smeared propagator.
- Example: `prop = apply_momentum_smear_source(D,src,gauge,k,4,40,mrhs=12)`.

## pyquda_momentum_smear.apply_momentum_smear_sink

`apply_momentum_smear_sink(propagator, base_gauge, k_mode, rho, n_steps)`
applies the spatial kernel to the sink end without inversion.

- Returns: sink-smeared propagator.
- Example: `prop_ss = apply_momentum_smear_sink(prop,gauge,kf,4,40)`.

## pyquda_momentum_smear.apply_momentum_smear_seqprop

`apply_momentum_smear_seqprop(dirac, sequential_source, t_sink, base_gauge,
sink_k_mode, rho, n_steps, mrhs=1, restart=0,*,already_smeared=None)` calls
`core.invertSequential` under an explicit one-smear contract. The flag may not
be omitted. `already_smeared=True` requires `sink_k_mode=None` and directly
inverts the supplied source. `already_smeared=False` requires an explicit mode
and applies the kernel exactly once before inversion.
The same non-Boolean `mrhs/restart` and lattice-decomposition checks apply.
Time, one-smear flag, optional mode, and smearing domain are validated before
the `core` import or either smearing/inversion path.

- Returns: sequential propagator.
- Example: `seqprop = apply_momentum_smear_seqprop(D,seq,ts,gauge,None,4,40,mrhs=12,already_smeared=True)`.

## pyquda_momentum_smear.momentum_smear_propagator

`momentum_smear_propagator(..., sink_k_mode=None, mrhs=1, restart=0)` is the
generic one-line source plus optional sink wrapper. Both source and optional
sink modes are canonicalized before source inversion, so an invalid sink mode
cannot consume a solve first.

- Returns: `SingleMomentumSmearResult`.
- Example: `line = momentum_smear_propagator(D,src,gauge,ki,4,40,sink_k_mode=kf)`.
- Provenance: the returned `sink_k_mode` is `None` when no independent sink
  smearing was requested and the validated canonical three-tuple otherwise.

## pyquda_momentum_smear.momentum_smear_meson

`momentum_smear_meson(latt_info, base_gauge, dirac, x_src, k_mode, radius,
n_steps, sink_mode='s2s', sink_k_mode=None, mrhs=1, restart=0)` constructs
source `+k/-k` and optional sink `+kf/-kf` lines. Every pure source/sink mode,
coordinate, solver, smearing, and decomposition check precedes point-source
construction and both inversions.
`sink_k_mode` is accepted only when `sink_mode='s2s'`; S-to-P rejects any
non-`None` value before import, point-source construction, or inversion.

- Returns: `MomentumSmearResult`.
- Example: `meson = momentum_smear_meson(info,gauge,D,x0,ki,4,40,sink_k_mode=kf,mrhs=12)`.

## pyquda_momentum_smear.momentum_smear_baryon_degenerate

Same controls, but returns one equal-mode degenerate-baryon line.
The same S-to-P prohibition on `sink_k_mode` applies.

- Returns: `BaryonMomentumSmearResult`.
- Example: `bline = momentum_smear_baryon_degenerate(info,gauge,D,x0,k,4,40)`.

## pyquda_momentum_smear.compatibility_aliases

`momentum_smear_s_to_p`, `momentum_smear_s_to_s`,
`momentum_smear_meson_s_to_p`, `momentum_smear_meson_s_to_s`,
`momentum_smear_baryon_s_to_p`, and `momentum_smear_baryon_s_to_s` forward to
the two primary hadron APIs with a fixed `sink_mode`.
Passing a second `sink_mode` keyword is rejected rather than silently
overriding or duplicating the alias-owned value.
The S-to-P aliases also reject a supplied `sink_k_mode`; the S-to-S aliases
retain the optional independent sink mode.

- Returns: the corresponding primary result container.
- Example: `result = momentum_smear_s_to_p(info,gauge,D,x0,k,4,40)`.

## pyquda_momentum_smear.fourier_phase_pair

`fourier_phase_pair(latt_info, momentum, x_src=(0,0,0,0))` returns sink
`exp(-iP.(x-x0))` and positive `exp(+iP.(x-x0))` arrays. The positive array is
a physical sequential-source choice only if the downstream contraction
explicitly daggers/conjugates that line. Neither this helper nor PyQUDA
`source.sequential12` performs that dagger; `sequential12` only selects the
sink time slice.

- Returns: `(sink_phase,sequential_phase)`.
- Example: `phase_sink,phase_seq = fourier_phase_pair(info,Pf,x0)`.

## pyquda_momentum_smear.build_sequential_source

`build_sequential_source(result, latt_info, tseq, sequential_source_phase,
gamma_sink_bar, gamma_source_bar, base_gauge, radius, n_steps)` builds a meson
fixed-sink source and applies active-leg sink smearing exactly once when needed.
`build_meson_sequential_source` is a compatibility alias.
Record consistency, time, layout, backend, gamma shape, phase shape, smearing
domain, and optional active mode all fail before importing `opt_einsum` or
`pyquda_utils` and before allocating the sequential field.

The caller-supplied phase multiplies the full checkerboard spectator block
before `source.sequential12` selects the global sink time. The builder requires
exact `(e,t,z,y,xh,4,4,3,3)` storage with `e=2`, one NumPy/CuPy backend for the
phase, propagator, and gamma kernels, and matching lattice decomposition
metadata.
On multi-rank layouts this equality includes `grid_coord`; equal shapes on
different rank-local blocks are not considered compatible.

- Returns: device-backed sequential source.
- Example: `seq = build_sequential_source(result,info,ts,phase_seq,Gs,G0,gauge,4,40)`.

Solver convergence, boundary conditions, gauge revision, actual performance,
and the optimal relation between `k` and hadron momentum require target-run
evidence; this API does not infer them.

## Executable example registry

The quality contract maps every public symbol below to one real AST-direct
local test. CPU/stub/structural/rejection examples support at most E1; they
do not establish GPU, MPI, QUDA, runtime, interacting-physics, or production
evidence.

| Public symbol | Example ID | Test ID | Mode |
|---|---|---|---|
| `BaryonMomentumSmearResult` | `API-EX-042` | `test_momentum_public_api_invalid_domains_and_aliases_fail_closed` | `cpu-structural` |
| `MesonMomentumSmearResult` | `API-EX-042` | `test_momentum_public_api_invalid_domains_and_aliases_fail_closed` | `cpu-structural` |
| `MomentumSmearResult` | `API-EX-042` | `test_momentum_public_api_invalid_domains_and_aliases_fail_closed` | `cpu-structural` |
| `SingleMomentumSmearResult` | `API-EX-042` | `test_momentum_public_api_invalid_domains_and_aliases_fail_closed` | `cpu-structural` |
| `apply_momentum_smear_seqprop` | `API-EX-058` | `test_sequential_inversion_requires_explicit_one_smear_contract` | `cpu-synthetic` |
| `apply_momentum_smear_sink` | `API-EX-043` | `test_momentum_public_api_invalid_domains_and_aliases_fail_closed` | `rejection-only-e1` |
| `apply_momentum_smear_source` | `API-EX-039` | `test_momentum_base_interfaces_are_preserved` | `cpu-synthetic` |
| `build_meson_sequential_source` | `API-EX-009` | `test_compatibility_aliases_delegate_to_primary_public_apis` | `cpu-synthetic` |
| `build_sequential_source` | `API-EX-009` | `test_compatibility_aliases_delegate_to_primary_public_apis` | `cpu-synthetic` |
| `fourier_phase_pair` | `API-EX-017` | `test_fourier_phase_pair_and_sequential_builder_contract` | `cpu-synthetic` |
| `momentum_smear_baryon_degenerate` | `API-EX-041` | `test_momentum_metadata_fails_before_imports_or_inversions` | `rejection-only-e1` |
| `momentum_smear_baryon_s_to_p` | `API-EX-043` | `test_momentum_public_api_invalid_domains_and_aliases_fail_closed` | `rejection-only-e1` |
| `momentum_smear_baryon_s_to_s` | `API-EX-043` | `test_momentum_public_api_invalid_domains_and_aliases_fail_closed` | `rejection-only-e1` |
| `momentum_smear_kernel` | `API-EX-036` | `test_link_phase_and_wuppertal_forwarding_agree` | `cpu-synthetic` |
| `momentum_smear_meson` | `API-EX-041` | `test_momentum_metadata_fails_before_imports_or_inversions` | `rejection-only-e1` |
| `momentum_smear_meson_s_to_p` | `API-EX-009` | `test_compatibility_aliases_delegate_to_primary_public_apis` | `cpu-synthetic` |
| `momentum_smear_meson_s_to_s` | `API-EX-009` | `test_compatibility_aliases_delegate_to_primary_public_apis` | `cpu-synthetic` |
| `momentum_smear_propagator` | `API-EX-040` | `test_momentum_lattice_signature_tracks_rank_coordinate_and_sink_mode` | `cpu-synthetic` |
| `momentum_smear_s_to_p` | `API-EX-009` | `test_compatibility_aliases_delegate_to_primary_public_apis` | `cpu-synthetic` |
| `momentum_smear_s_to_s` | `API-EX-009` | `test_compatibility_aliases_delegate_to_primary_public_apis` | `cpu-synthetic` |
| `negate_mode` | `API-EX-039` | `test_momentum_base_interfaces_are_preserved` | `cpu-synthetic` |
| `phase_spatial_links` | `API-EX-002` | `test_all_eight_skills_match_independent_synthetic_golden` | `cpu-synthetic` |
