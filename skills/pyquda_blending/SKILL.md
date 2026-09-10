---
name: pyquda_blending
description: >
  Generate or review PyQUDA array kernels for the blended all-to-all method:
  low Laplacian modes plus an explicitly constructed stochastic complement,
  blending weights, meson/baryon elementals, projected perambulators, and
  two-point contractions. Trigger on blending, stochastic distillation,
  blended perambulator, n_ev/n_st, or arXiv:2505.01719. Do not use for generic
  momentum smearing, RI renormalization, fitting, plotting, or effective-mass
  analysis.
---

# PyQUDA blending kernels

## Progressive disclosure

Read `USER_GUIDE.md` only when the user asks for principles, formulas,
derivations, natural-language examples, a complete workflow, or validation
boundaries. For ordinary API generation or review, use this file plus
`reference/API.md` and the bundled script.

Read `reference/PHYSICS_CONTRACT.md` before making an unbiased-estimator,
distillation-limit, dilution, perambulator-ownership, or production claim.

Read `VALIDATION.md` when the user asks for the current release label, direct
evidence, blockers, local verification commands, or the E2E promotion path.

Read `reference/API.md`, then import `scripts/Def_blending.py`. Generate a
short production driver that calls these kernels; do not duplicate their
weights, phase logic, layout helpers, or validation.

## Function index

- Weights: `omega`, `blending_tuple_weight`
- Optional private smearing: `phase_spatial_links_internal`, `momentum_smear_internal`
- Global elementals: `spatial_fourier_phase`, `meson_elemental`, `baryon_elemental`
- Projected solves: `generate_perambulator`
- Contractions: `meson_two_point`, `baryon_two_point`

## Composition ownership

A quark source/propagator produced by `$pyquda_momentum_smear` is not a
blending `eigvecs` basis. Keep that line field and the low-mode plus
orthogonal-stochastic basis as independent artifacts. They may meet only at
an explicitly defined source callback, solve, or observable-assembly boundary;
never pass the completed source field to `meson_elemental` merely because the
two stages are listed in sequence. If `momentum_smear_internal` is deliberately
applied to a blending basis vector, that transformation is owned and recorded
inside the blending basis construction rather than inherited from a generic
source-smearing result.

## pyquda_blending.meson_elemental

Contract local `(N,t,z,y,x,c)` basis vectors, use global coordinates and the
global color-space dimension `3*V3`, then sum over a same-timeslice spatial
communicator. A CuPy call requires CUDA-aware MPI; no host fallback is hidden.

## pyquda_blending.baryon_elemental

Apply the color epsilon contraction and rank-three blending weight with the
same global-coordinate and spatial-communicator contract.

## pyquda_blending.generate_perambulator

Return `(t,sink_spin,source_spin,sink_mode,source_mode)`. The callback API is
serial orchestration; for production mode counts the caller must provide an
MRHS/batched solve boundary and record true residuals and total cost.

## pyquda_blending.meson_two_point

Use explicit gamma5 Hermiticity for the backward line. A bare conjugate
transpose is not interchangeable for a nontrivial spin kernel.

## Required physical contract

1. Supply stochastic complement vectors already projected orthogonal to the
   retained low-mode subspace. State the noise ensemble and dilution scheme;
   stochastic unbiasedness is an ensemble/noise expectation, not a property
   of one sample.
2. `n_st=0` is the unit-weight distillation limit. For `n_st>0`, use the global
   high-mode dimension and weights in the unnumbered definition immediately
   after arXiv:2505.01719v2 Eq. (1), with tuple classifications in Eqs. (3)
   and (37).
3. `local_spatial_offset` is `(z0,y0,x0)`, momentum is `(px,py,pz)`, and the
   phase is `exp(-2*pi*i*(px*x/Gx+py*y/Gy+pz*z/Gz))`.
4. On more than one spatial rank, require each rank's global shape and offset,
   use global `3V3`, and call buffer `Allreduce` only across pieces of the same
   global time slices. Set `cuda_aware_mpi=True` for CuPy only after target
   runtime verification. The opt-in must be a Python/NumPy Boolean and must be
   checked before any elemental contraction. Do not use a world reduction
   when time is decomposed.
5. The bundled array code constructs kernels and callbacks; it does not create
   stochastic noise, perform low-mode projection/dilution, or load
   configurations. The scheduler-free E2E validator defines a contract-only
   metadata schema for hash-bound LapH/perambulator/exact-reference artifacts,
   but does not create those artifacts or define a runtime output format.
   Generate the numerical producer and output steps explicitly in the driver.
6. Reject Boolean mode provenance in built-in, `UserList`, custom non-string
   `Sequence`, and object-array inputs before numerical coercion. Correlator
   `t_source` must be a non-Boolean integer before any array access.
7. A link-phased quark source and a blended Laplacian/stochastic basis are
   different artifact types. Record separate IDs and ownership; composition
   does not imply a direct field-to-`eigvecs` handoff.

References: arXiv:2505.01719v2 for blending; arXiv:2009.10691v1 for the optional
momentum-smeared distillation context. Real QUDA/MPI production remains to be
validated on the target cluster.
