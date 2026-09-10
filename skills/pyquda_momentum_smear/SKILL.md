---
name: pyquda_momentum_smear
description: >
  Generate or review PyQUDA link-phased Wuppertal momentum-smearing code for
  quark sources, sink propagators, and sequential sources, including S-to-P,
  S-to-S, nonforward source/sink modes, MRHS, and Fourier-phase signs. Trigger
  on momentum smearing, phased spatial links, boosted-hadron source, or
  arXiv:1602.05525. Do not use for distillation/blending, fitting, plotting,
  effective masses, or ensemble analysis.
---

# PyQUDA momentum smearing

## Progressive disclosure

Read `USER_GUIDE.md` only when the user asks for principles, formulas,
derivations, natural-language examples, a complete workflow, or validation
boundaries. For ordinary API generation or review, use this file plus
`reference/API.md` and the bundled script.

Read `reference/PHYSICS_CONTRACT.md` for the full source/sink/Fourier/
sequential-dagger sign chain and before composing a nonforward workflow.

Read `VALIDATION.md` when the user asks for the current release label, direct
evidence, runtime blockers, local verification, or E2E promotion path.

Read `reference/API.md`, then import `scripts/Def_momentum_smear.py`. Generate
a compact per-configuration driver and call the bundled kernels directly.

## Function index

- Link/kernel: `phase_spatial_links`, `momentum_smear_kernel`
- One line: `apply_momentum_smear_source`, `apply_momentum_smear_sink`, `momentum_smear_propagator`
- Hadrons: `momentum_smear_meson`, `momentum_smear_baryon_degenerate`
- Sequential: `fourier_phase_pair`, `build_sequential_source`, `apply_momentum_smear_seqprop`

## pyquda_momentum_smear.momentum_smear_kernel

Apply `U_j(x) -> exp(+2*pi*i*k_j/L_j) U_j(x)` before every Wuppertal step.
`k_j` may be fractional. PyQUDA uses
`alpha=1/(4*n_steps/rho^2-6)`, so this wrapper rejects nonpositive alpha.

## pyquda_momentum_smear.momentum_smear_meson

Construct source `+k/-k` lines and optional independent sink `+k_f/-k_f`
lines. The exact hadron momentum still comes from the final Fourier
projection; the smearing mode only controls overlap.
`sink_k_mode` is meaningful only for S-to-S. S-to-P must omit it and fails
before point-source construction or inversion if it is supplied.

## pyquda_momentum_smear.momentum_smear_baryon_degenerate

Return one reusable line for a degenerate baryon with equal smearing modes.
For unequal masses/flavors, construct and document independent `k_i` per line.

## pyquda_momentum_smear.fourier_phase_pair

PyQUDA `MomentumPhase.getPhase(P)` is `exp(+iP.(x-x0))`. This function returns
the physical `+P` sink phase `exp(-iP.(x-x0))` and the conjugate sequential
phase candidate `exp(+iP.(x-x0))`. Use the positive candidate only after the
downstream contraction explicitly establishes a dagger/conjugation of that
line. Neither this helper nor PyQUDA `source.sequential12` performs that
dagger; the latter only selects the sink time slice.

`apply_momentum_smear_seqprop` has a fail-closed one-smear contract. Pass
`already_smeared=True,sink_k_mode=None` for a source already completed by
`build_sequential_source`; pass `already_smeared=False` plus an explicit
`sink_k_mode` only when this call must apply the required smearing once.
Validate every pure mode, coordinate, time, layout, solver option, smearing
domain, and one-smear flag before PyQUDA imports, source construction,
smearing, or inversion. In particular, canonicalize an optional sink mode
before spending a source solve.

## Required physical contract

1. Keep source/sink smearing modes separate from physical Fourier momentum and
   insertion transfer `q=P_f-P_i`.
2. Pass `mrhs` and `restart` explicitly when comparing solver performance;
   report setup, solve, and total time plus true residuals.
3. Verify global/local lattice extents, MPI grid, precision, boundary
   conditions, and gauge revision before reusing a phased gauge.
4. Primary reference: arXiv:1602.05525. arXiv:2009.10691 is supplementary for
   momentum-smeared distillation, not the primary citation for this wrapper.

No real CUDA/QUDA/MPI inversion was run in the local audit.

## Cross-skill artifact boundary

This skill returns quark source/propagator/sequential-line artifacts. It does
not construct the `(N,t,z,y,x,c)` Laplacian plus stochastic-complement basis
consumed by blending elementals. When both skills are explicitly requested,
keep their artifacts independent and join them only at a documented source
callback, solve, or observable-assembly boundary. A completed smeared line is
never automatically a blending `eigvecs` input.
