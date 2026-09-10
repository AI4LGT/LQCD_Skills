---
name: pyquda_enhanced_interpolator
description: >
  Generate or review array kernels for kinematically enhanced boosted-hadron
  interpolators: Euclidean gamma-plus/minus kernels, enhanced meson bilinears,
  nucleon source/sink diquark kernels, direct-minus-exchange contractions, and
  momentum projection. Trigger on enhanced interpolator, kinematic
  enhancement, gamma-plus nucleon, arXiv:2606.02447, or arXiv:2501.00729. Do
  not use for generic baryon spectroscopy, smearing, fitting, or plotting.
---

# Enhanced interpolator kernels

## Progressive disclosure

Read `USER_GUIDE.md` only when the user asks for principles, formulas,
derivations, natural-language examples, a complete workflow, or validation
boundaries. For ordinary API generation or review, use this file plus
`reference/API.md` and the bundled script.

Read `reference/PHYSICS_CONTRACT.md` before choosing a gamma basis, mapping a
source diquark, extending a paper operator, or claiming overlap/SNR gain.

Read `VALIDATION.md` when the user asks for the current release label, direct
evidence, blockers, local verification commands, or the E2E promotion path.

Read `reference/API.md` and import
`scripts/Def_enhanced_interpolator.py`. These are local NumPy/CuPy operator
and contraction kernels, not a complete PyQUDA inversion or I/O pipeline.

## Function index

- Contract: `EnhancedKernels`
- Gamma operations: `euclidean_adjoint`, `euclidean_source_conjugate`, `lightcone_gammas`
- Enhanced leg/kernel: `plus_quark_projector`, `enhanced_meson_kernel`, `enhanced_pion_kernel`, `enhanced_baryon_kernels`
- Contractions: `meson_contraction`, `baryon_contraction`, `momentum_project`

## pyquda_enhanced_interpolator.enhanced_baryon_kernels

Return both `diquark_sink` and `diquark_source_bar`. The source kernel uses
`gamma_t * D.conj() * gamma_t`; do not silently replace it by
`gamma_t * D.dagger() * gamma_t`.

## pyquda_enhanced_interpolator.baryon_contraction

Inputs end in `(sink_spin,source_spin,sink_color,source_color)` and the output
is the direct-minus-exchange local `uud` contraction. Use a small explicit
spin/color loop as an oracle when changing the einsum.

## pyquda_enhanced_interpolator.momentum_project

Inputs have exact checkerboard layout `(e,t,z,y,xh)`. The supplied phase must
already use global coordinates. `spatial_comm` is a same-global-timeslice
communicator, exact `e=2` parity extent, and mpi4py-style buffer `Allreduce`.
Set `cuda_aware_mpi=True` for CuPy only after target-runtime verification.

## Required physical contract

1. State the Euclidean gamma convention and boost direction before forming
   `gamma_±=(gamma_t ± i gamma_parallel)/sqrt(2)`.
2. The historical function name `plus_quark_projector` returns the paper's
   enhanced quark-component kernel; do not infer idempotence without checking
   the chosen normalization and gamma convention.
3. Use arXiv:2606.02447 for the nucleon construction and arXiv:2501.00729 for
   the meson extension. Do not cite the nucleon paper alone as proof of a
   generic meson operator, and do not treat arbitrary `bilinear_gamma` input
   as a paper-validated operator whitelist.
4. The bundled code neither constructs global Fourier phases nor performs
   configuration I/O. Keep source/sink kernels and MPI ownership explicit in
   the generated driver.

Only algebra/shape and CPU-stub evidence is available locally; the physical
operator and multi-rank pipeline still require target-environment validation.
