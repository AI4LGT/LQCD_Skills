---
name: pyquda_gluon_renorm
description: >
  Generate or review PyQUDA/CuPy bare pure-gauge observables: plaquettes,
  link-centered A_mu, clover F_mu_nu and dual tensors, F-W-F operators,
  gluonic EMT candidates, gauge-dependent Chern-Simons/topological currents,
  Fourier transforms, and momentum masks. Trigger on gluon renormalization,
  gluon field-strength, gluon EMT, F-tilde-F, or K_mu. Do not use this skill as
  proof that a bare observable is renormalized, and do not use for quark RI
  vertices or propagator inversion.
---

# Bare gluon and gauge-observable kernels

## Progressive disclosure

Read `USER_GUIDE.md` only when the user asks for principles, formulas,
derivations, natural-language examples, a complete workflow, or validation
boundaries. For ordinary API generation or review, use this file plus
`reference/API.md` and the bundled script.

Read `reference/PHYSICS_CONTRACT.md` before interpreting lattice units,
gauge dependence, EMT/current candidates, mixing, or renormalization.

Read `VALIDATION.md` when the user asks for the current release label, direct
evidence, blockers, local verification commands, or the E2E promotion path.

Read `reference/API.md`, then inspect
`scripts/Def_gluon_renorm.py` for the exact contraction. The file contains
bare observable kernels; the historical skill name does not supply a complete
renormalization prescription.

## Function index

- Gauge tensors: `g_0`, `plaq_munu`, `A_mu`, `F_munu`, `F_and_tildeF`
- Wilson-line products: `FW_FW`, `FW_tildeFW`, `FF`
- Fourier kernels: `FT_Phase`, `FT_Gauge_1mom`, `FT_Prop_1mom`, `FFT_Gauge_Allmom_MPI`
- EMT/topology: `gaugeEMT_*`, `ExA_t`, `Topological_current_Kmu*`
- Setup: `generate_pselect_mask`

## pyquda_gluon_renorm.F_munu

Construct a clover field-strength tensor with directions `0,1,2,3=x,y,z,t`.
The array storage remains checkerboard `e,t,z,y,xh` before explicit conversion.

## pyquda_gluon_renorm.FFT_Gauge_Allmom_MPI

This is a single-rank CuPy FFT despite its historical name. It rejects a
nontrivial process grid and applies the direction-specific half-link factor
`exp(-i p_mu/2)` to `A_mu`.

## pyquda_gluon_renorm.FW_FW

Use device Wilson-line shifts and reduce only the time series. Time-decomposed
output (`Gt>1`) is rejected because a world reduction would mix different
global time slices.

## pyquda_gluon_renorm.Topological_current_Kmu

Treat `A_mu`, `E x A`, and Chern-Simons `K_mu` as gauge dependent. Require the
gauge-fixing functional, stopping tolerance, residual gauge convention, and
Gribov-copy treatment before interpreting them physically.

## Required physical contract

1. `g_0` is selected by gauge-action normalization, not by the fermion action.
   The helper supports `wilson` and explicitly named `legacy_tadpole_10`; for
   any other action pass a separately verified coupling. Omit `u_0` for
   `wilson`; supply it explicitly for `legacy_tadpole_10`. The ambiguous
   positional call `g_0(beta,u_0)` must fail closed.
2. Distinguish gauge-invariant `F W F` products from gauge-fixed `A_mu/K_mu`,
   and distinguish bare fields from renormalized operators.
3. For every time-resolved MPI observable, state whether values are local in
   global time, reduced over spatial ranks, or globally assembled.
4. `generate_pselect_mask` allocates global dense four-dimensional arrays and
   is setup-only; estimate host memory before large-volume use.
5. Do not route `Topological_current_Kmu_t` as a full four-component current
   unless the requested component contract has been checked in the source.
6. Validate Wilson-line direction, length, orientation, Lorentz indices,
   coupling, explicit/gauge lattice signatures, and time decomposition before
   constructing `F_munu`; invalid controls or mismatched global/local extents,
   process grids, and rank coordinates must not reach `gauge.loop`.

## Mandatory completion gate

Do not complete an interacting, final, or renormalized gluon-observable claim
with only a generic request for an "independent oracle". Require a separate
`clover_or_reference` artifact with all of the following literal fields and
checks:

- `kind=independent_clover|supplied_reference` and `independent=true`; the
  reference must not be produced by the same contraction or control flow as
  the observable under test;
- a stable `source`, its `checksum` as a SHA-256 digest, and component-resolved
  complex reference values that preserve both real and imaginary parts;
- a component-wise complex comparison of the produced bare observable against
  those reference values, with declared absolute and relative tolerance,
  per-component differences, and an explicit pass/fail result. The comparison
  must pass before completion.

This independent-reference gate validates only the tested bare contraction; it
does not supply operator normalization or renormalization. Keep `bare`,
`gauge_fixed`, and `renormalized` as separate artifacts. A renormalized result
additionally requires the operator definition and normalization, the complete
mixing basis/matrix and covariance, scheme and scale, ensemble provenance, and
continuum matching/running equations with sources. If any item is absent,
return the available bare value with the renormalized value unset and name the
blocker. This source-level contract is not runtime evidence and does not by
itself promote a claim to E2, E3, E4, or E5.

Method references: arXiv:hep-lat/0203008 for improved/clover field-strength
construction; arXiv:1310.4263 for the gauge-dependent gluon-spin/Coulomb-gauge
context. The exact EMT, Chern-Simons, and Wilson-line normalization in this
legacy file still requires equation-by-equation provenance before production.
