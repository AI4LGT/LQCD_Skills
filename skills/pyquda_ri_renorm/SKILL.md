---
name: pyquda_ri_renorm
description: >
  Generate or review NumPy/CuPy array kernels for gauge-fixed off-shell
  RI-prime/MOM and nonexceptional bilinear kinematics, spin-color inversion,
  vertex amputation, explicit gamma_mu or qslash projectors, and raw
  propagator-Zq/ZA/ZV/ZS/ZP/ZT. Trigger on
  RI/SMOM projector, amputated vertex, momentum-source NPR, or raw Z factors.
  Do not use for MS-bar conversion, fitting, gauge loading, QUDA inversion, or
  claim that the bundled arrays form a complete production NPR pipeline.
---

# RI projection and amputation kernels

## Progressive disclosure

Read `USER_GUIDE.md` only when the user asks for principles, formulas,
derivations, natural-language examples, a complete workflow, or validation
boundaries. For ordinary API generation or review, use this file plus
`reference/API.md` and the bundled script.

Read `reference/PHYSICS_CONTRACT.md` before assigning a standard RI/SMOM
scheme identity, applying Ward identities, or dispatching continuum matching.

Read `VALIDATION.md` when the user asks for the current release label, direct
evidence, scheme/runtime blockers, local verification, or E2E promotion path.

Read `reference/API.md`, then import `scripts/Def_ri_renorm.py`. These kernels
start from momentum-space propagators and Green functions already produced by
a gauge-fixed PyQUDA workflow.

## Function index

- Contracts: `RIKinematics`, `RIConstants`, `validate_kinematics`
- Spin-color algebra: `inverse_propagator`, `amputate_vertex`, `slash`
- Projectors: `project_quark_field`, `project_vertex`, `project_multiplet`
- Raw factors: `compute_ri_constants`

## pyquda_ri_renorm.validate_kinematics

Require `p_in^2=p_out^2=mu^2` and `q^2=omega*mu^2`. The caller must label the
actual momentum components (`continuum_ap`, `kinetic_sin_ap`, improved, etc.).
For nonexceptional kinematics, explicitly choose `projector_scheme='gamma_mu'`
or `'qslash'`; `omega` alone does not define an RI/SMOM scheme.

## pyquda_ri_renorm.amputate_vertex

Compute `S_out^{-1} G_O S_in^{-1}` with independent external propagators in
`(...,4,4,3,3)` layout. Do not manufacture a general nonexceptional second
leg solely by gamma5 Hermiticity.

## pyquda_ri_renorm.compute_ri_constants

Require `S`, `P`, four `V`, four `A`, and six `T(mu<nu)` amputated vertices.
The qslash axial projector uses `q_mu * qslash * gamma5`, matching the tree
vertex order `gamma_mu * gamma5`.
Choose `zq_definition='incoming'`, `'outgoing'`, `'arithmetic_mean'`, or
`'geometric_mean'`; the last uses the backend principal square-root branch.
Complex statistical fluctuations can cross its branch cut; never silently
replace the result by `real`, `abs`, or another branch. The deprecated
`'propagator'` alias preserves the old arithmetic mean. A vector-vertex Zq is
not silently mixed in. Returns raw tree-normalized factors only. The returned
`RIConstants` records a full custom `scheme_identity` and
`continuum_matching_status='unmapped_fail_closed'`; do not dispatch a standard
RI/SMOM matching factor from the kinematic label alone.

## Required physical contract

Kinematic four-vectors reject Boolean provenance in built-in, `UserList`,
custom non-string `Sequence`, and object-array inputs before NumPy coercion.
Caller-created float arrays have already lost that provenance.

1. Record gauge-fixing functional/tolerance, gamma basis, color normalization,
   incoming/outgoing momentum sign, and lattice momentum definition.
2. Treat `symmetric_nonexceptional_gamma_mu/qslash` as projector-family
   labels only. Because this module uses propagator Zq and implements no
   vector-vertex Zq, its nonexceptional outputs are custom/unmatched rather
   than automatically standard `RI/SMOM_gamma_mu` or `RI/SMOM_qslash`.
3. arXiv:2310.00814v2 Appendix B.1 ("Vector normalization and Zq"), PDF
   pp. 13-14, supports propagator-Zq in Eq. (32) and independent-leg
   amputation in Eq. (34); the paper cross-references this appendix as Sec.
   IV.B.1. It does not validate the custom
   qslash or leg-averaging choices.
   arXiv:0901.2599 is the primary RI/SMOM scheme reference. Verify the exact
   projector convention before continuum matching.
4. This module performs no momentum-source construction, QUDA solve, MPI
   reduction, fit, extrapolation, or MS-bar conversion.

Tree-level CPU tests are available; interacting gauge-fixed, GPU, and MPI
validation remains outstanding.
