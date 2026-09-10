---
name: pyquda_helicity_renorm
description: >
  Review or cautiously adapt the legacy nf=3 helicity/EMT perturbative
  matching and two-operator RG code in Def_helicity_renorm.py. Trigger on
  helicity matching matrix, R12/R21, helicity renormalization, or legacy
  helicity running. Do not use for bare RI vertex projection, gauge-field
  production, or as a production coefficient source unless the user supplies
  the exact paper, operator basis, scheme, and equation mapping.
---

# Legacy helicity matching and running

## Progressive disclosure

Read `USER_GUIDE.md` only when the user asks for principles, formulas,
derivations, natural-language examples, a complete workflow, or validation
boundaries. For ordinary API generation or review, use this file plus
`reference/API.md` and the bundled script.

Read `reference/PHYSICS_CONTRACT.md` before using any built-in matching or
running coefficient. It records the unresolved basis/equation provenance and
the exact boundary of the literal legacy contract.
Read `reference/SOURCE_VERIFICATION.md` for the primary beta-function map and
the entry-by-entry fail-closed record for every unmapped matching/running
coefficient.

Read `VALIDATION.md` when the user asks for the current release label,
explicit-only gate, direct evidence, blockers, or reference-reproduction path.

Read `reference/API.md` before importing
`scripts/Def_helicity_renorm.py`. This skill is conservative: array algebra
and the one-loop matrix placement were audited, but the original external
file did not record provenance for every matching/anomalous-dimension number.

## Function index

- Coupling: `beta`, `alpha_s`
- Matching: `EMT_MatchingCoeff`, `Helicity_MatchingCoeff_tmp`, `Helicity_MatchingCoeff`, `Helicity_MatchingCoeff_array`
- Running: `Helicity_RunningFactorCalculator`, `Get_Helicity_Running`
- Optional host postprocessing: `momentum_interpolate`, `AA_subtract`

## pyquda_helicity_renorm.Helicity_MatchingCoeff

Return `(R11,R12,R21,R22)` with fixed row/column *tuple placement* only; the
physical operator basis is unverified. The unused `g_0` argument was removed.
Loop-one off-diagonal entries now follow the same placement as higher orders.
Four-loop output is a Padé estimate only when
`is_pade=1`; otherwise it reproduces the explicit three-loop truncation.

## pyquda_helicity_renorm.Helicity_RunningFactorCalculator

`run`/`run_array` are the public methods. The bundled legacy anomalous-
dimension table is restricted to `nf=3`; other flavor numbers are rejected.

## pyquda_helicity_renorm.momentum_interpolate

Sort unique references, reject extrapolation, and interpolate the two
bracketing means. With no covariance input, endpoint standard deviations are
propagated under an explicit independence assumption.
Reject Boolean provenance in built-in, `UserList`, custom non-string `Sequence`,
and object-array scale inputs before NumPy dtype inference; do not claim
provenance recovery for a float ndarray constructed by the caller earlier.

## Mandatory provenance gate

Before numerical production, require all of the following from the user or
an original source: operator basis order, RI scheme and gauge, coupling
normalization, flavor dependence, loop convention, and an equation/section
for every coefficient table. If any item is missing, return an audit or accept
caller-supplied coefficient matrices instead of presenting legacy decimals as
verified QCD results.

arXiv:1606.08659 supports the five-loop QCD beta-function context. It does not
by itself validate the helicity matching or anomalous-dimension tables in this
file. No GPU volume work belongs in this skill.
