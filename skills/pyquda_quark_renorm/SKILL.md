---
name: pyquda_quark_renorm
description: >
  Review or adapt quark-renormalization postprocessing: legacy RI/MOM,
  RI-prime/MOM and RI/SMOM-to-MS-bar conversion/running tables, legacy
  diagonal-error host fits, and NumPy/CuPy spin-color inversion or amputation
  kernels. Trigger
  on MS-bar conversion, anomalous-dimension running, ZS/ZT matching, or
  Def_quark_renorm. For raw gauge-fixed propagator/vertex projection and Zq,
  use pyquda_ri_renorm instead. Do not silently trust an unmapped coefficient
  table or use this skill for gauge loading and QUDA inversion.
---

# Quark renormalization postprocessing

## Progressive disclosure

Read `USER_GUIDE.md` only when the user asks for principles, formulas,
derivations, natural-language examples, a complete workflow, or validation
boundaries. For ordinary API generation or review, use this file plus
`reference/API.md` and the bundled script.

Read `reference/PHYSICS_CONTRACT.md` before using a conversion, running,
Padé, scalar-inverse, tensor, fit, or uncertainty-propagation result.
Read `reference/SOURCE_VERIFICATION.md` before treating a coefficient as
source-mapped: it distinguishes exact primary-PDF reductions from candidate
literature searches and `UNVERIFIED_LEGACY` tables.

Read `VALIDATION.md` when the user asks for the current release label,
explicit-only gate, direct evidence, blockers, or reference-reproduction path.

Read `reference/API.md`. `scripts/Def_quark_renorm.py` is a legacy mixed
module; heavy fit packages are imported lazily, while the spin-color kernels
can use NumPy or CuPy. The companion `Def_qcd_analysis.py` provides only
jackknife resampling and is not an independent skill.

## Mandatory named raw-to-continuum handoff

When a request starts from gauge-fixed off-shell propagators or bilinear
Green functions, explicitly name both stage owners:

1. `pyquda_ri_renorm` owns independent incoming/outgoing-leg amputation,
   projector selection/application, the `Zq` definition, and raw `Z` factors
   with their samples/covariance and custom scheme identity.
2. `pyquda_quark_renorm` owns only the later source-mapped continuum
   conversion and running, after the exact coefficient equation, scheme,
   gauge, expansion variable, flavor count, scales, and uncertainty rule have
   passed the provenance gate.

Join the stages with an immutable manifest carrying input hashes, leg and
momentum ownership, gauge/projector/`Zq` conventions, raw samples/covariance,
scheme identity, scales, and coefficient provenance. Local inversion or
legacy amputation helpers in this skill do not transfer ownership of the raw
NPR stage and do not authorize selection of an unmapped conversion table.

## Function index

- Array kernels: `inverse_propagator`, `adj`, `Lambda_O_con`, `Lambda_O_dis`
- Legacy matching: `alpha_s`, named `*_conversion_*` functions, Padé helpers
- Running/systematics: `anomalous_dimension`, `scale_running`, `matching_systematic_error`
- Host analysis: `jackknife_resampling`, `read_data`, `ma_fit`, `a2p2_fit`, `ratio_fit`

## pyquda_quark_renorm.inverse_propagator

Input layout is `(...,sink_spin,source_spin,sink_color,source_color)` and the
last four extents are `(4,4,3,3)`. The backend is inferred from the array.

## pyquda_quark_renorm.adj

Compute `gamma5 * S.dagger() * gamma5`; the dagger exchanges both spin and
color source/sink axes. Color-diagonal tests are insufficient.

## pyquda_quark_renorm.Lambda_O_con

Amputate a connected Green function after optional jackknife resampling.
Gauge fixing, external-leg momentum definitions, and source/sink momentum
ownership remain caller contracts.

## pyquda_quark_renorm.Lambda_O_dis

The input current is a raw disconnected loop. The caller must state whether
vacuum subtraction has already been performed and preserve configuration
correlations through the jackknife.

## Mandatory provenance gate

The bundled conversion module contains many historical numerical tables. The
tensor RI-prime/MOM third-order normalization is now mapped to Gracey,
arXiv:hep-ph/0304113v1, Eq. (4.11), but this does not promote the neighboring
tables. Do not change a coefficient by pattern matching, and do not call the
tables production-verified until every public series is mapped to an original
equation and independently tested. General references are
arXiv:hep-ph/9910332 for RI/RI-prime matching and arXiv:0901.2599 for RI/SMOM;
they do not automatically validate every number in this file.

`matching_systematic_error` currently supports only its legacy
`scale0=2.0, Lambda=0.332, nf=3` table and rejects other values instead of
silently ignoring them. Fit code must expose covariance, priors, momentum
definition, window, and systematic flags.
Reject Boolean numerical inputs in built-in, `UserList`, custom non-string
`Sequence`, and object-array inputs before float conversion.
`read_data(pade_flag=0)` must not evaluate any Padé factor.
`pade_matching_factor.scale0` is a deprecated validated compatibility no-op;
do not evaluate discarded tensor/running branches for its scalar-only return.
