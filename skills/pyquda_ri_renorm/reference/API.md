# pyquda_ri_renorm API

Fields end in `(sink_spin,source_spin,sink_color,source_color)=(4,4,3,3)`.
All arrays in one call must use the same NumPy/CuPy backend.
Every standalone momentum is one finite real four-vector and every Dirac
matrix/projector/tree vertex is numerical, finite, and exact `(4,4)`;
`NaN`, `Inf`, object/string matrices and implicit batch broadcasting are not
part of the API.

## Contents

- Kinematics and result containers
- Spin-color inversion and amputation
- Explicit projector families
- Raw bilinear factors

## pyquda_ri_renorm.RIKinematics

Immutable output of `validate_kinematics`; records `omega`, a safe
kinematic/projector-family label, and momentum definition. This is not a full
continuum scheme identity because it does not yet include the Zq definition.
Although callers should construct it through `validate_kinematics`,
`compute_ri_constants` defensively revalidates every field of a manually built
record. In particular, complex/Boolean `q` and complex/Boolean/nonfinite `mu2`
fail before any propagator, vertex, or projector computation.

- Returns: data container.
- Example: `kin = validate_kinematics(...)`.

## pyquda_ri_renorm.RIConstants

Immutable container for `Zq,ZA,ZV,ZS,ZP,ZT`, projected vertices, validated
`kinematics`, canonical `zq_definition`, full custom `scheme_identity`, and
`continuum_matching_status`.

- Returns: data container from `compute_ri_constants`.
- Example: `zs = result.ZS`.

## pyquda_ri_renorm.validate_kinematics

`validate_kinematics(p_in,p_out,omega,rtol=1e-8,atol=1e-12,*,
projector_scheme=None,momentum_definition=None)` checks equal virtualities and
`q^2=omega*mu^2`. Nonexceptional calls require `gamma_mu` or `qslash`.
Exceptional propagator-Zq kinematics are labeled `RIprime/MOM`; symmetric and
generalized labels deliberately remain kinematic/projector-family labels.
Each four-vector rejects Boolean provenance in built-in, `UserList`, custom
non-string `Sequence`, and object-array inputs before NumPy dtype inference; a
caller-prepared float ndarray has already lost that provenance.

- Returns: `RIKinematics`.
- Example: `kin = validate_kinematics(pin,pout,1,projector_scheme='qslash',momentum_definition='kinetic_sin_ap')`.
- References: arXiv:0901.2599; arXiv:2310.00814v2 Appendix B.1 ("Vector
  normalization and Zq"), PDF pp. 13-14. The latter directly supports
  propagator-Zq in Eq. (32) and independent-leg amputation in Eq. (34), not
  the custom projector family; its internal cross-reference is Sec. IV.B.1.

## pyquda_ri_renorm.inverse_propagator

`inverse_propagator(propagator)` flattens spin-color to 12x12, performs a
batched inverse, and restores axes. Input and output have exact
`(...,4,4,3,3)` trailing layout.

- Returns: same-shape/backend inverse.
- Example: `Sinv = inverse_propagator(S)`.

## pyquda_ri_renorm.amputate_vertex

`amputate_vertex(propagator_out,green_function,propagator_in)` computes
`S_out^{-1} G S_in^{-1}` with independent external legs.
All three arrays must have exact equal shapes and one backend; `(N,...)` with
`(1,...)` is rejected rather than silently broadcast.

- Returns: same-shape/backend amputated vertex.
- Example: `Lambda = amputate_vertex(Sout,G,Sin)`.

## pyquda_ri_renorm.slash

`slash(momentum,gammas)` forms the supplied four-component slash. It does not
convert integer modes to `p`, `sin(ap)`, or improved momentum.
The momentum must be finite and real; all four gammas must be exact `(4,4)`
matrices on one backend.

- Returns: spin matrix.
- Example: `pslash = slash(kin.p_in,gammas)`.

## pyquda_ri_renorm.project_quark_field

`project_quark_field(inverse_prop,momentum,gammas)` implements the propagator
projector `-i Tr[slash(p)S^{-1}]/(12p^2)`.
The field ends in `(4,4,3,3)`, momentum is finite and nonzero, and the gamma
family obeys the exact Dirac-matrix/backend contract.

- Returns: backend scalar/batch `Zq` projection.
- Example: `zq = project_quark_field(Sinv,p,gammas)`.

## pyquda_ri_renorm.project_vertex

`project_vertex(vertex,projector,tree_vertex)` determines tree normalization
from the supplied matrices.
The vertex ends in `(4,4,3,3)` and both spin matrices are exact `(4,4)` on the
same backend; zero tree normalization fails closed.

- Returns: projected scalar/batch array.
- Example: `lp = project_vertex(Lp,g5,g5)`.
- CuPy boundary: checking the tree denominator transfers and synchronizes one
  Boolean scalar explicitly; no volume array is copied to host.

## pyquda_ri_renorm.project_multiplet

`project_multiplet(vertices,projectors,tree_vertices)` combines aligned V, A,
or T components under one tree normalization. All sequences are nonempty and
equal length; vertices have one exact shape and all projectors/tree matrices
are exact `(4,4)` on one backend.

- Returns: projected scalar/batch array.
- Example: `lv = project_multiplet(Lv,Pv,gammas)`.
- CuPy boundary: the shared-denominator zero check has the same explicit
  one-Boolean-scalar synchronization as `project_vertex`.

## pyquda_ri_renorm.compute_ri_constants

`compute_ri_constants(kinematics,propagator_inverse_in,
propagator_inverse_out,vertices,gammas,gamma5,*,zq_definition='propagator')`.
`vertices` must contain `S,P,V[4],A[4],T[6]`.
Both inverse propagators and every vertex must have exact equal batch/field
shapes, and the complete gamma family must be exact `(4,4)` on one backend.
The supplied record is reconstructed from validated `p_in`, `p_out`, `omega`,
projector family, and momentum definition; its stored `q`, `mu2`, and `scheme`
must agree exactly within the documented numerical tolerance.

For the qslash family, the exact longitudinal kernels are
`P_V,mu = q_mu * qslash` and `P_A,mu = q_mu * qslash * gamma5`; the axial
gamma order matches the tree vertex `gamma_mu * gamma5`.

`zq_definition` selects `incoming`, `outgoing`, `arithmetic_mean`, or
`geometric_mean`. The arithmetic choice is
`(Zq_in+Zq_out)/2`; the geometric choice is
`sqrt(Zq_in*Zq_out)` on the backend principal branch. The deprecated
`propagator` alias maps to `arithmetic_mean` and exists only to preserve old
calls. Complex noisy estimates may cross the principal branch cut; callers
must not silently take `real`, `abs`, or select another branch. New code should
select a named convention explicitly.

- Returns: an `RIConstants` record whose `scheme_identity` ends in
  `__custom_unmatched` and whose `continuum_matching_status` is
  `unmapped_fail_closed`.
- Example: `Z = compute_ri_constants(kin,Sinv,Soutv,vertices,gammas,g5,zq_definition="incoming")`.
- Legacy compatibility: omitting `zq_definition` still selects the deprecated
  `propagator -> arithmetic_mean` alias.
- Boundary: vector-vertex Zq is not implemented. Do not infer standard
  `RI/SMOM_gamma_mu` or `RI/SMOM_qslash` matching from `omega` and projector
  family alone. The module performs no gauge fixing, solve, MPI reduction,
  fit, or MS-bar conversion.

## Executable example registry

The quality contract maps every public symbol below to one real AST-direct
local test. CPU/stub/structural/rejection examples support at most E1; they
do not establish GPU, MPI, QUDA, runtime, interacting-physics, or production
evidence.

| Public symbol | Example ID | Test ID | Mode |
|---|---|---|---|
| `RIConstants` | `API-EX-053` | `test_ri_public_records_and_projection_invalid_domains_fail_closed` | `cpu-structural` |
| `RIKinematics` | `API-EX-053` | `test_ri_public_records_and_projection_invalid_domains_fail_closed` | `cpu-structural` |
| `amputate_vertex` | `API-EX-011` | `test_critical_physics_mutants_are_distinguishable` | `cpu-synthetic` |
| `compute_ri_constants` | `API-EX-057` | `test_ri_tree_level_normalization_for_all_schemes` | `cpu-synthetic` |
| `inverse_propagator` | `API-EX-002` | `test_all_eight_skills_match_independent_synthetic_golden` | `cpu-synthetic` |
| `project_multiplet` | `API-EX-054` | `test_ri_public_records_and_projection_invalid_domains_fail_closed` | `rejection-only-e1` |
| `project_quark_field` | `API-EX-054` | `test_ri_public_records_and_projection_invalid_domains_fail_closed` | `rejection-only-e1` |
| `project_vertex` | `API-EX-056` | `test_ri_rejects_nonfinite_momenta_and_non_dirac_matrices` | `rejection-only-e1` |
| `slash` | `API-EX-055` | `test_ri_qslash_axial_projector_follows_tree_vertex_order` | `cpu-synthetic` |
| `validate_kinematics` | `API-EX-057` | `test_ri_tree_level_normalization_for_all_schemes` | `cpu-synthetic` |
