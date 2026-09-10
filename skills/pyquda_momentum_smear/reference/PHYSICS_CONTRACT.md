# Momentum-smearing and conjugation physics contract

## Status and evidence ceiling

- TODO: `SPEC-MOM-01`
- Status: `PASS_LOCAL_CONJUGATION_CONTRACT`
- Highest supported evidence: E1 for link phases, metadata and synthetic
  source/sequential algebra.
- Runtime/interacting ceiling: E0; no CUDA/QUDA/MPI correlator was run.

The contract separates three momenta that must never be silently identified:

$$
k_{\rm smear},\qquad P_i,P_f,\qquad q=P_f-P_i.
$$

$k_{\rm smear}$ controls the center of a quark smearing wave packet. $P_i$ and
$P_f$ are exact hadron Fourier momenta. $q$ belongs to the insertion. A good
choice such as $k\approx P/2$ or $P/3$ is an empirical tuning hypothesis, not
an API identity.

## Link-phased Wuppertal kernel

For each spatial direction $j=x,y,z$, the code replaces

$$
U_j(x)\longrightarrow
e^{+i\kappa_j}U_j(x),\qquad
\kappa_j=\frac{2\pi k_j}{L_j^{\rm global}},
$$

leaves the temporal link unchanged, and then applies the ordinary PyQUDA
Gaussian/Wuppertal iteration. This is the implementation form of
arXiv:1602.05525v3 Eq. (24), whose forward hop has $e^{+ik\cdot\hat j}$ and
backward hop has the conjugate phase. Multiplying an already smeared field by
$e^{ikx}$ is a different operation and is forbidden.

PyQUDA's wrapper uses

$$
\alpha=\left(\frac{4N}{\rho^2}-6\right)^{-1}>0.
$$

The current API validates this domain but does not assert that $(\rho,N)$ is
optimal. Phased gauge artifacts are valid only for the same gauge revision,
global/local extents, process grid, precision and boundary conditions.

## Complete data flow and signs

With PyQUDA `MomentumPhase.getPhase(P,x0)=exp(+iP.(x-x0))`, the local contract
is:

```text
point/source field at x0
  -> source link phase +k_i (meson spectator) / -k_i (meson active)
     or +k_i for each explicitly assigned baryon line
  -> source Wuppertal kernel
  -> D^{-1}: source-smeared forward propagator
  -> optional S-to-S sink endpoint with explicit +k_f/-k_f line ownership
  -> physical sink projection for +P_f: exp[-i P_f.(x-x0)]
  -> sequential-source candidate: exp[+i P_f.(x-x0)]
  -> later, caller-proved dagger/conjugation of that sequential line
  -> current insertion phase carrying q=P_f-P_i
```

The positive sequential phase is not context-free. `fourier_phase_pair`
returns both conjugate arrays, while `build_sequential_source` only multiplies
the positive candidate and selects a sink time. Neither it nor PyQUDA
`source.sequential12` performs the later dagger. A workflow that does not
explicitly prove the downstream conjugation must fail closed rather than use
the candidate.

The local meson source convention uses opposite modes on spectator and active
quark lines, consistent with the quark/antiquark relation in
arXiv:1602.05525 Eqs. (26)-(28). Degenerate baryons may reuse one $+k$ line;
unequal flavors or masses require independent line records. These line-level
choices do not replace the hadron Fourier projection.

## S-to-P, S-to-S and sequential ownership

- S-to-P applies source smearing only and rejects a supplied `sink_k_mode`.
- S-to-S applies exactly one source and one sink endpoint kernel and records
  both modes.
- Compatibility aliases have identical semantics and cannot loosen these
  gates.
- `build_sequential_source` may apply the active S-to-S sink kernel. A later
  `apply_momentum_smear_seqprop` call must then use
  `already_smeared=True,sink_k_mode=None`.
- If the sequential source was not yet smeared, the inversion wrapper requires
  `already_smeared=False` and one explicit mode. The helper applies it exactly
  once.

This one-smear record prevents both zero and double sink smearing.

## Coordinates, axes and boundary conditions

`x_src` is global `(x,y,z,t)`. PyQUDA lattice arrays begin with checkerboard
`(e,t,z,y,xh,...)`, $e=2$. Integer Fourier modes are defined with global
extents; smearing modes may be fractional. Rank-local phases must include the
global coordinate offset. Anti-periodic temporal boundary conditions belong to
the Dirac propagator/source workflow; the spatial smearing kernel itself is
time-local and does not redefine them.

## Result provenance

Every returned line must retain source/sink mode, physical $P_i/P_f$, source
coordinate, gauge hash/revision, lattice decomposition, action/mass/boundary,
precision, solver parameters and artifact ownership. For runtime evidence also
record setup/solve/total time, iterations, QUDA residual and independently
computed true residual for every RHS.

## Verification boundary

Local tests independently verify link signs, global extents, the conjugate
phase pair, sequential phase placement, aliases and one-smear rejection. They
do not prove the full sign of an interacting two- or three-point contraction.
That promotion requires `E2E-MOM-01`, including sign-flip controls on the
complex correlator.

## Sources

- arXiv:1602.05525v3, especially Eq. (24) and Eqs. (26)-(28).
- Current read-only PyQUDA `MomentumPhase`, `gaussianSmear`,
  `invertPropagator` and `invertSequential` API snapshot.
- `scripts/Def_momentum_smear.py`: exact local ownership and metadata gates.

