# pyquda_blending API

Arrays may be NumPy or CuPy. Basis vectors use `(N,t,z,y,x,c)`; all public
momentum tuples are `(px,py,pz)`, while shapes/offsets are `(z,y,x)`.
Integer counts, extents, offsets, and mode indices reject Python/NumPy
Booleans instead of treating them as `0` or `1`.

## Contents

- Blending weights
- Link-phased optional smearing
- Global Fourier phases and elementals
- Perambulator orchestration and contractions
- Provenance and runtime boundary

## pyquda_blending.omega

`omega(n, volume_color_dim, n_ev, n_st)` implements the successive distinct
stochastic-label factor in the unnumbered definition immediately after
arXiv:2505.01719v2 Eq. (1).

- Returns: Python `float`.
- Example: `w0 = omega(0, 3*Lx*Ly*Lz, nev, nst)`.
- Domain: `n`, `volume_color_dim`, `n_ev`, and `n_st` are non-Boolean
  integers; the global high-mode dimension and `0 <= n < n_st` are validated.

## pyquda_blending.blending_tuple_weight

`blending_tuple_weight(indices, volume_color_dim, n_ev, n_st)` applies Eq. (3)
for mesons and Eq. (37) for baryons. Repeated high-mode labels count once.

- Returns: one tuple weight; `n_st=0` returns one for low-only tuples.
- Example: `w_ijk = blending_tuple_weight((i,j,k), 3*V3, nev, nst)`.
- Domain: every tuple index and both mode counts are validated before any
  indexing; only `0 <= index < n_ev+n_st` is legal.

## pyquda_blending.phase_spatial_links_internal

`phase_spatial_links_internal(base_gauge, k_mode)` copies the gauge and phases
only links `x,y,z`. It is self-contained and does not import the momentum skill.
Mixed raw/object mode sequences containing Boolean entries fail before float
conversion or gauge copying, including `UserList` and custom non-string
`collections.abc.Sequence` inputs rather than only built-in list/tuple.

- Returns: copied PyQUDA gauge.
- Example: `gk = phase_spatial_links_internal(gauge, (0.5,0,0))`.
- Reference: arXiv:2009.10691v1, Section II.A, Eq. (4), in the distillation
  context.

## pyquda_blending.momentum_smear_internal

`momentum_smear_internal(field, base_gauge, k_mode, rho, n_steps)` calls
PyQUDA `source.gaussianSmear` with the phased gauge.

- Returns: same-kind PyQUDA lattice field.
- Example: `eta_k = momentum_smear_internal(eta, gauge, k, 4.0, 40)`.
- Domain: `rho` must be a finite positive real value; `n_steps` must be a
  non-Boolean positive integer; positive PyQUDA alpha additionally requires
  `rho**2 < 2*n_steps/3`.
- Composition boundary: a generic momentum-smeared quark source or propagator
  is not an `(N,t,z,y,x,c)` blending basis. Do not feed it to
  `meson_elemental`/`baryon_elemental`. Applying this helper to an explicitly
  identified basis vector is a separate blending-owned construction whose
  input/output basis IDs must be recorded.

## pyquda_blending.spatial_fourier_phase

`spatial_fourier_phase(spatial_shape, momentum, array_module=np, *,
global_shape=None, local_offset=(0,0,0))` constructs
`exp(-2*pi*i*p.x_global/G)`.
Momentum entries must be finite real values; raw Boolean provenance is checked
before conversion in built-in, `UserList`, custom non-string `Sequence`, and
object-array inputs.

- Returns: backend array with local `(z,y,x)` shape.
- Example: `phase = spatial_fourier_phase(local_zyx,p,cp,global_shape=global_zyx,local_offset=offset_zyx)`.

## pyquda_blending.meson_elemental

`meson_elemental(eigvecs, momentum, n_ev, n_st, *,
global_spatial_shape=None, local_spatial_offset=None, spatial_comm=None,
cuda_aware_mpi=False)`. Color extent is exactly three. For more than one
spatial rank, both global shape and this rank's offset are mandatory, and
`spatial_comm` must connect pieces of the same global time slices. Reduction
uses mpi4py-style buffer `Allreduce`; set `cuda_aware_mpi=True` for CuPy only
after verifying the target MPI stack.
`cuda_aware_mpi` is a strict Python/NumPy Boolean opt-in; integers, strings,
and arbitrary truthy objects are rejected even when no reduction or a
single-rank early return would otherwise occur. This gate is the first public
entrypoint check, before shape access, phase allocation, or the contraction.
`n_ev` and `n_st` are validated as non-Boolean nonnegative integers before
the basis is sliced; their sum must be nonzero and cannot exceed the available
global color-spatial subspace.

- Returns: all-reduced `(t,N,N)` elemental with blended weights.
- Example: `phi = meson_elemental(v,p,nev,nst,global_spatial_shape=Gzyx,local_spatial_offset=o,spatial_comm=space_comm)`.
- Reference: arXiv:2505.01719v2 Eqs. (31)-(32).

## pyquda_blending.baryon_elemental

Same keywords and global-coordinate/buffer-collective contract as
`meson_elemental`; color extent must be three.

- Returns: all-reduced `(t,N,N,N)` epsilon-color elemental.
- Example: `phi3 = baryon_elemental(v,p,nev,nst,global_spatial_shape=Gzyx,local_spatial_offset=o,spatial_comm=space_comm)`.
- Reference: arXiv:2505.01719v2 Eq. (37) for the weights.

## pyquda_blending.generate_perambulator

`generate_perambulator(n_modes, n_spin, make_source, solve, project_sink)` calls
one source/solve/project callback per source mode and spin.
Both counts must be non-Boolean positive integers.

- Returns: `(t,sink_spin,source_spin,sink_mode,source_mode)` array.
- Example: `tau = generate_perambulator(N,4,make_source,solve,project_sink)`.
- Reference: arXiv:2505.01719v2 Eq. (4).
- Limit: callback orchestration is not an MRHS implementation.

The scheduler-free E2E validator (`scripts/e2e_validation.py`) requires the
frozen contract to bind a LapH eigenvector artifact (`laph.eigenvectors`) and
projected perambulator artifact (`perambulator`) to the same configuration,
with full SHA-256 digests, `(D,N)` basis metadata, eigenpair/orthonormality
residuals, the exact perambulator axis order, mode/spin counts, true-residual
metadata, seed IDs and dilution labels.  Both artifacts must be listed in
`io.input_artifacts`; a generic artifact ID/checksum pair is insufficient.

## pyquda_blending.meson_two_point

`meson_two_point(phi_source, phi_sink, perambulator, gamma_source_bar,
gamma_sink, gamma5, t_source)` uses gamma5 Hermiticity.

- Returns: backend `(t,)` connected correlator.
- Example: `c2 = meson_two_point(phi0,phit,tau,Gbar,G,g5,t0)`.
- Domain: `t_source` is a non-Boolean integer in the perambulator time range and
  is validated before array/backend access.
- Boundary: the full Wick, source-bar, Fourier, and overall-sign convention is
  this implementation's contract; the weight/elemental equations are not a
  blanket literature validation of the complete contraction.

For an E2E contract, `reference.kind` must be `exact_all_to_all`.  The
reference artifact is hash-bound and must declare an independent implementation,
contraction convention, observable/configuration IDs and separate input artifacts; it may
not consume the blended perambulator as its oracle.  Multiple distinct noise
seeds and ordered dilution labels are required and are checked against every
post-run solve record.  These checks establish metadata integrity only, never
GPU/MPI/QUDA execution or unbiasedness.

## pyquda_blending.baryon_two_point

`baryon_two_point(phi_source, phi_sink, tau_u1, tau_u2, tau_d,
spin_projector, diquark_sink, diquark_source_bar, t_source)` evaluates direct
minus exchange.

- Returns: backend `(t,)` `uud` correlator.
- Example: `c2 = baryon_two_point(phi0,phit,tu,tu2,td,P,D,Dbar,t0)`.
- Domain: `t_source` obeys the same pre-array non-Boolean integer gate.

Stochastic vectors, orthogonal projection, dilution, QUDA solves, MPI
communicator construction, and I/O are caller responsibilities. CPU tests do
not constitute target-cluster stochastic or multi-rank validation.
