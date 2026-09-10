# Producer-specific HISQ/staggered reconstruction contract

## Status and evidence ceiling

- TODO: `SPEC-HISQ-01`
- Status: `PARTIAL_PRODUCER_BLOCKED`
- Highest supported evidence: E1 for the generic explicit-gamma spin lift;
  E0 for producer compatibility.

No named producer adapter is currently bundled. Therefore the generic helper
must not be presented as automatically compatible with a MILC, Chroma, Grid or
other propagator solely because the input shape matches.

## Mathematical transformation

For global coordinates $x=(t,z,y,x)$, the local order is

$$
\Omega(x)=
\gamma_x^{x\bmod2}
\gamma_y^{y\bmod2}
\gamma_z^{z\bmod2}
\gamma_t^{t\bmod2},
$$

and the code constructs

$$
G_{\rm naive}(x,x_0)=
\Omega(x)G_\chi(x,x_0)\Omega^\dagger(x_0).
$$

Gamma matrices do not commute, so the displayed order is part of the API.
`gamma_ops` must contain exactly `x,y,z,t,unit`, use one documented Hermitian
Euclidean Clifford basis and match the producer. The output storage is

```text
(t,z,y,x,sink_spin,source_spin,sink_color,source_color)
```

with spin/color extents `(4,4,3,3)`. It is a naive four-spin lift in
Wilson-like storage, not a Wilson/clover-action propagator.

Kawamoto-Smit spin diagonalization supports this method class. HISQ
arXiv:hep-lat/0610092 establishes the action/taste context and the formal
naive-staggered relation, but neither source fixes an arbitrary file
producer's origin, taste, source or boundary conventions.

## Producer adapter schema

A production adapter must be versioned and provide all fields below before
calling the generic kernel:

```text
producer_name, producer_version, source_file_format, source_dataset
global_shape_tzyx, local_shape_tzyx, grid_size_xyzt, grid_coord_xyzt
input_layout, input_checkerboarding, rank_ownership
source_global_tzyx, source_normalization
gamma_basis_id, omega_gamma_order
staggered_eta_convention, taste_or_naive_component
lattice_origin_tzyx, coordinate_origin_base
boundary_signs_xyzt, temporal_wrap_already_applied
complex_dtype, gauge/configuration_id, checksum
```

The adapter, not the generic array shape, must establish that $G_\chi$ is the
one-component object expected by the formula. A taste-projected block, naive
spin block, checkerboard field or source with an extra phase needs a separate
adapter and conversion equation.

## Origin, source and boundary treatment

The source is a global `(t,z,y,x)` coordinate. The legacy scalar argument means
`(t_source,0,0,0)` only. MPI-local sink coordinates include the rank's global
offset before parity is taken.

The kernel inserts no additional staggered $\eta_\mu(x)$, taste phase,
anti-periodic temporal wrap sign or source normalization. Those factors must be
declared as already included or explicitly transformed by the producer
adapter. In particular, crossing a temporal boundary may change a propagator
sign even though $\Omega(x)$ depends only on coordinate parity; the two effects
must not be conflated.

`evengrid` is a historical function name. The expected input is a full
lexicographic grid, not an even-site subset or checkerboard half-volume.

## MPI and gather ownership

`LatticeInfo.size`, `grid_size` and `grid_coord` are `(x,y,z,t)`, while arrays
are `(t,z,y,x)`. Each rank reconstructs its local block from global coordinates.

- `gather=False`: preserves the NumPy/CuPy local field on every rank.
- `gather=True`: explicitly converts a CuPy volume field to host, calls
  PyQUDA `core.gatherLattice`, and returns the global array on rank 0 only.

This ownership is a contract, not verified multi-rank evidence. Production
analysis should usually contract locally and reduce compact observables rather
than gather the full spin-color field.

## Current blocker and fail-closed rule

The bundled function signature does not carry the producer metadata schema,
so it cannot itself detect a producer mismatch. Until a versioned adapter is
implemented, safe use requires an explicit external metadata record and a
caller-supplied `gamma_ops` mapping. Missing taste/origin/boundary/layout
metadata is a production blocker and must produce `BLOCKED_BY_PRODUCER_METADATA`
in the workflow, not a guessed reconstruction.

Promotion requires independent small-lattice oracles for a nonzero source,
nonzero rank offset, temporal boundary crossing and both complex64/complex128,
plus comparison with the producer's own spin-lift or an independent reference.

## Sources

- N. Kawamoto and J. Smit, Nucl. Phys. B192 (1981) 100,
  DOI 10.1016/0550-3213(81)90196-6: spin diagonalization method.
- arXiv:hep-lat/0610092: HISQ and naive/staggered taste context.
- `scripts/Def_hisq_recover.py`: exact local gamma/order/layout behavior.

