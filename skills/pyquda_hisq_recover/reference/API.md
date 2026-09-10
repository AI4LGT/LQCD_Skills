# pyquda_hisq_recover API

This module reconstructs a naive four-spin field from a staggered/HISQ
propagator. The result merely uses Wilson-like spin-color storage.

## Contents

- Default Euclidean gamma mapping
- Global-array reconstruction
- MPI-local reconstruction and optional gather

## pyquda_hisq_recover.gamma_ops

Dictionary keys are exactly `t,z,y,x,unit`; each value is a numerical, finite
`(4,4)` matrix. `NaN`, `Inf`, object/string matrices, missing/extra keys and
wrong shapes fail before reconstruction. Custom matrices may be supplied to
either function. In the currently inspected
PyQUDA checkout, `gamma.gamma(n)` uses `DeGrandRossiMatrix.matrix(n)`, and this
module maps `n=(8,4,2,1,0)` to `(t,z,y,x,unit)`. Treat that as a local API
contract, not a universal HISQ-file convention.

- Returns: configuration object, not a field.
- Example: `ops = dict(gamma_ops)`.

## pyquda_hisq_recover.hisq_to_wilson_evengrid

`hisq_to_wilson_evengrid(latt_info, hisq_prop, tsource, gamma_ops=gamma_ops)`.
`tsource` accepts a legacy scalar time, meaning `(t,0,0,0)`, or a full global
source `(t,z,y,x)`.

`evengrid` is a historical name: the input is the complete lexicographic grid,
not an even-site or checkerboard subset. `hisq_prop` must be a one-component
staggered/HISQ color propagator whose source, staggered-phase, origin, and
boundary-condition conventions match the supplied `Omega` construction. No
extra eta/taste/boundary phase is applied here.

- Input: complex `(GLt,GLz,GLy,GLx,3,3)` NumPy/CuPy array.
- Returns: same-dtype/backend `(GLt,GLz,GLy,GLx,4,4,3,3)` array.
- Example: `naive = hisq_to_wilson_evengrid(info,Gchi,(t0,z0,y0,x0))`.

## pyquda_hisq_recover.hisq_to_wilson_evengrid_MPI

`hisq_to_wilson_evengrid_MPI(latt_info, hisq_prop, tsource,
gamma_ops=gamma_ops, gather=True)` uses global offsets from `grid_coord`.

- `gather` must be a Python or NumPy Boolean. Strings, integers, and arbitrary
  truthy objects fail before lattice metadata access or local reconstruction.
- `gather=False` returns the local reconstructed NumPy/CuPy field.
- `gather=True` returns a global host field on rank 0 and `None` elsewhere.
- Root ownership is queried through `pyquda_utils.core.getMPIRank()` after
  `core.gatherLattice`. The pinned `LatticeInfo` also snapshots `mpi_rank`,
  but this helper deliberately treats the active communicator as the
  canonical ownership source.
- Returns: local field, root host field, or `None` according to that contract.
- Example: `local = hisq_to_wilson_evengrid_MPI(info,Gchi,x0,gather=False)`.

Reference: the formula is `Omega(x) G_chi(x,x0) Omega(x0)^dagger`. See Kawamoto and
Smit, Nucl. Phys. B192 (1981) 100; arXiv:hep-lat/0610092 supplies HISQ action
context. Verify the exact gamma basis against the producer of `G_chi`.
