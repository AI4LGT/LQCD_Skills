---
name: pyquda_hisq_recover
description: >
  Generate or review NumPy/CuPy code for staggered/HISQ-to-naive four-spin
  reconstruction in Wilson-like spin-color storage, including arbitrary
  global source coordinates, MPI-local parity, and optional host gather.
  Trigger on HISQ recovery, staggered-to-naive spin lift, Omega(x), or
  Wilson-like propagator storage. Do not describe the result as a
  Wilson-action propagator and do not use for Dirac inversion or fitting.
---

# HISQ/staggered four-spin reconstruction

## Progressive disclosure

Read `USER_GUIDE.md` only when the user asks for principles, formulas,
derivations, natural-language examples, a complete workflow, or validation
boundaries. For ordinary API generation or review, use this file plus
`reference/API.md` and the bundled script.

Read `reference/PHYSICS_CONTRACT.md` before accepting a producer, taste,
origin, boundary-condition, storage-layout, or MPI-ownership convention.

Read `VALIDATION.md` when the user asks for the current release label, direct
evidence, producer blockers, local verification, or E2E promotion path.

Read `reference/API.md` and import `scripts/Def_hisq_recover.py`.

## Function index

- `gamma_ops`: current PyQUDA DeGrand--Rossi mapping or caller-supplied basis
- `hisq_to_wilson_evengrid`: globally resident array reconstruction
- `hisq_to_wilson_evengrid_MPI`: MPI-local reconstruction with optional gather

## pyquda_hisq_recover.hisq_to_wilson_evengrid

Compute
`G_naive(x,x0)=Omega(x) G_chi(x,x0) Omega(x0)^dagger` and store the result as
`(t,z,y,x,sink_spin,source_spin,sink_color,source_color)`. A scalar source
argument remains backward compatible with `(t_source,0,0,0)`; pass the full
global `(t,z,y,x)` coordinate for a displaced source.

## pyquda_hisq_recover.hisq_to_wilson_evengrid_MPI

Use global coordinate parity on each local block. `gather=False` keeps the
field on its NumPy/CuPy backend. `gather=True` explicitly transfers a CuPy
field to host and returns the global field only on rank 0.

## Required physical contract

1. This is the staggered spin lift, not a conversion to a Wilson fermion
   action. “Wilson-like” describes storage only.
2. Coordinates and source coordinates use `(t,z,y,x)`; PyQUDA grid metadata is
   `(x,y,z,t)`. State every reorder.
3. The supplied `gamma_ops` must implement one documented Euclidean basis.
   The kernel casts them to the propagator dtype and preserves complex64 or
   complex128.
4. Verify the producer's staggered phases, lattice origin, taste/source
   convention, and boundary wrap signs. `evengrid` is a historical name for a
   full lexicographic grid, not an even-site subset.
5. For MPI, verify global parity offsets and rank-0 gather ownership. Prefer
   local contractions followed by compact reductions over a full-field gather.
   `gather` accepts only Python/NumPy Booleans and is validated before local
   reconstruction; never rely on string/integer truthiness.

Method references: Kawamoto and Smit, Nucl. Phys. B192 (1981) 100 for spin
diagonalization; arXiv:hep-lat/0610092 for the HISQ action context. These
references support the method class, not a target-cluster runtime claim.
