---
name: pyquda-rectangular-wilson-loop-codegen
description: >
  Wilson loop computation skill for lattice QCD using PyQUDA.
  Generates a research-style Python script to evaluate a single
  rectangular Wilson loop W(mu, nu; R, T) with explicit lattice construction.

  The script follows the full pipeline:
  gauge loading → Wilson line construction → loop closure →
  trace evaluation → lattice averaging.
---

PyQUDA Rectangular Wilson Loop Skill

Purpose

Given a gauge configuration on a lattice, construct and evaluate a single
rectangular Wilson loop observable W(mu, nu; R, T) using explicit PyQUDA operations.

This skill bridges:

continuum Wilson loop definition → lattice discretization → PyQUDA implementation

The output is a minimal, directly runnable Python script suitable for HPC usage.


Physics Definition

Wilson loop in the (mu, nu) plane:

W_mu_nu(x; R, T) = Tr[
L_mu(x, R)
L_nu(x + R mu, T)
L_mu^dagger(x + T nu, R)
L_nu^dagger(x, T)
]

Lattice average:

<W_mu_nu(R, T)> = (1 / (Nc * V)) * sum_x Tr[W_mu_nu(x; R, T)]

where:
- Nc = 3
- V = total lattice volume


Inputs

The generated script must read:

cfg = sys.argv[1]
mu_name = sys.argv[2]
nu_name = sys.argv[3]
R_len = int(sys.argv[4])
T_len = int(sys.argv[5])

Direction constraints:
- mu, nu ∈ {X, Y, Z, T}
- mu != nu
- negative directions are not independent inputs
- reverse directions only appear when closing the loop


Implementation Strategy

The script must explicitly construct the Wilson loop using gauge links.

Wilson line rule:

A Wilson line of length L in direction dir is constructed as:

- first link: gauge.pack(dir, field)
- remaining L - 1 links: repeated gauge.covDev(field, dir)

Rectangular loop construction:

1. propagate along mu direction for R_len steps
2. propagate along nu direction for T_len steps
3. propagate backward along mu direction
4. propagate backward along nu direction
5. close the loop at starting point

This corresponds to the ordered product of gauge links along the loop path.


Required PyQUDA Workflow

The script must strictly follow:

Step 0 - Imports

Use only:
sys, numpy, cupy, mpi4py
pyquda.init, LatticeInfo
pyquda_utils.core, io
direction constants X, Y, Z, T

Avoid any unrelated modules or helper abstractions.


Step 1 - Input parsing

Map mu_name and nu_name to PyQUDA direction constants.

Validate:
- mu != nu
- R_len >= 1
- T_len >= 1


Step 2 - Initialization

init([1, 1, 1, 4], resource_path=".cache")

Construct lattice information explicitly.


Step 3 - Gauge loading

Construct cfg_file using the standard Chroma QIO path pattern.

Load with:

gauge = io.readChromaQIOGauge(cfg_file)

Move to GPU:

gauge.toDevice()

Fail explicitly if loading fails.


Step 4 - Wilson loop construction

Use explicit field-based construction:

- core.LatticeFermion
- gauge.pack
- gauge.covDev
- core.LatticeGauge
- loops.unpack

Follow exactly the loop path described above.

Do not:
- use high-level wilson_loop helpers
- construct multiple loops
- average over orientations
- introduce alternative implementations


Step 5 - Data inspection

On rank 0 print:

type(loops.data)
loops.data.shape
loops.data.dtype

This defines how the observable should be reduced.


Step 6 - Trace and lattice sum

Procedure:

1. identify color matrix as final 3x3 block
2. compute trace over color indices
3. sum over lattice site indices
4. keep complex values until final stage

Notes:

- do not discard imaginary part early
- do not assume tensor layout blindly
- do not build a large automatic axis-detection system

Instead:
use a short, explicit reduction consistent with observed layout


Step 7 - MPI reduction

Use mpi4py:

- compute local complex sum
- reduce via MPI.COMM_WORLD.Allreduce
- reconstruct global complex value

Do not:
- use gather-based reduction
- average per rank


Step 8 - Normalization

W = total_sum / (Nc * V_global)

V_global must represent full lattice volume.


Step 9 - Output

On rank 0 print:

- (mu, nu)
- (R_len, T_len)
- real part of W
- imaginary part of W


Constraints

The generated script must:

- implement only one loop per execution
- remain minimal and explicit
- follow the physical construction exactly

The script must NOT:

- hardcode cfg numbers
- restrict to a fixed plane like (X,Y)
- generate multiple observables
- use wilson_loop helper
- use HDF5 or alternative IO
- silently replace invalid data
- drop imaginary part prematurely
- mix NumPy and CuPy incorrectly
- replace MPI reduction with gather operations
- implement a generic tensor inference engine


Style

Write as a computational physics script:

- direct
- minimal
- readable
- step-by-step
- no unnecessary abstraction


Output

Return exactly one complete Python script.

No explanation outside the code.
