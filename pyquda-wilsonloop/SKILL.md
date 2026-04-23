---
name: pyquda-wilsonloop
description: >
  Generate PyQUDA Python scripts for Wilson loop computations on lattice QCD gauge configurations.
  Use when: (1) user asks to compute Wilson loops with PyQUDA, (2) user provides gauge configuration file (.lime),
  (3) user specifies loop paths (e.g., rectangular R x T loops), (4) user needs gauge.loop(...) based implementation.
---

# PyQUDA Wilson Loop Skill

Generate minimal PyQUDA scripts that compute Wilson loops using `gauge.loop(...)`.

## When to Use

- User provides a gauge configuration file path
- User specifies loop shapes (rectangular loops R×T, or explicit direction paths)
- User wants to compute Wilson loop expectation values

## Allowed Imports

Only these imports are allowed in the generated script:

```python
import sys
import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T
```

Do **not** import: `mpi4py`, `json`, `h5py`, `pathlib`, `logging` unless the user explicitly asks.

## Core Rule: Use `gauge.loop(...)` Directly

Always compute Wilson loops via:

```python
res = gauge.loop(loop_list, coefficients)
```

Do **not** use: `gauge.pack`, `gauge.covDev`, `core.LatticeFermion`, `core.LatticeGauge`, or `loops.unpack`.

## Paths Must Be Explicitly Expanded

For a rectangular R×T loop in the (μ, ν) plane, the direction list must be written out fully in the code:

```
[μ repeated R times, ν repeated T times, -μ repeated R times, -ν repeated T times]
```

Do **not** use: `[mu] * R + [nu] * T + [-mu] * R + [-nu] * T` or any other symbolic/generated form.

## Postprocessing Style

Keep postprocessing minimal:

```python
Nc = gauge.latt_info.Nc
for i, obj in enumerate(res):
    U = obj.getHost()
    U = U.reshape(-1, Nc, Nc)
    tr = np.trace(U, axis1=-2, axis2=-1)
    # compute and print local_sum, local_n, local_mean (and/or global equivalents)
```

## What NOT to Generate

- MPI Allreduce unless user asks
- JSON or HDF5 output
- Plaquette sanity checks
- Orientation or plane averaging
- Path-building helper functions
- Geometry validation blocks
- Unnecessary CLI parsing

## Reference

See [references/main.py](references/main.py) for a complete working example on a 24×24×24×72 lattice with a 3×4 Wilson loop in the (X,T) plane.
