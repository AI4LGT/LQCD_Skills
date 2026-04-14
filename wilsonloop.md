---
name: pyquda-wilson-loop-codegen
description: >
  Generate a research-style Python script using PyQUDA to compute a basic
  Wilson loop (1x1 plaquette) in lattice QCD.

  The script must follow the explicit PyQUDA workflow, including:
  initialization (init, LatticeInfo), gauge loading (readChromaQIOGauge),
  GPU transfer (toDevice), Wilson loop construction via pack/covDev/unpack,
  and final trace computation with correct MPI reduction and normalization.

  The output code must be directly runnable, minimal, explicit, GPU-aware,
  MPI-aware, and suitable for computational physics research usage.
---

# PyQUDA Wilson Loop Code Generation Skill

## Goal

Generate a **research-style Python script** using PyQUDA to compute a basic Wilson loop (starting from 1x1 plaquette).

The script must be:
- directly runnable
- explicit and easy to debug
- minimal abstraction
- consistent with lattice QCD workflows

---

## Physics Target

Construct the 1x1 Wilson loop (plaquette) in the (X, Y) plane:

W(x) = Tr[
U_X(x)
U_Y(x + X)
U_X^\dagger(x + Y)
U_Y^\dagger(x)
]

Then compute the lattice average:

<W> = (1 / (Nc * V)) * sum_x Tr[W(x)]

where:
- Nc = 3
- V = total lattice volume

---

## Required Workflow

The generated script must follow this exact structure:

### Step 0: Imports

Use only necessary libraries:

```python
import sys
import numpy as np
import cupy as cp

from pyquda import init, LatticeInfo
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T
```
### Step 1: Read command-line input
```
cfg = sys.argv[1]
```
This cfg value MUST be used to construct the gauge file path.


### step 2: Initialize PyQUDA

Example:
```
init([1, 1, 1, 4], resource_path=".cache")
latt_info = LatticeInfo([24, 24, 24, 72], -1, 1.0)
```
### Step 3: Load gauge configuration

Use cfg to build path:
```
cfg_file = f"/path/to/config_cfg_{cfg}.lime"
gauge = io.readChromaQIOGauge(cfg_file)
gauge.toDevice()
```
Do NOT hardcode configuration numbers.

### Step 4: Construct Wilson loop

Use this exact pattern:
```
gauge_wil = core.LatticeFermion(gauge.latt_info)

gauge.pack(X, gauge_wil)
gauge_wil = gauge.covDev(gauge_wil, Y)
gauge_wil = gauge.covDev(gauge_wil, -X)
gauge_wil = gauge.covDev(gauge_wil, -Y)

loops = core.LatticeGauge(gauge.latt_info)
loops.unpack(X, gauge_wil)
```
This represents a closed loop in (X, Y).

### Step 5: Inspect data
```
loop_data = loops.data
```
Print shape and type on rank 0.

### Step 6: Compute trace

DO NOT drop imaginary part early.

Determine the actual layout of `loops.data` first.
Then sum only over non-color lattice/site indices, keeping the final 3x3 color matrix.
Then compute the color trace explicitly.
Do not assume the axes blindly without checking the printed shape.
### Step 7: MPI reduction and normalization

Must distinguish:

local sum
global sum

Final normalization:
```
W = total_sum / (Nc * global_volume)
```
Do NOT divide by extra factors like 4 unless justified.
If a direct PyQUDA reduction helper is not clearly known to exist, use mpi4py for explicit global reduction instead of inventing a fake API.
### Step 8: Output result

On rank 0, print:

real part
imaginary part
## Critical Constraints
### 1. Do NOT mix NumPy and CuPy incorrectly

If data is on GPU, use:

cp.sum
cp.trace

Only use .get() at the end.

### 2. Do NOT ignore MPI

Ensure global reduction is correct.

### 3. Do NOT over-engineer
No classes
No frameworks
No unnecessary abstraction
### 4. Keep code explicit

Write in a clear, step-by-step style like a computational physicist.

### 5. Only generate ONE loop

Only implement (X, Y) plaquette.

Do NOT generate multiple loop types.
### 6. Do not use a prebuilt high-level wilson_loop helper.
Construct the loop explicitly with pack / covDev / unpack.
## Output Requirement

Generate a single complete Python script only.

Do NOT include explanation outside the code.

The script must be ready to run.
### 7. Do NOT replace the PyQUDA workflow with a custom NumPy/HDF5 implementation

The script must use the actual PyQUDA / pyquda_utils workflow.

It must include the real operations:
- `init(...)`
- `LatticeInfo(...)`
- `io.readChromaQIOGauge(...)`
- `gauge.toDevice()`
- `gauge.pack(...)`
- `gauge.covDev(...)`
- `loops.unpack(...)`

Do NOT create custom replacement functions named `pack`, `covDev`, or `unpack`.

### 8. Do NOT change the input format

The input is a Chroma QIO gauge configuration, not an HDF5 file.

Do NOT use `h5py`.
Do NOT search HDF5 datasets.
Do NOT reinterpret the input as HDF5.

### 9. Do NOT use silent fallback identities

If the gauge field cannot be read, or if the required operation fails, raise a clear error and stop.

Do NOT replace missing links or failed reads with identity matrices.

### 10. Do NOT compute only one site

The script must compute the Wilson loop contribution over the full local lattice volume, then perform global MPI reduction, then normalize by the total global volume.

Do NOT evaluate only a single plaquette at site (0,0).

### 11. Do NOT average over MPI ranks directly

MPI must be used to sum local lattice contributions, not to average one scalar per rank.

The final normalization must be based on the full global lattice volume.

## Optional Extension (comment only)

Explain how to extend to other planes:

(X,Y), (X,Z), (X,T)
(Y,Z), (Y,T)
(Z,T)

But do not implement them in execution.

## Final Instruction

Now generate the Python script following all rules above.