---
name: pyquda-tool
description: >
  PyQUDA tool usage skill. Generates Python code that calls PyQUDA to solve
  quark propagators on lattice gauge configurations. Covers: configuration
  loading, quark parameter setup
  (Wilson/clover action, mass or kappa, clover coefficient, link smearing),
  multigrid solver configuration, source construction (point, Gaussian
  smearing with APE/HYP/stout links), propagator inversion, and residual
  verification. Reads ensemble parameters from ensemble_registry.yaml.
  Trigger on: "compute propagators", "solve propagator", "run inversions",
  "call PyQUDA", "solve Dirac equation", or when lqcd-physics has produced
  a propagator requirements list.
---

# PyQUDA Tool Usage

## Purpose

Translate a propagator specification into executable PyQUDA code
that reads gauge configurations and produces propagator or correlator data files.
This skill generates a **data production script only** — a single script that
computes and saves per-configuration results. It does not produce analysis code.

## Prerequisites

- PyQUDA installed with working QUDA backend and GPU access
- Gauge configurations accessible at paths specified in ensemble registry
- Python 3.8+, numpy, (optional) cupy for GPU arrays, (optional) h5py for output
- MPI environment to run PyQUDA with multiple GPUs

## Code style and execution model

**Flat, self-contained scripts**: Generated PyQUDA code should be a single-file, one-off script with all physical parameters (masses, clover coefficients, lattice dimensions, file paths, source positions, etc.) hardcoded as plain variables at the top of the file. Do not factor the code into many small functions or classes. The script should read top-to-bottom so that a collaborator can immediately see and verify every physical parameter. This is standard practice in lattice QCD: computation scripts are shared and cross-checked by multiple people, not maintained as reusable software. Command-line arguments (argparse) should be reserved for parameters that distinguish independent parallel jobs — typically the **configuration ID** (so the same script can be submitted as multiple jobs for different configs). Physical parameters that define the calculation itself must remain hardcoded in the file.

**Always run with MPI**: PyQUDA scripts must be launched via MPI, e.g. `mpirun -np 4 python script.py` or `srun -n 4 python script.py` on SLURM clusters. Even single-GPU runs use `mpirun -np 1`. The number of MPI ranks must equal the product of grid_size dimensions, with one GPU per MPI rank by default. Always include a comment at the top of the generated script showing the expected launch command, e.g. `# Run: mpirun -np 4 python this_script.py`.

**Heavy computation — no interactive monitoring**: PyQUDA propagator inversions are computationally intensive and typically run for minutes to hours per configuration. Do not add progress bars, interactive prompts, or suggest frequent status checking. QUDA prints solver iteration counts and residuals to stdout automatically — the user monitors progress via stdout or log redirection. Submit the script and let it run to completion.

## Before generating code (REQUIRED)

Do NOT generate any computation code until the following are resolved. Present the information gathered in step 1 to the user, then ask the questions in step 2 and wait for answers.

**Step 1 — Physics derivation**: The propagator requirements must be known before writing code. Use lqcd-physics reasoning to derive: what interpolating operators are needed, what the Wick contraction looks like, and which propagators (quark flavors, source→sink structure) are required. Present this derivation to the user.

**Step 2 — Ask the user** to confirm or specify the following source configuration, as the optimal choice depends on the target observable and computational budget:

- **Source type**: point, wall, smeared (Gaussian/Wuppertal), or volume
- **Smearing**: whether to apply Gaussian smearing to the source, and if so, the smearing parameters (radius `rho`, number of steps `n_steps`) and which gauge field to use for smearing
- **Source positions**: spatial position(s) `[x0, y0, z0]` and time slices `t_src`. Using multiple source time slices (e.g., `t_src = 0, T//4, T//2, 3*T//4`) on each configuration significantly improves the signal-to-noise ratio by multiplying the effective statistics

Momentum projection is **not** a user choice — it is determined by the correlator definition. If the target observable requires momentum $\vec{p}$, the source must be projected to that momentum accordingly.

**Step 3 — Generate code** only after the user has confirmed the source configuration.

## Workflow

### Step 0: Common conventions

PyQUDA is a Python wrapper of the QUDA library, which provides GPU-accelerated operations for lattice QCD. PyQUDA uses NumPy arrays to handle lattice fields, and uses CuPy/PyTorch/DPNP arrays when GPU accelerated linear algebra is needed. PyQUDA is designed to run on cluster environment which have a job scheduler like SLURM or PBS, thus it usually runs in an MPI environment. There are some key concepts to understand when using the package:

#### Grid

Grid in PyQUDA indicates how to partition the lattice across multiple MPI ranks, which is very close to the concept of a Cartesion communicator. For example, a grid size of `[2, 2, 1, 1]` means that the lattice will be partitioned into 4 sublattices in the x and y dimensions, while the z and t dimensions are not partitioned. The product of the grid dimensions must equal the total number of MPI ranks used to run the job. If one dimension is partitioned, there will be communication between MPI ranks in that dimension during the solver iterations. If the grid size is not specified, but the targeting lattice size is provided during the initialization, PyQUDA will automatically generate a grid size, trying to minimize the communication between different MPI ranks.

#### Device

Device in PyQUDA refers to the local GPU device ID that each MPI rank will use. By default, PyQUDA will assign GPU devices based on the local rank of each MPI process. For example, if you have 4 GPUs and 4 MPI ranks, each rank will be assigned to a different GPU (rank 0 to GPU 0, rank 1 to GPU 1, etc.). Sometimes, a cluster will offer a binding script to bind each MPI rank to a specific NUMA node, GPU, and NIC, and environment variables such as `CUDA_VISIBLE_DEVICES` are usually set by the script. You will have to set `enable_mps=True` during the initialization in this situation to allow all ranks in one node can use the same GPU ID 0, although they are actually using different devices.

#### Lattice information

`LatticeInfo` class in PyQUDA is used to store the lattice information, including the lattice dimensions, boundary conditions, and anisotropy. Assuiming the global lattice size is `[GLx, GLy, GLz, GLt]`, the grid size is `[Gx, Gy, Gz, Gt]`, and the local lattice size for each MPI rank is `[Lx, Ly, Lz, Lt]`, where `Lx = GLx // Gx`, `Ly = GLy // Gy`, `Lz = GLz // Gz`, and `Lt = GLt // Gt`. The `LatticeInfo` object is used to initialize the Dirac operator and to create lattice fields, ensuring that all operations are consistent with the lattice geometry and partitioning.

#### Lattice fields

Lattice fields are objects in PyQUDA to handle the data of fields used in LQCD. For example, the gauge field is hold by a `LatticeGauge` object, and the quark propagator is hold by a `LatticePropagator` object. The `data` attribute of these objects is a `numpy.ndarray` (or `cupy.ndarray`/`torch.Tensor`/`dpnp.ndarray` array if using GPU acceleration) that contains the actual field data. The layout of the field data is `[2, Lt, Lz, Ly, Lx // 2]`, which is the even-odd preconditioned layout. The first dimension of size 2 corresponds to the parity (even/odd) of the lattice sites, and the last dimension of size `Lx // 2` corresponds to the half-lattice size in the x direction due to the even-odd preconditioning. Note the order of dimensions is "xyzt" in most cases in PyQUDA, except for the data layout of a field (which is "tzyx"). If a filed has both source and sink spin/color indices, the order will always be `[snk, src]`. For example, the shape of a `LatticePropagator.data` will be `[2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc]`, and the meaning of each dimension is `[parity, t, z, y, x//2, spin_snk, spin_src, color_snk, color_src]`. The `LatticeGauge.data` will have a shape of `[4, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc]`, where the additional dimension of size 4 corresponds to the four directions of the gauge links.

#### Array location

PyQUDA can handle arrays in different locations (CPU or GPU) with different backends (NumPy, CuPy, PyTorch, DPNP). The `backend` parameter in the initialization determines which array library is used for handling lattice fields by default. If `backend="cupy"` is set, PyQUDA will create a CuPy array for the field data when creating a `LatticeGauge` or `LatticePropagator` object. But remember if a `LatticeGauge` or `LatticePropagator` is created by loading from disk, the field data will always be created as a NumPy array on CPU, and you will have to use the `toDevice()` to transfer the data to GPU memory. PyQUDA provides a consistent API to support array linear algebra operations on GPU with backends. You can check `pyquda_comm.array` module for the supported operations. It's clear that using the `"numpy"` backend will keep all arrays on CPU, saving GPU memory but without acceleration.

#### Gamma matrices

The basis for the gamma matrices provided by `pyquda_utils.gamma` module is the DeGrand-Rossi basis. PyQUDA uses a bit-field encoding for gamma matrices, where `gamma.gamma(1)` corresponds to γ₁, `gamma.gamma(2)` to γ₂, `gamma.gamma(4)` to γ₃, and `gamma.gamma(8)` to γ₄. Products of gamma matrices can be represented using bitwise OR. For example, `gamma.gamma(15)` corresponds to γ₁γ₂γ₃γ₄ = γ₅.

### Step 1: Load ensemble metadata

Read the ensemble registry (YAML/JSON) to obtain:
- Configuration file paths and format (ILDG, QIO, milc, ...)
- Lattice dimensions (Lx, Ly, Lz, Lt)
- Gauge action parameters (beta, tadpole factor if applicable)

### Step 2: Initialize PyQUDA context

```python
from pyquda_utils import core

grid_size = [Gx, Gy, Gz, Gt] # grid partitioning
latt_size = [Lx, Ly, Lz, Lt] # lattice dimensions
core.init(grid_size, latt_size, backend="cupy", resource_path="/path/to/quda/tunecache")
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)
```
Here we initialize the PyQUDA context with the MPI, and set the lattice partitioning `grid_size`, and then create a `LatticeInfo` object with the lattice dimensions `latt_size`. The `latt_size` in `core.init` will be ignored if `grid_size` is specified, and the MPI size must be equal to the product of `grid_size`. PyQUDA will automatically generate a `grid_size` if only `latt_size` is provided, trying to minimize the communication between different MPI ranks. The `t_boundary=-1` indicates anti-periodic boundary conditions in time, and `anisotropy=1.0` indicates no anisotropy (use `xi_0 / nu` if using an anisotropic lattice, where `xi_0` is the gauge anisotropy and `nu` is the input light speed). The `backend` parameter specifies which Array API we should use to handle lattice fields. `"cupy"` backend could be helpful for GPU acceleration in the "contraction to correlator" step. The `resource_path` is where PyQUDA saves and loads QUDA's **autotuning cache**. On the first run with a given lattice size, grid partitioning, and solver configuration, QUDA automatically benchmarks many GPU kernel launch parameters to find the optimal ones. This autotuning can take significant extra time (minutes to tens of minutes) on the first run, but subsequent runs with the same parameters load the cached settings and start computing immediately. Always set `resource_path` to a persistent directory (e.g., `"./"` or a shared project directory) so the tune cache is preserved across runs. If the cache was generated by a different QUDA version, delete it and retune or use another directory to avoid incompatible launch parameters.

This step establishes the runtime context used by nearly all later operations
(MPI topology, lattice geometry, backend, and tune-cache behavior). In normal
practice, scripts start from this context setup and then proceed to gauge I/O,
Dirac construction, inversion, and contraction.

### Step 3: Load gauge configuration

```python
from pyquda_utils import io

gauge = io.readChromaQIOGauge("/path/to/gauge/configuration")
```
Here we load the gauge configuration from disk using the appropriate read function based on the file format. The `readChromaQIOGauge` function is used for Chroma generated QIO formatted files. Make sure to replace the path with the actual location of your gauge configuration. You can read the PyQUDA source code to determine which read function to use for other formats (e.g. `readMILCGauge` for MILC format).

### Step 4: Configure quark and solver parameters

#### Wilson fermion:
```python
from pyquda_utils import core

dirac = core.getWilson(latt_info, mass, tol, maxiter, multigrid)
```
Here we create a Wilson Dirac operator with the specified mass, solver tolerance, maximum iterations, and multigrid settings. `multigrid` should be the parameter to determine the aggregation size of every level. If `multigrid` is `None`, BiCGStab is used. These parameters should be set according to the requirements of your calculation. For example, for light quarks, you might need a smaller mass and a looser solver tolerance, while for heavy quarks, you can use a larger mass and a tighter tolerance to ensure the accuracy in timeslices far from the source. The multigrid is usually very helpful for light quark propagators, because of the critical slowing down issue of LQCD.

#### Clover fermion:
```python
from pyquda_utils import core

dirac = core.getClover(latt_info, mass, tol, maxiter, xi_0, csw_t, csw_r, multigrid)
```
Here we create a Clover Dirac operator with the specified mass, solver tolerance, maximum iterations, gauge anisotropy `xi_0`, clover coefficients `csw_t` and `csw_r`, and multigrid settings. Note that `csw_t` and `csw_r` are the clover coefficients for the temporal and spatial components, respectively. If your lattice is isotropic, you can set `csw_t = csw_r = csw`. `xi_0` is the gauge anisotropy,  please clarify it from the fermion anisotropy `xi = xi_0 / nu`.

#### Load all required fields into GPU memory:
```python
from pyquda_utils import core

gauge.stoutSmear(1, rho, 4)
with dirac.useGauge(gauge):
    # Do something
    ...
```
Here we first apply 4-dimentional stout smearing of parameter `rho` to the gauge field for 1 time. This might be necessary to calculate the propagator if you found quarks.smearing in the ensemble registry. You can read the PyQUDA source code to determine the appropriate smearing algorithm and parameters. Then we use the `useGauge` context manager to ensure that the gauge field and all auxiliary fields are loaded into QUDA. Inside this context, you can solve quark propagators defined by the Dirac operator. If mutliple context managers are nested, the innermost one will take effect. `useGauge` is not a free operation, it will trigger the data transfer between CPU and GPU if the gauge field is not already on GPU, and it will also trigger the reorder operion to convert the gauge field data into the layout required by QUDA. So it's better to put all the operations that require the gauge field inside the same `useGauge` context to avoid unnecessary data transfer and reordering.

### Step 5: Construct source and solve propagator

Execution rule (correctness + performance): keep inversions inside the matching
`with dirac.useGauge(gauge_for_this_dirac):` context.

If inversion and shift use different gauge fields (common case: stout-smeared
gauge for inversion, unsmeared pure gauge for Wilson-line shift), do not keep
one long `useGauge` block around a loop that also performs shifts. Instead,
switch contexts explicitly: inversion block with `dirac.useGauge(...)`, then
shift/contraction block with pure gauge, then re-enter inversion context.

Only when a loop is purely inversion with one unchanged inversion gauge is it
safe to keep `useGauge` outside inner loops for performance.

#### Directly solve propagator from a simple (point, wall, volume) source:
```python
from pyquda_utils import core, phase_v2

phase = phase_v2.MomentumPhase(latt_info).getPhase([kx, ky, kz], [x0, y0, z0])

with dirac.useGauge(gauge):
  propag_pt = core.invert(dirac, "point", [x0, y0, z0, t0], phase.data)
  propag_wl = core.invert(dirac, "wall", t0, phase.data)
  propag_vl = core.invert(dirac, "volume", None, phase.data)
```
Here we call the `invert` function to solve for the propagator using different source types. For a point source, we specify the position `[x0, y0, z0, t0]`. For a wall source, we specify the time slice `t0`. For a volume source, no additional parameters are needed. Here we also apply a momentum phase to the source, which is necessary for computing momentum wall source propagators. If no `phase` is given, the values for all three types will default to 1.

#### Solve propagator from an existing propagator:
```python
from pyquda_utils import core, source

with dirac.useGauge(gauge):
  source_pt = source.propagator(latt_info, "point", [x0, y0, z0, t0])
  source_sh = source.gaussianSmear(source_pt, gauge, rho, n_steps)
  propag_sh = core.invertPropagator(dirac, source_sh)
```
Here we first create a point source propagator using the `source.propagator` function. Then we apply Gaussian smearing to this source using the `source.gaussianSmear` function, which takes the original source, the gauge field, the smearing radius in the momentum space `rho`, and the number of smearing steps `n_steps`. Note the gauge we used here for the gaussian smearing might be different from the one used for the Dirac operator, depending on the requirements. Finally, we solve for the smeared propagator using the `core.invertPropagator` function.

#### Solve sequential propagator from an existing propagator on a specific time slice:
```python
from pyquda_utils import core

with dirac.useGauge(gauge):
  propag_sq = core.invertSequential(dirac, propag_sh, t_seq)
```
Here we use the `core.invertSequential` function to solve for a sequential propagator from the smeared propagator. This is useful for three-point correlator calculations where we need to insert an operator at a specific time slice. The `t_seq` parameter specifies the time slice where the sequential source is defined.

#### Multiple source times

Structure the main computation as a loop over source times when multiple sources are used:

Memory rule (GPU): avoid pre-building and keeping all propagators for all
`t_src` outside the loop. Construct the propagator inside the `t_src` loop,
immediately perform the needed contraction/output, then release temporary
objects before moving to the next source time. This keeps peak GPU memory low
and avoids OOM on large lattices.

The same idea applies to wall-source momentum runs: avoid pre-building
propagators for all momenta outside the loop. For each `t_src` and each
momentum, build only the current momentum propagator, contract/save, then
release temporary objects before the next momentum. Avoid holding a full
`{momentum: propagator}` map in GPU memory.

For nonlocal wall 2pt at nonzero sink momentum, using one zero-phase wall
propagator and only sink extraction phase tends to bias the source sum toward
$p\!=\!0$. Using two source phases (for example `p_src1`, `p_src2`) keeps the
estimator aligned with the target momentum channel.

Common error pattern to avoid: building momentum propagator lists/maps such as
`forward_props`, `backward_props`, or `{mom: propagator}` first, and
contracting in a separate later loop. This usually increases memory pressure
without improving numerical quality.

```python
for t_src in t_srcs:
  with dirac.useGauge(gauge_solver):
    propag = core.invert(dirac, "point", [x0, y0, z0, t_src])
  # ... (optional) switch to pure-gauge context and do Wilson-line shift/contraction
  # ... contract to correlator and save with t_src label
  # ... release propag / temporary tensors before next t_src

for t_src in t_srcs:
  for mom in momenta:
    with dirac.useGauge(gauge_solver):
      phase = phase_builder.getPhase(mom, [x0, y0, z0])
      propag_mom = core.invert(dirac, "wall", t_src, phase.data)
    # ... (optional) switch to pure-gauge context and do Wilson-line shift/contraction
    # ... contract to correlator and save with (t_src, mom) label
    # ... release propag_mom / temporary tensors before next mom
```

#### Covariant shift for displaced operators (nonlocal 2pt only)

For local two-point correlators, no spatial shift is needed. For nonlocal
operators with a spatial displacement, the Wilson-line-shifted propagator is
generated by repeated calls to `covDev` on the propagator leg that carries the
displacement.

When the lattice is distributed across MPI ranks, `covDev` is important because
it handles halo communication and boundary transport consistently. A local
array roll does not include this communication and can distort the nonlocal
Wilson-line displacement.

Key usage points:

- `gauge.ensurePureGauge()` and `gauge.pure_gauge.loadGauge(gauge)` prepare the
  pure-gauge link field used by `covDev`
- `gauge.pure_gauge.covDev(fermion, mu)` applies one gauge-covariant hop in the
  direction `mu`
- `mu = 0,1,2,3` means `+x,+y,+z,+t`, while `mu = 4,5,6,7` means `-x,-y,-z,-t`

`covDev` is called from `gauge.pure_gauge` and takes a fermion field plus a direction `mu` as input. Therefore, a propagator shift is implemented blockwise: loop
over all source spin/color components, extract each fermion block with
`getFermion(spin, color)`, apply `gauge.pure_gauge.covDev(fermion, mu)`, and
write the shifted block back with `setFermion(...)`.

Template:

```python
for spin in range(4):
  for color in range(3):
    fermion = propag_1.getFermion(spin, color)
    fermion_shift = gauge.pure_gauge.covDev(fermion, 2)  # 2 means +z
    propag_1.setFermion(fermion_shift, spin, color)
```

For displacement by multiple lattice spacings, reuse the updated propagator and
repeat the same spin/color loop once per step. In other words, `z = 2`
means applying the one-step update twice in sequence, not trying to pass a
larger displacement directly into `covDev`.

For a displaced nonlocal 2pt, keep the role of each object fixed:

- shift only the forward propagator leg (usually the `S1` leg)
- keep the backward leg unshifted and use it through `S2^dagger`
- apply the extraction phase for `p_snk` at contraction time, not inside `covDev`

Do not mix the gauge used for inversion with the gauge used for the shift. The
Dirac operator may use a smeared gauge field, but `covDev` should use the pure
gauge links appropriate for the Wilson line definition. If the active QUDA gauge
state has changed after leaving a solver context, reload the pure gauge before
calling `covDev`; otherwise a gauge-allocation error can occur.

A robust mixed-gauge pattern is:
1. Enter `with dirac.useGauge(gauge_solver):` and do inversion.
2. Exit inversion context, load/refresh pure gauge, and apply `covDev` shifts.
3. Re-enter `with dirac.useGauge(gauge_solver):` before the next inversion.

### Step 6: Save propagator (optional)
This is not always necessary, but you can save the propagator to disk in a format of your choice (e.g. HDF5, NumPy binary) for later analysis.

```python
propag_sh.save("propag_sh.npy", use_fp32=False)
propag_sh.saveH5("propag_sh.h5", tag, annotation=annotation, check=True, use_fp32=False)
```
Here we save the smeared propagator in both NumPy binary format and HDF5 format. The `save` method saves the propagator as a `.npy` file, while the `saveH5` method saves it as an `.h5` file with additional metadata such as a tag, annotation, and a check for data integrity. The `use_fp32` parameter determines whether to save the data in single precision (float32) or double precision (float64).

### Step 7: Contraction to correlator (optional)
If you need to compute correlators, you can perform contractions of the propagators using NumPy or CuPy's contraction utilities or by manually implementing the necessary spin-color contractions.

Important convention for source/sink gamma structures:

- use project-appropriate names for gamma structures
  (for example `gamma_structure1`, `gamma_structure2`)
- these gamma-structure variables should store only the physical bilinear
  operators inserted at source and sink
- they should not include the extra `gamma_5` factors coming from
  `\gamma_5`-hermiticity of the backward propagator
- those hermiticity factors appear only in the contraction formula, typically as
  `gamma_5 @ gamma_structure1` on the sink side and
  `gamma_structure2.conj().T @ gamma_5` on the source side

For example, if the physical channel is `gtg5`, then define
`gamma_structure1 = gamma_structure2 = gamma_t @ gamma_5` (or equivalent
project naming) as the operator itself. Do not fold the additional hermiticity
`gamma_5` factors into these definitions.

For a meson two-point function after using `\gamma_5`-hermiticity, the spin
structure entering the contraction is

$$
C(t) \propto \sum_{\vec{x}} \mathrm{Tr}\left[
S^\dagger(\vec{x},t;\vec{x}_0,t_0)
(\gamma_5\Gamma_{\mathrm{snk}})
S(\vec{x},t;\vec{x}_0,t_0)
(\Gamma_{\mathrm{src}}^\dagger\gamma_5)
\right].
$$

Below is a minimal rho example showing this separation:
```python
import cupy as cp
from pyquda_utils import core, phase_v2, gamma

phase = phase_v2.MomentumPhase(latt_info).getPhase([-kx, -ky, -kz], [x0, y0, z0])

# PyQUDA gamma convention: bit-field encoding
# gamma(1)=γ₁, gamma(2)=γ₂, gamma(4)=γ₃, gamma(8)=γ₄
# Products use bitwise OR: gamma(15)=γ₁γ₂γ₃γ₄=γ₅
gamma_1 = gamma.gamma(1)   # bit 0 = γ₁
gamma_2 = gamma.gamma(2)   # bit 1 = γ₂
gamma_3 = gamma.gamma(4)   # bit 2 = γ₃, NOT γ₄
gamma_5 = gamma.gamma(15)  # all four bits = γ₅
gamma_structure1_list = [gamma_1, gamma_2, gamma_3]
gamma_structure2_list = [gamma_1, gamma_2, gamma_3]
rho_2pt_1 = cp.einsum('wtzyx,wtzyxjiba,jk,wtzyxklba,li->t', phase.data, propag_sh.data.conj(), gamma_5 @ gamma_structure1_list[0], propag_sh.data, gamma_structure2_list[0].conj().T @ gamma_5)
rho_2pt_2 = cp.einsum('wtzyx,wtzyxjiba,jk,wtzyxklba,li->t', phase.data, propag_sh.data.conj(), gamma_5 @ gamma_structure1_list[1], propag_sh.data, gamma_structure2_list[1].conj().T @ gamma_5)
rho_2pt_3 = cp.einsum('wtzyx,wtzyxjiba,jk,wtzyxklba,li->t', phase.data, propag_sh.data.conj(), gamma_5 @ gamma_structure1_list[2], propag_sh.data, gamma_structure2_list[2].conj().T @ gamma_5)
# Average over the three spatial polarizations of the rho meson
rho_2pt = (rho_2pt_1 + rho_2pt_2 + rho_2pt_3) / 3
rho_2pt = core.gatherLattice(rho_2pt.get(), [0, -1, -1, -1])
```
Please refer to "lqcd-physics" skills for more details on how to perform contractions for various hadronic correlators. The convention for the spin-color indices in the propagator is (parity, t, z, y, x, spin_snk, spin_src, color_snk, color_src). The `einsum` function is used to perform the necessary contractions. The phase multiplied during contraction is the extraction phase for $e^{-i\vec{p}\cdot\vec{x}}$. In wall-source momentum 2pt with two propagators, source phases are fixed at inversion, the sink-side sign comes from hermitian conjugation of the second propagator, and the extraction phase is applied only at contraction. Finally, we gather the lattice data from all MPI ranks to obtain the full correlator in the root rank as a function of time. The second argument of `gatherLattice` specifies the dimensions to gather in `tzyx` order, where `0` indicates that we want to gather the data in the $t$ dimension, and `-1` means performing reduction in all three spatial dimensions.

### Step 8: Save output

Save correlator data to HDF5 **per configuration**, with metadata sufficient to reproduce the result (source position, momentum, operator type, configuration ID). Make sure only the root rank (rank 0) writes the output file to avoid conflicts. The output should contain correlator data C(t) indexed by configuration and source time.

## Common issues

- **Grid/context is not initialized**: The runtime context setup may have been skipped or failed. Recheck the Step 2 context setup (`core.init` and `LatticeInfo`) in the current script.
- **GPU out of memory**: Use more GPUs (increase MPI size), use `backend="numpy"` when initializing PyQUDA.
- **Solver not converging**: Check gauge configuration integrity, try restarting with tighter intermediate tolerance
- **Spinor volume doesn't match gauge volume**: You likely solved propagators without first entering the matching `with dirac.useGauge(gauge)` context, or mixed inversion gauge context with `pure_gauge.covDev` context and did not reload the Dirac gauge context before the next inversion.
