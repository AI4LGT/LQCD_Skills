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

Translate a propagator specification (from S1) into executable PyQUDA code
that reads gauge configurations and produces propagator data files.

## Prerequisites

- PyQUDA installed with working QUDA backend and GPU access
- Gauge configurations accessible at paths specified in ensemble registry
- Python 3.8+, numpy, (optional) cupy for GPU arrays, (optional) h5py for output
- MPI environment to run PyQUDA with multiple GPUs

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
Here we initialize the PyQUDA context with the MPI, and set the lattice partitioning `grid_size`, and then create a `LatticeInfo` object with the lattice dimensions `latt_size`. The `latt_size` in `core.init` will be ignored if `grid_size` is specified, and the MPI size must be equal to the product of `grid_size`. PyQUDA will automatically generate a `grid_size` if only `latt_size` is provided, trying to minimize the communication between different MPI ranks. The `t_boundary=-1` indicates anti-periodic boundary conditions in time, and `anisotropy=1.0` indicates no anisotropy (use `xi_0 / nu` if using an anisotropic lattice, where `xi_0` is the gauge anisotropy and `nu` is the input light speed). The `backend` parameter specifies which Array API we should use to handle lattice fields. `"cupy"` backend could be helpful for GPU acceleration in the "contraction to correlator" step. The `resource_path` is where PyQUDA will look for tuning cache files. Note that if the existing tuning cache is generated by another version of QUDA, you might need to relocate the directory and retune to avoid potential issues with incompatible launching parameters.

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

### Step 6: Save propagator (optional)
This is not always necessary, but you can save the propagator to disk in a format of your choice (e.g. HDF5, NumPy binary) for later analysis.

```python
propag_sh.save("propag_sh.npy", use_fp32=False)
propag_sh.saveH5("propag_sh.h5", tag, annotation=annotation, check=True, use_fp32=False)
```
Here we save the smeared propagator in both NumPy binary format and HDF5 format. The `save` method saves the propagator as a `.npy` file, while the `saveH5` method saves it as an `.h5` file with additional metadata such as a tag, annotation, and a check for data integrity. The `use_fp32` parameter determines whether to save the data in single precision (float32) or double precision (float64).

### Step 7: Contraction to correlator (optional)
If you need to compute correlators, you can perform contractions of the propagators using NumPy or CuPy's contraction utilities or by manually implementing the necessary spin-color contractions. For example, for a rho two-point correlator:
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
rho_2pt_1 = cp.einsum('wtzyx,wtzyxjiba,jk,wtzyxklba,li->t', phase.data, propag_sh.data.conj(), gamma_5 @ gamma_1, propag_sh.data, gamma_1.conj().T @ gamma_5)
rho_2pt_2 = cp.einsum('wtzyx,wtzyxjiba,jk,wtzyxklba,li->t', phase.data, propag_sh.data.conj(), gamma_5 @ gamma_2, propag_sh.data, gamma_2.conj().T @ gamma_5)
rho_2pt_3 = cp.einsum('wtzyx,wtzyxjiba,jk,wtzyxklba,li->t', phase.data, propag_sh.data.conj(), gamma_5 @ gamma_3, propag_sh.data, gamma_3.conj().T @ gamma_5)
# Average over the three spatial polarizations of the rho meson
rho_2pt = (rho_2pt_1 + rho_2pt_2 + rho_2pt_3) / 3
rho_2pt = core.gatherLattice(rho_2pt, [0, -1, -1, -1])
```
Please refer to "lqcd-physics" skills for more details on how to perform contractions for various hadronic correlators. The convention for the spin-color indices in the propagator is (parity, t, z, y, x, spin_snk, spin_src, color_snk, color_src). The `einsum` function is used to perform the necessary contractions to compute the two-point correlator for the rho meson. The `phase.data` is applied to account for the momentum projection $e^{-i\vec{p}\cdot\vec{x}}$. This is reversed comparing to the phase we applied to the source, due to the conjugation in the source operator, please refer to "lqcd-physics" to get the explation of this. Finally, we gather the lattice data from all MPI ranks to obtain the full correlator in the root rank as a function of time. The second argument of `gatherLattice` specifies the dimensions to gather in `tzyx` order, where `0` indicates that we want to gather the data in the $t$ dimension, and `-1` means performing reduction in all three spatial dimensions.

### Step 8: Save output

Save propagators to HDF5 with metadata you need to rebuild the result. Make sure only the root rank (rank 0) writes the output file to avoid conflicts. The output file is named based on the specified path and prefix `/path/to/output/{output_prefix}` and includes a group for the source type and position, with a dataset containing the correlator data.

## Common issues

- **GPU out of memory**: Use more GPUs (increase MPI size), use `backend="numpy"` when initializing PyQUDA.
- **Solver not converging**: Check gauge configuration integrity, try restarting with tighter intermediate tolerance
