import numpy as np

try:
    import cupy as cp
except ImportError:  # Pure NumPy reconstruction remains available with custom gammas.
    cp = None

try:
    from pyquda_utils import core, gamma
except ImportError:  # Deferred unless default gammas or MPI gather are requested.
    core = None
    gamma = None


gamma_ops = None if gamma is None else {
    't': gamma.gamma(8),
    'z': gamma.gamma(4),
    'y': gamma.gamma(2),
    'x': gamma.gamma(1),
    'unit': gamma.gamma(0),
}


def _validate_hisq_shape(hisq_prop, lattice_shape):
    expected = tuple(lattice_shape) + (3, 3)
    if hisq_prop.shape != expected:
        raise ValueError(f"hisq_prop must have shape {expected}, got {hisq_prop.shape}")


def _normalize_source_coordinate(source, global_shape):
    """Return a global source coordinate in `(t,z,y,x)` order.

    A scalar source is retained for backward compatibility and means
    `(t_source,0,0,0)`.
    """

    if isinstance(source, (bool, np.bool_)):
        raise TypeError("source coordinate must contain integers, not booleans")
    if np.isscalar(source):
        try:
            numeric = float(source)
        except (TypeError, ValueError) as error:
            raise TypeError("source coordinate must contain finite integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("source coordinate must contain finite integers")
        coordinate = (int(numeric), 0, 0, 0)
    else:
        if len(source) != 4:
            raise ValueError("source must be t_source or global (t,z,y,x)")
        coordinate_values = []
        for value in source:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("source coordinate must contain integers")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError("source coordinate must contain finite integers") from error
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError("source coordinate must contain finite integers")
            coordinate_values.append(int(numeric))
        coordinate = tuple(coordinate_values)
    if any(value < 0 or value >= extent for value, extent in zip(coordinate, global_shape)):
        raise ValueError("source coordinate lies outside the global lattice")
    return coordinate


def _validate_recovery_inputs(gamma_ops, source, global_shape):
    if gamma_ops is None:
        raise ImportError(
            "default gamma_ops require PyQUDA; pass an explicit NumPy gamma_ops mapping"
        )
    required = {'t', 'z', 'y', 'x', 'unit'}
    if set(gamma_ops) != required:
        raise ValueError(f"gamma_ops must contain exactly {sorted(required)}")
    if any(gamma_ops[name].shape != (4, 4) for name in required):
        raise ValueError("every gamma_ops matrix must have shape (4,4)")
    for name in required:
        matrix = gamma_ops[name]
        xp = cp if cp is not None and isinstance(matrix, cp.ndarray) else np
        try:
            finite = xp.isfinite(matrix).all()
        except TypeError as error:
            raise TypeError("every gamma_ops matrix must be numerical") from error
        if xp is np:
            is_finite = bool(np.asarray(finite).item())
        else:
            is_finite = bool(np.asarray(cp.asnumpy(finite)).item())
        if not is_finite:
            raise ValueError("every gamma_ops matrix must be finite")
    return _normalize_source_coordinate(source, global_shape)


def _xp(array):
    return cp if cp is not None and isinstance(array, cp.ndarray) else np


def _recover_local(hisq_prop, coords, source_coord, gamma_ops):
    """Apply `Omega(x) G_chi(x,x0) Omega(x0)^dagger`."""

    xp = _xp(hisq_prop)
    # The public contract deliberately supports only NumPy/CuPy complex64 and
    # complex128.  ``np.issubdtype(..., np.complexfloating)`` also accepts
    # platform-dependent extended complex dtypes (for example complex256),
    # which would silently widen the reconstruction and violate the declared
    # precision boundary.  Compare exact dtypes before any allocation or
    # matrix arithmetic so unsupported precision fails closed.
    if np.dtype(hisq_prop.dtype) not in {
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    }:
        raise TypeError("hisq_prop must use complex64 or complex128")
    matrices = {
        name: (
            cp.asnumpy(value).astype(hisq_prop.dtype, copy=False)
            if cp is not None and xp is np and isinstance(value, cp.ndarray)
            else xp.asarray(value, dtype=hisq_prop.dtype)
        )
        for name, value in gamma_ops.items()
    }
    lattice_shape = hisq_prop.shape[:-2]
    left = xp.broadcast_to(matrices['unit'], lattice_shape + (4, 4)).copy()
    for direction, coordinate in zip(('t', 'z', 'y', 'x'), coords):
        gamma_sink = xp.where(
            (coordinate % 2)[..., None, None] == 1,
            matrices[direction],
            matrices['unit'],
        )
        left = gamma_sink @ left
    source = matrices['unit']
    for direction, coordinate in zip(('t', 'z', 'y', 'x'), source_coord):
        if coordinate % 2:
            source = matrices[direction] @ source
    spin_factor = left @ source.conj().T
    return spin_factor[..., :, :, None, None] * hisq_prop[..., None, None, :, :]


def hisq_to_wilson_evengrid(latt_info, hisq_prop, tsource, gamma_ops = gamma_ops):
    """Reconstruct a naive four-spin field in Wilson-like storage.

    ``tsource`` accepts either the legacy global time index (spatial source at
    the origin) or the full global source coordinate ``(t,z,y,x)``.
    """

    Nx, Ny, Nz, Nt = latt_info.GLx, latt_info.GLy, latt_info.GLz, latt_info.GLt
    source_coord = _validate_recovery_inputs(gamma_ops, tsource, (Nt, Nz, Ny, Nx))
    _validate_hisq_shape(hisq_prop, (Nt, Nz, Ny, Nx))
    xp = _xp(hisq_prop)
    coords = xp.meshgrid(
        xp.arange(Nt), xp.arange(Nz), xp.arange(Ny), xp.arange(Nx), indexing='ij'
    )
    return _recover_local(hisq_prop, coords, source_coord, gamma_ops)


def hisq_to_wilson_evengrid_MPI(
    latt_info, hisq_prop, tsource, gamma_ops = gamma_ops, gather = True
):
    if not isinstance(gather, (bool, np.bool_)):
        raise TypeError("gather must be a Python or NumPy Boolean")
    gather = bool(gather)
    Lx, Ly, Lz, Lt = latt_info.size
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    global_shape = (Lt * Gt, Lz * Gz, Ly * Gy, Lx * Gx)
    source_coord = _validate_recovery_inputs(gamma_ops, tsource, global_shape)
    _validate_hisq_shape(hisq_prop, (Lt, Lz, Ly, Lx))
    xp = _xp(hisq_prop)
    coords = xp.meshgrid(
        xp.arange(gt * Lt, (gt + 1) * Lt),
        xp.arange(gz * Lz, (gz + 1) * Lz),
        xp.arange(gy * Ly, (gy + 1) * Ly),
        xp.arange(gx * Lx, (gx + 1) * Lx),
        indexing='ij',
    )
    wilson_prop = _recover_local(hisq_prop, coords, source_coord, gamma_ops)
    if not gather:
        return wilson_prop

    # `core.gatherLattice` is NumPy/MPI-only. This optional final transfer is
    # the full field and can dominate runtime; production contractions should
    # use gather=False and reduce observables before moving them to the host.
    if core is None:
        raise ImportError("gather=True requires pyquda_utils.core")
    host_prop = cp.asnumpy(wilson_prop) if cp is not None and xp is cp else wilson_prop
    wilson_recover = core.gatherLattice(host_prop, [0, 1, 2, 3])
    # Rank ownership belongs to PyQUDA's communicator API. The pinned
    # LatticeInfo also snapshots `mpi_rank`, but ownership at this return
    # boundary should follow the active communicator rather than potentially
    # stale geometry-object metadata.
    return wilson_recover if core.getMPIRank() == 0 else None


__all__ = [
    "gamma_ops",
    "hisq_to_wilson_evengrid",
    "hisq_to_wilson_evengrid_MPI",
]
