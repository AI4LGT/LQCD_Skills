"""Blended all-to-all propagator and elemental kernels for PyQUDA."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from math import prod
from typing import Any, Callable, Sequence

import numpy as np


def _xp(array: Any):
    module = type(array).__module__.split(".")[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _contains_boolean(value: Any) -> bool:
    """Detect Boolean provenance before numerical coercion can erase it."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("numerical sequences must not be string-like")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            return True
        if value.dtype.kind != "O":
            return False
        return any(_contains_boolean(item) for item in value.flat)
    if isinstance(value, SequenceABC):
        return any(_contains_boolean(item) for item in value)
    return False


def _finite_real_tuple(
    values: Sequence[Any], name: str, length: int
) -> tuple[float, ...]:
    if _contains_boolean(values):
        raise TypeError(f"{name} entries must be real, not booleans")
    try:
        actual_length = len(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of length {length}") from error
    if actual_length != length:
        raise ValueError(f"{name} must have length {length}")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} entries must be finite real numbers") from error
    if not all(np.isfinite(value) for value in result):
        raise ValueError(f"{name} entries must be finite")
    return result


def _cuda_aware_mpi_flag(value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("cuda_aware_mpi must be a Boolean opt-in flag")
    return bool(value)


def _integral_tuple(values: Sequence[Any], name: str, length: int) -> tuple[int, ...]:
    """Validate integer-valued lattice metadata without silent truncation."""

    if len(values) != length:
        raise ValueError(f"{name} must have length {length}")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} entries must be integers, not booleans")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} entries must be finite integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} entries must be finite integers")
        result.append(int(numeric))
    return tuple(result)


def _nonboolean_integer(value: Any, name: str, minimum: int) -> int:
    """Return one integer while rejecting Python/NumPy Boolean aliases."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a non-boolean integer")
    integer = int(value)
    if integer < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return integer


def _validate_blending_dimensions(
    volume_color_dim: int, n_ev: int, n_st: int
) -> None:
    volume_color_dim = _nonboolean_integer(
        volume_color_dim, "volume_color_dim", 1
    )
    n_ev = _nonboolean_integer(n_ev, "n_ev", 0)
    n_st = _nonboolean_integer(n_st, "n_st", 0)
    if n_ev > volume_color_dim:
        raise ValueError("n_ev cannot exceed the global color-spatial dimension")
    if n_st > volume_color_dim - n_ev:
        raise ValueError("n_st cannot exceed the high-mode subspace dimension")


def _require_same_backend(reference: Any, *arrays: Any) -> None:
    xp = _xp(reference)
    if any(_xp(array) is not xp for array in arrays):
        raise TypeError("all contraction arrays must use one NumPy/CuPy backend")


def omega(n: int, volume_color_dim: int, n_ev: int, n_st: int) -> float:
    """Return omega_n for a stochastic high-mode label.

    arXiv:2505.01719v2, unnumbered definition immediately after Eq. (1):
    omega_n = (|L2| - n) / (N_st - n), with |L2| = 3V - N_ev.
    """

    _validate_blending_dimensions(volume_color_dim, n_ev, n_st)
    n = _nonboolean_integer(n, "n", 0)
    if n_st < 1:
        raise ValueError("n_st >= 1 is required")
    if n >= n_st:
        raise ValueError("omega_n requires n < n_st")
    high_dim = volume_color_dim - n_ev
    return (high_dim - n) / (n_st - n)


def blending_tuple_weight(
    indices: Sequence[int], volume_color_dim: int, n_ev: int, n_st: int
) -> float:
    """Weight one meson or baryon elemental mode tuple.

    This is Eq. (3) for two indices and Eq. (37) for three indices in
    arXiv:2505.01719v2. Each distinct stochastic high-mode label contributes
    the next omega factor. With n_st=0 only low indices are legal and the
    result is exactly one, i.e. standard distillation.
    """

    _validate_blending_dimensions(volume_color_dim, n_ev, n_st)
    indices = tuple(
        _nonboolean_integer(index, "elemental index", 0) for index in indices
    )
    if n_st == 0:
        if any(index < 0 or index >= n_ev for index in indices):
            raise ValueError(
                "n_st=0 is the distillation limit and only low-mode indices are legal"
            )
        return 1.0
    if any(index < 0 or index >= n_ev + n_st for index in indices):
        raise IndexError("elemental index is outside the blended basis")
    high_labels = {index for index in indices if index >= n_ev}
    weight = 1.0
    for order in range(len(high_labels)):
        weight *= omega(order, volume_color_dim, n_ev, n_st)
    return weight


def _blending_weight_tensor(
    rank: int, volume_color_dim: int, n_ev: int, n_st: int, xp
) -> Any:
    """Build every tuple weight with device broadcasting."""

    n_basis = n_ev + n_st
    if n_st == 0:
        return xp.ones((n_basis,) * rank, dtype=float)
    labels = xp.arange(n_basis) - n_ev
    grids = xp.meshgrid(*([labels] * rank), indexing="ij")
    distinct = xp.zeros((n_basis,) * rank, dtype=np.int32)
    for label in range(n_st):
        distinct += sum(grid == label for grid in grids) > 0
    factors = [1.0]
    for order in range(min(rank, n_st)):
        factors.append(factors[-1] * omega(order, volume_color_dim, n_ev, n_st))
    return xp.asarray(factors)[distinct]


def phase_spatial_links_internal(base_gauge: Any, k_mode: Sequence[float]) -> Any:
    """Independent momentum-smearing link phase used only by blending.

    arXiv:2009.10691v1, Section II.A, Eq. (4):
    U_j(x) -> exp(i 2*pi*k_j/L_j) U_j(x).
    """

    mode = _finite_real_tuple(k_mode, "k_mode", 3)
    lengths = _integral_tuple((
        base_gauge.latt_info.GLx,
        base_gauge.latt_info.GLy,
        base_gauge.latt_info.GLz,
    ), "global spatial extents", 3)
    if any(length <= 0 for length in lengths):
        raise ValueError("global spatial extents must be positive")
    gauge = base_gauge.copy()
    for direction, (component, length) in enumerate(zip(mode, lengths)):
        gauge.data[direction] *= np.exp(2j * np.pi * component / length)
    return gauge


def momentum_smear_internal(
    field: Any,
    base_gauge: Any,
    k_mode: Sequence[float],
    rho: float,
    n_steps: int,
) -> Any:
    """Apply blending's private phased-link Wuppertal kernel."""

    if isinstance(rho, (bool, np.bool_)) or not isinstance(
        rho, (int, float, np.integer, np.floating)
    ):
        raise TypeError("rho must be a finite real number")
    rho_value = float(rho)
    if not np.isfinite(rho_value) or rho_value <= 0:
        raise ValueError("rho must be finite and positive")
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(
        n_steps, (int, np.integer)
    ):
        raise TypeError("n_steps must be a non-boolean integer")
    n_steps_value = int(n_steps)
    if n_steps_value < 1:
        raise ValueError("n_steps must be at least one")
    if rho_value * rho_value >= 2.0 * n_steps_value / 3.0:
        raise ValueError("rho^2 must be smaller than 2*n_steps/3 for positive alpha")
    mode = _finite_real_tuple(k_mode, "k_mode", 3)
    # Validate link metadata before importing PyQUDA or allocating a gauge copy.
    _integral_tuple(
        (
            base_gauge.latt_info.GLx,
            base_gauge.latt_info.GLy,
            base_gauge.latt_info.GLz,
        ),
        "global spatial extents",
        3,
    )
    from pyquda_utils import source

    gauge = phase_spatial_links_internal(base_gauge, mode)
    return source.gaussianSmear(field, gauge, rho_value, n_steps_value)


def spatial_fourier_phase(
    spatial_shape: Sequence[int],
    momentum: Sequence[float],
    array_module=np,
    *,
    global_shape: Sequence[int] | None = None,
    local_offset: Sequence[int] = (0, 0, 0),
) -> Any:
    """Return ``exp(-i p.x_global)`` in local ``(z,y,x)`` order.

    ``momentum`` is the Fourier mode tuple ``(px,py,pz)``.  ``global_shape``
    and ``local_offset`` use array order ``(Gz,Gy,Gx)`` and ``(z0,y0,x0)``.
    Supplying both is mandatory for a spatially decomposed lattice; omitting
    them retains the single-rank convention.
    """

    px, py, pz = _finite_real_tuple(momentum, "momentum", 3)
    nz, ny, nx = _integral_tuple(spatial_shape, "spatial_shape", 3)
    if global_shape is None:
        global_shape = (nz, ny, nx)
    gnz, gny, gnx = _integral_tuple(global_shape, "global_shape", 3)
    z0, y0, x0 = _integral_tuple(local_offset, "local_offset", 3)
    if min(nz, ny, nx, gnz, gny, gnx) < 1 or min(z0, y0, x0) < 0:
        raise ValueError("lattice extents must be positive and offsets nonnegative")
    if z0 + nz > gnz or y0 + ny > gny or x0 + nx > gnx:
        raise ValueError("the local spatial block lies outside global_shape")
    z, y, x = array_module.meshgrid(
        array_module.arange(z0, z0 + nz),
        array_module.arange(y0, y0 + ny),
        array_module.arange(x0, x0 + nx),
        indexing="ij",
    )
    return array_module.exp(
        -2j * array_module.pi * (px * x / gnx + py * y / gny + pz * z / gnz)
    )


def _spatial_allreduce(
    array: Any, spatial_comm: Any | None, cuda_aware_mpi: bool
) -> Any:
    """Sum a local timeslice contraction over spatial ranks.

    ``spatial_comm`` must connect ranks that own pieces of the same global
    timeslices.  For CuPy arrays it must be CUDA-aware; this helper never hides
    a full device-to-host transfer.
    """

    cuda_aware_mpi = _cuda_aware_mpi_flag(cuda_aware_mpi)
    if spatial_comm is None:
        return array
    size = getattr(spatial_comm, "size", None)
    if size is None and hasattr(spatial_comm, "Get_size"):
        size = spatial_comm.Get_size()
    if size == 1:
        return array
    if not hasattr(spatial_comm, "Allreduce"):
        raise TypeError("spatial_comm must provide mpi4py-style buffer Allreduce")
    xp = _xp(array)
    if xp is not np and not cuda_aware_mpi:
        raise RuntimeError(
            "CuPy spatial reduction requires cuda_aware_mpi=True after runtime verification"
        )
    send = xp.ascontiguousarray(array)
    reduced = xp.empty_like(send)
    try:
        spatial_comm.Allreduce(send, reduced)
    except Exception as error:
        if xp is not np:
            raise RuntimeError(
                "CuPy buffer Allreduce failed; verify CUDA-aware mpi4py/MPI support"
            ) from error
        raise
    return reduced


def meson_elemental(
    eigvecs: Any,
    momentum: Sequence[float],
    n_ev: int,
    n_st: int,
    *,
    global_spatial_shape: Sequence[int] | None = None,
    local_spatial_offset: Sequence[int] | None = None,
    spatial_comm: Any | None = None,
    cuda_aware_mpi: bool = False,
) -> Any:
    """Build the blended meson elemental Phi_ab(t).

    `eigvecs` has shape `(N,t,z,y,x,3)`. The contraction is the pion
    elemental of arXiv:2505.01719v2 Eq. (31); n_st=0 gives the standard
    distillation operator of Eq. (32).
    """

    cuda_aware_mpi = _cuda_aware_mpi_flag(cuda_aware_mpi)
    n_ev = _nonboolean_integer(n_ev, "n_ev", 0)
    n_st = _nonboolean_integer(n_st, "n_st", 0)
    if n_ev + n_st == 0:
        raise ValueError("n_ev and n_st must be nonnegative with a nonempty basis")
    xp = _xp(eigvecs)
    n_basis = n_ev + n_st
    if eigvecs.ndim != 6 or eigvecs.shape[0] < n_basis or eigvecs.shape[-1] != 3:
        raise ValueError("eigvecs must have shape (N,t,z,y,x,3)")
    vectors = eigvecs[:n_basis]
    local_spatial_shape = tuple(int(value) for value in vectors.shape[2:5])
    comm_size = None if spatial_comm is None else getattr(spatial_comm, "size", None)
    if spatial_comm is not None and comm_size is None and hasattr(spatial_comm, "Get_size"):
        comm_size = spatial_comm.Get_size()
    if spatial_comm is not None and comm_size != 1:
        if global_spatial_shape is None or local_spatial_offset is None:
            raise ValueError(
                "multi-rank elementals require global_spatial_shape and local_spatial_offset"
            )
    if global_spatial_shape is None:
        global_spatial_shape = local_spatial_shape
    if local_spatial_offset is None:
        local_spatial_offset = (0, 0, 0)
    global_spatial_shape = _integral_tuple(
        global_spatial_shape, "global_spatial_shape", 3
    )
    volume_color_dim = int(prod(global_spatial_shape) * 3)
    _validate_blending_dimensions(volume_color_dim, n_ev, n_st)
    phase = spatial_fourier_phase(
        local_spatial_shape,
        momentum,
        xp,
        global_shape=global_spatial_shape,
        local_offset=local_spatial_offset,
    )
    elemental = xp.einsum(
        "atzyxc,zyx,btzyxc->tab", vectors.conj(), phase, vectors
    )
    elemental = _spatial_allreduce(elemental, spatial_comm, cuda_aware_mpi)
    weights = _blending_weight_tensor(2, volume_color_dim, n_ev, n_st, xp)
    return elemental * weights[None]


def baryon_elemental(
    eigvecs: Any,
    momentum: Sequence[float],
    n_ev: int,
    n_st: int,
    *,
    global_spatial_shape: Sequence[int] | None = None,
    local_spatial_offset: Sequence[int] | None = None,
    spatial_comm: Any | None = None,
    cuda_aware_mpi: bool = False,
) -> Any:
    """Build epsilon_abc V_i^a V_j^b V_k^c with blended weights."""

    cuda_aware_mpi = _cuda_aware_mpi_flag(cuda_aware_mpi)
    n_ev = _nonboolean_integer(n_ev, "n_ev", 0)
    n_st = _nonboolean_integer(n_st, "n_st", 0)
    if n_ev + n_st == 0:
        raise ValueError("n_ev and n_st must be nonnegative with a nonempty basis")
    xp = _xp(eigvecs)
    n_basis = n_ev + n_st
    if eigvecs.ndim != 6 or eigvecs.shape[0] < n_basis or eigvecs.shape[-1] != 3:
        raise ValueError("eigvecs must have shape (N,t,z,y,x,3)")
    vectors = eigvecs[:n_basis]
    local_spatial_shape = tuple(int(value) for value in vectors.shape[2:5])
    comm_size = None if spatial_comm is None else getattr(spatial_comm, "size", None)
    if spatial_comm is not None and comm_size is None and hasattr(spatial_comm, "Get_size"):
        comm_size = spatial_comm.Get_size()
    if spatial_comm is not None and comm_size != 1:
        if global_spatial_shape is None or local_spatial_offset is None:
            raise ValueError(
                "multi-rank elementals require global_spatial_shape and local_spatial_offset"
            )
    if global_spatial_shape is None:
        global_spatial_shape = local_spatial_shape
    if local_spatial_offset is None:
        local_spatial_offset = (0, 0, 0)
    global_spatial_shape = _integral_tuple(
        global_spatial_shape, "global_spatial_shape", 3
    )
    volume_color_dim = int(prod(global_spatial_shape) * 3)
    _validate_blending_dimensions(volume_color_dim, n_ev, n_st)
    phase = spatial_fourier_phase(
        local_spatial_shape,
        momentum,
        xp,
        global_shape=global_spatial_shape,
        local_offset=local_spatial_offset,
    )
    epsilon = xp.zeros((3, 3, 3), dtype=vectors.dtype)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
    elemental = xp.einsum(
        "abc,itzyxa,jtzyxb,ktzyxc,zyx->tijk",
        epsilon,
        vectors,
        vectors,
        vectors,
        phase,
        optimize=True,
    )
    elemental = _spatial_allreduce(elemental, spatial_comm, cuda_aware_mpi)
    weights = _blending_weight_tensor(3, volume_color_dim, n_ev, n_st, xp)
    return elemental * weights[None]


def generate_perambulator(
    n_modes: int,
    n_spin: int,
    make_source: Callable[[int, int], Any],
    solve: Callable[[Any], Any],
    project_sink: Callable[[Any], Any],
) -> Any:
    """Invert and project every mode-spin source into a perambulator.

    The callback boundary keeps PyQUDA field construction explicit in the
    generated batch program. This implements the projected all-to-all
    propagator of arXiv:2505.01719v2 Eq. (4).
    """

    n_modes = _nonboolean_integer(n_modes, "n_modes", 1)
    n_spin = _nonboolean_integer(n_spin, "n_spin", 1)
    columns = []
    projected_shape = None
    # Source construction and solve submission are Python orchestration, while
    # each inversion remains in QUDA and each projected column stays on its
    # array backend. For many modes, supply MRHS-aware callbacks or add a batched
    # `solve_many` API; moving these solves to CPU would be a severe bottleneck.
    for source_mode in range(n_modes):
        spin_columns = []
        for source_spin in range(n_spin):
            solution = solve(make_source(source_mode, source_spin))
            projected = project_sink(solution)
            if projected.ndim != 3:
                raise ValueError(
                    "project_sink must return (t,sink_spin,sink_mode)"
                )
            if projected.shape[1:] != (n_spin, n_modes):
                raise ValueError(
                    "project_sink output must have sink dimensions (n_spin,n_modes)"
                )
            if projected_shape is None:
                projected_shape = projected.shape
                projected_backend = _xp(projected)
            elif projected.shape != projected_shape:
                raise ValueError("all projected perambulator columns must have one shape")
            elif _xp(projected) is not projected_backend:
                raise TypeError("all projected perambulator columns must use one backend")
            spin_columns.append(projected)
        columns.append(spin_columns)
    xp = _xp(columns[0][0])
    # (t,sink_spin,sink_mode) ->
    # (t,sink_spin,source_spin,sink_mode,source_mode)
    return xp.stack([xp.stack(column, axis=2) for column in columns], axis=4)


def meson_two_point(
    phi_source: Any,
    phi_sink: Any,
    perambulator: Any,
    gamma_source_bar: Any,
    gamma_sink: Any,
    gamma5: Any,
    t_source: int,
) -> Any:
    """Contract a connected blended meson two-point function.

    ``t_source`` is validated before array/backend access. The backward line is
    reconstructed as
    `(gamma5 x I) tau^dagger (gamma5 x I)`. This is the bundled
    implementation convention; arXiv:2505.01719v2 supports the blending
    weights/elementals but is not used here as blanket validation of this full
    Wick contraction. A bare `tau.conj().T` is not equivalent for nontrivial
    spin structure.
    """

    t_source = _nonboolean_integer(t_source, "t_source", 0)
    xp = _xp(perambulator)
    _require_same_backend(
        perambulator,
        phi_source,
        phi_sink,
        gamma_source_bar,
        gamma_sink,
        gamma5,
    )
    if perambulator.ndim != 5:
        raise ValueError(
            "perambulator must have (t,sink_spin,source_spin,sink_mode,source_mode)"
        )
    nt, ns, _, n_modes, _ = perambulator.shape
    if perambulator.shape[2] != ns or perambulator.shape[4] != n_modes:
        raise ValueError("perambulator spin and mode source/sink dimensions must match")
    if phi_source.shape != (nt, n_modes, n_modes) or phi_sink.shape != phi_source.shape:
        raise ValueError("source and sink elementals must have shape (t,mode,mode)")
    if gamma_source_bar.shape != (ns, ns) or gamma_sink.shape != (ns, ns):
        raise ValueError("source and sink gamma matrices must match the spin dimension")
    if gamma5.shape != (ns, ns):
        raise ValueError("gamma5 must match the spin dimension")
    if not 0 <= t_source < nt:
        raise ValueError("t_source is outside the perambulator time extent")
    tau = perambulator.transpose(0, 1, 3, 2, 4).reshape(nt, ns * n_modes, ns * n_modes)
    source_op = xp.kron(gamma_source_bar, phi_source[t_source].conj())
    gamma5_mode = xp.kron(gamma5, xp.eye(n_modes, dtype=perambulator.dtype))
    sink_op = xp.einsum("ij,tab->tiajb", gamma_sink, phi_sink).reshape(
        nt, ns * n_modes, ns * n_modes
    )
    tau_backward = gamma5_mode @ tau.conj().swapaxes(-1, -2) @ gamma5_mode
    return -xp.einsum(
        "tij,tjk,kl,tli->t", sink_op, tau, source_op, tau_backward, optimize=True
    )


def baryon_two_point(
    phi_source: Any,
    phi_sink: Any,
    tau_u1: Any,
    tau_u2: Any,
    tau_d: Any,
    spin_projector: Any,
    diquark_sink: Any,
    diquark_source_bar: Any,
    t_source: int,
) -> Any:
    """Contract direct-minus-exchange uud terms after strict source-time validation."""

    t_source = _nonboolean_integer(t_source, "t_source", 0)
    xp = _xp(tau_u1)
    taus = (tau_u1, tau_u2, tau_d)
    if any(_xp(tau) is not xp for tau in taus[1:]):
        raise TypeError("all perambulators must use one array backend")
    _require_same_backend(
        tau_u1,
        phi_source,
        phi_sink,
        spin_projector,
        diquark_sink,
        diquark_source_bar,
    )
    if any(tau.shape != tau_u1.shape for tau in taus[1:]):
        raise ValueError("all perambulators must have one shape")
    if tau_u1.ndim != 5:
        raise ValueError(
            "perambulators must have (t,sink_spin,source_spin,sink_mode,source_mode)"
        )
    nt, ns, source_ns, n_modes, source_n_modes = tau_u1.shape
    if source_ns != ns or source_n_modes != n_modes:
        raise ValueError("perambulator source/sink spin and mode extents must match")
    expected_phi = (nt, n_modes, n_modes, n_modes)
    if phi_source.shape != expected_phi or phi_sink.shape != expected_phi:
        raise ValueError("baryon elementals must have shape (t,mode,mode,mode)")
    if any(
        matrix.shape != (ns, ns)
        for matrix in (spin_projector, diquark_sink, diquark_source_bar)
    ):
        raise ValueError("baryon spin kernels must match the perambulator spin extent")
    if not 0 <= t_source < nt:
        raise ValueError("t_source is outside the perambulator time extent")
    source = phi_source[t_source].conj()
    direct = xp.einsum(
        "tabc,ABC,iI,jk,JK,tiIaA,tjJbB,tkKcC->t",
        phi_sink,
        source,
        spin_projector,
        diquark_sink,
        diquark_source_bar,
        tau_u1,
        tau_u2,
        tau_d,
        optimize=True,
    )
    exchange = xp.einsum(
        "tabc,ABC,iI,jk,JK,tiJaB,tjIbA,tkKcC->t",
        phi_sink,
        source,
        spin_projector,
        diquark_sink,
        diquark_source_bar,
        tau_u1,
        tau_u2,
        tau_d,
        optimize=True,
    )
    return direct - exchange


__all__ = [
    "omega",
    "blending_tuple_weight",
    "phase_spatial_links_internal",
    "momentum_smear_internal",
    "spatial_fourier_phase",
    "meson_elemental",
    "baryon_elemental",
    "generate_perambulator",
    "meson_two_point",
    "baryon_two_point",
]
