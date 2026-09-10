"""PyQUDA momentum-smearing kernels for propagators and sequential sources."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np


SpatialMode = Tuple[float, float, float]


@dataclass(frozen=True)
class MomentumSmearResult:
    """Matched meson lines; ``k_mode`` records the source spectator +k."""

    mode: str
    k_mode: SpatialMode
    spectator: Any
    active: Any
    spectator_at_sink: Any
    active_at_sink: Any
    active_sink_k: Optional[SpatialMode]


@dataclass(frozen=True)
class BaryonMomentumSmearResult:
    """One reusable baryon line; ``k_mode`` records its source-smearing k."""

    mode: str
    k_mode: SpatialMode
    propagator: Any
    propagator_at_sink: Any
    active_sink_k: Optional[SpatialMode]


@dataclass(frozen=True)
class SingleMomentumSmearResult:
    """Generic single-line compatibility result."""

    mode: SpatialMode
    propagator: Any
    sink_propagator: Any
    sink_k_mode: Optional[SpatialMode] = None


def _xp(array: Any):
    if type(array).__module__.split(".")[0] == "cupy":
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


def _global_extents_xyzt(latt_info: Any) -> tuple[int, int, int, int]:
    if all(hasattr(latt_info, name) for name in ("GLx", "GLy", "GLz", "GLt")):
        values = tuple(getattr(latt_info, name) for name in ("GLx", "GLy", "GLz", "GLt"))
    elif hasattr(latt_info, "global_size"):
        values = tuple(latt_info.global_size)
    elif hasattr(latt_info, "size") and tuple(getattr(latt_info, "grid_size", (1, 1, 1, 1))) == (1, 1, 1, 1):
        values = tuple(latt_info.size)
    else:
        raise ValueError("latt_info must expose global (x,y,z,t) extents")
    if len(values) != 4:
        raise ValueError("global lattice extents must contain (x,y,z,t)")
    extents = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("global lattice extents must be non-boolean integers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("global lattice extents must be finite integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
            raise ValueError("global lattice extents must be positive integers")
        extents.append(int(numeric))
    return tuple(extents)


def _integer_coordinate(
    values: Sequence[Any],
    name: str,
    extents: Sequence[int],
) -> tuple[int, ...]:
    if len(values) != len(extents):
        raise ValueError(f"{name} must have length {len(extents)}")
    coordinate = []
    for value, extent in zip(values, extents):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} entries must be integers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} entries must be finite integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} entries must be finite integers")
        integer = int(numeric)
        if integer < 0 or integer >= extent:
            raise ValueError(f"{name} lies outside the global lattice")
        coordinate.append(integer)
    return tuple(coordinate)


def _integer_modes(values: Sequence[Any], name: str, length: int) -> tuple[int, ...]:
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}")
    modes = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} entries must be integers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} entries must be finite integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} entries must be finite integers")
        modes.append(int(numeric))
    return tuple(modes)


def _global_time(time: Any, name: str, latt_info: Any) -> int:
    return _integer_coordinate((time,), name, (_global_extents_xyzt(latt_info)[3],))[0]


def _mode3(mode: Sequence[float]) -> SpatialMode:
    if _contains_boolean(mode):
        raise TypeError("momentum-smearing mode entries must be real, not booleans")
    try:
        length = len(mode)
    except TypeError as error:
        raise TypeError("momentum-smearing mode must be a three-component sequence") from error
    if length != 3:
        raise ValueError("momentum-smearing mode must contain (kx, ky, kz)")
    try:
        result = tuple(float(component) for component in mode)
    except (TypeError, ValueError) as error:
        raise TypeError("momentum-smearing mode entries must be finite real numbers") from error
    if not all(np.isfinite(component) for component in result):
        raise ValueError("momentum-smearing mode entries must be finite")
    return result


def _spatial_extents_xyz(latt_info: Any) -> tuple[int, int, int]:
    if not all(hasattr(latt_info, name) for name in ("GLx", "GLy", "GLz")):
        raise ValueError("lattice metadata must expose GLx, GLy, and GLz")
    extents = []
    for value in (latt_info.GLx, latt_info.GLy, latt_info.GLz):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("global spatial extents must be non-boolean integers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("global spatial extents must be finite integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
            raise ValueError("global spatial extents must be positive integers")
        extents.append(int(numeric))
    return tuple(extents)


def negate_mode(mode: Sequence[float]) -> SpatialMode:
    """Return a validated three-component mode with reversed sign."""

    return tuple(-component for component in _mode3(mode))


def phase_spatial_links(base_gauge: Any, k_mode: Sequence[float]) -> Any:
    """Return a gauge copy with momentum phases on spatial links.

    arXiv:1602.05525, Eq. (24), implements momentum smearing through
    U_j(x) -> exp(i 2*pi*k_j/L_j) U_j(x). Time links are unchanged.
    The components of ``k_mode`` may be fractional mode numbers.
    """

    mode = _mode3(k_mode)
    if not hasattr(base_gauge, "latt_info"):
        raise ValueError("base_gauge must expose latt_info")
    lengths = _spatial_extents_xyz(base_gauge.latt_info)
    phased_gauge = base_gauge.copy()
    for direction, (component, length) in enumerate(zip(mode, lengths)):
        phased_gauge.data[direction] *= np.exp(2j * np.pi * component / length)
    return phased_gauge


def _lattice_signature(latt_info: Any) -> tuple[Any, ...]:
    """Return lattice extents *and rank coordinates* for compatibility checks.

    Equal extents do not identify the same local block on a decomposed lattice.
    ``grid_coord`` is therefore mandatory whenever ``prod(grid_size)>1``.
    """

    signature = []
    for name in ("global_size", "size", "grid_size"):
        if hasattr(latt_info, name):
            values = _integer_modes(getattr(latt_info, name), name, 4)
            if name != "grid_coord" and any(value < 1 for value in values):
                raise ValueError(f"{name} entries must be positive")
            signature.append((name, values))
    if not signature and all(
        hasattr(latt_info, name) for name in ("GLx", "GLy", "GLz", "GLt")
    ):
        signature.append(
            (
                "global_size",
                _global_extents_xyzt(latt_info),
            )
        )
    grid_size = None
    if hasattr(latt_info, "grid_size"):
        grid_size = _integer_modes(latt_info.grid_size, "grid_size", 4)
        if any(value < 1 for value in grid_size):
            raise ValueError("grid_size entries must be positive")
    if hasattr(latt_info, "grid_coord"):
        grid_coord = _integer_modes(latt_info.grid_coord, "grid_coord", 4)
        if any(value < 0 for value in grid_coord):
            raise ValueError("grid_coord entries must be nonnegative")
        if grid_size is not None and any(
            coordinate >= extent
            for coordinate, extent in zip(grid_coord, grid_size)
        ):
            raise ValueError("grid_coord lies outside grid_size")
        signature.append(("grid_coord", grid_coord))
    elif grid_size is not None and int(np.prod(grid_size)) > 1:
        raise ValueError("multi-rank lattice metadata must expose grid_coord")
    return tuple(signature)


def _validate_smearing_domain(radius: float, n_steps: int) -> None:
    if isinstance(radius, (bool, np.bool_)) or not isinstance(
        radius, (int, float, np.integer, np.floating)
    ):
        raise TypeError("radius must be a finite real number")
    radius_value = float(radius)
    if not np.isfinite(radius_value) or radius_value <= 0:
        raise ValueError("radius must be finite and positive")
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(
        n_steps, (int, np.integer)
    ):
        raise TypeError("n_steps must be a non-boolean integer")
    n_steps_value = int(n_steps)
    if n_steps_value < 1:
        raise ValueError("n_steps must be at least one")
    denominator = 4.0 * n_steps_value / radius_value**2 - 6.0
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            "PyQUDA gaussianSmear requires positive alpha: radius^2 < 2*n_steps/3"
        )


def _validate_solver_options(mrhs: int, restart: int) -> None:
    if isinstance(mrhs, (bool, np.bool_)) or not isinstance(
        mrhs, (int, np.integer)
    ):
        raise TypeError("mrhs must be a non-boolean integer")
    if isinstance(restart, (bool, np.bool_)) or not isinstance(
        restart, (int, np.integer)
    ):
        raise TypeError("restart must be a non-boolean integer")
    if mrhs < 1:
        raise ValueError("mrhs must be a positive integer")
    if restart < 0:
        raise ValueError("restart must be a nonnegative integer")


def _validate_kernel_metadata(
    field: Any,
    base_gauge: Any,
    latt_info: Any,
    k_mode: Sequence[float],
    radius: float,
    n_steps: int,
) -> SpatialMode:
    """Validate every pure kernel input before importing or smearing."""

    _validate_smearing_domain(radius, n_steps)
    mode = _mode3(k_mode)
    if not hasattr(base_gauge, "latt_info"):
        raise ValueError("base_gauge must expose latt_info")
    _spatial_extents_xyz(base_gauge.latt_info)
    gauge_signature = _lattice_signature(base_gauge.latt_info)
    requested_signature = _lattice_signature(latt_info)
    if gauge_signature != requested_signature:
        raise ValueError("base_gauge and latt_info global/local/grid extents must match")
    if (
        hasattr(field, "latt_info")
        and _lattice_signature(field.latt_info) != requested_signature
    ):
        raise ValueError("field and gauge lattice decompositions must match")
    return mode


def momentum_smear_kernel(
    field: Any,
    base_gauge: Any,
    latt_info: Any,
    k_mode: Sequence[float],
    radius: float,
    n_steps: int,
) -> Any:
    """Apply link-phased Wuppertal smearing to a PyQUDA lattice field.

    The phase is inserted before every hopping iteration through the phased
    gauge links. Multiplying a completed Gaussian-smeared field by a plane
    wave is not the momentum-smearing operator of arXiv:1602.05525 Eq. (24).
    """

    mode = _validate_kernel_metadata(
        field, base_gauge, latt_info, k_mode, radius, n_steps
    )
    from pyquda_utils import source

    phased_gauge = phase_spatial_links(base_gauge, mode)
    return source.gaussianSmear(field, phased_gauge, radius, n_steps)


def apply_momentum_smear_source(
    dirac: Any,
    source_field: Any,
    base_gauge: Any,
    k_mode: Sequence[float],
    rho: float,
    n_steps: int,
    mrhs: int = 1,
    restart: int = 0,
) -> Any:
    """Smear a source and invert it, returning a PyQUDA propagator."""

    _validate_solver_options(mrhs, restart)
    mode = _validate_kernel_metadata(
        source_field,
        base_gauge,
        base_gauge.latt_info,
        k_mode,
        rho,
        n_steps,
    )
    from pyquda_utils import core

    smeared_source = momentum_smear_kernel(
        source_field, base_gauge, base_gauge.latt_info, mode, rho, n_steps
    )
    with dirac.useGauge(base_gauge):
        return core.invertPropagator(dirac, smeared_source, mrhs=mrhs, restart=restart)


def apply_momentum_smear_sink(
    propagator: Any,
    base_gauge: Any,
    k_mode: Sequence[float],
    rho: float,
    n_steps: int,
) -> Any:
    """Apply the same spatial kernel to the sink end of a propagator."""

    _validate_smearing_domain(rho, n_steps)
    mode = _mode3(k_mode)
    if not hasattr(base_gauge, "latt_info"):
        raise ValueError("base_gauge must expose latt_info")
    return momentum_smear_kernel(
        propagator, base_gauge, base_gauge.latt_info, mode, rho, n_steps
    )


def apply_momentum_smear_seqprop(
    dirac: Any,
    sequential_source: Any,
    t_sink: int,
    base_gauge: Any,
    sink_k_mode: Optional[Sequence[float]],
    rho: float,
    n_steps: int,
    mrhs: int = 1,
    restart: int = 0,
    *,
    already_smeared: Optional[bool] = None,
) -> Any:
    """Perform a sequential inversion with an explicit one-smear contract.

    ``already_smeared=True`` requires ``sink_k_mode=None`` and inverts the
    supplied source directly. ``already_smeared=False`` requires a sink mode
    and applies the momentum-smearing kernel exactly once before inversion.
    Omitting the flag fails closed so a source returned by
    :func:`build_sequential_source` cannot be silently smeared twice.
    """

    _validate_solver_options(mrhs, restart)
    _validate_smearing_domain(rho, n_steps)
    t_sink = _global_time(t_sink, "t_sink", base_gauge.latt_info)
    if not isinstance(already_smeared, (bool, np.bool_)):
        raise ValueError("already_smeared must be explicitly True or False")
    source_signature = (
        _lattice_signature(sequential_source.latt_info)
        if hasattr(sequential_source, "latt_info")
        else None
    )
    gauge_signature = _lattice_signature(base_gauge.latt_info)
    if source_signature is not None and source_signature != gauge_signature:
        raise ValueError("sequential source and gauge lattice decompositions must match")
    canonical_sink_mode = None
    if already_smeared:
        if sink_k_mode is not None:
            raise ValueError(
                "sink_k_mode must be None when the sequential source is already smeared"
            )
    else:
        if sink_k_mode is None:
            raise ValueError(
                "sink_k_mode is required when the sequential source is not already smeared"
            )
        canonical_sink_mode = _validate_kernel_metadata(
            sequential_source,
            base_gauge,
            base_gauge.latt_info,
            sink_k_mode,
            rho,
            n_steps,
        )
    from pyquda_utils import core

    if already_smeared:
        inversion_source = sequential_source
    else:
        inversion_source = momentum_smear_kernel(
            sequential_source,
            base_gauge,
            base_gauge.latt_info,
            canonical_sink_mode,
            rho,
            n_steps,
        )
    with dirac.useGauge(base_gauge):
        return core.invertSequential(
            dirac, inversion_source, t_sink, mrhs=mrhs, restart=restart
        )


def momentum_smear_propagator(
    dirac: Any,
    source_field: Any,
    base_gauge: Any,
    source_k_mode: Sequence[float],
    rho: float,
    n_steps: int,
    sink_k_mode: Optional[Sequence[float]] = None,
    mrhs: int = 1,
    restart: int = 0,
) -> SingleMomentumSmearResult:
    """Produce a source-smeared propagator and optional S-to-S endpoint."""

    _validate_smearing_domain(rho, n_steps)
    _validate_solver_options(mrhs, restart)
    mode = _mode3(source_k_mode)
    canonical_sink_mode = (
        None if sink_k_mode is None else _mode3(sink_k_mode)
    )
    propagator = apply_momentum_smear_source(
        dirac, source_field, base_gauge, mode, rho, n_steps, mrhs, restart
    )
    sink = propagator
    if canonical_sink_mode is not None:
        sink = apply_momentum_smear_sink(
            propagator, base_gauge, canonical_sink_mode, rho, n_steps
        )
    return SingleMomentumSmearResult(
        mode=mode,
        propagator=propagator,
        sink_propagator=sink,
        sink_k_mode=canonical_sink_mode,
    )


def momentum_smear_meson(
    latt_info: Any,
    base_gauge: Any,
    dirac: Any,
    x_src: Sequence[int],
    k_mode: Sequence[float],
    radius: float,
    n_steps: int,
    sink_mode: str = "s2s",
    sink_k_mode: Optional[Sequence[float]] = None,
    mrhs: int = 1,
    restart: int = 0,
) -> MomentumSmearResult:
    """Prepare a +k/-k meson pair with optional independent sink momentum.

    Link phasing follows arXiv:1602.05525 Eq. (24). The
    spectator uses +k and the active line uses -k at each smeared endpoint.
    ``k_mode`` controls the source. For S-to-S, ``sink_k_mode`` controls the
    sink and defaults to ``k_mode``, preserving forward-matrix-element calls.
    S-to-P has no sink-smearing endpoint and therefore rejects a non-``None``
    ``sink_k_mode`` instead of silently discarding it.
    The exact hadron momentum is imposed by the full-hadron Fourier phase.
    """

    if sink_mode not in ("s2p", "s2s"):
        raise ValueError("sink_mode must be 's2p' or 's2s'")
    if sink_mode == "s2p" and sink_k_mode is not None:
        raise ValueError("sink_k_mode is only valid for sink_mode='s2s'")
    _validate_solver_options(mrhs, restart)
    _validate_smearing_domain(radius, n_steps)
    if _lattice_signature(base_gauge.latt_info) != _lattice_signature(latt_info):
        raise ValueError("base_gauge and latt_info lattice decompositions must match")
    _spatial_extents_xyz(base_gauge.latt_info)
    x_src = _integer_coordinate(
        x_src, "x_src", _global_extents_xyzt(latt_info)
    )
    k_plus = _mode3(k_mode)
    k_minus = negate_mode(k_plus)
    supplied_sink_mode = (
        None if sink_k_mode is None else _mode3(sink_k_mode)
    )
    sink_plus = k_plus if supplied_sink_mode is None else supplied_sink_mode
    sink_minus = negate_mode(sink_plus)
    from pyquda_utils import core, source

    point = source.propagator(latt_info, "point", list(x_src))
    source_plus = momentum_smear_kernel(
        point, base_gauge, latt_info, k_plus, radius, n_steps
    )
    source_minus = momentum_smear_kernel(
        point, base_gauge, latt_info, k_minus, radius, n_steps
    )
    with dirac.useGauge(base_gauge):
        spectator = core.invertPropagator(
            dirac, source_plus, mrhs=mrhs, restart=restart
        )
        active = core.invertPropagator(
            dirac, source_minus, mrhs=mrhs, restart=restart
        )

    if sink_mode == "s2s":
        spectator_at_sink = momentum_smear_kernel(
            spectator, base_gauge, latt_info, sink_plus, radius, n_steps
        )
        active_at_sink = momentum_smear_kernel(
            active, base_gauge, latt_info, sink_minus, radius, n_steps
        )
        active_sink_k = sink_minus
    else:
        spectator_at_sink = spectator
        active_at_sink = active
        active_sink_k = None
    return MomentumSmearResult(
        mode=sink_mode,
        k_mode=k_plus,
        spectator=spectator,
        active=active,
        spectator_at_sink=spectator_at_sink,
        active_at_sink=active_at_sink,
        active_sink_k=active_sink_k,
    )


def momentum_smear_baryon_degenerate(
    latt_info: Any,
    base_gauge: Any,
    dirac: Any,
    x_src: Sequence[int],
    k_mode: Sequence[float],
    radius: float,
    n_steps: int,
    sink_mode: str = "s2s",
    sink_k_mode: Optional[Sequence[float]] = None,
    mrhs: int = 1,
    restart: int = 0,
) -> BaryonMomentumSmearResult:
    """Prepare one degenerate-baryon line with an optional independent sink k.

    ``k_mode`` controls source smearing. For S-to-S, ``sink_k_mode`` controls
    sink smearing and defaults to ``k_mode`` for backward compatibility.
    S-to-P rejects a supplied ``sink_k_mode`` because no sink kernel is run.
    """

    if sink_mode not in ("s2p", "s2s"):
        raise ValueError("sink_mode must be 's2p' or 's2s'")
    if sink_mode == "s2p" and sink_k_mode is not None:
        raise ValueError("sink_k_mode is only valid for sink_mode='s2s'")
    _validate_solver_options(mrhs, restart)
    _validate_smearing_domain(radius, n_steps)
    if _lattice_signature(base_gauge.latt_info) != _lattice_signature(latt_info):
        raise ValueError("base_gauge and latt_info lattice decompositions must match")
    _spatial_extents_xyz(base_gauge.latt_info)
    x_src = _integer_coordinate(
        x_src, "x_src", _global_extents_xyzt(latt_info)
    )
    mode = _mode3(k_mode)
    supplied_sink_mode = (
        None if sink_k_mode is None else _mode3(sink_k_mode)
    )
    sink_k = mode if supplied_sink_mode is None else supplied_sink_mode
    from pyquda_utils import core, source

    point = source.propagator(latt_info, "point", list(x_src))
    smeared_source = momentum_smear_kernel(
        point, base_gauge, latt_info, mode, radius, n_steps
    )
    with dirac.useGauge(base_gauge):
        propagator = core.invertPropagator(
            dirac, smeared_source, mrhs=mrhs, restart=restart
        )
    if sink_mode == "s2s":
        propagator_at_sink = momentum_smear_kernel(
            propagator, base_gauge, latt_info, sink_k, radius, n_steps
        )
        active_sink_k = sink_k
    else:
        propagator_at_sink = propagator
        active_sink_k = None
    return BaryonMomentumSmearResult(
        mode=sink_mode,
        k_mode=mode,
        propagator=propagator,
        propagator_at_sink=propagator_at_sink,
        active_sink_k=active_sink_k,
    )


def momentum_smear_s_to_p(*args: Any, **kwargs: Any) -> MomentumSmearResult:
    if "sink_mode" in kwargs:
        raise ValueError("sink_mode is fixed by momentum_smear_s_to_p")
    return momentum_smear_meson(*args, **kwargs, sink_mode="s2p")


def momentum_smear_s_to_s(*args: Any, **kwargs: Any) -> MomentumSmearResult:
    if "sink_mode" in kwargs:
        raise ValueError("sink_mode is fixed by momentum_smear_s_to_s")
    return momentum_smear_meson(*args, **kwargs, sink_mode="s2s")


def momentum_smear_baryon_s_to_p(
    *args: Any, **kwargs: Any
) -> BaryonMomentumSmearResult:
    if "sink_mode" in kwargs:
        raise ValueError("sink_mode is fixed by momentum_smear_baryon_s_to_p")
    return momentum_smear_baryon_degenerate(*args, **kwargs, sink_mode="s2p")


def momentum_smear_baryon_s_to_s(
    *args: Any, **kwargs: Any
) -> BaryonMomentumSmearResult:
    if "sink_mode" in kwargs:
        raise ValueError("sink_mode is fixed by momentum_smear_baryon_s_to_s")
    return momentum_smear_baryon_degenerate(*args, **kwargs, sink_mode="s2s")


def fourier_phase_pair(
    latt_info: Any,
    momentum: Sequence[int],
    x_src: Sequence[int] = (0, 0, 0, 0),
) -> Tuple[Any, Any]:
    """Return the conjugate ``-Pf``/``+Pf`` phase pair.

    For a nonforward 3pt, pass the final hadron momentum ``Pf`` here. The
    current-insertion phase carrying ``Pf-Pi`` belongs to the contraction.
    The positive phase is physically the sequential-source choice only when
    the downstream contraction explicitly daggers/conjugates that line; this
    helper does not establish that downstream contract.
    """

    momentum = _integer_modes(momentum, "momentum", 3)
    x_src = _integer_coordinate(
        x_src, "x_src", _global_extents_xyzt(latt_info)
    )
    from pyquda_utils.phase_v2 import MomentumPhase

    sequential = MomentumPhase(latt_info).getPhase(momentum, list(x_src)).data
    return sequential.conj(), sequential


def build_sequential_source(
    result: MomentumSmearResult,
    latt_info: Any,
    tseq: int,
    sequential_source_phase: Any,
    gamma_sink_bar: Any,
    gamma_source_bar: Any,
    base_gauge: Any,
    radius: float,
    n_steps: int,
) -> Any:
    """Build the phase- and smearing-matched meson sequential source."""

    if not isinstance(result, MomentumSmearResult):
        raise TypeError("result must be a MomentumSmearResult")
    if result.mode not in {"s2p", "s2s"}:
        raise ValueError("result.mode must be 's2p' or 's2s'")
    _mode3(result.k_mode)
    _validate_smearing_domain(radius, n_steps)
    canonical_active_sink_k = (
        None
        if result.active_sink_k is None
        else _mode3(result.active_sink_k)
    )
    if result.mode == "s2p" and canonical_active_sink_k is not None:
        raise ValueError("s2p results must not carry an active sink-smearing mode")
    if result.mode == "s2s" and canonical_active_sink_k is None:
        raise ValueError("s2s results must carry an active sink-smearing mode")
    tseq = _global_time(tseq, "tseq", latt_info)
    lattice_signature = _lattice_signature(latt_info)
    if _lattice_signature(base_gauge.latt_info) != lattice_signature:
        raise ValueError("base_gauge and latt_info lattice decompositions must match")
    spectator = result.spectator_at_sink
    if hasattr(spectator, "latt_info") and _lattice_signature(spectator.latt_info) != lattice_signature:
        raise ValueError("spectator and sequential-source lattice decompositions must match")
    spectator_data = spectator.data
    if spectator_data.ndim != 9 or spectator_data.shape[0] != 2:
        raise ValueError(
            "spectator_at_sink.data must have (e,t,z,y,xh,4,4,3,3) layout"
        )
    if spectator_data.shape[-4:] != (4, 4, 3, 3):
        raise ValueError("spectator spin-color extents must be (4,4,3,3)")
    if getattr(sequential_source_phase, "shape", None) != spectator_data.shape[:5]:
        raise ValueError(
            "sequential_source_phase must match checkerboard (e,t,z,y,xh) axes"
        )
    if (
        getattr(gamma_sink_bar, "shape", None) != (4, 4)
        or getattr(gamma_source_bar, "shape", None) != (4, 4)
    ):
        raise ValueError("sequential-source gamma kernels must have shape (4,4)")
    xp = _xp(spectator_data)
    if any(
        _xp(array) is not xp
        for array in (sequential_source_phase, gamma_sink_bar, gamma_source_bar)
    ):
        raise TypeError("phase, propagator, and gamma kernels must use one backend")
    from opt_einsum import contract
    from pyquda_utils import core, source

    sink_block = core.LatticePropagator(latt_info)
    sink_block.data = contract(
        "AB,wtzyxBCab,CD->wtzyxADab",
        gamma_sink_bar,
        spectator_data,
        gamma_source_bar,
    )
    sink_block.data = contract(
        "wtzyx,wtzyxADab->wtzyxADab",
        sequential_source_phase,
        sink_block.data,
    )
    sequential = source.sequential12(sink_block, tseq)
    if canonical_active_sink_k is not None:
        sequential = momentum_smear_kernel(
            sequential,
            base_gauge,
            latt_info,
            canonical_active_sink_k,
            radius,
            n_steps,
        )
    return sequential


MesonMomentumSmearResult = MomentumSmearResult
momentum_smear_meson_s_to_p = momentum_smear_s_to_p
momentum_smear_meson_s_to_s = momentum_smear_s_to_s
build_meson_sequential_source = build_sequential_source


__all__ = [
    "MomentumSmearResult",
    "MesonMomentumSmearResult",
    "BaryonMomentumSmearResult",
    "SingleMomentumSmearResult",
    "negate_mode",
    "phase_spatial_links",
    "momentum_smear_kernel",
    "momentum_smear_meson",
    "momentum_smear_baryon_degenerate",
    "momentum_smear_s_to_p",
    "momentum_smear_s_to_s",
    "momentum_smear_meson_s_to_p",
    "momentum_smear_meson_s_to_s",
    "momentum_smear_baryon_s_to_p",
    "momentum_smear_baryon_s_to_s",
    "apply_momentum_smear_source",
    "apply_momentum_smear_sink",
    "apply_momentum_smear_seqprop",
    "momentum_smear_propagator",
    "fourier_phase_pair",
    "build_sequential_source",
    "build_meson_sequential_source",
]
