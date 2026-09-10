"""Bare pure-gauge observable kernels for PyQUDA/CuPy."""

import numpy as np
import cupy as cp
from itertools import permutations
from mpi4py import MPI
from opt_einsum import contract
from typing import Any, List, Mapping
from numbers import Number
import re

from pyquda_utils import core
from pyquda_utils.core import LatticeInfo
from pyquda.field import LatticeGauge, Nc


def _e2e_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _e2e_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _e2e_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be Boolean")
    return bool(value)


def _e2e_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive integer") from error
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _e2e_mode(value: Any, name: str) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 4:
        raise ValueError(f"{name} must contain four integer modes in (x,y,z,t) order")
    components = []
    for index, component in enumerate(value):
        if isinstance(component, (bool, np.bool_)):
            raise TypeError(f"{name}[{index}] must be an integer mode")
        try:
            numeric = float(component)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name}[{index}] must be an integer mode") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name}[{index}] must be a finite integer mode")
        components.append(int(numeric))
    return tuple(components)


def _e2e_array(value: Any, name: str, *, complex_required: bool = False) -> tuple[int, ...]:
    """Check array residency/type without accepting paths, summaries, or lists."""

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        raise TypeError(f"{name} must be an actual numerical array, not a path or summary")
    try:
        normalized_shape = tuple(int(item) for item in shape)
        normalized_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose a numerical array shape and dtype") from error
    if not normalized_shape or any(item <= 0 for item in normalized_shape):
        raise ValueError(f"{name} must be non-empty")
    if not np.issubdtype(normalized_dtype, np.number):
        raise TypeError(f"{name} must use a numerical dtype")
    if complex_required and not np.issubdtype(normalized_dtype, np.complexfloating):
        raise TypeError(f"{name} must retain complex components")
    return normalized_shape


def _e2e_complex_tolerance(value: Any, name: str) -> tuple[float, float]:
    tolerance = _e2e_mapping(value, name)
    if tolerance.get("comparison") != "real_and_imaginary":
        raise ValueError(
            f"{name}.comparison must be 'real_and_imaginary'; magnitude-only checks are forbidden"
        )
    atol = _finite_real_scalar(tolerance.get("atol"), f"{name}.atol")
    rtol = _finite_real_scalar(tolerance.get("rtol"), f"{name}.rtol")
    if atol < 0 or rtol < 0:
        raise ValueError(f"{name}.atol and {name}.rtol must be non-negative")
    return atol, rtol


def _e2e_componentwise_close(observed: Any, expected: Any, tolerance: Any, name: str) -> None:
    observed_shape = _e2e_array(observed, f"{name}.observed", complex_required=True)
    expected_shape = _e2e_array(expected, f"{name}.expected", complex_required=True)
    if observed_shape != expected_shape:
        raise ValueError(f"{name}.observed and {name}.expected must have one exact shape")
    atol, rtol = _e2e_complex_tolerance(tolerance, f"{name}.tolerance")
    observed_backend = type(observed).__module__.split(".")[0]
    expected_backend = type(expected).__module__.split(".")[0]
    if observed_backend != expected_backend:
        raise TypeError(f"{name} must not compare host and device arrays implicitly")
    xp = cp if observed_backend == "cupy" else np
    real_ok = xp.allclose(observed.real, expected.real, rtol=rtol, atol=atol)
    imag_ok = xp.allclose(observed.imag, expected.imag, rtol=rtol, atol=atol)
    if xp is np:
        close = bool(real_ok) and bool(imag_ok)
    else:
        close = bool(np.asarray(cp.asnumpy(real_ok)).item()) and bool(
            np.asarray(cp.asnumpy(imag_ok)).item()
        )
    if not close:
        raise ValueError(f"{name} fails the frozen component-wise complex tolerance")


def _e2e_gauge_fixing(value: Any, name: str) -> None:
    gauge_fixing = _e2e_mapping(value, name)
    for key in ("functional", "algorithm"):
        _e2e_text(gauge_fixing.get(key), f"{name}.{key}")
    residual = _finite_real_scalar(gauge_fixing.get("residual"), f"{name}.residual")
    if residual < 0:
        raise ValueError(f"{name}.residual must be non-negative")


def validate_gluon_e2e_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a gluon E2E evidence record without invoking PyQUDA or a scheduler.

    This readiness gate only inspects already-produced arrays and metadata.  It
    cannot load a configuration, call a clover kernel, or submit a GPU/MPI job;
    absent provenance and controls are rejected rather than filled in.
    """

    record = _e2e_mapping(record, "record")
    gauge_input = _e2e_mapping(record.get("interacting_gauge_input"), "interacting_gauge_input")
    if not _e2e_bool(gauge_input.get("interacting"), "interacting_gauge_input.interacting"):
        raise ValueError("interacting_gauge_input.interacting must be true")
    checksum = _e2e_text(gauge_input.get("checksum"), "interacting_gauge_input.checksum")
    if re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", checksum) is None:
        raise ValueError("interacting_gauge_input.checksum must be a SHA-256 digest")
    gauge_shape = _e2e_array(gauge_input.get("array"), "interacting_gauge_input.array", complex_required=True)
    if len(gauge_shape) < 8 or gauge_shape[:2] != (4, 2) or gauge_shape[-2:] != (3, 3):
        raise ValueError("interacting_gauge_input.array must use (direction,e,t,z,y,xh,...,3,3) storage")

    geometry = _e2e_mapping(record.get("geometry"), "geometry")
    if geometry.get("checkerboard_axes") != "e,t,z,y,xh":
        raise ValueError("geometry.checkerboard_axes must be 'e,t,z,y,xh'")
    local = tuple(_e2e_positive_integer(value, f"geometry.local_extents_xyzt[{index}]") for index, value in enumerate(geometry.get("local_extents_xyzt", ())))
    global_extents = tuple(_e2e_positive_integer(value, f"geometry.global_extents_xyzt[{index}]") for index, value in enumerate(geometry.get("global_extents_xyzt", ())))
    grid = tuple(_e2e_positive_integer(value, f"geometry.process_grid_xyzt[{index}]") for index, value in enumerate(geometry.get("process_grid_xyzt", ())))
    coordinate = _e2e_mode(geometry.get("process_coordinate_xyzt"), "geometry.process_coordinate_xyzt")
    if not (len(local) == len(global_extents) == len(grid) == 4):
        raise ValueError("geometry extents and process grid must each have four components")
    if local[0] % 2:
        raise ValueError("geometry.local_extents_xyzt[0] must be even for checkerboard storage")
    if tuple(local[index] * grid[index] for index in range(4)) != global_extents:
        raise ValueError("geometry global extents must equal local extents times process grid")
    if any(component < 0 or component >= grid[index] for index, component in enumerate(coordinate)):
        raise ValueError("geometry.process_coordinate_xyzt lies outside geometry.process_grid_xyzt")
    if gauge_shape[2:6] != (local[3], local[2], local[1], local[0] // 2):
        raise ValueError("gauge array checkerboard extents disagree with geometry.local_extents_xyzt")

    momentum = _e2e_mapping(record.get("momentum_consistency"), "momentum_consistency")
    _e2e_mode(momentum.get("mode_xyzt"), "momentum_consistency.mode_xyzt")
    _e2e_componentwise_close(
        momentum.get("single_mode"),
        momentum.get("all_modes_selected"),
        momentum.get("tolerance"),
        "momentum_consistency",
    )

    reference = _e2e_mapping(record.get("clover_or_reference"), "clover_or_reference")
    if reference.get("kind") not in {"independent_clover", "supplied_reference"}:
        raise ValueError("clover_or_reference.kind must be independent_clover or supplied_reference")
    if not _e2e_bool(reference.get("independent"), "clover_or_reference.independent"):
        raise ValueError("clover_or_reference.independent must be true")
    _e2e_text(reference.get("source"), "clover_or_reference.source")
    reference_checksum = _e2e_text(reference.get("checksum"), "clover_or_reference.checksum")
    if re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", reference_checksum) is None:
        raise ValueError("clover_or_reference.checksum must be a SHA-256 digest")
    _e2e_array(reference.get("values"), "clover_or_reference.values")

    controls = _e2e_mapping(record.get("gauge_transformation_controls"), "gauge_transformation_controls")
    if set(controls) != {"invariant", "dependent"}:
        raise ValueError("gauge_transformation_controls must contain invariant and dependent records")
    invariant = _e2e_mapping(controls["invariant"], "gauge_transformation_controls.invariant")
    if invariant.get("quantity_class") != "gauge_invariant":
        raise ValueError("invariant control must be labelled gauge_invariant")
    _e2e_componentwise_close(
        invariant.get("baseline"), invariant.get("transformed"), invariant.get("tolerance"),
        "gauge_transformation_controls.invariant",
    )
    dependent = _e2e_mapping(controls["dependent"], "gauge_transformation_controls.dependent")
    if dependent.get("quantity_class") != "gauge_dependent":
        raise ValueError("dependent control must be labelled gauge_dependent")
    _e2e_gauge_fixing(dependent.get("baseline_gauge_fixing"), "gauge_transformation_controls.dependent.baseline_gauge_fixing")
    _e2e_gauge_fixing(dependent.get("transformed_gauge_fixing"), "gauge_transformation_controls.dependent.transformed_gauge_fixing")
    _e2e_componentwise_close(
        dependent.get("baseline_refixed"), dependent.get("transformed_refixed"), dependent.get("tolerance"),
        "gauge_transformation_controls.dependent",
    )

    states = _e2e_mapping(record.get("observable_states"), "observable_states")
    if set(states) != {"bare", "gauge_fixed", "renormalized"}:
        raise ValueError("observable_states must contain exactly bare, gauge_fixed, and renormalized")
    bare = _e2e_mapping(states["bare"], "observable_states.bare")
    gauge_fixed = _e2e_mapping(states["gauge_fixed"], "observable_states.gauge_fixed")
    renormalized = _e2e_mapping(states["renormalized"], "observable_states.renormalized")
    if bare.get("state") != "bare" or gauge_fixed.get("state") != "gauge_fixed":
        raise ValueError("bare and gauge_fixed records must keep their literal state labels")
    bare_values = bare.get("values")
    gauge_fixed_values = gauge_fixed.get("values")
    _e2e_array(bare_values, "observable_states.bare.values")
    _e2e_array(gauge_fixed_values, "observable_states.gauge_fixed.values")
    if bare is gauge_fixed or bare_values is gauge_fixed_values:
        raise ValueError("bare and gauge-fixed outputs must be stored as distinct objects")
    _e2e_gauge_fixing(gauge_fixed.get("gauge_fixing"), "observable_states.gauge_fixed.gauge_fixing")
    if renormalized.get("state") == "not_available":
        _e2e_text(renormalized.get("reason"), "observable_states.renormalized.reason")
    elif renormalized.get("state") == "renormalized":
        renormalized_values = renormalized.get("values")
        _e2e_array(renormalized_values, "observable_states.renormalized.values")
        if renormalized_values is bare_values or renormalized_values is gauge_fixed_values:
            raise ValueError("renormalized output must not alias bare or gauge-fixed values")
        _e2e_text(renormalized.get("scheme_identity"), "observable_states.renormalized.scheme_identity")
    else:
        raise ValueError("observable_states.renormalized.state must be not_available or renormalized")

    return {
        "status": "validated_e2e_record_no_submission",
        "checked_controls": (
            "checkerboard_global_coordinates",
            "one_vs_all_momentum",
            "independent_clover_or_reference",
            "gauge_invariant_and_dependent_controls",
            "separate_observable_states",
        ),
    }


def _finite_real_scalar(value, name, *, positive=False):
    """Validate a host scalar without Boolean or complex coercion."""

    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _choice_integer(value, name, allowed):
    """Validate a selector/index without accepting Boolean aliases."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a non-boolean integer")
    result = int(value)
    if result not in allowed:
        choices = ", ".join(str(item) for item in sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}")
    return result


def _lorentz_indices(*indices):
    return tuple(
        _choice_integer(index, "Lorentz index", {0, 1, 2, 3})
        for index in indices
    )


def _wilson_line_controls(link_dir, link_length, forward_flag):
    """Validate all link controls before any field-strength construction."""

    link_dir = _choice_integer(link_dir, "link_dir", {0, 1, 2, 3})
    if isinstance(link_length, (bool, np.bool_)) or not isinstance(
        link_length, (int, np.integer)
    ):
        raise TypeError("link_length must be a non-boolean integer")
    link_length = int(link_length)
    if link_length < 0:
        raise ValueError("link_length must be nonnegative")
    forward_flag = _choice_integer(forward_flag, "forward_flag", {0, 1})
    return link_dir, link_length, forward_flag


def _require_time_undecomposed(latt_info):
    grid_size = tuple(getattr(latt_info, "grid_size", (1, 1, 1, 1)))
    if len(grid_size) != 4:
        raise ValueError("latt_info.grid_size must contain (Gx,Gy,Gz,Gt)")
    canonical = []
    for value in grid_size:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("grid_size entries must be non-boolean integers")
        if int(value) < 1:
            raise ValueError("grid_size entries must be positive")
        canonical.append(int(value))
    if canonical[3] != 1:
        raise NotImplementedError(
            "time-resolved Wilson-line output currently requires Gt=1; a world "
            "reduction would mix different global time slices"
        )


def _lattice_vector(values, name, *, positive):
    """Return one strict four-component integer lattice vector."""

    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a four-component integer sequence") from error
    if len(values) != 4:
        raise ValueError(f"{name} must contain (x,y,z,t)")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} entries must be non-boolean integers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} entries must be finite integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} entries must be finite integers")
        integer = int(numeric)
        if positive and integer < 1:
            raise ValueError(f"{name} entries must be positive")
        if not positive and integer < 0:
            raise ValueError(f"{name} entries must be nonnegative")
        result.append(integer)
    return tuple(result)


def _lattice_signature(latt_info):
    """Canonicalize global/local extents, process grid, and rank coordinate."""

    if latt_info is None:
        raise ValueError("latt_info is required")
    if hasattr(latt_info, "size"):
        local_size = _lattice_vector(latt_info.size, "size", positive=True)
    elif all(hasattr(latt_info, name) for name in ("Lx", "Ly", "Lz", "Lt")):
        local_size = _lattice_vector(
            (latt_info.Lx, latt_info.Ly, latt_info.Lz, latt_info.Lt),
            "size",
            positive=True,
        )
    else:
        raise ValueError("latt_info must expose local size=(Lx,Ly,Lz,Lt)")

    grid_size = _lattice_vector(
        getattr(latt_info, "grid_size", (1, 1, 1, 1)),
        "grid_size",
        positive=True,
    )
    derived_global = tuple(
        local * grid for local, grid in zip(local_size, grid_size)
    )
    if hasattr(latt_info, "global_size"):
        global_size = _lattice_vector(
            latt_info.global_size, "global_size", positive=True
        )
    elif all(hasattr(latt_info, name) for name in ("GLx", "GLy", "GLz", "GLt")):
        global_size = _lattice_vector(
            (latt_info.GLx, latt_info.GLy, latt_info.GLz, latt_info.GLt),
            "global_size",
            positive=True,
        )
    else:
        global_size = derived_global
    if global_size != derived_global:
        raise ValueError("global_size must equal size * grid_size")

    if hasattr(latt_info, "grid_coord"):
        grid_coord = _lattice_vector(
            latt_info.grid_coord, "grid_coord", positive=False
        )
    elif int(np.prod(grid_size)) == 1:
        grid_coord = (0, 0, 0, 0)
    else:
        raise ValueError("multi-rank lattice metadata must expose grid_coord")
    if any(coord >= extent for coord, extent in zip(grid_coord, grid_size)):
        raise ValueError("grid_coord lies outside grid_size")
    return global_size, local_size, grid_size, grid_coord


def _require_matching_lattice_info(latt_info, gauge):
    if not hasattr(gauge, "latt_info"):
        raise ValueError("gauge must expose latt_info")
    if _lattice_signature(latt_info) != _lattice_signature(gauge.latt_info):
        raise ValueError("gauge and latt_info lattice decompositions must match")


######################################### SU(3) Structural Constants #########################################

def epsilon(n: int):
    shape = (n,) * n
    epsilon = cp.zeros(shape)
    for idx in permutations(range(n)):
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if idx[i] > idx[j]:
                    inversions += 1
        sign = (-1) ** inversions
        if len(set(idx)) == n:
            epsilon[idx] = sign
    return epsilon


def apply_antisymmetry(tensor):
    indices = cp.arange(8)
    for i in indices:
        for j in indices:
            for k in indices:
                if i < j and j < k:
                    current_value = tensor[i, j, k]
                    tensor[j, i, k] = -current_value
                    tensor[i, k, j] = -current_value
                    tensor[k, j, i] = -current_value
                    tensor[j, k, i] = current_value
                    tensor[k, i, j] = current_value
                    tensor[i, j, k] = current_value
    return tensor


def SU3f():
    su3f = cp.zeros((8, 8, 8))
    su3f[0, 1, 2] = 1.0
    su3f[0, 3, 6] = su3f[1, 3, 5] = su3f[1, 4, 6] = su3f[2, 3, 4] = 1 / 2
    su3f[0, 4, 5] = su3f[2, 5, 6] = -1 / 2
    su3f[3, 4, 7] = su3f[5, 6, 7] = cp.sqrt(3) / 2
    su3f = apply_antisymmetry(su3f)
    return su3f


def SU3_phase_angle(gauge: LatticeGauge):
    phase_angles = cp.angle(gauge.data)
    return phase_angles



######################################### Plaquette, Gauge Potential, Field Strength, et al. #########################################

def g_0(beta, u_0=None, normalization="wilson"):
    """Return a bare coupling for an explicitly named gauge normalization.

    ``normalization='wilson'`` means beta=6/g0^2.  The legacy tadpole formula
    beta=10/(g0^2 u0^4) is available only under the explicit name
    ``'legacy_tadpole_10'``. Fermion-action labels such as DWF or clover do not
    determine the gauge-action normalization and are intentionally rejected.
    """

    beta = _finite_real_scalar(beta, "beta", positive=True)
    if not isinstance(normalization, str):
        raise TypeError("normalization must be a string")
    if normalization == "wilson":
        if u_0 is not None:
            raise ValueError(
                "u_0 is not part of the Wilson beta=6/g0^2 convention; "
                "omit u_0 or explicitly select legacy_tadpole_10"
            )
        return cp.sqrt(6 / beta)
    if normalization == "legacy_tadpole_10":
        if u_0 is None:
            raise ValueError(
                "legacy_tadpole_10 requires an explicit finite positive u_0"
            )
        u_0 = _finite_real_scalar(u_0, "u_0", positive=True)
        return cp.sqrt(10 / (beta * (u_0 ** 4)))
    raise ValueError("normalization must be 'wilson' or 'legacy_tadpole_10'")


def trace_less(array_a, array_b):
    #etzyxab type traceless
    if array_a.shape[:-2] != array_b.shape or array_a.shape[-1] != array_a.shape[-2]:
        raise ValueError("array_b must match array_a without its square matrix axes")
    identity = cp.eye(array_a.shape[-1], dtype=array_a.dtype)
    return array_a - array_b[..., None, None] * identity


def _validated_gauge_data(gauge):
    """Validate the public gauge-field storage before allocating a plaquette.

    The accepted PyQUDA layout is ``(direction,e,t,z,y,xh,color,color)``.
    Keeping this check ahead of the first CuPy allocation turns malformed
    wrapper objects into a deterministic input error rather than an indexing
    failure inside the plaquette loop.
    """

    data = getattr(gauge, "data", None)
    shape = getattr(data, "shape", None)
    if shape is None or len(shape) != 8:
        raise ValueError(
            "gauge.data must have (direction,e,t,z,y,xh,color,color) layout"
        )
    if shape[0] != 4 or shape[1] != 2 or tuple(shape[-2:]) != (3, 3):
        raise ValueError(
            "gauge.data must have (4,2,Lt,Lz,Ly,Lx/2,3,3) extents"
        )
    if any(int(extent) < 1 for extent in shape[2:6]):
        raise ValueError("gauge.data lattice extents must be positive")
    return data


def plaq_munu(gauge: LatticeGauge):
    #plaquette_munu(x) = U_mu(x)U_nu(x + mu)U^dag_mu(x + nu)U^dag_nu(x), 44etzyxab (d = xyzt)
    #mu = nu, diagonal elements are SU(3) unitray element
    gauge_data = _validated_gauge_data(gauge)
    other_dim = gauge_data.shape[1:]
    plaq = cp.zeros((4, 4, *other_dim), dtype=gauge_data.dtype)
    for mu in range(4):
        for ic in range(3):
            plaq[mu, mu, :, :, :, :, :, ic, ic] = 1.0 # P_{\mu\mu}
        for nu in range(mu + 1, 4):
            U_mu = gauge_data[mu]  # U_mu(x)
            U_nu = gauge_data[nu]  # U_nu(x)
            # pyquda.field shifts one component by (step, direction). Calling
            # LatticeGauge.shift with one displacement vector belongs to a
            # different gauge helper and fails for this imported class.
            U_mu_shift_nu = gauge[mu].shift(1, nu).data  # U_mu(x + nu)
            U_nu_shift_mu = gauge[nu].shift(1, mu).data  # U_nu(x + mu)
            U_mu_dagger_shift_nu = U_mu_shift_nu.conj().transpose(0, 1, 2, 3, 4, 6, 5)
            U_nu_dagger = U_nu.conj().transpose(0, 1, 2, 3, 4, 6, 5)
            plaq[mu, nu] = contract(
                "...ab,...bc,...cd,...de->...ae",
                U_mu, U_nu_shift_mu, U_mu_dagger_shift_nu, U_nu_dagger) # P_{\mu\nu}
            plaq[nu, mu] = plaq[mu, nu].conj().transpose(0, 1, 2, 3, 4, 6, 5) # P_{\nu\mu}
    return plaq


def A_mu(gauge: LatticeGauge, g_0, half_flag = 1):
    #A_mu with integer(half_flag = 0) and half(half_flag = 1) - lattice definition, 4etzyxab (d = xyzt)
    half_flag = _choice_integer(half_flag, "half_flag", {0, 1})
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    if half_flag == 0 :
        gauge_shift = gauge.shift([-1, -1, -1, -1], [0, 1, 2, 3]) #U_mu(x - mu)
        A_mu = (gauge.data - gauge.data.conj().transpose(0, 1, 2, 3, 4, 5, 7, 6) + \
        gauge_shift.data - gauge_shift.data.conj().transpose(0, 1, 2, 3, 4, 5, 7, 6)) / (4j)
    elif half_flag == 1 :
        A_mu = (gauge.data - gauge.data.conj().transpose(0, 1, 2, 3, 4, 5, 7, 6)) / (2j)
    A_mu_trace = contract("detzyxaa->detzyx", A_mu) / 3
    A_mu_traceless = trace_less(A_mu, A_mu_trace)
    A_mu_all = A_mu_traceless / g_0
    return A_mu_all


def F_munu(gauge: LatticeGauge, g_0, save_flag = 1):
    """Return clover field strengths in the ``0,1,2,3=x,y,z,t`` convention.

    ``save_flag=1`` returns ``(F_ti, F_ij)`` with pair order
    ``F_ij=(F_xy,F_xz,F_yz)=(B_z,-B_y,B_x)``. ``save_flag=0`` returns the
    full antisymmetric tensor; diagonal entries are zero by construction.
    """

    save_flag = _choice_integer(save_flag, "save_flag", {0, 1})
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    gauge_F4i = gauge.loop(
        [[[0, 3, 4, 7], [3, 4, 7, 0], [4, 7, 0, 3], [7, 0, 3, 4],],
         [[1, 3, 5, 7], [3, 5, 7, 1], [5, 7, 1, 3], [7, 1, 3, 5],],
         [[2, 3, 6, 7], [3, 6, 7, 2], [6, 7, 2, 3], [7, 2, 3, 6],],
         [[3, 7, 3, 7], [3, 7, 3, 7], [3, 7, 3, 7], [3, 7, 3, 7],],],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],)
    E4i_all = -(-1) * (gauge_F4i.data[:3] - gauge_F4i.data[:3].conj().transpose(0, 1, 2, 3, 4, 5, 7, 6)) / (2j * g_0) #E_{x, y, z}
    gauge_Fij = gauge.loop(
        [[[0, 1, 4, 5], [1, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 4],],
         [[0, 2, 4, 6], [2, 4, 6, 0], [4, 6, 0, 2], [6, 0, 2, 4],],
         [[1, 2, 5, 6], [2, 5, 6, 1], [5, 6, 1, 2], [6, 1, 2, 5],],
         [[3, 7, 3, 7], [3, 7, 3, 7], [3, 7, 3, 7], [3, 7, 3, 7],],],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],)
    # Pair order is (F_xy,F_xz,F_yz)=(B_z,-B_y,B_x), not (B_x,B_y,B_z).
    Bij_all = -(gauge_Fij.data[:3] - gauge_Fij.data[:3].conj().transpose(0, 1, 2, 3, 4, 5, 7, 6)) / (2j * g_0)
    if save_flag == 1 :
        return E4i_all, Bij_all
    else :
        other_dim = E4i_all.shape[1:]
        F_munu_all = cp.zeros((4, 4, *other_dim), dtype = E4i_all.dtype)
        for i in range(3):
           F_munu_all[3, i] =  E4i_all[i]
           F_munu_all[i, 3] = -E4i_all[i]
        F_munu_all[0, 1] =  Bij_all[0]
        F_munu_all[1, 0] = -Bij_all[0]
        F_munu_all[0, 2] =  Bij_all[1]
        F_munu_all[2, 0] = -Bij_all[1]
        F_munu_all[1, 2] =  Bij_all[2]
        F_munu_all[2, 1] = -Bij_all[2]
        return F_munu_all


def F_and_tildeF(gauge: LatticeGauge, g_0, mu1, nu1, mu2, nu2):
    # Conventional dual tensor: tildeF_{op} = (1/2) epsilon_{opmn} F_{mn}.
    # The 1/2 prevents double counting the antisymmetric (m,n) and (n,m) pair.
    mu1, nu1, mu2, nu2 = _lorentz_indices(mu1, nu1, mu2, nu2)
    Fmunu = F_munu(gauge, g_0, 0)
    tildeFmunu = 0.5 * contract("opmn,mnetzyxab->opetzyxab", epsilon(4), Fmunu)
    return Fmunu[mu1, nu1], tildeFmunu[mu2, nu2]


def _dagger(field):
    return field.conj().swapaxes(-1, -2)


def _checked_real(field, observable_name):
    """Return the real part only after an explicit imaginary-residual check."""

    dtype = np.dtype(field.dtype)
    if dtype == np.dtype(np.complex64):
        rtol, atol = 1.0e-4, 1.0e-6
    else:
        rtol, atol = 1.0e-10, 1.0e-12
    imag_max = float(cp.asnumpy(cp.max(cp.abs(field.imag)))) if field.size else 0.0
    real_max = float(cp.asnumpy(cp.max(cp.abs(field.real)))) if field.size else 0.0
    if imag_max > atol + rtol * max(1.0, real_max):
        raise ValueError(
            f"{observable_name} has a non-negligible imaginary residual "
            f"({imag_max:.6e}); check Hermiticity and contraction conventions"
        )
    return field.real


def _wilson_line_operator_gpu(
    latt_info: LatticeInfo,
    gauge: LatticeGauge,
    left_field,
    right_field,
    link_dir: int,
    link_length: int,
    forward_flag: int,
):
    """Contract F(x) W F(x+n) W^dag on-device and reduce spatial sites.

    `LatticeLink.shift` performs the required neighbor/halo movement while the
    full lattice matrices remain in the configured PyQUDA array backend. On a
    multi-rank boundary PyQUDA currently stages only the halo through host MPI;
    avoiding that small transfer requires CUDA-aware support in pyquda_comm.
    """

    link_dir, link_length, forward_flag = _wilson_line_controls(
        link_dir, link_length, forward_flag
    )
    _require_time_undecomposed(latt_info)

    identity = cp.eye(Nc, dtype=gauge.data.dtype)
    wilson = cp.broadcast_to(identity, gauge.data.shape[1:]).copy()
    link = gauge[link_dir]
    direction = 1 if forward_flag else -1
    for step in range(link_length):
        offset = step if direction == 1 else -(step + 1)
        shifted_link = link.shift(offset, link_dir).data
        if direction == -1:
            shifted_link = _dagger(shifted_link)
        wilson = wilson @ shifted_link

    right_shifted = link.__class__(latt_info, right_field).shift(
        direction * link_length, link_dir
    ).data
    density = -0.5 * cp.trace(
        left_field @ wilson @ right_shifted @ _dagger(wilson), axis1=-2, axis2=-1
    )
    local_t = density.sum(axis=(0, 2, 3, 4))

    # MPI has no portable CuPy reduction in this project. Transfer only Lt
    # complex values after the O(V) contraction, never the gauge/F tensors.
    return SumMPI(cp.asnumpy(local_t))


def FW_tildeFW(latt_info: LatticeInfo, gauge: LatticeGauge, g_0, mu1, nu1, mu2, nu2, link_dir = 2, link_length = 0):
    #calculate (-1/2)Tr[F_{\mu1\nu1}(0)W(0,?)\tilde{F}_{\mu2\nu2}(?)W(?,0)]
    #link_dir = 0123, W(0,xyzt), link_length = int
    link_dir, link_length, _ = _wilson_line_controls(
        link_dir, link_length, 1
    )
    mu1, nu1, mu2, nu2 = _lorentz_indices(mu1, nu1, mu2, nu2)
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    _require_matching_lattice_info(latt_info, gauge)
    _require_time_undecomposed(latt_info)
    F, tildeF = F_and_tildeF(gauge, g_0, mu1, nu1, mu2, nu2)
    return _wilson_line_operator_gpu(
        latt_info, gauge, F, tildeF, link_dir, link_length, 1
    )


def FW_FW(latt_info: LatticeInfo, gauge: LatticeGauge, g_0, mu1, nu1, mu2, nu2, link_dir = 2, link_length = 0, forward_flag = 1):
    #calculate (-1/2)Tr[F_{\mu1\nu1}(0)W(0,?)F_{\mu2\nu2}(?)W(?,0)]
    #link_dir = 0123, W(0,xyzt), link_length = int
    link_dir, link_length, forward_flag = _wilson_line_controls(
        link_dir, link_length, forward_flag
    )
    mu1, nu1, mu2, nu2 = _lorentz_indices(mu1, nu1, mu2, nu2)
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    _require_matching_lattice_info(latt_info, gauge)
    _require_time_undecomposed(latt_info)
    F = F_munu(gauge, g_0, 0)
    Fmunu, Frhosigma = F[mu1, nu1], F[mu2, nu2]
    return _wilson_line_operator_gpu(
        latt_info, gauge, Fmunu, Frhosigma, link_dir, link_length, forward_flag
    )


def FF(latt_info: LatticeInfo, gauge: LatticeGauge, g_0, mu1, nu1, mu2, nu2):
    #calculate F_{\mu1\nu1}F_{\mu2\nu2}
    mu1, nu1, mu2, nu2 = _lorentz_indices(mu1, nu1, mu2, nu2)
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    _require_matching_lattice_info(latt_info, gauge)
    _require_time_undecomposed(latt_info)
    F = F_munu(gauge, g_0, 0)
    FF = contract("etzyxab,etzyxba->t", F[mu1, nu1], F[mu2, nu2])
    return SumMPI(cp.asnumpy(FF))



######################################### Fourier Transform #########################################

def SumMPI(data_local: np.ndarray):
    # Let mpi4py infer the NumPy datatype; the previous hard-coded complex128
    # datatype corrupted or rejected real and single-precision inputs. Resolve
    # PyQUDA's communicator at call time instead of capturing COMM_WORLD before
    # core.init, since PyQUDA may run on a custom communicator.
    return core.getMPIComm().reduce(np.asarray(data_local), op=MPI.SUM, root=0)


def _checkerboard_to_lexico_gpu(field):
    """Convert one `etzyx...` field to `tzyx...` without leaving CuPy."""

    if field.ndim < 5 or field.shape[0] != 2:
        raise ValueError("field must begin with checkerboard axes (e,t,z,y,xh)")
    _, Lt, Lz, Ly, Lxh = field.shape[:5]
    t, z, y, x = cp.indices((Lt, Lz, Ly, 2 * Lxh))
    parity = (t + z + y + x) & 1
    return field[parity, t, z, y, x // 2]


def _assemble_site_fields(fields):
    """Return one-rank device fields or explicit root-owned MPI host gathers.

    A multi-rank root receives a tuple of global NumPy arrays. Every non-root
    rank receives a same-length tuple of ``None`` values from ``gatherLattice``.
    """

    local = tuple(_checkerboard_to_lexico_gpu(field) for field in fields)
    if core.getMPISize() == 1:
        return local
    # PyQUDA currently exposes only a NumPy global gather. This O(V) transfer
    # is unavoidable when the caller explicitly requests a global site field.
    return tuple(
        core.gatherLattice(cp.asnumpy(field), [0, 1, 2, 3]) for field in local
    )


def _canonical_fft_modes(p: List[Number], latt_info: LatticeInfo):
    if len(p) != 4:
        raise ValueError("p must contain four integer modes (px,py,pz,pt)")
    modes = []
    for component in p:
        if isinstance(component, (bool, np.bool_)):
            raise TypeError("momentum modes must be integers, not booleans")
        try:
            numeric = float(component)
        except (TypeError, ValueError) as error:
            raise TypeError("momentum modes must be finite integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("momentum modes must be finite integers")
        modes.append(int(numeric))
    Lx, Ly, Lz, Lt = latt_info.size
    Gx, Gy, Gz, Gt = latt_info.grid_size
    extents = (Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt)
    if any(extent <= 0 for extent in extents):
        raise ValueError("global lattice extents must be positive")
    return tuple(
        (mode + extent // 2) % extent - extent // 2
        for mode, extent in zip(modes, extents)
    )


def _validate_checkerboard_geometry(latt_info: LatticeInfo):
    """Return local ``(Lx,Ly,Lz,Lt)`` after validating checkerboard storage."""

    if len(latt_info.size) != 4:
        raise ValueError("latt_info.size must contain (Lx,Ly,Lz,Lt)")
    extents = []
    for value in latt_info.size:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("local lattice extents must be non-boolean integers")
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
            raise ValueError("local lattice extents must be positive integers")
        extents.append(int(numeric))
    if extents[0] % 2:
        raise ValueError("checkerboard x storage requires even local Lx")
    return tuple(extents)


def FT_Phase(p: List[Number], latt_info: LatticeInfo):
    px, py, pz, pt = _canonical_fft_modes(p, latt_info)
    Lx, Ly, Lz, Lt = _validate_checkerboard_geometry(latt_info)
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    x = ((cp.arange(gx * Lx, (gx + 1) * Lx) + (Lx * Gx) // 2) % (Lx * Gx) - (Lx * Gx) // 2)
    y = ((cp.arange(gy * Ly, (gy + 1) * Ly) + (Ly * Gy) // 2) % (Ly * Gy) - (Ly * Gy) // 2)
    z = ((cp.arange(gz * Lz, (gz + 1) * Lz) + (Lz * Gz) // 2) % (Lz * Gz) - (Lz * Gz) // 2)
    t = ((cp.arange(gt * Lt, (gt + 1) * Lt) + (Lt * Gt) // 2) % (Lt * Gt) - (Lt * Gt) // 2)
    tt, zz, yy, xx = cp.meshgrid(t, z, y, x, indexing="ij")
    phase_lex = cp.exp(
        -2j * cp.pi * (
            (px / (Lx * Gx)) * xx
            + (py / (Ly * Gy)) * yy
            + (pz / (Lz * Gz)) * zz
            + (pt / (Lt * Gt)) * tt
        )
    )
    checkerboard = cp.empty((2, Lt, Lz, Ly, Lx // 2), dtype=phase_lex.dtype)
    parity = (tt + zz + yy) % 2
    checkerboard[0] = cp.where(parity[..., 0::2] == 0, phase_lex[..., 0::2], phase_lex[..., 1::2])
    checkerboard[1] = cp.where(parity[..., 0::2] == 0, phase_lex[..., 1::2], phase_lex[..., 0::2])
    return checkerboard


def FT_Gauge_1mom(
    gauge_data: cp.ndarray,
    p: List[Number],
    latt_info: LatticeInfo,
    half_flag=1,
):
    """Transform one four-direction gauge field at a single momentum.

    `half_flag=1` means component mu is link-centered at x+mu/2 and receives
    the direction-specific exp(-i p_mu/2) origin correction. Use
    `half_flag=0` only for an explicitly site-centered input field.
    """

    half_flag = _choice_integer(half_flag, "half_flag", {0, 1})
    Lx, Ly, Lz, Lt = _validate_checkerboard_geometry(latt_info)
    expected = (4, 2, Lt, Lz, Ly, Lx // 2)
    if gauge_data.ndim != 8 or gauge_data.shape[:6] != expected:
        raise ValueError(
            "gauge_data must have exact (4,2,Lt,Lz,Ly,Lx/2,Nc,Nc) layout"
        )
    if gauge_data.shape[-2] != gauge_data.shape[-1]:
        raise ValueError("gauge_data color matrices must be square")
    phase = FT_Phase(p, latt_info)
    gauge_ft_local_sum = contract(
        "detzyxab,etzyx->dab", gauge_data, phase
    )
    gauge_ft = SumMPI(cp.asnumpy(gauge_ft_local_sum))
    if core.getMPIRank() == 0:
        gauge_ft /= latt_info.global_volume
        if half_flag == 1:
            px, py, pz, pt = _canonical_fft_modes(p, latt_info)
            Lx, Ly, Lz, Lt = latt_info.size
            Gx, Gy, Gz, Gt = latt_info.grid_size
            half_link_phase = np.exp(
                -1j
                * np.pi
                * np.asarray(
                    [
                        px / (Lx * Gx),
                        py / (Ly * Gy),
                        pz / (Lz * Gz),
                        pt / (Lt * Gt),
                    ]
                )
            )
            gauge_ft *= half_link_phase[:, None, None]
    return gauge_ft


def FT_Prop_1mom(prop_data: cp.ndarray, p: List[Number], latt_info: LatticeInfo):
    Lx, Ly, Lz, Lt = _validate_checkerboard_geometry(latt_info)
    expected = (2, Lt, Lz, Ly, Lx // 2, 4, 4, 3, 3)
    if prop_data.ndim != 9 or prop_data.shape != expected:
        raise ValueError(
            "prop_data must have exact (2,Lt,Lz,Ly,Lx/2,4,4,3,3) layout"
        )
    phase = FT_Phase(p, latt_info)
    prop_ft_local_sum = contract(
        "etzyxijab,etzyx->ijab", prop_data, phase
    )
    prop_ft = SumMPI(cp.asnumpy(prop_ft_local_sum))
    if core.getMPIRank() == 0:
        prop_ft /= latt_info.global_volume
    return prop_ft


def FFT_Gauge_Allmom_MPI(gauge_data: cp.ndarray, latt_info: LatticeInfo, comm: MPI.Comm, half_flag = 1) -> cp.ndarray:
    #FFT transformation on gauge, cupy 4p_{tzyx}ab
    #Only use for Gx, Gy, Gz, Gt = 1 1 1 * !!!!!!

    half_flag = _choice_integer(half_flag, "half_flag", {0, 1})
    Lx, Ly, Lz, Lt = _validate_checkerboard_geometry(latt_info)
    Gx, Gy, Gz, Gt = latt_info.grid_size
    if (Gx, Gy, Gz, Gt) != (1, 1, 1, 1) or comm.size != 1:
        raise ValueError(
            "FFT_Gauge_Allmom_MPI currently requires one GPU; a correct multi-GPU "
            "global FFT needs a distributed CUDA FFT backend"
        )
    expected = (4, 2, Lt, Lz, Ly, Lx // 2)
    if gauge_data.ndim != 8 or gauge_data.shape[:6] != expected:
        raise ValueError(
            "gauge_data must have exact (4,2,Lt,Lz,Ly,Lx/2,Nc,Nc) layout"
        )
    if gauge_data.shape[-2] != gauge_data.shape[-1]:
        raise ValueError("gauge_data color matrices must be square")

    # Convert checkerboard x storage to lexicographic x with device indexing.
    t, z, y, x = cp.indices((Lt, Lz, Ly, Lx))
    parity = (t + z + y + x) & 1
    gauge_data_lex = gauge_data[:, parity, t, z, y, x // 2]
    result = cp.fft.fftn(gauge_data_lex, axes=(1, 2, 3, 4)) / latt_info.global_volume
    if half_flag == 1:
        p = [cp.fft.fftfreq(length) for length in (Lt, Lz, Ly, Lx)]
        pt, pz, py, px = cp.meshgrid(*p, indexing="ij")
        # A_mu lives at x + mu/2.  Each link direction therefore receives its
        # own exp(-i p_mu/2) Fourier-origin correction; a common phase built
        # from p_x+p_y+p_z+p_t is not the transform of a link-centered field.
        phase_factor = cp.stack(
            [
                cp.exp(-1j * cp.pi * px),
                cp.exp(-1j * cp.pi * py),
                cp.exp(-1j * cp.pi * pz),
                cp.exp(-1j * cp.pi * pt),
            ],
            axis=0,
        )
        result *= phase_factor[..., None, None]
    return result



######################################### Gluon Propagator and Etc. #########################################

def gaugeEMT_munu(gauge: LatticeGauge, g_0):
    #generate CPU,rank = 0 gauge_EMT_munu (4,4 = xyzt)
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    T_munu = cp.zeros((4, 4), dtype = cp.complex128)
    F_munu_all = F_munu(gauge, g_0, 0)
    Fmurho_Fnurho = 2 * contract("mretzyxab,nretzyxba->mn", F_munu_all, F_munu_all)
    F_sq = 2 * contract("mnetzyxab,mnetzyxba->", F_munu_all, F_munu_all) / 4.0
    for mu in range(4):
        for nu in range(4):
            T_munu[mu, nu] = Fmurho_Fnurho[mu, nu]
            if mu == nu:
                T_munu[mu, nu] -= F_sq
    # Spatial/lattice axes were already contracted locally, so use a direct
    # MPI reduction instead of pretending the 4x4 matrix has t,z,y,x axes.
    T_munu_all = SumMPI(cp.asnumpy(_checked_real(T_munu, "gaugeEMT_munu")))
    return T_munu_all


def gaugeEMT_mumu(gauge: LatticeGauge, g_0, def_type = 1):
    #generate CPU,rank = 0 gauge_EMT_mumu (4 = xyzt)
    #def_type = 1, 2, T_{mu mu} = 2sum_x Tr[F_{mu rho}F_{mu rho}-(1/4) g_{mu mu}F^2]
    #def_type = 0,    T_{mu mu} = -4/g_0^2 (sum_{nu!=mu,x}Tr[P_{mu,nu}(x)]-(1/4)sum_{rho!=nu,x}P_{rho,nu}(x))
    def_type = _choice_integer(def_type, "def_type", {0, 1, 2})
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    T_mumu = cp.zeros(4, dtype = cp.complex128)
    if def_type == 2 :
        F_munu_all = F_munu(gauge, g_0, 0)
        Fmurho_Fnurho = 2 * contract("mretzyxab,nretzyxba->mn", F_munu_all, F_munu_all)
        F_sq = 2 * contract("mnetzyxab,mnetzyxba->", F_munu_all, F_munu_all) / 4.0
        for mu in range(4):
            T_mumu[mu] = Fmurho_Fnurho[mu, mu] - F_sq
    elif def_type == 1 :
        E4i_all, Bij_all = F_munu(gauge, g_0, 1)
        Ftrhosq = 2 * (contract("etzyxab,etzyxba->", E4i_all[0], E4i_all[0]) + \
                       contract("etzyxab,etzyxba->", E4i_all[1], E4i_all[1]) + \
                       contract("etzyxab,etzyxba->", E4i_all[2], E4i_all[2]))
        Fzrhosq = 2 * (contract("etzyxab,etzyxba->", Bij_all[1], Bij_all[1]) + \
                       contract("etzyxab,etzyxba->", Bij_all[2], Bij_all[2]) + \
                       contract("etzyxab,etzyxba->", E4i_all[2], E4i_all[2]))
        Fyrhosq = 2 * (contract("etzyxab,etzyxba->", Bij_all[0], Bij_all[0]) + \
                       contract("etzyxab,etzyxba->", Bij_all[2], Bij_all[2]) + \
                       contract("etzyxab,etzyxba->", E4i_all[1], E4i_all[1]))
        Fxrhosq = 2 * (contract("etzyxab,etzyxba->", Bij_all[0], Bij_all[0]) + \
                       contract("etzyxab,etzyxba->", Bij_all[1], Bij_all[1]) + \
                       contract("etzyxab,etzyxba->", E4i_all[0], E4i_all[0]))
        E_sq = contract("detzyxab,detzyxba->", E4i_all, E4i_all) #E^2
        B_sq = contract("detzyxab,detzyxba->", Bij_all, Bij_all) #B^2
        F_sq = 2 * 2 * (E_sq + B_sq) / 4.0
        T_mumu[3] = Ftrhosq - F_sq
        T_mumu[2] = Fzrhosq - F_sq
        T_mumu[1] = Fyrhosq - F_sq
        T_mumu[0] = Fxrhosq - F_sq
    elif def_type == 0:
        plaq = plaq_munu(gauge)
        P_rhonu = 0.0
        for nu in range(4):
            for rho in range(4):
                if rho != nu:
                    P_rhonu += contract("etzyxaa->", plaq[rho, nu])
        for mu in range(4):
            P_munu  = 0.0
            for nu in range(4):
                if nu != mu:
                    P_munu += contract("etzyxaa->", plaq[mu, nu])
            T_mumu[mu] = (-4.0 / g_0**2) * (P_munu - P_rhonu / 4.0) #(d = xyzt)
    T_mumu_all = SumMPI(cp.asnumpy(_checked_real(T_mumu, "gaugeEMT_mumu")))
    return T_mumu_all


def gaugeEMT_mumu_tzyx(gauge: LatticeGauge, g_0, def_type = 1):
    def_type = _choice_integer(def_type, "def_type", {1, 2})
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    if def_type == 2 :
        F_munu_all = F_munu(gauge, g_0, 0)
        Fmurho_Fnurho = 2 * contract("mretzyxab,nretzyxba->mnetzyx", F_munu_all, F_munu_all)
        F_sq = 2 * contract("mnetzyxab,mnetzyxba->etzyx", F_munu_all, F_munu_all) / 4.0
        T_xx = Fmurho_Fnurho[0, 0] - F_sq
        T_yy = Fmurho_Fnurho[1, 1] - F_sq
        T_zz = Fmurho_Fnurho[2, 2] - F_sq
        T_tt = Fmurho_Fnurho[3, 3] - F_sq
    elif def_type == 1 :
        E4i_all, Bij_all = F_munu(gauge, g_0, 1)
        Ftrhosq = 2 * (contract("etzyxab,etzyxba->etzyx", E4i_all[0], E4i_all[0]) + \
                       contract("etzyxab,etzyxba->etzyx", E4i_all[1], E4i_all[1]) + \
                       contract("etzyxab,etzyxba->etzyx", E4i_all[2], E4i_all[2]))
        Fzrhosq = 2 * (contract("etzyxab,etzyxba->etzyx", Bij_all[1], Bij_all[1]) + \
                       contract("etzyxab,etzyxba->etzyx", Bij_all[2], Bij_all[2]) + \
                       contract("etzyxab,etzyxba->etzyx", E4i_all[2], E4i_all[2]))
        Fyrhosq = 2 * (contract("etzyxab,etzyxba->etzyx", Bij_all[0], Bij_all[0]) + \
                       contract("etzyxab,etzyxba->etzyx", Bij_all[2], Bij_all[2]) + \
                       contract("etzyxab,etzyxba->etzyx", E4i_all[1], E4i_all[1]))
        Fxrhosq = 2 * (contract("etzyxab,etzyxba->etzyx", Bij_all[0], Bij_all[0]) + \
                       contract("etzyxab,etzyxba->etzyx", Bij_all[1], Bij_all[1]) + \
                       contract("etzyxab,etzyxba->etzyx", E4i_all[0], E4i_all[0]))
        E_sq = contract("detzyxab,detzyxba->etzyx", E4i_all, E4i_all) #E^2
        B_sq = contract("detzyxab,detzyxba->etzyx", Bij_all, Bij_all) #B^2
        F_sq = 2 * 2 * (E_sq + B_sq) / 4.0
        T_xx = Fxrhosq - F_sq
        T_yy = Fyrhosq - F_sq
        T_zz = Fzrhosq - F_sq
        T_tt = Ftrhosq - F_sq
    return _assemble_site_fields((T_xx, T_yy, T_zz, T_tt))


def ExA_t(gauge: LatticeGauge, g_0, half_flag = 1):
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    half_flag = _choice_integer(half_flag, "half_flag", {0, 1})
    if tuple(getattr(gauge.latt_info, "grid_size", (1, 1, 1, 1)))[3] != 1:
        raise NotImplementedError("time-resolved ExA_t currently requires Gt=1")
    A_mu_all = A_mu(gauge, g_0, half_flag)
    E4i_all, Bij_all = F_munu(gauge, g_0)
    ExA = cp.einsum("ijk,jetzyxab,ketzyxba->it", epsilon(3), E4i_all, A_mu_all[:3])
    return SumMPI(cp.asnumpy(_checked_real(ExA, "ExA_t")))


def Topological_current_Kmu(gauge: LatticeGauge, g_0, half_flag = 1):
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    half_flag = _choice_integer(half_flag, "half_flag", {0, 1})
    A_mu_all = A_mu(gauge, g_0, half_flag)
    E4i_all, Bij_all = F_munu(gauge, g_0)
    fac = (-1j) * (2/3) * g_0
    sign = epsilon(4)

    e3201_AB  = sign[3,2,0,1] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[2], Bij_all[0]))
    e3201_3A  = sign[3,2,0,1] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[2], A_mu_all[0], A_mu_all[1])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[2], A_mu_all[1], A_mu_all[0]))
    e3102_AB  = sign[3,1,0,2] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[1], Bij_all[1]))
    e3102_3A  = sign[3,1,0,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[1], A_mu_all[0], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[1], A_mu_all[2], A_mu_all[0]))
    e3012_AB  = sign[3,0,1,2] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[0], Bij_all[2]))
    e3012_3A  = sign[3,0,1,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[0], A_mu_all[1], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[0], A_mu_all[2], A_mu_all[1]))
    Kt = e3201_AB + e3102_AB + e3012_AB + e3201_3A + e3102_3A + e3012_3A

    e2301_AtB = sign[2,3,0,1] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[3], Bij_all[0]))
    e2301_3A  = sign[2,3,0,1] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[3], A_mu_all[0], A_mu_all[1])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[3], A_mu_all[1], A_mu_all[0]))
    e2103_AE  = sign[2,1,0,3] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[1], -E4i_all[0]))
    e2103_3A  = sign[2,1,0,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[1], A_mu_all[0], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[1], A_mu_all[3], A_mu_all[0]))
    e2013_AE  = sign[2,0,1,3] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[0], -E4i_all[1]))
    e2013_3A  = sign[2,0,1,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[0], A_mu_all[1], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[0], A_mu_all[3], A_mu_all[1]))
    Kz = e2301_AtB + e2103_AE + e2013_AE + e2301_3A + e2103_3A + e2013_3A

    e1302_AtB = sign[1,3,0,2] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[3], Bij_all[1]))
    e1302_3A  = sign[1,3,0,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[3], A_mu_all[0], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[3], A_mu_all[2], A_mu_all[0]))
    e1203_AE  = sign[1,2,0,3] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[2], -E4i_all[0]))
    e1203_3A  = sign[1,2,0,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[2], A_mu_all[0], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[2], A_mu_all[3], A_mu_all[0]))
    e1023_AE  = sign[1,0,2,3] * 2 *   (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[0], -E4i_all[2]))
    e1023_3A  = sign[1,0,2,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[0], A_mu_all[2], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[0], A_mu_all[3], A_mu_all[2]))
    Ky = e1302_AtB + e1203_AE + e1023_AE + e1302_3A + e1203_3A + e1023_3A

    e0312_AtB = sign[0,3,1,2] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[3], Bij_all[2]))
    e0312_3A  = sign[0,3,1,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[3], A_mu_all[1], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[3], A_mu_all[2], A_mu_all[1]))
    e0213_AE  = sign[0,2,1,3] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[2], -E4i_all[1]))
    e0213_3A  = sign[0,2,1,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[2], A_mu_all[1], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[2], A_mu_all[3], A_mu_all[1]))
    e0123_AE  = sign[0,1,2,3] * 2   * (+cp.einsum("etzyxab,etzyxba->",         A_mu_all[1], -E4i_all[2]))
    e0123_3A  = sign[0,1,2,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[1], A_mu_all[2], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->", A_mu_all[1], A_mu_all[3], A_mu_all[2]))
    Kx = e0312_AtB + e0213_AE + e0123_AE + e0312_3A + e0213_3A + e0123_3A

    reduced = SumMPI(cp.asnumpy(cp.stack((Kx, Ky, Kz, Kt))))
    if reduced is None:
        return None, None, None, None
    return tuple(reduced)


def Topological_current_Kmu_t(gauge: LatticeGauge, g_0, half_flag = 1):
    """Return selected Kt/Kz decomposition terms as local-time series.

    Despite the historical name, this is not a complete four-component K_mu
    time series. It also requires an undecomposed time direction so that the
    final world reduction cannot mix different global time slices.
    """

    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    half_flag = _choice_integer(half_flag, "half_flag", {0, 1})
    if tuple(getattr(gauge.latt_info, "grid_size", (1, 1, 1, 1)))[3] != 1:
        raise NotImplementedError(
            "Topological_current_Kmu_t currently requires Gt=1"
        )
    A_mu_all = A_mu(gauge, g_0, half_flag)
    E4i_all, Bij_all = F_munu(gauge, g_0)
    fac = (-1j) * (2/3) * g_0
    sign = epsilon(4)

    e3201_AB  = sign[3,2,0,1] * 2   * (+cp.einsum("etzyxab,etzyxba->t",         A_mu_all[2], Bij_all[0]))
    e3201_3A  = sign[3,2,0,1] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[2], A_mu_all[0], A_mu_all[1])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[2], A_mu_all[1], A_mu_all[0]))
    e3102_AB  = sign[3,1,0,2] * 2   * (+cp.einsum("etzyxab,etzyxba->t",         A_mu_all[1], Bij_all[1]))
    e3102_3A  = sign[3,1,0,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[1], A_mu_all[0], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[1], A_mu_all[2], A_mu_all[0]))
    e3012_AB  = sign[3,0,1,2] * 2   * (+cp.einsum("etzyxab,etzyxba->t",         A_mu_all[0], Bij_all[2]))
    e3012_3A  = sign[3,0,1,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[0], A_mu_all[1], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[0], A_mu_all[2], A_mu_all[1]))
    Kt_AF  = e3201_AB + e3102_AB + e3012_AB
    Kt_3A  = e3201_3A + e3102_3A + e3012_3A

    e2301_AtB = sign[2,3,0,1] * 2   * (+cp.einsum("etzyxab,etzyxba->t",         A_mu_all[3], Bij_all[0]))
    e2301_3A  = sign[2,3,0,1] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[3], A_mu_all[0], A_mu_all[1])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[3], A_mu_all[1], A_mu_all[0]))
    e2103_AE  = sign[2,1,0,3] * 2   * (+cp.einsum("etzyxab,etzyxba->t",         A_mu_all[1], -E4i_all[0]))
    e2103_3A  = sign[2,1,0,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[1], A_mu_all[0], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[1], A_mu_all[3], A_mu_all[0]))
    e2013_AE  = sign[2,0,1,3] * 2   * (+cp.einsum("etzyxab,etzyxba->t",         A_mu_all[0], -E4i_all[1]))
    e2013_3A  = sign[2,0,1,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[0], A_mu_all[1], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->t", A_mu_all[0], A_mu_all[3], A_mu_all[1]))
    Kz_AtB = e2301_AtB
    Kz_ExA = e2103_AE + e2013_AE
    Kz_3A  = e2301_3A + e2103_3A + e2013_3A

    components = cp.stack((Kt_AF, Kt_3A, Kz_AtB, Kz_ExA, Kz_3A))
    reduced = SumMPI(
        cp.asnumpy(_checked_real(components, "Topological_current_Kmu_t"))
    )
    if reduced is None:
        return None, None, None, None, None
    return tuple(reduced)


def Topological_current_Kmu_tzyx(gauge: LatticeGauge, g_0, half_flag = 1):
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    half_flag = _choice_integer(half_flag, "half_flag", {0, 1})
    A_mu_all = A_mu(gauge, g_0, half_flag)
    E4i_all, Bij_all = F_munu(gauge, g_0)
    fac = (-1j) * (2/3) * g_0
    sign = epsilon(4)

    e3201_AB  = sign[3,2,0,1] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[2], Bij_all[0]))
    e3201_3A  = sign[3,2,0,1] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[2], A_mu_all[0], A_mu_all[1])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[2], A_mu_all[1], A_mu_all[0]))
    e3102_AB  = sign[3,1,0,2] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[1], Bij_all[1]))
    e3102_3A  = sign[3,1,0,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[1], A_mu_all[0], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[1], A_mu_all[2], A_mu_all[0]))
    e3012_AB  = sign[3,0,1,2] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[0], Bij_all[2]))
    e3012_3A  = sign[3,0,1,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[0], A_mu_all[1], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[0], A_mu_all[2], A_mu_all[1]))
    Kt = e3201_AB + e3102_AB + e3012_AB + e3201_3A + e3102_3A + e3012_3A

    e2301_AtB = sign[2,3,0,1] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[3], Bij_all[0]))
    e2301_3A  = sign[2,3,0,1] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[3], A_mu_all[0], A_mu_all[1])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[3], A_mu_all[1], A_mu_all[0]))
    e2103_AE  = sign[2,1,0,3] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[1], -E4i_all[0]))
    e2103_3A  = sign[2,1,0,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[1], A_mu_all[0], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[1], A_mu_all[3], A_mu_all[0]))
    e2013_AE  = sign[2,0,1,3] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[0], -E4i_all[1]))
    e2013_3A  = sign[2,0,1,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[0], A_mu_all[1], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[0], A_mu_all[3], A_mu_all[1]))
    Kz = e2301_AtB + e2103_AE + e2013_AE + e2301_3A + e2103_3A + e2013_3A

    e1302_AtB = sign[1,3,0,2] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[3], Bij_all[1]))
    e1302_3A  = sign[1,3,0,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[3], A_mu_all[0], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[3], A_mu_all[2], A_mu_all[0]))
    e1203_AE  = sign[1,2,0,3] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[2], -E4i_all[0]))
    e1203_3A  = sign[1,2,0,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[2], A_mu_all[0], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[2], A_mu_all[3], A_mu_all[0]))
    e1023_AE  = sign[1,0,2,3] * 2 *   (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[0], -E4i_all[2]))
    e1023_3A  = sign[1,0,2,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[0], A_mu_all[2], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[0], A_mu_all[3], A_mu_all[2]))
    Ky = e1302_AtB + e1203_AE + e1023_AE + e1302_3A + e1203_3A + e1023_3A

    e0312_AtB = sign[0,3,1,2] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[3], Bij_all[2]))
    e0312_3A  = sign[0,3,1,2] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[3], A_mu_all[1], A_mu_all[2])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[3], A_mu_all[2], A_mu_all[1]))
    e0213_AE  = sign[0,2,1,3] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[2], -E4i_all[1]))
    e0213_3A  = sign[0,2,1,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[2], A_mu_all[1], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[2], A_mu_all[3], A_mu_all[1]))
    e0123_AE  = sign[0,1,2,3] * 2   * (+cp.einsum("etzyxab,etzyxba->etzyx",         A_mu_all[1], -E4i_all[2]))
    e0123_3A  = sign[0,1,2,3] * fac * (+cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[1], A_mu_all[2], A_mu_all[3])
                                       -cp.einsum("etzyxab,etzyxbc,etzyxca->etzyx", A_mu_all[1], A_mu_all[3], A_mu_all[2]))
    Kx = e0312_AtB + e0213_AE + e0123_AE + e0312_3A + e0213_3A + e0123_3A

    return _assemble_site_fields((Kx, Ky, Kz, Kt))



######################################### Momentum selection #########################################

def generate_pselect_mask(latt_info: LatticeInfo, mommin, mommax, cut_list, mode_list):
    #select specific momentum mode
    #mode = [1:7]  p_{xyzt} = 0001, 0010, 0011. 0100, 0101, 0110, 0111
    #mode = [8:15] p_{xyzt} = 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111
    #cut_value >= 0.25(4*1), 0.33(3*1), 0.50(2*1), 1.00(1*1)
    if isinstance(mommin, (bool, np.bool_)) or isinstance(mommax, (bool, np.bool_)):
        raise TypeError("mommin and mommax must be integers")
    if not isinstance(mommin, (int, np.integer)) or not isinstance(mommax, (int, np.integer)):
        raise TypeError("mommin and mommax must be integers")
    if mommin < 0 or mommax < mommin:
        raise ValueError("momentum bounds require 0 <= mommin <= mommax")
    if len(mode_list) != len(cut_list):
        raise ValueError("mode_list and cut_list must have the same length")
    modes = []
    cuts = []
    for mode in mode_list:
        if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, (int, np.integer)):
            raise TypeError("mode_list entries must be integers")
        if mode not in range(1, 16):
            raise ValueError("mode_list entries must be in 1..15")
        modes.append(int(mode))
    for cut_value in cut_list:
        cut = _finite_real_scalar(cut_value, "cut_list entry", positive=True)
        if cut > 1.0:
            raise ValueError("cut_list entries must satisfy 0 < cut <= 1")
        cuts.append(cut)

    Lx, Ly, Lz, Lt = latt_info.size
    Gx, Gy, Gz, Gt = latt_info.grid_size
    ap_x = 2 * np.pi * np.fft.fftfreq(Lx * Gx, d=1.0)
    ap_y = 2 * np.pi * np.fft.fftfreq(Ly * Gy, d=1.0)
    ap_z = 2 * np.pi * np.fft.fftfreq(Lz * Gz, d=1.0)
    ap_t = 2 * np.pi * np.fft.fftfreq(Lt * Gt, d=1.0)
    apt, apz, apy, apx = np.meshgrid(ap_t, ap_z, ap_y, ap_x, indexing="ij")
    extents = (Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt)

    def add_candidate(select_mask, momsqr_lst, ix, iy, iz, it, cut_value):
        """Canonicalize one integer label to NumPy FFT bins before selection."""

        canonical = tuple(
            (int(component) + extent // 2) % extent - extent // 2
            for component, extent in zip((ix, iy, iz, it), extents)
        )
        cix, ciy, ciz, cit = canonical
        components = 2 * np.pi * np.asarray(canonical, dtype=float) / np.asarray(
            extents, dtype=float
        )
        p2 = float(np.sum(components**2))
        p4 = float(np.sum(components**4))
        if p2 <= 0 or p4 / (p2**2) > cut_value:
            return select_mask
        if any(np.isclose(p2, previous) for previous in momsqr_lst):
            return select_mask
        momsqr_lst.append(p2)
        mask_condition = (
            np.isclose(apx, 2 * np.pi * cix / extents[0])
            & np.isclose(apy, 2 * np.pi * ciy / extents[1])
            & np.isclose(apz, 2 * np.pi * ciz / extents[2])
            & np.isclose(apt, 2 * np.pi * cit / extents[3])
        )
        return select_mask | mask_condition

    masks = []
    for mode, cut_value in zip(modes, cuts):
        binary_pattern = f'{mode:04b}'
        momsqr_lst = []
        select_mask = np.zeros_like(apx, dtype = bool)
        if mode in range(8, 16):
            for ix in range(mommin, mommax + 1):
                if binary_pattern[0] == '0' and ix != 0:
                    continue
                iy_min, iy_max = (0, 0) if binary_pattern[1] == '0' else (ix - 1, ix + 1)
                for iy in range(iy_min, iy_max + 1):
                    iz_min, iz_max = (0, 0) if binary_pattern[2] == '0' else (ix - 1, ix + 1)
                    for iz in range(iz_min, iz_max + 1):
                        it0 = (ix * (Lt * Gt)) // (Lx * Gx)
                        it_min, it_max = (0, 0) if binary_pattern[3] == '0' else (it0 - 1, it0 + 1)
                        for it in range(it_min, it_max + 1):
                            select_mask = add_candidate(
                                select_mask, momsqr_lst, ix, iy, iz, it, cut_value
                            )
        elif mode in range(4, 8):
            for iy in range(mommin, mommax + 1):
                if binary_pattern[1] == '0' and iy != 0:
                    continue
                ix_min, ix_max = (0, 0) if binary_pattern[0] == '0' else (iy - 1, iy + 1)
                for ix in range(ix_min, ix_max + 1):
                    iz_min, iz_max = (0, 0) if binary_pattern[2] == '0' else (iy - 1, iy + 1)
                    for iz in range(iz_min, iz_max + 1):
                        it0 = (iy * (Lt * Gt)) // (Ly * Gy)
                        it_min, it_max = (0, 0) if binary_pattern[3] == '0' else (it0 - 1, it0 + 1)
                        for it in range(it_min, it_max + 1):
                            select_mask = add_candidate(
                                select_mask, momsqr_lst, ix, iy, iz, it, cut_value
                            )
        elif mode in range(2, 4):
            for iz in range(mommin, mommax + 1):
                if binary_pattern[2] == '0' and iz != 0:
                    continue
                ix_min, ix_max = (0, 0) if binary_pattern[0] == '0' else (iz - 1, iz + 1)
                for ix in range(ix_min, ix_max + 1):
                    iy_min, iy_max = (0, 0) if binary_pattern[1] == '0' else (iz - 1, iz + 1)
                    for iy in range(iy_min, iy_max + 1):
                        it0 = (iz * (Lt * Gt)) // (Lz * Gz)
                        it_min, it_max = (0, 0) if binary_pattern[3] == '0' else (it0 - 1, it0 + 1)
                        for it in range(it_min, it_max + 1):
                            select_mask = add_candidate(
                                select_mask, momsqr_lst, ix, iy, iz, it, cut_value
                            )
        elif mode == 1:
            for it in range(mommin, mommax + 1):
                if binary_pattern[3] == '0' and it != 0:
                    continue
                ix0 = (it * (Lx * Gx)) // (Lt * Gt)
                ix_min, ix_max = (0, 0) if binary_pattern[0] == '0' else (ix0 - 1, ix0 + 1)
                for ix in range(ix_min, ix_max + 1):
                    iy0 = (it * (Ly * Gy)) // (Lt * Gt)
                    iy_min, iy_max = (0, 0) if binary_pattern[1] == '0' else (iy0 - 1, iy0 + 1)
                    for iy in range(iy_min, iy_max + 1):
                        iz0 = (it * (Lz * Gz)) // (Lt * Gt)
                        iz_min, iz_max = (0, 0) if binary_pattern[2] == '0' else (iz0 - 1, iz0 + 1)
                        for iz in range(iz_min, iz_max + 1):
                            select_mask = add_candidate(
                                select_mask, momsqr_lst, ix, iy, iz, it, cut_value
                            )
        masks.append(select_mask)
    return masks


__all__ = [
    "g_0",
    "plaq_munu",
    "A_mu",
    "F_munu",
    "F_and_tildeF",
    "FW_tildeFW",
    "FW_FW",
    "FF",
    "FT_Phase",
    "FT_Gauge_1mom",
    "FT_Prop_1mom",
    "FFT_Gauge_Allmom_MPI",
    "gaugeEMT_munu",
    "gaugeEMT_mumu",
    "gaugeEMT_mumu_tzyx",
    "ExA_t",
    "Topological_current_Kmu",
    "Topological_current_Kmu_t",
    "Topological_current_Kmu_tzyx",
    "generate_pselect_mask",
]
