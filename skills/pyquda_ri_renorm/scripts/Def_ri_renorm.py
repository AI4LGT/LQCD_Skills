"""RI-prime/MOM and custom nonexceptional RI projection building blocks."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _contains_boolean(value: Any) -> bool:
    """Detect Boolean provenance before NumPy dtype inference can erase it."""

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


def _xp(array: Any):
    if type(array).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return np


def _require_same_backend(reference: Any, *arrays: Any) -> None:
    xp = _xp(reference)
    if any(_xp(array) is not xp for array in arrays):
        raise TypeError("all spin-color fields and spin matrices must use one backend")


def _finite_real_scalar(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_four_vector(values: Sequence[float], name: str) -> np.ndarray:
    if _contains_boolean(values):
        raise TypeError(f"{name} components must be real, not booleans")
    raw = np.asarray(values)
    if raw.shape != (4,):
        raise ValueError(f"{name} must be one four-vector")
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TypeError(f"{name} components must be real")
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} components must be finite real numbers") from error
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} components must be finite")
    return vector


def _require_spin_matrices(reference: Any, *matrices: Any) -> None:
    _require_same_backend(reference, *matrices)
    xp = _xp(reference)
    for matrix in (reference, *matrices):
        if getattr(matrix, "shape", None) != (4, 4):
            raise ValueError("all Dirac spin matrices must have exact shape (4,4)")
        try:
            finite = xp.isfinite(matrix).all()
        except TypeError as error:
            raise TypeError("all Dirac spin matrices must be numerical") from error
        if xp is np:
            is_finite = bool(np.asarray(finite).item())
        else:
            is_finite = bool(np.asarray(xp.asnumpy(finite)).item())
        if not is_finite:
            raise ValueError("all Dirac spin matrices must be finite")


def _require_spin_color_field(field: Any, name: str) -> None:
    if getattr(field, "ndim", 0) < 4 or field.shape[-4:] != (4, 4, 3, 3):
        raise ValueError(f"{name} must end in (4,4,3,3)")


@dataclass(frozen=True)
class RIKinematics:
    p_in: Any
    p_out: Any
    q: Any
    mu2: float
    omega: float
    scheme: str
    projector_scheme: str
    momentum_definition: str


@dataclass(frozen=True)
class RIConstants:
    Zq: Any
    ZA: Any
    ZV: Any
    ZS: Any
    ZP: Any
    ZT: Any
    projected_vertices: Mapping[str, Any]
    kinematics: RIKinematics
    zq_definition: str
    scheme_identity: str
    continuum_matching_status: str


@dataclass(frozen=True)
class RIE2EValidation:
    """A fail-closed readiness result; it never starts a runtime job."""

    status: str
    scheme_identity: str
    checked_controls: tuple[str, ...]


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


def _e2e_nonnegative_scalar(value: Any, name: str) -> float:
    return _finite_real_scalar(value, name, minimum=0.0)


def _e2e_complex_array(
    value: Any, name: str, *, field: bool = False, allow_scalar: bool = False
) -> None:
    """Accept real runtime arrays only; paths, lists, and scalar claims fail closed."""

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        raise TypeError(f"{name} must be an actual complex array, not a path or summary")
    try:
        normalized_shape = tuple(int(item) for item in shape)
        normalized_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose a numerical array shape and dtype") from error
    if (not normalized_shape and not allow_scalar) or any(item <= 0 for item in normalized_shape):
        raise ValueError(f"{name} must be non-empty")
    if not np.issubdtype(normalized_dtype, np.complexfloating):
        raise TypeError(f"{name} must retain complex components")
    # Evidence arrays are consumed as already-produced numerical results.  A
    # shape/dtype-only check would let NaN/Inf payloads through and still mark
    # the E2E record as validated, even though such values cannot support a
    # trustworthy NPR claim.  Check finiteness on the originating backend and
    # transfer only the resulting scalar for CuPy arrays.
    xp = _xp(value)
    try:
        finite = xp.isfinite(value).all()
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numerical complex values") from error
    if xp is np:
        is_finite = bool(np.asarray(finite).item())
    else:
        is_finite = bool(np.asarray(xp.asnumpy(finite)).item())
    if not is_finite:
        raise ValueError(f"{name} must contain only finite values")
    if field and (len(normalized_shape) < 4 or normalized_shape[-4:] != (4, 4, 3, 3)):
        raise ValueError(f"{name} must end in the spin-color layout (4,4,3,3)")


def _e2e_complex_tolerance(value: Any, name: str) -> Mapping[str, Any]:
    tolerance = _e2e_mapping(value, name)
    if tolerance.get("comparison") != "real_and_imaginary":
        raise ValueError(
            f"{name}.comparison must be 'real_and_imaginary'; magnitude-only checks are forbidden"
        )
    _e2e_nonnegative_scalar(tolerance.get("atol"), f"{name}.atol")
    _e2e_nonnegative_scalar(tolerance.get("rtol"), f"{name}.rtol")
    return tolerance


def _e2e_componentwise_close(observed: Any, expected: Any, tolerance: Any, name: str) -> None:
    _e2e_complex_array(observed, f"{name}.observed", allow_scalar=True)
    _e2e_complex_array(expected, f"{name}.expected", allow_scalar=True)
    if tuple(observed.shape) != tuple(expected.shape):
        raise ValueError(f"{name}.observed and {name}.expected must have one exact shape")
    if _xp(observed) is not _xp(expected):
        raise TypeError(f"{name} must not compare host and device arrays implicitly")
    checked_tolerance = _e2e_complex_tolerance(tolerance, f"{name}.tolerance")
    atol = _e2e_nonnegative_scalar(checked_tolerance["atol"], f"{name}.tolerance.atol")
    rtol = _e2e_nonnegative_scalar(checked_tolerance["rtol"], f"{name}.tolerance.rtol")
    xp = _xp(observed)
    real_ok = xp.allclose(observed.real, expected.real, rtol=rtol, atol=atol)
    imag_ok = xp.allclose(observed.imag, expected.imag, rtol=rtol, atol=atol)
    if xp is np:
        close = bool(real_ok) and bool(imag_ok)
    else:
        close = bool(np.asarray(xp.asnumpy(real_ok)).item()) and bool(
            np.asarray(xp.asnumpy(imag_ok)).item()
        )
    if not close:
        raise ValueError(f"{name} fails the frozen component-wise complex tolerance")


def _e2e_control(control: Any, name: str) -> None:
    record = _e2e_mapping(control, name)
    applicable = _e2e_bool(record.get("applicable"), f"{name}.applicable")
    if not applicable:
        _e2e_text(record.get("reason"), f"{name}.reason")
        return
    if record.get("status") != "passed":
        raise ValueError(f"{name}.status must be 'passed' when the control is applicable")
    _e2e_componentwise_close(
        record.get("observed"), record.get("expected"), record.get("tolerance"), name
    )


def _e2e_vertex_channels(value: Any, name: str) -> None:
    vertices = _e2e_mapping(value, name)
    required = {"S", "P", "V", "A", "T"}
    if set(vertices) != required:
        raise ValueError(f"{name} must contain exactly {sorted(required)}")
    for channel in ("S", "P"):
        _e2e_complex_array(vertices[channel], f"{name}.{channel}", field=True)
    for channel, count in (("V", 4), ("A", 4), ("T", 6)):
        components = vertices[channel]
        if isinstance(components, (str, bytes)) or not isinstance(components, SequenceABC):
            raise TypeError(f"{name}.{channel} must be a {count}-component sequence")
        if len(components) != count:
            raise ValueError(f"{name}.{channel} must contain exactly {count} components")
        for index, component in enumerate(components):
            _e2e_complex_array(component, f"{name}.{channel}[{index}]", field=True)


def validate_ri_e2e_record(record: Mapping[str, Any]) -> RIE2EValidation:
    """Validate an RI E2E evidence record without loading data or submitting work.

    The record must already hold the actual gauge-fixed propagator/vertex arrays
    and the small complex control arrays.  This function intentionally has no
    scheduler, MPI, PyQUDA, or filesystem side effects; a missing prerequisite
    is an error rather than an invitation to synthesize or run it.
    """

    record = _e2e_mapping(record, "record")
    gauge_fixing = _e2e_mapping(record.get("gauge_fixing"), "gauge_fixing")
    for key in ("functional", "algorithm"):
        _e2e_text(gauge_fixing.get(key), f"gauge_fixing.{key}")
    _e2e_nonnegative_scalar(gauge_fixing.get("residual"), "gauge_fixing.residual")

    propagator_in = record.get("propagator_in")
    propagator_out = record.get("propagator_out")
    _e2e_complex_array(propagator_in, "propagator_in", field=True)
    _e2e_complex_array(propagator_out, "propagator_out", field=True)
    if propagator_in is propagator_out:
        raise ValueError("independent incoming/outgoing legs must not alias one array")
    _e2e_vertex_channels(record.get("green_functions"), "green_functions")
    _e2e_vertex_channels(record.get("amputated_vertices"), "amputated_vertices")

    amputation = _e2e_mapping(record.get("amputation"), "amputation")
    if not _e2e_bool(amputation.get("independent_legs"), "amputation.independent_legs"):
        raise ValueError("amputation.independent_legs must be true")
    if amputation.get("formula") != "S_out^-1 G S_in^-1":
        raise ValueError("amputation.formula must be exactly 'S_out^-1 G S_in^-1'")
    if amputation.get("incoming_leg") != "propagator_in" or amputation.get("outgoing_leg") != "propagator_out":
        raise ValueError("amputation must explicitly name the incoming and outgoing leg ownership")

    tree_check = _e2e_mapping(record.get("tree_or_weak_field_check"), "tree_or_weak_field_check")
    if tree_check.get("kind") not in {"tree", "weak_field"}:
        raise ValueError("tree_or_weak_field_check.kind must be 'tree' or 'weak_field'")
    if not _e2e_bool(tree_check.get("passed"), "tree_or_weak_field_check.passed"):
        raise ValueError("tree_or_weak_field_check.passed must be true")
    _e2e_componentwise_close(
        tree_check.get("observed"),
        tree_check.get("expected"),
        tree_check.get("tolerance"),
        "tree_or_weak_field_check",
    )

    ward_controls = _e2e_mapping(record.get("ward_controls"), "ward_controls")
    if set(ward_controls) != {"vector", "axial"}:
        raise ValueError("ward_controls must contain exactly vector and axial controls")
    _e2e_control(ward_controls["vector"], "ward_controls.vector")
    _e2e_control(ward_controls["axial"], "ward_controls.axial")

    renormalization = _e2e_mapping(record.get("renormalization"), "renormalization")
    zq_definition = _e2e_text(renormalization.get("zq_definition"), "renormalization.zq_definition")
    projector = _e2e_text(renormalization.get("projector"), "renormalization.projector")
    scheme_identity = _e2e_text(renormalization.get("scheme_identity"), "renormalization.scheme_identity")
    if zq_definition not in {"incoming", "outgoing", "arithmetic_mean", "geometric_mean"}:
        raise ValueError("renormalization.zq_definition is not an explicit supported leg rule")
    if projector not in {"gamma_mu", "qslash"}:
        raise ValueError("renormalization.projector must be gamma_mu or qslash")

    raw_ri = _e2e_mapping(record.get("raw_ri"), "raw_ri")
    if raw_ri.get("state") != "raw_ri":
        raise ValueError("raw_ri.state must be raw_ri")
    if raw_ri.get("scheme_identity") != scheme_identity:
        raise ValueError("raw_ri.scheme_identity must equal renormalization.scheme_identity")
    factors = _e2e_mapping(raw_ri.get("factors"), "raw_ri.factors")
    if set(factors) != {"Zq", "ZA", "ZV", "ZS", "ZP", "ZT"}:
        raise ValueError("raw_ri.factors must contain exactly Zq/ZA/ZV/ZS/ZP/ZT")
    for name, value in factors.items():
        _e2e_complex_array(value, f"raw_ri.factors.{name}", allow_scalar=True)

    matching = _e2e_mapping(record.get("continuum_matching"), "continuum_matching")
    if matching is raw_ri:
        raise ValueError("raw_ri and continuum_matching must be separate records")
    if matching.get("state") not in {"not_requested", "unmapped_fail_closed", "converted"}:
        raise ValueError("continuum_matching.state must be not_requested, unmapped_fail_closed, or converted")
    if matching.get("state") == "converted":
        _e2e_text(matching.get("target_scheme"), "continuum_matching.target_scheme")
        _e2e_complex_array(matching.get("factors"), "continuum_matching.factors")

    return RIE2EValidation(
        status="validated_e2e_record_no_submission",
        scheme_identity=scheme_identity,
        checked_controls=("tree_or_weak_field", "vector_ward", "axial_ward", "raw_ri_vs_matching"),
    )


def validate_kinematics(
    p_in: Sequence[float],
    p_out: Sequence[float],
    omega: float,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-12,
    *,
    projector_scheme: str | None = None,
    momentum_definition: str | None = None,
) -> RIKinematics:
    """Validate p_in^2=p_out^2=mu^2 and q^2=omega*mu^2.

    `scheme` is a kinematic/projector-family label, not a complete continuum
    scheme identity. The latter also requires the Zq definition and is built
    by `compute_ri_constants`. Exceptional propagator-Zq kinematics are named
    RIprime/MOM; nonexceptional labels deliberately avoid claiming a standard
    RI/SMOM scheme while vector-vertex Zq is not implemented.
    """

    if not isinstance(momentum_definition, str) or not momentum_definition.strip():
        raise ValueError(
            "momentum_definition is required (for example continuum, kinetic, or improved)"
        )
    pin = _finite_four_vector(p_in, "p_in")
    pout = _finite_four_vector(p_out, "p_out")
    omega = _finite_real_scalar(omega, "omega", minimum=0.0)
    rtol = _finite_real_scalar(rtol, "rtol", minimum=0.0)
    atol = _finite_real_scalar(atol, "atol", minimum=0.0)
    pin2 = float(pin @ pin)
    pout2 = float(pout @ pout)
    if pin2 <= 0 or not np.isclose(pin2, pout2, rtol=rtol, atol=atol):
        raise ValueError("kinematics require p_in^2 = p_out^2 = mu^2 > 0")
    q = pout - pin
    if not np.isclose(float(q @ q), omega * pin2, rtol=rtol, atol=atol):
        raise ValueError("kinematics require (p_out-p_in)^2 = omega*mu^2")
    if np.isclose(omega, 0.0, rtol=rtol, atol=atol):
        if projector_scheme is None:
            projector_scheme = "gamma_mu"
        if projector_scheme != "gamma_mu":
            raise ValueError("exceptional kinematics require projector_scheme='gamma_mu'")
        scheme = "RIprime/MOM"
    else:
        if projector_scheme not in {"gamma_mu", "qslash"}:
            raise ValueError(
                "nonexceptional kinematics require explicit projector_scheme="
                "'gamma_mu' or 'qslash'"
            )
        prefix = (
            "symmetric_nonexceptional"
            if np.isclose(omega, 1.0, rtol=rtol, atol=atol)
            else "generalized_nonexceptional"
        )
        scheme = f"{prefix}_{projector_scheme}"
    return RIKinematics(
        pin,
        pout,
        q,
        pin2,
        float(omega),
        scheme,
        projector_scheme,
        str(momentum_definition),
    )


def _to_spin_color_matrix(field: Any) -> Any:
    """Map `(...,4,4,3,3)` to `(...,12,12)`."""

    if field.shape[-4:] != (4, 4, 3, 3):
        raise ValueError("field must end in (4,4,3,3)")
    return field.swapaxes(-3, -2).reshape(field.shape[:-4] + (12, 12))


def _from_spin_color_matrix(matrix: Any) -> Any:
    if matrix.shape[-2:] != (12, 12):
        raise ValueError("matrix must end in (12,12)")
    return matrix.reshape(matrix.shape[:-2] + (4, 3, 4, 3)).swapaxes(-3, -2)


def inverse_propagator(propagator: Any) -> Any:
    """Invert all spin-color blocks of a momentum-space propagator."""

    _require_spin_color_field(propagator, "propagator")
    xp = _xp(propagator)
    return _from_spin_color_matrix(xp.linalg.inv(_to_spin_color_matrix(propagator)))


def amputate_vertex(
    propagator_out: Any,
    green_function: Any,
    propagator_in: Any,
) -> Any:
    """Return S_out^-1 G_O S_in^-1.

    This is the standard bilinear amputation algebra used by momentum-
    subtraction NPR workflows. It does not by itself identify a continuum
    RI/SMOM scheme.
    """

    _require_same_backend(propagator_out, green_function, propagator_in)
    _require_spin_color_field(propagator_out, "propagator_out")
    _require_spin_color_field(green_function, "green_function")
    _require_spin_color_field(propagator_in, "propagator_in")
    if not (
        propagator_out.shape == green_function.shape == propagator_in.shape
    ):
        raise ValueError(
            "amputation requires exact equal shapes; implicit batch broadcasting is forbidden"
        )

    sinv_out = _to_spin_color_matrix(inverse_propagator(propagator_out))
    sinv_in = _to_spin_color_matrix(inverse_propagator(propagator_in))
    green = _to_spin_color_matrix(green_function)
    return _from_spin_color_matrix(sinv_out @ green @ sinv_in)


def slash(momentum: Sequence[float], gammas: Sequence[Any]) -> Any:
    if len(gammas) != 4:
        raise ValueError("gammas must have four components")
    momentum = _finite_four_vector(momentum, "momentum")
    _require_spin_matrices(gammas[0], *gammas[1:])
    result = 0 * gammas[0]
    for component, gamma_mu in zip(momentum, gammas):
        result = result + float(component) * gamma_mu
    return result


def project_quark_field(
    inverse_prop: Any,
    momentum: Sequence[float],
    gammas: Sequence[Any],
) -> Any:
    """Compute the per-momentum RI quark-field projector Zq.

    The `-i Tr[slash(p) S^-1]/(12 p^2)` normalization is the RI propagator
    projection used in arXiv:2310.00814v2 Appendix B.1, Eq. (32).
    """

    _require_spin_color_field(inverse_prop, "inverse_prop")
    if len(gammas) != 4:
        raise ValueError("gammas must contain four Dirac matrices")
    _require_spin_matrices(gammas[0], *gammas[1:])
    xp = _xp(inverse_prop)
    _require_same_backend(inverse_prop, *gammas)
    p = _finite_four_vector(momentum, "momentum")
    p2 = float(p @ p)
    if p2 <= 0:
        raise ValueError("momentum must be nonzero")
    spin_projector = slash(p, gammas)
    projector = xp.kron(spin_projector, xp.eye(3, dtype=spin_projector.dtype))
    matrix = _to_spin_color_matrix(inverse_prop)
    return (-1j / (12.0 * p2)) * xp.trace(projector @ matrix, axis1=-2, axis2=-1)


def _scalar_isclose_zero(value: Any, xp) -> bool:
    """Check one backend scalar, making a CuPy synchronization explicit."""

    close = xp.isclose(value, 0.0)
    if xp is np:
        return bool(np.asarray(close).item())
    # This transfers exactly one Boolean scalar to host. It is an intentional
    # validation synchronization, not a hidden volume-array fallback.
    return bool(np.asarray(xp.asnumpy(close)).item())


def project_vertex(vertex: Any, projector: Any, tree_vertex: Any) -> Any:
    """Project a vertex with automatic tree-level normalization."""

    _require_spin_color_field(vertex, "vertex")
    _require_spin_matrices(projector, tree_vertex)
    xp = _xp(vertex)
    _require_same_backend(vertex, projector, tree_vertex)
    color_identity = xp.eye(3, dtype=vertex.dtype)
    pmat = xp.kron(projector, color_identity)
    tmat = xp.kron(tree_vertex, color_identity)
    vmat = _to_spin_color_matrix(vertex)
    denominator = xp.trace(pmat.conj().T @ tmat)
    if _scalar_isclose_zero(denominator, xp):
        raise ValueError("projector has zero tree-level normalization")
    return xp.trace(pmat.conj().T @ vmat, axis1=-2, axis2=-1) / denominator


def project_multiplet(
    vertices: Sequence[Any],
    projectors: Sequence[Any],
    tree_vertices: Sequence[Any],
) -> Any:
    """Average tree-normalized components of V, A, or T."""

    if not (len(vertices) == len(projectors) == len(tree_vertices)):
        raise ValueError("vertex, projector, and tree multiplets must align")
    if not vertices:
        raise ValueError("multiplets must be nonempty")
    for vertex in vertices:
        _require_spin_color_field(vertex, "multiplet vertex")
    _require_spin_matrices(projectors[0], *projectors[1:], *tree_vertices)
    if any(vertex.shape != vertices[0].shape for vertex in vertices[1:]):
        raise ValueError("all multiplet vertices must have one exact shape")
    xp = _xp(vertices[0])
    _require_same_backend(
        vertices[0], *vertices[1:], *projectors, *tree_vertices
    )
    color_identity = xp.eye(3, dtype=vertices[0].dtype)
    numerator = 0.0
    denominator = 0.0
    for vertex, projector, tree in zip(vertices, projectors, tree_vertices):
        pmat = xp.kron(projector, color_identity)
        tmat = xp.kron(tree, color_identity)
        numerator = numerator + xp.trace(
            pmat.conj().T @ _to_spin_color_matrix(vertex), axis1=-2, axis2=-1
        )
        denominator = denominator + xp.trace(pmat.conj().T @ tmat)
    if _scalar_isclose_zero(denominator, xp):
        raise ValueError("multiplet projector has zero tree-level normalization")
    return numerator / denominator


def _longitudinal_projectors(
    q: Sequence[float], gammas: Sequence[Any], gamma5: Any, axial: bool
) -> list[Any]:
    """Build q_mu slash(q) projectors for a nonexceptional RI family.

    The axial order follows the contracted tree vertex
    ``qslash @ gamma5``. For every omega this function supplies only the stated
    algebraic qslash family; it does not establish a continuum matching scheme
    or literature result.
    """

    qslash = slash(q, gammas)
    base = qslash @ gamma5 if axial else qslash
    return [float(component) * base for component in q]


def compute_ri_constants(
    kinematics: RIKinematics,
    propagator_inverse_in: Any,
    propagator_inverse_out: Any,
    vertices: Mapping[str, Any],
    gammas: Sequence[Any],
    gamma5: Any,
    *,
    zq_definition: str = "propagator",
) -> RIConstants:
    """Compute raw per-momentum Zq, ZA, ZV, ZS, ZP, and ZT.

    `vertices` contains amputated `V` and `A` four-tuples, one `S`, one `P`,
    and a `T` six-tuple ordered by mu<nu. The propagator projections can use
    the incoming leg, outgoing leg, arithmetic mean, or principal-branch
    geometric mean. Complex statistical noise can cross the principal
    square-root branch cut. The backend principal square root is used exactly;
    this function never takes a real part, absolute value, or a different
    branch. The legacy `propagator` label aliases `arithmetic_mean`; no
    vector-vertex RI/SMOM definition is mixed in.
    """

    if not isinstance(kinematics, RIKinematics):
        raise TypeError("kinematics must be an RIKinematics record")
    checked_kinematics = validate_kinematics(
        kinematics.p_in,
        kinematics.p_out,
        kinematics.omega,
        projector_scheme=kinematics.projector_scheme,
        momentum_definition=kinematics.momentum_definition,
    )
    supplied_q = _finite_four_vector(kinematics.q, "kinematics.q")
    if not np.allclose(supplied_q, checked_kinematics.q):
        raise ValueError("kinematics.q must equal p_out-p_in")
    supplied_mu2 = _finite_real_scalar(
        kinematics.mu2, "kinematics.mu2", minimum=0.0
    )
    if not np.isclose(supplied_mu2, checked_kinematics.mu2):
        raise ValueError("kinematics.mu2 is inconsistent with the external legs")
    if kinematics.scheme != checked_kinematics.scheme:
        raise ValueError("kinematics.scheme is inconsistent with omega/projector_scheme")
    kinematics = checked_kinematics

    valid_zq_definitions = {
        "propagator",
        "incoming",
        "outgoing",
        "arithmetic_mean",
        "geometric_mean",
    }
    if zq_definition not in valid_zq_definitions:
        raise ValueError(
            "zq_definition must be incoming, outgoing, arithmetic_mean, "
            "geometric_mean, or the deprecated propagator alias"
        )
    required = {"V", "A", "S", "P", "T"}
    if set(vertices) != required:
        raise ValueError(f"vertices must contain exactly {sorted(required)}")
    if len(vertices["V"]) != 4 or len(vertices["A"]) != 4 or len(vertices["T"]) != 6:
        raise ValueError("V/A require four components and T requires six")
    if len(gammas) != 4:
        raise ValueError("gammas must contain four Dirac matrices")
    _require_spin_matrices(gammas[0], *gammas[1:], gamma5)
    _require_spin_color_field(propagator_inverse_in, "propagator_inverse_in")
    _require_spin_color_field(propagator_inverse_out, "propagator_inverse_out")
    if propagator_inverse_in.shape != propagator_inverse_out.shape:
        raise ValueError("incoming and outgoing inverse propagators must have one shape")
    xp = _xp(propagator_inverse_in)
    all_vertices = [
        *vertices["V"],
        *vertices["A"],
        vertices["S"],
        vertices["P"],
        *vertices["T"],
    ]
    for vertex in all_vertices:
        _require_spin_color_field(vertex, "amputated vertex")
        if vertex.shape != propagator_inverse_in.shape:
            raise ValueError(
                "all inverse propagators and vertices must have exact equal batch/field shapes"
            )
    _require_same_backend(
        propagator_inverse_in,
        propagator_inverse_out,
        gamma5,
        *gammas,
        *all_vertices,
    )
    zq_in = project_quark_field(propagator_inverse_in, kinematics.p_in, gammas)
    zq_out = project_quark_field(propagator_inverse_out, kinematics.p_out, gammas)
    canonical_zq_definition = (
        "arithmetic_mean" if zq_definition == "propagator" else zq_definition
    )
    if canonical_zq_definition == "incoming":
        zq = zq_in
    elif canonical_zq_definition == "outgoing":
        zq = zq_out
    elif canonical_zq_definition == "geometric_mean":
        zq = xp.sqrt(zq_in * zq_out)
    else:
        zq = 0.5 * (zq_in + zq_out)

    vector_tree = list(gammas)
    axial_tree = [gamma_mu @ gamma5 for gamma_mu in gammas]
    if kinematics.projector_scheme == "gamma_mu":
        vector_projectors = vector_tree
        axial_projectors = axial_tree
    else:
        vector_projectors = _longitudinal_projectors(
            kinematics.q, gammas, gamma5, axial=False
        )
        axial_projectors = _longitudinal_projectors(
            kinematics.q, gammas, gamma5, axial=True
        )

    lambda_v = project_multiplet(
        vertices["V"], vector_projectors, vector_tree
    )
    lambda_a = project_multiplet(
        vertices["A"], axial_projectors, axial_tree
    )
    scalar_tree = xp.eye(gamma5.shape[0], dtype=gamma5.dtype)
    lambda_s = project_vertex(vertices["S"], scalar_tree, scalar_tree)
    lambda_p = project_vertex(vertices["P"], gamma5, gamma5)

    tensor_tree = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            tensor_tree.append(0.5 * (gammas[mu] @ gammas[nu] - gammas[nu] @ gammas[mu]))
    lambda_t = project_multiplet(vertices["T"], tensor_tree, tensor_tree)
    projected = {
        "V": lambda_v,
        "A": lambda_a,
        "S": lambda_s,
        "P": lambda_p,
        "T": lambda_t,
    }
    scheme_identity = (
        f"{kinematics.scheme}__Zq_propagator_{canonical_zq_definition}"
        "__custom_unmatched"
    )
    return RIConstants(
        Zq=zq,
        ZA=zq / lambda_a,
        ZV=zq / lambda_v,
        ZS=zq / lambda_s,
        ZP=zq / lambda_p,
        ZT=zq / lambda_t,
        projected_vertices=projected,
        kinematics=kinematics,
        zq_definition=canonical_zq_definition,
        scheme_identity=scheme_identity,
        continuum_matching_status="unmapped_fail_closed",
    )


__all__ = [
    "RIKinematics",
    "RIConstants",
    "validate_kinematics",
    "inverse_propagator",
    "amputate_vertex",
    "slash",
    "project_quark_field",
    "project_vertex",
    "project_multiplet",
    "compute_ri_constants",
]
