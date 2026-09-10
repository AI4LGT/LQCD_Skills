"""Fail-closed, scheduler-free E2E validation helpers for enhanced operators.

The helpers validate a pre-registered comparison and analyse paired data that
has already been produced elsewhere.  They never construct a PyQUDA solve,
read a configuration, submit a job, or convert a CPU/mock check into a claim
about interacting overlap, signal-to-noise, or a plateau.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
REQUIRED_METRICS = ("overlap", "snr", "plateau", "fit_stability")
COST_COMPONENTS = ("setup_s", "solve_s", "contraction_s", "communication_s", "io_s")


class EnhancedE2EContractError(ValueError):
    """Raised when the matched enhanced/conventional E2E contract is incomplete."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnhancedE2EContractError(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnhancedE2EContractError(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise EnhancedE2EContractError(f"{name} must be a non-Boolean integer")
    result = int(value)
    if result < minimum:
        raise EnhancedE2EContractError(f"{name} must be at least {minimum}")
    return result


def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise EnhancedE2EContractError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise EnhancedE2EContractError(f"{name} is outside its allowed finite range")
    return result


def _required_true(data: Mapping[str, Any], key: str, context: str) -> None:
    if data.get(key) is not True:
        raise EnhancedE2EContractError(f"{context}.{key} must be explicitly true")


def _matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise EnhancedE2EContractError(f"{name} must be a finite (4,4) Dirac matrix")
    return matrix


def _propagator(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if (
        result.ndim < 4
        or result.shape[-4:] != (4, 4, 3, 3)
        or not np.isfinite(result).all()
    ):
        raise EnhancedE2EContractError(
            f"{name} must end in (sink_spin,source_spin,sink_color,source_color)=(4,4,3,3)"
        )
    return result


def _artifact_ids(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise EnhancedE2EContractError(f"{name} must be a nonempty sequence")
    return tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))


def validate_euclidean_gamma_basis(
    gamma_t: Any,
    gamma_parallel: Any,
    gamma5: Any,
    *,
    atol: float = 1.0e-12,
) -> dict[str, float]:
    """Check the declared Hermitian Euclidean boost-plane gamma convention."""

    tolerance = _finite(atol, "atol", 0.0)
    time = _matrix(gamma_t, "gamma_t")
    parallel = _matrix(gamma_parallel, "gamma_parallel")
    gamma_five = _matrix(gamma5, "gamma5")
    identity = np.eye(4, dtype=np.complex128)
    residuals = {
        "gamma_t_hermiticity": float(np.max(np.abs(time.conj().T - time))),
        "gamma_parallel_hermiticity": float(np.max(np.abs(parallel.conj().T - parallel))),
        "gamma5_hermiticity": float(np.max(np.abs(gamma_five.conj().T - gamma_five))),
        "gamma_t_square": float(np.max(np.abs(time @ time - identity))),
        "gamma_parallel_square": float(np.max(np.abs(parallel @ parallel - identity))),
        "gamma5_square": float(np.max(np.abs(gamma_five @ gamma_five - identity))),
        "boost_plane_anticommutator": float(np.max(np.abs(time @ parallel + parallel @ time))),
        "gamma5_time_anticommutator": float(np.max(np.abs(gamma_five @ time + time @ gamma_five))),
        "gamma5_parallel_anticommutator": float(np.max(np.abs(gamma_five @ parallel + parallel @ gamma_five))),
    }
    if max(residuals.values()) > tolerance:
        raise EnhancedE2EContractError("declared matrices fail the Euclidean gamma-basis checks")
    return residuals


def transform_dirac_matrix(matrix: Any, basis_transform: Any) -> np.ndarray:
    """Apply ``S M S†`` to a one-in/one-out Dirac matrix."""

    source = _matrix(matrix, "matrix")
    transform = _matrix(basis_transform, "basis_transform")
    _validate_unitary(transform)
    return transform @ source @ transform.conj().T


def transform_diquark_kernel(matrix: Any, basis_transform: Any) -> np.ndarray:
    """Apply the stated two-sink-spin transformation ``S D S^T``."""

    source = _matrix(matrix, "diquark_kernel")
    transform = _matrix(basis_transform, "basis_transform")
    _validate_unitary(transform)
    return transform @ source @ transform.T


def validate_diquark_source_conjugation(
    diquark_sink: Any, gamma_t: Any, diquark_source_bar: Any, *, atol: float = 1.0e-12
) -> float:
    """Check the source-spin ``gamma_t D.conj() gamma_t`` convention.

    This is deliberately distinct from a Hermitian adjoint and is checked in
    whichever basis the caller supplies.  Basis covariance still requires the
    driver to transform all source and sink spin indices consistently.
    """

    tolerance = _finite(atol, "atol", 0.0)
    sink = _matrix(diquark_sink, "diquark_sink")
    time = _matrix(gamma_t, "gamma_t")
    source = _matrix(diquark_source_bar, "diquark_source_bar")
    residual = float(np.max(np.abs(source - time @ sink.conj() @ time)))
    if residual > tolerance:
        raise EnhancedE2EContractError(
            "diquark source must use gamma_t @ diquark_sink.conj() @ gamma_t"
        )
    return residual


def transform_propagator(propagator: Any, basis_transform: Any) -> np.ndarray:
    """Transform trailing propagator spin axes ``(sink_spin, source_spin)``."""

    values = _propagator(propagator, "propagator")
    transform = _matrix(basis_transform, "basis_transform")
    _validate_unitary(transform)
    return np.einsum("ik,...klab,lj->...ijab", transform, values, transform.conj().T, optimize=True)


def _validate_unitary(transform: np.ndarray, *, atol: float = 1.0e-12) -> float:
    residual = float(np.max(np.abs(transform.conj().T @ transform - np.eye(4))))
    if residual > atol:
        raise EnhancedE2EContractError("basis_transform must be unitary")
    return residual


def validate_basis_transform(
    *,
    basis_transform: Any,
    gamma_matrices: Mapping[str, Any],
    transformed_gamma_matrices: Mapping[str, Any],
    propagators: Mapping[str, Any],
    transformed_propagators: Mapping[str, Any],
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Independently check gamma and interacting-propagator basis transforms.

    The diquark source is intentionally not guessed from an arbitrary basis
    change.  A production driver must separately retain its source-kernel
    convention and verify ``gamma_t D.conj() gamma_t`` in that basis.
    """

    tolerance = _finite(atol, "atol", 0.0)
    transform = _matrix(basis_transform, "basis_transform")
    unitarity = _validate_unitary(transform, atol=tolerance)
    original_gammas = _mapping(gamma_matrices, "gamma_matrices")
    moved_gammas = _mapping(transformed_gamma_matrices, "transformed_gamma_matrices")
    original_props = _mapping(propagators, "propagators")
    moved_props = _mapping(transformed_propagators, "transformed_propagators")
    if set(original_gammas) != set(moved_gammas) or not original_gammas:
        raise EnhancedE2EContractError("gamma transform inputs must have the same nonempty keys")
    if set(original_props) != set(moved_props) or not original_props:
        raise EnhancedE2EContractError("propagator transform inputs must have the same nonempty keys")
    gamma_residuals = {
        name: float(np.max(np.abs(transform_dirac_matrix(value, transform) - _matrix(moved_gammas[name], f"transformed_gamma_matrices.{name}"))))
        for name, value in original_gammas.items()
    }
    prop_residuals = {
        name: float(
            np.max(
                np.abs(
                    transform_propagator(value, transform)
                    - _propagator(
                        moved_props[name], f"transformed_propagators.{name}"
                    )
                )
            )
        )
        for name, value in original_props.items()
    }
    if max((*gamma_residuals.values(), *prop_residuals.values())) > tolerance:
        raise EnhancedE2EContractError("gamma or propagator basis transform does not match the declared S")
    return {
        "unitarity_residual": unitarity,
        "gamma_residuals": gamma_residuals,
        "propagator_residuals": prop_residuals,
        "source_sink_note": "source and sink kernels require their declared index-specific transform checks",
    }


def _validate_operator_endpoint(value: Any, name: str) -> tuple[float, float, float]:
    endpoint = _mapping(value, name)
    for key in (
        "operator_id",
        "gamma_basis_id",
        "smearing_id",
        "artifact_id",
        "time_slice",
        "boundary_condition",
    ):
        _string(endpoint.get(key), f"{name}.{key}")
    momentum = endpoint.get("momentum")
    if isinstance(momentum, (str, bytes)) or not isinstance(momentum, Sequence) or len(momentum) != 3:
        raise EnhancedE2EContractError(f"{name}.momentum must have (px,py,pz)")
    values = tuple(_finite(component, f"{name}.momentum[{index}]") for index, component in enumerate(momentum))
    return values


def _validate_resources(resources: Mapping[str, Any]) -> None:
    conventional = _mapping(resources.get("conventional"), "resources.conventional")
    enhanced = _mapping(resources.get("enhanced"), "resources.enhanced")
    for label, row in (("conventional", conventional), ("enhanced", enhanced)):
        _string(row.get("precision"), f"resources.{label}.precision")
        _string(row.get("solver_name"), f"resources.{label}.solver_name")
        _finite(row.get("true_residual_target"), f"resources.{label}.true_residual_target", 0.0)
        _integer(row.get("source_count"), f"resources.{label}.source_count", 1)
        _string(row.get("solve_budget_id"), f"resources.{label}.solve_budget_id")
    for key in ("precision", "solver_name", "true_residual_target", "source_count", "solve_budget_id"):
        if conventional[key] != enhanced[key]:
            raise EnhancedE2EContractError(f"resources must use equal {key} for the matched comparison")
    _required_true(resources, "shared_propagator_cost", "resources")
    _required_true(resources, "equal_solve_budget", "resources")


def validate_enhanced_e2e_preflight(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen, unsubmitted interacting-operator comparison plan."""

    data = _mapping(contract, "contract")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise EnhancedE2EContractError(f"schema_version must be {SCHEMA_VERSION}")
    _string(data.get("experiment_id"), "experiment_id")

    basis = _mapping(data.get("gamma_basis"), "gamma_basis")
    if basis.get("convention") != "Euclidean Hermitian":
        raise EnhancedE2EContractError("gamma_basis.convention must be 'Euclidean Hermitian'")
    _string(basis.get("boost_direction"), "gamma_basis.boost_direction")
    _string(basis.get("basis_id"), "gamma_basis.basis_id")
    for key in ("gamma_algebra_checked", "basis_transform_checked", "source_sink_transform_checked"):
        _required_true(basis, key, "gamma_basis")

    comparison = _mapping(data.get("comparison"), "comparison")
    _required_true(comparison, "interacting_propagators", "comparison")
    _required_true(comparison, "shared_propagators", "comparison")
    _artifact_ids(comparison.get("propagator_ids"), "comparison.propagator_ids")
    _artifact_ids(comparison.get("configuration_ids"), "comparison.configuration_ids")
    _artifact_ids(comparison.get("source_ids"), "comparison.source_ids")

    operators = _mapping(data.get("operators"), "operators")
    conventional = _mapping(operators.get("conventional"), "operators.conventional")
    enhanced = _mapping(operators.get("enhanced"), "operators.enhanced")
    conventional_source = _validate_operator_endpoint(conventional.get("source"), "operators.conventional.source")
    conventional_sink = _validate_operator_endpoint(conventional.get("sink"), "operators.conventional.sink")
    enhanced_source = _validate_operator_endpoint(enhanced.get("source"), "operators.enhanced.source")
    enhanced_sink = _validate_operator_endpoint(enhanced.get("sink"), "operators.enhanced.sink")
    if conventional_source != enhanced_source or conventional_sink != enhanced_sink:
        raise EnhancedE2EContractError("conventional and enhanced operators must use identical source/sink momentum")
    for label, left, right in (
        ("source", conventional["source"], enhanced["source"]),
        ("sink", conventional["sink"], enhanced["sink"]),
    ):
        for key in ("gamma_basis_id", "smearing_id", "time_slice", "boundary_condition"):
            if left[key] != right[key]:
                raise EnhancedE2EContractError(
                    f"conventional and enhanced {label} controls must share {key}"
                )
    for endpoint in (conventional, enhanced):
        _required_true(endpoint, "source_sink_convention_frozen", "operator")
    _validate_resources(_mapping(data.get("resources"), "resources"))

    analysis = _mapping(data.get("analysis"), "analysis")
    if analysis.get("correlated_resampling") not in {"jackknife", "bootstrap"}:
        raise EnhancedE2EContractError("analysis.correlated_resampling must be jackknife or bootstrap")
    _string(analysis.get("resample_unit"), "analysis.resample_unit")
    _string(analysis.get("fit_window_rule"), "analysis.fit_window_rule")
    _required_true(analysis, "frozen_before_run", "analysis")
    _required_true(analysis, "no_improvement_is_valid", "analysis")
    metric_defs = _mapping(analysis.get("metrics"), "analysis.metrics")
    if set(metric_defs) != set(REQUIRED_METRICS):
        raise EnhancedE2EContractError("analysis.metrics must define overlap, snr, plateau, and fit_stability")
    for name in REQUIRED_METRICS:
        definition = _mapping(metric_defs[name], f"analysis.metrics.{name}")
        _string(definition.get("definition"), f"analysis.metrics.{name}.definition")
        if definition.get("direction") not in {"higher", "lower"}:
            raise EnhancedE2EContractError(f"analysis.metrics.{name}.direction must be higher or lower")

    io = _mapping(data.get("io"), "io")
    _artifact_ids(io.get("input_artifact_ids"), "io.input_artifact_ids")
    _string(io.get("output_schema"), "io.output_schema")
    cost_plan = _mapping(data.get("cost_plan"), "cost_plan")
    if cost_plan.get("units") != "seconds" or tuple(cost_plan.get("components", ())) != COST_COMPONENTS:
        raise EnhancedE2EContractError("cost_plan must account for setup, solve, contraction, communication and I/O in seconds")
    execution = _mapping(data.get("execution"), "execution")
    if execution.get("state") != "NOT_RUN" or _integer(
        execution.get("jobs_submitted"), "execution.jobs_submitted", 0
    ) != 0:
        raise EnhancedE2EContractError("preflight accepts only an unsubmitted comparison")
    if execution.get("submission_policy") != "external_human_authorization_required":
        raise EnhancedE2EContractError("execution must require external human authorisation")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": data["experiment_id"],
        "status": "READY_FOR_EXPLICIT_HUMAN_AUTHORIZATION",
        "jobs_submitted": 0,
        "no_submission_performed": True,
        "required_metrics": list(REQUIRED_METRICS),
        "evidence_limit": "contract-only; no interacting overlap, SNR, plateau, fit, QUDA, or MPI result is established",
    }


def validate_cost_accounting(cost: Mapping[str, Any], *, atol: float = 1.0e-9) -> dict[str, float]:
    """Validate one operator's complete timing record."""

    row = _mapping(cost, "cost")
    if row.get("units") != "seconds":
        raise EnhancedE2EContractError("cost.units must be 'seconds'")
    components = {name: _finite(row.get(name), f"cost.{name}", 0.0) for name in COST_COMPONENTS}
    total = _finite(row.get("total_s"), "cost.total_s", 0.0)
    tolerance = _finite(atol, "atol", 0.0)
    if not np.isclose(total, sum(components.values()), rtol=0.0, atol=tolerance):
        raise EnhancedE2EContractError("cost.total_s must equal all reported cost components")
    return {**components, "total_s": total}


def validate_matched_cost_accounting(
    conventional: Mapping[str, Any], enhanced: Mapping[str, Any], *, atol: float = 1.0e-9
) -> dict[str, Any]:
    """Verify equal shared setup/solve cost while retaining operator costs.

    The conventional and enhanced contractions can have different post-solve
    costs, so those are reported rather than forced equal.  Equal setup and
    solve records are the required comparable-cost condition when propagators
    are shared.
    """

    tolerance = _finite(atol, "atol", 0.0)
    base = validate_cost_accounting(conventional, atol=tolerance)
    candidate = validate_cost_accounting(enhanced, atol=tolerance)
    for component in ("setup_s", "solve_s"):
        if not np.isclose(base[component], candidate[component], rtol=0.0, atol=tolerance):
            raise EnhancedE2EContractError(
                f"matched comparison requires equal shared {component} records"
            )
    return {
        "conventional": base,
        "enhanced": candidate,
        "shared_setup_s": base["setup_s"],
        "shared_solve_s": base["solve_s"],
        "operator_cost_difference_s": float(candidate["total_s"] - base["total_s"]),
    }


def summarize_correlated_metric(
    conventional: Sequence[float],
    enhanced: Sequence[float],
    *,
    direction: str,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Analyse a paired metric without requiring improvement as an outcome."""

    if direction not in {"higher", "lower"}:
        raise EnhancedE2EContractError("direction must be higher or lower")
    level = _finite(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise EnhancedE2EContractError("confidence must be strictly between zero and one")
    if isinstance(conventional, (str, bytes)) or isinstance(enhanced, (str, bytes)):
        raise EnhancedE2EContractError("metric samples must be numerical sequences")
    try:
        base = np.asarray(conventional, dtype=float)
        candidate = np.asarray(enhanced, dtype=float)
    except (TypeError, ValueError) as error:
        raise EnhancedE2EContractError("metric samples must be numerical") from error
    if base.ndim != 1 or candidate.ndim != 1 or base.size != candidate.size or base.size < 2:
        raise EnhancedE2EContractError("paired metric samples must be one-dimensional, equally sized, and have at least two entries")
    if not np.isfinite(base).all() or not np.isfinite(candidate).all():
        raise EnhancedE2EContractError("metric samples must be finite")
    signed_difference = candidate - base if direction == "higher" else base - candidate
    mean_difference = float(np.mean(signed_difference))
    standard_error = float(np.std(signed_difference, ddof=1) / np.sqrt(base.size))
    critical = NormalDist().inv_cdf(0.5 + level / 2.0)
    interval = (mean_difference - critical * standard_error, mean_difference + critical * standard_error)
    if interval[0] > 0.0:
        classification = "SUPPORTED_IMPROVEMENT"
    elif interval[1] < 0.0:
        classification = "SUPPORTED_DEGRADATION"
    else:
        classification = "NO_CONFIRMED_IMPROVEMENT"
    correlation = None
    if np.std(base) > 0.0 and np.std(candidate) > 0.0:
        correlation = float(np.corrcoef(base, candidate)[0, 1])
    return {
        "sample_count": int(base.size),
        "direction": direction,
        "conventional_mean": float(np.mean(base)),
        "enhanced_mean": float(np.mean(candidate)),
        "paired_improvement_mean": mean_difference,
        "paired_improvement_standard_error": standard_error,
        "confidence": level,
        "paired_improvement_confidence_interval": [float(interval[0]), float(interval[1])],
        "conventional_enhanced_correlation": correlation,
        "classification": classification,
        "physical_claim": "none; this is a paired statistical summary pending target-runtime provenance",
    }


def summarize_enhanced_metrics(
    metrics: Mapping[str, Mapping[str, Any]], *, confidence: float = 0.95
) -> dict[str, Any]:
    """Summarise all pre-registered paired metrics with a no-improvement path."""

    rows = _mapping(metrics, "metrics")
    if set(rows) != set(REQUIRED_METRICS):
        raise EnhancedE2EContractError("metrics must contain exactly the four pre-registered metric names")
    result = {
        name: summarize_correlated_metric(
            _mapping(rows[name], f"metrics.{name}").get("conventional"),
            _mapping(rows[name], f"metrics.{name}").get("enhanced"),
            direction=_mapping(rows[name], f"metrics.{name}").get("direction"),
            confidence=confidence,
        )
        for name in REQUIRED_METRICS
    }
    classifications = {name: row["classification"] for name, row in result.items()}
    if all(value == "SUPPORTED_IMPROVEMENT" for value in classifications.values()):
        overall = "ALL_PREREGISTERED_METRICS_SUPPORT_IMPROVEMENT"
    elif any(value == "SUPPORTED_DEGRADATION" for value in classifications.values()):
        overall = "MIXED_OR_DEGRADED"
    else:
        overall = "NO_CONFIRMED_IMPROVEMENT"
    return {
        "metrics": result,
        "overall": overall,
        "no_improvement_is_valid": True,
        "evidence_limit": "paired statistics only; bind to shared interacting propagator provenance before a physics claim",
    }
