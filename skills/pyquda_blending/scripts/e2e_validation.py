"""Fail-closed, scheduler-free E2E validation helpers for blending.

This module deliberately has no PyQUDA, MPI, Slurm, filesystem, or scheduler
dependency.  It validates the scientific contract which must be frozen before
an externally authorised cluster run, and it analyses already-collected seed
measurements.  A successful result is *not* evidence that a QUDA solve, an MPI
collective, or an interacting correlator has run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from statistics import NormalDist
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
COST_COMPONENTS = (
    "setup_s",
    "basis_s",
    "noise_s",
    "projection_s",
    "solve_s",
    "contraction_s",
    "communication_s",
    "io_s",
)


class BlendingE2EContractError(ValueError):
    """Raised when E2E evidence is incomplete or internally inconsistent."""


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PERAMBULATOR_AXIS_ORDER = "(t,sink_spin,source_spin,sink_mode,source_mode)"
_BASIS_LAYOUT = "(D,N) color-spatial columns"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BlendingE2EContractError(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlendingE2EContractError(f"{name} must be a nonempty string")
    return value


def _checksum(value: Any, name: str) -> str:
    result = _string(value, name)
    if _SHA256_RE.fullmatch(result) is None:
        raise BlendingE2EContractError(
            f"{name} must be a lowercase sha256:<64-hex-character> digest"
        )
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise BlendingE2EContractError(f"{name} must be a non-Boolean integer")
    result = int(value)
    if result < minimum:
        raise BlendingE2EContractError(f"{name} must be at least {minimum}")
    return result


def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise BlendingE2EContractError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "finite" if minimum is None else f"finite and at least {minimum}"
        raise BlendingE2EContractError(f"{name} must be {qualifier}")
    return result


def _required_true(data: Mapping[str, Any], key: str, context: str) -> None:
    if data.get(key) is not True:
        raise BlendingE2EContractError(f"{context}.{key} must be explicitly true")


def _artifact_rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise BlendingE2EContractError(f"{name} must be a nonempty artifact sequence")
    rows = tuple(_mapping(row, f"{name}[{index}]") for index, row in enumerate(value))
    artifact_ids: set[str] = set()
    for index, row in enumerate(rows):
        _string(row.get("artifact_id"), f"{name}[{index}].artifact_id")
        _checksum(row.get("checksum"), f"{name}[{index}].checksum")
        artifact_id = row["artifact_id"]
        if artifact_id in artifact_ids:
            raise BlendingE2EContractError(f"{name} contains duplicate artifact_id {artifact_id!r}")
        artifact_ids.add(artifact_id)
    return rows


def _string_sequence(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BlendingE2EContractError(f"{name} must be a sequence of strings")
    if not allow_empty and not value:
        raise BlendingE2EContractError(f"{name} must be nonempty")
    result = tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise BlendingE2EContractError(f"{name} must not contain duplicates")
    return result


def _artifact_index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows):
        artifact_id = _string(row.get("artifact_id"), f"{name}[{row_number}].artifact_id")
        if artifact_id in index:
            raise BlendingE2EContractError(f"{name} contains duplicate artifact_id {artifact_id!r}")
        index[artifact_id] = row
    return index


def z4_noise(dimension: int, n_st: int, seed: int) -> np.ndarray:
    """Generate an explicit unit-modulus Z4 noise matrix with one RNG seed.

    This helper provides a deterministic CPU contract/oracle.  Production
    drivers must record their actual RNG implementation, seed derivation and
    device placement instead of treating this helper as a GPU noise generator.
    """

    dimension = _integer(dimension, "dimension", 1)
    n_st = _integer(n_st, "n_st", 0)
    seed = _integer(seed, "seed", 0)
    phases = np.random.default_rng(seed).integers(0, 4, size=(dimension, n_st))
    return np.exp(0.5j * np.pi * phases)


def complement_noise(
    low_modes: Any,
    n_st: int,
    seed: int,
    *,
    atol: float = 1.0e-12,
) -> np.ndarray:
    """Project Z4 noise into the low-mode complement and orthonormalise it.

    ``low_modes`` has column layout ``(D, N_ev)`` with ``D=3V3``.  The QR
    stage makes one realisation mutually orthonormal; it does not by itself
    prove isotropy or independence across production configurations.
    """

    n_st = _integer(n_st, "n_st", 0)
    seed = _integer(seed, "seed", 0)
    matrix = np.asarray(low_modes, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise BlendingE2EContractError("low_modes must have shape (D, N_ev)")
    if not np.isfinite(matrix).all():
        raise BlendingE2EContractError("low_modes must be finite")
    dimension, n_ev = matrix.shape
    if n_st > dimension - n_ev:
        raise BlendingE2EContractError("n_st exceeds the complement dimension")
    validate_complement_basis(matrix, np.empty((dimension, 0)), atol=atol)
    if n_st == 0:
        return np.empty((dimension, 0), dtype=np.complex128)
    raw_noise = z4_noise(dimension, n_st, seed)
    projected = raw_noise - matrix @ (matrix.conj().T @ raw_noise)
    orthonormal, upper = np.linalg.qr(projected, mode="reduced")
    if np.min(np.abs(np.diag(upper))) <= atol:
        raise BlendingE2EContractError("projected noise is rank deficient")
    result = orthonormal[:, :n_st]
    validate_complement_basis(matrix, result, atol=atol)
    return result


def validate_complement_basis(
    low_modes: Any, stochastic_modes: Any, *, atol: float = 1.0e-12
) -> dict[str, float]:
    """Numerically verify low-space and complement orthogonality conditions."""

    tolerance = _finite(atol, "atol", 0.0)
    low = np.asarray(low_modes, dtype=np.complex128)
    stochastic = np.asarray(stochastic_modes, dtype=np.complex128)
    if low.ndim != 2 or stochastic.ndim != 2 or low.shape[0] != stochastic.shape[0]:
        raise BlendingE2EContractError(
            "low_modes and stochastic_modes must be two-dimensional with common D"
        )
    if not np.isfinite(low).all() or not np.isfinite(stochastic).all():
        raise BlendingE2EContractError("basis vectors must be finite")
    low_residual = float(np.max(np.abs(low.conj().T @ low - np.eye(low.shape[1])))) if low.shape[1] else 0.0
    complement_residual = float(np.max(np.abs(low.conj().T @ stochastic))) if low.shape[1] and stochastic.shape[1] else 0.0
    stochastic_residual = (
        float(
            np.max(
                np.abs(
                    stochastic.conj().T @ stochastic - np.eye(stochastic.shape[1])
                )
            )
        )
        if stochastic.shape[1]
        else 0.0
    )
    if max(low_residual, complement_residual, stochastic_residual) > tolerance:
        raise BlendingE2EContractError(
            "basis fails low-mode orthonormality or complement orthogonality"
        )
    return {
        "low_mode_orthonormal_residual": low_residual,
        "low_complement_residual": complement_residual,
        "stochastic_orthonormal_residual": stochastic_residual,
    }


def complete_dilution_projectors(dimension: int) -> tuple[np.ndarray, ...]:
    """Return rank-one complete dilution projectors in the stated subspace."""

    dimension = _integer(dimension, "dimension", 1)
    return tuple(np.diag(np.eye(dimension, dtype=np.complex128)[index]) for index in range(dimension))


def validate_dilution_projectors(
    projectors: Sequence[Any], dimension: int, *, require_complete: bool = True, atol: float = 1.0e-12
) -> dict[str, float | bool | int]:
    """Check orthogonal/idempotent dilution projectors and optional completeness."""

    dimension = _integer(dimension, "dimension", 1)
    if not isinstance(require_complete, (bool, np.bool_)):
        raise BlendingE2EContractError("require_complete must be Boolean")
    tolerance = _finite(atol, "atol", 0.0)
    if isinstance(projectors, (str, bytes)) or not isinstance(projectors, Sequence) or not projectors:
        raise BlendingE2EContractError("projectors must be a nonempty sequence")
    arrays = tuple(np.asarray(projector, dtype=np.complex128) for projector in projectors)
    if any(array.shape != (dimension, dimension) for array in arrays):
        raise BlendingE2EContractError("every dilution projector must have shape (dimension, dimension)")
    if not all(np.isfinite(array).all() for array in arrays):
        raise BlendingE2EContractError("dilution projectors must be finite")
    identity = np.eye(dimension, dtype=np.complex128)
    idempotence = max(float(np.max(np.abs(projector @ projector - projector))) for projector in arrays)
    hermiticity = max(float(np.max(np.abs(projector.conj().T - projector))) for projector in arrays)
    pairwise = max(
        (float(np.max(np.abs(left @ right))) for left_index, left in enumerate(arrays) for right in arrays[left_index + 1 :]),
        default=0.0,
    )
    completeness = float(np.max(np.abs(sum(arrays) - identity)))
    if max(idempotence, hermiticity, pairwise) > tolerance:
        raise BlendingE2EContractError("dilution projectors are not orthogonal projectors")
    if require_complete and completeness > tolerance:
        raise BlendingE2EContractError("dilution projectors do not resolve the stated subspace")
    return {
        "projector_count": len(arrays),
        "idempotence_residual": idempotence,
        "hermiticity_residual": hermiticity,
        "pairwise_orthogonality_residual": pairwise,
        "completeness_residual": completeness,
        "complete": bool(completeness <= tolerance),
    }


def validate_cost_accounting(cost: Mapping[str, Any], *, atol: float = 1.0e-9) -> dict[str, float]:
    """Require explicit, nonnegative component costs and an auditable total."""

    row = _mapping(cost, "cost")
    if row.get("units") != "seconds":
        raise BlendingE2EContractError("cost.units must be 'seconds'")
    values = {name: _finite(row.get(name), f"cost.{name}", 0.0) for name in COST_COMPONENTS}
    total = _finite(row.get("total_s"), "cost.total_s", 0.0)
    tolerance = _finite(atol, "atol", 0.0)
    component_total = float(sum(values.values()))
    if not np.isclose(total, component_total, rtol=0.0, atol=tolerance):
        raise BlendingE2EContractError(
            "cost.total_s must equal setup+basis+noise+projection+solve+contraction+communication+io"
        )
    return {**values, "total_s": total}


def _validate_seed_contract(noise: Mapping[str, Any]) -> int:
    _string(noise.get("distribution"), "noise.distribution")
    _string(noise.get("generator"), "noise.generator")
    _string(noise.get("normalization"), "noise.normalization")
    _required_true(noise, "independent_seeds", "noise")
    scope = noise.get("seed_scope")
    if isinstance(scope, (str, bytes)) or not isinstance(scope, Sequence):
        raise BlendingE2EContractError("noise.seed_scope must identify every seed coordinate")
    required_scope = {"configuration_id", "time_slice", "noise_label"}
    scope_names = {_string(value, f"noise.seed_scope[{index}]") for index, value in enumerate(scope)}
    if not required_scope <= scope_names:
        raise BlendingE2EContractError("noise.seed_scope must include configuration_id, time_slice, noise_label")
    rows = noise.get("seed_records")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) < 2:
        raise BlendingE2EContractError("noise.seed_records must contain at least two independent seeds")
    seeds: list[int] = []
    coordinates: set[tuple[str, str, str]] = set()
    for index, value in enumerate(rows):
        record = _mapping(value, f"noise.seed_records[{index}]")
        seeds.append(_integer(record.get("seed"), f"noise.seed_records[{index}].seed", 0))
        coordinate = tuple(
            _string(record.get(field), f"noise.seed_records[{index}].{field}")
            for field in ("configuration_id", "time_slice", "noise_label")
        )
        if coordinate in coordinates:
            raise BlendingE2EContractError("noise seed records reuse a configuration/time/noise label")
        coordinates.add(coordinate)
    if len(set(seeds)) != len(seeds):
        raise BlendingE2EContractError("noise seeds must be distinct for independent samples")
    return len(rows)


def _validate_laph_perambulator_reference(
    data: Mapping[str, Any],
    input_artifacts: Sequence[Mapping[str, Any]],
    *,
    n_ev: int,
    n_st: int,
    observable_id: str,
) -> dict[str, Any]:
    """Validate the producer metadata needed by a LapH/BLD adapter.

    IDs and checksums are deliberately cross-bound here.  A generic list of
    files is not enough: the LapH basis, projected perambulator and exact
    reference must identify the same configuration and compatible basis, while
    the reference must remain a separately produced artifact.
    """

    artifacts = _artifact_index(input_artifacts, "io.input_artifacts")
    laph = _mapping(data.get("laph"), "laph")
    eigen = _mapping(laph.get("eigenvectors"), "laph.eigenvectors")
    eigen_id = _string(eigen.get("artifact_id"), "laph.eigenvectors.artifact_id")
    eigen_checksum = _checksum(eigen.get("checksum"), "laph.eigenvectors.checksum")
    if eigen_id not in artifacts or artifacts[eigen_id].get("checksum") != eigen_checksum:
        raise BlendingE2EContractError(
            "laph.eigenvectors must be declared in io.input_artifacts with the same checksum"
        )
    eigen_configuration = _string(
        eigen.get("configuration_id"), "laph.eigenvectors.configuration_id"
    )
    gauge_id = _string(eigen.get("gauge_artifact_id"), "laph.eigenvectors.gauge_artifact_id")
    if gauge_id == eigen_id or gauge_id not in artifacts:
        raise BlendingE2EContractError(
            "laph.eigenvectors.gauge_artifact_id must identify a separate input artifact"
        )
    _string(eigen.get("operator_convention"), "laph.eigenvectors.operator_convention")
    if _string(eigen.get("basis_layout"), "laph.eigenvectors.basis_layout") != _BASIS_LAYOUT:
        raise BlendingE2EContractError(
            f"laph.eigenvectors.basis_layout must be {_BASIS_LAYOUT!r}"
        )
    if _integer(eigen.get("n_ev"), "laph.eigenvectors.n_ev", 0) != n_ev:
        raise BlendingE2EContractError("laph.eigenvectors.n_ev must equal mode_counts.n_ev")
    _string(eigen.get("normalization"), "laph.eigenvectors.normalization")
    _finite(eigen.get("max_eigenpair_residual"), "laph.eigenvectors.max_eigenpair_residual", 0.0)
    _finite(eigen.get("orthonormality_residual"), "laph.eigenvectors.orthonormality_residual", 0.0)

    perambulator = _mapping(data.get("perambulator"), "perambulator")
    peram_id = _string(perambulator.get("artifact_id"), "perambulator.artifact_id")
    peram_checksum = _checksum(perambulator.get("checksum"), "perambulator.checksum")
    if peram_id not in artifacts or artifacts[peram_id].get("checksum") != peram_checksum:
        raise BlendingE2EContractError(
            "perambulator must be declared in io.input_artifacts with the same checksum"
        )
    if peram_id == eigen_id or peram_id == gauge_id:
        raise BlendingE2EContractError("perambulator must be a distinct artifact")
    if _string(perambulator.get("configuration_id"), "perambulator.configuration_id") != eigen_configuration:
        raise BlendingE2EContractError("perambulator and LapH basis must use one configuration")
    if _string(
        perambulator.get("laph_eigenvector_artifact_id"),
        "perambulator.laph_eigenvector_artifact_id",
    ) != eigen_id:
        raise BlendingE2EContractError(
            "perambulator.laph_eigenvector_artifact_id must bind to laph.eigenvectors"
        )
    if _string(perambulator.get("axis_order"), "perambulator.axis_order") != _PERAMBULATOR_AXIS_ORDER:
        raise BlendingE2EContractError(
            f"perambulator.axis_order must be {_PERAMBULATOR_AXIS_ORDER!r}"
        )
    if _integer(perambulator.get("n_modes"), "perambulator.n_modes", 1) != n_ev + n_st:
        raise BlendingE2EContractError("perambulator.n_modes must equal n_ev+n_st")
    if _integer(perambulator.get("n_spin"), "perambulator.n_spin", 1) != 4:
        raise BlendingE2EContractError("perambulator.n_spin must be four-component Dirac spin")
    _string(perambulator.get("action_id"), "perambulator.action_id")
    _string(perambulator.get("boundary_conditions"), "perambulator.boundary_conditions")
    if perambulator.get("residual_kind") != "true":
        raise BlendingE2EContractError("perambulator.residual_kind must be 'true'")
    _finite(perambulator.get("max_true_residual"), "perambulator.max_true_residual", 0.0)

    reference = _mapping(data.get("reference"), "reference")
    reference_id = _string(reference.get("artifact_id"), "reference.artifact_id")
    reference_checksum = _checksum(reference.get("checksum"), "reference.checksum")
    if reference_id not in artifacts or artifacts[reference_id].get("checksum") != reference_checksum:
        raise BlendingE2EContractError(
            "reference must be declared in io.input_artifacts with the same checksum"
        )
    if len({eigen_id, peram_id, reference_id}) != 3:
        raise BlendingE2EContractError(
            "LapH, perambulator and exact reference artifacts must be distinct"
        )
    if reference.get("kind") != "exact_all_to_all":
        raise BlendingE2EContractError(
            "BLD adapter requires an exact_all_to_all reference artifact"
        )
    _string(reference.get("implementation_id"), "reference.implementation_id")
    _required_true(reference, "independent_implementation", "reference")
    _string(reference.get("contraction_convention"), "reference.contraction_convention")
    if _string(reference.get("observable_id"), "reference.observable_id") != observable_id:
        raise BlendingE2EContractError("exact reference and BLD contract must identify one observable")
    reference_inputs = _string_sequence(reference.get("input_artifact_ids"), "reference.input_artifact_ids")
    missing_reference_inputs = [artifact_id for artifact_id in reference_inputs if artifact_id not in artifacts]
    if missing_reference_inputs:
        raise BlendingE2EContractError(
            "reference.input_artifact_ids must be declared in io.input_artifacts"
        )
    if peram_id in reference_inputs:
        raise BlendingE2EContractError(
            "exact reference must not use the blended perambulator as its input"
        )
    if _string(reference.get("configuration_id"), "reference.configuration_id") != eigen_configuration:
        raise BlendingE2EContractError("exact reference and LapH basis must use one configuration")

    return {
        "laph_eigenvectors": eigen_id,
        "perambulator": peram_id,
        "exact_reference": reference_id,
        "configuration_id": eigen_configuration,
    }


def validate_blending_e2e_preflight(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen, not-yet-submitted blending E2E contract.

    The return value intentionally retains ``jobs_submitted=0``.  Callers must
    obtain human authorisation and use their own cluster tooling to execute a
    run; this module never builds or invokes a scheduler command.
    """

    data = _mapping(contract, "contract")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise BlendingE2EContractError(f"schema_version must be {SCHEMA_VERSION}")
    _string(data.get("experiment_id"), "experiment_id")
    observable_id = _string(data.get("observable_id"), "observable_id")
    lattice = _mapping(data.get("lattice"), "lattice")
    dimension = _integer(lattice.get("volume_color_dim"), "lattice.volume_color_dim", 1)
    _string(lattice.get("basis_layout"), "lattice.basis_layout")
    counts = _mapping(data.get("mode_counts"), "mode_counts")
    n_ev = _integer(counts.get("n_ev"), "mode_counts.n_ev", 0)
    n_st = _integer(counts.get("n_st"), "mode_counts.n_st", 0)
    if n_ev > dimension or n_st > dimension - n_ev:
        raise BlendingE2EContractError("mode_counts exceed the global color-spatial dimension")
    noise = _mapping(data.get("noise"), "noise")
    seed_count = _validate_seed_contract(noise)

    basis = _mapping(data.get("basis"), "basis")
    if basis.get("complement_projection") != "I-P_low":
        raise BlendingE2EContractError("basis.complement_projection must be 'I-P_low'")
    for key in ("low_modes_orthonormal", "noise_orthogonal_to_low", "noise_mutually_orthonormal"):
        _required_true(basis, key, "basis")

    dilution = _mapping(data.get("dilution"), "dilution")
    _string(dilution.get("scheme"), "dilution.scheme")
    if _integer(dilution.get("subspace_dimension"), "dilution.subspace_dimension", 0) != n_st:
        raise BlendingE2EContractError("dilution.subspace_dimension must equal n_st")
    dilution_labels = _string_sequence(
        dilution.get("labels"), "dilution.labels", allow_empty=(n_st == 0)
    )
    if _integer(dilution.get("label_count"), "dilution.label_count", 0) != len(dilution_labels):
        raise BlendingE2EContractError("dilution.label_count must equal len(dilution.labels)")
    for key in ("projectors_orthogonal", "projectors_complete", "all_diluted_solves_included"):
        _required_true(dilution, key, "dilution")

    reference = _mapping(data.get("reference"), "reference")
    if reference.get("kind") != "exact_all_to_all":
        raise BlendingE2EContractError("reference.kind must be exact_all_to_all")
    for key in ("frozen_before_run", "same_observable"):
        _required_true(reference, key, "reference")
    _string(reference.get("artifact_id"), "reference.artifact_id")
    _string(reference.get("checksum"), "reference.checksum")

    controls = _mapping(data.get("controls"), "controls")
    nst_zero = _mapping(controls.get("n_st_zero"), "controls.n_st_zero")
    if (
        nst_zero.get("expected") != "distillation_weight_one"
        or _integer(nst_zero.get("n_st"), "controls.n_st_zero.n_st", 0) != 0
    ):
        raise BlendingE2EContractError("N_st=0 control must test the unit-weight distillation limit")
    _required_true(nst_zero, "enabled", "controls.n_st_zero")
    complete = _mapping(controls.get("complete_dilution"), "controls.complete_dilution")
    if complete.get("expected") != "exact_with_all_solves":
        raise BlendingE2EContractError("complete-dilution control must require all solves and exact reference")
    _required_true(complete, "enabled", "controls.complete_dilution")

    solver = _mapping(data.get("solver"), "solver")
    for key in ("solver_name", "precision", "solve_strategy"):
        _string(solver.get(key), f"solver.{key}")
    if solver.get("residual_kind") != "true":
        raise BlendingE2EContractError("solver.residual_kind must be 'true'")
    _finite(solver.get("true_residual_target"), "solver.true_residual_target", 0.0)
    _integer(solver.get("rhs_count"), "solver.rhs_count", 1)
    _required_true(solver, "all_diluted_solves_recorded", "solver")

    io = _mapping(data.get("io"), "io")
    input_artifacts = _artifact_rows(io.get("input_artifacts"), "io.input_artifacts")
    _string(io.get("output_schema"), "io.output_schema")
    _string(io.get("output_location_policy"), "io.output_location_policy")
    producer_metadata = _validate_laph_perambulator_reference(
        data, input_artifacts, n_ev=n_ev, n_st=n_st, observable_id=observable_id
    )
    perambulator = _mapping(data["perambulator"], "perambulator")
    perambulator_seed_ids = _string_sequence(
        perambulator.get("seed_ids"), "perambulator.seed_ids"
    )
    declared_seed_ids = {str(record["seed"]) for record in noise["seed_records"]}
    if set(perambulator_seed_ids) != declared_seed_ids:
        raise BlendingE2EContractError(
            "perambulator.seed_ids must match the declared independent noise seeds"
        )
    perambulator_dilution_labels = _string_sequence(
        perambulator.get("dilution_labels"), "perambulator.dilution_labels", allow_empty=(n_st == 0)
    )
    if perambulator_dilution_labels != dilution_labels:
        raise BlendingE2EContractError(
            "perambulator.dilution_labels must match dilution.labels in order"
        )

    cost_plan = _mapping(data.get("cost_plan"), "cost_plan")
    if cost_plan.get("units") != "seconds" or tuple(cost_plan.get("components", ())) != COST_COMPONENTS:
        raise BlendingE2EContractError("cost_plan must list every cost component in seconds")

    execution = _mapping(data.get("execution"), "execution")
    if execution.get("state") != "NOT_RUN" or _integer(
        execution.get("jobs_submitted"), "execution.jobs_submitted", 0
    ) != 0:
        raise BlendingE2EContractError("preflight accepts only an unsubmitted E2E contract")
    if execution.get("submission_policy") != "external_human_authorization_required":
        raise BlendingE2EContractError("execution must require external human authorisation")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": data["experiment_id"],
        "status": "READY_FOR_EXPLICIT_HUMAN_AUTHORIZATION",
        "jobs_submitted": 0,
        "no_submission_performed": True,
        "seed_count": seed_count,
        "mode_counts": {"n_ev": n_ev, "n_st": n_st, "volume_color_dim": dimension},
        "producer_metadata": producer_metadata,
        "evidence_limit": "contract-only; no QUDA, MPI, I/O, solve, or physics result is established",
    }


def summarize_blending_estimator(
    seed_values: Sequence[complex],
    reference_value: complex,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Report mean, variance and component-wise normal confidence intervals.

    The samples must come from distinct predeclared seeds.  Compatibility with
    a reference is a finite-sample diagnostic, not a proof of unbiasedness.
    """

    if isinstance(seed_values, (str, bytes)) or not isinstance(seed_values, Sequence) or len(seed_values) < 2:
        raise BlendingE2EContractError("at least two independent seed values are required")
    level = _finite(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise BlendingE2EContractError("confidence must lie strictly between zero and one")
    try:
        values = np.asarray(seed_values, dtype=np.complex128)
        reference = complex(reference_value)
    except (TypeError, ValueError) as error:
        raise BlendingE2EContractError("seed_values and reference_value must be numerical") from error
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(reference):
        raise BlendingE2EContractError("estimator values and reference must be finite scalars")
    count = values.size
    mean = complex(values.mean())
    components = np.column_stack((values.real, values.imag))
    covariance = np.cov(components, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    standard_error = np.sqrt(np.diag(covariance) / count)
    critical = NormalDist().inv_cdf(0.5 + level / 2.0)
    difference = np.array([mean.real - reference.real, mean.imag - reference.imag])
    interval = np.column_stack((difference - critical * standard_error, difference + critical * standard_error))
    compatible = bool(np.all(interval[:, 0] <= 0.0) and np.all(interval[:, 1] >= 0.0))
    z_scores = [
        float(delta / error) if error > 0 else (0.0 if delta == 0 else float(np.copysign(np.inf, delta)))
        for delta, error in zip(difference, standard_error)
    ]
    return {
        "sample_count": int(count),
        "confidence": level,
        "mean": {"real": float(mean.real), "imag": float(mean.imag)},
        "sample_variance": {"real": float(covariance[0, 0]), "imag": float(covariance[1, 1])},
        "mean_standard_error": {"real": float(standard_error[0]), "imag": float(standard_error[1])},
        "difference_to_reference": {"real": float(difference[0]), "imag": float(difference[1])},
        "difference_confidence_interval": {
            "real": [float(interval[0, 0]), float(interval[0, 1])],
            "imag": [float(interval[1, 0]), float(interval[1, 1])],
        },
        "z_score": {"real": z_scores[0], "imag": z_scores[1]},
        "compatible_with_reference_at_confidence": compatible,
        "unbiasedness_proven": False,
        "interpretation": "finite-seed compatibility only; independent target-runtime validation remains required",
    }


def validate_blending_limit_controls(
    nst_zero_observable: Any,
    distillation_reference: Any,
    complete_dilution_observable: Any,
    exact_reference: Any,
    *,
    atol: float = 1.0e-12,
) -> dict[str, float]:
    """Check the two non-stochastic limits against frozen references.

    ``N_st=0`` is compared with the selected low-mode distillation reference;
    complete dilution is compared with the full exact reference.  Neither
    check asserts that an incomplete stochastic estimator is unbiased.
    """

    tolerance = _finite(atol, "atol", 0.0)

    def residual(value: Any, reference: Any, name: str) -> float:
        try:
            observed = np.asarray(value, dtype=np.complex128)
            expected = np.asarray(reference, dtype=np.complex128)
        except (TypeError, ValueError) as error:
            raise BlendingE2EContractError(f"{name} must be numerical") from error
        if observed.shape != expected.shape or observed.size == 0:
            raise BlendingE2EContractError(f"{name} and its reference must have one nonempty shape")
        if not np.isfinite(observed).all() or not np.isfinite(expected).all():
            raise BlendingE2EContractError(f"{name} and its reference must be finite")
        return float(np.max(np.abs(observed - expected)))

    nst_zero_residual = residual(
        nst_zero_observable, distillation_reference, "nst_zero_observable"
    )
    complete_dilution_residual = residual(
        complete_dilution_observable, exact_reference, "complete_dilution_observable"
    )
    if max(nst_zero_residual, complete_dilution_residual) > tolerance:
        raise BlendingE2EContractError(
            "N_st=0 distillation or complete-dilution exact control failed"
        )
    return {
        "nst_zero_distillation_residual": nst_zero_residual,
        "complete_dilution_exact_residual": complete_dilution_residual,
    }


def validate_blending_result_metadata(
    *,
    contract: Mapping[str, Any],
    solve_records: Sequence[Mapping[str, Any]],
    io_artifacts: Sequence[Mapping[str, Any]],
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate post-run solve/I-O metadata without claiming a run occurred.

    ``contract`` is the frozen preflight declaration.  Requiring it here
    prevents a result receipt from introducing unbound seed or dilution labels
    after the run and makes the LapH/perambulator/reference provenance
    checkable at the same boundary as the measurements.
    """

    preflight = validate_blending_e2e_preflight(contract)
    expected_seed_records = tuple(
        _mapping(value, f"contract.noise.seed_records[{index}]")
        for index, value in enumerate(_mapping(contract["noise"], "contract.noise")["seed_records"])
    )
    expected_seed_coordinates = {
        (
            int(record["seed"]),
            str(record["configuration_id"]),
            str(record["time_slice"]),
            str(record["noise_label"]),
        )
        for record in expected_seed_records
    }
    dilution = _mapping(contract["dilution"], "contract.dilution")
    expected_dilution_labels = _string_sequence(
        dilution["labels"], "contract.dilution.labels", allow_empty=(int(contract["mode_counts"]["n_st"]) == 0)
    )
    producer_metadata = _mapping(preflight.get("producer_metadata"), "preflight.producer_metadata")
    expected_laph_id = _string(producer_metadata.get("laph_eigenvectors"), "preflight.producer_metadata.laph_eigenvectors")
    expected_peram_id = _string(producer_metadata.get("perambulator"), "preflight.producer_metadata.perambulator")

    if isinstance(solve_records, (str, bytes)) or not isinstance(solve_records, Sequence) or not solve_records:
        raise BlendingE2EContractError("solve_records must be nonempty")
    observed_keys: set[tuple[int, str, str, str, str]] = set()
    for index, value in enumerate(solve_records):
        record = _mapping(value, f"solve_records[{index}]")
        seed = _integer(record.get("seed"), f"solve_records[{index}].seed", 0)
        configuration_id = _string(record.get("configuration_id"), f"solve_records[{index}].configuration_id")
        time_slice = _string(record.get("time_slice"), f"solve_records[{index}].time_slice")
        noise_label = _string(record.get("noise_label"), f"solve_records[{index}].noise_label")
        dilution_label = _string(record.get("dilution_label"), f"solve_records[{index}].dilution_label")
        seed_coordinate = (seed, configuration_id, time_slice, noise_label)
        if seed_coordinate not in expected_seed_coordinates:
            raise BlendingE2EContractError(
                f"solve_records[{index}] is not bound to a declared noise seed coordinate"
            )
        if dilution_label not in expected_dilution_labels:
            raise BlendingE2EContractError(
                f"solve_records[{index}].dilution_label is not declared in dilution.labels"
            )
        key = (*seed_coordinate, dilution_label)
        if key in observed_keys:
            raise BlendingE2EContractError("solve_records must not duplicate a seed/dilution solve")
        observed_keys.add(key)
        if record.get("laph_eigenvector_artifact_id") != expected_laph_id:
            raise BlendingE2EContractError(
                f"solve_records[{index}] must bind the declared LapH eigenvector artifact"
            )
        if record.get("perambulator_artifact_id") != expected_peram_id:
            raise BlendingE2EContractError(
                f"solve_records[{index}] must bind the declared perambulator artifact"
            )
        if record.get("residual_kind") != "true":
            raise BlendingE2EContractError("every solve record must report a true residual")
        residual = _finite(record.get("true_residual"), f"solve_records[{index}].true_residual", 0.0)
        target = _finite(
            _mapping(contract["solver"], "contract.solver").get("true_residual_target"),
            "contract.solver.true_residual_target",
            0.0,
        )
        if residual > target:
            raise BlendingE2EContractError(
                f"solve_records[{index}].true_residual exceeds contract solver target"
            )
        _finite(record.get("solve_s"), f"solve_records[{index}].solve_s", 0.0)
    expected_keys = {
        (*coordinate, dilution_label)
        for coordinate in expected_seed_coordinates
        for dilution_label in expected_dilution_labels
    }
    if observed_keys != expected_keys:
        raise BlendingE2EContractError(
            "solve_records must cover every declared seed coordinate and dilution label"
        )
    artifacts = _artifact_rows(io_artifacts, "io_artifacts")
    return {
        "solve_record_count": len(solve_records),
        "io_artifact_count": len(artifacts),
        "cost": validate_cost_accounting(cost),
        "bound_seed_count": len(expected_seed_records),
        "bound_dilution_label_count": len(expected_dilution_labels),
        "evidence_limit": "metadata integrity only; caller must bind these records to target-runtime logs",
    }
