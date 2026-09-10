#!/usr/bin/env python3
"""Fail-closed E2E validator for a momentum-smearing correlator workflow.

This module deliberately has no scheduler, subprocess, PyQUDA, CuPy, or MPI
dependency.  A cluster-side driver supplies the two inversion and contraction
hooks; this validator makes their provenance, residual evidence, sign controls
and complex reference comparison explicit before it can report a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "pyquda-momentum-e2e/v1"
CONTROL_NAMES = (
    "source_smear_sign_flip",
    "sink_fourier_sign_flip",
    "sequential_phase_sign_flip",
)


class MomentumE2EValidationError(ValueError):
    """Raised when an E2E request is incomplete or its evidence is invalid."""


@dataclass(frozen=True)
class MomentumE2ERequest:
    """Immutable run request delivered to externally owned cluster hooks."""

    contract: Mapping[str, Any]
    control: str | None = None

    def with_control(self, control: str) -> "MomentumE2ERequest":
        if control not in CONTROL_NAMES:
            raise MomentumE2EValidationError(f"unsupported sign control: {control}")
        return replace(self, control=control)

    @property
    def source_k_mode(self) -> tuple[float, float, float]:
        source = tuple(self.contract["smearing"]["source_k_mode"])
        if self.control == "source_smear_sign_flip":
            return tuple(-value for value in source)
        return source

    @property
    def sink_fourier_momentum(self) -> tuple[int, int, int]:
        momentum = tuple(self.contract["momenta"]["P_f"])
        if self.control == "sink_fourier_sign_flip":
            return tuple(-value for value in momentum)
        return momentum

    @property
    def sequential_phase_momentum(self) -> tuple[int, int, int]:
        momentum = tuple(self.contract["momenta"]["P_f"])
        if self.control == "sequential_phase_sign_flip":
            return tuple(-value for value in momentum)
        return momentum


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MomentumE2EValidationError(f"{name} must be a mapping")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MomentumE2EValidationError(f"{name} must be a nonempty string")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    value = _nonempty_string(value, name).lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise MomentumE2EValidationError(f"{name} must be a SHA-256 hexadecimal digest")
    return value


def _integer_vector(value: Any, name: str, *, length: int = 3) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise MomentumE2EValidationError(f"{name} must contain {length} integer entries")
    result: list[int] = []
    for entry in value:
        if isinstance(entry, (bool, np.bool_)):
            raise MomentumE2EValidationError(f"{name} entries must be integers")
        try:
            numeric = float(entry)
        except (TypeError, ValueError) as error:
            raise MomentumE2EValidationError(f"{name} entries must be integers") from error
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise MomentumE2EValidationError(f"{name} entries must be finite integers")
        result.append(int(numeric))
    return tuple(result)


def _real_vector(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise MomentumE2EValidationError(f"{name} must contain three real modes")
    result: list[float] = []
    for entry in value:
        if isinstance(entry, (bool, np.bool_)):
            raise MomentumE2EValidationError(f"{name} entries must be real numbers")
        try:
            numeric = float(entry)
        except (TypeError, ValueError) as error:
            raise MomentumE2EValidationError(f"{name} entries must be real numbers") from error
        if not math.isfinite(numeric):
            raise MomentumE2EValidationError(f"{name} entries must be finite")
        result.append(numeric)
    return tuple(result)


def _positive_real(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MomentumE2EValidationError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise MomentumE2EValidationError(f"{name} must be a real number") from error
    if not math.isfinite(numeric) or (numeric < 0 if allow_zero else numeric <= 0):
        relation = "nonnegative" if allow_zero else "positive"
        raise MomentumE2EValidationError(f"{name} must be finite and {relation}")
    return numeric


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MomentumE2EValidationError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise MomentumE2EValidationError(f"{name} must be a positive integer")
    return int(value)


def _canonical_checksum(array: Any) -> str:
    materialized = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(materialized.tobytes()).hexdigest()


def validate_momentum_e2e_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a cluster E2E request without running hooks.

    The returned object contains no machine-local data and does not authorize a
    scheduler operation.  It only establishes that a caller *may* invoke the
    supplied runtime hooks.
    """

    contract = _mapping(contract, "contract")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise MomentumE2EValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}"
        )

    ensemble = _mapping(contract.get("ensemble"), "ensemble")
    configuration = _mapping(contract.get("configuration"), "configuration")
    momenta = _mapping(contract.get("momenta"), "momenta")
    smearing = _mapping(contract.get("smearing"), "smearing")
    sequential = _mapping(contract.get("sequential"), "sequential")
    solver = _mapping(contract.get("solver"), "solver")
    comparison = _mapping(contract.get("comparison"), "comparison")
    execution = _mapping(contract.get("execution"), "execution")
    reference_checksums = _mapping(
        contract.get("reference_checksums"), "reference_checksums"
    )

    volume = _integer_vector(
        ensemble.get("volume_xyzt"), "ensemble.volume_xyzt", length=4
    )
    if any(extent < 1 for extent in volume):
        raise MomentumE2EValidationError("ensemble.volume_xyzt must be positive")
    canonical_ensemble = {
        "id": _nonempty_string(ensemble.get("id"), "ensemble.id"),
        "action": _nonempty_string(ensemble.get("action"), "ensemble.action"),
        "mass": _nonempty_string(ensemble.get("mass"), "ensemble.mass"),
        "lattice_spacing": _nonempty_string(
            ensemble.get("lattice_spacing"), "ensemble.lattice_spacing"
        ),
        "boundary_conditions": _nonempty_string(
            ensemble.get("boundary_conditions"), "ensemble.boundary_conditions"
        ),
        "volume_xyzt": volume,
    }
    canonical_configuration = {
        "id": _nonempty_string(configuration.get("id"), "configuration.id"),
        "gauge_checksum": _sha256(
            configuration.get("gauge_checksum"), "configuration.gauge_checksum"
        ),
        "gauge_revision": _nonempty_string(
            configuration.get("gauge_revision"), "configuration.gauge_revision"
        ),
    }

    p_i = _integer_vector(momenta.get("P_i"), "momenta.P_i")
    p_f = _integer_vector(momenta.get("P_f"), "momenta.P_f")
    q = _integer_vector(momenta.get("q"), "momenta.q")
    if q != tuple(final - initial for initial, final in zip(p_i, p_f)):
        raise MomentumE2EValidationError("momenta.q must equal P_f - P_i")

    endpoint = smearing.get("endpoint")
    source_mode = smearing.get("source_mode")
    sink_mode = smearing.get("sink_mode")
    if endpoint not in {"s2p", "s2s"}:
        raise MomentumE2EValidationError("smearing.endpoint must be 's2p' or 's2s'")
    if source_mode != "momentum_wuppertal":
        raise MomentumE2EValidationError(
            "smearing.source_mode must be 'momentum_wuppertal'"
        )
    allowed_sink = {"none", "momentum_wuppertal"}
    if sink_mode not in allowed_sink:
        raise MomentumE2EValidationError(
            "smearing.sink_mode must be 'none' or 'momentum_wuppertal'"
        )
    sink_k_raw = smearing.get("sink_k_mode")
    if endpoint == "s2p":
        if sink_mode != "none" or sink_k_raw is not None:
            raise MomentumE2EValidationError(
                "s2p requires sink_mode='none' and sink_k_mode=None"
            )
        sink_k_mode = None
    else:
        if sink_mode != "momentum_wuppertal":
            raise MomentumE2EValidationError(
                "s2s requires sink_mode='momentum_wuppertal'"
            )
        sink_k_mode = _real_vector(sink_k_raw, "smearing.sink_k_mode")
    rho = _positive_real(smearing.get("rho"), "smearing.rho")
    n_steps = _positive_int(smearing.get("n_steps"), "smearing.n_steps")
    if rho**2 >= 2.0 * n_steps / 3.0:
        raise MomentumE2EValidationError(
            "Wuppertal domain requires rho^2 < 2*n_steps/3"
        )

    owner = _mapping(sequential.get("downstream_dagger_owner"), "sequential.downstream_dagger_owner")
    if owner.get("operation") != "dagger":
        raise MomentumE2EValidationError(
            "sequential.downstream_dagger_owner.operation must be 'dagger'"
        )
    canonical_owner = {
        "component": _nonempty_string(owner.get("component"), "downstream dagger component"),
        "operation": "dagger",
    }

    if comparison.get("mode") != "full_complex":
        raise MomentumE2EValidationError(
            "comparison.mode must be 'full_complex'; magnitude-only comparisons are forbidden"
        )
    absolute_tolerance = _positive_real(
        comparison.get("absolute_tolerance"), "comparison.absolute_tolerance", allow_zero=True
    )
    relative_tolerance = _positive_real(
        comparison.get("relative_tolerance"), "comparison.relative_tolerance", allow_zero=True
    )
    if absolute_tolerance == 0.0 and relative_tolerance == 0.0:
        raise MomentumE2EValidationError("at least one complex comparison tolerance must be nonzero")
    control_minimum = _positive_real(
        comparison.get("control_minimum_separation"),
        "comparison.control_minimum_separation",
        allow_zero=True,
    )

    runtime_kind = execution.get("runtime_kind")
    if runtime_kind not in {"cpu_mock", "quda"}:
        raise MomentumE2EValidationError(
            "execution.runtime_kind must be 'cpu_mock' or 'quda'"
        )
    mpi_size = _positive_int(execution.get("mpi_size"), "execution.mpi_size")
    precision = _nonempty_string(solver.get("precision"), "solver.precision")

    return {
        "schema_version": SCHEMA_VERSION,
        "ensemble": canonical_ensemble,
        "configuration": canonical_configuration,
        "momenta": {"P_i": p_i, "P_f": p_f, "q": q},
        "smearing": {
            "endpoint": endpoint,
            "source_mode": source_mode,
            "sink_mode": sink_mode,
            "source_k_mode": _real_vector(smearing.get("source_k_mode"), "smearing.source_k_mode"),
            "sink_k_mode": sink_k_mode,
            "rho": rho,
            "n_steps": n_steps,
        },
        "sequential": {"downstream_dagger_owner": canonical_owner},
        "solver": {
            "precision": precision,
            "tolerance": _positive_real(solver.get("tolerance"), "solver.tolerance"),
        },
        "comparison": {
            "mode": "full_complex",
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "control_minimum_separation": control_minimum,
        },
        "execution": {"runtime_kind": runtime_kind, "mpi_size": mpi_size},
        "reference_checksums": {
            "two_point": _sha256(reference_checksums.get("two_point"), "reference_checksums.two_point"),
            "three_point": _sha256(reference_checksums.get("three_point"), "reference_checksums.three_point"),
        },
    }


def compare_full_complex(
    actual: Any,
    expected: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    label: str,
) -> dict[str, float]:
    """Compare complex correlators component-wise, never by magnitude alone."""

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise MomentumE2EValidationError(
            f"{label} shape mismatch: {actual_array.shape} != {expected_array.shape}"
        )
    if not np.iscomplexobj(actual_array) or not np.iscomplexobj(expected_array):
        raise MomentumE2EValidationError(
            f"{label} must supply complex-valued actual and reference arrays"
        )
    if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
        raise MomentumE2EValidationError(f"{label} contains non-finite values")
    delta = actual_array - expected_array
    absolute_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    reference_scale = max(1.0, float(np.max(np.abs(expected_array))) if expected_array.size else 0.0)
    relative_error = absolute_error / reference_scale
    threshold = absolute_tolerance + relative_tolerance * reference_scale
    if absolute_error > threshold:
        raise MomentumE2EValidationError(
            f"{label} full-complex comparison failed: {absolute_error} > {threshold}"
        )
    return {
        "absolute_max_error": absolute_error,
        "relative_max_error": relative_error,
        "reference_scale": reference_scale,
        "threshold": threshold,
    }


def _inversion_record(value: Any, label: str) -> tuple[Mapping[str, Any], dict[str, Any]]:
    record = _mapping(value, f"{label} inversion result")
    required = (
        "solution",
        "rhs",
        "apply_operator",
        "true_residual",
        "reported_residual",
        "iterations",
        "setup_seconds",
        "solve_seconds",
        "total_seconds",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise MomentumE2EValidationError(f"{label} inversion result is missing {missing}")
    operator = record["apply_operator"]
    if not callable(operator):
        raise MomentumE2EValidationError(f"{label}.apply_operator must be callable")
    solution = np.asarray(record["solution"])
    rhs = np.asarray(record["rhs"])
    if solution.shape != rhs.shape or solution.size == 0:
        raise MomentumE2EValidationError(f"{label} solution/rhs must have the same nonempty shape")
    applied = np.asarray(operator(solution))
    if applied.shape != rhs.shape:
        raise MomentumE2EValidationError(f"{label}.apply_operator returned the wrong shape")
    denominator = float(np.linalg.norm(rhs.ravel()))
    if denominator == 0.0:
        raise MomentumE2EValidationError(f"{label}.rhs must have nonzero norm for a true residual")
    computed_true = float(np.linalg.norm((applied - rhs).ravel()) / denominator)
    captured_true = _positive_real(record["true_residual"], f"{label}.true_residual", allow_zero=True)
    capture_tolerance = max(1.0e-14, 32.0 * np.finfo(float).eps * max(1.0, computed_true))
    if abs(captured_true - computed_true) > capture_tolerance:
        raise MomentumE2EValidationError(
            f"{label}.true_residual does not match independently recomputed residual"
        )
    setup_seconds = _positive_real(record["setup_seconds"], f"{label}.setup_seconds", allow_zero=True)
    solve_seconds = _positive_real(record["solve_seconds"], f"{label}.solve_seconds", allow_zero=True)
    total_seconds = _positive_real(record["total_seconds"], f"{label}.total_seconds", allow_zero=True)
    if total_seconds + 1.0e-15 < setup_seconds + solve_seconds:
        raise MomentumE2EValidationError(f"{label}.total_seconds is less than setup+solve")
    return record, {
        "true_residual": computed_true,
        "reported_residual": _positive_real(record["reported_residual"], f"{label}.reported_residual", allow_zero=True),
        "iterations": _positive_int(record["iterations"], f"{label}.iterations"),
        "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": total_seconds,
    }


def _hook(hooks: Mapping[str, Any], name: str) -> Callable[..., Any]:
    value = hooks.get(name) if isinstance(hooks, Mapping) else None
    if not callable(value):
        raise MomentumE2EValidationError(f"hooks.{name} must be a callable external hook")
    return value


def _run_once(request: MomentumE2ERequest, hooks: Mapping[str, Any]) -> dict[str, Any]:
    forward, forward_summary = _inversion_record(
        _hook(hooks, "forward_invert")(request), "forward"
    )
    sequential, sequential_summary = _inversion_record(
        _hook(hooks, "sequential_invert")(request), "sequential"
    )
    daggered = _hook(hooks, "downstream_dagger")(sequential["solution"], request)
    if daggered is None:
        raise MomentumE2EValidationError("hooks.downstream_dagger must return the daggered sequential line")
    two_point = _hook(hooks, "contract_two_point")(forward["solution"], request)
    three_point = _hook(hooks, "contract_three_point")(
        forward["solution"], daggered, request
    )
    return {
        "two_point": np.asarray(two_point),
        "three_point": np.asarray(three_point),
        "inversions": {"forward": forward_summary, "sequential": sequential_summary},
    }


def run_momentum_e2e(
    contract: Mapping[str, Any],
    hooks: Mapping[str, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    """Run caller-provided inversions/contractions and validate their E2E evidence.

    The caller owns source construction, real PyQUDA/QUDA inversions, MPI and
    the downstream dagger implementation.  This function calls every supplied
    hook and fail-closes on absent residuals, a magnitude-only comparison, or a
    nondiscriminating sign mutation.
    """

    canonical = validate_momentum_e2e_contract(contract)
    references = _mapping(references, "references")
    if set(references) != {"two_point", "three_point"}:
        raise MomentumE2EValidationError("references must contain exactly two_point and three_point")
    for name in ("two_point", "three_point"):
        if _canonical_checksum(references[name]) != canonical["reference_checksums"][name]:
            raise MomentumE2EValidationError(f"reference checksum mismatch for {name}")

    request = MomentumE2ERequest(canonical)
    baseline = _run_once(request, hooks)
    comparisons = {
        name: compare_full_complex(
            baseline[name],
            references[name],
            absolute_tolerance=canonical["comparison"]["absolute_tolerance"],
            relative_tolerance=canonical["comparison"]["relative_tolerance"],
            label=name,
        )
        for name in ("two_point", "three_point")
    }
    # ``compare_full_complex`` is an equality checker.  Compute mutation
    # separation directly so sign controls are required to *differ*.
    controls: dict[str, dict[str, float]] = {}
    baseline_three = baseline["three_point"]
    for control in CONTROL_NAMES:
        mutated = _run_once(request.with_control(control), hooks)
        compare_full_complex(
            mutated["two_point"], mutated["two_point"],
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            label=f"{control} two_point finite check",
        )
        mutant_three = mutated["three_point"]
        if not np.iscomplexobj(mutant_three) or mutant_three.shape != baseline_three.shape:
            raise MomentumE2EValidationError(f"{control} did not produce a compatible complex 3pt")
        separation = float(np.max(np.abs(mutant_three - baseline_three)))
        scale = max(1.0, float(np.max(np.abs(baseline_three))))
        relative_separation = separation / scale
        if relative_separation <= canonical["comparison"]["control_minimum_separation"]:
            raise MomentumE2EValidationError(
                f"{control} is not discriminated on the full complex 3pt correlator"
            )
        controls[control] = {
            "absolute_max_separation": separation,
            "relative_max_separation": relative_separation,
        }

    runtime_kind = canonical["execution"]["runtime_kind"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "PASS_CPU_MOCK_NOT_RUNTIME"
            if runtime_kind == "cpu_mock"
            else "PASS_EXTERNAL_QUDA_HOOKS_NOT_INDEPENDENTLY_AUDITED"
        ),
        "runtime_kind": runtime_kind,
        "mpi_size": canonical["execution"]["mpi_size"],
        "jobs_submitted": 0,
        "comparison_mode": "full_complex",
        "downstream_dagger_owner": canonical["sequential"]["downstream_dagger_owner"],
        "comparisons": comparisons,
        "controls": controls,
        "inversions": baseline["inversions"],
    }


def preflight_momentum_e2e(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the static contract; never imports or submits a runtime job."""

    canonical = validate_momentum_e2e_contract(contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "READY_FOR_EXTERNAL_HOOKS_NOT_EXECUTED",
        "runtime_kind": canonical["execution"]["runtime_kind"],
        "jobs_submitted": 0,
        "commands": [],
        "required_hooks": [
            "forward_invert",
            "sequential_invert",
            "downstream_dagger",
            "contract_two_point",
            "contract_three_point",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = preflight_momentum_e2e(json.loads(args.contract.read_text(encoding="utf-8")))
        exit_code = 0
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED_BY_INVALID_CONTRACT",
            "error_type": type(error).__name__,
            "error": str(error),
            "jobs_submitted": 0,
        }
        exit_code = 2
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
