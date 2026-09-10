#!/usr/bin/env python3
"""Fail-closed E2E validator for producer-specific HISQ spin reconstruction.

The generic reconstruction kernel cannot infer a staggered propagator producer's
taste, eta, boundary or storage convention.  This module makes that producer
contract executable while leaving data I/O, MPI transport and reference spin
lift ownership with explicitly supplied hooks.  It never launches a process or
submits a scheduler job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "pyquda-hisq-producer/v1"
SUPPORTED_DTYPES = ("complex64", "complex128")


class HISQE2EValidationError(ValueError):
    """Raised when producer metadata or a spin-lift comparison is incomplete."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HISQE2EValidationError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HISQE2EValidationError(f"{name} must be a nonempty string")
    return value.strip()


def _supported_dtype_declaration(value: Any, name: str) -> str:
    """Validate a single or ``+``-joined declaration of supported dtypes.

    The metadata contract uses one field for the source representation, while
    the E2E fixture may deliberately declare both precision cases as
    ``complex64+complex128``.  Accept that explicit union, but reject
    unsupported, empty, or duplicated dtype tokens before any runtime hook is
    called.
    """

    declaration = _text(value, name)
    tokens = tuple(token.strip() for token in declaration.split("+"))
    if (
        not tokens
        or any(token not in SUPPORTED_DTYPES for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        supported = "+".join(SUPPORTED_DTYPES)
        raise HISQE2EValidationError(
            f"{name} must contain each supported dtype at most once ({supported})"
        )
    return declaration


def _sha256(value: Any, name: str) -> str:
    value = _text(value, name).lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HISQE2EValidationError(f"{name} must be a SHA-256 hexadecimal digest")
    return value


def _integer_vector(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    allow_negative: bool = False,
) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise HISQE2EValidationError(f"{name} must contain four integer entries")
    result: list[int] = []
    for entry in value:
        if isinstance(entry, (bool, np.bool_)):
            raise HISQE2EValidationError(f"{name} entries must be integers")
        try:
            numeric = float(entry)
        except (TypeError, ValueError) as error:
            raise HISQE2EValidationError(f"{name} entries must be integers") from error
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise HISQE2EValidationError(f"{name} entries must be finite integers")
        integer = int(numeric)
        if (positive and integer < 1) or (not positive and not allow_negative and integer < 0):
            relation = "positive" if positive else "nonnegative"
            raise HISQE2EValidationError(f"{name} entries must be {relation}")
        result.append(integer)
    return tuple(result)


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise HISQE2EValidationError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise HISQE2EValidationError(f"{name} must be a real number") from error
    if not math.isfinite(numeric) or (numeric < 0 if allow_zero else numeric <= 0):
        relation = "nonnegative" if allow_zero else "positive"
        raise HISQE2EValidationError(f"{name} must be finite and {relation}")
    return numeric


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise HISQE2EValidationError(f"{name} must be a positive integer")
    if int(value) < 1:
        raise HISQE2EValidationError(f"{name} must be a positive integer")
    return int(value)


def _full_complex_comparison(
    actual: Any,
    expected: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    label: str,
) -> dict[str, float]:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise HISQE2EValidationError(
            f"{label} shape mismatch: {actual_array.shape} != {expected_array.shape}"
        )
    if not np.iscomplexobj(actual_array) or not np.iscomplexobj(expected_array):
        raise HISQE2EValidationError(f"{label} must compare complex arrays")
    if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
        raise HISQE2EValidationError(f"{label} contains non-finite values")
    error = float(np.max(np.abs(actual_array - expected_array))) if actual_array.size else 0.0
    scale = max(1.0, float(np.max(np.abs(expected_array))) if expected_array.size else 0.0)
    threshold = absolute_tolerance + relative_tolerance * scale
    if error > threshold:
        raise HISQE2EValidationError(f"{label} full-complex comparison failed: {error} > {threshold}")
    return {
        "absolute_max_error": error,
        "relative_max_error": error / scale,
        "reference_scale": scale,
        "threshold": threshold,
    }


def validate_hisq_producer_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a versioned producer adapter contract without reading data."""

    metadata = _mapping(metadata, "metadata")
    if metadata.get("contract_version") != SCHEMA_VERSION:
        raise HISQE2EValidationError(
            f"contract_version must be {SCHEMA_VERSION!r}"
        )
    producer = _mapping(metadata.get("producer"), "producer")
    input_record = _mapping(metadata.get("input"), "input")
    physics = _mapping(metadata.get("physics"), "physics")
    coordinates = _mapping(metadata.get("coordinates"), "coordinates")
    boundary = _mapping(metadata.get("boundary"), "boundary")
    layout = _mapping(metadata.get("layout"), "layout")
    execution = _mapping(metadata.get("execution"), "execution")
    reference = _mapping(metadata.get("reference"), "reference")
    validation = _mapping(metadata.get("validation"), "validation")

    global_shape = _integer_vector(layout.get("global_shape_tzyx"), "layout.global_shape_tzyx", positive=True)
    local_shape = _integer_vector(layout.get("local_shape_tzyx"), "layout.local_shape_tzyx", positive=True)
    grid_size = _integer_vector(layout.get("grid_size_xyzt"), "layout.grid_size_xyzt", positive=True)
    grid_coord = _integer_vector(layout.get("grid_coord_xyzt"), "layout.grid_coord_xyzt")
    if any(coordinate >= extent for coordinate, extent in zip(grid_coord, grid_size)):
        raise HISQE2EValidationError("layout.grid_coord_xyzt lies outside layout.grid_size_xyzt")
    expected_global = tuple(
        local_shape[axis] * grid_size[3 - axis] for axis in range(4)
    )
    if global_shape != expected_global:
        raise HISQE2EValidationError(
            "global_shape_tzyx must equal local_shape_tzyx times grid_size_xyzt"
        )
    expected_offset = (
        grid_coord[3] * local_shape[0],
        grid_coord[2] * local_shape[1],
        grid_coord[1] * local_shape[2],
        grid_coord[0] * local_shape[3],
    )
    if _integer_vector(layout.get("rank_offsets_tzyx"), "layout.rank_offsets_tzyx") != expected_offset:
        raise HISQE2EValidationError(
            "layout.rank_offsets_tzyx must be derived from grid_coord and local_shape"
        )
    mpi_size = _positive_int(execution.get("mpi_size"), "execution.mpi_size")
    expected_mpi_size = int(np.prod(grid_size))
    if mpi_size != expected_mpi_size:
        raise HISQE2EValidationError("execution.mpi_size must equal product(grid_size_xyzt)")
    rank = execution.get("rank")
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise HISQE2EValidationError("execution.rank must be a non-boolean integer")
    if int(rank) < 0 or int(rank) >= mpi_size:
        raise HISQE2EValidationError("execution.rank lies outside execution.mpi_size")
    runtime_kind = execution.get("runtime_kind")
    if runtime_kind not in {"cpu_mock", "mpi"}:
        raise HISQE2EValidationError("execution.runtime_kind must be 'cpu_mock' or 'mpi'")

    if layout.get("input_layout") != "tzyx_color_color":
        raise HISQE2EValidationError(
            "generic recovery accepts only input_layout='tzyx_color_color'"
        )
    if layout.get("even_odd_storage") != "full_lexicographic":
        raise HISQE2EValidationError(
            "generic recovery accepts only even_odd_storage='full_lexicographic'"
        )
    ownership = _mapping(layout.get("rank_ownership"), "layout.rank_ownership")
    if ownership != {
        "local_field": "all_ranks",
        "global_field": "root_only",
        "root_rank": 0,
    }:
        raise HISQE2EValidationError(
            "layout.rank_ownership must explicitly declare local all-ranks and global root-only ownership"
        )

    source_coordinate = _integer_vector(
        coordinates.get("source_global_tzyx"), "coordinates.source_global_tzyx"
    )
    if any(value >= extent for value, extent in zip(source_coordinate, global_shape)):
        raise HISQE2EValidationError("coordinates.source_global_tzyx lies outside global_shape_tzyx")
    if coordinates.get("coordinate_origin_base") != 0:
        raise HISQE2EValidationError("coordinates.coordinate_origin_base must be 0")
    if _integer_vector(coordinates.get("lattice_origin_tzyx"), "coordinates.lattice_origin_tzyx") != (0, 0, 0, 0):
        raise HISQE2EValidationError("coordinates.lattice_origin_tzyx must be explicit zero origin")

    signs = _integer_vector(
        boundary.get("signs_xyzt"), "boundary.signs_xyzt", allow_negative=True
    )
    if any(sign not in {-1, 1} for sign in signs):
        raise HISQE2EValidationError("boundary.signs_xyzt entries must be +1 or -1")
    if not isinstance(boundary.get("temporal_wrap_already_applied"), (bool, np.bool_)):
        raise HISQE2EValidationError("boundary.temporal_wrap_already_applied must be Boolean")

    validation_dtypes = validation.get("dtypes")
    if not isinstance(validation_dtypes, Sequence) or isinstance(validation_dtypes, (str, bytes)):
        raise HISQE2EValidationError("validation.dtypes must list complex64 and complex128")
    if tuple(validation_dtypes) != SUPPORTED_DTYPES:
        raise HISQE2EValidationError("validation.dtypes must be exactly ['complex64', 'complex128']")
    if validation.get("comparison_mode") != "full_complex":
        raise HISQE2EValidationError(
            "validation.comparison_mode must be 'full_complex'; magnitude-only is forbidden"
        )
    absolute = _mapping(validation.get("absolute_tolerance"), "validation.absolute_tolerance")
    relative = _mapping(validation.get("relative_tolerance"), "validation.relative_tolerance")
    tolerances = {
        dtype: {
            "absolute": _positive_number(absolute.get(dtype), f"absolute_tolerance.{dtype}", allow_zero=True),
            "relative": _positive_number(relative.get(dtype), f"relative_tolerance.{dtype}", allow_zero=True),
        }
        for dtype in SUPPORTED_DTYPES
    }
    if any(values["absolute"] == 0.0 and values["relative"] == 0.0 for values in tolerances.values()):
        raise HISQE2EValidationError("every dtype needs a nonzero full-complex tolerance")

    implementation_owner = _text(reference.get("implementation_owner"), "reference.implementation_owner")
    reference_owner = _text(reference.get("producer_reference_owner"), "reference.producer_reference_owner")
    if implementation_owner == reference_owner:
        raise HISQE2EValidationError("producer reference owner must be independent from implementation owner")

    return {
        "contract_version": SCHEMA_VERSION,
        "producer": {
            "name": _text(producer.get("name"), "producer.name"),
            "version": _text(producer.get("version"), "producer.version"),
        },
        "input": {
            "file_format": _text(input_record.get("file_format"), "input.file_format"),
            "dataset": _text(input_record.get("dataset"), "input.dataset"),
            "checksum": _sha256(input_record.get("checksum"), "input.checksum"),
            "complex_dtype": _supported_dtype_declaration(
                input_record.get("complex_dtype"), "input.complex_dtype"
            ),
        },
        "physics": {
            "taste": _text(physics.get("taste"), "physics.taste"),
            "staggered_eta_convention": _text(physics.get("staggered_eta_convention"), "physics.staggered_eta_convention"),
            "gamma_basis_id": _text(physics.get("gamma_basis_id"), "physics.gamma_basis_id"),
            "omega_gamma_order": _text(physics.get("omega_gamma_order"), "physics.omega_gamma_order"),
            "source_normalization": _text(physics.get("source_normalization"), "physics.source_normalization"),
        },
        "coordinates": {
            "source_global_tzyx": source_coordinate,
            "lattice_origin_tzyx": (0, 0, 0, 0),
            "coordinate_origin_base": 0,
        },
        "boundary": {
            "signs_xyzt": signs,
            "temporal_wrap_already_applied": bool(boundary["temporal_wrap_already_applied"]),
        },
        "layout": {
            "input_layout": "tzyx_color_color",
            "even_odd_storage": "full_lexicographic",
            "global_shape_tzyx": global_shape,
            "local_shape_tzyx": local_shape,
            "grid_size_xyzt": grid_size,
            "grid_coord_xyzt": grid_coord,
            "rank_offsets_tzyx": expected_offset,
            "rank_ownership": dict(ownership),
        },
        "execution": {"runtime_kind": runtime_kind, "mpi_size": mpi_size, "rank": int(rank)},
        "reference": {
            "implementation_owner": implementation_owner,
            "producer_reference_owner": reference_owner,
        },
        "validation": {"dtypes": SUPPORTED_DTYPES, "tolerances": tolerances},
    }


def _hook(hooks: Mapping[str, Any], name: str) -> Callable[..., Any]:
    hook = hooks.get(name) if isinstance(hooks, Mapping) else None
    if not callable(hook):
        raise HISQE2EValidationError(f"hooks.{name} must be a callable external hook")
    return hook


def _field_for_dtype(value: Any, dtype_name: str, local_shape: tuple[int, int, int, int]) -> np.ndarray:
    field = np.asarray(value)
    expected_shape = tuple(local_shape) + (3, 3)
    if field.shape != expected_shape:
        raise HISQE2EValidationError(f"{dtype_name} input has shape {field.shape}, expected {expected_shape}")
    if field.dtype != np.dtype(dtype_name):
        raise HISQE2EValidationError(f"{dtype_name} input must use dtype {dtype_name}")
    if not np.isfinite(field).all():
        raise HISQE2EValidationError(f"{dtype_name} input contains non-finite values")
    return field


def _output_for_dtype(value: Any, dtype_name: str, shape: tuple[int, ...], label: str) -> np.ndarray:
    field = np.asarray(value)
    if field.shape != shape:
        raise HISQE2EValidationError(f"{label} shape {field.shape} does not match expected {shape}")
    if field.dtype != np.dtype(dtype_name):
        raise HISQE2EValidationError(f"{label} must preserve {dtype_name}")
    if not np.iscomplexobj(field):
        raise HISQE2EValidationError(f"{label} must be complex")
    return field


def run_hisq_e2e(
    metadata: Mapping[str, Any],
    local_inputs: Mapping[str, Any],
    hooks: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare production reconstruction with an independently owned spin lift.

    ``recover_local`` and ``recover_global`` are the implementation path.
    ``producer_spin_lift_local`` and ``producer_spin_lift_global`` must come
    from the named producer or another independently owned reference.  On a
    non-root rank both global hooks must return ``None`` by contract.
    """

    canonical = validate_hisq_producer_metadata(metadata)
    local_inputs = _mapping(local_inputs, "local_inputs")
    if set(local_inputs) != set(SUPPORTED_DTYPES):
        raise HISQE2EValidationError("local_inputs must contain exactly complex64 and complex128")
    recover_local = _hook(hooks, "recover_local")
    reference_local = _hook(hooks, "producer_spin_lift_local")
    recover_global = _hook(hooks, "recover_global")
    reference_global = _hook(hooks, "producer_spin_lift_global")
    if recover_local is reference_local or recover_global is reference_global:
        raise HISQE2EValidationError("implementation and producer reference hooks must be distinct callables")

    local_shape = canonical["layout"]["local_shape_tzyx"]
    global_shape = canonical["layout"]["global_shape_tzyx"]
    rank = canonical["execution"]["rank"]
    root_rank = canonical["layout"]["rank_ownership"]["root_rank"]
    checks: dict[str, Any] = {}
    for dtype_name in SUPPORTED_DTYPES:
        input_field = _field_for_dtype(local_inputs[dtype_name], dtype_name, local_shape)
        input_digest = hashlib.sha256(np.ascontiguousarray(input_field).tobytes()).hexdigest()
        actual_local = _output_for_dtype(
            recover_local(input_field, canonical),
            dtype_name,
            tuple(local_shape) + (4, 4, 3, 3),
            f"{dtype_name} implementation local result",
        )
        if hashlib.sha256(np.ascontiguousarray(input_field).tobytes()).hexdigest() != input_digest:
            raise HISQE2EValidationError(f"{dtype_name} implementation modified its local input")
        expected_local = _output_for_dtype(
            reference_local(input_field, canonical),
            dtype_name,
            tuple(local_shape) + (4, 4, 3, 3),
            f"{dtype_name} producer reference local result",
        )
        if hashlib.sha256(np.ascontiguousarray(input_field).tobytes()).hexdigest() != input_digest:
            raise HISQE2EValidationError(f"{dtype_name} producer reference modified its local input")
        tolerance = canonical["validation"]["tolerances"][dtype_name]
        local_comparison = _full_complex_comparison(
            actual_local,
            expected_local,
            absolute_tolerance=tolerance["absolute"],
            relative_tolerance=tolerance["relative"],
            label=f"{dtype_name} local spin lift",
        )
        actual_global = recover_global(actual_local, canonical)
        expected_global = reference_global(expected_local, canonical)
        if rank == root_rank:
            actual_global = _output_for_dtype(
                actual_global,
                dtype_name,
                tuple(global_shape) + (4, 4, 3, 3),
                f"{dtype_name} implementation global result",
            )
            expected_global = _output_for_dtype(
                expected_global,
                dtype_name,
                tuple(global_shape) + (4, 4, 3, 3),
                f"{dtype_name} producer reference global result",
            )
            global_comparison: dict[str, float] | None = _full_complex_comparison(
                actual_global,
                expected_global,
                absolute_tolerance=tolerance["absolute"],
                relative_tolerance=tolerance["relative"],
                label=f"{dtype_name} global spin lift",
            )
        else:
            if actual_global is not None or expected_global is not None:
                raise HISQE2EValidationError(
                    "non-root ranks must return None for both global reconstruction outputs"
                )
            global_comparison = None
        checks[dtype_name] = {
            "input_checksum": input_digest,
            "local": local_comparison,
            "global": global_comparison,
        }

    runtime_kind = canonical["execution"]["runtime_kind"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "PASS_CPU_MOCK_NOT_MPI_RUNTIME"
            if runtime_kind == "cpu_mock"
            else "PASS_EXTERNAL_MPI_HOOKS_NOT_INDEPENDENTLY_AUDITED"
        ),
        "runtime_kind": runtime_kind,
        "producer": canonical["producer"],
        "rank": rank,
        "mpi_size": canonical["execution"]["mpi_size"],
        "global_owner_rank": root_rank,
        "jobs_submitted": 0,
        "comparison_mode": "full_complex",
        "checks": checks,
    }


def preflight_hisq_e2e(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate metadata only; it has no data, MPI import, or scheduler side effect."""

    canonical = validate_hisq_producer_metadata(metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "READY_FOR_EXTERNAL_HOOKS_NOT_EXECUTED",
        "runtime_kind": canonical["execution"]["runtime_kind"],
        "jobs_submitted": 0,
        "commands": [],
        "required_hooks": [
            "recover_local",
            "producer_spin_lift_local",
            "recover_global",
            "producer_spin_lift_global",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = preflight_hisq_e2e(json.loads(args.metadata.read_text(encoding="utf-8")))
        exit_code = 0
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED_BY_INVALID_PRODUCER_METADATA",
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
