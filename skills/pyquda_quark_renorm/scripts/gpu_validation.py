#!/usr/bin/env python3
"""Single-DCU validation for quark spin-color inverse and gamma5 adjoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import traceback
from pathlib import Path

import numpy as np


SEED = 20260822


def _field_from_matrix(matrix):
    leading = matrix.shape[:-2]
    field = np.empty(leading + (4, 4, 3, 3), dtype=matrix.dtype)
    for sink_spin in range(4):
        for source_spin in range(4):
            for sink_color in range(3):
                for source_color in range(3):
                    field[..., sink_spin, source_spin, sink_color, source_color] = matrix[
                        ...,
                        3 * sink_spin + sink_color,
                        3 * source_spin + source_color,
                    ]
    return field


def _matrix_from_field(field):
    leading = field.shape[:-4]
    matrix = np.empty(leading + (12, 12), dtype=field.dtype)
    for sink_spin in range(4):
        for source_spin in range(4):
            for sink_color in range(3):
                for source_color in range(3):
                    matrix[
                        ...,
                        3 * sink_spin + sink_color,
                        3 * source_spin + source_color,
                    ] = field[..., sink_spin, source_spin, sink_color, source_color]
    return matrix


def _relative_max_error(actual, expected):
    scale = max(1.0, float(np.max(np.abs(expected))))
    return float(np.max(np.abs(actual - expected)) / scale)


def _device_info(cp):
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties.get("name", properties.get(b"name", "unknown"))
    arch = properties.get("gcnArchName", properties.get(b"gcnArchName", "unknown"))
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    if isinstance(arch, bytes):
        arch = arch.decode(errors="replace")
    return {"id": int(device.id), "name": str(name), "arch": str(arch)}


def _spin_only_adjoint_mutant(field, gamma5):
    dagger = field.conj().swapaxes(-4, -3)
    return np.einsum(
        "ij,...jkab,kl->...ilab", gamma5, dagger, gamma5, optimize=True
    )


def run_validation():
    import cupy as cp

    import Def_quark_renorm as implementation

    rng = np.random.default_rng(SEED)
    random_blocks = (
        rng.standard_normal((2, 3, 12, 12))
        + 1j * rng.standard_normal((2, 3, 12, 12))
    )
    gamma5 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)
    checks = []
    for dtype_name, tolerance in (("complex64", 5.0e-5), ("complex128", 5.0e-12)):
        dtype = np.dtype(dtype_name)
        identity = np.eye(12, dtype=dtype)
        matrices = 4.5 * identity + 0.06 * random_blocks.astype(dtype)
        fields = _field_from_matrix(matrices)
        device_fields = cp.asarray(fields)
        device_gamma5 = cp.asarray(gamma5.astype(dtype))
        actual_inverse = implementation.inverse_propagator(device_fields)
        actual_adjoint = implementation.adj(device_fields, device_gamma5)
        cp.cuda.get_current_stream().synchronize()
        if not isinstance(actual_inverse, cp.ndarray) or not isinstance(actual_adjoint, cp.ndarray):
            raise AssertionError("quark kernels did not preserve the CuPy backend")
        inverse_host = _matrix_from_field(cp.asnumpy(actual_inverse))
        adjoint_host = _matrix_from_field(cp.asnumpy(actual_adjoint))
        expected_inverse = np.linalg.inv(matrices)
        gamma5_color = np.kron(gamma5.astype(dtype), np.eye(3, dtype=dtype))
        expected_adjoint = gamma5_color @ matrices.conj().swapaxes(-1, -2) @ gamma5_color
        mutant_field = _spin_only_adjoint_mutant(fields, gamma5.astype(dtype))
        mutant_adjoint = _matrix_from_field(mutant_field)
        inverse_error = _relative_max_error(inverse_host, expected_inverse)
        adjoint_error = _relative_max_error(adjoint_host, expected_adjoint)
        mutant_separation = _relative_max_error(mutant_adjoint, expected_adjoint)
        residual = _relative_max_error(
            matrices @ inverse_host,
            np.broadcast_to(identity, matrices.shape),
        )
        condition_number = float(max(np.linalg.cond(block) for block in matrices.reshape(-1, 12, 12)))
        residual_bound = 200.0 * np.finfo(dtype).eps * condition_number
        if max(inverse_error, adjoint_error) > tolerance:
            raise AssertionError("{} quark differential failed".format(dtype_name))
        if residual > residual_bound:
            raise AssertionError("inverse residual exceeds conditioning-aware bound")
        if mutant_separation <= 10.0 * tolerance:
            raise AssertionError("spin-only adjoint mutant is not discriminated")
        involution = implementation.adj(actual_adjoint, device_gamma5)
        cp.cuda.get_current_stream().synchronize()
        involution_error = _relative_max_error(cp.asnumpy(involution), fields)
        if involution_error > tolerance:
            raise AssertionError("gamma5 adjoint is not an involution")
        checks.append(
            {
                "dtype": dtype_name,
                "shape": list(actual_inverse.shape),
                "inverse_relative_max_error": inverse_error,
                "adjoint_relative_max_error": adjoint_error,
                "adjoint_involution_error": involution_error,
                "forward_residual": residual,
                "residual_bound": float(residual_bound),
                "condition_number_max": condition_number,
                "tolerance": tolerance,
                "mutant": "Hermitian adjoint swaps spin axes but not color axes",
                "mutant_separation": mutant_separation,
            }
        )
    return {
        "schema_version": 1,
        "skill": "pyquda_quark_renorm",
        "status": "PASS",
        "seed": SEED,
        "input_sha256": hashlib.sha256(
            np.ascontiguousarray(random_blocks).tobytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": _device_info(cp),
        "gpu_kernel_executed": True,
        "quda_calls": 0,
        "oracle": "independent compound-index NumPy inverse and gamma5 M^dagger gamma5",
        "checks": checks,
        "evidence_scope": "single-DCU spin-color kernels only; legacy matching/running coefficient provenance remains unverified",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run_validation()
        exit_code = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "skill": "pyquda_quark_renorm",
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
