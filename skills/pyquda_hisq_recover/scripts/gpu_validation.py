#!/usr/bin/env python3
"""Single-DCU numerical validation for HISQ four-spin reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import traceback
from pathlib import Path

import numpy as np


SEED = 20260822
SOURCE_CASES = (
    ("three_odd_gamma_order", (1, 1, 0, 1), "reverse_source_order"),
    ("two_odd_source_dagger", (0, 1, 0, 1), "source_conjugate_without_transpose"),
)


def _gamma_basis(dtype):
    sigma1 = np.array([[0, 1], [1, 0]], dtype=dtype)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=dtype)
    identity2 = np.eye(2, dtype=dtype)
    return {
        "t": np.kron(sigma2, identity2),
        "z": np.kron(sigma1, sigma3),
        "y": np.kron(sigma1, sigma2),
        "x": np.kron(sigma1, sigma1),
        "unit": np.eye(4, dtype=dtype),
    }


def _omega_tzyx(coordinate, gammas, *, reverse=False):
    t, z, y, x = coordinate
    factors = {
        "t": gammas["t"] if t % 2 else gammas["unit"],
        "z": gammas["z"] if z % 2 else gammas["unit"],
        "y": gammas["y"] if y % 2 else gammas["unit"],
        "x": gammas["x"] if x % 2 else gammas["unit"],
    }
    if reverse:
        return factors["t"] @ factors["z"] @ factors["y"] @ factors["x"]
    return factors["x"] @ factors["y"] @ factors["z"] @ factors["t"]


def _oracle(
    field,
    source,
    gammas,
    *,
    mutant_source_no_transpose=False,
    mutant_source_reverse_order=False,
):
    if mutant_source_no_transpose and mutant_source_reverse_order:
        raise ValueError("select exactly one source mutant")
    lattice_shape = field.shape[:-2]
    result = np.empty(lattice_shape + (4, 4, 3, 3), dtype=field.dtype)
    source_omega = _omega_tzyx(
        source, gammas, reverse=mutant_source_reverse_order
    )
    source_factor = (
        source_omega.conj()
        if mutant_source_no_transpose
        else source_omega.conj().T
    )
    for coordinate in np.ndindex(lattice_shape):
        sink_omega = _omega_tzyx(coordinate, gammas)
        spin_factor = sink_omega @ source_factor
        result[coordinate] = (
            spin_factor[:, :, None, None] * field[coordinate][None, None, :, :]
        )
    return result


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


def run_validation():
    import cupy as cp

    import Def_hisq_recover as implementation

    class LatticeInfo:
        GLx = 4
        GLy = 2
        GLz = 2
        GLt = 2

    rng = np.random.default_rng(SEED)
    base_real = rng.standard_normal((2, 2, 2, 4, 3, 3))
    base_imag = rng.standard_normal((2, 2, 2, 4, 3, 3))
    checks = []
    for dtype_name, tolerance in (("complex64", 5.0e-5), ("complex128", 5.0e-12)):
        dtype = np.dtype(dtype_name)
        host_field = (base_real + 1j * base_imag).astype(dtype)
        host_gammas = _gamma_basis(dtype)
        device_field = cp.asarray(host_field)
        device_gammas = {name: cp.asarray(value) for name, value in host_gammas.items()}
        for case_name, source_tzyx, mutant_name in SOURCE_CASES:
            actual = implementation.hisq_to_wilson_evengrid(
                LatticeInfo(), device_field, source_tzyx, device_gammas
            )
            cp.cuda.get_current_stream().synchronize()
            if not isinstance(actual, cp.ndarray):
                raise AssertionError("reconstruction did not return a CuPy array")
            if actual.device.id != cp.cuda.Device().id:
                raise AssertionError("reconstruction returned on the wrong device")
            if actual.dtype != device_field.dtype:
                raise AssertionError("reconstruction changed dtype")
            expected = _oracle(host_field, source_tzyx, host_gammas)
            if mutant_name == "reverse_source_order":
                mutant = _oracle(
                    host_field,
                    source_tzyx,
                    host_gammas,
                    mutant_source_reverse_order=True,
                )
            else:
                mutant = _oracle(
                    host_field,
                    source_tzyx,
                    host_gammas,
                    mutant_source_no_transpose=True,
                )
            actual_host = cp.asnumpy(actual)
            error = _relative_max_error(actual_host, expected)
            mutant_separation = _relative_max_error(mutant, expected)
            if error > tolerance:
                raise AssertionError(
                    "{} {} reconstruction error {} exceeds {}".format(
                        dtype_name, case_name, error, tolerance
                    )
                )
            if mutant_separation <= 10.0 * tolerance:
                raise AssertionError(
                    "{} mutant is not discriminated for {}".format(
                        mutant_name, case_name
                    )
                )
            if not np.array_equal(cp.asnumpy(device_field), host_field):
                raise AssertionError("reconstruction modified its input field")
            checks.append(
                {
                    "case": case_name,
                    "source_tzyx": list(source_tzyx),
                    "dtype": dtype_name,
                    "shape": list(actual.shape),
                    "relative_max_error": error,
                    "tolerance": tolerance,
                    "mutant": mutant_name,
                    "mutant_separation": mutant_separation,
                }
            )
    return {
        "schema_version": 1,
        "skill": "pyquda_hisq_recover",
        "status": "PASS",
        "seed": SEED,
        "input_sha256": hashlib.sha256(
            np.ascontiguousarray(base_real + 1j * base_imag).tobytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": _device_info(cp),
        "gpu_kernel_executed": True,
        "quda_calls": 0,
        "oracle": "independent host site loop for Omega(x) Gchi Omega(x0)^dagger",
        "checks": checks,
        "evidence_scope": "single-DCU synthetic reconstruction algebra; no MPI gather or interacting HISQ solve",
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
            "skill": "pyquda_hisq_recover",
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
