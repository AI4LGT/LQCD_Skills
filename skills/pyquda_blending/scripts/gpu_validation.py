#!/usr/bin/env python3
"""Single-DCU numerical validation for blended meson elementals."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import traceback
from pathlib import Path

import numpy as np


SEED = 20260822
N_EV = 1
N_ST = 2
GLOBAL_ZYX = (5, 6, 8)
OFFSET_ZYX = (1, 2, 3)
MOMENTUM_XYZ = (1.25, -0.5, 0.75)


def _omega(order, volume_color_dim):
    high_dim = volume_color_dim - N_EV
    return (high_dim - order) / (N_ST - order)


def _tuple_weight(indices, volume_color_dim):
    high_labels = {index for index in indices if index >= N_EV}
    weight = 1.0
    for order in range(len(high_labels)):
        weight *= _omega(order, volume_color_dim)
    return weight


def _phase(local_shape, *, sign=-1.0, offset=OFFSET_ZYX):
    nz, ny, nx = local_shape
    gnz, gny, gnx = GLOBAL_ZYX
    z0, y0, x0 = offset
    px, py, pz = MOMENTUM_XYZ
    result = np.empty(local_shape, dtype=np.complex128)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                angle = 2.0 * np.pi * (
                    px * (x + x0) / gnx
                    + py * (y + y0) / gny
                    + pz * (z + z0) / gnz
                )
                result[z, y, x] = np.exp(sign * 1j * angle)
    return result


def _meson_oracle(vectors, phase):
    n_basis, nt, nz, ny, nx, nc = vectors.shape
    volume_color_dim = int(np.prod(GLOBAL_ZYX) * nc)
    result = np.zeros((nt, n_basis, n_basis), dtype=np.complex128)
    for t in range(nt):
        for left in range(n_basis):
            for right in range(n_basis):
                value = 0.0j
                for z in range(nz):
                    for y in range(ny):
                        for x in range(nx):
                            for color in range(nc):
                                value += (
                                    vectors[left, t, z, y, x, color].conjugate()
                                    * phase[z, y, x]
                                    * vectors[right, t, z, y, x, color]
                                )
                result[t, left, right] = value * _tuple_weight(
                    (left, right), volume_color_dim
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

    import Def_blending as implementation

    rng = np.random.default_rng(SEED)
    vectors = (
        rng.standard_normal((N_EV + N_ST, 2, 2, 2, 3, 3))
        + 1j * rng.standard_normal((N_EV + N_ST, 2, 2, 2, 3, 3))
    ).astype(np.complex128)
    original_sha256 = hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()
    device_vectors = cp.asarray(vectors)
    actual = implementation.meson_elemental(
        device_vectors,
        MOMENTUM_XYZ,
        N_EV,
        N_ST,
        global_spatial_shape=GLOBAL_ZYX,
        local_spatial_offset=OFFSET_ZYX,
        spatial_comm=None,
        cuda_aware_mpi=False,
    )
    cp.cuda.get_current_stream().synchronize()
    if not isinstance(actual, cp.ndarray):
        raise AssertionError("meson_elemental did not return a CuPy array")
    expected = _meson_oracle(vectors, _phase(vectors.shape[2:5]))
    sign_mutant = _meson_oracle(vectors, _phase(vectors.shape[2:5], sign=1.0))
    offset_mutant = _meson_oracle(
        vectors, _phase(vectors.shape[2:5], offset=(0, 0, 0))
    )
    actual_host = cp.asnumpy(actual)
    error = _relative_max_error(actual_host, expected)
    sign_separation = _relative_max_error(sign_mutant, expected)
    offset_separation = _relative_max_error(offset_mutant, expected)
    tolerance = 5.0e-12
    if error > tolerance:
        raise AssertionError("blending elemental differential failed")
    if min(sign_separation, offset_separation) <= 10.0 * tolerance:
        raise AssertionError("Fourier sign/offset mutants are not discriminated")
    post_sha256 = hashlib.sha256(
        np.ascontiguousarray(cp.asnumpy(device_vectors)).tobytes()
    ).hexdigest()
    if post_sha256 != original_sha256:
        raise AssertionError("meson_elemental mutated its input basis")
    return {
        "schema_version": 1,
        "skill": "pyquda_blending",
        "status": "PASS",
        "seed": SEED,
        "input_sha256": original_sha256,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": _device_info(cp),
        "gpu_kernel_executed": True,
        "quda_calls": 0,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "relative_max_error": error,
        "tolerance": tolerance,
        "mutations": {
            "fourier_sign_flip_separation": sign_separation,
            "ignore_global_offset_separation": offset_separation,
        },
        "oracle": "independent host loops for global-coordinate phase, color-space sum, and distinct stochastic-label weights",
        "evidence_scope": "single-DCU deterministic meson elemental; no stochastic unbiasedness, dilution study, solve, MPI, or I/O",
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
            "skill": "pyquda_blending",
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
