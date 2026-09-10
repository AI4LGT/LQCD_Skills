#!/usr/bin/env python3
"""Single-DCU differential validation for gluon Fourier kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import traceback
from pathlib import Path

import numpy as np


SEED = 20260822
LATTICE_XYZT = (4, 6, 8, 10)
MOMENTUM_XYZT = (1, -2, 3, 4)


def _resource_path():
    raw = os.environ.get("PYQUDA_RESOURCE_PATH")
    if not raw:
        raise RuntimeError("PYQUDA_RESOURCE_PATH must point inside LQCD_Master")
    path = Path(raw).resolve()
    repository = Path(__file__).resolve().parents[3]
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise RuntimeError("QUDA resource path lies outside writable LQCD_Master") from error
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkerboard_from_lexico(lexico):
    directions, lt, lz, ly, lx, nc, _ = lexico.shape
    checkerboard = np.empty(
        (directions, 2, lt, lz, ly, lx // 2, nc, nc), dtype=lexico.dtype
    )
    for t in range(lt):
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    parity = (t + z + y + x) & 1
                    checkerboard[:, parity, t, z, y, x // 2] = lexico[:, t, z, y, x]
    return checkerboard


def _direct_dft(lexico, momentum, *, common_half_phase=False):
    px, py, pz, pt = momentum
    _, lt, lz, ly, lx, _, _ = lexico.shape
    result = np.zeros(lexico.shape[:1] + lexico.shape[-2:], dtype=np.complex128)
    for t in range(lt):
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    phase = np.exp(
                        -2j
                        * np.pi
                        * (px * x / lx + py * y / ly + pz * z / lz + pt * t / lt)
                    )
                    result += lexico[:, t, z, y, x] * phase
    result /= lx * ly * lz * lt
    fractions = np.asarray([px / lx, py / ly, pz / lz, pt / lt], dtype=float)
    if common_half_phase:
        result *= np.exp(-1j * np.pi * np.sum(fractions))
    else:
        result *= np.exp(-1j * np.pi * fractions)[:, None, None]
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
    from pyquda_utils import core

    resource_path = _resource_path()
    core.init([1, 1, 1, 1], list(LATTICE_XYZT), resource_path=str(resource_path))
    import Def_gluon_renorm as implementation

    if core.getMPISize() != 1:
        raise AssertionError("minimal gluon validation requires exactly one MPI rank")
    latt_info = core.LatticeInfo(list(LATTICE_XYZT))
    lx, ly, lz, lt = LATTICE_XYZT
    rng = np.random.default_rng(SEED)
    amplitude = (
        rng.standard_normal((4, 3, 3)) + 1j * rng.standard_normal((4, 3, 3))
    ).astype(np.complex128)
    lexico = np.empty((4, lt, lz, ly, lx, 3, 3), dtype=np.complex128)
    px, py, pz, pt = MOMENTUM_XYZT
    noise = (
        rng.standard_normal(lexico.shape) + 1j * rng.standard_normal(lexico.shape)
    ).astype(np.complex128)
    for t in range(lt):
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    source_phase = np.exp(
                        2j
                        * np.pi
                        * (px * x / lx + py * y / ly + pz * z / lz + pt * t / lt)
                    )
                    lexico[:, t, z, y, x] = amplitude * source_phase
    lexico += 0.01 * noise
    checkerboard = _checkerboard_from_lexico(lexico)
    input_sha256 = hashlib.sha256(
        np.ascontiguousarray(checkerboard).tobytes()
    ).hexdigest()
    device_checkerboard = cp.asarray(checkerboard)

    one_momentum = implementation.FT_Gauge_1mom(
        device_checkerboard, list(MOMENTUM_XYZT), latt_info, half_flag=1
    )
    all_momenta = implementation.FFT_Gauge_Allmom_MPI(
        device_checkerboard,
        latt_info,
        core.getMPIComm(),
        half_flag=1,
    )
    cp.cuda.get_current_stream().synchronize()
    if not isinstance(all_momenta, cp.ndarray):
        raise AssertionError("all-momentum FFT did not return a CuPy array")
    selected_fft = cp.asnumpy(
        all_momenta[
            :,
            pt % lt,
            pz % lz,
            py % ly,
            px % lx,
        ]
    )
    expected = _direct_dft(lexico, MOMENTUM_XYZT)
    common_phase_mutant = _direct_dft(
        lexico, MOMENTUM_XYZT, common_half_phase=True
    )
    one_momentum_error = _relative_max_error(np.asarray(one_momentum), expected)
    all_momentum_error = _relative_max_error(selected_fft, expected)
    mutant_separation = _relative_max_error(common_phase_mutant, expected)
    tolerance = 5.0e-12
    if max(one_momentum_error, all_momentum_error) > tolerance:
        raise AssertionError("gluon direct-DFT differential failed")
    if mutant_separation <= 10.0 * tolerance:
        raise AssertionError("common half-link phase mutant is not discriminated")
    post_sha256 = hashlib.sha256(
        np.ascontiguousarray(cp.asnumpy(device_checkerboard)).tobytes()
    ).hexdigest()
    if post_sha256 != input_sha256:
        raise AssertionError("Fourier kernels mutated the input gauge field")
    return {
        "schema_version": 1,
        "skill": "pyquda_gluon_renorm",
        "status": "PASS",
        "seed": SEED,
        "input_sha256": input_sha256,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": _device_info(cp),
        "mpi_size": int(core.getMPISize()),
        "quda_runtime_initialized": True,
        "quda_kernel_calls": 0,
        "gpu_kernel_executed": True,
        "lattice_xyzt": list(LATTICE_XYZT),
        "momentum_xyzt": list(MOMENTUM_XYZT),
        "one_momentum_relative_max_error": one_momentum_error,
        "all_momentum_relative_max_error": all_momentum_error,
        "tolerance": tolerance,
        "mutant": "one common half-link phase for all four directions",
        "mutant_separation": mutant_separation,
        "oracle": "independent checkerboard map and explicit host DFT with direction-specific link-center phases",
        "evidence_scope": "single-rank single-DCU synthetic Fourier kernels; no distributed FFT, interacting gauge EMT, mixing, or renormalization claim",
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
            "skill": "pyquda_gluon_renorm",
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
