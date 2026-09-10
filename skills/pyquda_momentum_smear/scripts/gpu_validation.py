#!/usr/bin/env python3
"""Single-DCU QUDA validation for link-phased Wuppertal smearing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import traceback
from pathlib import Path

import numpy as np


LATTICE_XYZT = (4, 6, 8, 10)
PLANE_WAVE_Q = (1, 1, 2)
SMEAR_MODE_K = (0.25, -0.5, 0.75)
RHO = 0.5
N_STEPS = 1
FOURIER_MOMENTUM = (1, -2, 3)
SOURCE_XYZT = (1, 2, 3, 1)


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


def _plane_wave_field():
    lx, ly, lz, lt = LATTICE_XYZT
    qx, qy, qz = PLANE_WAVE_Q
    field = np.zeros((lt, lz, ly, lx, 4, 3), dtype=np.complex128)
    for t in range(lt):
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    angle = 2.0 * np.pi * (qx * x / lx + qy * y / ly + qz * z / lz)
                    field[t, z, y, x, 0, 0] = np.exp(1j * angle)
    return field


def _checkerboard_phase(momentum, source, sign):
    lx, ly, lz, lt = LATTICE_XYZT
    px, py, pz = momentum
    x0, y0, z0, _ = source
    result = np.empty((2, lt, lz, ly, lx // 2), dtype=np.complex128)
    for t in range(lt):
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    parity = (t + z + y + x) & 1
                    angle = 2.0 * np.pi * (
                        px * (x - x0) / lx
                        + py * (y - y0) / ly
                        + pz * (z - z0) / lz
                    )
                    result[parity, t, z, y, x // 2] = np.exp(sign * 1j * angle)
    return result


def _wuppertal_eigenvalue(mode):
    lengths = LATTICE_XYZT[:3]
    alpha = 1.0 / (4.0 * N_STEPS / RHO**2 - 6.0)
    angles = [
        2.0 * np.pi * (q_component + k_component) / length
        for q_component, k_component, length in zip(PLANE_WAVE_Q, mode, lengths)
    ]
    return 1.0 + 2.0 * alpha * sum(np.cos(angle) for angle in angles)


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
    import Def_momentum_smear as implementation

    if core.getMPISize() != 1:
        raise AssertionError("minimal momentum validation requires one MPI rank")
    latt_info = core.LatticeInfo(list(LATTICE_XYZT), 1, 1.0)
    gauge = core.LatticeGauge(latt_info)
    plane_wave = _plane_wave_field()
    input_sha256 = hashlib.sha256(
        np.ascontiguousarray(plane_wave).tobytes()
    ).hexdigest()
    field = core.LatticeFermion(
        latt_info, cp.asarray(latt_info.evenodd(plane_wave, False))
    )

    output_zero = implementation.momentum_smear_kernel(
        field, gauge, latt_info, (0.0, 0.0, 0.0), RHO, N_STEPS
    )
    output_mode = implementation.momentum_smear_kernel(
        field, gauge, latt_info, SMEAR_MODE_K, RHO, N_STEPS
    )
    cp.cuda.get_current_stream().synchronize()
    zero_lexico = output_zero.lexico(False)[..., 0, 0]
    mode_lexico = output_mode.lexico(False)[..., 0, 0]
    plane_device = cp.asarray(plane_wave[..., 0, 0])
    ratio_zero_sites = zero_lexico / plane_device
    ratio_mode_sites = mode_lexico / plane_device
    ratio_zero = complex(cp.asnumpy(cp.mean(ratio_zero_sites)).item())
    ratio_mode = complex(cp.asnumpy(cp.mean(ratio_mode_sites)).item())
    spread = max(
        float(cp.asnumpy(cp.max(cp.abs(ratio_zero_sites - ratio_zero))).item()),
        float(cp.asnumpy(cp.max(cp.abs(ratio_mode_sites - ratio_mode))).item()),
    )
    observed_ratio = ratio_mode / ratio_zero
    expected_ratio = _wuppertal_eigenvalue(SMEAR_MODE_K) / _wuppertal_eigenvalue((0.0, 0.0, 0.0))
    sign_mutant_ratio = _wuppertal_eigenvalue(tuple(-value for value in SMEAR_MODE_K)) / _wuppertal_eigenvalue((0.0, 0.0, 0.0))
    eigenvalue_error = abs(observed_ratio - expected_ratio) / max(1.0, abs(expected_ratio))
    mutant_separation = abs(sign_mutant_ratio - expected_ratio) / max(1.0, abs(expected_ratio))
    tolerance = 3.0e-5
    if spread > tolerance:
        raise AssertionError("smeared plane wave is not an eigenvector")
    if eigenvalue_error > tolerance:
        raise AssertionError("link-phased Wuppertal eigenvalue ratio failed")
    if mutant_separation <= 10.0 * tolerance:
        raise AssertionError("link-phase sign mutant is not discriminated")

    phased_gauge = implementation.phase_spatial_links(gauge, SMEAR_MODE_K)
    link_errors = []
    for direction, (mode, length) in enumerate(zip(SMEAR_MODE_K, LATTICE_XYZT[:3])):
        expected_link = gauge.data[direction] * np.exp(2j * np.pi * mode / length)
        link_errors.append(
            float(cp.asnumpy(cp.max(cp.abs(phased_gauge.data[direction] - expected_link))).item())
        )
    time_link_error = float(
        cp.asnumpy(cp.max(cp.abs(phased_gauge.data[3] - gauge.data[3]))).item()
    )
    if max(link_errors + [time_link_error]) > 5.0e-12:
        raise AssertionError("spatial/time link phase ownership failed")

    sink_phase, sequential_phase = implementation.fourier_phase_pair(
        latt_info, FOURIER_MOMENTUM, SOURCE_XYZT
    )
    cp.cuda.get_current_stream().synchronize()
    if not isinstance(sink_phase, cp.ndarray) or not isinstance(sequential_phase, cp.ndarray):
        raise AssertionError("fourier_phase_pair did not return device arrays")
    expected_sink = _checkerboard_phase(FOURIER_MOMENTUM, SOURCE_XYZT, -1.0)
    expected_sequential = _checkerboard_phase(
        FOURIER_MOMENTUM, SOURCE_XYZT, 1.0
    )
    sink_error = _relative_max_error(cp.asnumpy(sink_phase), expected_sink)
    sequential_error = _relative_max_error(
        cp.asnumpy(sequential_phase), expected_sequential
    )
    phase_swap_separation = _relative_max_error(expected_sink, expected_sequential)
    if max(sink_error, sequential_error) > 5.0e-12:
        raise AssertionError("Fourier phase pair differential failed")
    if phase_swap_separation <= 5.0e-11:
        raise AssertionError("sink/sequential phase swap is not discriminated")
    post_sha256 = hashlib.sha256(
        np.ascontiguousarray(cp.asnumpy(field.lexico(False))).tobytes()
    ).hexdigest()
    if post_sha256 != input_sha256:
        raise AssertionError("momentum smearing mutated the source field")
    return {
        "schema_version": 1,
        "skill": "pyquda_momentum_smear",
        "status": "PASS",
        "input_sha256": input_sha256,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": _device_info(cp),
        "mpi_size": int(core.getMPISize()),
        "gpu_kernel_executed": True,
        "quda_runtime_initialized": True,
        "quda_wuppertal_calls": 2,
        "lattice_xyzt": list(LATTICE_XYZT),
        "plane_wave_q": list(PLANE_WAVE_Q),
        "smear_mode_k": list(SMEAR_MODE_K),
        "rho": RHO,
        "n_steps": N_STEPS,
        "observed_eigenvalue_ratio": [observed_ratio.real, observed_ratio.imag],
        "expected_eigenvalue_ratio": expected_ratio,
        "eigenvalue_relative_error": float(eigenvalue_error),
        "plane_wave_ratio_spread": spread,
        "tolerance": tolerance,
        "link_phase_max_errors": link_errors,
        "time_link_max_error": time_link_error,
        "fourier_sink_relative_max_error": sink_error,
        "fourier_sequential_relative_max_error": sequential_error,
        "mutations": {
            "link_phase_sign_flip_separation": float(mutant_separation),
            "sink_sequential_swap_separation": phase_swap_separation,
        },
        "phase_v2_boundary": "target PyQUDA builds phase on host then transfers to configured backend",
        "oracle": "normalization-free free-field Wuppertal eigenvalue ratio and independent checkerboard coordinate phases",
        "evidence_scope": "single-DCU unit-gauge one-step kernel and phase helper; no inversion, true residual, MPI, interacting 2pt/3pt, or downstream dagger closure",
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
            "skill": "pyquda_momentum_smear",
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
