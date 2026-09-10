#!/usr/bin/env python3
"""Single-DCU numerical validation for RI inversion and projection kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import traceback
from pathlib import Path

import numpy as np


SEED = 20260822


def _gamma_basis(dtype=np.complex128):
    sigma1 = np.array([[0, 1], [1, 0]], dtype=dtype)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=dtype)
    identity2 = np.eye(2, dtype=dtype)
    return [
        np.kron(sigma1, sigma1),
        np.kron(sigma1, sigma2),
        np.kron(sigma1, sigma3),
        np.kron(sigma2, identity2),
    ]


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


def _tree_level_check(implementation, cp):
    gammas_host = _gamma_basis()
    gamma5_host = gammas_host[3] @ gammas_host[2] @ gammas_host[1] @ gammas_host[0]
    gammas = [cp.asarray(value) for value in gammas_host]
    gamma5 = cp.asarray(gamma5_host)
    color_identity = np.eye(3, dtype=np.complex128)
    p_in = np.array([1.0, 0.0, 0.0, 0.0])
    p_out = np.array([0.0, 1.0, 0.0, 0.0])

    def spin_color(spin):
        return cp.asarray(_field_from_matrix(np.kron(spin, color_identity)))

    inverse_in = spin_color(1j * sum(component * gamma for component, gamma in zip(p_in, gammas_host)))
    inverse_out = spin_color(1j * sum(component * gamma for component, gamma in zip(p_out, gammas_host)))
    tensor_spin = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            tensor_spin.append(0.5 * (gammas_host[mu] @ gammas_host[nu] - gammas_host[nu] @ gammas_host[mu]))
    vertices = {
        "V": [spin_color(gamma) for gamma in gammas_host],
        "A": [spin_color(gamma @ gamma5_host) for gamma in gammas_host],
        "S": spin_color(np.eye(4, dtype=np.complex128)),
        "P": spin_color(gamma5_host),
        "T": [spin_color(value) for value in tensor_spin],
    }
    kinematics = implementation.validate_kinematics(
        p_in,
        p_out,
        2.0,
        projector_scheme="qslash",
        momentum_definition="continuum",
    )
    if not np.array_equal(kinematics.q, p_out - p_in):
        raise AssertionError("kinematics.q does not equal p_out-p_in")
    definitions = {}
    for definition in ("incoming", "outgoing", "arithmetic_mean", "geometric_mean"):
        constants = implementation.compute_ri_constants(
            kinematics,
            inverse_in,
            inverse_out,
            vertices,
            gammas,
            gamma5,
            zq_definition=definition,
        )
        values = {
            name: complex(cp.asnumpy(getattr(constants, name)).item())
            for name in ("Zq", "ZA", "ZV", "ZS", "ZP", "ZT")
        }
        error = max(abs(value - 1.0) for value in values.values())
        if error > 5.0e-12:
            raise AssertionError("tree-level {} error {}".format(definition, error))
        if constants.continuum_matching_status != "unmapped_fail_closed":
            raise AssertionError("custom RI result did not remain fail-closed")
        definitions[definition] = {
            "max_abs_error_from_one": float(error),
            "scheme_identity": constants.scheme_identity,
        }
    return definitions


def run_validation():
    import cupy as cp

    import Def_ri_renorm as implementation

    rng = np.random.default_rng(SEED)
    random_blocks = (
        rng.standard_normal((3, 2, 12, 12))
        + 1j * rng.standard_normal((3, 2, 12, 12))
    )
    checks = []
    for dtype_name, tolerance in (("complex64", 5.0e-5), ("complex128", 5.0e-12)):
        dtype = np.dtype(dtype_name)
        perturbation = random_blocks.astype(dtype)
        identity = np.eye(12, dtype=dtype)
        propagator_out = 4.0 * identity + 0.07 * perturbation[0]
        propagator_in = 5.0 * identity + 0.05 * perturbation[1]
        green = 0.2 * identity + 0.03 * perturbation[2]
        field_out = _field_from_matrix(propagator_out)
        field_in = _field_from_matrix(propagator_in)
        field_green = _field_from_matrix(green)
        device_out = cp.asarray(field_out)
        device_in = cp.asarray(field_in)
        device_green = cp.asarray(field_green)
        actual_inverse = implementation.inverse_propagator(device_out)
        actual_amputated = implementation.amputate_vertex(
            device_out, device_green, device_in
        )
        cp.cuda.get_current_stream().synchronize()
        if not isinstance(actual_amputated, cp.ndarray):
            raise AssertionError("amputation did not return a CuPy array")
        expected_inverse = np.linalg.inv(propagator_out)
        expected_amputated = (
            np.linalg.inv(propagator_out) @ green @ np.linalg.inv(propagator_in)
        )
        swapped_mutant = (
            np.linalg.inv(propagator_in) @ green @ np.linalg.inv(propagator_out)
        )
        inverse_host = _matrix_from_field(cp.asnumpy(actual_inverse))
        amputated_host = _matrix_from_field(cp.asnumpy(actual_amputated))
        inverse_error = _relative_max_error(inverse_host, expected_inverse)
        amputation_error = _relative_max_error(amputated_host, expected_amputated)
        mutant_separation = _relative_max_error(swapped_mutant, expected_amputated)
        residual = _relative_max_error(
            propagator_out @ inverse_host,
            np.broadcast_to(identity, propagator_out.shape),
        )
        if max(inverse_error, amputation_error) > tolerance:
            raise AssertionError("{} inverse/amputation differential failed".format(dtype_name))
        if mutant_separation <= 10.0 * tolerance:
            raise AssertionError("swapped-leg mutant is not discriminated")
        condition_number = float(max(np.linalg.cond(block) for block in propagator_out))
        residual_bound = 200.0 * np.finfo(dtype).eps * condition_number
        if residual > residual_bound:
            raise AssertionError("inverse residual exceeds conditioning-aware bound")
        checks.append(
            {
                "dtype": dtype_name,
                "shape": list(actual_amputated.shape),
                "inverse_relative_max_error": inverse_error,
                "amputation_relative_max_error": amputation_error,
                "forward_residual": residual,
                "residual_bound": float(residual_bound),
                "condition_number_max": condition_number,
                "tolerance": tolerance,
                "mutant": "swap incoming and outgoing amputation legs",
                "mutant_separation": mutant_separation,
            }
        )
    tree_level = _tree_level_check(implementation, cp)
    cp.cuda.get_current_stream().synchronize()
    return {
        "schema_version": 1,
        "skill": "pyquda_ri_renorm",
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
        "oracle": "independent compound-index NumPy inverse/amputation and analytic tree vertices",
        "checks": checks,
        "tree_level": tree_level,
        "evidence_scope": "single-DCU synthetic raw RI algebra; no gauge fixing, QUDA solve, MPI, or continuum matching",
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
            "skill": "pyquda_ri_renorm",
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
