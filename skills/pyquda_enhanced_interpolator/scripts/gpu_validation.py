#!/usr/bin/env python3
"""Single-DCU numerical validation for enhanced baryon contractions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import traceback
from pathlib import Path

import numpy as np


SEED = 20260822
EPSILON_TERMS = (
    ((0, 1, 2), 1.0),
    ((1, 2, 0), 1.0),
    ((2, 0, 1), 1.0),
    ((0, 2, 1), -1.0),
    ((2, 1, 0), -1.0),
    ((1, 0, 2), -1.0),
)


def _gamma_basis(dtype=np.complex128):
    sigma1 = np.array([[0, 1], [1, 0]], dtype=dtype)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=dtype)
    identity2 = np.eye(2, dtype=dtype)
    return (
        np.kron(sigma1, sigma1),
        np.kron(sigma1, sigma2),
        np.kron(sigma1, sigma3),
        np.kron(sigma2, identity2),
    )


def _baryon_oracle(u1, u2, d, projector, diquark_sink, diquark_source):
    direct = 0.0j
    exchange = 0.0j
    for (colors_sink, epsilon_sink), (colors_source, epsilon_source) in itertools.product(
        EPSILON_TERMS, repeat=2
    ):
        a, b, c = colors_sink
        color_a, color_b, color_c = colors_source
        color_sign = epsilon_sink * epsilon_source
        for i, source_i, j, k, source_j, source_k in itertools.product(
            range(4), repeat=6
        ):
            spin_weight = (
                projector[i, source_i]
                * diquark_sink[j, k]
                * diquark_source[source_j, source_k]
            )
            common = color_sign * spin_weight * d[k, source_k, c, color_c]
            direct += (
                common
                * u1[i, source_i, a, color_a]
                * u2[j, source_j, b, color_b]
            )
            exchange += (
                common
                * u1[i, source_j, a, color_b]
                * u2[j, source_i, b, color_a]
            )
    return direct, exchange


def _relative_error(actual, expected):
    return float(abs(actual - expected) / max(1.0, abs(expected)))


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

    import Def_enhanced_interpolator as implementation

    rng = np.random.default_rng(SEED)
    propagators = (
        rng.standard_normal((3, 4, 4, 3, 3))
        + 1j * rng.standard_normal((3, 4, 4, 3, 3))
    ).astype(np.complex128)
    projector = (
        rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    ).astype(np.complex128)
    diquark_sink = (
        rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    ).astype(np.complex128)
    gamma_x, gamma_y, gamma_z, gamma_t = _gamma_basis()
    gamma5 = gamma_t @ gamma_z @ gamma_y @ gamma_x

    device_gamma_t = cp.asarray(gamma_t)
    device_diquark_sink = cp.asarray(diquark_sink)
    device_diquark_source = implementation.euclidean_source_conjugate(
        device_diquark_sink, device_gamma_t
    )
    cp.cuda.get_current_stream().synchronize()
    expected_source = gamma_t @ diquark_sink.conj() @ gamma_t
    source_error = _relative_max_error(
        cp.asnumpy(device_diquark_source), expected_source
    )
    adjoint_mutant = gamma_t @ diquark_sink.conj().T @ gamma_t
    source_mutant_separation = _relative_max_error(adjoint_mutant, expected_source)
    if source_error > 5.0e-12:
        raise AssertionError("Euclidean source conjugation differential failed")
    if source_mutant_separation <= 5.0e-11:
        raise AssertionError("source-adjoint mutant is not discriminated")

    actual = implementation.baryon_contraction(
        cp.asarray(propagators[0]),
        cp.asarray(propagators[1]),
        cp.asarray(propagators[2]),
        cp.asarray(projector),
        device_diquark_sink,
        device_diquark_source,
    )
    cp.cuda.get_current_stream().synchronize()
    if not isinstance(actual, cp.ndarray):
        raise AssertionError("baryon contraction did not return a CuPy scalar")
    direct, exchange = _baryon_oracle(
        propagators[0],
        propagators[1],
        propagators[2],
        projector,
        diquark_sink,
        expected_source,
    )
    expected = direct - exchange
    actual_host = complex(cp.asnumpy(actual).item())
    contraction_error = _relative_error(actual_host, expected)
    plus_mutant = direct + exchange
    exchange_mutant_separation = _relative_error(plus_mutant, expected)
    if contraction_error > 5.0e-11:
        raise AssertionError("explicit baryon Wick-loop differential failed")
    if exchange_mutant_separation <= 5.0e-10:
        raise AssertionError("direct-plus-exchange mutant is not discriminated")

    gamma_plus, gamma_minus = implementation.lightcone_gammas(
        cp.asarray(gamma_t), cp.asarray(gamma_z)
    )
    projector_plus = implementation.plus_quark_projector(gamma_plus, gamma_minus)
    gamma_plus_squared = gamma_plus @ gamma_plus
    gamma_minus_squared = gamma_minus @ gamma_minus
    lightcone_anticommutator = (
        gamma_plus @ gamma_minus + gamma_minus @ gamma_plus
    )
    projector_squared = projector_plus @ projector_plus
    cp.cuda.get_current_stream().synchronize()
    identity = np.eye(4, dtype=np.complex128)
    nilpotency_error = max(
        _relative_max_error(cp.asnumpy(gamma_plus_squared), np.zeros((4, 4))),
        _relative_max_error(cp.asnumpy(gamma_minus_squared), np.zeros((4, 4))),
    )
    anticommutator_error = _relative_max_error(
        cp.asnumpy(lightcone_anticommutator),
        2.0 * identity,
    )
    projector_expected = (
        (gamma_t - 1j * gamma_z)
        @ (gamma_t + 1j * gamma_z)
        / (2.0 * np.sqrt(2.0))
    )
    projector_error = _relative_max_error(
        cp.asnumpy(projector_plus),
        projector_expected,
    )
    scaled_projector_error = _relative_max_error(
        cp.asnumpy(projector_squared),
        np.sqrt(2.0) * projector_expected,
    )
    normalization_mutant = (
        (gamma_t - 1j * gamma_z)
        @ (gamma_t + 1j * gamma_z)
        / np.sqrt(2.0)
    )
    order_mutant = (
        (gamma_t + 1j * gamma_z)
        @ (gamma_t - 1j * gamma_z)
        / (2.0 * np.sqrt(2.0))
    )
    normalization_mutant_separation = _relative_max_error(
        normalization_mutant, projector_expected
    )
    order_mutant_separation = _relative_max_error(order_mutant, projector_expected)
    if max(
        nilpotency_error,
        anticommutator_error,
        projector_error,
        scaled_projector_error,
    ) > 5.0e-12:
        raise AssertionError("lightcone Clifford/projector identities failed")
    if min(normalization_mutant_separation, order_mutant_separation) <= 5.0e-11:
        raise AssertionError("lightcone projector mutant is not discriminated")

    checksum_payload = np.concatenate(
        [propagators.reshape(-1), projector.reshape(-1), diquark_sink.reshape(-1)]
    )
    return {
        "schema_version": 1,
        "skill": "pyquda_enhanced_interpolator",
        "status": "PASS",
        "seed": SEED,
        "input_sha256": hashlib.sha256(
            np.ascontiguousarray(checksum_payload).tobytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": _device_info(cp),
        "gpu_kernel_executed": True,
        "quda_calls": 0,
        "source_conjugation_relative_max_error": source_error,
        "baryon_contraction_relative_error": contraction_error,
        "lightcone_nilpotency_relative_max_error": nilpotency_error,
        "lightcone_anticommutator_relative_max_error": anticommutator_error,
        "projector_relative_max_error": projector_error,
        "scaled_projector_relative_max_error": scaled_projector_error,
        "mutations": {
            "source_conjugate_to_adjoint_separation": source_mutant_separation,
            "direct_minus_to_plus_exchange_separation": exchange_mutant_separation,
            "missing_lightcone_normalization_separation": normalization_mutant_separation,
            "gamma_plus_minus_order_swap_separation": order_mutant_separation,
        },
        "oracle": "independent host spin-color Wick loops plus analytic Euclidean conjugation and Clifford identities",
        "evidence_scope": "single-DCU local contraction algebra; no inversion, MPI, interacting overlap, SNR, plateau, or fit claim",
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
            "skill": "pyquda_enhanced_interpolator",
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
