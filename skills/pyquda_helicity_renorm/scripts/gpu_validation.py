#!/usr/bin/env python3
"""Cluster-host validation for the intentionally CPU helicity RG kernels."""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from pathlib import Path

import numpy as np


def _relative_max_error(actual, expected):
    scale = max(1.0, float(np.max(np.abs(expected))))
    return float(np.max(np.abs(actual - expected)) / scale)


def run_validation():
    import scipy

    import Def_helicity_renorm as implementation

    mu = 3.0
    alpha_five_loop = float(implementation.alpha_s(mu, 3)[4])
    expansion = alpha_five_loop / (4.0 * np.pi)
    matching = np.asarray(
        implementation.Helicity_MatchingCoeff_tmp(mu, mu, 1, 0), dtype=float
    ).reshape(2, 2)
    expected_r12 = 8.0 * expansion
    expected_r21 = -6.0 * expansion
    matching_error = max(
        abs(matching[0, 1] - expected_r12),
        abs(matching[1, 0] - expected_r21),
    )
    if matching_error > 5.0e-14:
        raise AssertionError("one-loop off-diagonal checkpoint failed")
    swapped_mutant = np.array([expected_r21, expected_r12])
    correct_pair = np.array([expected_r12, expected_r21])
    mutant_separation = _relative_max_error(swapped_mutant, correct_pair)
    if mutant_separation <= 1.0e-4:
        raise AssertionError("R12/R21 swap mutant is not discriminated")

    calculator = implementation.Helicity_RunningFactorCalculator(
        nf=3, ope_type="ghelicity"
    )
    identity = np.eye(2)
    equal_scale = calculator.run(2.5, 2.5)
    equal_scale_error = _relative_max_error(equal_scale, identity)
    if equal_scale_error > 5.0e-12:
        raise AssertionError("equal-scale running is not identity")
    running_2_to_3 = calculator.run(2.0, 3.0)
    running_3_to_4 = calculator.run(3.0, 4.0)
    running_2_to_4 = calculator.run(2.0, 4.0)
    composition = running_3_to_4 @ running_2_to_3
    composition_error = _relative_max_error(composition, running_2_to_4)
    # Each segment reinitializes the truncated beta ODE from the external
    # five-loop alpha_s value, so this is a bounded diagnostic rather than an
    # exact group-property oracle.
    if composition_error > 5.0e-4:
        raise AssertionError("RG segmented-composition diagnostic is unstable")
    noncommuting_initial = np.array([[1.1, 0.3], [-0.2, 0.9]])
    running_with_initial = calculator.run(2.0, 3.0, noncommuting_initial)
    initial_basis_error = _relative_max_error(running_with_initial, running_2_to_3)
    if initial_basis_error > 2.0e-8:
        raise AssertionError("left-acting RG evolution depends on R_init basis")
    step = 1.0e-5
    step_mu = 2.5
    short_running = calculator.run(step_mu, step_mu * np.exp(step / 2.0))
    derivative = (short_running - identity) / step
    local_a_s = float(implementation.alpha_s(step_mu, 3)[4]) / np.pi
    expected_derivative = calculator._get_gamma_matrix(local_a_s)
    derivative_error = _relative_max_error(derivative, expected_derivative)
    if derivative_error > 2.0e-6:
        raise AssertionError("RG Gamma@R orientation derivative failed")

    return {
        "schema_version": 1,
        "skill": "pyquda_helicity_renorm",
        "status": "PASS",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "gpu_applicable": False,
        "gpu_kernel_executed": False,
        "quda_calls": 0,
        "matching": {
            "muR": mu,
            "mu_scale": mu,
            "alpha_s_five_loop": alpha_five_loop,
            "expansion_alpha_over_4pi": expansion,
            "observed_R12": float(matching[0, 1]),
            "expected_R12": expected_r12,
            "observed_R21": float(matching[1, 0]),
            "expected_R21": expected_r21,
            "max_abs_error": float(matching_error),
            "mutant": "swap R12 and R21",
            "mutant_separation": mutant_separation,
        },
        "running": {
            "equal_scale_relative_max_error": equal_scale_error,
            "composition_relative_max_error": composition_error,
            "composition_semantics": "bounded diagnostic; each segment reinitializes alpha_s",
            "noncommuting_initial_basis_error": initial_basis_error,
            "small_step_gamma_derivative_error": derivative_error,
        },
        "oracle": "independent one-loop R12=8a/R21=-6a, identity, noncommuting-basis invariance, and small-step Gamma@R derivative",
        "evidence_scope": "target cluster CPU/SciPy algebra only; no GPU implementation and no coefficient/basis/scheme provenance promotion",
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
            "skill": "pyquda_helicity_renorm",
            "status": "FAIL",
            "gpu_applicable": False,
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
