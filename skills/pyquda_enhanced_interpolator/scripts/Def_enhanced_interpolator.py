"""Kinematically enhanced meson and baryon interpolator contractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _xp(array: Any):
    if type(array).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return np


def _require_same_backend(reference: Any, *arrays: Any) -> None:
    xp = _xp(reference)
    if any(_xp(array) is not xp for array in arrays):
        raise TypeError("all propagators and spin kernels must use one backend")


def _require_spin_matrices(reference: Any, *matrices: Any) -> None:
    """Require one backend and the four-dimensional Dirac spin space."""

    _require_same_backend(reference, *matrices)
    xp = _xp(reference)
    for matrix in (reference, *matrices):
        if getattr(matrix, "shape", None) != (4, 4):
            raise ValueError("all Dirac spin matrices must have exact shape (4,4)")
        try:
            finite = xp.isfinite(matrix).all()
        except TypeError as error:
            raise TypeError("all Dirac spin matrices must be numerical") from error
        if xp is np:
            is_finite = bool(np.asarray(finite).item())
        else:
            is_finite = bool(np.asarray(xp.asnumpy(finite)).item())
        if not is_finite:
            raise ValueError("all Dirac spin matrices must be finite")


@dataclass(frozen=True)
class EnhancedKernels:
    gamma_plus: Any
    gamma_minus: Any
    quark_plus: Any
    projector: Any
    diquark_sink: Any
    diquark_source_bar: Any

    @property
    def diquark(self) -> Any:
        """Backward-compatible alias for the sink diquark kernel."""

        return self.diquark_sink


def euclidean_adjoint(matrix: Any, gamma_t: Any) -> Any:
    """Return gamma_t M^dagger gamma_t for a Euclidean bilinear."""

    _require_spin_matrices(matrix, gamma_t)
    return gamma_t @ matrix.conj().T @ gamma_t


def euclidean_source_conjugate(matrix: Any, gamma_t: Any) -> Any:
    """Return the baryon-source kernel `gamma_t M* gamma_t`.

    For the diquark factor in the nucleon contraction the source operation is
    complex conjugation, not the bilinear Hermitian adjoint.  Keeping this as
    a separate operation prevents an accidental transpose of spin indices.
    """

    _require_spin_matrices(matrix, gamma_t)
    return gamma_t @ matrix.conj() @ gamma_t


def lightcone_gammas(gamma_t: Any, gamma_parallel: Any) -> tuple[Any, Any]:
    """Construct gamma_+ and gamma_- for the chosen boost direction.

    arXiv:2606.02447v2, Section 2.1 before Eq. (2.2), defines
    gamma_+/-=(gamma_t^M -/+ gamma_z^M)/sqrt(2). With the paper's Euclidean
    mapping gamma_parallel^E=i gamma_parallel^M this becomes
    gamma_+/-=(gamma_t^E +/- i gamma_parallel^E)/sqrt(2).
    """

    _require_spin_matrices(gamma_t, gamma_parallel)
    gamma_plus = (gamma_t + 1j * gamma_parallel) / np.sqrt(2.0)
    gamma_minus = (gamma_t - 1j * gamma_parallel) / np.sqrt(2.0)
    return gamma_plus, gamma_minus


def plus_quark_projector(gamma_plus: Any, gamma_minus: Any) -> Any:
    """Return Q_+=gamma_- gamma_+ / sqrt(2), Section 2.1 before Eq. (2.2)."""

    _require_spin_matrices(gamma_plus, gamma_minus)
    return gamma_minus @ gamma_plus / np.sqrt(2.0)


def enhanced_meson_kernel(
    bilinear_gamma: Any,
    quark_projector: Any,
    gamma_t: Any,
    antiquark_projector: Any | None = None,
) -> Any:
    """Project a Dirac-bar meson kernel on both quark legs.

    ``bilinear_gamma`` is the matrix in ``bar(q) Gamma q``.  It is *not* the
    matrix between ``q^dagger`` and ``q``.  This generic two-leg projection is
    an implementation-level algebraic extension; callers remain responsible
    for selecting a paper-supported operator.  In particular, the canonical
    projected-pion identity starts from ``q^dagger gamma5 q`` and therefore
    passes ``Gamma = gamma_t @ gamma5`` to this Dirac-bar API.
    """

    left_q = quark_projector if antiquark_projector is None else antiquark_projector
    _require_spin_matrices(bilinear_gamma, quark_projector, gamma_t, left_q)
    right = quark_projector
    left = euclidean_adjoint(left_q, gamma_t)
    return left @ bilinear_gamma @ right


def enhanced_pion_kernel(
    gamma5: Any,
    quark_projector: Any,
    gamma_t: Any,
) -> Any:
    """Return the canonical projected-pion Dirac-bar kernel.

    The projected-field relation in arXiv:2501.00729 is written as
    ``u_+^dagger gamma5 d_+``.  Since :func:`enhanced_meson_kernel` accepts the
    matrix in ``bar(u) Gamma d`` and ``bar(u)=u^dagger gamma_t``, the required
    input is ``Gamma=gamma_t gamma5``.  Keeping this wrapper separate prevents
    the tempting but identically zero ``Gamma=gamma5`` call.
    """

    _require_spin_matrices(gamma5, quark_projector, gamma_t)
    return enhanced_meson_kernel(
        gamma_t @ gamma5,
        quark_projector,
        gamma_t,
    )


def enhanced_baryon_kernels(
    charge_conjugation: Any,
    gamma5: Any,
    gamma_t: Any,
    gamma_parallel: Any,
    projector_kind: str = "plus",
    diquark_kind: str = "plus",
) -> EnhancedKernels:
    """Construct the nucleon projector T and Gamma'=C gamma5 Gamma.

    arXiv:2606.02447v2 Eqs. (2.6), (2.9)-(2.11) and Section 2.2 use
    Gamma in {1,gamma_t,gamma_+} and
    T in {P_+=(1+gamma_t)/2,gamma_t,gamma_+}.
    """

    _require_spin_matrices(
        charge_conjugation, gamma5, gamma_t, gamma_parallel
    )
    xp = _xp(gamma_t)
    gamma_plus, gamma_minus = lightcone_gammas(gamma_t, gamma_parallel)
    quark_plus = plus_quark_projector(gamma_plus, gamma_minus)
    identity = xp.eye(gamma_t.shape[0], dtype=gamma_t.dtype)
    choices = {"identity": identity, "time": gamma_t, "plus": gamma_plus}
    projectors = {
        "parity": 0.5 * (identity + gamma_t),
        "time": gamma_t,
        "plus": gamma_plus,
    }
    if diquark_kind not in choices:
        raise ValueError("diquark_kind must be identity, time, or plus")
    if projector_kind not in projectors:
        raise ValueError("projector_kind must be parity, time, or plus")
    diquark_sink = charge_conjugation @ gamma5 @ choices[diquark_kind]
    return EnhancedKernels(
        gamma_plus=gamma_plus,
        gamma_minus=gamma_minus,
        quark_plus=quark_plus,
        projector=projectors[projector_kind],
        diquark_sink=diquark_sink,
        diquark_source_bar=euclidean_source_conjugate(diquark_sink, gamma_t),
    )


def meson_contraction(
    propagator_quark: Any,
    propagator_antiquark: Any,
    gamma_sink: Any,
    gamma_source_bar: Any,
    gamma5: Any,
) -> Any:
    """Return the local enhanced meson contraction before momentum sum."""

    xp = _xp(propagator_quark)
    if _xp(propagator_antiquark) is not xp:
        raise TypeError("quark and antiquark propagators must use one backend")
    if propagator_quark.shape != propagator_antiquark.shape:
        raise ValueError("quark and antiquark propagators must have one shape")
    _require_same_backend(
        propagator_quark, gamma_sink, gamma_source_bar, gamma5
    )
    if propagator_quark.ndim < 4 or propagator_quark.shape[-4:] != (4, 4, 3, 3):
        raise ValueError(
            "propagators must end in (sink_spin,source_spin,sink_color,source_color)=(4,4,3,3)"
        )
    if any(matrix.shape != (4, 4) for matrix in (gamma_sink, gamma_source_bar, gamma5)):
        raise ValueError("all meson spin kernels must have shape (4,4)")
    antiquark_backward = xp.einsum(
        "ij,...kjba,kl->...ilab",
        gamma5,
        propagator_antiquark.conj(),
        gamma5,
        optimize=True,
    )
    return -xp.einsum(
        "ij,...jkab,kl,...liba->...",
        gamma_sink,
        propagator_quark,
        gamma_source_bar,
        antiquark_backward,
        optimize=True,
    )


def baryon_contraction(
    propagator_u1: Any,
    propagator_u2: Any,
    propagator_d: Any,
    spin_projector: Any,
    diquark_sink: Any,
    diquark_source_bar: Any,
) -> Any:
    """Contract the enhanced local uud interpolator, direct minus exchange.

    The operator is arXiv:2606.02447v2 Eq. (2.6), and the two terms are the
    Wick contraction structure entering Eq. (2.7). Propagators end in
    `(sink_spin,source_spin,sink_color,source_color)`.
    """

    xp = _xp(propagator_u1)
    propagators = (propagator_u1, propagator_u2, propagator_d)
    if any(_xp(prop) is not xp for prop in propagators[1:]):
        raise TypeError("all baryon propagators must use one backend")
    _require_same_backend(
        propagator_u1, spin_projector, diquark_sink, diquark_source_bar
    )
    if any(prop.shape != propagator_u1.shape for prop in propagators[1:]):
        raise ValueError("all baryon propagators must have one shape")
    if propagator_u1.ndim < 4 or propagator_u1.shape[-4:] != (4, 4, 3, 3):
        raise ValueError(
            "propagators must end in (sink_spin,source_spin,sink_color,source_color)=(4,4,3,3)"
        )
    if any(
        matrix.shape != (4, 4)
        for matrix in (spin_projector, diquark_sink, diquark_source_bar)
    ):
        raise ValueError("all baryon spin kernels must have shape (4,4)")
    epsilon = xp.zeros((3, 3, 3), dtype=propagator_u1.dtype)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
    direct = xp.einsum(
        "abc,ABC,iI,jk,JK,...iIaA,...jJbB,...kKcC->...",
        epsilon,
        epsilon,
        spin_projector,
        diquark_sink,
        diquark_source_bar,
        propagator_u1,
        propagator_u2,
        propagator_d,
        optimize=True,
    )
    exchange = xp.einsum(
        "abc,ABC,iI,jk,JK,...iJaB,...jIbA,...kKcC->...",
        epsilon,
        epsilon,
        spin_projector,
        diquark_sink,
        diquark_source_bar,
        propagator_u1,
        propagator_u2,
        propagator_d,
        optimize=True,
    )
    return direct - exchange


def momentum_project(
    local_contraction: Any,
    phase: Any,
    *,
    spatial_comm: Any | None = None,
    cuda_aware_mpi: bool = False,
) -> Any:
    """Project a local checkerboard field and optionally sum spatial ranks.

    Both arrays must have exactly ``(e,t,z,y,xh)`` layout and the phase must
    already use global coordinates. ``spatial_comm`` must contain only ranks
    that own spatial pieces of the same global time slices. For CuPy arrays it
    must be CUDA-aware; this function performs no hidden host transfer.
    """

    if not isinstance(cuda_aware_mpi, (bool, np.bool_)):
        raise TypeError("cuda_aware_mpi must be a Boolean opt-in flag")
    cuda_aware_mpi = bool(cuda_aware_mpi)
    xp = _xp(local_contraction)
    if _xp(phase) is not xp:
        raise TypeError("phase and local_contraction must use the same backend")
    if local_contraction.ndim != 5 or phase.ndim != 5:
        raise ValueError("phase and contraction must have exact (e,t,z,y,xh) layout")
    if local_contraction.shape != phase.shape:
        raise ValueError("phase and contraction checkerboard lattice axes must match")
    if local_contraction.shape[0] != 2:
        raise ValueError("checkerboard parity extent must be e=2")
    local = xp.einsum(
        "etzyx,etzyx->t", local_contraction, phase, optimize=True
    )
    if spatial_comm is None:
        return local
    size = getattr(spatial_comm, "size", None)
    if size is None and hasattr(spatial_comm, "Get_size"):
        size = spatial_comm.Get_size()
    if size == 1:
        return local
    if not hasattr(spatial_comm, "Allreduce"):
        raise TypeError("spatial_comm must provide mpi4py-style buffer Allreduce")
    if xp is not np and not cuda_aware_mpi:
        raise RuntimeError(
            "CuPy spatial reduction requires cuda_aware_mpi=True after runtime verification"
        )
    send = xp.ascontiguousarray(local)
    reduced = xp.empty_like(send)
    try:
        spatial_comm.Allreduce(send, reduced)
    except Exception as error:
        if xp is not np:
            raise RuntimeError(
                "CuPy buffer Allreduce failed; verify CUDA-aware mpi4py/MPI support"
            ) from error
        raise
    return reduced


__all__ = [
    "EnhancedKernels",
    "euclidean_adjoint",
    "euclidean_source_conjugate",
    "lightcone_gammas",
    "plus_quark_projector",
    "enhanced_meson_kernel",
    "enhanced_pion_kernel",
    "enhanced_baryon_kernels",
    "meson_contraction",
    "baryon_contraction",
    "momentum_project",
]
