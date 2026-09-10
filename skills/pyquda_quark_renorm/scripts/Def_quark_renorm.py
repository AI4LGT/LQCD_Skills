from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
import numpy as np
import math
import importlib.util
import sys
import types
from pathlib import Path
from scipy.special import zeta
from types import MappingProxyType
from typing import Callable, Union

# ``dataclasses`` resolves postponed annotations through ``sys.modules``.  A
# few consumers intentionally load this skill with ``spec_from_file_location``
# without registering the module first; install a lightweight namespace for
# that execution mode so the pure-NumPy path remains importable on Python 3.13
# while retaining Python 3.8-compatible postponed annotations.
if __name__ not in sys.modules:
    sys.modules[__name__] = types.ModuleType(__name__)

try:
    import cupy as cp
except ImportError:  # CPU perturbative routines do not require CuPy.
    cp = None

try:
    from pyquda_utils import gamma
except ImportError:  # Deferred until the spin-color amputation kernels are used.
    gamma = None

# Perturbative coefficients, pandas input, gvar, and lsqfit below intentionally
# remain on CPU: they operate on small scale/momentum tables and their libraries
# do not accept CuPy arrays. The spin-color inversion/amputation kernels at the
# end of this module dispatch to CuPy and must receive unreduced tensor data;
# transfer only compact fit inputs into the CPU-only routines.

try:
    from . import Def_qcd_analysis as analy
except ImportError:
    # Skill scripts are often loaded by absolute path rather than as a package.
    # Resolve the companion relative to this file instead of the process CWD.
    _analysis_path = Path(__file__).with_name("Def_qcd_analysis.py")
    _analysis_spec = importlib.util.spec_from_file_location(
        f"{__name__}_qcd_analysis", _analysis_path
    )
    if _analysis_spec is None or _analysis_spec.loader is None:
        raise ImportError(f"cannot load companion module: {_analysis_path}")
    analy = importlib.util.module_from_spec(_analysis_spec)
    sys.modules[_analysis_spec.name] = analy
    _analysis_spec.loader.exec_module(analy)

def flatten(seq):
    """Flatten a nested numerical sequence without evaluating source text."""

    return np.asarray(seq, dtype=object).reshape(-1).tolist()


def _validate_nf(nf):
    if isinstance(nf, (bool, np.bool_)) or not isinstance(nf, (int, np.integer)):
        raise TypeError("nf must be a non-boolean integer")
    nf = int(nf)
    if not 0 <= nf <= 16:
        raise ValueError("nf must lie in the asymptotically-free range 0..16")
    return nf


def _finite_real_scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _contains_boolean(value):
    """Detect Boolean provenance before NumPy dtype inference can erase it."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("numerical sequences must not be string-like")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            return True
        if value.dtype.kind != "O":
            return False
        return any(_contains_boolean(item) for item in value.flat)
    if isinstance(value, SequenceABC):
        return any(_contains_boolean(item) for item in value)
    return False


def _finite_real_array(value, name):
    """Convert one real-domain host array without silent narrowing."""

    if _contains_boolean(value):
        raise TypeError(f"{name} must be a finite real array, not booleans")
    raw = np.asarray(value)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TypeError(f"{name} must be a finite real array")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real array") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _resolve_lambda(Lambda, nf):
    nf = _validate_nf(nf)
    if Lambda is None:
        if nf != 3:
            raise ValueError(
                "the default Lambda=0.332 GeV is defined only for nf=3; "
                "pass an explicit flavor-appropriate Lambda"
            )
        Lambda = 0.332
    return _finite_real_scalar(Lambda, "Lambda", positive=True), nf


def _nonboolean_integer(value, name, *, minimum=None, allowed=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a non-boolean integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if allowed is not None and result not in allowed:
        choices = ", ".join(str(item) for item in sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}")
    return result


######################################### Physical-profile registry #########################################

# The numerical functions below predate this registry and deliberately retain
# their literal signatures.  The registry is the only API that attaches the
# ``physical=True`` assertion to a formula.  In particular, a direct call to a
# historical decimal table remains a literal replay, not a provenance claim.

_PHYSICAL_PROFILE_STATUSES = frozenset({"MAPPED", "MAPPED_DERIVED"})
_UNVERIFIED_LEGACY = "UNVERIFIED_LEGACY"


@dataclass(frozen=True)
class QuarkRenormProfile:
    """Machine-readable provenance for one perturbative QRK formula.

    ``projector_or_definition`` records the paper's conversion-function
    definition when a continuum conversion does not use a lattice vertex
    projector.  It does not certify an upstream lattice NPR projector.
    """

    symbol: str
    family: str
    status: str
    paper: str | None
    paper_version: str | None
    page: str | None
    equation: str | None
    source_scheme: str | None
    target_scheme: str | None
    operator_basis: str | None
    projector_or_definition: str | None
    gauge: str | None
    direction: str | None
    expansion_variable: str | None
    normalization: str | None
    source_check: str | None
    reason: str | None = None

    @property
    def complete_provenance(self) -> bool:
        """Whether this entry contains every field needed by ``physical=True``."""

        required = (
            self.paper,
            self.paper_version,
            self.page,
            self.equation,
            self.source_scheme,
            self.target_scheme,
            self.operator_basis,
            self.projector_or_definition,
            self.gauge,
            self.direction,
            self.expansion_variable,
            self.normalization,
            self.source_check,
        )
        return self.status in _PHYSICAL_PROFILE_STATUSES and all(required)

    def provenance_dict(self) -> dict[str, str | None]:
        """Return a stable, serialization-ready record without a callable."""

        return {
            "symbol": self.symbol,
            "family": self.family,
            "status": self.status,
            "paper": self.paper,
            "paper_version": self.paper_version,
            "page": self.page,
            "equation": self.equation,
            "source_scheme": self.source_scheme,
            "target_scheme": self.target_scheme,
            "operator_basis": self.operator_basis,
            "projector_or_definition": self.projector_or_definition,
            "gauge": self.gauge,
            "direction": self.direction,
            "expansion_variable": self.expansion_variable,
            "normalization": self.normalization,
            "source_check": self.source_check,
            "reason": self.reason,
        }


class QuarkRenormProfileError(ValueError):
    """Base error for provenance-gated quark-renormalization requests."""

    def __init__(self, symbol: str, status: str, detail: str):
        self.symbol = symbol
        self.status = status
        self.detail = detail
        super().__init__(f"{status}: {symbol}: {detail}")


class UnverifiedLegacyError(QuarkRenormProfileError):
    """Fail-closed result for a request to use a legacy formula physically."""

    def __init__(self, profile: QuarkRenormProfile):
        super().__init__(
            profile.symbol,
            _UNVERIFIED_LEGACY,
            profile.reason
            or "no equation-level provenance is stored for this legacy path",
        )


class UnknownQuarkRenormProfileError(QuarkRenormProfileError):
    """Fail-closed result for a symbol absent from the profile registry."""

    def __init__(self, symbol: str):
        super().__init__(
            symbol,
            "UNKNOWN_PROFILE",
            "no physical-profile record exists; do not infer one from a function name",
        )


@dataclass(frozen=True)
class QuarkRenormLiteral:
    """Explicit nonphysical wrapper around an ``UNVERIFIED_LEGACY`` function."""

    profile: QuarkRenormProfile
    function: Callable[..., object]
    physical: bool = False

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def _mapped_profile(
    symbol,
    family,
    *,
    paper,
    paper_version,
    page,
    equation,
    source_scheme,
    target_scheme,
    operator_basis,
    projector_or_definition,
    gauge,
    direction,
    expansion_variable,
    normalization,
    source_check,
    status="MAPPED",
):
    return QuarkRenormProfile(
        symbol=symbol,
        family=family,
        status=status,
        paper=paper,
        paper_version=paper_version,
        page=page,
        equation=equation,
        source_scheme=source_scheme,
        target_scheme=target_scheme,
        operator_basis=operator_basis,
        projector_or_definition=projector_or_definition,
        gauge=gauge,
        direction=direction,
        expansion_variable=expansion_variable,
        normalization=normalization,
        source_check=source_check,
    )


def _legacy_profile(symbol, family, reason):
    return QuarkRenormProfile(
        symbol=symbol,
        family=family,
        status=_UNVERIFIED_LEGACY,
        paper=None,
        paper_version=None,
        page=None,
        equation=None,
        source_scheme=None,
        target_scheme=None,
        operator_basis=None,
        projector_or_definition=None,
        gauge=None,
        direction=None,
        expansion_variable=None,
        normalization=None,
        source_check=None,
        reason=reason,
    )


# These strings transcribe only the records in reference/SOURCE_VERIFICATION.md.
# They bind formula-level perturbative provenance; they do not bind a raw lattice
# NPR vertex, its gauge fixing, or its continuum extrapolation.
_MAPPED_QUARK_RENORM_PROFILES = {
    "quark_field_conversion_ms_bar_over_rimom": _mapped_profile(
        "quark_field_conversion_ms_bar_over_rimom",
        "ri_mom_field_conversion",
        paper="Chetyrkin and Retey, Nucl. Phys. B583 (2000) 3-34",
        paper_version="arXiv:hep-ph/9910332v2",
        page="PDF p. 11",
        equation="Eq. (34), with Eqs. (9)-(12) notation",
        source_scheme="RI/MOM C_2 field conversion function",
        target_scheme="MS-bar C_2 field conversion function",
        operator_basis="SU(3) quark two-point field factor C_2",
        projector_or_definition="CR2000 C_2 conversion-function definition",
        gauge="Landau gauge",
        direction="API follows the literal C_2^RI direction; external Z_q direction remains caller-owned",
        expansion_variable="paper a=alpha_s/(4*pi); module a_s=alpha_s/pi",
        normalization="a=a_s/4, so a^2=a_s^2/16 and a^3=a_s^3/64",
        source_check="coefficient-by-coefficient transcription of Eq. (34)",
    ),
    "quark_field_conversion_ms_bar_over_rimom_prime": _mapped_profile(
        "quark_field_conversion_ms_bar_over_rimom_prime",
        "ri_prime_mom_field_conversion",
        paper="Chetyrkin and Retey, Nucl. Phys. B583 (2000) 3-34",
        paper_version="arXiv:hep-ph/9910332v2",
        page="PDF p. 11",
        equation="Eq. (36), with Eqs. (9)-(12) notation",
        source_scheme="RI-prime/MOM C_2 field conversion function",
        target_scheme="MS-bar C_2 field conversion function",
        operator_basis="SU(3) quark two-point field factor C_2",
        projector_or_definition="CR2000 C_2 conversion-function definition",
        gauge="Landau gauge",
        direction="API follows the literal C_2^RI-prime direction; external Z_q direction remains caller-owned",
        expansion_variable="paper a=alpha_s/(4*pi); module a_s=alpha_s/pi",
        normalization="a=a_s/4, so a^2=a_s^2/16 and a^3=a_s^3/64",
        source_check="coefficient-by-coefficient transcription of Eq. (36)",
    ),
    "quark_mass_conversion_ms_bar_over_rimom_prime": _mapped_profile(
        "quark_mass_conversion_ms_bar_over_rimom_prime",
        "ri_prime_mom_mass_conversion",
        paper="Chetyrkin and Retey, Nucl. Phys. B583 (2000) 3-34",
        paper_version="arXiv:hep-ph/9910332v2",
        page="PDF pp. 11-12",
        equation="Eqs. (37), (41), with Eqs. (9)-(12) notation",
        source_scheme="RI-prime/MOM quark mass m-prime",
        target_scheme="MS-bar quark mass m",
        operator_basis="SU(3) quark mass conversion function C_m",
        projector_or_definition="CR2000 C_m conversion-function definition",
        gauge="Landau gauge",
        direction="m_MS-bar = C_m^RI-prime * m_RI-prime",
        expansion_variable="paper a=alpha_s/(4*pi); module a_s=alpha_s/pi",
        normalization="a=a_s/4, so a=a_s/4, a^2=a_s^2/16, a^3=a_s^3/64",
        source_check="Eq. (41) nf=4, a_s=0.1 reference decomposition plus Eq. (37) transcription",
    ),
    "vector_conversion_ms_bar_over_rimom_prime": _mapped_profile(
        "vector_conversion_ms_bar_over_rimom_prime",
        "derived_ri_prime_over_ri_field_ratio",
        paper="Chetyrkin and Retey, Nucl. Phys. B583 (2000) 3-34",
        paper_version="arXiv:hep-ph/9910332v2",
        page="PDF p. 11",
        equation="fixed-order ratio of Eqs. (36)/(34)",
        source_scheme="RI/MOM C_2 field conversion function",
        target_scheme="RI-prime/MOM C_2 field conversion function",
        operator_basis="SU(3) quark two-point field factor C_2",
        projector_or_definition="ratio of the CR2000 C_2 conversion-function definitions",
        gauge="Landau gauge",
        direction="C_2^RI-prime / C_2^RI through O(a^3); historical vector name is not a vector-current map",
        expansion_variable="paper a=alpha_s/(4*pi); module a_s=alpha_s/pi",
        normalization="fixed-order ratio retains a_s^2/16 and a_s^3/64 terms",
        source_check="algebraic division of the mapped Eq. (36) and Eq. (34) series",
        status="MAPPED_DERIVED",
    ),
    "quark_field_conversion_rimom_prime_over_rimom": _mapped_profile(
        "quark_field_conversion_rimom_prime_over_rimom",
        "derived_ri_prime_over_ri_field_ratio",
        paper="Chetyrkin and Retey, Nucl. Phys. B583 (2000) 3-34",
        paper_version="arXiv:hep-ph/9910332v2",
        page="PDF p. 11",
        equation="fixed-order ratio of Eqs. (36)/(34)",
        source_scheme="RI/MOM C_2 field conversion function",
        target_scheme="RI-prime/MOM C_2 field conversion function",
        operator_basis="SU(3) quark two-point field factor C_2",
        projector_or_definition="ratio of the CR2000 C_2 conversion-function definitions",
        gauge="Landau gauge",
        direction="C_2^RI-prime / C_2^RI through O(a^3); external Z_q direction remains caller-owned",
        expansion_variable="paper a=alpha_s/(4*pi); module a_s=alpha_s/pi",
        normalization="fixed-order ratio retains a_s^2/16 and a_s^3/64 terms",
        source_check="algebraic division of the mapped Eq. (36) and Eq. (34) series",
        status="MAPPED_DERIVED",
    ),
    "tensor_conversion_ms_bar_over_rimom_prime": _mapped_profile(
        "tensor_conversion_ms_bar_over_rimom_prime",
        "ri_prime_mom_tensor_conversion",
        paper="J. A. Gracey, Nucl. Phys. B662 (2003) 247-278",
        paper_version="arXiv:hep-ph/0304113v1",
        page="PDF p. 23",
        equation="Eq. (4.11)",
        source_scheme="RI-prime/MOM tensor Z_T",
        target_scheme="MS-bar tensor Z_T",
        operator_basis="non-singlet tensor current",
        projector_or_definition="G2003 tensor conversion function C_T=Z_T^RI-prime/Z_T^MS-bar",
        gauge="Landau gauge (xi=0 specialization)",
        direction="API returns the fixed-order inverse Z_T^MS-bar/Z_T^RI-prime through O(a^3)",
        expansion_variable="paper a=alpha_s/(4*pi); module a_s=alpha_s/pi",
        normalization="fixed-order inverse uses -k2*a_s^2/16-k3*a_s^3/64",
        source_check="Eq. (4.11) coefficient transcription and fixed-order inverse",
    ),
    "beta_coupling_constant": _mapped_profile(
        "beta_coupling_constant",
        "ms_bar_coupling_beta",
        paper="Baikov, Chetyrkin and Kuehn, Phys. Rev. Lett. 118 (2017) 082002",
        paper_version="arXiv:1606.08659v2",
        page="PDF p. 1",
        equation="Eqs. (1)-(5)",
        source_scheme="MS-bar coupling a_s",
        target_scheme="MS-bar coupling a_s",
        operator_basis="SU(3) QCD beta function",
        projector_or_definition="not applicable: continuum MS-bar coupling beta function",
        gauge="not applicable: MS-bar beta-function coefficients",
        direction="d a_s / d ln(mu^2) = -sum_i beta_i a_s^(i+2)",
        expansion_variable="a_s=alpha_s/pi",
        normalization="code stores beta_i/beta_0 for i>0 after transcribing beta_i",
        source_check="SU(3) Eqs. (2)-(5) reproduce the five stored coefficients",
    ),
    "strong_coupling_constant": _mapped_profile(
        "strong_coupling_constant",
        "ms_bar_coupling_running",
        paper="Baikov, Chetyrkin and Kuehn, Phys. Rev. Lett. 118 (2017) 082002",
        paper_version="arXiv:1606.08659v2",
        page="PDF p. 1",
        equation="Eqs. (1)-(5)",
        source_scheme="MS-bar coupling a_s at Lambda",
        target_scheme="MS-bar coupling a_s(mu)",
        operator_basis="SU(3) QCD beta function",
        projector_or_definition="not applicable: continuum MS-bar coupling beta function",
        gauge="not applicable: MS-bar beta-function coefficients",
        direction="asymptotic solution for a_s(mu) from the mapped beta coefficients",
        expansion_variable="a_s=alpha_s/pi",
        normalization="L=2*log(mu/Lambda), returned as successive one- through five-loop sums",
        source_check="uses the mapped BCK2016 beta coefficients; expansion domain is validated locally",
        status="MAPPED_DERIVED",
    ),
    "quark_mass_anomalous_dimension_under_ms_bar": _mapped_profile(
        "quark_mass_anomalous_dimension_under_ms_bar",
        "ms_bar_mass_running",
        paper="Baikov, Chetyrkin and Kuehn, JHEP 10 (2014) 076",
        paper_version="arXiv:1402.6611v1",
        page="PDF pp. 3 and 5",
        equation="Eqs. (3.1)-(3.4), (4.7)-(4.12)",
        source_scheme="MS-bar quark mass at mu0",
        target_scheme="MS-bar quark mass at mu",
        operator_basis="MS-bar quark mass anomalous dimension",
        projector_or_definition="not applicable: continuum MS-bar mass anomalous dimension",
        gauge="not applicable: MS-bar quark-mass running",
        direction="API returns c(a_s(mu0))/c(a_s(mu)), the inverse of the Eq. (4.7) mass ratio",
        expansion_variable="a_s=alpha_s/pi",
        normalization="code maps paper d2,d3 into c-function coefficients before truncation",
        source_check="BCK2014 coefficient transcription plus equal-scale and composition algebra checks",
    ),
    "scalar_anomalous_dimension_under_ms_bar": _mapped_profile(
        "scalar_anomalous_dimension_under_ms_bar",
        "derived_ms_bar_scalar_running",
        paper="Baikov, Chetyrkin and Kuehn, JHEP 10 (2014) 076",
        paper_version="arXiv:1402.6611v1",
        page="PDF pp. 3 and 5",
        equation="Eqs. (3.1)-(3.4), (4.7)-(4.12)",
        source_scheme="MS-bar scalar factor at mu0 under Z_S=1/Z_m",
        target_scheme="MS-bar scalar factor at mu under Z_S=1/Z_m",
        operator_basis="scalar reciprocal of the mapped MS-bar mass factor",
        projector_or_definition="not applicable: local reciprocal wrapper, no independent lattice projector",
        gauge="not applicable: inherits the mapped MS-bar mass-running formula",
        direction="entrywise reciprocal of the mapped mass-running output",
        expansion_variable="a_s=alpha_s/pi",
        normalization="numerical reciprocal of each returned truncation; it is not re-expanded",
        source_check="algebraic reciprocal of the mapped BCK2014 mass-running function",
        status="MAPPED_DERIVED",
    ),
}

_LEGACY_QUARK_RENORM_PROFILES = {
    name: _legacy_profile(
        name,
        "smom_decimal_conversion",
        "decimal RI/SMOM path lacks a frozen RI/SMOM versus RI/SMOM-gamma_mu projector, xi, matching-scale, and equation-to-decimal reduction",
    )
    for name in (
        "quark_mass_conversion_ms_bar_over_rismom",
        "quark_mass_conversion_ms_bar_over_rismom_mu",
        "scalar_conversion_ms_bar_over_rismom",
        "scalar_conversion_ms_bar_over_rismom_mu",
        "tensor_conversion_ms_bar_over_rismom",
        "tensor_conversion_ms_bar_over_rismom_mu",
    )
}
_LEGACY_QUARK_RENORM_PROFILES.update(
    {
        name: _legacy_profile(
            name,
            "general_xi_rimom_prime2",
            "general-xi RI-prime/MOM table lacks a frozen paper version, equation, ordered definition, gauge convention, direction, and normalization binding",
        )
        for name in (
            "quark_mass_conversion_ms_bar_over_rimom_prime2",
            "quark_field_conversion_rimom_prime_over_rimom2",
            "scalar_conversion_ms_bar_over_rimom_prime2",
            "tensor_conversion_ms_bar_over_rimom_prime2",
        )
    }
)
_LEGACY_QUARK_RENORM_PROFILES.update(
    {
        name: _legacy_profile(
            name,
            "unmapped_anomalous_dimension",
            "quark-field/tensor anomalous-dimension coefficients lack a coefficient-level paper, scheme, gauge, direction, and expansion-variable map",
        )
        for name in (
            "quark_field_anomalous_dimension_under_ms_bar",
            "tensor_anomalous_dimension_under_ms_bar",
        )
    }
)
_LEGACY_QUARK_RENORM_PROFILES.update(
    {
        name: _legacy_profile(
            name,
            "pade_literal_estimate",
            "Padé order, construction equation, fit input, pole domain, and truncation prescription are not source-mapped",
        )
        for name in (
            "scalar_conversion_ms_bar_over_rimom_pade_3loop",
            "scalar_conversion_ms_bar_over_rimom_pade_4loop",
            "tensor_conversion_ms_bar_over_rimom_pade_3loop",
            "tensor_conversion_ms_bar_over_rimom_pade_4loop",
            "pade_matching_factor",
        )
    }
)
_LEGACY_QUARK_RENORM_PROFILES["matching_systematic_error"] = _legacy_profile(
    "matching_systematic_error",
    "matching_systematic_error",
    "the seven-channel systematic-error table has no frozen source map for its channel definitions, coefficients, matching direction, or uncertainty prescription",
)

QUARK_RENORM_PROFILE_REGISTRY = MappingProxyType(
    {**_MAPPED_QUARK_RENORM_PROFILES, **_LEGACY_QUARK_RENORM_PROFILES}
)
QUARK_RENORM_PHYSICAL_PROFILES = MappingProxyType(
    {
        symbol: profile
        for symbol, profile in QUARK_RENORM_PROFILE_REGISTRY.items()
        if profile.complete_provenance
    }
)


def _profile_for_symbol(symbol):
    if not isinstance(symbol, str) or not symbol:
        raise TypeError("profile symbol must be a non-empty string")
    try:
        return QUARK_RENORM_PROFILE_REGISTRY[symbol]
    except KeyError as error:
        raise UnknownQuarkRenormProfileError(symbol) from error


def get_quark_renorm_profile(symbol, *, physical=True):
    """Request formula provenance, fail-closed for nonphysical paths.

    The default is deliberately ``physical=True``.  Passing ``physical=False``
    exposes metadata only; calling an unmapped table requires the separate,
    conspicuously named :func:`get_quark_renorm_legacy_literal` route.
    """

    if not isinstance(physical, (bool, np.bool_)):
        raise TypeError("physical must be a boolean")
    profile = _profile_for_symbol(symbol)
    if physical:
        if profile.status == _UNVERIFIED_LEGACY:
            raise UnverifiedLegacyError(profile)
        if not profile.complete_provenance:
            raise QuarkRenormProfileError(
                profile.symbol,
                "INCOMPLETE_PROVENANCE",
                "a mapped status without every provenance field cannot be requested physically",
            )
    return profile


def invoke_quark_renorm_profile(symbol, *args, physical=True, **kwargs):
    """Invoke a complete, source-mapped perturbative formula.

    This verifies formula provenance only.  It never supplies a gauge-fixed
    lattice NPR input, continuum limit, or physical uncertainty budget.
    """

    profile = get_quark_renorm_profile(symbol, physical=physical)
    if not physical:
        raise QuarkRenormProfileError(
            profile.symbol,
            "NONPHYSICAL_ROUTE_REQUIRED",
            "use get_quark_renorm_legacy_literal(..., physical=False) for a literal legacy replay",
        )
    function = globals().get(profile.symbol)
    if not callable(function):
        raise RuntimeError(f"profile callable is unavailable: {profile.symbol}")
    return function(*args, **kwargs)


def get_quark_renorm_legacy_literal(symbol, *, physical=False):
    """Return an explicit ``UNVERIFIED_LEGACY`` literal-only callable.

    The wrapper preserves historical numerical access but carries
    ``physical=False`` permanently.  Requesting the same row physically raises
    :class:`UnverifiedLegacyError` before any numerical table is evaluated.
    """

    if not isinstance(physical, (bool, np.bool_)):
        raise TypeError("physical must be a boolean")
    profile = _profile_for_symbol(symbol)
    if physical:
        # Use the ordinary gate to preserve the typed status and message.
        get_quark_renorm_profile(symbol, physical=True)
        raise AssertionError("a verified profile is not a legacy literal")
    if profile.status != _UNVERIFIED_LEGACY:
        raise QuarkRenormProfileError(
            profile.symbol,
            "MAPPED_PROFILE_NOT_LEGACY",
            "use invoke_quark_renorm_profile(..., physical=True) for this source-mapped row",
        )
    function = globals().get(profile.symbol)
    if not callable(function):
        raise RuntimeError(f"legacy literal callable is unavailable: {profile.symbol}")
    return QuarkRenormLiteral(profile=profile, function=function)


######################################### Strong Coupling Constant #########################################

def beta_coupling_constant(nf):
    nf = _validate_nf(nf)
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    beta0 = float((11.0 - 2.0*nf/3)/4)
    beta1 = float((102.0 - 38.0*nf/3)/16)
    beta2 = float((2857.0 - 5033.0*nf/9 + 325.0*nf*nf/27)/128)
    beta3 = float((149753.0/6 + 3564.0*z3 - (1078361.0/162 + 6508.0*z3/27)*nf
    + (50065.0/162 + 6472.0*z3/81)*nf*nf + 1093.0*nf*nf*nf/729)/256)
    beta4 = float((8157455.0/16 + 621885.0*z3/2 - 88209.0*z4/2 - 288090.0*z5
    + (-336460813.0/1944 - 4811164.0*z3/81 + 33935.0*z4/6 + 1358995.0*z5/27)*nf
    + (25960913.0/1944 + 698531.0*z3/81 - 10526.0*z4/9 - 381760.0*z5/81)*nf*nf
    + (-630559.0/5832 - 48722.0*z3/243 + 1618.0*z4/27 + 460.0*z5/9)*nf*nf*nf
    + (1205.0/2916 - 152.0*z3/81)*nf*nf*nf*nf)/1024)
    beta = np.array([beta0, beta1 / beta0, beta2 / beta0, beta3 / beta0, beta4 / beta0])
    return beta


def strong_coupling_constant(scale, Lambda=None, nf=3):
    scale = _finite_real_scalar(scale, "scale", positive=True)
    Lambda, nf = _resolve_lambda(Lambda, nf)
    beta = beta_coupling_constant(nf)
    b0, b1, b2, b3, b4 = beta[0], beta[1], beta[2], beta[3], beta[4]
    l = float(scale / Lambda)
    # The asymptotic expansion is not valid here. The original fallback set
    # l=0 and immediately evaluated log(0), silently producing NaNs.
    if l <= 3:
        raise ValueError("strong-coupling expansion requires scale / Lambda > 3")
    L = 2*np.log(l)
    LL, bL = np.log(L), b0 * L
    as_1 = 1/ bL
    as_2 = -b1*LL/bL/bL
    as_3 = (b1*b1 * (LL*LL - LL - 1) + b2)/(pow(bL,3))
    as_4 = (pow(b1,3) * (-pow(LL,3) + 2.5*LL*LL + 2*LL - 1.0/2) - 3*b1*b2*LL + b3/2)/pow(bL,4)
    as_5 = (pow(b1,4) * (pow(LL,4) - 13.0/3*LL*LL*LL - 3.0/2*LL*LL + 4*LL + 7.0/6)\
    + 3*b1*b1*b2*(2*LL*LL - LL - 1) - b1*b3*(2*LL + 1.0/6) + 5.0/3*b2*b2 + b4/3)/pow(bL,5)
    ret0 = as_1
    ret1 = ret0 + as_2
    ret2 = ret1 + as_3
    ret3 = ret2 + as_4
    ret4 = ret3 + as_5
    ret = np.array([ret0, ret1, ret2, ret3, ret4])
    return ret


def alpha_s(scale, nf=3, Lambda=None):
    #5-loop alpha_s
    alpha = strong_coupling_constant(scale, Lambda, nf)[4] * np.pi
    return alpha



######################################### RI/RI* to MSbar Matching #########################################

def vector_conversion_ms_bar_over_rimom_prime(scale, Lambda=None, nf=3):
    z3 = zeta(3)
    a_s = strong_coupling_constant(scale, Lambda, nf)
    a2 = float((8.0*nf - 134.0)/3/4)
    a3 = float((-52321.0/18 + 607.0*z3 + (8944.0/27 - 32.0*z3)*nf - 208.0*nf*nf/27)/4.0)
    C_V = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_V[i] = 1 + a2*a_s[i]*a_s[i]/16 + a3*a_s[i]*a_s[i]*a_s[i]/64
    return C_V


def quark_mass_conversion_ms_bar_over_rismom(scale, Lambda=None, nf=3):
    a_s = strong_coupling_constant(scale, Lambda, nf)
    C_m = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_m[i] = 1 - 0.1613797*a_s[i] - 0.66044182*a_s[i]*a_s[i]
    return C_m

def quark_mass_conversion_ms_bar_over_rismom_mu(scale, Lambda=None, nf=3):
    a_s = strong_coupling_constant(scale, Lambda, nf)
    C_m = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_m[i] = 1 - 0.494713025*a_s[i] - (55.03243483-6.161687618*nf)*a_s[i]*a_s[i]/16
    return C_m


def quark_field_conversion_rimom_prime_over_rimom(scale, Lambda=None, nf=3):
    return vector_conversion_ms_bar_over_rimom_prime(scale, Lambda, nf)


def quark_field_conversion_ms_bar_over_rimom(scale, Lambda=None, nf=3):
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    a_s = strong_coupling_constant(scale, Lambda, nf)
    a2 = float(-517.0/18 + 12.0*z3 + 5.0*nf/3)
    a3 = float(-1287283.0/648 + 14197.0*z3/12 + 79.0*z4/4 - 1165.0*z5/3
    +18014.0*nf/81 - 368.0*z3*nf/9 - 1102.0*nf*nf/243)
    C_q = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_q[i] = 1 + a2*a_s[i]*a_s[i]/16 + a3*a_s[i]*a_s[i]*a_s[i]/64
    return C_q


def quark_field_conversion_ms_bar_over_rimom_prime(scale, Lambda=None, nf=3):
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    a_s = strong_coupling_constant(scale, Lambda, nf)
    a2 = float(-359.0/9 + 12.0*z3 + 7.0*nf/3)
    a3 = float(-439543.0/162 + 8009.0*z3/6 + 79.0*z4/3 - 1165.0*z5/3
    + 24722.0*nf/81 - 440.0*z3*nf/9 - 1570.0*nf*nf/243)
    C_q = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_q[i] = 1 + a2*a_s[i]*a_s[i]/16 + a3*a_s[i]*a_s[i]*a_s[i]/64
    return C_q


def quark_mass_conversion_ms_bar_over_rimom_prime(scale, Lambda=None, nf=3):
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    a_s = strong_coupling_constant(scale, Lambda, nf)
    a1 = float(-16.0/3)
    a2 = float(-3779.0/18+152.0/3*z3+83.0*nf/9)
    a3 = float(-3115807.0/324 + 195809.0*z3/54 - 2960.0*z5/9 + 217390.0*nf/243
    -4720.0*z3*nf/27 + 80.0*z4*nf/3 - 7514.0*nf*nf/729 - 32.0*z3*nf*nf/27)
    C_m = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_m[i] = 1 + a1*a_s[i]/4 + a2*a_s[i]*a_s[i]/16 + a3*pow(a_s[i],3)/64
    return C_m


def scalar_conversion_ms_bar_over_rismom(scale, Lambda=None, nf=3):
    a_s = quark_mass_conversion_ms_bar_over_rismom(scale, Lambda, nf)
    for i in range(len(a_s)):
        a_s[i] = 1.0/a_s[i]
    return a_s


def scalar_conversion_ms_bar_over_rismom_mu(scale, Lambda=None, nf=3):
    a_s = quark_mass_conversion_ms_bar_over_rismom_mu(scale, Lambda, nf)
    for i in range(len(a_s)):
        a_s[i] = 1.0/a_s[i]
    return a_s


def scalar_conversion_ms_bar_over_rimom_prime(scale, Lambda=None, nf=3):
    a_s = quark_mass_conversion_ms_bar_over_rimom_prime(scale, Lambda, nf)
    for i in range(len(a_s)):
        a_s[i] = 1.0/a_s[i]
    return a_s


def tensor_conversion_ms_bar_over_rismom(scale, Lambda=None, nf=3):
    a_s = strong_coupling_constant(scale, Lambda, nf)
    C_T = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_T[i] = 1 - 0.05379324*a_s[i] - 1.94213215*a_s[i]*a_s[i]
    return C_T


def tensor_conversion_ms_bar_over_rismom_mu(scale, Lambda, nf = 3):
    a_s = strong_coupling_constant(scale, Lambda, nf)
    C_T = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_T[i] = 1 + 0.279540095*a_s[i] - (8.607630493-1.955130440*nf)*a_s[i]*a_s[i]/16
    return C_T


def tensor_conversion_ms_bar_over_rimom_prime(scale, Lambda=None, nf=3):
    """Return the Landau-gauge RI-prime/MOM-to-MS tensor conversion series.

    Gracey, Nucl. Phys. B662 (2003) 247--278, arXiv:hep-ph/0304113v1,
    Eq. (4.11), uses ``a = alpha_s/(4*pi)``.  At ``xi=0`` it gives
    ``C_T = 1 + a2*a**2 + a3*a**3`` for RI-prime over MS.  This historical
    API returns the inverse through O(a**3), hence
    ``1 - a2*a**2 - a3*a**3``.  With the module's ``a_s = alpha_s/pi``, the
    denominators are 16 and 64.

    This fixes the formerly literal ``/6`` third-order normalization.  The
    function remains explicit-only at the skill-routing boundary; a numerical
    call alone is not a complete NPR analysis.
    """

    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    a_s = strong_coupling_constant(scale, Lambda, nf)
    a2 = float(3847.0/54-313.0/81*nf-184.0*z3/9)
    a3 = float((-678473.0*z3/486 - 1072.0*z4/81 + 10040.0*z5/27 + 9858659.0/2916)
    +(2096.0*z3/27 - 80.0*z4/9 - 286262.0/729)*nf +(32.0*z3/81 + 13754.0/2187)*nf*nf)
    C_T = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_T[i] = 1 - a2*a_s[i]*a_s[i]/16 - a3*a_s[i]*a_s[i]*a_s[i]/64
    return C_T



######################################### RI/RI* to MSbar Matching 2 #########################################

def quark_mass_conversion_ms_bar_over_rimom_prime2(scale, xi, Lambda=None, nf=3):
    xi = _finite_real_scalar(xi, "xi")
    z3, z4, z5, z6, z7 = zeta(3), zeta(4), zeta(5), zeta(6), zeta(7)
    Cf = 4.0/3
    CA = 3.0
    Tf = 1.0/2
    d4FA = 7.5
    d4FF = 0.416667
    a_s = strong_coupling_constant(scale, Lambda, nf)
    coef = np.empty((len(a_s)))
    coef[0] = float(-Cf*(4 + xi))
    coef[1] = float((Cf*(4*nf*Tf*(249 + 40*xi) +\
    9*Cf*(19 + 32*xi - 96*z3 + 8*pow(xi,2)) -\
    CA*(3855 + 446*xi - 1296*z3 + 90*pow(xi,2) + 18*pow(xi,3))))/72)
    coef[2] = float(-(Cf*(32*(3757 + 600*xi + 432*z3)*pow(nf,2)*pow(Tf,2) +\
    216*Cf*nf*Tf*(-2218 + 768*z3 + xi*(115 + 144*z3) + 432*z4 + 80*pow(xi,2)) +\
    162*pow(Cf,2)*(6454 - 1392*z3 + 3*xi*(31 + 48*z3) - 2880*z5 -\
    48*(-2 + 3*z3)*pow(xi,2) + 24*pow(xi,3)) -\
    2*CA*(8*nf*Tf*(95387 - 16632*z3 + 12*xi*(994 + 81*z3) + 5832*z4 +\
    1350*pow(xi,2) + 270*pow(xi,3)) +\
    27*Cf*(18781 + xi*(6089 - 3096*z3) - 23088*z3 + 1440*z5 -\
    8*(-170 + 27*z3)*pow(xi,2) + 288*pow(xi,3) + 36*pow(xi,4))) +\
    pow(CA,2)*(3360023 + xi*(340212 - 63180*z3) - 1684422*z3 + 233280*z5 +\
    54*(1256 + 63*z3)*pow(xi,2) + 22734*pow(xi,3) + 4131*pow(xi,4) + 486*pow(xi,5)\
    )))/3888)
    coef[3] = float(152*(0.333333333333)*d4FA - 334*nf*(0.333333333333)*d4FF +\
    96979./729*Cf*pow (Tf, 3)*pow (nf, 3) -\
    2254577./1944*pow (Cf, 2)*pow (Tf, 2)*pow (nf, 2) +\
    391853./144*pow (Cf, 3)*Tf*nf + 676301./384*pow (Cf, 4) -\
    4818313./1944*CA*Cf*pow (Tf, 2)*pow (nf, 2) +\
    7489579./3888*CA*pow (Cf, 2)*Tf*nf - 8105797./576*CA*pow (Cf, 3) +\
    24807575./1944*pow (CA, 2)*Cf*Tf*nf +\
    347228707./31104*pow (CA, 2)*pow (Cf, 2) -\
    854108555./46656*pow (CA, 3)*Cf +\
    11618./243*xi*pow (Cf, 2)*pow (Tf, 2)*pow (nf, 2) -\
    959./9*xi*pow (Cf, 3)*Tf*nf + 3317./12*xi*pow (Cf, 4) -\
    13801./243*xi*CA*Cf*pow (Tf, 2)*pow (nf, 2) -\
    124682./243*xi*CA*pow (Cf, 2)*Tf*nf -\
    50831./144*xi*CA*pow (Cf, 3) +\
    902513./1944*xi*pow (CA, 2)*Cf*Tf*nf +\
    5644277./3888*xi*pow (CA, 2)*pow (Cf, 2) -\
    12316633./15552*xi*pow (CA, 3)*Cf +\
    107./6*pow (xi, 2)*pow (Cf, 3)*Tf*nf +\
    43./8*pow (xi, 2)*pow (Cf, 4) -\
    403./8*pow (xi, 2)*CA*pow (Cf, 2)*Tf*nf -\
    9955./96*pow (xi, 2)*CA*pow (Cf, 3) +\
    285./8*pow (xi, 2)*pow (CA, 2)*Cf*Tf*nf +\
    7237./32*pow (xi, 2)*pow (CA, 2)*pow (Cf, 2) -\
    9341./64*pow (xi, 2)*pow (CA, 3)*Cf + 4*pow (xi, 3)*pow (Cf, 4) -\
    23*pow (xi, 3)*CA*pow (Cf, 3) +\
    679./16*pow (xi, 3)*pow (CA, 2)*pow (Cf, 2) -\
    5743./192*pow (xi, 3)*pow (CA, 3)*Cf + pow (xi, 4)*pow (Cf, 4) -\
    15./4*pow (xi, 4)*CA*pow (Cf, 3) +\
    97./16*pow (xi, 4)*pow (CA, 2)*pow (Cf, 2) -\
    289./64*pow (xi, 4)*pow (CA, 3)*Cf -\
    52689./16*z7*(0.333333333333)*d4FA -\
    1764*z7*nf*(0.333333333333)*d4FF - 11550*z7*pow (Cf, 4) -\
    882*z7*CA*pow (Cf, 2)*Tf*nf + 16583*z7*CA*pow (Cf, 3) +\
    1617./4*z7*pow (CA, 2)*Cf*Tf*nf -\
    310023./32*z7*pow (CA, 2)*pow (Cf, 2) +\
    566069./256*z7*pow (CA, 3)*Cf +\
    1687./8*z7*xi*(0.333333333333)*d4FA +\
    161./16*z7*xi*pow (CA, 2)*pow (Cf, 2) -\
    3913./192*z7*xi*pow (CA, 3)*Cf +\
    399./16*z7*pow (xi, 2)*(0.333333333333)*d4FA +\
    441./32*z7*pow (xi, 2)*pow (CA, 2)*pow (Cf, 2) -\
    1953./256*z7*pow (xi, 2)*pow (CA, 3)*Cf +\
    300*z6*pow (Cf, 3)*Tf*nf - 50*z6*CA*pow (Cf, 2)*Tf*nf -\
    250*z6*pow (CA, 2)*Cf*Tf*nf - 275*z6*pow (CA, 2)*pow (Cf, 2) +\
    275*z6*pow (CA, 3)*Cf + 1335./8*z5*(0.333333333333)*d4FA +\
    200*z5*nf*(0.333333333333)*d4FF -\
    112*z5*pow (Cf, 2)*pow (Tf, 2)*pow (nf, 2) -\
    1060./3*z5*pow (Cf, 3)*Tf*nf + 4020*z5*pow (Cf, 4) +\
    2128./9*z5*CA*Cf*pow (Tf, 2)*pow (nf, 2) +\
    288*z5*CA*pow (Cf, 2)*Tf*nf + 4730./3*z5*CA*pow (Cf, 3) -\
    14459./18*z5*pow (CA, 2)*Cf*Tf*nf +\
    5205./8*z5*pow (CA, 2)*pow (Cf, 2) - 13055./36*z5*pow (CA, 3)*Cf +\
    675./2*z5*xi*(0.333333333333)*d4FA - 120*z5*xi*pow (Cf, 4) +\
    20*z5*xi*CA*pow (Cf, 2)*Tf*nf - 310*z5*xi*CA*pow (Cf, 3) +\
    2585./8*z5*xi*pow (CA, 2)*pow (Cf, 2) -\
    2805./32*z5*xi*pow (CA, 3)*Cf -\
    25./4*z5*pow (xi, 2)*(0.333333333333)*d4FA +\
    5./3*z5*pow (xi, 2)*pow (CA, 2)*Cf*Tf*nf -\
    15*z5*pow (xi, 2)*pow (CA, 2)*pow (Cf, 2) -\
    185./48*z5*pow (xi, 2)*pow (CA, 3)*Cf -\
    5./4*z5*pow (xi, 3)*pow (CA, 3)*Cf -\
    5./8*z5*pow (xi, 4)*(0.333333333333)*d4FA -\
    5./96*z5*pow (xi, 4)*pow (CA, 3)*Cf - 90*z4*(0.333333333333)*d4FA +\
    180*z4*nf*(0.333333333333)*d4FF -\
    16./3*z4*Cf*pow (Tf, 3)*pow (nf, 3) +\
    60*z4*pow (Cf, 2)*pow (Tf, 2)*pow (nf, 2) -\
    111*z4*pow (Cf, 3)*Tf*nf + 126*z4*pow (Cf, 4) -\
    60*z4*CA*Cf*pow (Tf, 2)*pow (nf, 2) - 234*z4*CA*pow (Cf, 2)*Tf*nf -\
    237./2*z4*CA*pow (Cf, 3) + 671./2*z4*pow (CA, 2)*Cf*Tf*nf +\
    57*z4*pow (CA, 2)*pow (Cf, 2) - 709./12*z4*pow (CA, 3)*Cf +\
    24*z4*xi*pow (Cf, 3)*Tf*nf - 36*z4*xi*CA*pow (Cf, 2)*Tf*nf +\
    9*z4*xi*pow (CA, 2)*Cf*Tf*nf - 9./32*z4*xi*pow (CA, 3)*Cf -\
    3./8*z4*pow (xi, 2)*pow (CA, 3)*Cf -\
    3./32*z4*pow (xi, 3)*pow (CA, 3)*Cf -\
    10285./8*z3*(0.333333333333)*d4FA +\
    1676*z3*nf*(0.333333333333)*d4FF +\
    80./27*z3*Cf*pow (Tf, 3)*pow (nf, 3) +\
    6628./9*z3*pow (Cf, 2)*pow (Tf, 2)*pow (nf, 2) -\
    3893./3*z3*pow (Cf, 3)*Tf*nf + 2755*z3*pow (Cf, 4) +\
    92*z3*CA*Cf*pow (Tf, 2)*pow (nf, 2) +\
    2678./9*z3*CA*pow (Cf, 2)*Tf*nf - 19955./12*z3*CA*pow (Cf, 3) -\
    36590./9*z3*pow (CA, 2)*Cf*Tf*nf -\
    61505./9*z3*pow (CA, 2)*pow (Cf, 2) +\
    18335207./1728*z3*pow (CA, 3)*Cf -\
    223./2*z3*xi*(0.333333333333)*d4FA -\
    128./3*z3*xi*pow (Cf, 2)*pow (Tf, 2)*pow (nf, 2) +\
    368./3*z3*xi*pow (Cf, 3)*Tf*nf - 22*z3*xi*pow (Cf, 4) +\
    16*z3*xi*CA*Cf*pow (Tf, 2)*pow (nf, 2) +\
    1180./3*z3*xi*CA*pow (Cf, 2)*Tf*nf + 1963./6*z3*xi*CA*pow (Cf, 3) -\
    386./3*z3*xi*pow (CA, 2)*Cf*Tf*nf -\
    5591./4*z3*xi*pow (CA, 2)*pow (Cf, 2) +\
    34741./96*z3*xi*pow (CA, 3)*Cf +\
    35./4*z3*pow (xi, 2)*(0.333333333333)*d4FA -\
    76./3*z3*pow (xi, 2)*pow (Cf, 3)*Tf*nf -\
    15*z3*pow (xi, 2)*pow (Cf, 4) +\
    68./3*z3*pow (xi, 2)*CA*pow (Cf, 2)*Tf*nf +\
    431./3*z3*pow (xi, 2)*CA*pow (Cf, 3) -\
    28./3*z3*pow (xi, 2)*pow (CA, 2)*Cf*Tf*nf -\
    1393./12*z3*pow (xi, 2)*pow (CA, 2)*pow (Cf, 2) +\
    4637./96*z3*pow (xi, 2)*pow (CA, 3)*Cf +\
    3./2*z3*pow (xi, 3)*(0.333333333333)*d4FA -\
    6*z3*pow (xi, 3)*pow (Cf, 4) + 9*z3*pow (xi, 3)*CA*pow (Cf, 3) -\
    51./4*z3*pow (xi, 3)*pow (CA, 2)*pow (Cf, 2) +\
    201./32*z3*pow (xi, 3)*pow (CA, 3)*Cf +\
    7./8*z3*pow (xi, 4)*(0.333333333333)*d4FA +\
    z3*pow (xi, 4)*CA*pow (Cf, 3) -\
    3./2*z3*pow (xi, 4)*pow (CA, 2)*pow (Cf, 2) +\
    95./192*z3*pow (xi, 4)*pow (CA, 3)*Cf +\
    4935./2*pow (z3, 2)*(0.333333333333)*d4FA -\
    192*pow (z3, 2)*nf*(0.333333333333)*d4FF -\
    200*pow (z3, 2)*pow (Cf, 3)*Tf*nf + 1536*pow (z3, 2)*pow (Cf, 4) +\
    652*pow (z3, 2)*CA*pow (Cf, 2)*Tf*nf -\
    3632*pow (z3, 2)*CA*pow (Cf, 3) -\
    300*pow (z3, 2)*pow (CA, 2)*Cf*Tf*nf +\
    3130*pow (z3, 2)*pow (CA, 2)*pow (Cf, 2) -\
    22617./16*pow (z3, 2)*pow (CA, 3)*Cf -\
    270*pow (z3, 2)*xi*(0.333333333333)*d4FA +\
    24*pow (z3, 2)*xi*CA*pow (Cf, 3) -\
    207./2*pow (z3, 2)*xi*pow (CA, 2)*pow (Cf, 2) +\
    813./16*pow (z3, 2)*xi*pow (CA, 3)*Cf -\
    33./2*pow (z3, 2)*pow (xi, 2)*(0.333333333333)*d4FA -\
    3./2*pow (z3, 2)*pow (xi, 2)*pow (CA, 2)*pow (Cf, 2) +\
    13./8*pow (z3, 2)*pow (xi, 2)*pow (CA, 3)*Cf)
    C_m = np.ones(len(a_s))
    for i in range(len(a_s)):
        for j in range(i):
            C_m[i] += coef[j]*pow(a_s[i]/4,j+1)
    return C_m


def quark_field_conversion_rimom_prime_over_rimom2(scale, xi, Lambda=None, nf=3):
    xi = _finite_real_scalar(xi, "xi")
    z3, z4, z5, z6, z7 = zeta(3), zeta(4), zeta(5), zeta(6), zeta(7)
    Cf = 4.0/3
    CA = 3.0
    Tf = 1.0/2
    d4FA = 7.5
    d4FF = 0.416667
    Nc = 3.0
    a_s = strong_coupling_constant(scale, Lambda, nf)
    # Only coefficients 0..2 below are active. The source's four-loop block is
    # intentionally disabled, so allocate only known coefficients instead of
    # reading an uninitialized coef[3] in the highest truncation.
    coef = np.empty((3))
    coef[0] = float(- (Cf*xi)/2)
    coef[1] = float((Cf*(54*Cf + 8*nf*Tf*(9 + 10*xi) -\
    CA*(225 + 223*xi + 45*pow(xi,2) + 9*pow(xi,3))))/72)
    coef[2] = float(- (Cf*(8*(54*Cf*nf*Tf*(14 - 55*xi + 48*xi*z3) + 243*pow(Cf,2) +\
    16*(117 + 50*xi)*pow(nf,2)*pow(Tf,2)) -\
    8*CA*(27*Cf*(242 - 72*z3 + 9*pow(xi,2) + 3*pow(xi,3)) +\
    4*nf*Tf*(3438 - 324*z3 + 2*xi*(994 + 81*z3) + 225*pow(xi,2) + 45*pow(xi,3))) +\
    pow(CA,2)*(179811 - 39690*z3 - 4*xi*(-28270 + 5751*z3) + 18*(1256 + 63*z3)*pow(xi,2) +\
    7578*pow(xi,3) + 1377*pow(xi,4) + 162*pow(xi,5))))/2592)
    '''coef[3] = float(((57*CA*Cf + 38*xi*CA*Cf + 6*pow(xi,2)*CA*Cf + pow(Cf,2) -
    4*pow(xi,2)*pow(Cf,2) - 20*Cf*nf*Tf - 24*CA*Cf*z3 - 24*xi*CA*Cf*z3)*
    (-82*CA*Cf - 52*xi*CA*Cf - 9*pow(xi,2)*CA*Cf + 5*pow(Cf,2) +
    8*pow(xi,2)*pow(Cf,2) + 28*Cf*nf*Tf + 24*CA*Cf*z3 + 24*xi*CA*Cf*z3))/64. +
    (xi*Cf*(-1274056*pow(CA,2)*Cf - 358191*xi*pow(CA,2)*Cf -
    63747*pow(xi,2)*pow(CA,2)*Cf - 11880*pow(xi,3)*pow(CA,2)*Cf +
    215352*CA*pow(Cf,2) + 85536*xi*CA*pow(Cf,2) +
    59616*pow(xi,2)*CA*pow(Cf,2) + 12312*pow(xi,3)*CA*pow(Cf,2) +
    31536*pow(Cf,3) - 11016*xi*pow(Cf,3) - 5184*pow(xi,3)*pow(Cf,3) +
    760768*CA*Cf*nf*Tf + 124056*xi*CA*Cf*nf*Tf + 68256*pow(Cf,2)*nf*Tf -
    28512*xi*pow(Cf,2)*nf*Tf - 100480*Cf*pow(nf,2)*pow(Tf,2) +
    678024*pow(CA,2)*Cf*z3 + 181440*xi*pow(CA,2)*Cf*z3 +
    25272*pow(xi,2)*pow(CA,2)*Cf*z3 + 1728*pow(xi,3)*pow(CA,2)*Cf*z3 -
    228096*CA*pow(Cf,2)*z3 + 57024*xi*CA*pow(Cf,2)*z3 -
    31104*pow(xi,2)*CA*pow(Cf,2)*z3 - 5184*pow(xi,3)*CA*pow(Cf,2)*z3 +
    3456*pow(xi,3)*pow(Cf,3)*z3 - 89856*CA*Cf*nf*Tf*z3 -
    41472*xi*CA*Cf*nf*Tf*z3 - 82944*pow(Cf,2)*nf*Tf*z3 + 22356*pow(CA,2)*Cf*z4 -
    1944*xi*pow(CA,2)*Cf*z4 - 972*pow(xi,2)*pow(CA,2)*Cf*z4 -
    31104*CA*pow(Cf,2)*z4 - 213840*pow(CA,2)*Cf*z5 -
    12960*xi*pow(CA,2)*Cf*z5 - 6480*pow(xi,2)*pow(CA,2)*Cf*z5 +
    103680*CA*pow(Cf,2)*z5 - 103680*xi*CA*pow(Cf,2)*z5))/10368. -
    (xi*Cf*(457217*pow(CA,2)*Cf + 119790*xi*pow(CA,2)*Cf +
    19926*pow(xi,2)*pow(CA,2)*Cf + 3753*pow(xi,3)*pow(CA,2)*Cf -
    55404*CA*pow(Cf,2) - 11016*xi*CA*pow(Cf,2) -
    7128*pow(xi,2)*CA*pow(Cf,2) - 2106*pow(xi,3)*CA*pow(Cf,2) -
    17712*pow(Cf,3) + 5022*xi*pow(Cf,3) - 270368*CA*Cf*nf*Tf -
    43128*xi*CA*Cf*nf*Tf - 40176*pow(Cf,2)*nf*Tf + 3240*xi*pow(Cf,2)*nf*Tf +
    35264*Cf*pow(nf,2)*pow(Tf,2) - 299322*pow(CA,2)*Cf*z3 -
    71604*xi*pow(CA,2)*Cf*z3 - 11178*pow(xi,2)*pow(CA,2)*Cf*z3 -
    864*pow(xi,3)*pow(CA,2)*Cf*z3 + 98496*CA*pow(Cf,2)*z3 -
    40176*xi*CA*pow(Cf,2)*z3 + 3888*pow(xi,2)*CA*pow(Cf,2)*z3 +
    2592*pow(xi,3)*CA*pow(Cf,2)*z3 - 1728*pow(xi,3)*pow(Cf,3)*z3 +
    34560*CA*Cf*nf*Tf*z3 + 15552*xi*CA*Cf*nf*Tf*z3 + 41472*pow(Cf,2)*nf*Tf*z3 -
    11178*pow(CA,2)*Cf*z4 + 972*xi*pow(CA,2)*Cf*z4 +
    486*pow(xi,2)*pow(CA,2)*Cf*z4 + 15552*CA*pow(Cf,2)*z4 +
    106920*pow(CA,2)*Cf*z5 + 6480*xi*pow(CA,2)*Cf*z5 +
    3240*pow(xi,2)*pow(CA,2)*Cf*z5 - 51840*CA*pow(Cf,2)*z5 +
    51840*xi*CA*pow(Cf,2)*z5))/2592. +
    (169827840*d4FA - 3359232*xi*d4FA - 3936128712*pow(CA,3)*Cf*Nc -
    842973089*xi*pow(CA,3)*Cf*Nc - 137303235*pow(xi,2)*pow(CA,3)*Cf*Nc -
    29024406*pow(xi,3)*pow(CA,3)*Cf*Nc - 4399272*pow(xi,4)*pow(CA,3)*Cf*Nc +
    1182863520*pow(CA,2)*pow(Cf,2)*Nc + 383068800*xi*pow(CA,2)*pow(Cf,2)*Nc +
    148359600*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc +
    32437584*pow(xi,3)*pow(CA,2)*pow(Cf,2)*Nc +
    5239080*pow(xi,4)*pow(CA,2)*pow(Cf,2)*Nc - 732667680*CA*pow(Cf,3)*Nc -
    77962176*xi*CA*pow(Cf,3)*Nc - 18907344*pow(xi,2)*CA*pow(Cf,3)*Nc -
    13436928*pow(xi,3)*CA*pow(Cf,3)*Nc - 3079296*pow(xi,4)*CA*pow(Cf,3)*Nc +
    250746840*pow(Cf,4)*Nc - 3421440*xi*pow(Cf,4)*Nc +
    2706048*pow(xi,2)*pow(Cf,4)*Nc + 746496*pow(xi,4)*pow(Cf,4)*Nc -
    353839104*d4FF*nf + 3260707776*pow(CA,2)*Cf*Nc*nf*Tf +
    498291368*xi*pow(CA,2)*Cf*Nc*nf*Tf +
    33851952*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf + 719195328*CA*pow(Cf,2)*Nc*nf*Tf -
    46660896*xi*CA*pow(Cf,2)*Nc*nf*Tf -
    39776832*pow(xi,2)*CA*pow(Cf,2)*Nc*nf*Tf - 109615680*pow(Cf,3)*Nc*nf*Tf -
    3359232*xi*pow(Cf,3)*Nc*nf*Tf + 5598720*pow(xi,2)*pow(Cf,3)*Nc*nf*Tf -
    862500096*CA*Cf*Nc*pow(nf,2)*pow(Tf,2) -
    60475904*xi*CA*Cf*Nc*pow(nf,2)*pow(Tf,2) -
    265714560*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2) +
    27076608*xi*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2) +
    65713152*Cf*Nc*pow(nf,3)*pow(Tf,3) - 28600128*d4FA*z3 - 188116992*xi*d4FA*z3 -
    2146176*pow(xi,2)*d4FA*z3 + 2985984*pow(xi,3)*d4FA*z3 +
    1632960*pow(xi,4)*d4FA*z3 + 2782825632*pow(CA,3)*Cf*Nc*z3 +
    463870368*xi*pow(CA,3)*Cf*Nc*z3 + 66892392*pow(xi,2)*pow(CA,3)*Cf*Nc*z3 +
    10723104*pow(xi,3)*pow(CA,3)*Cf*Nc*z3 +
    756216*pow(xi,4)*pow(CA,3)*Cf*Nc*z3 +
    517197312*pow(CA,2)*pow(Cf,2)*Nc*z3 -
    115375104*xi*pow(CA,2)*pow(Cf,2)*Nc*z3 -
    89548416*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*z3 -
    19315584*pow(xi,3)*pow(CA,2)*pow(Cf,2)*Nc*z3 -
    2301696*pow(xi,4)*pow(CA,2)*pow(Cf,2)*Nc*z3 -
    3016683648*CA*pow(Cf,3)*Nc*z3 + 106282368*xi*CA*pow(Cf,3)*Nc*z3 +
    2239488*pow(xi,2)*CA*pow(Cf,3)*Nc*z3 +
    9704448*pow(xi,3)*CA*pow(Cf,3)*Nc*z3 +
    1866240*pow(xi,4)*CA*pow(Cf,3)*Nc*z3 + 1451561472*pow(Cf,4)*Nc*z3 -
    2239488*pow(xi,2)*pow(Cf,4)*Nc*z3 - 497664*pow(xi,4)*pow(Cf,4)*Nc*z3 -
    839061504*d4FF*nf*z3 - 1450783872*pow(CA,2)*Cf*Nc*nf*Tf*z3 -
    133429248*xi*pow(CA,2)*Cf*Nc*nf*Tf*z3 -
    10461312*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf*z3 -
    81243648*CA*pow(Cf,2)*Nc*nf*Tf*z3 - 136899072*xi*CA*pow(Cf,2)*Nc*nf*Tf*z3 +
    19906560*pow(xi,2)*CA*pow(Cf,2)*Nc*nf*Tf*z3 +
    76889088*pow(Cf,3)*Nc*nf*Tf*z3 + 5971968*xi*pow(Cf,3)*Nc*nf*Tf*z3 -
    2985984*pow(xi,2)*pow(Cf,3)*Nc*nf*Tf*z3 +
    69672960*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z3 +
    9842688*xi*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z3 +
    198070272*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2)*z3 -
    7962624*xi*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2)*z3 +
    1327104*Cf*Nc*pow(nf,3)*pow(Tf,3)*z3 - 114983712*d4FA*pow(z3,2) +
    1306368*xi*d4FA*pow(z3,2) + 1632960*pow(xi,2)*d4FA*pow(z3,2) +
    2239488*pow(xi,3)*d4FA*pow(z3,2) + 443232*pow(xi,4)*d4FA*pow(z3,2) -
    125912880*pow(CA,3)*Cf*Nc*pow(z3,2) -
    67332384*xi*pow(CA,3)*Cf*Nc*pow(z3,2) +
    1407456*pow(xi,2)*pow(CA,3)*Cf*Nc*pow(z3,2) +
    46656*pow(xi,3)*pow(CA,3)*Cf*Nc*pow(z3,2) +
    19440*pow(xi,4)*pow(CA,3)*Cf*Nc*pow(z3,2) +
    72223488*pow(CA,2)*pow(Cf,2)*Nc*pow(z3,2) +
    44229888*xi*pow(CA,2)*pow(Cf,2)*Nc*pow(z3,2) +
    3359232*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*pow(z3,2) -
    116453376*CA*pow(Cf,3)*Nc*pow(z3,2) + 59719680*pow(Cf,4)*Nc*pow(z3,2) +
    286654464*d4FF*nf*pow(z3,2) - 62705664*pow(CA,2)*Cf*Nc*nf*Tf*pow(z3,2) -
    10450944*xi*pow(CA,2)*Cf*Nc*nf*Tf*pow(z3,2) -
    746496*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf*pow(z3,2) +
    119439360*CA*pow(Cf,2)*Nc*nf*Tf*pow(z3,2) +
    23887872*xi*CA*pow(Cf,2)*Nc*nf*Tf*pow(z3,2) - 38946096*d4FA*z4 -
    14136768*xi*d4FA*z4 - 769824*pow(xi,2)*d4FA*z4 + 419904*pow(xi,3)*d4FA*z4 +
    244944*pow(xi,4)*d4FA*z4 + 27158166*pow(CA,3)*Cf*Nc*z4 -
    8161884*xi*pow(CA,3)*Cf*Nc*z4 - 1933308*pow(xi,2)*pow(CA,3)*Cf*Nc*z4 -
    253692*pow(xi,3)*pow(CA,3)*Cf*Nc*z4 - 1458*pow(xi,4)*pow(CA,3)*Cf*Nc*z4 +
    58926528*pow(CA,2)*pow(Cf,2)*Nc*z4 -
    2239488*xi*pow(CA,2)*pow(Cf,2)*Nc*z4 +
    279936*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*z4 +
    139968*pow(xi,3)*pow(CA,2)*pow(Cf,2)*Nc*z4 -
    237385728*CA*pow(Cf,3)*Nc*z4 + 4478976*xi*CA*pow(Cf,3)*Nc*z4 +
    111974400*pow(Cf,4)*Nc*z4 + 419904*pow(CA,2)*Cf*Nc*nf*Tf*z4 +
    17076096*xi*pow(CA,2)*Cf*Nc*nf*Tf*z4 +
    326592*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf*z4 -
    21275136*xi*CA*pow(Cf,2)*Nc*nf*Tf*z4 + 17915904*pow(Cf,3)*Nc*nf*Tf*z4 -
    8957952*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z4 -
    1492992*xi*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z4 +
    8957952*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2)*z4 - 289617120*d4FA*z5 +
    179159040*xi*d4FA*z5 + 34292160*pow(xi,2)*d4FA*z5 +
    5598720*pow(xi,3)*d4FA*z5 - 2216160*pow(xi,4)*d4FA*z5 -
    1461908088*pow(CA,3)*Cf*Nc*z5 + 66843792*xi*pow(CA,3)*Cf*Nc*z5 -
    14380416*pow(xi,2)*pow(CA,3)*Cf*Nc*z5 -
    1784592*pow(xi,3)*pow(CA,3)*Cf*Nc*z5 -
    68040*pow(xi,4)*pow(CA,3)*Cf*Nc*z5 - 155737728*pow(CA,2)*pow(Cf,2)*Nc*z5 -
    144374400*xi*pow(CA,2)*pow(Cf,2)*Nc*z5 -
    18662400*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*z5 +
    933120*pow(xi,3)*pow(CA,2)*pow(Cf,2)*Nc*z5 +
    304819200*CA*pow(Cf,3)*Nc*z5 - 201553920*xi*CA*pow(Cf,3)*Nc*z5 +
    14929920*pow(xi,2)*CA*pow(Cf,3)*Nc*z5 + 724101120*pow(Cf,4)*Nc*z5 +
    925655040*d4FF*nf*z5 + 661156992*pow(CA,2)*Cf*Nc*nf*Tf*z5 -
    92358144*xi*pow(CA,2)*Cf*Nc*nf*Tf*z5 +
    1544832*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf*z5 -
    88086528*CA*pow(Cf,2)*Nc*nf*Tf*z5 + 85764096*xi*CA*pow(Cf,2)*Nc*nf*Tf*z5 -
    4976640*pow(Cf,3)*Nc*nf*Tf*z5 - 36495360*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z5 +
    19906560*xi*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z5 + 49863600*d4FA*z6 +
    25660800*xi*d4FA*z6 + 4082400*pow(xi,2)*d4FA*z6 - 291600*pow(xi,4)*d4FA*z6 +
    60434100*pow(CA,3)*Cf*Nc*z6 + 11178000*xi*pow(CA,3)*Cf*Nc*z6 +
    923400*pow(xi,2)*pow(CA,3)*Cf*Nc*z6 - 24300*pow(xi,4)*pow(CA,3)*Cf*Nc*z6 -
    366249600*pow(CA,2)*pow(Cf,2)*Nc*z6 -
    2332800*xi*pow(CA,2)*pow(Cf,2)*Nc*z6 + 671846400*CA*pow(Cf,3)*Nc*z6 -
    298598400*pow(Cf,4)*Nc*z6 + 37324800*pow(CA,2)*Cf*Nc*nf*Tf*z6 -
    74649600*CA*pow(Cf,2)*Nc*nf*Tf*z6 + 438286464*d4FA*z7 + 36088416*xi*d4FA*z7 -
    31679424*pow(xi,2)*d4FA*z7 - 10287648*pow(xi,3)*d4FA*z7 +
    514123848*pow(CA,3)*Cf*Nc*z7 + 24603264*xi*pow(CA,3)*Cf*Nc*z7 -
    435456*pow(xi,2)*pow(CA,3)*Cf*Nc*z7 -
    857304*pow(xi,3)*pow(CA,3)*Cf*Nc*z7 -
    1301469120*pow(CA,2)*pow(Cf,2)*Nc*z7 -
    32985792*xi*pow(CA,2)*pow(Cf,2)*Nc*z7 +
    6531840*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*z7 +
    3801530880*CA*pow(Cf,3)*Nc*z7 + 148925952*xi*CA*pow(Cf,3)*Nc*z7 -
    2633637888*pow(Cf,4)*Nc*z7 + 164602368*pow(CA,2)*Cf*Nc*nf*Tf*z7 -
    658409472*CA*pow(Cf,2)*Nc*nf*Tf*z7)/(746496.*Nc) +
    (-144820224*d4FA + 2985984*xi*d4FA + 2620877064*pow(CA,3)*Cf*Nc +
    552068009*xi*pow(CA,3)*Cf*Nc + 83106459*pow(xi,2)*pow(CA,3)*Cf*Nc +
    17860014*pow(xi,3)*pow(CA,3)*Cf*Nc + 2713824*pow(xi,4)*pow(CA,3)*Cf*Nc -
    704859840*pow(CA,2)*pow(Cf,2)*Nc - 69254496*xi*pow(CA,2)*pow(Cf,2)*Nc -
    38214504*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc -
    10001880*pow(xi,3)*pow(CA,2)*pow(Cf,2)*Nc -
    1879848*pow(xi,4)*pow(CA,2)*pow(Cf,2)*Nc + 595681776*CA*pow(Cf,3)*Nc +
    39945312*xi*CA*pow(Cf,3)*Nc - 93312*pow(xi,2)*CA*pow(Cf,3)*Nc +
    559872*pow(xi,3)*CA*pow(Cf,3)*Nc + 419904*pow(xi,4)*CA*pow(Cf,3)*Nc -
    202889448*pow(Cf,4)*Nc - 3950208*xi*pow(Cf,4)*Nc -
    326592*pow(xi,2)*pow(Cf,4)*Nc + 306063360*d4FF*nf -
    2083649472*pow(CA,2)*Cf*Nc*nf*Tf - 326170088*xi*pow(CA,2)*Cf*Nc*nf*Tf -
    20554992*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf - 736307712*CA*pow(Cf,2)*Nc*nf*Tf -
    81376992*xi*CA*pow(Cf,2)*Nc*nf*Tf +
    10881216*pow(xi,2)*CA*pow(Cf,2)*Nc*nf*Tf + 90051264*pow(Cf,3)*Nc*nf*Tf -
    11632896*xi*pow(Cf,3)*Nc*nf*Tf + 559872*pow(xi,2)*pow(Cf,3)*Nc*nf*Tf +
    534249216*CA*Cf*Nc*pow(nf,2)*pow(Tf,2) +
    39277568*xi*CA*Cf*Nc*pow(nf,2)*pow(Tf,2) +
    229530240*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2) -
    7695360*xi*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2) -
    38065152*Cf*Nc*pow(nf,3)*pow(Tf,3) - 23328000*d4FA*z3 + 169267968*xi*d4FA*z3 +
    1119744*pow(xi,2)*d4FA*z3 - 2426112*pow(xi,3)*d4FA*z3 -
    1306368*pow(xi,4)*d4FA*z3 - 2184238872*pow(CA,3)*Cf*Nc*z3 -
    342164304*xi*pow(CA,3)*Cf*Nc*z3 - 49174776*pow(xi,2)*pow(CA,3)*Cf*Nc*z3 -
    8378640*pow(xi,3)*pow(CA,3)*Cf*Nc*z3 -
    571536*pow(xi,4)*pow(CA,3)*Cf*Nc*z3 -
    658191744*pow(CA,2)*pow(Cf,2)*Nc*z3 -
    36288000*xi*pow(CA,2)*pow(Cf,2)*Nc*z3 +
    21617280*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*z3 +
    7278336*pow(xi,3)*pow(CA,2)*pow(Cf,2)*Nc*z3 +
    1368576*pow(xi,4)*pow(CA,2)*pow(Cf,2)*Nc*z3 +
    2701289088*CA*pow(Cf,3)*Nc*z3 - 60372864*xi*CA*pow(Cf,3)*Nc*z3 -
    14556672*pow(xi,2)*CA*pow(Cf,3)*Nc*z3 -
    1866240*pow(xi,3)*CA*pow(Cf,3)*Nc*z3 -
    373248*pow(xi,4)*CA*pow(Cf,3)*Nc*z3 - 1302262272*pow(Cf,4)*Nc*z3 +
    2239488*pow(xi,2)*pow(Cf,4)*Nc*z3 - 248832*pow(xi,4)*pow(Cf,4)*Nc*z3 +
    839061504*d4FF*nf*z3 + 1169976960*pow(CA,2)*Cf*Nc*nf*Tf*z3 +
    83414016*xi*pow(CA,2)*Cf*Nc*nf*Tf*z3 +
    6977664*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf*z3 +
    85722624*CA*pow(Cf,2)*Nc*nf*Tf*z3 + 122964480*xi*CA*pow(Cf,2)*Nc*nf*Tf*z3 -
    10948608*pow(xi,2)*CA*pow(Cf,2)*Nc*nf*Tf*z3 -
    53001216*pow(Cf,3)*Nc*nf*Tf*z3 + 11943936*xi*pow(Cf,3)*Nc*nf*Tf*z3 +
    2985984*pow(xi,2)*pow(Cf,3)*Nc*nf*Tf*z3 -
    55738368*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z3 -
    3870720*xi*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z3 -
    162238464*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2)*z3 +
    7962624*xi*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2)*z3 -
    1327104*Cf*Nc*pow(nf,3)*pow(Tf,3)*z3 + 114983712*d4FA*pow(z3,2) -
    1306368*xi*d4FA*pow(z3,2) - 1632960*pow(xi,2)*d4FA*pow(z3,2) -
    2239488*pow(xi,3)*d4FA*pow(z3,2) - 443232*pow(xi,4)*d4FA*pow(z3,2) +
    125912880*pow(CA,3)*Cf*Nc*pow(z3,2) +
    67332384*xi*pow(CA,3)*Cf*Nc*pow(z3,2) -
    1407456*pow(xi,2)*pow(CA,3)*Cf*Nc*pow(z3,2) -
    46656*pow(xi,3)*pow(CA,3)*Cf*Nc*pow(z3,2) -
    19440*pow(xi,4)*pow(CA,3)*Cf*Nc*pow(z3,2) -
    65505024*pow(CA,2)*pow(Cf,2)*Nc*pow(z3,2) -
    30792960*xi*pow(CA,2)*pow(Cf,2)*Nc*pow(z3,2) +
    3359232*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*pow(z3,2) +
    116453376*CA*pow(Cf,3)*Nc*pow(z3,2) - 59719680*pow(Cf,4)*Nc*pow(z3,2) -
    286654464*d4FF*nf*pow(z3,2) + 62705664*pow(CA,2)*Cf*Nc*nf*Tf*pow(z3,2) +
    10450944*xi*pow(CA,2)*Cf*Nc*nf*Tf*pow(z3,2) +
    746496*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf*pow(z3,2) -
    119439360*CA*pow(Cf,2)*Nc*nf*Tf*pow(z3,2) -
    23887872*xi*CA*pow(Cf,2)*Nc*nf*Tf*pow(z3,2) + 38946096*d4FA*z4 +
    14136768*xi*d4FA*z4 + 769824*pow(xi,2)*d4FA*z4 - 419904*pow(xi,3)*d4FA*z4 -
    244944*pow(xi,4)*d4FA*z4 - 27158166*pow(CA,3)*Cf*Nc*z4 +
    8056908*xi*pow(CA,3)*Cf*Nc*z4 + 1793340*pow(xi,2)*pow(CA,3)*Cf*Nc*z4 +
    218700*pow(xi,3)*pow(CA,3)*Cf*Nc*z4 + 1458*pow(xi,4)*pow(CA,3)*Cf*Nc*z4 -
    58926528*pow(CA,2)*pow(Cf,2)*Nc*z4 -
    2589408*xi*pow(CA,2)*pow(Cf,2)*Nc*z4 +
    139968*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*z4 +
    69984*pow(xi,3)*pow(CA,2)*pow(Cf,2)*Nc*z4 + 237385728*CA*pow(Cf,3)*Nc*z4 +
    2239488*xi*CA*pow(Cf,3)*Nc*z4 - 111974400*pow(Cf,4)*Nc*z4 -
    419904*pow(CA,2)*Cf*Nc*nf*Tf*z4 - 13716864*xi*pow(CA,2)*Cf*Nc*nf*Tf*z4 -
    326592*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf*z4 +
    16796160*xi*CA*pow(Cf,2)*Nc*nf*Tf*z4 - 17915904*pow(Cf,3)*Nc*nf*Tf*z4 +
    8957952*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z4 +
    1492992*xi*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z4 -
    8957952*pow(Cf,2)*Nc*pow(nf,2)*pow(Tf,2)*z4 + 329508000*d4FA*z5 -
    158630400*xi*d4FA*z5 - 31026240*pow(xi,2)*d4FA*z5 -
    5598720*pow(xi,3)*d4FA*z5 + 1982880*pow(xi,4)*d4FA*z5 +
    1340894088*pow(CA,3)*Cf*Nc*z5 - 66143952*xi*pow(CA,3)*Cf*Nc*z5 +
    11542176*pow(xi,2)*pow(CA,3)*Cf*Nc*z5 +
    1318032*pow(xi,3)*pow(CA,3)*Cf*Nc*z5 +
    48600*pow(xi,4)*pow(CA,3)*Cf*Nc*z5 - 55147392*pow(CA,2)*pow(Cf,2)*Nc*z5 +
    122757120*xi*pow(CA,2)*pow(Cf,2)*Nc*z5 +
    17729280*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*z5 +
    466560*pow(xi,3)*pow(CA,2)*pow(Cf,2)*Nc*z5 +
    232657920*CA*pow(Cf,3)*Nc*z5 + 179159040*xi*CA*pow(Cf,3)*Nc*z5 +
    7464960*pow(xi,2)*CA*pow(Cf,3)*Nc*z5 - 962979840*pow(Cf,4)*Nc*z5 -
    925655040*d4FF*nf*z5 - 569711232*pow(CA,2)*Cf*Nc*nf*Tf*z5 +
    94846464*xi*pow(CA,2)*Cf*Nc*nf*Tf*z5 -
    922752*pow(xi,2)*pow(CA,2)*Cf*Nc*nf*Tf*z5 -
    1492992*CA*pow(Cf,2)*Nc*nf*Tf*z5 - 65857536*xi*CA*pow(Cf,2)*Nc*nf*Tf*z5 +
    4976640*pow(Cf,3)*Nc*nf*Tf*z5 + 36495360*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z5 -
    19906560*xi*CA*Cf*Nc*pow(nf,2)*pow(Tf,2)*z5 - 49863600*d4FA*z6 -
    25660800*xi*d4FA*z6 - 4082400*pow(xi,2)*d4FA*z6 + 291600*pow(xi,4)*d4FA*z6 -
    60434100*pow(CA,3)*Cf*Nc*z6 - 11178000*xi*pow(CA,3)*Cf*Nc*z6 -
    923400*pow(xi,2)*pow(CA,3)*Cf*Nc*z6 + 24300*pow(xi,4)*pow(CA,3)*Cf*Nc*z6 +
    366249600*pow(CA,2)*pow(Cf,2)*Nc*z6 +
    2332800*xi*pow(CA,2)*pow(Cf,2)*Nc*z6 - 671846400*CA*pow(Cf,3)*Nc*z6 +
    298598400*pow(Cf,4)*Nc*z6 - 37324800*pow(CA,2)*Cf*Nc*nf*Tf*z6 +
    74649600*CA*pow(Cf,2)*Nc*nf*Tf*z6 - 438286464*d4FA*z7 - 36088416*xi*d4FA*z7 +
    31679424*pow(xi,2)*d4FA*z7 + 10287648*pow(xi,3)*d4FA*z7 -
    514123848*pow(CA,3)*Cf*Nc*z7 - 24603264*xi*pow(CA,3)*Cf*Nc*z7 +
    435456*pow(xi,2)*pow(CA,3)*Cf*Nc*z7 +
    857304*pow(xi,3)*pow(CA,3)*Cf*Nc*z7 +
    1301469120*pow(CA,2)*pow(Cf,2)*Nc*z7 +
    32985792*xi*pow(CA,2)*pow(Cf,2)*Nc*z7 -
    6531840*pow(xi,2)*pow(CA,2)*pow(Cf,2)*Nc*z7 -
    3801530880*CA*pow(Cf,3)*Nc*z7 - 148925952*xi*CA*pow(Cf,3)*Nc*z7 +
    2633637888*pow(Cf,4)*Nc*z7 - 164602368*pow(CA,2)*Cf*Nc*nf*Tf*z7 +
    658409472*CA*pow(Cf,2)*Nc*nf*Tf*z7)/(746496.*Nc))'''
    C_q = np.ones(len(a_s))
    for i in range(len(a_s)):
        if i > len(coef):
            # No four-loop coefficient is available in the supplied source.
            # Preserve the highest fully known truncation instead of mixing a
            # five-loop coupling with an unknown conversion coefficient.
            C_q[i] = C_q[len(coef)]
            continue
        for j in range(min(i, len(coef))):
            C_q[i] += coef[j]*pow(a_s[i]/4,j+1)
    return C_q


def scalar_conversion_ms_bar_over_rimom_prime2(scale, xi, Lambda=None, nf=3):
    xi = _finite_real_scalar(xi, "xi")
    a_s = quark_mass_conversion_ms_bar_over_rimom_prime2(scale, xi, Lambda, nf)
    for i in range(len(a_s)):
        a_s[i]=1.0/a_s[i]
    return a_s


def tensor_conversion_ms_bar_over_rimom_prime2(scale, xi, Lambda=None, nf=3):
    xi = _finite_real_scalar(xi, "xi")
    z3, z4, z5, z6, z7 = zeta(3), zeta(4), zeta(5), zeta(6), zeta(7)
    Cf = 4.0/3
    CA = 3.0
    Tf = 1.0/2
    d4FA = 7.5
    d4FF = 0.416667
    a_s = strong_coupling_constant(scale, Lambda, nf)
    coef = np.empty((len(a_s)))
    coef[0] = float(-Cf*xi)
    coef[1] = float(- (Cf*(-4*nf*Tf*(313 + 120*xi) + 45*Cf*(-107 + 96*z3) + CA*(5987 + 1338*xi - 3024*z3 + 270*pow(xi,2) + 54*pow(xi,3))))/216)
    coef[2] = float((-14094322*pow(CA,2)*Cf + 18830268*CA*pow(Cf,2) - 4390632*pow(Cf,3) +\
    5968864*CA*Cf*nf*Tf - 3124512*pow(Cf,2)*nf*Tf -\
    440128*Cf*pow(nf,2)*pow(Tf,2) - 1092771*pow(CA,2)*Cf*xi +\
    788508*CA*pow(Cf,2)*xi - 531684*pow(Cf,3)*xi + 340200*CA*Cf*nf*Tf*xi -\
    150768*pow(Cf,2)*nf*Tf*xi - 215055*pow(CA,2)*Cf*pow(xi,2) +\
    58320*CA*pow(Cf,2)*pow(xi,2) - 46656*pow(Cf,3)*pow(xi,2) -\
    39366*pow(CA,2)*Cf*pow(xi,3) + 11664*CA*pow(Cf,2)*pow(xi,3) +\
    8876196*pow(CA,2)*Cf*z3 - 15505344*CA*pow(Cf,2)*z3 + 4017600*pow(Cf,3)*z3 -\
    1900800*CA*Cf*nf*Tf*z3 + 2239488*pow(Cf,2)*nf*Tf*z3 -\
    27648*Cf*pow(nf,2)*pow(Tf,2)*z3 + 324648*pow(CA,2)*Cf*xi*z3 -\
    614304*CA*pow(Cf,2)*xi*z3 + 513216*pow(Cf,3)*xi*z3 - 93312*CA*Cf*nf*Tf*xi*z3 +\
    62208*pow(Cf,2)*nf*Tf*xi*z3 + 26244*pow(CA,2)*Cf*pow(xi,2)*z3 -\
    23328*CA*pow(Cf,2)*pow(xi,2)*z3 + 46656*pow(Cf,3)*pow(xi,2)*z3 +\
    466560*pow(CA,2)*Cf*z4 - 1306368*CA*pow(Cf,2)*z4 + 746496*pow(Cf,3)*z4 +\
    186624*CA*Cf*nf*Tf*z4 - 186624*pow(Cf,2)*nf*Tf*z4 - 1321920*pow(CA,2)*Cf*z5 +\
    1399680*CA*pow(Cf,2)*z5 - 311040*pow(Cf,3)*z5 + 77760*pow(CA,2)*Cf*xi*z5)/\
    23328.)
    coef[3] = float((-7769141516*pow(CA,3)*Cf + 13044283638*pow(CA,2)*pow(Cf,2) -\
    5906379708*CA*pow(Cf,3) + 1090674882*pow(Cf,4) + 7174656*d4FA -\
    112845312*d4FF*nf + 5293901856*pow(CA,2)*Cf*nf*Tf -\
    4652127888*CA*pow(Cf,2)*nf*Tf + 484101360*pow(Cf,3)*nf*Tf -\
    1012343136*CA*Cf*pow(nf,2)*pow(Tf,2) +\
    301037280*pow(Cf,2)*pow(nf,2)*pow(Tf,2) +\
    51645184*Cf*pow(nf,3)*pow(Tf,3) - 457740324*pow(CA,3)*Cf*xi +\
    569620464*pow(CA,2)*pow(Cf,2)*xi - 623352240*CA*pow(Cf,3)*xi +\
    129169728*pow(Cf,4)*xi - 124416*d4FA*xi + 264983328*pow(CA,2)*Cf*nf*Tf*xi -\
    163128192*CA*pow(Cf,2)*nf*Tf*xi + 109451520*pow(Cf,3)*nf*Tf*xi -\
    31797504*CA*Cf*pow(nf,2)*pow(Tf,2)*xi +\
    15705600*pow(Cf,2)*pow(nf,2)*pow(Tf,2)*xi -\
    82158300*pow(CA,3)*Cf*pow(xi,2) +\
    55312632*pow(CA,2)*pow(Cf,2)*pow(xi,2) -\
    49958856*CA*pow(Cf,3)*pow(xi,2) + 11757312*pow(Cf,4)*pow(xi,2) +\
    19945440*pow(CA,2)*Cf*nf*Tf*pow(xi,2) -\
    10303200*CA*pow(Cf,2)*nf*Tf*pow(xi,2) + 4727808*pow(Cf,3)*nf*Tf*pow(xi,2) -\
    16746588*pow(CA,3)*Cf*pow(xi,3) + 8398080*pow(CA,2)*pow(Cf,2)*pow(xi,3) -\
    4199040*CA*pow(Cf,3)*pow(xi,3) + 1119744*pow(Cf,4)*pow(xi,3) -\
    2528172*pow(CA,3)*Cf*pow(xi,4) + 1189728*pow(CA,2)*pow(Cf,2)*pow(xi,4) -\
    279936*CA*pow(Cf,3)*pow(xi,4) + 5427947484*pow(CA,3)*Cf*z3 -\
    10302090624*pow(CA,2)*pow(Cf,2)*z3 + 3471501888*CA*pow(Cf,3)*z3 +\
    406342656*pow(Cf,4)*z3 - 161898912*d4FA*z3 + 97293312*d4FF*nf*z3 -\
    2217673728*pow(CA,2)*Cf*nf*Tf*z3 + 2863641600*CA*pow(Cf,2)*nf*Tf*z3 -\
    188179200*pow(Cf,3)*nf*Tf*z3 + 160496640*CA*Cf*pow(nf,2)*pow(Tf,2)*z3 -\
    134618112*pow(Cf,2)*pow(nf,2)*pow(Tf,2)*z3 +\
    1437696*Cf*pow(nf,3)*pow(Tf,3)*z3 + 123245064*pow(CA,3)*Cf*xi*z3 -\
    339593472*pow(CA,2)*pow(Cf,2)*xi*z3 + 534086784*CA*pow(Cf,3)*xi*z3 -\
    121056768*pow(Cf,4)*xi*z3 - 6998400*d4FA*xi*z3 -\
    59263488*pow(CA,2)*Cf*nf*Tf*xi*z3 + 74400768*CA*pow(Cf,2)*nf*Tf*xi*z3 -\
    88584192*pow(Cf,3)*nf*Tf*xi*z3 + 8957952*CA*Cf*pow(nf,2)*pow(Tf,2)*xi*z3 -\
    7962624*pow(Cf,2)*pow(nf,2)*pow(Tf,2)*xi*z3 +\
    26117640*pow(CA,3)*Cf*pow(xi,2)*z3 -\
    35474112*pow(CA,2)*pow(Cf,2)*pow(xi,2)*z3 +\
    49206528*CA*pow(Cf,3)*pow(xi,2)*z3 - 11757312*pow(Cf,4)*pow(xi,2)*z3 -\
    1041984*d4FA*pow(xi,2)*z3 - 5225472*pow(CA,2)*Cf*nf*Tf*pow(xi,2)*z3 +\
    5723136*CA*pow(Cf,2)*nf*Tf*pow(xi,2)*z3 -\
    4727808*pow(Cf,3)*nf*Tf*pow(xi,2)*z3 + 3516696*pow(CA,3)*Cf*pow(xi,3)*z3 -\
    5878656*pow(CA,2)*pow(Cf,2)*pow(xi,3)*z3 +\
    5038848*CA*pow(Cf,3)*pow(xi,3)*z3 - 1119744*pow(Cf,4)*pow(xi,3)*z3 +\
    279936*d4FA*pow(xi,3)*z3 + 277020*pow(CA,3)*Cf*pow(xi,4)*z3 -\
    839808*pow(CA,2)*pow(Cf,2)*pow(xi,4)*z3 +\
    559872*CA*pow(Cf,3)*pow(xi,4)*z3 + 163296*d4FA*pow(xi,4)*z3 -\
    859248*pow(CA,3)*Cf*pow(z3,2) - 663541632*pow(CA,2)*pow(Cf,2)*pow(z3,2) +\
    1204098048*CA*pow(Cf,3)*pow(z3,2) - 355332096*pow(Cf,4)*pow(z3,2) +\
    236794752*d4FA*pow(z3,2) - 155271168*d4FF*nf*pow(z3,2) -\
    179159040*pow(CA,2)*Cf*nf*Tf*pow(z3,2) +\
    575548416*CA*pow(Cf,2)*nf*Tf*pow(z3,2) -\
    443418624*pow(Cf,3)*nf*Tf*pow(z3,2) - 15524784*pow(CA,3)*Cf*xi*pow(z3,2) +\
    32192640*pow(CA,2)*pow(Cf,2)*xi*pow(z3,2) -\
    22394880*CA*pow(Cf,3)*xi*pow(z3,2) + 373248*d4FA*xi*pow(z3,2) +\
    1492992*pow(CA,2)*Cf*nf*Tf*xi*pow(z3,2) -\
    93312*pow(CA,3)*Cf*pow(xi,2)*pow(z3,2) +\
    1026432*d4FA*pow(xi,2)*pow(z3,2) + 97962048*pow(CA,3)*Cf*z4 -\
    228614400*pow(CA,2)*pow(Cf,2)*z4 - 23607936*CA*pow(Cf,3)*z4 +\
    139968000*pow(Cf,4)*z4 - 4852224*d4FA*z4 + 2239488*d4FF*nf*z4 +\
    68584320*pow(CA,2)*Cf*nf*Tf*z4 - 56360448*CA*pow(Cf,2)*nf*Tf*z4 +\
    186624*pow(Cf,3)*nf*Tf*z4 - 17169408*CA*Cf*pow(nf,2)*pow(Tf,2)*z4 +\
    17169408*pow(Cf,2)*pow(nf,2)*pow(Tf,2)*z4 -\
    995328*Cf*pow(nf,3)*pow(Tf,3)*z4 - 157464*pow(CA,3)*Cf*xi*z4 -\
    11197440*pow(CA,2)*pow(Cf,2)*xi*z4 + 31352832*CA*pow(Cf,3)*xi*z4 -\
    17915904*pow(Cf,4)*xi*z4 + 5038848*pow(CA,2)*Cf*nf*Tf*xi*z4 -\
    11197440*CA*pow(Cf,2)*nf*Tf*xi*z4 + 4478976*pow(Cf,3)*nf*Tf*xi*z4 -\
    209952*pow(CA,3)*Cf*pow(xi,2)*z4 - 52488*pow(CA,3)*Cf*pow(xi,3)*z4 -\
    307910160*pow(CA,3)*Cf*z5 - 907933536*pow(CA,2)*pow(Cf,2)*z5 +\
    2880479232*CA*pow(Cf,3)*z5 - 2238243840*pow(Cf,4)*z5 + 943993440*d4FA*z5 -\
    126904320*d4FF*nf*z5 - 271216512*pow(CA,2)*Cf*nf*Tf*z5 +\
    535735296*CA*pow(Cf,2)*nf*Tf*z5 + 97293312*pow(Cf,3)*nf*Tf*z5 +\
    103845888*CA*Cf*pow(nf,2)*pow(Tf,2)*z5 -\
    140341248*pow(Cf,2)*pow(nf,2)*pow(Tf,2)*z5 + 120680280*pow(CA,3)*Cf*xi*z5 -\
    96228000*pow(CA,2)*pow(Cf,2)*xi*z5 - 74649600*CA*pow(Cf,3)*xi*z5 +\
    7464960*pow(Cf,4)*xi*z5 + 777600*d4FA*xi*z5 -\
    26956800*pow(CA,2)*Cf*nf*Tf*xi*z5 + 33592320*CA*pow(Cf,2)*nf*Tf*xi*z5 -\
    1807920*pow(CA,3)*Cf*pow(xi,2)*z5 -\
    5365440*pow(CA,2)*pow(Cf,2)*pow(xi,2)*z5 + 3032640*d4FA*pow(xi,2)*z5 +\
    933120*pow(CA,2)*Cf*nf*Tf*pow(xi,2)*z5 - 699840*pow(CA,3)*Cf*pow(xi,3)*z5 -\
    29160*pow(CA,3)*Cf*pow(xi,4)*z5 - 116640*d4FA*pow(xi,4)*z5 +\
    62208000*pow(CA,3)*Cf*z6 - 270604800*pow(CA,2)*pow(Cf,2)*z6 +\
    569203200*CA*pow(Cf,3)*z6 - 373248000*pow(Cf,4)*z6 + 24883200*d4FA*z6 -\
    46656000*pow(CA,2)*Cf*nf*Tf*z6 - 9331200*CA*pow(Cf,2)*nf*Tf*z6 +\
    55987200*pow(Cf,3)*nf*Tf*z6 - 226816443*pow(CA,3)*Cf*z7 +\
    2066061816*pow(CA,2)*pow(Cf,2)*z7 - 3784548096*CA*pow(Cf,3)*z7 +\
    1706116608*pow(Cf,4)*z7 - 989709840*d4FA*z7 + 329204736*d4FF*nf*z7 +\
    20575296*pow(CA,2)*Cf*nf*Tf*z7 + 891324*pow(CA,3)*Cf*xi*z7 -\
    61644240*pow(CA,2)*pow(Cf,2)*xi*z7 + 82301184*CA*pow(Cf,3)*xi*z7 +\
    12791520*d4FA*xi*z7 + 290871*pow(CA,3)*Cf*pow(xi,2)*z7 -\
    2571912*pow(CA,2)*pow(Cf,2)*pow(xi,2)*z7 - 1877904*d4FA*pow(xi,2)*z7)/\
    559872.)
    C_T = np.ones(len(a_s))
    for i in range(len(a_s)):
        for j in range(i):
            C_T[i] += coef[j]*pow(a_s[i]/4,j+1)
    return C_T



######################################### MSbar Matching 2GeV Running #########################################

def anomalous_dimension(gamma, nf):
    gamma = _finite_real_array(gamma, "gamma").copy()
    if gamma.ndim != 1 or not 3 <= gamma.size <= 5:
        raise ValueError("gamma must contain one vector of three to five coefficients")
    nf = _validate_nf(nf)
    beta = beta_coupling_constant(nf)
    b0 = beta[0]
    b1 = beta[1]
    b2 = beta[2]
    b3 = beta[3]
    b4 = beta[4]
    for i in range(len(gamma)):
        gamma[i]/=b0
    b0 = b0/b0
    r0 = gamma[0]
    r1 = gamma[1]
    r2 = gamma[2]
    d1 = -b1*r0 + b0*r1
    d2 = b1*b1*r0 - b0*b2*r0 - b0*b1*r1 + b0*b0*r2
    dim = np.empty((len(gamma)))
    dim[0] = r0
    dim[1] = d1
    dim[2] = (d1*d1 + d2)/2.0
    if len(gamma) >= 4 :
        r3 = gamma[3]
        d3 = -b1*b1*b1*r0 + b0*b1*b1*r1 + b0*b1*(2*b2*r0 - b0*r2) + b0*b0*(-b3*r0 - b2*r1 + b0*r3)
        dim[3] = (d1*d2 + d1*(d1*d1 + d2)/2.0 + d3)/3.0
    if len(gamma) >= 5 :
        r4 = gamma[4]
        d4 = pow(b1,4)*r0 - b0*b1*b1*b1*r1 + b0*b1*b1*(-3*b2*r0+b0*r2) + b0*b0*b1*(2*b3*r0+2*b2*r1-b0*r3) + b0*b0*(b2*b2*r0 - b0*b2*r2 - b0*b3*r1)
        dim[4] = (d2*(d1*d1 + d2)/2.0 + d1*d3 + d1*(d1*d2 +d1*(d1*d1 + d2)/2.0 + d3)/3.0 + d4 + b0*(b0*r4-b4*r0))/4.0
    return dim


def scale_running(scale, scale0, dim, Lambda, nf):
    dim = _finite_real_array(dim, "dim")
    if dim.ndim != 1 or not 3 <= dim.size <= 5:
        raise ValueError("dim must contain one vector of three to five coefficients")
    scale = _finite_real_scalar(scale, "scale", positive=True)
    scale0 = _finite_real_scalar(scale0, "scale0", positive=True)
    Lambda, nf = _resolve_lambda(Lambda, nf)
    a_s = strong_coupling_constant(scale, Lambda, nf)[4]
    a_2 = strong_coupling_constant(scale0, Lambda, nf)[4]
    gamma_m = np.empty((len(dim)))
    c_s0 = pow(a_s, dim[0])
    c_20 = pow(a_2, dim[0])
    gamma_m[0] = c_20 / c_s0
    c_s1 = c_s0*(1+dim[1]*a_s)
    c_21 = c_20*(1+dim[1]*a_2)
    gamma_m[1] = c_21 / c_s1
    c_s2 = c_s1 + c_s0*dim[2]*a_s*a_s
    c_22 = c_21 + c_20*dim[2]*a_2*a_2
    gamma_m[2] = c_22 / c_s2
    if len(dim) >= 4 :
        c_s3 = c_s2 + c_s0*dim[3]*a_s*a_s*a_s
        c_23 = c_22 + c_20*dim[3]*a_2*a_2*a_2
        gamma_m[3] = c_23 / c_s3
    if len(dim) == 5 :
        c_s4 = c_s3 + c_s0*dim[4]*pow(a_s,4)
        c_24 = c_23 + c_20*dim[4]*pow(a_2,4)
        gamma_m[4] = c_24 / c_s4
    return gamma_m


def quark_mass_anomalous_dimension_under_ms_bar(scale, scale0=2.0, Lambda=None, nf=3):
    z3, z4, z5, z6, z7 = zeta(3), zeta(4), zeta(5), zeta(6), zeta(7)
    r0 = float(1)
    r1 = float((202.0/3 - 20.0*nf/9)/16)
    r2 = float((1249.0 - (2216.0/27+160.0*z3/3)*nf - 140.0*nf*nf/81)/64)
    r3 = float((4603055.0/162 + 135680.0*z3/27 - 8800.0*z5 + (-91723.0/27 - 34192.0*z3/9 + 880.0*z4 + 18400.0*z5/9)*nf
    +(5242.0/243 + 800.0*z3/9 - 160.0*z4/3)*nf*nf + (-332.0/243+64.0*z3/27)*nf*nf*nf)/256)
    r4 = float((99512327.0/162 + 46402466.0*z3/243 + 96800.0*z3*z3 - 698126.0*z4/9 - 231757160.0*z5/243 + 242000.0*z6 + 412720.0*z7
    +(-150736283.0/1458 - 12538016.0*z3/81 - 75680.0*z3*z3/9 + 2038742.0*z4/27 + 49876180.0*z5/243 - 638000.0*z6/9 - 1820000.0*z7/27)*nf
    +(1320742.0/729 + 2010824.0*z3/243 + 46400.0*z3*z3/27 - 166300.0*z4/27 - 264040.0*z5/81 + 92000.0*z6/27)*nf*nf
    +(91865.0/1458 + 12848.0*z3/81 + 448.0*z4/9 - 5120.0*z5/27)*nf*nf*nf + (-260.0/243 - 320.0*z3/243 + 64.0*z4/27)*nf*nf*nf*nf)/1024)
    gamma = np.array([r0, r1, r2, r3, r4])
    dim=anomalous_dimension(gamma, nf)
    return scale_running(scale, scale0, dim, Lambda, nf)


def quark_field_anomalous_dimension_under_ms_bar(scale, scale0=2.0, Lambda=None, nf=3):
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    r0 = 0
    r1 = (67.0/3.0 - 4.0*nf/3.0)/16.0
    r2 = ((20729.0/36 - 79.0*z3/2) - 550.0*nf/9 + 20.0*nf*nf/27)/64
    r3 = (2109389.0/162 - 565939.0*z3/324 + 2607.0*z4/4 - 761525.0*z5/1296
    -(162103.0/81 + 2291.0*z3/27+79.0*z4/2+160.0*z5/3)*nf
    +(3853.0/81 + 160.0*z3/9)*nf*nf + 140.0*nf*nf*nf/243)/256
    gamma = np.array([r0*2, r1*2, r2*2, r3*2, 0])
    dim=anomalous_dimension(gamma, nf)
    return scale_running(scale, scale0, dim, Lambda, nf)


def scalar_anomalous_dimension_under_ms_bar(scale, scale0=2.0, Lambda=None, nf=3):
    gamma_m = quark_mass_anomalous_dimension_under_ms_bar(scale, scale0, Lambda, nf)
    for i in range(len(gamma_m)):
        gamma_m[i] = 1.0/gamma_m[i]
    return gamma_m


def tensor_anomalous_dimension_under_ms_bar(scale, scale0=2.0, Lambda=None, nf=3):
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    r0 = float(1.0/3)
    r1 = float((1086.0/27 - 52.0*nf/27)/16)
    r2 = float((52555.0/81 - 4.0*nf*nf/9 - 160.0*z3*nf/9 - 5240.0*nf/81 - 928.0*z3/27)/64)
    r3 = float((2208517.0/162 - 123728.0*z3/243 + 5104.0*z4/9 - 669760.0*z5/243
    +(-1537379.0/729 - 303664.0*z3/243 + 6992.0*z4/27 + 18400.0*z5/27)*nf
    +(19922.0/729 + 3680.0*z3/81 - 160.0*z4/9)*nf*nf + (28.0/243+64.0*z3/81)*nf*nf*nf)/256)
    gamma = np.array([r0, r1, r2, r3, 0])
    dim=anomalous_dimension(gamma,nf)
    return scale_running(scale,scale0,dim,Lambda,nf)



######################################### Systematic error analysis #########################################

def matching_systematic_error(mu, MOM_flag, error_flag, scale0=2.0, Lambda=None, nf=3):
    #error_flag 0-3: statistical, truncation, perturbative_running, lmabda_QCD
    MOM_flag = _nonboolean_integer(MOM_flag, "MOM_flag", allowed={0, 1})
    error_flag = _nonboolean_integer(
        error_flag, "error_flag", allowed={0, 1, 2, 3}
    )
    mu = _finite_real_scalar(mu, "mu", positive=True)
    scale0 = _finite_real_scalar(scale0, "scale0", positive=True)
    Lambda, nf = _resolve_lambda(Lambda, nf)
    if (scale0, Lambda, nf) != (2.0, 0.332, 3):
        raise NotImplementedError(
            "this legacy systematic-error table is currently defined only for "
            "scale0=2.0, Lambda=0.332, nf=3"
        )
    alpha_s = strong_coupling_constant(mu,Lambda,nf)[4]*3.1415926535898
    fac = np.empty((7))
    trunZq_smom=-1.417
    trunZs_smom=0.5831
    trunZT_smom=-1.5188
    trunZq_mom=-2.00312
    trunZs_mom=25.0835
    trunZT_mom=-2.0497
    if error_flag == 0 :
        if MOM_flag == 0 :
            fac[0]=scalar_conversion_ms_bar_over_rismom(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[4]
            fac[1]=fac[0]
            fac[2]=tensor_conversion_ms_bar_over_rismom(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[4]
            fac[3]=scalar_conversion_ms_bar_over_rismom_mu(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[4]
            fac[4]=fac[3]
            fac[5]=tensor_conversion_ms_bar_over_rismom_mu(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[4]
            fac[6]=quark_field_conversion_ms_bar_over_rimom_prime(mu)[4]*quark_field_anomalous_dimension_under_ms_bar(mu)[4]
        else :
            fac[0]=scalar_conversion_ms_bar_over_rimom_prime(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[4]
            fac[1]=fac[0]
            fac[2]=tensor_conversion_ms_bar_over_rimom_prime(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[4]
            fac[3]=fac[0]/quark_field_conversion_rimom_prime_over_rimom(mu)[4]
            fac[4]=fac[3]
            fac[5]=fac[2]/quark_field_conversion_rimom_prime_over_rimom(mu)[4]
            fac[6]=quark_field_conversion_ms_bar_over_rimom(mu)[4]*quark_field_anomalous_dimension_under_ms_bar(mu)[4]
    elif error_flag == 1 :
        if MOM_flag == 0 :
            fac[0]=scalar_conversion_ms_bar_over_rismom(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[4]
            fac[1]=fac[0]
            fac[2]=tensor_conversion_ms_bar_over_rismom(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[4]
            fac[3]=scalar_conversion_ms_bar_over_rismom_mu(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[4]
            fac[4]=fac[3]
            fac[5]=tensor_conversion_ms_bar_over_rismom_mu(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[4]
            fac[6]=quark_field_conversion_ms_bar_over_rimom_prime(mu)[4]*quark_field_anomalous_dimension_under_ms_bar(mu)[4]
        else :
            fac[0]=scalar_conversion_ms_bar_over_rimom_prime(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[4]
            fac[1]=fac[0]
            fac[2]=tensor_conversion_ms_bar_over_rimom_prime(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[4]
            fac[3]=(1+trunZs_mom*alpha_s*alpha_s*alpha_s*alpha_s*alpha_s)*fac[0]/quark_field_conversion_rimom_prime_over_rimom(mu)[4]
            fac[4]=fac[3]
            fac[5]=(1+trunZT_mom*alpha_s*alpha_s*alpha_s*alpha_s*alpha_s)*fac[2]/quark_field_conversion_rimom_prime_over_rimom(mu)[4]
            fac[6]=(1+trunZq_mom*alpha_s*alpha_s*alpha_s*alpha_s*alpha_s)*quark_field_conversion_ms_bar_over_rimom(mu)[4]*quark_field_anomalous_dimension_under_ms_bar(mu)[4]
    elif error_flag == 2 :
        if MOM_flag == 0 :
            fac[0]=scalar_conversion_ms_bar_over_rismom(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[2]
            fac[1]=fac[0]
            fac[2]=tensor_conversion_ms_bar_over_rismom(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[2]
            fac[3]=scalar_conversion_ms_bar_over_rismom_mu(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[2]
            fac[4]=fac[3]
            fac[5]=tensor_conversion_ms_bar_over_rismom_mu(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[2]
            fac[6]=quark_field_conversion_ms_bar_over_rimom_prime(mu)[4]*quark_field_anomalous_dimension_under_ms_bar(mu)[2]
        else :
            fac[0]=scalar_conversion_ms_bar_over_rimom_prime(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[2]
            fac[1]=fac[0]
            fac[2]=tensor_conversion_ms_bar_over_rimom_prime(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[2]
            fac[3]=fac[0]/quark_field_conversion_rimom_prime_over_rimom(mu)[4]
            fac[4]=fac[3]
            fac[5]=fac[2]/quark_field_conversion_rimom_prime_over_rimom(mu)[4]
            fac[6]=quark_field_conversion_ms_bar_over_rimom(mu)[4]*quark_field_anomalous_dimension_under_ms_bar(mu)[2]
    elif error_flag == 3 :
        if MOM_flag == 0 :
            fac[0]=scalar_conversion_ms_bar_over_rismom(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[4]
            fac[1]=fac[0]
            fac[2]=tensor_conversion_ms_bar_over_rismom(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[4]
            fac[3]=scalar_conversion_ms_bar_over_rismom_mu(mu)[4]*scalar_anomalous_dimension_under_ms_bar(mu)[4]
            fac[4]=fac[3]
            fac[5]=tensor_conversion_ms_bar_over_rismom_mu(mu)[4]*tensor_anomalous_dimension_under_ms_bar(mu)[4]
            fac[6]=quark_field_conversion_ms_bar_over_rimom_prime(mu,0.349)[4]*quark_field_anomalous_dimension_under_ms_bar(mu)[4]
        else :
#            fac[0]=scalar_conversion_ms_bar_over_rimom_prime2(mu,0.0,0.349,nf)[4]*scalar_anomalous_dimension_under_ms_bar(mu,2.0,0.349,nf)[4]
#            fac[1]=fac[0]
#            fac[2]=tensor_conversion_ms_bar_over_rimom_prime2(mu,0.0,0.349,nf)[4]*tensor_anomalous_dimension_under_ms_bar(mu,2.0,0.349,nf)[4]
#            fac[3]=fac[0]/quark_field_conversion_rimom_prime_over_rimom2(mu,0.0,0.349,nf)[4]
#            fac[4]=fac[3]
#            fac[5]=fac[2]/quark_field_conversion_rimom_prime_over_rimom2(mu,0.0,0.349,nf)[4]
#            fac[6]=quark_field_conversion_ms_bar_over_rimom(mu,0.349,nf)[4]*quark_field_anomalous_dimension_under_ms_bar(mu,2.0,0.349,nf)[4]
            fac[0]=scalar_conversion_ms_bar_over_rimom_prime(mu,0.349,nf)[4]*scalar_anomalous_dimension_under_ms_bar(mu,2.0,0.349,nf)[4]
            fac[1]=fac[0]
            fac[2]=tensor_conversion_ms_bar_over_rimom_prime(mu,0.349,nf)[4]*tensor_anomalous_dimension_under_ms_bar(mu,2.0,0.349,nf)[4]
            fac[3]=fac[0]/quark_field_conversion_rimom_prime_over_rimom(mu,0.349,nf)[4]
            fac[4]=fac[3]
            fac[5]=fac[2]/quark_field_conversion_rimom_prime_over_rimom(mu,0.349,nf)[4]
            fac[6]=quark_field_conversion_ms_bar_over_rimom(mu,0.349,nf)[4]*quark_field_anomalous_dimension_under_ms_bar(mu,2.0,0.349,nf)[4]
    return fac

######################################### Pade matching factor #########################################

def scalar_conversion_ms_bar_over_rimom_pade_3loop(scale, Lambda=None, nf=3):
    a_s = strong_coupling_constant(scale, Lambda, nf)
    C_t = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_t[i]=(1 - 7.15994 * a_s[i] - 1.37291 * a_s[i] * a_s[i])/(1 - 8.49197 *  a_s[i])
    return C_t


def scalar_conversion_ms_bar_over_rimom_pade_4loop(scale, Lambda=None, nf=3):
    a_s = strong_coupling_constant(scale, Lambda, nf)
    C_t = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_t[i]=(1 - 8.2047 * a_s[i] - 2.76457 * a_s[i] * a_s[i] - 10.3835 * a_s[i] * a_s[i] * a_s[i])/(1 - 9.53673 * a_s[i])
    return C_t


def tensor_conversion_ms_bar_over_rimom_pade_3loop(scale, Lambda=None, nf=3):
    a_s = strong_coupling_constant(scale, Lambda, nf)
    C_t = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_t[i]=(1 - 8.22443 * a_s[i] - 1.6196 * a_s[i] * a_s[i])/(1 - 8.22443 *  a_s[i])
    return C_t


def tensor_conversion_ms_bar_over_rimom_pade_4loop(scale, Lambda=None, nf=3):
    a_s = strong_coupling_constant(scale, Lambda, nf)
    C_t = np.empty((len(a_s)))
    for i in range(len(a_s)):
        C_t[i]=(1 - 6.86236 * a_s[i] - 1.6196 * a_s[i] * a_s[i] - 2.206 * a_s[i] * a_s[i] * a_s[i])/(1 - 6.86236 * a_s[i])
    return C_t


def pade_matching_factor(mu, scale0=2.0, Lambda=None, nf=3):
    """Return the three historical scalar Padé matching variants.

    ``scale0`` is a deprecated compatibility no-op because no running factor is
    part of the public three-channel return. It remains validated so malformed
    legacy calls fail closed, while discarded tensor/running branches are not
    evaluated.
    """

    mu = _finite_real_scalar(mu, "mu", positive=True)
    _finite_real_scalar(scale0, "scale0", positive=True)
    Lambda, nf = _resolve_lambda(Lambda, nf)
    return np.asarray(
        [
            scalar_conversion_ms_bar_over_rimom_prime(mu, Lambda, nf)[4]
            / quark_field_conversion_rimom_prime_over_rimom(mu, Lambda, nf)[4],
            scalar_conversion_ms_bar_over_rimom_pade_3loop(mu, Lambda, nf)[4],
            scalar_conversion_ms_bar_over_rimom_pade_4loop(mu, Lambda, nf)[4],
        ]
    )



######################################### RI/MOM fitting form #########################################

def fcn_ma(x, p):
    f=p['A']/pow(x,2) + p['B'] + p['C']*x
    return f


def fcn_a2p2(x, p):
    f=p['C0'] + p['C1']*x + p['C2']*x**2 + p['C3']*x**3
    return f


def gvar_fcn_a2p2(x, p):
    return p['C0'] + p['C1']*x + p['C2']*x**2 + p['C3']*x**3



######################################### Get 2-loop matching #########################################

def find_closest_indices(ensemble, ens_small):
    """Return original indices that bracket every target momentum."""

    targets = _finite_real_array(ensemble, "ensemble")
    references = _finite_real_array(ens_small, "ens_small")
    if targets.ndim != 1 or references.ndim != 1 or references.size < 2:
        raise ValueError("targets/references must be 1D with at least two references")
    if not np.isfinite(targets).all() or not np.isfinite(references).all():
        raise ValueError("interpolation momenta must be finite")
    order = np.argsort(references)
    sorted_references = references[order]
    if np.any(np.diff(sorted_references) <= 0):
        raise ValueError("reference momenta must be unique")
    if targets.size and (
        targets.min() < sorted_references[0] or targets.max() > sorted_references[-1]
    ):
        raise ValueError("target momenta must lie inside the reference range")
    index_pairs = []
    for value in targets:
        if value == sorted_references[0]:
            left, right = 0, 1
        elif value == sorted_references[-1]:
            left, right = sorted_references.size - 2, sorted_references.size - 1
        else:
            right = int(np.searchsorted(sorted_references, value, side="right"))
            left = right - 1
        index_pairs.append((int(order[left]), int(order[right])))
    return index_pairs


def linear_interpolation(x1, y1, x2, y2, target_x):
    slope = (y2 - y1) / (x2 - x1)
    target_y = y1 + slope * (target_x - x1)
    return target_y


def linear_interpolation_with_error(x1, y1, x2, y2, target_x):
    if x2 == x1:
        raise ValueError("interpolation endpoints must have distinct momenta")
    weight = (target_x - x1) / (x2 - x1)
    # Arithmetic on gvar inputs preserves their covariance. For independent
    # endpoints this also gives the standard weighted quadrature uncertainty.
    return (1 - weight) * y1 + weight * y2



######################################### Get 2-loop matching #########################################

def readlist(nm, naplllow, napllow, naplow, napmid, naphigh, am1, ZA):
    return nm, naplllow, napllow, naplow, napmid, naphigh, am1, ZA


def _fit_window_bounds(a2p2, fitmin, upper_limit):
    """Return inclusive indices for ``fitmin < a2p2 <= upper_limit``."""

    momenta = _finite_real_array(a2p2, "a2p2")
    if momenta.ndim != 1 or not np.isfinite(momenta).all():
        raise ValueError("a2p2 must be one finite one-dimensional array")
    # The caller slices every aligned data column with one contiguous range.
    # Without this guard an unsorted momentum list can silently discard valid
    # in-window points (or include out-of-window points) because the previous
    # implementation treated the first out-of-range entry as the endpoint.
    if np.any(np.diff(momenta) < 0):
        raise ValueError(
            "a2p2 must be sorted in nondecreasing order for contiguous window selection"
        )
    start_candidates = np.flatnonzero(momenta > fitmin)
    if start_candidates.size == 0:
        raise ValueError("no momentum satisfies a2p2 > fitmin")
    start = int(start_candidates[0])
    end_candidates = np.flatnonzero(momenta > upper_limit)
    end = int(end_candidates[0]) - 1 if end_candidates.size else momenta.size - 1
    if end < start:
        raise ValueError("the selected momentum window is empty")
    return start, end


def read_data(name, nm, naplllow, napllow, naplow, napmid, naphigh, \
              am1, fitmin, pade_flag, lattice_flag, range_flag, \
              mass_flag, data_type, pade34 = None):
    nm = _nonboolean_integer(nm, "nm", minimum=1)
    naplllow = _nonboolean_integer(naplllow, "naplllow", minimum=1)
    napllow = _nonboolean_integer(napllow, "napllow", minimum=0)
    naplow = _nonboolean_integer(naplow, "naplow", minimum=0)
    napmid = _nonboolean_integer(napmid, "napmid", minimum=0)
    naphigh = _nonboolean_integer(naphigh, "naphigh", minimum=0)
    nap = naplllow + napllow + naplow + napmid + naphigh
    if nap <= 0:
        raise ValueError("total momentum count must be positive")
    am1 = _finite_real_array(am1, "am1")
    if am1.ndim != 1 or am1.size == 0 or not np.isfinite(am1).all() or np.any(am1 <= 0):
        raise ValueError("am1 must be one nonempty finite positive array")
    lattice_flag = _nonboolean_integer(lattice_flag, "lattice_flag", minimum=0)
    if lattice_flag >= len(am1):
        raise ValueError("lattice_flag is outside am1")
    pade_flag = _nonboolean_integer(pade_flag, "pade_flag", allowed={0, 1})
    range_flag = _nonboolean_integer(range_flag, "range_flag", allowed={0, 1, 2})
    mass_flag = _nonboolean_integer(mass_flag, "mass_flag", allowed={0, 1})
    fitmin = _finite_real_scalar(fitmin, "fitmin")
    if not isinstance(data_type, str) or not data_type:
        raise TypeError("data_type must be a nonempty column name")
    if not isinstance(name, (str, Path)):
        raise TypeError("name must be a table path")
    if pade_flag == 1:
        if pade34 is None:
            raise ValueError("pade34 is required when pade_flag=1")
        pade34 = _nonboolean_integer(
            pade34, "pade34", allowed={0, 1, 2}
        )
    if mass_flag == 1 and nm <= 2:
        raise ValueError("mass_flag=1 requires at least three mass points")
    napmas = naplllow
    import pandas as pd

    lop4 = pd.read_table(name, sep=r'\s+')
    required_columns = {"SMOM", "Mass", "a2p2", data_type, data_type + "err"}
    missing_columns = required_columns - set(lop4.columns)
    if missing_columns:
        raise ValueError(f"input table is missing columns {sorted(missing_columns)}")
    lop4 = lop4.drop(lop4[lop4['SMOM'] == 0].index)
    lop4.reset_index(drop = True, inplace = True)
    if len(lop4) < nm * nap:
        raise ValueError("input table has fewer rows than the declared mass/momentum layout")
    numeric_columns = ["Mass", "a2p2", data_type, data_type + "err"]
    if not np.isfinite(lop4[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("input table contains nonfinite numerical data")
    if np.any(lop4[data_type + "err"].to_numpy(dtype=float) < 0):
        raise ValueError("input table uncertainties must be nonnegative")
    mqa=np.empty((nm))
    for im in range(0,nm):
        mqa[im] = lop4.loc[im*napmas+0,'Mass']
    lop4pade = np.empty((nm, nap))
    lop4pade_err = np.empty((nm, nap))

    a2p2=np.empty((nap))
    for illl in range(0,naplllow):
        a2p2[illl] = lop4.loc[0*naplllow+illl,'a2p2']
    for ill in range(0,napllow):
        a2p2[naplllow+ill]=lop4.loc[nm*naplllow+0*napllow+ill,'a2p2']
    for il in range(0,naplow):
        a2p2[naplllow+napllow+il]=lop4.loc[nm*(naplllow+napllow)+0*naplow+il,'a2p2']
    for imd in range(0,napmid):
        a2p2[naplllow+napllow+naplow+imd]=lop4.loc[nm*(naplllow+napllow+naplow)+0*napmid+imd,'a2p2']
    for ih in range(0,naphigh):
        a2p2[naplllow+napllow+naplow+napmid+ih]=lop4.loc[nm*(naplllow+napllow+naplow+napmid)+0*naphigh+ih,'a2p2']

    upper_limit = {0: 18.0, 1: 17.0, 2: 13.0}[range_flag]
    deletedata, deletedata2 = _fit_window_bounds(a2p2, fitmin, upper_limit)

    p = np.sqrt(a2p2)*am1[lattice_flag]
    fac = None
    if pade_flag == 1:
        fac = np.empty((nap, 3))
        for i in range(nap):
            fac[i] = pade_matching_factor(p[i])
    a2p2 = a2p2[deletedata:deletedata2+1]
    p = p[deletedata:deletedata2+1]

    data_errtype = data_type+'err'
    if pade_flag == 0 :
        for im in range(0,nm):
            for illl in range(0,naplllow):
                lop4pade[im][illl]=lop4.loc[im*naplllow+illl,data_type]
                lop4pade_err[im][illl]=lop4.loc[im*naplllow+illl,data_errtype]
            for ill in range(0,napllow):
                lop4pade[im][naplllow+ill]=lop4.loc[nm*naplllow+im*napllow+ill,data_type]
                lop4pade_err[im][naplllow+ill]=lop4.loc[nm*naplllow+im*napllow+ill,data_errtype]
            for il in range(0,naplow):
                lop4pade[im][naplllow+napllow+il]=lop4.loc[nm*(naplllow+napllow)+im*naplow+il,data_type]
                lop4pade_err[im][naplllow+napllow+il]=lop4.loc[nm*(naplllow+napllow)+im*naplow+il,data_errtype]
            for imd in range(0,napmid):
                lop4pade[im][naplllow+napllow+naplow+imd]=lop4.loc[nm*(naplllow+napllow+naplow)+im*napmid+imd,data_type]
                lop4pade_err[im][naplllow+napllow+naplow+imd]=lop4.loc[nm*(naplllow+napllow+naplow)+im*napmid+imd,data_errtype]
            for ih in range(0,naphigh):
                lop4pade[im][naplllow+napllow+naplow+napmid+ih]=lop4.loc[nm*(naplllow+napllow+naplow+napmid)+im*naphigh+ih,data_type]
                lop4pade_err[im][naplllow+napllow+naplow+napmid+ih]=lop4.loc[nm*(naplllow+napllow+naplow+napmid)+im*naphigh+ih,data_errtype]
    else :
        for im in range(0,nm):
            for illl in range(0,naplllow):
                lop4pade[im][illl]=lop4.loc[im*naplllow+illl,data_type]/fac[illl][0]*fac[illl][pade34]
                lop4pade_err[im][illl]=lop4.loc[im*naplllow+illl,data_errtype]/fac[illl][0]*fac[illl][pade34]
            for ill in range(0,napllow):
                lop4pade[im][naplllow+ill]=lop4.loc[nm*naplllow+im*napllow+ill,data_type]/fac[naplllow+ill][0]*fac[naplllow+ill][pade34]
                lop4pade_err[im][naplllow+ill]=lop4.loc[nm*naplllow+im*napllow+ill,data_errtype]/fac[naplllow+ill][0]*fac[naplllow+ill][pade34]
            for il in range(0,naplow):
                lop4pade[im][naplllow+napllow+il]=lop4.loc[nm*(naplllow+napllow)+im*naplow+il,data_type]/fac[naplllow+napllow+il][0]*fac[naplllow+napllow+il][pade34]
                lop4pade_err[im][naplllow+napllow+il]=lop4.loc[nm*(naplllow+napllow)+im*naplow+il,data_errtype]/fac[naplllow+napllow+il][0]*fac[naplllow+napllow+il][pade34]
            for imd in range(0,napmid):
                lop4pade[im][naplllow+napllow+naplow+imd]=lop4.loc[nm*(naplllow+napllow+naplow)+im*napmid+imd,data_type]/fac[naplllow+napllow+naplow+imd][0]*fac[naplllow+napllow+naplow+imd][pade34]
                lop4pade_err[im][naplllow+napllow+naplow+imd]=lop4.loc[nm*(naplllow+napllow+naplow)+im*napmid+imd,data_errtype]/fac[naplllow+napllow+naplow+imd][0]*fac[naplllow+napllow+naplow+imd][pade34]
            for ih in range(0,naphigh):
                lop4pade[im][naplllow+napllow+naplow+napmid+ih]=lop4.loc[nm*(naplllow+napllow+naplow+napmid)+im*naphigh+ih,data_type]/fac[naplllow+napllow+naplow+napmid+ih][0]*fac[naplllow+napllow+naplow+napmid+ih][pade34]
                lop4pade_err[im][naplllow+napllow+naplow+napmid+ih]=lop4.loc[nm*(naplllow+napllow+naplow+napmid)+im*naphigh+ih,data_errtype]/fac[naplllow+napllow+naplow+napmid+ih][0]*fac[naplllow+napllow+naplow+napmid+ih][pade34]

    lop4pade = lop4pade[:, deletedata:deletedata2+1]
    lop4pade_err = lop4pade_err[:, deletedata:deletedata2+1]
    if mass_flag == 1:
        mqa = mqa[2:]
        lop4pade = lop4pade[2:, :]
        lop4pade_err = lop4pade_err[2:, :]
    else:
        pass
    return mqa, lop4pade, lop4pade_err, a2p2, p



######################################### Fitting Data #########################################

def ma_fit(lop4pade, lop4pade_err, mqa, prior_ma, print_mafit_flag, a2p2 = None):
    if any(_contains_boolean(value) for value in (lop4pade, lop4pade_err, mqa)):
        raise TypeError("ma_fit numerical inputs must be real, not booleans")
    lop4pade = np.asarray(lop4pade)
    lop4pade_err = np.asarray(lop4pade_err)
    mqa = np.asarray(mqa)
    if any(np.iscomplexobj(array) for array in (lop4pade, lop4pade_err, mqa)):
        raise TypeError("ma_fit numerical inputs must be real")
    lop4pade = np.asarray(lop4pade, dtype=float)
    lop4pade_err = np.asarray(lop4pade_err, dtype=float)
    mqa = np.asarray(mqa, dtype=float)
    if lop4pade.ndim != 2 or lop4pade.shape != lop4pade_err.shape:
        raise ValueError("lop4pade and lop4pade_err must have one equal 2D shape")
    if lop4pade.shape[0] < 3 or lop4pade.shape[1] == 0:
        raise ValueError("ma_fit requires at least three masses and one momentum")
    if mqa.shape != (lop4pade.shape[0],):
        raise ValueError("mqa must match the mass axis of lop4pade")
    if not all(np.isfinite(array).all() for array in (lop4pade, lop4pade_err, mqa)):
        raise ValueError("ma_fit numerical inputs must be finite")
    if np.any(lop4pade_err < 0) or np.any(mqa == 0):
        raise ValueError("uncertainties must be nonnegative and mqa must be nonzero")
    print_mafit_flag = _nonboolean_integer(
        print_mafit_flag, "print_mafit_flag", allowed={0, 1}
    )
    if a2p2 is not None:
        if _contains_boolean(a2p2):
            raise TypeError("a2p2 must be real when supplied")
        a2p2 = np.asarray(a2p2)
        if np.iscomplexobj(a2p2):
            raise TypeError("a2p2 must be real when supplied")
        try:
            a2p2 = np.asarray(a2p2, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("a2p2 must be a finite real array") from error
        if a2p2.shape != (lop4pade.shape[1],) or not np.isfinite(a2p2).all():
            raise ValueError("a2p2 must match the momentum axis when supplied")
    import gvar as gv
    import lsqfit

    ZS_ma = list()
    ZS_ma_final = np.empty((lop4pade.shape[1]))
    ZS_ma_err = list()
    ZS_ma_err_final = np.empty((lop4pade_err.shape[1]))
    for i in range(lop4pade.shape[1]):
        RCs_ma_mean = np.empty((lop4pade.shape[0]))
        RCs_ma_sdev = np.empty((lop4pade.shape[0]))
        for im in range(lop4pade.shape[0]):
            RCs_ma_mean[im] = lop4pade[im][i]
            RCs_ma_sdev[im] = lop4pade_err[im][i]
        RCs_ma = gv.gvar(RCs_ma_mean, RCs_ma_sdev)
        fit_ma = lsqfit.nonlinear_fit(prior = prior_ma, data=(mqa, RCs_ma), fcn = fcn_ma)
        if print_mafit_flag == 1 :
            print(fit_ma.format(True))
        ZS_ma.append(fit_ma.p['B'].mean)
        ZS_ma_final0 = flatten(ZS_ma)
        ZS_ma_final[i] = ZS_ma_final0[i]
        ZS_ma_err.append(fit_ma.p['B'].sdev)
        ZS_ma_err_final0 = flatten(ZS_ma_err)
        ZS_ma_err_final[i] = ZS_ma_err_final0[i]
    return ZS_ma_final, ZS_ma_err_final


def gvar_output(x, p):
    return p['C0'] + p['C1']*x + p['C2']*x**2 + p['C3']*x**3


def _product_mean_sdev(left, right):
    """Return product statistics while preserving any shared gvar sources."""

    product = left * right
    return product.mean, product.sdev


def a2p2_fit(ZS_ma_final, ZS_ma_err_final, a2p2, am1, ZA, prior_a2p2, \
             print_flag, lattice_flag, matching_error_flag, gvar_flag = 0):
    if any(
        _contains_boolean(value)
        for value in (ZS_ma_final, ZS_ma_err_final, a2p2)
    ):
        raise TypeError("a2p2_fit data and momenta must be real, not booleans")
    arrays = [np.asarray(value) for value in (ZS_ma_final, ZS_ma_err_final, a2p2)]
    if any(np.iscomplexobj(array) for array in arrays):
        raise TypeError("a2p2_fit data and momenta must be real")
    ZS_ma_final, ZS_ma_err_final, a2p2 = [
        np.asarray(array, dtype=float) for array in arrays
    ]
    if not (
        ZS_ma_final.ndim == ZS_ma_err_final.ndim == a2p2.ndim == 1
        and ZS_ma_final.shape == ZS_ma_err_final.shape == a2p2.shape
        and a2p2.size > 0
    ):
        raise ValueError("ZS means/errors and a2p2 must be equal nonempty 1D arrays")
    if not all(np.isfinite(array).all() for array in (ZS_ma_final, ZS_ma_err_final, a2p2)):
        raise ValueError("a2p2_fit numerical inputs must be finite")
    if np.any(ZS_ma_err_final < 0) or np.any(a2p2 <= 0):
        raise ValueError("uncertainties must be nonnegative and a2p2 positive")
    am1 = _finite_real_array(am1, "am1")
    lattice_flag = _nonboolean_integer(lattice_flag, "lattice_flag", minimum=0)
    if (
        am1.ndim != 1
        or lattice_flag >= am1.size
        or np.any(am1 <= 0)
    ):
        raise ValueError("am1/lattice_flag must select one finite positive inverse spacing")
    ZA = _finite_real_scalar(ZA, "ZA")
    print_flag = _nonboolean_integer(print_flag, "print_flag", allowed={0, 1})
    matching_error_flag = _nonboolean_integer(
        matching_error_flag, "matching_error_flag", allowed={0, 1, 2, 3}
    )
    gvar_flag = _nonboolean_integer(gvar_flag, "gvar_flag", allowed={0, 1})
    mu = np.sqrt(a2p2)*am1[lattice_flag]
    fac = np.empty((mu.shape[0]))
    for i in range(mu.shape[0]):
        fac[i] = matching_systematic_error(mu[i], 1, matching_error_flag)[3]/\
        matching_systematic_error(mu[i], 1, 0)[3] #1 for MOM, 3 for ZS
    ZS_ma_final = ZS_ma_final * fac
    ZS_ma_err_final = ZS_ma_err_final * fac
    import gvar as gv
    import lsqfit

    RCs_a2p2 = gv.gvar(ZS_ma_final, ZS_ma_err_final)
    fit_a2p2 = lsqfit.nonlinear_fit(prior = prior_a2p2, data = (a2p2, RCs_a2p2), fcn = fcn_a2p2)
    if print_flag == 1 :
        print(fit_a2p2.format(True))
        print("ZS = ", gv.gvar(fit_a2p2.p['C0'].mean * ZA, fit_a2p2.p['C0'].sdev * ZA))
    Zs_ov_ZA = gv.gvar(fit_a2p2.p['C0'].mean * ZA,fit_a2p2.p['C0'].sdev * ZA)
    x_values = np.linspace(start = 0, stop = 20, num = 200)
    output_mean = np.empty((200))
    output_sdev = np.empty((200))
    for i, x in enumerate(x_values):
            output_mean[i] = gvar_fcn_a2p2(x, fit_a2p2.p).mean * ZA
            output_sdev[i] = gvar_fcn_a2p2(x, fit_a2p2.p).sdev * ZA
    if gvar_flag == 0:
        return Zs_ov_ZA
    else:
        return Zs_ov_ZA, output_mean, output_sdev


def ratio_fit(p, p0, a2p2, a2p2_0, ZS_all_small, ZS_all_0, am1, \
              am1_0, ZA, ZA0, prior_ratio, lattice_flag, \
              matching_error_flag, ZS_small):
    if any(_contains_boolean(value) for value in (p, p0, a2p2, a2p2_0)):
        raise TypeError("ratio_fit momentum inputs must be real, not booleans")
    raw_momenta = [np.asarray(value) for value in (p, p0, a2p2, a2p2_0)]
    if any(np.iscomplexobj(array) for array in raw_momenta):
        raise TypeError("ratio_fit momentum inputs must be real")
    try:
        p, p0, a2p2, a2p2_0 = [
            np.asarray(array, dtype=float) for array in raw_momenta
        ]
    except (TypeError, ValueError) as error:
        raise TypeError("ratio_fit momentum inputs must be finite real arrays") from error
    if any(array.ndim != 1 or array.size == 0 for array in (p, p0, a2p2, a2p2_0)):
        raise ValueError("p, p0, a2p2, and a2p2_0 must be nonempty 1D arrays")
    if p.shape != a2p2.shape or p0.shape != a2p2_0.shape:
        raise ValueError("each momentum array must align with its a2p2 array")
    if not all(np.isfinite(array).all() for array in (p, p0, a2p2, a2p2_0)):
        raise ValueError("ratio_fit momentum inputs must be finite")
    if np.any(p <= 0) or np.any(p0 <= 0) or np.any(a2p2 <= 0) or np.any(a2p2_0 <= 0):
        raise ValueError("ratio_fit momentum magnitudes and a2p2 must be positive")
    if np.shape(ZS_all_small) != p.shape or np.shape(ZS_all_0) != p0.shape:
        raise ValueError("ZS arrays must match their corresponding momentum axes")
    am1 = _finite_real_array(am1, "am1")
    am1_0 = _finite_real_array(am1_0, "am1_0")
    lattice_flag = _nonboolean_integer(lattice_flag, "lattice_flag", minimum=0)
    if (
        am1.ndim != 1
        or am1_0.ndim != 1
        or lattice_flag >= am1.size
        or lattice_flag >= am1_0.size
        or np.any(am1 <= 0)
        or np.any(am1_0 <= 0)
    ):
        raise ValueError("am1/am1_0 and lattice_flag must define positive finite scales")
    ZA = _finite_real_scalar(ZA, "ZA")
    ZA0 = _finite_real_scalar(ZA0, "ZA0")
    if ZA == 0 or ZA0 == 0:
        raise ValueError("ZA and ZA0 must be nonzero")
    matching_error_flag = _nonboolean_integer(
        matching_error_flag, "matching_error_flag", allowed={0, 1, 2, 3}
    )
    p_ens   = p0 #normal like 48I
    p_small = p #small like a03m310
    mu0 = np.sqrt(a2p2_0) * am1_0[lattice_flag]
    mu = np.sqrt(a2p2) * am1[lattice_flag]
    fac0 = np.empty((mu0.shape[0]))
    fac = np.empty((mu.shape[0]))

    for i in range(mu0.shape[0]):
        fac0[i] = matching_systematic_error(mu0[i],1,matching_error_flag)[3]/\
        matching_systematic_error(mu0[i],1,0)[3] #1 for MOM, 3 for ZS
    for i in range(mu.shape[0]):
        fac[i] = matching_systematic_error(mu[i],1,matching_error_flag)[3]/\
        matching_systematic_error(mu[i],1,0)[3] #1 for MOM, 3 for ZS

    value_ens   = np.empty(len(p_ens), dtype=object)
    value_small = ZS_all_small*fac

    interpolation_pairs = find_closest_indices(p_ens, p_small)
    for i, (find0, find1) in enumerate(interpolation_pairs):
        #print(find0, a2p2[find0], find1, a2p2[find1]) #test_flag
        value_ens[i] = linear_interpolation_with_error(p_small[find0], value_small[find0], p_small[find1], value_small[find1], p_ens[i])
    denominator_means = []
    for value in value_ens:
        mean_value = getattr(value, "mean", value)
        if callable(mean_value):
            mean_value = mean_value()
        denominator_means.append(float(mean_value))
    denominator_means = np.asarray(denominator_means)
    if not np.isfinite(denominator_means).all() or np.any(denominator_means == 0):
        raise ValueError("interpolated small-ensemble denominator is zero or nonfinite")
    a2p2_ens = (p_ens/am1[lattice_flag])**2 #small on normal
    ratio_ens = (ZS_all_0*fac0*ZA0)/(value_ens*ZA)
    #print(ZS_all_0*fac0*ZA0,value_ens*ZA)

    import lsqfit

    fit_ratio = lsqfit.nonlinear_fit(prior=prior_ratio, data=(a2p2_ens,ratio_ens), fcn=fcn_a2p2)
    ratio = fit_ratio.p['C0']

    x_values = np.linspace(start = 0, stop = 20, num = 200)
    output_mean_small = np.empty((200))
    output_sdev_small = np.empty((200))
    for i, x in enumerate(x_values):
        output_mean_small[i], output_sdev_small[i] = _product_mean_sdev(
            gvar_output(x, fit_ratio.p), ZS_small
        )
    return ratio, ratio_ens, output_mean_small, output_sdev_small



######################################### ZO1/ZO2 #########################################

def _xp(array):
    return cp if cp is not None and isinstance(array, cp.ndarray) else np


def _as_backend(array, xp):
    if cp is not None and xp is np and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return xp.asarray(array)


def inverse_propagator(prop: Union[np.ndarray, cp.ndarray], on_device = None):
    """Invert batched spin-color blocks on the input array's backend.

    `on_device` remains as a strict Boolean compatibility check; leaving it
    unset avoids a host fallback when a CuPy propagator is supplied.
    """

    xp = _xp(prop)
    inferred_device = xp is cp
    if on_device is not None:
        if not isinstance(on_device, (bool, np.bool_)):
            raise TypeError("on_device must be a Boolean compatibility check")
        if bool(on_device) != inferred_device:
            raise ValueError("on_device disagrees with the propagator array backend")
    shape = prop.shape
    if shape[-4:] != (4, 4, 3, 3):
        raise ValueError("prop must end in (4,4,3,3)")
    inv_shape = shape[:-4] + (shape[-4] * shape[-2], shape[-3] * shape[-1])
    kro_shape = shape[:-4] + (shape[-4], shape[-2], shape[-3], shape[-1])
    return xp.linalg.inv(prop.swapaxes(-3, -2).reshape(inv_shape)).reshape(
        kro_shape
    ).swapaxes(-3, -2)


def adj(prop, Gm5):
    """Return gamma5 * prop^dagger * gamma5 with arbitrary leading axes.

    The Hermitian adjoint exchanges both source/sink spin and source/sink
    color.  Exchanging only the spin axes is correct only for a color-diagonal
    propagator and silently corrupts a general NPR vertex.
    """

    if prop.ndim < 4 or prop.shape[-4:] != (4, 4, 3, 3):
        raise ValueError("prop must end in (4,4,3,3)")
    xp = _xp(prop)
    gamma5 = _as_backend(Gm5, xp)
    if gamma5.shape != (4, 4):
        raise ValueError("Gm5 must have shape (4,4)")
    dagger = prop.conj().swapaxes(-4, -3).swapaxes(-2, -1)
    return xp.einsum(
        "ij,...jkab,kl->...ilab",
        gamma5,
        dagger,
        gamma5,
        optimize=True,
    )


def print_a2p2(Nx, Nt, mom_list):
    ap_x = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0)
    ap_y = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0)
    ap_z = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0)
    ap_t = 2 * np.pi * np.fft.fftfreq(Nt, d=1.0)
    apt, apz, apy, apx = np.meshgrid(ap_t, ap_z, ap_y, ap_x, indexing="ij")
    a2_p2 = apx**2 + apy**2 + apz**2 + apt**2
    for ip in range(len(mom_list)):
        print(ip, "a2p2 = ", a2_p2[mom_list[ip][3], mom_list[ip][2], mom_list[ip][1], mom_list[ip][0]])


def Lambda_O_con(Sq, GreenB, is_jack = 1):
    is_jack = _nonboolean_integer(is_jack, "is_jack", allowed={0, 1})
    if getattr(Sq, "ndim", 0) != 5 or Sq.shape[-4:] != (4, 4, 3, 3):
        raise ValueError("Sq must have exact (configuration,4,4,3,3) layout")
    if getattr(GreenB, "shape", None) != Sq.shape:
        raise ValueError("GreenB must have exactly the same shape as Sq")
    if gamma is None:
        raise ImportError("Lambda_O_con requires pyquda_utils.gamma")
    xp = _xp(Sq)
    if _xp(GreenB) is not xp:
        raise TypeError("Sq and GreenB must use the same NumPy/CuPy backend")
    if is_jack == 1 : Sq_inv = inverse_propagator(analy.jackknife_resampling(Sq))
    else : Sq_inv = inverse_propagator(Sq)
    Sq_inv_adj = adj(Sq_inv, _as_backend(gamma.gamma(15), xp))
    if is_jack == 1 : BareGreen = analy.jackknife_resampling(GreenB)
    else : BareGreen = GreenB
    AmpuGreen = xp.einsum(
        'nijab,njkbc,nklca->nil', Sq_inv, BareGreen, Sq_inv_adj, optimize=True
    )
    return AmpuGreen


def Lambda_O_dis(Sq, current, is_jack = 1):
    is_jack = _nonboolean_integer(is_jack, "is_jack", allowed={0, 1})
    if getattr(Sq, "ndim", 0) != 5 or Sq.shape[-4:] != (4, 4, 3, 3):
        raise ValueError("Sq must have exact (configuration,4,4,3,3) layout")
    if getattr(current, "shape", None) != Sq.shape[:1]:
        raise ValueError("current must have one entry per Sq configuration")
    if gamma is None:
        raise ImportError("Lambda_O_dis requires pyquda_utils.gamma")
    xp = _xp(Sq)
    if _xp(current) is not xp:
        raise TypeError("Sq and current must use the same NumPy/CuPy backend")
    if is_jack == 1 : Sq_inv = inverse_propagator(analy.jackknife_resampling(Sq))
    else : Sq_inv = inverse_propagator(Sq)
    Sq_inv_adj = adj(Sq_inv, _as_backend(gamma.gamma(15), xp))
    disconnected = xp.einsum('nijab,n->nijab', Sq, current, optimize=True)
    if is_jack == 1 : BareGreen = analy.jackknife_resampling(disconnected)
    else : BareGreen = disconnected
    AmpuGreen = xp.einsum(
        'nijab,njkbc,nklca->nil', Sq_inv, BareGreen, Sq_inv_adj, optimize=True
    )
    return AmpuGreen


__all__ = [
    "QuarkRenormProfile",
    "QuarkRenormProfileError",
    "UnverifiedLegacyError",
    "UnknownQuarkRenormProfileError",
    "QuarkRenormLiteral",
    "QUARK_RENORM_PROFILE_REGISTRY",
    "QUARK_RENORM_PHYSICAL_PROFILES",
    "get_quark_renorm_profile",
    "invoke_quark_renorm_profile",
    "get_quark_renorm_legacy_literal",
    "beta_coupling_constant",
    "strong_coupling_constant",
    "alpha_s",
    "vector_conversion_ms_bar_over_rimom_prime",
    "quark_mass_conversion_ms_bar_over_rismom",
    "quark_mass_conversion_ms_bar_over_rismom_mu",
    "quark_field_conversion_rimom_prime_over_rimom",
    "quark_field_conversion_ms_bar_over_rimom",
    "quark_field_conversion_ms_bar_over_rimom_prime",
    "quark_mass_conversion_ms_bar_over_rimom_prime",
    "scalar_conversion_ms_bar_over_rismom",
    "scalar_conversion_ms_bar_over_rismom_mu",
    "scalar_conversion_ms_bar_over_rimom_prime",
    "tensor_conversion_ms_bar_over_rismom",
    "tensor_conversion_ms_bar_over_rismom_mu",
    "tensor_conversion_ms_bar_over_rimom_prime",
    "quark_mass_conversion_ms_bar_over_rimom_prime2",
    "quark_field_conversion_rimom_prime_over_rimom2",
    "scalar_conversion_ms_bar_over_rimom_prime2",
    "tensor_conversion_ms_bar_over_rimom_prime2",
    "anomalous_dimension",
    "scale_running",
    "quark_mass_anomalous_dimension_under_ms_bar",
    "quark_field_anomalous_dimension_under_ms_bar",
    "scalar_anomalous_dimension_under_ms_bar",
    "tensor_anomalous_dimension_under_ms_bar",
    "matching_systematic_error",
    "scalar_conversion_ms_bar_over_rimom_pade_3loop",
    "scalar_conversion_ms_bar_over_rimom_pade_4loop",
    "tensor_conversion_ms_bar_over_rimom_pade_3loop",
    "tensor_conversion_ms_bar_over_rimom_pade_4loop",
    "pade_matching_factor",
    "read_data",
    "ma_fit",
    "a2p2_fit",
    "ratio_fit",
    "inverse_propagator",
    "adj",
    "Lambda_O_con",
    "Lambda_O_dis",
]
