from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass

import numpy as np
import scipy

# Matching matrices are at most 2x2 and RG evolution uses SciPy's host-only ODE
# solver, so GPU offload would add synchronization without reducing lattice
# work. Do not feed full device fields to this module; project/reduce vertices
# on GPU first, then transfer only the small renormalization matrices. A future
# GPU ODE backend is useful only for very large batched scale scans.


# The numeric routines below predate an equation-level binding to a physical
# operator basis.  Keep their default output as an explicit literal replay.
# `physical=True` is deliberately opt-in and fail-closed: no profile is
# registered yet because the one directly relevant primary source contradicts
# the two legacy off-diagonal tree terms.
HELICITY_PHYSICAL_PROFILE_REQUIRED_FIELDS = (
    "profile_id",
    "operator_basis",
    "source_scheme",
    "target_scheme",
    "gauge",
    "flavor_count",
    "color_factors",
    "operator_normalization",
    "matrix_action",
    "matrix_direction",
    "coupling_variable",
    "derivative_variable",
    "logarithm_convention",
    "tree_level_terms",
    "coefficient_provenance",
    "truncation_closure",
    "pade_definition",
    "scale_domain",
    "uncertainty_policy",
)
HELICITY_COEFFICIENT_ENTRIES = ("R11", "R12", "R21", "R22", "Gamma")
HELICITY_PROVENANCE_LOCATORS = ("version", "page", "equation")
HELICITY_TRUNCATION_SECTORS = ("matching", "running")
HELICITY_UNCERTAINTY_SECTORS = ("input_covariance", "coupling", "scale", "truncation")
HELICITY_MATCHING_LEGACY_APIS = frozenset(
    (
        "Helicity_MatchingCoeff_tmp",
        "Helicity_MatchingCoeff",
        "Helicity_MatchingCoeff_array",
    )
)
ZHAO_V2_PROFILE_ID = "zhao-2512.24315v2-k-j5-ri-mom-to-modified-msbar-larin"

# This is a source-audit profile, not a registered calculator profile.  Its
# values record only direct statements needed to diagnose the contradiction;
# it must never make a legacy tuple physical.
ZHAO_V2_K_J5_SOURCE_PROFILE = {
    "profile_id": ZHAO_V2_PROFILE_ID,
    "operator_basis": ("K^mu", "J5^mu"),
    "source_scheme": "RI/MOM",
    "target_scheme": "modified-MSbar (Larin)",
    "gauge": "Landau",
    "flavor_count": 3,
    "color_factors": {"N_c": 3.0, "C_A": 3.0, "C_F": 4.0 / 3.0},
    "operator_normalization": (
        "tr(T^a T^b)=delta^(ab)/2; K^mu uses the source's stated normalization."
    ),
    "matrix_action": "O_target = R @ O_source",
    "matrix_direction": "RI/MOM -> modified-MSbar (Larin)",
    "coupling_variable": "a_s=alpha_s/(4*pi)",
    "derivative_variable": "d/d ln(mu^2)",
    "logarithm_convention": "L_mu=ln(mu^2/mu_RI^2)",
    "tree_level_terms": {"R12": 1.0, "R21": 1.0},
    "coefficient_provenance": {
        "R11": {
            "version": "arXiv:2512.24315v2",
            "page": "Supplemental p. 9",
            "equation": "(20)",
        },
        "R12": {
            "version": "arXiv:2512.24315v2",
            "page": "Supplemental p. 9",
            "equation": "(22)",
        },
        "R21": {
            "version": "arXiv:2512.24315v2",
            "page": "Supplemental p. 9",
            "equation": "(23)",
        },
        "R22": {
            "version": "arXiv:2512.24315v2",
            "page": "Supplemental p. 9",
            "equation": "(21)",
        },
        "Gamma": {
            "version": "arXiv:2512.24315v2",
            "page": "Supplemental p. 10",
            "equation": "(24)",
        },
    },
    "truncation_closure": {
        "matching": {
            "loop_order": 3,
            "closed": True,
            "description": "Fixed-order matching entries are recorded through a_s^3.",
        },
        "running": {
            "loop_order": 3,
            "closed": False,
            "description": (
                "The local Gamma22 truncation is not closed relative to "
                "Gamma22=-2*a_s*n_f*Gamma12."
            ),
        },
    },
    "pade_definition": (
        "No Padé coefficient is source-mapped; a physical profile must reject "
        "the legacy loop-4 estimate."
    ),
    "scale_domain": (
        "The source audit identifies the two scale roles but supplies no "
        "validated numerical domain for a production physical evaluation."
    ),
    "uncertainty_policy": {
        "input_covariance": "Not supplied by this source audit.",
        "coupling": "Not supplied by this source audit.",
        "scale": "Not supplied by this source audit.",
        "truncation": "Not supplied by this source audit.",
    },
}


@dataclass(frozen=True)
class HelicityPhysicalProfileResult:
    """Structured disposition for a requested physical HEL calculation."""

    status: str
    is_physical: bool
    message: str
    profile_id: str | None = None
    legacy_api: str | None = None
    missing_fields: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    profile_field_status: tuple[tuple[str, str], ...] = ()

    def as_dict(self):
        """Return a serializable audit record without changing its disposition."""

        return {
            "status": self.status,
            "is_physical": self.is_physical,
            "message": self.message,
            "profile_id": self.profile_id,
            "legacy_api": self.legacy_api,
            "missing_fields": list(self.missing_fields),
            "contradictions": list(self.contradictions),
            "profile_field_status": dict(self.profile_field_status),
        }


class HelicityPhysicalProfileError(ValueError):
    """Raised only when a caller requests ``physical=True`` without a PASS."""

    def __init__(self, result):
        self.result = result
        super().__init__(f"{result.status}: {result.message}")


def _is_present_text(value):
    return isinstance(value, str) and bool(value.strip())


def _profile_id(profile):
    if isinstance(profile, Mapping) and _is_present_text(profile.get("profile_id")):
        return profile["profile_id"]
    return None


def _profile_missing_fields(profile):
    """List required physical-map fields without coercing untrusted metadata."""

    if not isinstance(profile, Mapping):
        return ("physical_profile (mapping)",)

    missing = []
    for field in HELICITY_PHYSICAL_PROFILE_REQUIRED_FIELDS:
        value = profile.get(field)
        if value is None or value == "" or value == () or value == {}:
            missing.append(field)

    basis = profile.get("operator_basis")
    if basis is not None:
        if isinstance(basis, (str, bytes, bytearray)) or not isinstance(
            basis, SequenceABC
        ):
            missing.append("operator_basis=(K^mu,J5^mu)")
        elif tuple(basis) != ("K^mu", "J5^mu"):
            missing.append("operator_basis=(K^mu,J5^mu)")

    terms = profile.get("tree_level_terms")
    if isinstance(terms, Mapping):
        for entry in ("R12", "R21"):
            if entry not in terms:
                missing.append(f"tree_level_terms.{entry}")
    elif terms is not None:
        missing.append("tree_level_terms (mapping)")

    provenance = profile.get("coefficient_provenance")
    if isinstance(provenance, Mapping):
        for entry in HELICITY_COEFFICIENT_ENTRIES:
            locator = provenance.get(entry)
            if not isinstance(locator, Mapping):
                missing.append(f"coefficient_provenance.{entry}")
                continue
            for field in HELICITY_PROVENANCE_LOCATORS:
                if not _is_present_text(locator.get(field)):
                    missing.append(f"coefficient_provenance.{entry}.{field}")
    elif provenance is not None:
        missing.append("coefficient_provenance (mapping)")

    closure = profile.get("truncation_closure")
    if isinstance(closure, Mapping):
        for sector in HELICITY_TRUNCATION_SECTORS:
            record = closure.get(sector)
            if not isinstance(record, Mapping):
                missing.append(f"truncation_closure.{sector}")
                continue
            for field in ("loop_order", "closed", "description"):
                if field not in record or record[field] in (None, ""):
                    missing.append(f"truncation_closure.{sector}.{field}")
    elif closure is not None:
        missing.append("truncation_closure (mapping)")

    colors = profile.get("color_factors")
    if isinstance(colors, Mapping):
        for label in ("N_c", "C_A", "C_F"):
            if label not in colors:
                missing.append(f"color_factors.{label}")
    elif colors is not None:
        missing.append("color_factors (mapping)")

    uncertainty = profile.get("uncertainty_policy")
    if isinstance(uncertainty, Mapping):
        for sector in HELICITY_UNCERTAINTY_SECTORS:
            if not _is_present_text(uncertainty.get(sector)):
                missing.append(f"uncertainty_policy.{sector}")
    elif uncertainty is not None:
        missing.append("uncertainty_policy (mapping)")

    return tuple(dict.fromkeys(missing))


def _profile_field_status(missing_fields):
    """Return a JSON-safe completeness disposition for every required field."""

    missing = set(missing_fields)
    return tuple(
        (
            field,
            "MISSING"
            if any(item == field or item.startswith(field + ".") for item in missing)
            else "PRESENT",
        )
        for field in HELICITY_PHYSICAL_PROFILE_REQUIRED_FIELDS
    )


def _profile_issues(profile):
    """Return non-missing profile defects that must block physical promotion."""

    issues = []
    if not isinstance(profile, Mapping):
        return tuple(issues)

    for field in (
        "source_scheme",
        "target_scheme",
        "gauge",
        "operator_normalization",
        "matrix_action",
        "matrix_direction",
        "coupling_variable",
        "derivative_variable",
        "logarithm_convention",
        "pade_definition",
        "scale_domain",
    ):
        if field in profile and not _is_present_text(profile[field]):
            issues.append(f"{field} must be nonempty text")

    flavor = profile.get("flavor_count")
    if flavor is not None:
        if isinstance(flavor, (bool, np.bool_)) or not isinstance(flavor, (int, np.integer)):
            issues.append("flavor_count must be the integer 3 for the bundled calculator")
        elif int(flavor) != 3:
            issues.append("flavor_count must be 3 for the bundled calculator")

    colors = profile.get("color_factors")
    if isinstance(colors, Mapping):
        for label, expected in (("N_c", 3.0), ("C_A", 3.0), ("C_F", 4.0 / 3.0)):
            if label not in colors:
                continue
            try:
                value = _finite_real_scalar(colors[label], f"color_factors.{label}")
            except (TypeError, ValueError):
                issues.append(f"color_factors.{label} must be a finite real scalar")
                continue
            if value != expected:
                issues.append(
                    f"color_factors.{label} must match the bundled SU(3) value {expected}"
                )

    terms = profile.get("tree_level_terms")
    if isinstance(terms, Mapping):
        for entry in ("R12", "R21"):
            if entry not in terms:
                continue
            try:
                value = _finite_real_scalar(terms[entry], f"tree_level_terms.{entry}")
            except (TypeError, ValueError):
                issues.append(f"tree_level_terms.{entry} must be a finite real scalar")
                continue
            if value != 1.0:
                issues.append(
                    f"tree_level_terms.{entry} must retain the documented K^mu/J5^mu value 1"
                )

    closure = profile.get("truncation_closure")
    if isinstance(closure, Mapping):
        for sector in HELICITY_TRUNCATION_SECTORS:
            record = closure.get(sector)
            if not isinstance(record, Mapping) or "closed" not in record:
                continue
            if record["closed"] is not True:
                issues.append(f"truncation_closure.{sector} is not closed")
    return tuple(issues)


def validate_helicity_physical_profile(physical_profile, *, legacy_api=None):
    """Audit a HEL physical-profile request without evaluating coefficients.

    The only directly relevant source profile is intentionally supplied above
    as ``ZHAO_V2_K_J5_SOURCE_PROFILE``.  It documents the required metadata
    but is not registered because its off-diagonal tree terms contradict every
    built-in legacy matching tuple.  A future physical implementation must add
    a separately reviewed, complete calculator profile here; callers cannot
    promote a literal tuple by adding labels at runtime.
    """

    profile_id = _profile_id(physical_profile)
    missing_fields = _profile_missing_fields(physical_profile)
    profile_field_status = _profile_field_status(missing_fields)
    issues = _profile_issues(physical_profile)
    contradictions = ()
    if profile_id == ZHAO_V2_PROFILE_ID and legacy_api in HELICITY_MATCHING_LEGACY_APIS:
        contradictions = (
            "Zhao v2 has R12^(0)=R21^(0)=1 in the ordered (K^mu,J5^mu) basis.",
            f"{legacy_api} returns R12^(0)=R21^(0)=0, a -1 difference in both entries.",
        )
        return HelicityPhysicalProfileResult(
            status="PRIMARY_SOURCE_CONTRADICTION",
            is_physical=False,
            message=(
                "The Zhao v2 source profile cannot be bound to a built-in legacy "
                "tuple; use the explicit literal API only."
            ),
            profile_id=profile_id,
            legacy_api=legacy_api,
            missing_fields=missing_fields,
            contradictions=contradictions,
            profile_field_status=profile_field_status,
        )
    if missing_fields:
        return HelicityPhysicalProfileResult(
            status="INCOMPLETE_PHYSICAL_PROFILE",
            is_physical=False,
            message=(
                "physical=True requires ordered (K^mu,J5^mu) metadata, source/target "
                "schemes, gauge/flavor/color/normalization data, action/direction, "
                "coupling/derivative/log conventions, tree terms, coefficient locators, "
                "Padé/scale/uncertainty policies, and closed matching/running truncations."
            ),
            profile_id=profile_id,
            legacy_api=legacy_api,
            missing_fields=missing_fields,
            contradictions=issues,
            profile_field_status=profile_field_status,
        )
    if issues:
        return HelicityPhysicalProfileResult(
            status="UNVERIFIED_PHYSICAL_PROFILE",
            is_physical=False,
            message=(
                "The supplied HEL profile has an unresolved physical-map, "
                "tree-term, flavor/color, or truncation defect."
            ),
            profile_id=profile_id,
            legacy_api=legacy_api,
            contradictions=issues,
            profile_field_status=profile_field_status,
        )
    # Intentionally empty until a reviewed implementation and its exact
    # source-to-code map are added together.  Metadata alone cannot certify
    # these pre-existing numerical tables.
    return HelicityPhysicalProfileResult(
        status="UNVERIFIED_NO_MAPPED_PHYSICAL_PROFILE",
        is_physical=False,
        message=(
            "No fully documented HEL coefficient implementation is registered; "
            "the bundled tuples remain explicit UNVERIFIED_LEGACY replays."
        ),
        profile_id=profile_id,
        legacy_api=legacy_api,
        profile_field_status=profile_field_status,
    )


def _require_physical_profile(physical, physical_profile, *, legacy_api):
    if not isinstance(physical, (bool, np.bool_)):
        raise TypeError("physical must be a boolean")
    if not physical:
        return
    result = validate_helicity_physical_profile(
        physical_profile, legacy_api=legacy_api
    )
    if not result.is_physical:
        raise HelicityPhysicalProfileError(result)


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


def _finite_real_scalar(value, name, *, positive=False):
    """Validate one real scalar without accepting Boolean or complex aliases."""

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


def _choice_integer(value, name, allowed):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a non-boolean integer")
    result = int(value)
    if result not in allowed:
        choices = ", ".join(str(item) for item in sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}")
    return result


def _validate_nf(Nf):
    if isinstance(Nf, (bool, np.bool_)) or not isinstance(Nf, (int, np.integer)):
        raise TypeError("Nf must be a non-boolean integer")
    Nf = int(Nf)
    if not 0 <= Nf <= 16:
        raise ValueError("Nf must lie in the asymptotically-free range 0..16")
    return Nf


######################################### Strong Coupling Constant #########################################

def beta(Nf):
    Nf = _validate_nf(Nf)
    # beta(as = alphas / pi), unlike current mainstream choice of as = alphas / (4*pi)
    zeta = scipy.special.zeta
    beta0 = (11.0 - 2.0 * Nf / 3) / 4
    beta1 = (102.0 - 38.0 * Nf / 3) / 16
    beta2 = (2857.0 - 5033.0 * Nf / 9 + 325.0 * Nf * Nf / 27) / 128
    beta3 = (149753.0 / 6 + 3564.0 * zeta(3) - (1078361.0 / 162 + 6508.0 * zeta(3) / 27) * Nf + \
            (50065.0 / 162 + 6472.0 * zeta(3) / 81) * Nf * Nf + 1093.0 * Nf * Nf * Nf / 729) / 256
    beta4 = (8157455.0 / 16 + 621885.0 * zeta(3) / 2 - 88209.0 * zeta(4) / 2 - 288090.0 * zeta(5) + \
            (-336460813.0 / 1944 - 4811164.0 * zeta(3) / 81 + 33935.0 * zeta(4) / 6 + 1358995.0 * zeta(5) / 27) * Nf + \
            (25960913.0 / 1944 + 698531.0 * zeta(3) / 81 - 10526.0 * zeta(4) / 9 - 381760.0 * zeta(5) / 81) * Nf * Nf + \
            (-630559.0 / 5832 - 48722.0 * zeta(3) / 243 + 1618.0 * zeta(4) / 27 + 460.0 * zeta(5) / 9) * Nf * Nf * Nf + \
            (1205.0 / 2916 - 152.0 * zeta(3) / 81) * Nf * Nf * Nf * Nf) / 1024
    beta1 = beta1/beta0
    beta2 = beta2/beta0
    beta3 = beta3/beta0
    beta4 = beta4/beta0
    return beta0, beta1, beta2, beta3, beta4


def alpha_s(mu_scale, Nf, Lambda=None):
    # as = alphas / pi, unlike current mainstream choice of as = alphas / (4*pi)
    Nf = _validate_nf(Nf)
    mu_scale = _finite_real_scalar(mu_scale, "mu_scale", positive=True)
    if Lambda is None:
        if Nf != 3:
            raise ValueError(
                "the default Lambda=0.332 GeV is defined only for Nf=3; "
                "pass an explicit flavor-appropriate Lambda"
            )
        Lambda = 0.332
    Lambda = _finite_real_scalar(Lambda, "Lambda", positive=True)
    # This perturbative expansion was documented for l > 2. Evaluating below
    # that domain makes log(log(l^2)) invalid or physically uncontrolled.
    if mu_scale / Lambda <= 2:
        raise ValueError("alpha_s expansion requires mu_scale / Lambda > 2")
    b0, b1, b2, b3, b4 = beta(Nf)
    l = mu_scale / Lambda #l > 2!!!
    L = 2 * np.log(l)
    LL = np.log(L)
    bL = b0 * L
    as_tmp, a_s = np.zeros((5)), np.zeros((5))
    as_tmp[0] = 1 / bL
    as_tmp[1] = -b1 * LL / bL / bL
    as_tmp[2] = (b1 * b1 * (LL * LL - LL - 1) + b2)/(pow(bL, 3))
    as_tmp[3] = (pow(b1, 3) * (-pow(LL, 3) + 2.5 * LL * LL + 2 * LL - 1.0 / 2) - 3 * b1 * b2 * LL + b3 / 2)/pow(bL, 4)
    as_tmp[4] = (pow(b1, 4) * (pow(LL, 4) - 13.0 / 3 * LL * LL * LL - 3.0 / 2 * LL * LL + 4 * LL + 7.0 / 6)
            + 3 * b1 * b1 * b2 * (2 * LL * LL - LL - 1) - b1 * b3 * (2 * LL + 1.0 / 6) + 5.0 / 3 * b2 * b2 + b4 / 3) / pow(bL, 5)
    a_s[0] = as_tmp[0]
    a_s[1] = as_tmp[0] + as_tmp[1]
    a_s[2] = as_tmp[0] + as_tmp[1] + as_tmp[2]
    a_s[3] = as_tmp[0] + as_tmp[1] + as_tmp[2] + as_tmp[3]
    a_s[4] = as_tmp[0] + as_tmp[1] + as_tmp[2] + as_tmp[3] + as_tmp[4]
    return a_s * np.pi


def plot_alphas_mu(output_path=None, show=False):
    import matplotlib.pyplot as plt

    Nf, Lambda = 3, 0.332
    mu_min, mu_max, mu_step = 0.6, 10.0, 0.01
    mu_values = np.arange(mu_min, mu_max + mu_step, mu_step)
    alphas_values = np.zeros((5, len(mu_values)))
    for i, mu in enumerate(mu_values):
        alphas = alpha_s(mu, Nf, Lambda)
        for loop in range(5):
            alphas_values[loop, i] = alphas[loop]
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c', 'm']
    line_styles = ['-', '--', '-.', ':', '-']
    labels = ['1-loop', '2-loop', '3-loop', '4-loop', '5-loop']
    for loop in range(5):
        plt.plot(mu_values, alphas_values[loop],
                color=colors[loop], linestyle=line_styles[loop],
                linewidth=2, label=labels[loop])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlabel(r'$\mu$ [GeV]', fontsize=14)
    plt.ylabel(r'$\alpha_s(\mu)$', fontsize=14)
    plt.title(r'Strong coupling constant $\alpha_s$ vs. scale $\mu$ (Nf=3, $\Lambda$=0.332 GeV)', fontsize=16)
    plt.xlim(mu_min, mu_max)
    plt.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    if show:
        plt.show()
    return plt.gcf()

######################################### Helicity Matching Coefficient #########################################

def _validate_matching_inputs(muR, mu_scale, loop, is_pade):
    _finite_real_scalar(muR, "muR", positive=True)
    _finite_real_scalar(mu_scale, "mu_scale", positive=True)
    _choice_integer(loop, "loop", {1, 2, 3, 4})
    _choice_integer(is_pade, "is_pade", {0, 1})


def _pade_next_coefficient(r2, r3, label):
    """Return r3^2/r2 only when the legacy Padé estimate is well defined."""

    values = np.asarray([r2, r3], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Padé inputs for {label} must be finite")
    scale = max(1.0, abs(float(r3)))
    if abs(float(r2)) <= 1.0e-12 * scale:
        raise ValueError(
            f"Padé denominator for {label} is zero or numerically singular; "
            "use the explicit fixed-order result"
        )
    estimate = float(r3) ** 2 / float(r2)
    if not np.isfinite(estimate):
        raise ValueError(f"Padé estimate for {label} is nonfinite")
    return estimate

def EMT_MatchingCoeff(g_0, muR, mu_scale) :
    g_0 = _finite_real_scalar(g_0, "g_0", positive=True)
    muR = _finite_real_scalar(muR, "muR", positive=True)
    mu_scale = _finite_real_scalar(mu_scale, "mu_scale", positive=True)
    nf, Nc = 3, 3
    RMS = 1 + (g_0**2 * nf / (16 * np.pi**2)) * (2 * np.log(mu_scale**2 / muR**2) / 3 + 10 / 9) - \
        (g_0**2 * Nc / (16 * np.pi**2)) * (5 / 12)
    return RMS


def Helicity_MatchingCoeff_tmp(
    muR, mu_scale, loop, is_pade, *, physical=False, physical_profile=None
) :
    _require_physical_profile(
        physical, physical_profile, legacy_api="Helicity_MatchingCoeff_tmp"
    )
    _validate_matching_inputs(muR, mu_scale, loop, is_pade)
    nf, Nc = 3, 3
    CA, CF = Nc, (Nc**2 - 1) / (2 * Nc)
    zeta = scipy.special.zeta
    alpha_s_5loop = alpha_s(mu_scale, nf)[4]
    a_s = alpha_s_5loop / (4 * np.pi)
    r1_11 = (367 * CA - 40 * nf + 12 * (11 * CA - 2 * nf) * np.log(mu_scale**2 / muR**2)) / 36
    r2_11 = (12 * (4649 * CA**2 - 1354 * CA * nf + 16 * nf * (-27 * CF + 5 * nf)) * np.log(mu_scale**2 / muR**2) + \
            72 * (11 * CA - 2 * nf)**2 * np.log(mu_scale**2 / muR**2)**2 - 16 * CA * nf * (1381 + 648 * zeta(3)) + \
            CA**2 * (99047 + 2430 * zeta(3)) + 4 * nf * (200 * nf + 27 * CF * (-161 + 96 * zeta(3)))) / 648
    r3_11 = ((432 * (11 * CA - 2 * nf) * (5057 * CA**2 - 1474 * CA * nf - 612 * CF * nf + 80 * nf**2) * \
            np.log(mu_scale**2 / muR**2)**2 + 1728 * (11 * CA - 2 * nf)**3 * np.log(mu_scale**2 / muR**2)**3 + \
            72 * np.log(mu_scale**2 / muR**2) * (-18 * CA**2 * nf * (28805 + 6606 * zeta(3)) + \
            CA**3 * (1273537 + 26730 * zeta(3)) + 8 * nf * (1539 * CF**2 - 200 * nf**2 - \
            54 * CF * nf * (-101 + 48 * zeta(3))) + 12 * CA * nf * \
            (9 * CF * (-2735 + 1056 * zeta(3)) + nf * (4895 + 1728 * zeta(3)))) + \
            32 * nf * (-2000 * nf**2 - 9 * CF * nf * (-20375 + 11664 * zeta(3)) + \
            81 * CF**2 * (1333 + 2760 * zeta(3) - 4320 * zeta(5))) + \
            CA**3 * (141023224 - 606366 * zeta(3) - 7648425 * zeta(5)) - \
            24 * CA**2 * nf * (2061755 + 897858 * zeta(3) - 304560 * zeta(5)) + \
            48 * CA * nf * (nf * (89833 + 56592 * zeta(3)) + 3 * CF * (-331903 + 141048 * zeta(3) + \
            38880 * zeta(5))))) / 46656
    r1_12 = (12 * CF * (2 + np.log(mu_scale**2 / muR**2))) / 4
    r2_12 = -((CF * (693 * CF + 578 * nf + (-2436 * CA + 324 * CF + 312 * nf) * np.log(mu_scale**2 / muR**2) + \
            (-396 * CA + 72 * nf) * np.log(mu_scale**2 / muR**2)**2 - 864 * CF * zeta(3) + \
            CA * (-4643 + 1188 * zeta(3)))) / 9 ) / 4
    r3_12 = ((CF * (540 * (2437 * CA**2 - 297 * CA * CF - 752 * CA * nf - 36 * CF * nf + 52 * nf**2) * \
            np.log(mu_scale**2 / muR**2)**2 + 1080 * (11 * CA - 2 * nf)**2 * np.log(mu_scale**2 / muR**2)**3 - \
            4 * CA * nf * (511465 + 324 * np.pi**4 - 37800 * zeta(3)) + \
            20 * nf**2 * (5005 + 432 * zeta(3)) + 12 * CF * nf * (-64495 + 108 * np.pi**4 + \
            19440 * zeta(3)) + 15 * CA * CF * (-104701 + 144288 * zeta(3)) - \
            180 * np.log(mu_scale**2 / muR**2) * (-567 * CF**2 - 500 * nf**2 - 54 * CF * nf * (-25 + 8 * zeta(3)) + \
            CA**2 * (-30395 + 6534 * zeta(3)) + 2 * CA * (2943 * CF + 4148 * nf - \
            2376 * CF * zeta(3) + 54 * nf * zeta(3))) + \
            270 * CF**2*(707 + 768 * zeta(3) - 2880 * zeta(5)) + \
            CA**2 * (8458205 - 3893130 * zeta(3) + 846450 * zeta(5)))) / 810) / 4
    r1_21 = -4 * (nf / 2)
    r2_21 = -4 * ((nf * (367 * CA - 40 * nf + 12 * (11 * CA - 2 * nf) * np.log(mu_scale**2 / muR**2))) / 72)
    r3_21 = -4 * ((nf * (12 * (4649 * CA**2 - 1354 * CA * nf + 16 * nf * (-27 * CF + 5 * nf)) * \
            np.log(mu_scale**2 / muR**2) + 72 * (11 * CA - 2 * nf)**2 * np.log(mu_scale**2 / muR**2)**2 - \
            16 * CA * nf * (1381 + 648 * zeta(3)) + CA**2 * (99047 + 2430 * zeta(3)) + \
            4 * nf * (200 * nf + 27 * CF * (-161 + 96 * zeta(3))))) / 1296)
    r1_22 = 0
    r2_22 = - 6 * CF * nf * (2 + np.log(mu_scale**2 / muR**2))
    r3_22 = (CF * nf * (693 * CF + 578 * nf + (-2436 * CA + 324 * CF + 312 * nf) * np.log(mu_scale**2 / muR**2) + \
            (-396 * CA + 72 * nf) * np.log(mu_scale**2 / muR**2)**2 - 864 * CF * zeta(3) + \
            CA * (-4643 + 1188 * zeta(3)))) / 18
    if loop == 1 :
        R11, R12, R21, R22 = \
            1 + a_s * r1_11, a_s * r1_12, a_s * r1_21, 1 + a_s * r1_22
    elif loop == 2 :
        R11, R12, R21, R22 = \
            1 + a_s * r1_11 + a_s**2 * r2_11, a_s * r1_12 + a_s**2 * r2_12, \
            a_s * r1_21 + a_s**2 * r2_21, 1 + a_s * r1_22 + a_s**2 * r2_22
    elif loop == 3 :
        # Three-loop coefficients are known above. Replacing them by a [1/1]
        # estimate made r3_22 divide by r1_22 == 0 and discarded known data.
        R11, R12, R21, R22 = \
            1 + a_s * r1_11 + a_s**2 * r2_11 + a_s**3 * r3_11, a_s * r1_12 + a_s**2 * r2_12 + a_s**3 * r3_12, \
            a_s * r1_21 + a_s**2 * r2_21 + a_s**3 * r3_21, 1 + a_s * r1_22 + a_s**2 * r2_22 + a_s**3 * r3_22
    elif loop == 4 :
        if is_pade == 1 :
            r4_11 = _pade_next_coefficient(r2_11, r3_11, "R11")
            r4_12 = _pade_next_coefficient(r2_12, r3_12, "R12")
            r4_21 = _pade_next_coefficient(r2_21, r3_21, "R21")
            r4_22 = _pade_next_coefficient(r2_22, r3_22, "R22")
        else :
            r4_11, r4_12, r4_21, r4_22 = 0, 0, 0, 0
        R11, R12, R21, R22 = \
            1 + a_s * r1_11 + a_s**2 * r2_11 + a_s**3 * r3_11 + a_s**4 * r4_11, a_s * r1_12 + a_s**2 * r2_12 + a_s**3 * r3_12 + a_s**4 * r4_12, \
            a_s * r1_21 + a_s**2 * r2_21 + a_s**3 * r3_21 + a_s**4 * r4_21, 1 + a_s * r1_22 + a_s**2 * r2_22 + a_s**3 * r3_22 + a_s**4 * r4_22
    return R11, R12, R21, R22


def Helicity_MatchingCoeff(
    muR, mu_scale, loop, is_pade, *, physical=False, physical_profile=None
) :
    _require_physical_profile(
        physical, physical_profile, legacy_api="Helicity_MatchingCoeff"
    )
    _validate_matching_inputs(muR, mu_scale, loop, is_pade)
    nf, Nc = 3, 3
    CA, CF = Nc, (Nc**2 - 1) / (2 * Nc)
    zeta = scipy.special.zeta
    alpha_s_5loop = alpha_s(mu_scale, nf)[4]
    a_s = alpha_s_5loop / (4 * np.pi)
    r0_11 = 1 / (1 + a_s * (2 * nf - 33) * (5 + 3 * np.log(mu_scale**2 / muR**2)) / 9)
    r1_11 = 49 / 4
    r2_11 = (24949 + (8916 - 888 * nf) * np.log(mu_scale**2 / muR**2) + 810 * zeta(3) - 32 * nf * (73 + 20 * zeta(3))) / 24
    r3_11 = (432 * (-33 + 2 * nf) * (-7911 + 866 * nf) * np.log(mu_scale**2 / muR**2)**2 + \
            216 * np.log(mu_scale**2 / muR**2) * (3022011 - 502138 * nf + 17244 * nf**2 + \
            30 * (-33 + 2 * nf) * (-81 + 64 * nf) * zeta(3)) - 243 * (-4828712 + 22458 * zeta(3) + 283275 * zeta(5)) + \
            8 * nf * (-24087531 + 732998 * nf + 18 * (-231425 + 8496 * nf) * zeta(3) + 2844720 * zeta(5))) / 15552
    r1_12 = (12 * CF * (2 + np.log(mu_scale**2 / muR**2))) / 4
    r2_12 = -((CF * (693 * CF + 578 * nf + (-2436 * CA + 324 * CF + 312 * nf) * np.log(mu_scale**2 / muR**2) + \
            (-396 * CA + 72 * nf) * np.log(mu_scale**2 / muR**2)**2 - 864 * CF * zeta(3) + \
            CA * (-4643 + 1188 * zeta(3)))) / 9 ) / 4
    r3_12 = ((CF * (540 * (2437 * CA**2 - 297 * CA * CF - 752 * CA * nf - 36 * CF * nf + 52 * nf**2) * \
            np.log(mu_scale**2 / muR**2)**2 + 1080 * (11 * CA - 2 * nf)**2 * np.log(mu_scale**2 / muR**2)**3 - \
            4 * CA * nf * (511465 + 324 * np.pi**4 - 37800 * zeta(3)) + \
            20 * nf**2 * (5005 + 432 * zeta(3)) + 12 * CF * nf * (-64495 + 108 * np.pi**4 + \
            19440 * zeta(3)) + 15 * CA * CF * (-104701 + 144288 * zeta(3)) - \
            180 * np.log(mu_scale**2 / muR**2) * (-567 * CF**2 - 500 * nf**2 - 54 * CF * nf * (-25 + 8 * zeta(3)) + \
            CA**2 * (-30395 + 6534 * zeta(3)) + 2 * CA * (2943 * CF + 4148 * nf - \
            2376 * CF * zeta(3) + 54 * nf * zeta(3))) + \
            270 * CF**2*(707 + 768 * zeta(3) - 2880 * zeta(5)) + \
            CA**2 * (8458205 - 3893130 * zeta(3) + 846450 * zeta(5)))) / 810) / 4
    r1_21 = -4 * (nf / 2)
    r2_21 = -4 * ((nf * (367 * CA - 40 * nf + 12 * (11 * CA - 2 * nf) * np.log(mu_scale**2 / muR**2))) / 72)
    r3_21 = -4 * ((nf * (12 * (4649 * CA**2 - 1354 * CA * nf + 16 * nf * (-27 * CF + 5 * nf)) * \
            np.log(mu_scale**2 / muR**2) + 72 * (11 * CA - 2 * nf)**2 * np.log(mu_scale**2 / muR**2)**2 - \
            16 * CA * nf * (1381 + 648 * zeta(3)) + CA**2 * (99047 + 2430 * zeta(3)) + \
            4 * nf * (200 * nf + 27 * CF * (-161 + 96 * zeta(3))))) / 1296)
    r1_22 = 0
    r2_22 = - 6 * CF * nf * (2 + np.log(mu_scale**2 / muR**2))
    r3_22 = (CF * nf * (693 * CF + 578 * nf + (-2436 * CA + 324 * CF + 312 * nf) * np.log(mu_scale**2 / muR**2) + \
            (-396 * CA + 72 * nf) * np.log(mu_scale**2 / muR**2)**2 - 864 * CF * zeta(3) + \
            CA * (-4643 + 1188 * zeta(3)))) / 18
    if loop == 1 :
        R11, R12, R21, R22 = \
            r0_11 + a_s * r1_11, a_s * r1_12, a_s * r1_21, 1 + a_s * r1_22
    elif loop == 2 :
        R11, R12, R21, R22 = \
            r0_11 + a_s * r1_11 + a_s**2 * r2_11, a_s * r1_12 + a_s**2 * r2_12, \
            a_s * r1_21 + a_s**2 * r2_21, 1 + a_s * r1_22 + a_s**2 * r2_22
    elif loop == 3 :
        # Keep the explicit three-loop coefficients; r1_22 is exactly zero.
        R11, R12, R21, R22 = \
            r0_11 + a_s * r1_11 + a_s**2 * r2_11 + a_s**3 * r3_11, a_s * r1_12 + a_s**2 * r2_12 + a_s**3 * r3_12, \
            a_s * r1_21 + a_s**2 * r2_21 + a_s**3 * r3_21, 1 + a_s * r1_22 + a_s**2 * r2_22 + a_s**3 * r3_22
    elif loop == 4 :
        if is_pade == 1 :
            r4_11 = _pade_next_coefficient(r2_11, r3_11, "R11")
            r4_12 = _pade_next_coefficient(r2_12, r3_12, "R12")
            r4_21 = _pade_next_coefficient(r2_21, r3_21, "R21")
            r4_22 = _pade_next_coefficient(r2_22, r3_22, "R22")
        else :
            r4_11, r4_12, r4_21, r4_22 = 0, 0, 0, 0
        R11, R12, R21, R22 = \
            r0_11 + a_s * r1_11 + a_s**2 * r2_11 + a_s**3 * r3_11 + a_s**4 * r4_11, a_s * r1_12 + a_s**2 * r2_12 + a_s**3 * r3_12 + a_s**4 * r4_12, \
            a_s * r1_21 + a_s**2 * r2_21 + a_s**3 * r3_21 + a_s**4 * r4_21, 1 + a_s * r1_22 + a_s**2 * r2_22 + a_s**3 * r3_22 + a_s**4 * r4_22
    return R11, R12, R21, R22




def Helicity_MatchingCoeff_array(muR_array, loop, is_pade, is_fixing_order = 1,
                                muscale = None, muscaleOVmuR = None, tmp = 0,
                                *, physical=False, physical_profile=None):
    _choice_integer(loop, "loop", {1, 2, 3, 4})
    _choice_integer(is_pade, "is_pade", {0, 1})
    is_fixing_order = _choice_integer(
        is_fixing_order, "is_fixing_order", {0, 1}
    )
    tmp = _choice_integer(tmp, "tmp", {0, 1})
    _require_physical_profile(
        physical,
        physical_profile,
        legacy_api=("Helicity_MatchingCoeff_tmp" if tmp else "Helicity_MatchingCoeff"),
    )
    if _contains_boolean(muR_array):
        raise TypeError("muR_array entries must be real scales, not booleans")
    raw_muR_array = np.asarray(muR_array)
    if np.iscomplexobj(raw_muR_array):
        raise TypeError("muR_array must be real")
    try:
        muR_array = np.asarray(raw_muR_array, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("muR_array must be a finite real array") from error
    if (
        muR_array.ndim != 1
        or muR_array.size == 0
        or not np.isfinite(muR_array).all()
        or np.any(muR_array <= 0)
    ):
        raise ValueError(
            "muR_array must be a nonempty finite one-dimensional positive array"
    )
    if is_fixing_order == 0:
        if muscaleOVmuR is None:
            raise ValueError("muscaleOVmuR is required for proportional scales")
        muscaleOVmuR = _finite_real_scalar(
            muscaleOVmuR, "muscaleOVmuR", positive=True
        )
    else:
        if muscale is None:
            raise ValueError("muscale is required for a fixed scale")
        muscale = _finite_real_scalar(muscale, "muscale", positive=True)
    R = np.zeros((4, muR_array.shape[0]))
    for i in range(muR_array.shape[0]) :
        if is_fixing_order == 0 :
            if tmp == 0 :
                R[0,i], R[1,i], R[2,i], R[3,i] = Helicity_MatchingCoeff(muR = muR_array[i], \
                mu_scale = muR_array[i] * muscaleOVmuR, loop = loop, is_pade = is_pade,
                physical=physical, physical_profile=physical_profile)
            else :
                R[0,i], R[1,i], R[2,i], R[3,i] = Helicity_MatchingCoeff_tmp(muR = muR_array[i], \
                mu_scale = muR_array[i] * muscaleOVmuR, loop = loop, is_pade = is_pade,
                physical=physical, physical_profile=physical_profile)
        else :
            if tmp == 0 :
                R[0,i], R[1,i], R[2,i], R[3,i] = Helicity_MatchingCoeff(muR = muR_array[i], \
                mu_scale = muscale, loop = loop, is_pade = is_pade,
                physical=physical, physical_profile=physical_profile)
            else :
                R[0,i], R[1,i], R[2,i], R[3,i] = Helicity_MatchingCoeff_tmp(muR = muR_array[i], \
                mu_scale = muscale, loop = loop, is_pade = is_pade,
                physical=physical, physical_profile=physical_profile)
    return R


class Helicity_RunningFactorCalculator:
    def __init__(
        self, nf=3, ope_type = "ghelicity", *, physical=False, physical_profile=None
    ):
        _require_physical_profile(
            physical,
            physical_profile,
            legacy_api="Helicity_RunningFactorCalculator",
        )
        nf = _validate_nf(nf)
        if nf != 3:
            raise ValueError(
                "the bundled legacy anomalous-dimension coefficients are only defined for nf=3"
            )
        if ope_type != "ghelicity":
            raise ValueError("ope_type must be 'ghelicity'")
        self.nf = nf
        self.beta_coeffs = self._beta_coefficients()
        self.gamma_coeffs = self._ghelicity_gamma_coefficients()

    def _beta_coefficients(self):
        zeta = scipy.special.zeta
        beta0 = -(-11 + (2 * self.nf)/3) / 4
        beta1 = -(-102 + (38 * self.nf)/3) / 4**2
        beta2 = -(-2857/2 + (5033 * self.nf)/18 - (325 * self.nf**2)/54) / 4**3
        beta3 = -(-149753/6 + (1078361 * self.nf)/162 - (50065*self.nf**2)/162 - (1093*self.nf**3)/729
                - 3564*zeta(3) + (6508 * self.nf * zeta(3))/27 - (6472 * self.nf**2 * zeta(3))/81) / 4**4
        beta4 = -(-8157455/16 + (336460813 * self.nf)/1944 - (25960913 * self.nf**2)/1944
                + (630559 * self.nf**3)/5832 - (1205*self.nf**4)/2916 + (9801*np.pi**4)/20
                - (6787 * self.nf * np.pi**4)/108 + (5263 * self.nf**2 * np.pi**4)/405
                - (809 * self.nf**3 * np.pi**4)/1215 - (621885 * zeta(3))/2 + (4811164 * self.nf * zeta(3))/81
                - (698531 * self.nf**2 * zeta(3))/81 + (48722 * self.nf**3 * zeta(3))/243
                + (152 * self.nf**4 * zeta(3))/81 + 288090 * zeta(5) - (1358995 * self.nf * zeta(5))/27
                + (381760 * self.nf**2 * zeta(5))/81 - (460 * self.nf**3 * zeta(5))/9) / 4**5
        return beta0, beta1, beta2, beta3, beta4

    def _ghelicity_gamma_coefficients(self):
        gamma11_0, gamma11_1, gamma11_2 = 9.0/4.0, 64.0/16.0,  643.833/64.0
        gamma12_0, gamma12_1, gamma12_2 = 4.0/4.0, 76.0/16.0,  316.679/64.0
        gamma21_0, gamma21_1, gamma21_2 = 0.0/4.0, 0.0/16.0,   0.0/64.0
        gamma22_0, gamma22_1, gamma22_2 = 0.0/4.0, -24.0/16.0, -456.0/64.0
        gamma_coeffs = {
            '11': [gamma11_0, gamma11_1, gamma11_2, 0.0],
            '12': [gamma12_0, gamma12_1, gamma12_2, 0.0],
            '21': [gamma21_0, gamma21_1, gamma21_2, 0.0],
            '22': [gamma22_0, gamma22_1, gamma22_2, 0.0]}
        return gamma_coeffs

    def _get_gamma_matrix(self, a_s):
        coeffs = self.gamma_coeffs
        gamma11 = sum(c * (a_s**(i+1)) for i, c in enumerate(coeffs['11']))
        gamma12 = sum(c * (a_s**(i+1)) for i, c in enumerate(coeffs['12']))
        gamma21 = sum(c * (a_s**(i+1)) for i, c in enumerate(coeffs['21']))
        gamma22 = sum(c * (a_s**(i+1)) for i, c in enumerate(coeffs['22']))
        return np.array([[gamma11, gamma12], [gamma21, gamma22]])

    def _solve_coupled_rg_equations(self, mu_sq_start, mu_sq_end, R_init, a_s_init):
        if _contains_boolean(R_init):
            raise TypeError("R_init must be a finite real (2,2) array")
        raw_R_init = np.asarray(R_init)
        if raw_R_init.dtype.kind == "b" or np.iscomplexobj(raw_R_init):
            raise TypeError("R_init must be a finite real (2,2) array")
        try:
            R_init = np.asarray(raw_R_init, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("R_init must be a finite real (2,2) array") from error
        mu_sq_start = _finite_real_scalar(
            mu_sq_start, "mu_sq_start", positive=True
        )
        mu_sq_end = _finite_real_scalar(mu_sq_end, "mu_sq_end", positive=True)
        a_s_init = _finite_real_scalar(a_s_init, "a_s_init", positive=True)
        if R_init.shape != (2, 2):
            raise ValueError("R_init must have shape (2,2)")
        if not np.isfinite(R_init).all():
            raise ValueError("R_init must be finite")
        from scipy.integrate import solve_ivp

        def ode_system(ln_mu2, y):
            a_s, R = y[0], np.array([[y[1], y[2]], [y[3], y[4]]])
            beta0, beta1, beta2, beta3, beta4 = self.beta_coeffs
            beta = -(beta0 * a_s**2 + beta1 * a_s**3 + beta2 * a_s**4 + beta3 * a_s**5 + beta4 * a_s**6)
            gamma = self._get_gamma_matrix(a_s)

            da_s_dlnmu2 = beta # d(as)/d(ln mu^2) = beta(as)
            dR_dlnmu2 = gamma @ R # dR/d(ln mu^2) = Gamma(as)*R
            dydt = np.zeros(5)
            dydt[0], dydt[1], dydt[2], dydt[3], dydt[4] = \
                da_s_dlnmu2, dR_dlnmu2[0, 0], dR_dlnmu2[0, 1], dR_dlnmu2[1, 0], dR_dlnmu2[1, 1]
            return dydt # dy/d(ln μ²)
        y0 = np.zeros(5)
        y0[0], y0[1:5] = a_s_init, R_init.flatten(order='C')
        ln_mu2_start, ln_mu2_end = np.log(mu_sq_start), np.log(mu_sq_end)
        solution = solve_ivp(ode_system, [ln_mu2_start, ln_mu2_end],
            y0, method='RK45', dense_output=True, rtol=1e-8, atol=1e-10)
        if solution.success:
            y_final = solution.y[:, -1]
            a_s_final, R_final = y_final[0], np.array([[y_final[1], y_final[2]], [y_final[3], y_final[4]]])
            U = np.dot(R_final, np.linalg.inv(R_init)) # R_final = U * R_init
            return U, a_s_final, R_final
        else:
            raise RuntimeError(f"failed to solve RG equation: {solution.message}")

    def calculate_running_factor(self, mu1, mu2, R_at_mu1=None):
        mu1 = _finite_real_scalar(mu1, "mu1", positive=True)
        mu2 = _finite_real_scalar(mu2, "mu2", positive=True)
        a_s1 = alpha_s(mu1, self.nf)[4] / np.pi
        if R_at_mu1 is None: R_init = np.eye(2)
        else: R_init = R_at_mu1
        U, a_s2, R_final = self._solve_coupled_rg_equations(mu1**2, mu2**2, R_init, a_s1)
        return U

    def calculate_running_array(self, mu_from, mu_to, R_init=None):
        mu_to = _finite_real_scalar(mu_to, "mu_to", positive=True)
        if np.isscalar(mu_from):
            U = self.calculate_running_factor(mu_from, mu_to, R_init)
            return U
        elif isinstance(mu_from, (list, tuple, np.ndarray)):
            if _contains_boolean(mu_from):
                raise TypeError("mu_from entries must be real scales, not booleans")
            values = np.asarray(mu_from)
            if values.ndim != 1 or values.size == 0 or np.iscomplexobj(values):
                raise ValueError("mu_from must be a nonempty real one-dimensional array")
            results = []
            for mu in values:
                U = self.calculate_running_factor(mu, mu_to, R_init)
                results.append(U)
            return np.array(results)
        else: raise TypeError("mu_from should be a scalar or numpy_array.")

    def run(self, mu1, mu2, R_at_mu1=None):
        """Stable public alias for `calculate_running_factor`."""

        return self.calculate_running_factor(mu1, mu2, R_at_mu1)

    def run_array(self, mu_from, mu_to, R_init=None):
        """Stable public alias for `calculate_running_array`."""

        return self.calculate_running_array(mu_from, mu_to, R_init)


def Get_Helicity_Running(
    mu_from, muscaleOVmuR, mu_to, *, physical=False, physical_profile=None
):
    _require_physical_profile(
        physical, physical_profile, legacy_api="Get_Helicity_Running"
    )
    muscaleOVmuR = _finite_real_scalar(
        muscaleOVmuR, "muscaleOVmuR", positive=True
    )
    mu_to = _finite_real_scalar(mu_to, "mu_to", positive=True)
    if _contains_boolean(mu_from):
        raise TypeError("mu_from entries must be real scales, not booleans")
    calc = Helicity_RunningFactorCalculator(
        nf=3,
        ope_type="ghelicity",
        physical=physical,
        physical_profile=physical_profile,
    )
    raw_from = np.asarray(mu_from)
    if raw_from.ndim == 0:
        scaled_from = _finite_real_scalar(
            raw_from.item(), "mu_from", positive=True
        ) * muscaleOVmuR
    elif raw_from.ndim == 1 and raw_from.size > 0:
        validated = [
            _finite_real_scalar(value, f"mu_from[{index}]", positive=True)
            for index, value in enumerate(raw_from)
        ]
        scaled_from = np.asarray(validated, dtype=float) * muscaleOVmuR
    else:
        raise ValueError("mu_from must be a scalar or nonempty one-dimensional array")
    U_values = calc.calculate_running_array(scaled_from, mu_to)
    return U_values.real


def Plot_Helicity_MatchingCoeff(apx, apy, apz, apt, lattspac, is_pade, \
                                is_a2p2 = 1, is_fixing_order = 1, muscale = None, muscaleOVmuR = None, RG_mask = None,
                                output_path = None, show = False) :
    import matplotlib.pyplot as plt

    a2p2 = apx**2 + apy**2 + apz**2 + apt**2
    a2hatp2 = 4 * (np.sin(apx / 2)**2 + np.sin(apy / 2)**2 + np.sin(apz / 2)**2 + np.sin(apt / 2)**2)
    muR_array = np.sqrt(a2p2) * 0.19733 / lattspac
    muR_array, a2p2, a2hatp2 = muR_array[RG_mask], a2p2[RG_mask], a2hatp2[RG_mask]
    for loop in range(2, 4) :
        R_array = Helicity_MatchingCoeff_array(muR_array, loop, is_pade, is_fixing_order, muscale, muscaleOVmuR)
        R_matrix = np.array([[R_array[0], R_array[1]],[R_array[2],R_array[3]]])
        if is_fixing_order == 0 :
            U_values = Get_Helicity_Running(muR_array, muscaleOVmuR, muscale)
            for i in range(R_matrix.shape[2]): R_matrix[..., i] = U_values[i, ...] @ R_matrix[..., i]
        if loop == 3 : als = 0.9
        elif loop == 2 : als = 0.2
        if is_a2p2 == 1 :
            plt.xlabel(r'$a^2p^2$', fontsize=25)
            if loop == 3 :
                plt.errorbar(a2p2, R_matrix[0, 0, :], capsize=4, alpha=als, fmt = 'o', color='red',   label=r"$R_{11}$")
                plt.errorbar(a2p2, R_matrix[0, 1, :], capsize=4, alpha=als, fmt = 's', color='green', label=r"$R_{12}$")
                plt.errorbar(a2p2, R_matrix[1, 0, :], capsize=4, alpha=als, fmt = '^', color='gray',  label=r"$R_{21}$")
                plt.errorbar(a2p2, R_matrix[1, 1, :], capsize=4, alpha=als, fmt = 'd', color='blue',  label=r"$R_{22}$")
            elif loop == 2 :
                plt.errorbar(a2p2, R_matrix[0, 0, :], capsize=4, alpha=als, fmt = 'o', color='red')
                plt.errorbar(a2p2, R_matrix[0, 1, :], capsize=4, alpha=als, fmt = 's', color='green')
                plt.errorbar(a2p2, R_matrix[1, 0, :], capsize=4, alpha=als, fmt = '^', color='gray')
                plt.errorbar(a2p2, R_matrix[1, 1, :], capsize=4, alpha=als, fmt = 'd', color='blue')
        else :
            plt.xlabel(r'$a^2\hat{p}^2$', fontsize=25)
            if loop == 3 :
                plt.errorbar(a2hatp2, R_matrix[0, 0, :], capsize=4, alpha=als, fmt = 'o', color='red',   label=r"$R_{11}$")
                plt.errorbar(a2hatp2, R_matrix[0, 1, :], capsize=4, alpha=als, fmt = 's', color='green', label=r"$R_{12}$")
                plt.errorbar(a2hatp2, R_matrix[1, 0, :], capsize=4, alpha=als, fmt = '^', color='gray',  label=r"$R_{21}$")
                plt.errorbar(a2hatp2, R_matrix[1, 1, :], capsize=4, alpha=als, fmt = 'd', color='blue',  label=r"$R_{22}$")
            elif loop == 2 :
                plt.errorbar(a2hatp2, R_matrix[0, 0, :], capsize=4, alpha=als, fmt = 'o', color='red')
                plt.errorbar(a2hatp2, R_matrix[0, 1, :], capsize=4, alpha=als, fmt = 's', color='green')
                plt.errorbar(a2hatp2, R_matrix[1, 0, :], capsize=4, alpha=als, fmt = '^', color='gray')
                plt.errorbar(a2hatp2, R_matrix[1, 1, :], capsize=4, alpha=als, fmt = 'd', color='blue')
    plt.title(r'$R_{ij}(a^2p^2,\mu^2=$' +f'{muscale**2:.2f}' + r'$\mathrm{GeV}^2)$', fontsize = 22)
    plt.legend()
    plt.grid()
    plt.axis([3, 11, -1.0, 4.0])
    if output_path is not None:
        plt.savefig(output_path)
    if show:
        plt.show()
    return plt.gcf()


def Plot_Helicity11_MatchingCoeff(apx, apy, apz, apt, lattspac, is_pade, \
                                is_a2p2 = 1, is_fixing_order = 1, muscale = None, muscaleOVmuR = None, RG_mask = None,
                                output_path = None, show = False) :
    import matplotlib.pyplot as plt

    a2p2 = apx**2 + apy**2 + apz**2 + apt**2
    a2hatp2 = 4 * (np.sin(apx / 2)**2 + np.sin(apy / 2)**2 + np.sin(apz / 2)**2 + np.sin(apt / 2)**2)
    muR_array = np.sqrt(a2p2) * 0.19733 / lattspac
    muR_array, a2p2, a2hatp2 = muR_array[RG_mask], a2p2[RG_mask], a2hatp2[RG_mask]
    for ilop in range(1, 4) :
        R_array = Helicity_MatchingCoeff_array(muR_array, ilop, is_pade, is_fixing_order, muscale, muscaleOVmuR)
        R_matrix = np.array([[R_array[0], R_array[1]],[R_array[2],R_array[3]]])
        if is_fixing_order == 0 :
            U_values = Get_Helicity_Running(muR_array, muscaleOVmuR, muscale)
            for i in range(R_matrix.shape[2]): R_matrix[..., i] = U_values[i, ...] @ R_matrix[..., i]

        if is_a2p2 == 1 :
            plt.xlabel(r'$a^2p^2$', fontsize=25)
            plt.errorbar(a2p2, R_matrix[0, 0, :], capsize=4, alpha=0.9, fmt = 'o', label=f"{ilop}-loop")
        else :
            plt.xlabel(r'$a^2\hat{p}^2$', fontsize=25)
            plt.errorbar(a2hatp2, R_matrix[0, 0, :], capsize=4, alpha=0.9, fmt = 'o', label=f"{ilop}-loop")
    plt.title(r'$R_{11}(a^2\hat{p}^2,\mu^2=$' +f'{muscale**2:.2f}' + r'$\mathrm{GeV}^2)$', fontsize = 22)
    plt.legend()
    plt.grid()
    plt.axis([3, 11, 1.0, 3.0])
    if output_path is not None:
        plt.savefig(output_path)
    if show:
        plt.show()
    return plt.gcf()



######################################### Momentum Interpolation #########################################

def momentum_interpolate(a2_hatp2, a2_hatp2_tmp, datasets, datasets_tmp, Ncase):
    for raw_values, name in (
        (a2_hatp2, "target momenta"),
        (a2_hatp2_tmp, "reference momenta"),
    ):
        if _contains_boolean(raw_values):
            raise TypeError(f"{name} must be finite real arrays, not booleans")
    raw_targets = np.asarray(a2_hatp2)
    raw_references = np.asarray(a2_hatp2_tmp)
    for values, name in (
        (raw_targets, "target momenta"),
        (raw_references, "reference momenta"),
    ):
        if values.dtype.kind == "b" or np.iscomplexobj(values):
            raise TypeError(f"{name} must be finite real arrays")
    try:
        targets = np.asarray(raw_targets, dtype=float)
        references = np.asarray(raw_references, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "target and reference momenta must be finite real arrays"
        ) from error
    datasets = np.asarray(datasets)
    datasets_tmp = np.asarray(datasets_tmp)
    if targets.ndim != 1 or references.ndim != 1:
        raise ValueError("target and reference momenta must be one-dimensional")
    if references.size < 2:
        raise ValueError("at least two reference momenta are required")
    if not np.isfinite(targets).all() or not np.isfinite(references).all():
        raise ValueError("target and reference momenta must be finite")
    if isinstance(Ncase, (bool, np.bool_)) or not isinstance(
        Ncase, (int, np.integer)
    ):
        raise TypeError("Ncase must be a non-boolean integer")
    if Ncase < 1:
        raise ValueError("Ncase must be a positive integer")
    if datasets.ndim != 3 or datasets_tmp.ndim != 3:
        raise ValueError("datasets and datasets_tmp must have three axes")
    if datasets.shape[0] != Ncase or datasets_tmp.shape[0] != Ncase:
        raise ValueError("Ncase must match the leading dataset axes")
    if datasets.shape[1] <= 9 or datasets_tmp.shape[1] <= 9:
        raise ValueError("datasets require mean/error slots at axes 8 and 9")
    if datasets.shape[2] != targets.size or datasets_tmp.shape[2] != references.size:
        raise ValueError("dataset momentum axes must match target/reference arrays")

    order = np.argsort(references)
    references = references[order]
    reference_data = datasets_tmp[..., order]
    if np.any(np.diff(references) <= 0):
        raise ValueError("reference momenta must be unique")
    if targets.size and (targets.min() < references[0] or targets.max() > references[-1]):
        raise ValueError("target momenta must lie inside the reference range")

    output_dtype = np.result_type(datasets.dtype, datasets_tmp.dtype, np.float64)
    Z_prime_Im_mean = np.empty((Ncase, targets.size), dtype=output_dtype)
    Z_prime_Im_sdev = np.empty((Ncase, targets.size), dtype=output_dtype)
    for i, mom in enumerate(targets):
        exact = np.flatnonzero(np.isclose(references, mom, rtol=0.0, atol=1.0e-14))
        if exact.size:
            index = int(exact[0])
            Z_prime_Im_mean[:, i] = reference_data[:, 8, index]
            Z_prime_Im_sdev[:, i] = reference_data[:, 9, index]
            continue
        right_idx = int(np.searchsorted(references, mom, side="right"))
        left_idx = right_idx - 1
        x0, x1 = references[left_idx], references[right_idx]
        weight = (mom - x0) / (x1 - x0)
        y0, y1 = reference_data[:, 8, left_idx], reference_data[:, 8, right_idx]
        z0, z1 = reference_data[:, 9, left_idx], reference_data[:, 9, right_idx]
        Z_prime_Im_mean[:, i] = (1 - weight) * y0 + weight * y1
        # No covariance is supplied by this legacy API; propagate endpoint
        # standard deviations under the explicit independence assumption.
        Z_prime_Im_sdev[:, i] = np.sqrt(
            (1 - weight) ** 2 * z0**2 + weight**2 * z1**2
        )
    return Z_prime_Im_mean, Z_prime_Im_sdev



######################################### Smear Outer leg Subtraction #########################################

def poly_func(x, p):
    return sum(p_i * x**i for i, p_i in enumerate(p))


def AA_subtract(a2p2, AovA, a2p2_low, a2p2_high, n_poly, is_show_fitting_result = 0):
    if _contains_boolean(a2p2):
        raise TypeError("a2p2 must be real, not boolean")
    a2p2 = np.asarray(a2p2)
    if np.iscomplexobj(a2p2):
        raise TypeError("a2p2 must be real")
    try:
        a2p2 = np.asarray(a2p2, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("a2p2 must be a finite real array") from error
    if a2p2.ndim != 1 or a2p2.size == 0 or not np.isfinite(a2p2).all():
        raise ValueError("a2p2 must be one nonempty finite one-dimensional array")
    if _contains_boolean(AovA):
        raise TypeError("AovA must be real, not boolean")
    AovA = np.asarray(AovA)
    if AovA.ndim != 2 or AovA.shape[0] == 0 or AovA.shape[1] != a2p2.size:
        raise ValueError("AovA must have shape (samples, len(a2p2))")
    if np.iscomplexobj(AovA):
        imag_max = float(np.max(np.abs(AovA.imag))) if AovA.size else 0.0
        real_max = float(np.max(np.abs(AovA.real))) if AovA.size else 0.0
        if imag_max > 1.0e-12 + 1.0e-10 * max(1.0, real_max):
            raise ValueError(
                "AA_subtract received non-negligible complex data; choose and "
                "document the projected real observable before fitting"
            )
        AovA = AovA.real
    try:
        AovA = np.asarray(AovA, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("AovA must be a finite real sample array") from error
    if not np.isfinite(AovA).all():
        raise ValueError("AovA must contain only finite values")
    a2p2_low = _finite_real_scalar(a2p2_low, "a2p2_low")
    a2p2_high = _finite_real_scalar(a2p2_high, "a2p2_high")
    if a2p2_low > a2p2_high:
        raise ValueError("a2p2_low must not exceed a2p2_high")
    if isinstance(n_poly, (bool, np.bool_)) or not isinstance(
        n_poly, (int, np.integer)
    ):
        raise TypeError("n_poly must be a non-boolean integer")
    if n_poly < 0:
        raise ValueError("n_poly must be nonnegative")
    is_show_fitting_result = _choice_integer(
        is_show_fitting_result, "is_show_fitting_result", {0, 1}
    )
    mask = (a2p2 >= a2p2_low) & (a2p2 <= a2p2_high)
    if int(mask.sum()) < n_poly + 1:
        raise ValueError("fit window must contain at least n_poly+1 momentum points")
    denominator = np.mean(AovA, axis=0)
    scale = max(1.0, float(np.max(np.abs(denominator))))
    if np.any(np.abs(denominator) <= np.finfo(float).eps * scale):
        raise ValueError("sample mean denominator is zero or numerically singular")
    import gvar as gv
    import lsqfit

    x_data, y_mean, y_sdev = \
        a2p2[mask], np.mean(AovA[:, mask], axis = 0), np.std(AovA[:, mask], axis = 0)
    y_data = [gv.gvar(mean, sdev) for mean, sdev in zip(y_mean, y_sdev)]
    prior = gv.gvar([0.0] * (n_poly + 1), [10.0] * (n_poly + 1))
    fit = lsqfit.nonlinear_fit(data=(x_data, y_data), fcn=poly_func, prior=prior)
    if is_show_fitting_result == 1 : print("AA_subtract results: ", fit.format(True))
    subtract_fac = fit.p[0].mean / denominator
    if not np.isfinite(subtract_fac).all():
        raise ValueError("AA_subtract produced a nonfinite subtraction factor")
    return subtract_fac


__all__ = [
    "beta",
    "alpha_s",
    "EMT_MatchingCoeff",
    "Helicity_MatchingCoeff_tmp",
    "Helicity_MatchingCoeff",
    "Helicity_MatchingCoeff_array",
    "Helicity_RunningFactorCalculator",
    "Get_Helicity_Running",
    "momentum_interpolate",
    "AA_subtract",
]
