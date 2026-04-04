---
name: lqcd-analysis
description: >
  Lattice QCD analysis pipeline skill. Use whenever you need to process
  correlator data, perform statistical analysis, or extract physics results
  from lattice measurements. Covers: correlator folding (periodic/anti-periodic
  boundary conditions), effective mass computation, jackknife/bootstrap
  resampling, fit range determination (t_min scan), lsqfit usage with
  Bayesian priors, covariance matrix conditioning (SVD cut, shrinkage),
  fit quality diagnostics (chi2/dof, Q-value, AIC, prior-posterior consistency),
  and unit conversion via scale setting. Trigger on: "analyze correlators",
  "fit the data", "extract mass", "effective mass", "statistical analysis",
  or when propagator contraction is complete and physics results are needed.
---

# LQCD Analysis Pipeline

## Purpose

Take raw correlator data C(t) measured on N_cfg gauge configurations and
extract physics results (masses, amplitudes, matrix elements) with
controlled statistical and systematic uncertainties.

## Workflow overview

```
Raw C(t) per config
  → Fold (exploit boundary conditions)
  → Resample (jackknife or bootstrap)
  → Effective mass (visual diagnostic)
  → t_min scan (determine fit range)
  → Correlated fit with lsqfit
  → Diagnostics (χ²/dof, Q, AIC, prior-posterior)
  → Scale conversion (lattice → physical units)
```

## Step 1: Correlator folding

Fermions with **anti-periodic** temporal boundary conditions:
  C_folded(t) = C(t) - C(T - t)    (for t = 0, ..., T/2)

This assumes the standard meson two-point function sign convention where
the backward-propagating state picks up a minus sign from the anti-periodic
BC. The spectral decomposition becomes:
  C_folded(t) ∝ A * (e^{-mt} + e^{-m(T-t)})

**Important**: verify the sign convention by checking that C_folded(t) is
positive (for pseudoscalar mesons with the standard operator normalization).
If it comes out negative, flip the overall sign — this is a convention issue,
not a physics error.

For **periodic** boundary conditions (typically bosonic observables):
  C_folded(t) = (C(t) + C(T - t)) / 2

## Step 2: Resampling

Use **jackknife** (default) or **bootstrap** for error estimation:

```python
import numpy as np

def jackknife_samples(data):
    """data shape: (N_cfg, T)"""
    n = data.shape[0]
    mean = data.mean(axis=0)
    return n * mean - (n - 1) * np.array([
        np.delete(data, i, axis=0).mean(axis=0) for i in range(n)
    ])
```

Jackknife is preferred when N_cfg is small (< 200) because bootstrap
can underestimate errors in that regime. For N_cfg > 500, both give
consistent results.

## Step 3: Effective mass

The effective mass provides a model-independent visualization of the
ground-state mass:

```python
def effective_mass_cosh(C, T):
    """Solve m_eff from cosh formula, appropriate for folded correlators."""
    import numpy as np
    m_eff = np.zeros(T // 2 - 1)
    for t in range(1, T // 2):
        ratio = (C[t-1] + C[t+1]) / (2 * C[t])
        if ratio > 1:
            m_eff[t-1] = np.arccosh(ratio)
        else:
            m_eff[t-1] = np.nan  # Signal lost — noise dominates
    return m_eff
```

Alternative (log ratio, simpler but less accurate near T/2):
  m_eff(t) = log(C(t) / C(t+1))

**Interpretation**: a plateau in m_eff(t) indicates the region where the
ground state dominates. The plateau value is your mass estimate. If no
clear plateau exists, you need better statistics or smearing.

## Step 4: Fit range determination (t_min scan)

This is the most judgment-intensive step. Protocol:

1. Fix t_max = T/2 (or T/2 - 1 to avoid contact terms)
2. Scan t_min from 2 to T/4
3. For each t_min, perform a correlated fit (see Step 5)
4. Record: m(t_min), σ_m(t_min), χ²/dof, Q-value, AIC

**Selection criteria** (apply in order):
- Q > 0.05 (fit is statistically acceptable)
- m(t_min) is stable: |m(t_min) - m(t_min+1)| < σ_m
- Choose the smallest t_min satisfying the above (maximizes data used)
- Cross-check with AIC: if AIC strongly favors a larger t_min, prefer that

**If no t_min gives Q > 0.05**: the covariance matrix may be ill-conditioned.
Try an uncorrelated fit first to see if the mass is sensible, then apply
SVD cut (see Step 5).

## Step 5: Correlated fit with lsqfit

```python
import gvar as gv
import lsqfit

# --- Prepare data ---
# C_mean: central values, shape (T_fit,)
# C_cov:  covariance matrix, shape (T_fit, T_fit)
# Both computed from jackknife samples

y = gv.gvar(C_mean, C_cov)
t = np.arange(t_min, t_max + 1)

# --- Define fit function ---
def two_state_meson(t, p, T):
    return (
        p['A0'] * (np.exp(-p['m0'] * t) + np.exp(-p['m0'] * (T - t))) +
        p['A1'] * (np.exp(-p['m1'] * t) + np.exp(-p['m1'] * (T - t)))
    )

# --- Set priors ---
# Ground state: informed by effective mass plateau
# Excited state: broad prior, typically m1 ~ m0 + 0.5 ± 0.5
m_eff_plateau = ...  # read from effective mass plot
prior = {
    'm0': gv.gvar(m_eff_plateau, 0.1),    # informed
    'A0': gv.gvar(1e-5, 1e-4),            # broad, positive
    'm1': gv.gvar(m_eff_plateau + 0.5, 0.5),  # excited state
    'A1': gv.gvar(1e-5, 1e-4),
}

# --- Fit ---
fit = lsqfit.nonlinear_fit(
    data=y,
    fcn=lambda t, p: two_state_meson(t, p, T=T),
    prior=prior,
)
print(fit)
```

### Covariance matrix conditioning

When N_cfg is comparable to or smaller than the number of data points,
the sample covariance matrix becomes singular or ill-conditioned.

**SVD cut** (built into lsqfit):
```python
fit = lsqfit.nonlinear_fit(data=y, fcn=fcn, prior=prior, svdcut=1e-4)
```
Rule of thumb: svdcut ≈ max(0, 1 - N_cfg / N_data) as a starting point.
Increase until χ²/dof ≈ 1.

**Alternative**: Use an uncorrelated fit as a cross-check. If correlated
and uncorrelated fits give consistent central values but different errors,
the correlated fit is likely fine — the issue is just error estimation.

## Step 6: Fit quality diagnostics

After fitting, check ALL of the following:

| Diagnostic          | Acceptable range     | Action if failed              |
|---------------------|----------------------|-------------------------------|
| χ²/dof              | 0.5 – 2.0            | Adjust fit range or SVD cut   |
| Q-value             | > 0.05               | Fit range too aggressive      |
| m0 posterior vs prior| Posterior narrower   | If prior dominates → data has |
|                     | than prior            | no constraining power         |
| m1 (excited state)  | m1 > m0              | If not, fit is unphysical     |
| AIC comparison      | Lower is better      | Compare 1-state vs 2-state    |

```python
# Quick diagnostic printout
print(f"chi2/dof = {fit.chi2 / fit.dof:.2f}")
print(f"Q = {fit.Q:.3f}")
print(f"m0 = {fit.p['m0']}")
print(f"m1 = {fit.p['m1']}")

# Prior-posterior comparison
for key in prior:
    pr = prior[key]
    po = fit.p[key]
    pull = abs(gv.mean(po) - gv.mean(pr)) / gv.sdev(pr)
    width_ratio = gv.sdev(po) / gv.sdev(pr)
    print(f"{key}: pull = {pull:.2f}, width_ratio = {width_ratio:.2f}")
```

## Step 7: Scale conversion

The fitted mass m0 is in lattice units. Convert to physical units:

  m_phys [MeV] = m0 / a

where `a` is the lattice spacing from the ensemble registry.

Alternatively, using the scale parameter:
  m_phys [MeV] = m0 * (hbar*c / a)

with hbar*c = 197.3269804 MeV·fm.

The lattice spacing `a` and its uncertainty should be propagated through
using gvar to maintain proper error budgets.
