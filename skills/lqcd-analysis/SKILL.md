---
name: lqcd-analysis
description: >
  Lattice QCD analysis pipeline skill. Takes correlator data from lattice
  measurements and extracts physics results with statistical and systematic
  error control. Covers: source-time shifting, correlator folding,
  jackknife/bootstrap resampling, gvar/lsqfit integration, effective mass,
  multi-state correlated fits (hadron spectrum), matrix element extraction
  (ratio method, summation method, simultaneous C₂+C₃ two-state fit),
  dispersion relation and speed of light, fit diagnostics (χ²/dof, Q-value,
  AIC, SVD cut), and scale conversion to physical units. Uses fit function
  templates from lqcd-physics-spectrum. Trigger on: "analyze correlators",
  "fit the data", "extract mass", "effective mass", "matrix element",
  "dispersion relation", "speed of light", "jackknife", "resampling",
  "lsqfit", or when contractions are done and results needed.
---

# LQCD Analysis Pipeline

## Purpose

Take raw correlator data C(t) measured on N_cfg gauge configurations and
extract physics results (masses, amplitudes, matrix elements, form factors)
with controlled statistical and systematic uncertainties. This skill generates
an **analysis script only** — it assumes the correlator data already exists
on disk. It does not produce data generation or propagator computation code.

## Code style

Generated analysis code should be a **flat, self-contained script** with all parameters (file paths, fit ranges, number of states, lattice dimensions, lattice spacing, etc.) hardcoded as plain variables at the top. Write the script to read top-to-bottom so a collaborator can immediately see and verify every analysis choice. This is standard practice in lattice QCD: analysis scripts are shared between collaborators for cross-checking, not maintained as reusable software. Command-line arguments (argparse) may be used sparingly for parameters that distinguish independent runs, but physical and analysis parameters must remain hardcoded in the file.

**Inline vs. function**: Data processing operations that carry physical meaning (source-time shifting, correlator folding, sign corrections, momentum projection averaging, etc.) should be written **inline** in the script, not wrapped in utility functions. This makes the physics reasoning visible at the point where it happens — a collaborator can immediately see *why* a roll or sign flip occurs. The exception is **fit model functions** (e.g., `make_c2_model`): lsqfit requires a callable, so these must be defined as functions. Note that the code examples in this skill document the algorithm as functions for clarity, but when generating actual scripts, inline the data processing steps.

## Input data

This skill takes **pre-computed correlator data** from disk as input: per-configuration correlator data (e.g., HDF5 files containing C(t) for each config and source time). Single-configuration data has no physical meaning — only ensemble-averaged quantities with statistical errors carry physical significance. Never include propagator computation or gauge configuration loading in an analysis script.

## Workflow overview

```
Raw C(t; t_src) per config
  → Shift to common t_src = 0
  → Fold (exploit time-reversal symmetry)
  → Construct gvar dataset (jackknife/bootstrap → mean + covariance)
  → Effective mass (visual diagnostic with error bars)
  → Fit range determination (t_min scan)
  → Correlated fit with lsqfit:
      • Spectrum: multi-state fit to C₂(t)
      • Matrix elements: ratio / summation / simultaneous C₂+C₃ fit
  → Diagnostics (χ²/dof, Q, AIC, prior-posterior consistency)
  → [Optional] Dispersion relation → speed of light
  → Scale conversion (lattice → physical units)
```

---

## Step 1: Correlator preprocessing

### 1a: Source time shifting

Different configurations often use different source time slices t_src
(randomized to reduce autocorrelations). All correlators must be shifted
to a common reference t_src = 0 **per configuration**, before folding or
ensemble averaging.

```python
def shift_correlator(C_raw, t_src, T):
    """Shift correlator so that source sits at t=0.
    C_raw: shape (..., T), last axis is time.
    """
    return np.roll(C_raw, -t_src, axis=-1)
```

**Anti-periodic boundary conditions**: If the correlator was built from
quark propagators without absorbing the temporal boundary phase, time
slices that roll past the boundary acquire a sign flip:

```python
def shift_correlator_apbc(C_raw, t_src, T):
    """Shift with sign correction for anti-periodic BC."""
    C = np.roll(C_raw, -t_src, axis=-1)
    if t_src > 0:
        C[..., T - t_src:] *= -1
    return C
```

Whether this sign correction is needed depends on how the contraction code
handles the boundary. Verify by inspecting C(T-1) on a single configuration
before and after shifting. If the contraction already folds in the boundary
phase (as PyQUDA does), a plain roll suffices.

### 1b: Correlator folding

After shifting to t_src = 0, exploit time-reversal symmetry to improve
statistics. Folding must be done **per configuration**.

**Mesons** (meson correlators are effectively periodic in time even with
anti-periodic quark boundary conditions):

```python
def fold_correlator(C, T):
    """Fold: C_fold(t) = (C(t) + C(T-t)) / 2, t = 0, ..., T/2.
    C: shape (..., T).  Returns: shape (..., T//2 + 1).
    """
    Thalf = T // 2
    C_fwd = C[..., :Thalf + 1]                                      # C(0..T/2)
    C_bwd = np.concatenate([C[..., :1],                              # C(T)=C(0)
                            C[..., -1:Thalf - 1:-1]], axis=-1)       # C(T-1)..C(T/2)
    return (C_fwd + C_bwd) / 2
```

**Baryons** with parity projector $P^+ = (1+\gamma_4)/2$: the forward- and
backward-propagating states have **opposite parity**. Do NOT fold — use the
full unfolded correlator and fit both parities (see Step 4).

**Sign check**: After folding, mesonic correlators should be positive (for
standard operator normalizations). If consistently negative, flip the sign.

---

## Step 2: Resampling and gvar dataset

### Jackknife resampling

```python
def jackknife_resample(data):
    """Generate leave-one-out jackknife resamples.
    data:    shape (N_cfg, ...)
    Returns: shape (N_cfg, ...) — sample i is the mean of all configs except i.
    """
    n = data.shape[0]
    total = data.sum(axis=0)
    return (total[np.newaxis] - data) / (n - 1)
```

### Constructing gvar arrays

**Method 1 — from raw per-config data** (simplest):

```python
import gvar as gv

y = gv.dataset.avg_data(C_all)   # C_all shape: (N_cfg, T)
# Returns gvar array of shape (T,) with mean and full covariance.
```

**Method 2 — from jackknife samples** (explicit control):

```python
def make_gvar_jackknife(jk_samples):
    """Create gvar array from jackknife samples.
    jk_samples: shape (N_cfg, ...)
    """
    n = jk_samples.shape[0]
    mean = jk_samples.mean(axis=0)
    # Jackknife covariance: (n-1) * sample_cov(ddof=0)
    cov = np.cov(jk_samples.reshape(n, -1).T, ddof=0) * (n - 1)
    return gv.gvar(mean.ravel(), cov).reshape(mean.shape)
```

### Two workflows for error propagation

**Workflow A — gvar native (preferred)**:
Build a gvar dataset once; all subsequent arithmetic and fits automatically
propagate errors and correlations through gvar.

```python
C_gvar = preprocess(C_raw_all, t_src_all, T, fold=True)
fit = lsqfit.nonlinear_fit(data=(t, C_gvar[t_min:t_max+1]),
                           fcn=model, prior=prior)
m0 = fit.p['dE0']          # gvar: central value + error
```

**Workflow B — jackknife refit** (for non-trivial derived quantities):
When subsequent analysis involves steps that cannot be expressed as simple
gvar arithmetic (thresholding, model selection, etc.), fit each jackknife
sample and compute statistics from the distribution of results.

```python
jk = jackknife_resample(C_all)
# Use the FULL-sample covariance for every fit (only the mean varies)
cov = np.cov(C_all[:, t_min:t_max+1].T) / C_all.shape[0]
masses = []
for i in range(len(jk)):
    y = gv.gvar(jk[i, t_min:t_max+1], cov)
    fit = lsqfit.nonlinear_fit(data=(t, y), fcn=model, prior=prior)
    masses.append(gv.mean(fit.p['dE0']))
m_mean = np.mean(masses)
m_err  = np.sqrt((len(masses) - 1) * np.var(masses))
```

### Complete preprocessing helper

```python
def preprocess(C_raw_all, t_src_all, T, fold=True):
    """Raw per-config data → gvar dataset.
    C_raw_all: (N_cfg, T), t_src_all: (N_cfg,).
    """
    C = np.array([shift_correlator(C_raw_all[i], t_src_all[i], T)
                  for i in range(len(C_raw_all))])
    if fold:
        C = np.array([fold_correlator(C[i], T) for i in range(len(C))])
    return gv.dataset.avg_data(C)
```

---

## Step 3: Effective mass

### Cosh effective mass (folded meson correlators)

```python
def effective_mass_cosh(C, T):
    """C: gvar array of shape (T//2+1,).
    Returns list of gvar (or None where ratio ≤ 1).
    """
    m_eff = []
    for t in range(1, len(C) - 1):
        ratio = (C[t - 1] + C[t + 1]) / (2 * C[t])
        if gv.mean(ratio) > 1:
            m_eff.append(np.arccosh(ratio))   # works on gvar via ufunc
        else:
            m_eff.append(None)
    return m_eff
```

### Log effective mass (unfolded / baryon correlators)

```python
def effective_mass_log(C):
    """C: gvar array. Returns gvar array of length len(C)-1."""
    return np.log(C[:-1] / C[1:])
```

When the input is a gvar array, error bars are propagated automatically.
Use `gv.mean()` and `gv.sdev()` to extract central values and errors
for plotting.

---

## Step 4: Two-point function fitting (hadron spectrum)

The spectral decomposition (see `lqcd-physics-spectrum`) determines the fit
function.
The fit amplitudes $A_n$ absorb all normalization factors. The fit model
below handles both mesons and baryons:

### Fit function with guaranteed energy ordering

Parametrize energies as cumulative sums of positive gaps:

$$E_0 = \Delta E_0,\quad E_n = E_{n-1} + \Delta E_n \quad (\Delta E_n > 0)$$

Use `log(...)` keys in the prior dict to enforce $\Delta E_n > 0$
automatically (lsqfit fits $\log\Delta E_n$ and exponentiates internally).

```python
import lsqfit

def make_c2_model(T, n_states, baryonic=False):
    """General two-point fit function.

    Parameters
    ----------
    T : int         Temporal extent of the lattice.
    n_states : int  Number of exponential states per parity channel.
    baryonic : bool If True, include backward opposite-parity states.
    """
    def model(t, p):
        ans = 0
        # Forward-propagating states
        E = 0
        for i in range(n_states):
            E = E + p[f'dE{i}']            # E_i = sum of gaps
            A = p[f'A{i}']
            if baryonic:
                ans += A * np.exp(-E * t)
            else:
                ans += A * (np.exp(-E * t) + np.exp(-E * (T - t)))
        # Backward opposite-parity states (baryons only)
        if baryonic:
            Eb = 0
            for i in range(n_states):
                Eb = Eb + p[f'dEb{i}']
                ans += p[f'Ab{i}'] * np.exp(-Eb * (T - t))
        return ans
    return model
```

### Prior setting

```python
def make_c2_prior(m_eff_est, n_states, baryonic=False, gap=0.4):
    """Build priors from an effective-mass plateau estimate.

    Parameters
    ----------
    m_eff_est : float  Rough ground-state mass in lattice units.
    gap : float        Expected energy gap to first excited state (lattice units).
    """
    prior = {}
    # Ground-state gap (≈ mass)
    prior['log(dE0)'] = gv.gvar(np.log(m_eff_est), 0.5)
    # Excited-state gaps
    for i in range(1, n_states):
        prior[f'log(dE{i})'] = gv.gvar(np.log(gap), 0.5)
    # Amplitudes — broad, sign unconstrained
    for i in range(n_states):
        prior[f'A{i}'] = gv.gvar(0, 10)
    # Opposite-parity channel for baryons
    if baryonic:
        prior['log(dEb0)'] = gv.gvar(np.log(m_eff_est + 0.3), 0.5)
        for i in range(1, n_states):
            prior[f'log(dEb{i})'] = gv.gvar(np.log(gap), 0.5)
        for i in range(n_states):
            prior[f'Ab{i}'] = gv.gvar(0, 10)
    return prior
```

### Performing the fit

```python
# 1. Preprocess
C_gvar = preprocess(C_raw_all, t_src_all, T, fold=True)

# 2. Effective mass → rough estimate for priors
m_eff = effective_mass_cosh(C_gvar, T)
m_est = gv.mean(m_eff[T // 4])      # pick a safe plateau point

# 3. Model + prior
model = make_c2_model(T, n_states=2)
prior = make_c2_prior(m_est, n_states=2)

# 4. Fit
t = np.arange(t_min, t_max + 1)
y = C_gvar[t_min:t_max + 1]
fit = lsqfit.nonlinear_fit(data=(t, y), fcn=model, prior=prior)
print(fit)

# 5. Ground-state mass
m0 = fit.p['dE0']   # gvar with central value and error
```

### Fit range determination (t_min scan)

```python
results = []
for tmin in range(2, T // 4):
    t = np.arange(tmin, t_max + 1)
    y = C_gvar[tmin:t_max + 1]
    fit = lsqfit.nonlinear_fit(data=(t, y), fcn=model, prior=prior)
    results.append(dict(tmin=tmin, m0=fit.p['dE0'],
                        chi2_dof=fit.chi2 / fit.dof, Q=fit.Q))
```

**Selection criteria** (apply in order):

1. $Q > 0.05$ (fit is statistically acceptable)
2. $m(t_\min)$ is stable: $|m(t_\min) - m(t_\min + 1)| < \sigma_m$
3. Choose the smallest qualifying $t_\min$ (maximizes data usage)
4. Cross-check with AIC: if AIC strongly favors a larger $t_\min$, prefer it

If no $t_\min$ gives $Q > 0.05$, the covariance matrix may be
ill-conditioned — apply SVD cut (see below).

### Covariance matrix conditioning (SVD cut)

When $N_\text{cfg} \lesssim N_\text{data}$, the sample covariance becomes
singular or ill-conditioned.

**SVD cut** (built into lsqfit):

```python
fit = lsqfit.nonlinear_fit(data=(t, y), fcn=model, prior=prior,
                           svdcut=1e-3)
```

**Automated SVD diagnosis**:

```python
s = gv.dataset.svd_diagnosis(C_all[:, t_min:t_max + 1])
print(s.svdcut)   # recommended SVD cut value
```

**Cross-check**: compare correlated and uncorrelated fits. If central values
agree but errors differ, the correlated fit is reliable; the discrepancy
is only in error estimation.

---

## Step 5: Matrix element extraction (three-point functions)

### General framework

Matrix elements are extracted from three-point functions $C_3(\tau, t_\text{sep})$
with current insertion $J$ at time $\tau$ between source ($t=0$) and sink
($t=t_\text{sep}$). The spectral decomposition (see
`lqcd-physics-spectrum`) gives:

$$C_3(\tau,\, t_\text{sep}) = \sum_{n,m} B_{nm}\; e^{-E_n(t_\text{sep} - \tau)}\, e^{-E_m\,\tau}$$

where $B_{nm} \propto Z_n \,\mathcal{M}_{nm}\, Z_m$ factorizes into overlap
factors (shared with $C_2$) and the matrix element
$\mathcal{M}_{nm} = \langle n | J | m \rangle$. The target is
$\mathcal{M}_{00}$.

**Preprocessing**: shift $t_\text{src} \to 0$ per configuration (same as
for $C_2$). Folding does NOT apply to three-point functions.

### 5a: Ratio method

A model-independent extraction that cancels overlap factors and exponential
time dependence, yielding a plateau whose value is $\mathcal{M}$.

**General formula** (works for elastic and inelastic cases, any momentum):

$$R(\tau,\, t_\text{sep}) = \frac{C_3(\tau,\, t_\text{sep})}{C_2^\text{snk}(t_\text{sep})}\;\sqrt{\frac{C_2^\text{src}(t_\text{sep} - \tau)\; C_2^\text{snk}(\tau)\; C_2^\text{snk}(t_\text{sep})}{C_2^\text{snk}(t_\text{sep} - \tau)\; C_2^\text{src}(\tau)\; C_2^\text{src}(t_\text{sep})}}$$

In the ground-state limit ($\Delta E \cdot \tau \gg 1$ and
$\Delta E \cdot (t_\text{sep} - \tau) \gg 1$), the square root factor
cancels the exponential mismatch and overlap-factor ratio, leaving:

$$R \;\xrightarrow{\;\text{plateau}\;}\; \mathcal{M}$$

**Simplification** — for the zero-momentum elastic case
($C_2^\text{src} = C_2^\text{snk} \equiv C_2$), the square root equals 1
and the ratio reduces to:

$$R(\tau,\, t_\text{sep}) = \frac{C_3(\tau,\, t_\text{sep})}{C_2(t_\text{sep})}$$

```python
def ratio_method(C3, C2_src, C2_snk, tau_arr, t_sep):
    """General ratio for matrix element extraction.
    All inputs are gvar arrays / values.
    tau_arr : array of current insertion times.
    """
    R = np.array([
        C3[tau] / C2_snk[t_sep] * gv.sqrt(
            C2_src[t_sep - tau] * C2_snk[tau] * C2_snk[t_sep] /
            (C2_snk[t_sep - tau] * C2_src[tau] * C2_src[t_sep])
        )
        for tau in tau_arr
    ])
    return R   # should plateau in tau
```

**Extract $\mathcal{M}$**: fit a constant to the plateau region, or average
over a visually identified plateau range:

```python
plateau = R[delta:-delta]         # exclude contact-term region
M = lsqfit.wavg(plateau)         # weighted average → gvar
```

### 5b: Summation method

Sum the ratio over insertion times to gain an extra power of excited-state
suppression:

$$S(t_\text{sep}) = \sum_{\tau=\delta}^{t_\text{sep} - \delta} R(\tau,\, t_\text{sep})$$

where $\delta \ge 1$ excludes contact terms at source and sink. At large
$t_\text{sep}$:

$$S(t_\text{sep}) = c_0 + \mathcal{M}\,t_\text{sep} + O\!\left(e^{-\Delta E\, t_\text{sep}}\right)$$

The slope in $t_\text{sep}$ gives the matrix element. Excited-state
contamination is $O(e^{-\Delta E\, t_\text{sep}})$, improved by one power
compared to the ratio method's $O(e^{-\Delta E\, \tau})$.

```python
def summation_method(R_dict, t_seps, delta=1):
    """R_dict: {t_sep: gvar array R(tau=0..t_sep)}.
    Returns the matrix element from a linear fit in t_sep.
    """
    S_vals = []
    for ts in t_seps:
        S_vals.append(sum(R_dict[ts][delta:ts - delta + 1]))
    t_arr = np.array(t_seps)
    S_arr = np.array(S_vals)                   # gvar array
    fit = lsqfit.nonlinear_fit(
        data=(t_arr, S_arr),
        fcn=lambda t, p: p['c0'] + p['M'] * t,
        prior={'c0': gv.gvar(0, 10), 'M': gv.gvar(0, 10)},
    )
    return fit.p['M']   # matrix element (gvar)
```

### 5c: Simultaneous two-state fit (C₂ + C₃)

The most robust approach: fit two-point and three-point functions together,
sharing energies and overlap factors. Excited-state contamination is
parametrized explicitly.

```python
def make_c2c3_model(T, n_states, t_seps):
    """Simultaneous C₂ + C₃ fit model.

    Parameters
    ----------
    T : int            Temporal extent.
    n_states : int     Number of states per channel.
    t_seps : list[int] Source-sink separations included in the fit.

    Data dict keys expected:
      'c2'          → two-point correlator values at times data['t2']
      'c3_{tsep}'   → three-point values at insertion times data['tau_{tsep}']
    """
    def model(data, p):
        result = {}
        # --- Two-point function ---
        t2 = data['t2']
        c2 = 0
        for i in range(n_states):
            E = sum(p[f'dE{j}'] for j in range(i + 1))
            c2 += p[f'Z{i}'] ** 2 * (np.exp(-E * t2)
                                      + np.exp(-E * (T - t2)))
        result['c2'] = c2
        # --- Three-point functions ---
        for ts in t_seps:
            tau = data[f'tau_{ts}']
            c3 = 0
            for n in range(n_states):
                En = sum(p[f'dE{j}'] for j in range(n + 1))
                for m in range(n_states):
                    Em = sum(p[f'dE{j}'] for j in range(m + 1))
                    c3 += (p[f'Z{n}'] * p[f'M{n}{m}'] * p[f'Z{m}']
                           * np.exp(-En * (ts - tau))
                           * np.exp(-Em * tau))
            result[f'c3_{ts}'] = c3
        return result
    return model
```

**Priors** — energies shared with C₂; overlap factors $Z_n$ connect C₂
and C₃; matrix element priors informed by the ratio plateau:

```python
prior = {}
# Energy gaps (same as C₂-only fit)
prior['log(dE0)'] = gv.gvar(np.log(m_est), 0.5)
prior['log(dE1)'] = gv.gvar(np.log(0.4), 0.5)
# Overlap factors (Z^2 plays the role of the C₂ amplitude A)
for i in range(n_states):
    prior[f'Z{i}'] = gv.gvar(0, 10)
# Matrix elements
prior['M00'] = gv.gvar(R_plateau_est, 1.0)    # informed by ratio
prior['M01'] = gv.gvar(0, 1)
prior['M10'] = gv.gvar(0, 1)
prior['M11'] = gv.gvar(0, 1)
```

**Data dict construction and fit**:

```python
x = {'t2': np.arange(t_min, t_max + 1)}
y = {'c2': C2_gvar[t_min:t_max + 1]}
for ts in t_seps:
    tau_range = np.arange(delta, ts - delta + 1)
    x[f'tau_{ts}'] = tau_range
    y[f'c3_{ts}'] = C3_gvar[ts][tau_range]

fit = lsqfit.nonlinear_fit(data=(x, y), fcn=model, prior=prior)
M = fit.p['M00']   # ground-state matrix element (gvar)
```

### Choosing the method

| Situation | Recommended method |
|---|---|
| Quick first look, single $t_\text{sep}$ | Ratio method |
| Multiple $t_\text{sep}$ available, moderate statistics | Summation method |
| High-precision result, excited states matter | Simultaneous fit |
| Non-zero momentum transfer | Ratio (general formula) |

### Notes on generality

**Limitation**: The simultaneous fit model above uses the meson cosh form
for $C_2$. For baryons, replace the cosh term with the forward + backward
parity structure from `make_c2_model(..., baryonic=True)`.

The framework above is **operator-agnostic**: the specific Lorentz/Dirac
structure of the current $J$ does not affect the fitting procedure — it
only enters through:

- The contraction that produces C₃ (handled by the
  `lqcd-physics-correlator` skill)
- Kinematic prefactors relating $\mathcal{M}$ to physical form factors
  (e.g., $g_A$, $f_+$, $g_M$), which depend on the operator and spin
  projection
- Operator renormalization (multiplicative $Z_J$ factor)

These channel-specific details should be applied **after** extracting
$\mathcal{M}$ from the fit.

---

## Step 6: Dispersion relation and speed of light

### Lattice momenta

On a periodic spatial lattice with extent $L$ (in lattice units):

$$p_i = \frac{2\pi n_i}{L},\quad n_i \in \mathbb{Z}$$

The lattice-improved momentum (matching the free-field lattice dispersion):

$$\hat{p}_i = 2\sin\!\left(\frac{\pi n_i}{L}\right)$$

### Fitting procedure

1. Compute two-point correlators at several momenta
   $\vec{n} = (0,0,0),\,(1,0,0),\,(1,1,0),\,(1,1,1),\,(2,0,0),\,\ldots$
2. Average over equivalent momenta related by cubic symmetry
   (e.g., $(1,0,0)$, $(0,1,0)$, $(0,0,1)$, and sign flips)
3. Extract energy $E(\vec{n})$ at each momentum from a two-point fit

**Important**: use gvar throughout so that correlations between energies
at different momenta (which share the same propagator data) are preserved.

4. Fit the dispersion relation:

**Continuum form** (valid for small $|p|$):

$$E^2(\vec{n}) = m^2 + c^2 \sum_i p_i^2$$

**Lattice-improved form** (better for larger momenta):

$$E^2(\vec{n}) = m^2 + c^2 \sum_i \hat{p}_i^2$$

The speed of light $c$ should equal 1 in the continuum limit. Deviation
from 1 measures the size of discretization effects.

```python
def fit_dispersion(energies, momenta_n, L):
    """Fit the dispersion relation.

    Parameters
    ----------
    energies : list[gvar]      Energy at each momentum (from C₂ fits).
    momenta_n : list[tuple]    Integer momentum vectors (nx, ny, nz).
    L : int                    Spatial lattice extent.

    Returns
    -------
    lsqfit fit object with keys 'm' (mass) and 'c2' (c²).
    """
    p2_hat = np.array([
        sum(4 * np.sin(np.pi * ni / L) ** 2 for ni in nvec)
        for nvec in momenta_n
    ])
    E2 = np.array([E ** 2 for E in energies])      # gvar array

    def model(x, p):
        return p['m'] ** 2 + p['c2'] * x

    prior = {
        'm':  gv.gvar(gv.mean(energies[0]), 0.1),
        'c2': gv.gvar(1.0, 0.5),
    }
    fit = lsqfit.nonlinear_fit(data=(p2_hat, E2), fcn=model, prior=prior)
    c = fit.p['c2'] ** 0.5
    print(f"m   = {fit.p['m']}")
    print(f"c   = {c}")
    print(f"c^2 = {fit.p['c2']}")
    return fit
```

If $c$ deviates significantly from 1, higher-order terms can be included:

$$E^2 = m^2 + c_2\,\hat{p}^2 + c_4\,(\hat{p}^2)^2$$

**Note**: The above assumes an isotropic lattice (same $c$ for all
spatial directions). For anisotropic lattices, fit separate coefficients
$c_i$ per direction.

---

## Step 7: Fit quality diagnostics

After every fit, check **all** of the following:

| Diagnostic | Acceptable range | Action if failed |
|---|---|---|
| $\chi^2/\text{dof}$ | 0.5 – 2.0 | Adjust fit range or SVD cut |
| Q-value | > 0.05 | Fit range too aggressive |
| Posterior vs prior | Posterior narrower than prior | If prior dominates → data has no constraining power |
| Energy ordering | $E_1 > E_0$ | If violated, fit is unphysical |
| AIC comparison | Lower is better | Compare $N$-state vs $(N\!+\!1)$-state |

```python
def print_diagnostics(fit, prior):
    """Print comprehensive fit diagnostics."""
    print(f"chi2/dof = {fit.chi2 / fit.dof:.2f},  Q = {fit.Q:.3f}")
    print(f"AIC = {fit.chi2 + 2 * fit.p.size:.1f}")
    for key in prior:
        if key not in fit.p:
            continue
        pr, po = prior[key], fit.p[key]
        pull   = abs(gv.mean(po) - gv.mean(pr)) / gv.sdev(pr)
        shrink = gv.sdev(po) / gv.sdev(pr)
        print(f"  {key:12s}: {po}  (pull={pull:.2f}, shrink={shrink:.2f})")
```

---

## Step 8: Scale conversion

Fitted masses are in **lattice units**. Convert to physical units:

$$m_\text{phys}\;\text{[MeV]} = \frac{m_\text{lat}}{a}\;\hbar c = m_\text{lat} \times \frac{197.3269804\;\text{MeV·fm}}{a\;\text{[fm]}}$$

Propagate the lattice-spacing uncertainty via gvar:

```python
a = gv.gvar(0.10530, 0.00018)       # fm — from ensemble registry
hbarc = 197.3269804                  # MeV·fm
m_phys = fit.p['dE0'] * hbarc / a   # gvar — full error propagation
print(f"m = {m_phys} MeV")
```

---

## Summary of conventions and pitfalls

1. **Always shift before folding**: $t_\text{src}$ alignment is the first
   step; forgetting it silently corrupts the ensemble average.
2. **Fold only mesons**: baryon correlators with $P^+$ projection must NOT
   be folded; the backward state is a different particle (opposite parity).
3. **Use `log(...)` keys for energy gaps**: this guarantees positive gaps
   and prevents unphysical state reordering during the fit.
4. **Covariance matrix**: when $N_\text{cfg} < N_\text{data}$, always apply
   SVD cut. Use `gv.dataset.svd_diagnosis` for a data-driven recommendation.
5. **Correlations across momenta**: when extracting energies at several
   momenta from the same propagator data, keep them as gvar arrays from a
   single dataset so that cross-correlations are preserved in the
   dispersion-relation fit.
6. **Matrix element vs form factor**: the fit gives the bare lattice matrix
   element $\mathcal{M}$. Kinematic decomposition (Lorentz structure) and
   operator renormalization ($Z_J$) must be applied afterward.
