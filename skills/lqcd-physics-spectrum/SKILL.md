---
name: lqcd-physics-spectrum
description: >
  Lattice QCD physics reasoning skill. Starting from a correlator
  definition, derives its spectral decomposition: completeness
  relation, overlap factors, backward-state structure, and fit
  function templates for lqcd-analysis. Assumes the operator and
  correlator have already been identified, typically by
  lqcd-physics-correlator. Trigger on: spectral decomposition,
  excited-state contamination, two-point or three-point fit models,
  backward-propagating states, or "what fitting functions should I
  use?".
---

# LQCD Physics Reasoning

## Purpose

Given a physics observable (hadron mass, decay constant, form factor, ...),
derive the complete chain:

  **Correlator(s) → Spectrum(s) → Fitting function(s)**

so that downstream tools (analysis pipeline) know exactly what to
compute.

Use this skill after the operator content and correlator definition are
already known. If those are still missing, first use
`lqcd-physics-correlator`.

## Spectral decomposition

Once the correlator has been specified, one should understand its time
dependence by inserting complete sets of energy eigenstates. This
determines the **fit function** that the analysis pipeline
(`lqcd-analysis`) will use.

### Completeness relation

An interpolating operator $\mathcal{O}$ with quantum numbers $J^{PC}$
couples to **all** eigenstates carrying those quantum numbers:

$$\langle 0 | \mathcal{O} | n, \vec{p} \rangle = Z_n(\vec{p})$$

where $|n, \vec{p}\rangle$ is the $n$-th eigenstate (ordered by energy,
$n = 0$ being the ground state) with three-momentum $\vec{p}$.

On a finite lattice with spatial volume $V = L^3$ and relativistic state
normalization $\langle n, \vec{p} | m, \vec{q} \rangle = 2 E_n V\,
\delta_{\vec{p}\vec{q}}\,\delta_{nm}$, the completeness relation reads:

$$\mathbf{1} = |0\rangle\langle 0| + \sum_{n \geq 1}\sum_{\vec{p}} \frac{1}{2 E_n(\vec{p})\,V}\;|n, \vec{p}\rangle\langle n, \vec{p}|$$

### Two-point function

Insert completeness between $\mathcal{O}$ and $\mathcal{O}^\dagger$
in the two-point function $C_2(\vec{p};\,t) = \langle \mathcal{O}(\vec{p},t)\,\mathcal{O}^\dagger(\vec{p},0)\rangle$.

At **zero temperature** ($T \to \infty$):

$$C_2(\vec{p};\,t) = \sum_n \frac{|Z_n(\vec{p})|^2}{2 E_n(\vec{p})}\;e^{-E_n(\vec{p})\,t}$$

(An overall volume factor may appear depending on whether one or both
operators are momentum-projected; in practice it is absorbed into $Z_n$.)

At **finite temporal extent** $T$, backward-propagating contributions appear.
Their sign depends on the boundary condition of the **composite** state:

- **Mesons** — composed of two anti-periodic quarks → effective **periodic**
  BC ($(-1)^2 = +1$):

$$C_2^\text{meson}(t) = \sum_n A_n\left(e^{-E_n t} + e^{-E_n(T-t)}\right)$$

- **Baryons** — composed of three anti-periodic quarks → effective
  **anti-periodic** BC ($(-1)^3 = -1$).  With parity projector
  $P^+ = (1+\gamma_4)/2$, the forward state has positive parity and the
  backward state has negative parity (the opposite-parity partner):

$$C_2^{P^+}(t) = \sum_n A_n^+ e^{-E_n^+ t} - \sum_n A_n^- e^{-E_n^-(T-t)}$$

Here the **fit amplitude** is defined as

$$A_n \equiv \frac{|Z_n|^2}{2 E_n}$$

absorbing all convention-dependent normalization factors into $Z_n$. The
amplitude $A_n$ is always positive for physical states; its magnitude
encodes how strongly the operator couples to the $n$-th state.

**Key physics**: smeared sources enhance $|Z_0|$ relative to excited-state
overlaps $|Z_{n \geq 1}|$, producing a cleaner plateau in the effective
mass. The spectral decomposition makes this quantitative: the excited-state
contamination in the effective mass is proportional to $(A_1/A_0)\,
e^{-\Delta E\, t}$ where $\Delta E = E_1 - E_0$.

### Three-point function

For a three-point function with current insertion $J$ at Euclidean time
$\tau$ ($0 < \tau < t_\text{sep}$):

$$C_3(\tau,\,t_\text{sep}) = \langle \mathcal{O}_f(\vec{p}_f,\,t_\text{sep})\;J(\vec{q},\,\tau)\;\mathcal{O}_i^\dagger(\vec{p}_i,\,0)\rangle$$

Insert completeness on **both** sides of the current:

$$C_3(\tau,\,t_\text{sep}) = \sum_{n,m} \frac{Z_n^f\,(Z_m^i)^*}{4\,E_n\,E_m}\;\langle n | J | m \rangle\;e^{-E_n(t_\text{sep}-\tau)}\,e^{-E_m\,\tau}$$

The crucial **factorization**: each coefficient separates into three
independent factors —

$$\underbrace{Z_n^f}_{\text{sink overlap}} \;\times\; \underbrace{\langle n | J | m \rangle}_{\text{matrix element}} \;\times\; \underbrace{(Z_m^i)^*}_{\text{source overlap}}$$

The overlap factors $Z_n$ are the **same** as those in the two-point
function.  This is what enables the simultaneous fit in lqcd-analysis:

- $C_2$ determines $E_n$ and $Z_n$ (or equivalently $A_n$)
- $C_3$, sharing $E_n$ and $Z_n$, determines the matrix elements
  $\mathcal{M}_{nm} \equiv \langle n | J | m \rangle$
- The ground-state matrix element $\mathcal{M}_{00}$ is the physics target

### Thermal effects

For $t_\text{sep} \ll T$, backward-propagating contributions to the
three-point function are exponentially suppressed and usually negligible.
When $t_\text{sep}$ is not small compared to $T$, additional thermal terms
appear and must be included in the fit model.

### Summary: spectral decomposition → analysis handoff

Given the operator choice and boundary conditions, the spectral
decomposition fully determines the **fit function template**:

| Correlator | Fit function | Free parameters |
|---|---|---|
| $C_2^\text{meson}(t)$ | $\sum_n A_n(e^{-E_n t} + e^{-E_n(T-t)})$ | $\{E_n,\,A_n\}$ |
| $C_2^{P^+\text{baryon}}(t)$ | $\sum_n A_n^+ e^{-E_n^+ t} - \sum_n A_n^- e^{-E_n^- (T-t)}$ | $\{E_n^+,\,A_n^+,\,E_n^-,\,A_n^-\}$ |
| $C_3(\tau,t_\text{sep})$ | $\sum_{n,m} B_{nm}\,e^{-E_n(t_\text{sep}-\tau)} e^{-E_m\tau}$ | $\{E_n,\,B_{nm}\}$ with $B_{nm} \propto Z_n\,\mathcal{M}_{nm}\,Z_m$ |

The analysis skill (lqcd-analysis) takes these templates as its fit models,
using the energy-gap parametrization $E_n = \sum_{k=0}^{n}\Delta E_k$
with $\Delta E_k > 0$ to ensure proper state ordering.

---

## Common pitfalls

1. **Periodic vs anti-periodic BC**: Fermions use anti-periodic temporal
   boundary conditions, but the **composite** state's BC depends on the
   number of quarks:
     Mesons:  (-1)² = +1 → C(t) ∝ e^{-mt} + e^{-m(T-t)}  (cosh-like)
     Baryons: (-1)³ = -1 → backward state has opposite parity
   See the Spectral Decomposition section above for the fit function
   templates.
