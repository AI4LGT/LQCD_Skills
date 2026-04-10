---
name: lqcd-physics
description: >
  Lattice QCD physics reasoning skill. Derives the full chain from a
  physics observable to computable quantities: interpolating operator
  construction (mesons, baryons), correlator definition (two-point,
  three-point), spectral decomposition (completeness relation, overlap
  factors, fit function templates for lqcd-analysis), Wick contraction
  with γ₅-hermiticity and flavor symmetry, propagator requirements, and
  einsum expressions for contractions. Uses DeGrand-Rossi gamma basis
  (PyQUDA convention). Trigger on: hadron masses, decay constants, form
  factors, matrix elements, operator construction, Wick contraction,
  spectral decomposition, or "what correlators/propagators do I need".
---

# LQCD Physics Reasoning

## Purpose

Given a physics observable (hadron mass, decay constant, form factor, ...),
derive the complete chain:

  **Observable →  Wick contraction → Correlator(s) → Propagator(s) → Einsum(s)**

so that downstream tools (PyQUDA, analysis pipeline) know exactly what to
compute.

## Core workflow

### Step 1: Identify the interpolating operator(s)

For a target hadron with quantum numbers $J^{PC}$ and flavor content, write the interpolating operator. Use Dirac bilinears for mesons, and appropriate diquark-quark structures for baryons.

For example, the simplest interpolating operator for a $\pi^+$ meson is usually written as $\mathcal{O}_{\pi^+} = \bar{d}^a \gamma_5 u^a$, and the simplest operator for a proton is $\mathcal{O}_p = \epsilon^{abc} (u^a C\gamma_5 d^b) u^c$. The simplest local operators are usually sufficient for ground state mass extraction, but interpolating operators can be constructed with gamma matrices and gauge covariant derivatives to access different quantum numbers, excited states, and observables related to hadron structure in general. For example, if we want to compute pion distribution amplitudes, we should use non-local operators with quark fields separated by a Wilson line, e.g. $\mathcal{O}_{\pi^+}(z) = \bar{d}^a(0) \gamma_5 W(0,z) u^a(z)$.

**Gamma matrices convention**: Use the DeGrand-Rossi basis as the Euclidean Dirac basis, which is the default gamma basis in PyQUDA convention:

$$
\gamma_1 = \begin{pmatrix} 0 & i \sigma_1 \nonumber \\ -i \sigma_1 & 0 \end{pmatrix},\quad
\gamma_2 = \begin{pmatrix} 0 & -i \sigma_2 \\ i \sigma_2 & 0 \end{pmatrix},\quad
\gamma_3 = \begin{pmatrix} 0 & i \sigma_3 \\ -i \sigma_3 & 0 \end{pmatrix},\quad
\gamma_4 = \begin{pmatrix} 0 & I_{2 \times 2}\\ I_{2 \times 2} & 0 \end{pmatrix}, \\
\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4=\begin{pmatrix} I_{2 \times 2} & 0 \\ 0 & -I_{2 \times 2} \end{pmatrix},\quad
C = \gamma_2\gamma_4 = \begin{pmatrix} -i\sigma_2 & 0\\ 0 & i\sigma_2 \end{pmatrix}
$$

Clearly we have $\gamma_\mu^\dagger = \gamma_\mu$ for $\mu = 1,2,3,4$ in the DeGrand-Rossi basis. And we have gamma anticommutation relations $\{\gamma_\mu, \gamma_\nu\} = 2\delta_{\mu\nu}$ and $\{\gamma_5, \gamma_\mu\} = 0$.

### Step 2: Write the correlator

The basic observable in lattice QCD is the correlator of interpolating operators. For a mass extraction, this is a two-point function. For a matrix element or form factor, this is a three-point function with an appropriate current insertion.

A two-point function for a hadron is typically

$$C_2(\vec{p};t_f,t_i) = \langle \mathcal{O}(\vec{p},t_f) \mathcal{O}^\dagger(\vec{p},t_i) \rangle$$

Where $\vec{p}$ is the momentum, and the interpolating operators are projected onto this momentum by Fourier transformation $\mathcal{O}(\vec{p},t) = \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \mathcal{O}(\vec{x},t)$. By utilizing the time translation invariance of the vacuum, we can shift the time and average over equivalent time slices to enhance the signal. The operator for initial and final states can be the same or different depending on the observable (e.g. for a form factor you might have different operators at source and sink).

For a matrix element or form factor, you need a three-point function

$$C_3(\vec{q}; t_f,t_i,\tau) = \langle \mathcal{O}_\text{snk}(\vec{p}_f,t_f) J(\vec{q},\tau) \mathcal{O}^\dagger_\text{src}(\vec{p}_i,t_i) \rangle$$

with an appropriate current insertion J. Here, $t_f$ is the sink time, $\tau$ is the current insertion time, and $t_i$ is the source time. (We can shift the time slices to average over equivalent time slices to enhance the signal.) The transfer momentum is calculated by $\vec{q} = \vec{p}_i - \vec{p}_f$.

### Step 3: Wick contraction and propagator determination

Expand the correlator by contracting all quark-antiquark pairs into
propagators $S_f(x, y)$. Apply:

- **γ₅-hermiticity**: $S_f(x, y) = \gamma_5 S_f^\dagger(y, x) \gamma_5$
  → converts backward propagators into forward ones (saves inversions)
- **Flavor symmetry**: for degenerate u/d quarks, $S_u = S_d = S_l$
  → reduces number of distinct propagators needed
- **Charge conjugation / isospin**: may relate different diagram topologies

### Step 4: Determine propagator

A typical 2-point correlator $C_\pi(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_l^\dagger(\vec{x},t; \vec{y},0) S_l(\vec{x},t; \vec{y},0) ]$ requires summing over both source $\vec{y}$ and sink $\vec{x}$, which is impossible in the real computation. Instead, we usually use point source propagator or wall source propator to estimate the correlator. Sometimes we can also use volume source propagator with stochastic estimation, but this is less common for two-point functions. Gaussian smearing can be applied to all types of sources to enhance ground state overlap, and APE/HYP smearing can be applied to the gauge links used in the source construction to further improve the signal. The choice of source type and smearing parameters depends on the specific observable and the desired balance between computational cost and statistical precision.

**Point source**: We can use a point source at a fixed spatial location and time slice (e.g., $\vec{y} = \vec{y}_0$ at $t=t_0$), set the phase $e^{i\vec{p}\cdot\vec{y}_0}$ at this point, and compute the propagator from this source point to all spatial points at all time slices, namely $S_{l,\text{point}(\vec{p},\vec{y}_0,t_0)}(\vec{x},t)\equiv e^{i\vec{p}\cdot\vec{y}_0}S_l(\vec{x},t;\vec{y}_0,t_0)$. This gives us an estimate of the correlator: $C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{point}(\vec{-p}_2,\vec{y}_0,t_0)}^{\dagger}(\vec{x},t) S_{l,\text{point}(\vec{p}_1,\vec{y}_0,t_0)}(\vec{x},t) ]$, where $\vec{p}=\vec{p}_1+\vec{p}_2$. The extra negative sign on $\vec{p}_2$ comes from the conjugate transpose of the propagator. This is how to use point source propagators to calculate the correlator. Note the momentum phase here is only a complex factor, we can just ignore it and set it to 1 without affecting any physical results. Then the momentum index $\vec{0}$ can be eliminated and the correlator estimated can be written as $C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{point}(\vec{y}_0,t_0)}^{\dagger}(\vec{x},t) S_{l,\text{point}(\vec{y}_0,t_0)}(\vec{x},t) ]$. For better statistics, we can also use multiple point sources at different spatial locations and time slices, and average the resulting correlators.

**Wall source**: We can also use a wall source that spans the entire spatial volume at a fixed time slice (e.g., $t=t_0$), set the phase $e^{i\vec{p}\cdot\vec{x}}$ for each spatial point, and compute the propagator from this source to all spatial points at all time slices, namely $S_{l,\text{wall}(\vec{p},t_0)}(\vec{x},t)\equiv\sum_{y}e^{i\vec{p}\cdot\vec{y}}S_l(\vec{x},t;\vec{y},t_0)$. This gives us an estimate of the correlator: $C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{wall}(\vec{p},t_0)}^{\dagger}(\vec{x},t) S_{l,\text{wall}(\vec{p},t_0)}(\vec{x},t) ]$. For better statistics, we can also use multiple wall sources at different time slices, and average the resulting correlators.

**Volume source**: Generally we do not use volume sources for two-point functions, but they can be used for all-to-all propagator with stochastic estimation. A volume source is defined as $S_{l,\text{volume}(\vec{p},E)}(\vec{x},t)\equiv\sum_{\vec{y},\tau}e^{i(\vec{p}\cdot\vec{y}+E\tau)}S(\vec{x},t;\vec{y},\tau)$, which have the 4-momentum phase at each spatial point and time slice.

Output a list of propagators specifying:
- Quark flavor / mass parameter
- Source type (point, smeared-point with Gaussian/Wuppertal parameters)
- Source position(s) (time slice, number of sources per configuration)
- Whether APE/HYP smeared links are used for the source construction
- Sink treatment (point, smeared, or both → for SS/SP correlator matrix)

---

## Worked examples

See `examples/{pion,rho,proton}.md` for step-by-step demonstrations of the Wick contraction and propagator determination workflow.

**Einsum conventions** (used by all example files): propagators use the
data layout
`[parity][t][z][y][x][spin_snk][spin_src][color_snk][color_src]`,
and the momentum phase $e^{-i\vec{p}\cdot(\vec{x}-\vec{x}_0)}$ has layout
`[parity][t][z][y][x]`. The `parity` index reflects the even-odd
preconditioned lattice layout. Einsum contracts over spatial indices
(`w,z,y,x`) and spin-color indices, leaving only `t`. The hermitian
conjugate $S^\dagger$ is implemented via `.conj()` with transposed spin
and color index order (`jiba` instead of `ijab`).

---

## Spectral decomposition

After constructing the correlator (Step 2) and before computing it via
Wick contraction (Step 3), one should understand its time dependence by
inserting complete sets of energy eigenstates. This determines the **fit
function** that the analysis pipeline (lqcd-analysis) will use.

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

## Decision rules for source strategy

| Situation                        | Recommendation                     |
|----------------------------------|------------------------------------|
| Quick first look / debugging     | Point source, 1 per config         |
| Production meson spectroscopy    | Smeared source, 4 sources/config   |
| Baryon spectroscopy              | Smeared source essential           |
| Disconnected diagrams needed     | Stochastic volume sources (Z₂/Z₄) |
| Form factor / 3pt function       | Sequential source or stochastic    |

**Multiple source times**: Computing propagators from multiple source time slices per configuration (e.g., `t_src = 0, T/4, T/2, 3T/4`) multiplies the effective statistics and improves the signal-to-noise ratio, especially for baryons and excited states. Each source time yields an independent correlator measurement after shifting to `t_src = 0`. The "4 sources/config" in the table above refers to 4 different source time positions. When determining propagator requirements, **ask the user** to confirm the source type, smearing, positions, and number of source times, as the optimal choice depends on the target observable and available computational budget.

## Common pitfalls

1. **Forgetting disconnected diagrams**: Flavor-singlet mesons (η, η', σ)
   have disconnected quark-loop contributions. These are computationally
   expensive and require different techniques (stochastic estimation).
   For flavor non-singlet mesons (π⁺, K⁺, ρ⁺), there are no disconnected
   diagrams.

2. **Wrong sign convention**: The overall sign of C(t) depends on the
   operator normalization and the number of fermion anticommutations in the
   Wick contraction. Always verify the sign is consistent with the expected
   spectral decomposition: C(t) should be dominated by a positive or
   negative exponential depending on convention.

3. **Periodic vs anti-periodic BC**: Fermions use anti-periodic temporal
   boundary conditions, but the **composite** state's BC depends on the
   number of quarks:
     Mesons:  (-1)² = +1 → C(t) ∝ e^{-mt} + e^{-m(T-t)}  (cosh-like)
     Baryons: (-1)³ = -1 → backward state has opposite parity
   See the Spectral Decomposition section above for the fit function
   templates.
