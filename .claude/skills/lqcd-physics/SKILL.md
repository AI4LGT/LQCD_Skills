---
name: lqcd-physics
description: >
  Lattice QCD physics reasoning skill. Use whenever the user specifies a
  hadronic physics goal (e.g. "compute the pion mass", "measure the nucleon
  axial charge") and you need to derive the required correlators and quark
  propagators. Covers: interpolating operator construction, Wick contraction,
  gamma-matrix algebra, γ₅-hermiticity, flavor/isospin symmetry exploitation,
  and source/sink smearing strategy. Trigger on any mention of hadron masses,
  decay constants, form factors, two-point or three-point functions, or
  quark-level contractions in a lattice QCD context.
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

For a target hadron with quantum numbers $J^{PC}$ and flavor content, write the interpolating operator. Use Dirac bilinears for mesons, and
appropriate diquark-quark structures for baryons.

For example, the simplest interpolating operator for a $\pi^+$ meson is usually written as $\mathcal{O}_{\pi^+} = \bar{d}^a \gamma_5 u^a$, and the simplest operator for a proton is $\mathcal{O}_p = \epsilon^{abc} (u^a C\gamma_5 d^b) u^c$.

**Convention**: Use the DeGrand-Rossi basis as the Euclidean Dirac basis, which is the default gamma basis in PyQUDA convention:

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

See the examples below for a step-by-step demonstration of the Wick contraction process.

### Step 4: Determine propagator requirements

Output a list of propagators specifying:
- Quark flavor / mass parameter
- Source type (point, smeared-point with Gaussian/Wuppertal parameters)
- Source position(s) (time slice, number of sources per configuration)
- Whether APE/HYP smeared links are used for the source construction
- Sink treatment (point, smeared, or both → for SS/SP correlator matrix)

---

## Worked examples

### Example 1: Pion mass (π⁺ channel)

**Goal**: Extract $m_\pi$

**Step 1 — Operator**:
  $$\mathcal{O}_{\pi^+} = \bar{d} \gamma_5 u$$

**Step 2 — Correlator**:
  $$C_\pi(\vec{p}; t,0) = \langle \mathcal{O}_{\pi^+}(\vec{p},t) \mathcal{O}^\dagger_{\pi^+}(\vec{p},0) \rangle$$

**Step 3a - Quark fields**: Expand the operator in terms of quark fields and Fourier transform:
  $$C_\pi(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \langle \bar{d}(\vec{x},t) \gamma_5 u(\vec{x},t) \bar{u}(\vec{y},0) \gamma_4 \gamma_5 \gamma_4 d(\vec{y},0) \rangle$$

**Step 3b — Wick contraction**:
One connected diagram (no disconnected pieces for charged pion, and the negative sign arises from the anticommutation of fermion fields):

  $$C_\pi(\vec{p}; t,0) = -\sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_d(\vec{y},0; \vec{x},t) \gamma_5 S_u(\vec{x},t; \vec{y},0) \gamma_4 \gamma_5 \gamma_4 ]$$

**Step 3c - Simplification**:
  1. Apply the $\gamma_5$-hermiticity and the flavor symmetry:
  $$C_\pi(\vec{p}; t,0) = - \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ \gamma_5 S_l^\dagger(\vec{x},t; \vec{y},0) \gamma_5 \gamma_5 S_l(\vec{x},t; \vec{y},0) \gamma_4 \gamma_5 \gamma_4 ]$$
  2. Apply the cyclic property to simplify the gamma matrix structure:
  $$C_\pi(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_l^\dagger(\vec{x},t; \vec{y},0) S_l(\vec{x},t; \vec{y},0) ]$$

**Step 4 — Propagators needed**:
We need the light quark propagator $S_l(x,y)$ from all source points to all sink points. To optimize, we usually do not calculate all source points. Instead, we estimate the correlator using a point source at a specific spatial location and time slice ($\vec{y}=\vec{x}_0, t=0$ here) and compute the propagator from this source point to all spatial points at all time slices $S_l(\vec{x},t; \vec{x}_0,0)$.
$$C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot (\vec{x} - \vec{x}_0)} \text{Tr}[ S_l^\dagger(\vec{x},t; \vec{x}_0,0) S_l(\vec{x},t; \vec{x}_0,0) ]$$

**Step 5 - Einsum to correlators**
The trace over spin and color indices can be expressed as an einsum operation. Consider a propagator with the data layout `[parity][t][z][y][x][spin_snk][spin_src][color_snk][color_src]`, and a momentum phase $e^{-i \vec{p} \cdot (\vec{x} - \vec{x}_0)}$ with the data layout `[parity][t][z][y][x]`. Here the `parity` index indicates we are using an even-odd preconditioned layout lattice. The trace over spin and color can be expressed as:
```
numpy.einsum('wtzyx,wtzyxjiba,wtzyxijab->t', phase, S_l.conj(), S_l)
```
Note we are performing a sum over the spatial indices (`w`, `z`, `y`, `x`) and the spin/color indices (`i`, `j`, `a`, `b`). The resulting array has only the time index `t`, which is what we need for the correlator. The conjugate of the propagator is taken care of by the `conj()` function, and the transpose is applied with the reversed order of `ij` and `ab` indices.

**Optional enhancement**:
- Gaussian-smeared source (N_smear ~ 50-100, r_smear ~ 4-6 in lattice units) with APE-smeared links (N_APE ~ 25, α_APE ~ 2.5) for better ground-state overlap
- If using smeared source: compute both SS and SP correlators for a variational/matrix analysis, or at minimum to cross-check plateau quality

---

### Example 2: Rho meson mass (ρ⁺ channel)

**Goal**: Extract $m_\rho$

**Step 1 — Operator**:
  $$\mathcal{O}_{\rho^+_i} = \bar{d} \gamma_i u \quad (i = 1, 2, 3 \text{ for the three polarizations})$$

**Step 2 — Correlator**: Average over polarizations for better statistics:
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \langle \mathcal{O}_{\rho^+_i}(\vec{p},t) \mathcal{O}^\dagger_{\rho^+_i}(\vec{p},0) \rangle$$

**Step 3a — Quark fields**: Expand the operator in terms of quark fields and Fourier transform:
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \sum_{\vec{x}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \langle \bar{d}(\vec{x},t) \gamma_i u(\vec{x},t) \bar{u}(\vec{y},0) \gamma_4 \gamma_i \gamma_4 d(\vec{y},0) \rangle$$

**Step 3b — Wick contraction**:
Same topology as pion, just replace $\gamma_5 \to \gamma_i$:

  $$C_\rho(\vec{p}; t,0) = -\frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_d(\vec{y},0; \vec{x},t) \gamma_i S_u(\vec{x},t; \vec{y},0) \gamma_4 \gamma_i \gamma_4 ]$$

**Step 3c — Simplification**:
  1. Apply the $\gamma_5$-hermiticity and the flavor symmetry:
  $$C_\rho(\vec{p}; t,0) = -\frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ \gamma_5 S_l^\dagger(\vec{x},t; \vec{y},0) \gamma_5 \gamma_i S_l(\vec{x},t; \vec{y},0) \gamma_4 \gamma_i \gamma_4 ]$$
  2. Apply the cyclic property to simplify the gamma matrix structure (Unlike the pion case, $\gamma_5 \gamma_i$ do not trivially simplify to identity matrix, you must explicitly perform the spin-color matvec to evaluate the trace):
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ \gamma_5 S_l^\dagger(\vec{x},t; \vec{y},0) (\gamma_5 \gamma_i) S_l(\vec{x},t; \vec{y},0) \gamma_i \gamma_5 ]$$

**Step 4 — Propagators needed**:
Same as the pion case, we need the point-source light quark propagator $S_l(\vec{x},t; \vec{x}_0,0)$.
$$C_\rho(\vec{p}; t,0) \approx \frac{1}{3}\sum_i\sum_{\vec{x}} e^{-i \vec{p} \cdot (\vec{x} - \vec{x}_0)} \text{Tr}[ S_l^\dagger(\vec{x},t; \vec{x}_0,0) \gamma_5\gamma_i S_l(\vec{x},t; \vec{x}_0,0) \gamma_i \gamma_5 ]$$

**Step 5 - Einsum to correlators**
The trace over spin and color indices can be expressed as an einsum operation. Consider a propagator with the data layout `[parity][t][z][y][x][spin_snk][spin_src][color_snk][color_src]`, and a momentum phase $e^{-i \vec{p} \cdot (\vec{x} - \vec{x}_0)}$ with the data layout `[parity][t][z][y][x]`. Here the `parity` index indicates we are using an even-odd preconditioned layout lattice. The trace over spin and color can be expressed as:
```
numpy.einsum('wtzyx,wtzyxjiba,jk,wtzyxklab,li->t', phase, S_l.conj(), gamma_5 @ gamma_i, S_l, gamma_i @ gamma_5)
```
Note we are performing a sum over the spatial indices (`w`, `z`, `y`, `x`) and the spin/color indices (`i`, `j`, `a`, `b`). The resulting array has only the time index `t`, which is what we need for the correlator. The conjugate of the propagator is taken care of by the `conj()` function, and the transpose is applied with the reversed order of `ij` and `ab` indices.

---

### Example 3: Nucleon mass (proton)

**Goal**: Extract $m_p$

**Step 1 — Operator**:
  $$\mathcal{O}_{p} = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) u^c$$

**Step 2 — Correlator**: We also need a positive parity projector to isolate the ground state nucleon:
  $$C_p(\vec{p}; t,0) = \mathrm{Tr} [P^+ \langle \mathcal{O}_{p}(\vec{p},t) \mathcal{O}^\dagger_{p}(\vec{p},0) \rangle],\;P^+ = \frac{1 + \gamma_4}{2}$$

**Step 3a — Quark fields**: Expand the operator in terms of quark fields and Fourier transform, and we have to write out all spin indices:
  $$C_p(\vec{p}; t,0) = P^+_{\gamma''\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} u^a_\alpha(\vec{x},t) (C\gamma_5)_{\alpha\beta} d^b_\beta(\vec{x},t) u^c_\gamma (\vec{x},t) \epsilon^{a'b'c'} \bar{u}^{c'}_{\gamma'}(\vec{y},0) (\gamma_4)_{\gamma'\gamma''} \bar{d}^{b'}_{\beta'}(\vec{y},0) (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} \bar{u}^{a'}_{\alpha'}(\vec{y},0)$$

**Step 3b — Wick contraction**:
Now we have two contraction paths:

  $$C_p(\vec{p}; t,0) = P^+_{\gamma''\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} \\
  [ S_{u\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4)_{\gamma'\gamma''} (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} \\
  -S_{u\alpha\gamma'}^{ac'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\alpha'}^{ca'} (\vec{x},t;\vec{y},0) (\gamma_4)_{\gamma'\beta'} (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} ]$$

**Step 3c — Simplification**:
  1. Swap color indices to make cancel all the minus signs from fermion anticommutation and the epsilon tensors:
  $$C_p(\vec{p}; t,0) = P^+_{\gamma''\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} \\
  [ S_{u\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4)_{\gamma'\gamma''} (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} \\
  + S_{u\alpha\gamma'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\alpha'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4)_{\gamma'\beta'} (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} ]$$
  2. Apply the cyclic property to simplify the gamma matrix structure:
  $$C_p(\vec{p}; t,0) = \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta'\alpha'} P^+_{\gamma'\gamma} \\
  [ S_{l\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) S_{l\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{l\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) \\
  + S_{l\alpha\gamma'}^{aa'}(\vec{x},t;\vec{y},0) S_{l\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{l\gamma\alpha'}^{cc'} (\vec{x},t;\vec{y},0) ]$$

**Step 4 — Propagators needed**:
Same as the pion case, we need the point-source light quark propagator $S_l(\vec{x},t; \vec{x}_0,0)$.
  $$C_p(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot (\vec{x} - \vec{x}_0)} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta'\alpha'} P^+_{\gamma'\gamma} \\
  [ S_{l\alpha\alpha'}^{aa'}(\vec{x},t;\vec{x}_0,0) S_{l\beta\beta'}^{bb'}(\vec{x},t;\vec{x}_0,0) S_{l\gamma\gamma'}^{cc'} (\vec{x},t;\vec{x}_0,0) \\
  + S_{l\alpha\gamma'}^{aa'}(\vec{x},t;\vec{x}_0,0) S_{l\beta\beta'}^{bb'}(\vec{x},t;\vec{x}_0,0) S_{l\gamma\alpha'}^{cc'} (\vec{x},t;\vec{x}_0,0) ]$$

**Step 5 - Einsum to correlators**
The trace over spin and color indices can be expressed as an einsum operation. Consider a propagator with the data layout `[parity][t][z][y][x][spin_snk][spin_src][color_snk][color_src]`, and a momentum phase $e^{-i \vec{p} \cdot (\vec{x} - \vec{x}_0)}$ with the data layout `[parity][t][z][y][x]`. Here the `parity` index indicates we are using an even-odd preconditioned layout lattice. The trace over spin and color can be expressed as:
```
numpy.einsum('wtzyx,abc,def,ij,kl,mn,wtzyxikad,wtzyxjlbe,wtzyxnmcf,li->t', phase, epsilon, epsilon, C @ gamma_5, C @ gamma_5, P_plus, S_l, S_l, S_l) + numpy.einsum('wtzyx,abc,def,ij,kl,mn,wtzyximad,wtzyxjlbe,wtzyxnkcf,li->t', phase, epsilon, epsilon, C @ gamma_5, C @ gamma_5, P_plus, S_l, S_l, S_l)
```
Note we are performing a sum over the spatial indices (`w`, `z`, `y`, `x`) and the spin/color indices (`i`, `j`, `a`, `b`). The resulting array has only the time index `t`, which is what we need for the correlator. Note this expression might have an extra minus compared to the formula in step 4 depending on the transpose property of the gamma matrices.

Since the einsum subscripts are quite complex in the baryon case, it is recommended to break down the expression into smaller pieces to reduce computational cost.

---

## Decision rules for source strategy

| Situation                        | Recommendation                     |
|----------------------------------|------------------------------------|
| Quick first look / debugging     | Point source, 1 per config         |
| Production meson spectroscopy    | Smeared source, 4 sources/config   |
| Baryon spectroscopy              | Smeared source essential           |
| Disconnected diagrams needed     | Stochastic volume sources (Z₂/Z₄) |
| Form factor / 3pt function       | Sequential source or stochastic    |

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

3. **Periodic vs anti-periodic BC**: Fermions typically use anti-periodic
   temporal boundary conditions, which affects the spectral decomposition:
     C(t) ∝ e^{-mt} - e^{-m(T-t)}  (anti-periodic, meson)
   rather than cosh. This matters for fitting — see S3 (Analysis Pipeline).
