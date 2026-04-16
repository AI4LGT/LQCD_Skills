---
name: lqcd-physics-correlator
description: >
  Lattice QCD physics reasoning skill. Derives the chain from a physics
  observable to a computable correlator: interpolating operator
  construction (mesons, baryons), correlator definition (two-point,
  three-point), Wick contraction with γ₅-hermiticity and flavor
  symmetry, propagator requirements, and einsum expressions for
  contractions. Hands spectral decomposition and fit templates off to
  lqcd-physics-spectrum. Uses DeGrand-Rossi gamma basis (PyQUDA
  convention). Trigger on: hadron masses, decay constants, form
  factors, matrix elements, operator construction, Wick contraction,
  disconnected diagrams, or "what correlators/propagators do I need".
---

# LQCD Physics Reasoning

## Purpose

Given a physics observable (hadron mass, decay constant, form factor, ...),
derive the complete chain:

  **Observable → Operator(s) → Correlator(s) → Wick contraction → Propagator(s) → Einsum(s)**

so that downstream tools (PyQUDA) know exactly what to compute.

This skill stops once the correlator expression, propagator list, and
contraction/einsum structure are fixed. For Euclidean-time dependence,
backward-state structure, and fit templates, hand off to
`lqcd-physics-spectrum`.

## Core workflow

### Step 1: Identify the interpolating operator(s)

For a target hadron with quantum numbers $J^{PC}$ and flavor content, write the interpolating operator. Use Dirac bilinears for mesons, and appropriate diquark-quark structures for baryons.

For example, the simplest interpolating operator for a $\pi^+$ meson is usually written as $\mathcal{O}_{\pi^+} = \bar{d}^a \gamma_5 u^a$, and the simplest operator for a proton is $\mathcal{O}_p = \epsilon^{abc} (u^a C\gamma_5 d^b) u^c$. The simplest local operators are usually sufficient for ground state mass extraction, but interpolating operators can be constructed with gamma matrices and gauge covariant derivatives to access different quantum numbers, excited states, and observables related to hadron structure in general. For example, if we want to compute pion distribution amplitudes, we should use non-local operators with quark fields separated by a Wilson line, e.g. $\mathcal{O}_{\pi^+}(x;z) = \bar{d}^a(0) \gamma_5 W(0,z) u^a(z)$. If we want to compute a matrix element of an axial vector current inserted on the $u$ quark, we should insert the current operator $J_\mu=\bar{u}\gamma_5\gamma_\mu u$ between the source and sink operators in the three-point function.

**Gamma matrices convention**: Use the DeGrand-Rossi basis as the Euclidean Dirac basis, which is the default gamma basis in PyQUDA convention:

$$
\gamma_1 = \begin{pmatrix} 0 & i \sigma_1 \nonumber \\ -i \sigma_1 & 0 \end{pmatrix},\quad
\gamma_2 = \begin{pmatrix} 0 & -i \sigma_2 \\ i \sigma_2 & 0 \end{pmatrix},\quad
\gamma_3 = \begin{pmatrix} 0 & i \sigma_3 \\ -i \sigma_3 & 0 \end{pmatrix},\quad
\gamma_4 = \begin{pmatrix} 0 & I_{2 \times 2}\\ I_{2 \times 2} & 0 \end{pmatrix}, \\
\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4=\begin{pmatrix} I_{2 \times 2} & 0 \\ 0 & -I_{2 \times 2} \end{pmatrix},\quad
C = \gamma_2\gamma_4 = \begin{pmatrix} -i\sigma_2 & 0\\ 0 & i\sigma_2 \end{pmatrix}
$$

Clearly we have $\gamma_\mu^\dagger = \gamma_\mu$ for $\mu = 1,2,3,4$ in the DeGrand-Rossi basis. And we have gamma anticommutation relations $\{\gamma_\mu, \gamma_\nu\} = 2\delta_{\mu\nu}$ and $\{\gamma_5, \gamma_\mu\} = 0$. The charge conjugation matrix $C=\gamma_2\gamma_4$.

### Step 2: Write the correlator

The basic observable in lattice QCD is the correlator of interpolating operators. For a mass extraction, this is a two-point function. For a matrix element or form factor, this is a three-point function with an appropriate current insertion.

A two-point function for a hadron is typically

$$C_2(\vec{p};t_f,t_i) = \langle \mathcal{O}(\vec{p},t_f) \mathcal{O}^\dagger(\vec{p},t_i) \rangle$$

Where $\vec{p}$ is the momentum, and the interpolating operators are projected onto this momentum by Fourier transformation $\mathcal{O}(\vec{p},t) = \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \mathcal{O}(\vec{x},t)$. By utilizing the time translation invariance of the vacuum, we can shift the time and average over equivalent time slices to enhance the signal. The operator for initial and final states can be the same or different depending on the observable (e.g. for a form factor you might have different operators at source and sink).

For a matrix element or form factor, you need a three-point function

$$C_3(\vec{q}; t_f,t_i,\tau) = \langle \mathcal{O}_\text{snk}(\vec{p}_f,t_f) J(\vec{q},\tau) \mathcal{O}^\dagger_\text{src}(\vec{p}_i,t_i) \rangle$$

with an appropriate current insertion $J$. Here, $t_f$ is the sink time, $\tau$ is the current insertion time, and $t_i$ is the source time. (We can shift the time slices to average over equivalent time slices to enhance the signal.) The transfer momentum is calculated by $\vec{q} = \vec{p}_i - \vec{p}_f$.

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

**Wall source**: We can also use a wall source that spans the entire spatial volume at a fixed time slice (e.g., $t=t_0$), set the phase $e^{i\vec{p}\cdot\vec{x}}$ for each spatial point, and compute the propagator from this source to all spatial points at all time slices, namely $S_{l,\text{wall}(\vec{p},t_0)}(\vec{x},t)\equiv\sum_{y}e^{i\vec{p}\cdot\vec{y}}S_l(\vec{x},t;\vec{y},t_0)$. This gives us an estimate of the correlator: $C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{wall}(-\vec{p}_2,t_0)}^{\dagger}(\vec{x},t) S_{l,\text{wall}(\vec{p}_1,t_0)}(\vec{x},t) ]$. Here we apply the similar momentum splitting strategy just like the point source case. But we cannot ignore the phase in the wall source, so we really need to calculate both $S_{l,\text{wall}(-\vec{p}_2,t_0)}$ and $S_{l,\text{wall}(-\vec{p}_2,t_0)}$. Assuming that we only have a momentum in the $z$ direction, and we have $\vec{p}_i=(0,0,p_z)$, and the best strategy to split the momentum is basically equally splitting, i.e.
1. If $p_z=1$, choose $\vec{p}_1=(0,0,1)$, $\vec{p}_2=(0,0,0)$;
2. If $p_z=2$, choose $\vec{p}_1=(0,0,1)$, $\vec{p}_2=(0,0,1)$;
3. If $p_z=3$, choose $\vec{p}_1=(0,0,2)$, $\vec{p}_2=(0,0,1)$;
4. If $p_z=4$, choose $\vec{p}_1=(0,0,2)$, $\vec{p}_2=(0,0,2)$;

and so on. Finally, for better statistics, we can also use multiple wall sources at different time slices, and average the resulting correlators. 

**Volume source**: Generally we do not use volume sources for two-point functions, but they can be used for all-to-all propagator with stochastic estimation. A volume source is defined as $S_{l,\text{volume}(\vec{p},E)}(\vec{x},t)\equiv\sum_{\vec{y},\tau}e^{i(\vec{p}\cdot\vec{y}+E\tau)}S(\vec{x},t;\vec{y},\tau)$, which have the 4-momentum phase at each spatial point and time slice.

**Shifted propagator**: When using a non-local operator like $\mathcal{O}_{\pi^+}(x;z) = \bar{d}^a(x) \gamma_5 W(x,z) u^a(z)$, we will see the final propagator need to be shifted by applying the corresponding Wilson line like $S_{u,W(\vec{z},t)}(\vec{x},t;\vec{y},0)\equiv W(\vec{z},t;\vec{x},t)S_u(\vec{z},t;\vec{y},0)$. This is because the quark field in the operator is located at $\vec{z}$ instead of $\vec{x}$, and the Wilson line connects the two points to make the operator gauge invariant. The same applies to current and baryon operators with non-local structures.

Output a list of propagators specifying:
- Quark flavor / mass parameter
- Source type (point, smeared-point with Gaussian/Wuppertal parameters)
- Source position(s) (time slice, number of sources per configuration)
- Whether APE/HYP smeared links are used for the source construction
- Sink treatment (point, smeared, or both → for SS/SP correlator matrix)

---

## Worked examples

See files in `reference` directory for step-by-step demonstrations of the Wick contraction and propagator determination workflow. The filename indicates the target observable, e.g. `pion_mass.md` for the pion mass extraction example, `rho_mass.md` for the rho meson mass extraction example, and `proton.md` for the proton mass extraction example. Each example follows the same workflow outlined above, with detailed explanations of each step and the resulting expressions for the correlator, propagators needed, and einsum structure. The examples cover a range of observables and hadron types to illustrate the generality of the workflow.

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
   large-t behavior of the channel. If you need the explicit fit template,
   hand off to `lqcd-physics-spectrum`.

3. **Periodic vs anti-periodic BC**: Fermions use anti-periodic temporal
   boundary conditions, but the **composite** state's BC depends on the
   number of quarks:
     Mesons:  (-1)² = +1 → C(t) ∝ e^{-mt} + e^{-m(T-t)}  (cosh-like)
     Baryons: (-1)³ = -1 → backward state has opposite parity
   See `lqcd-physics-spectrum` for the explicit fit-function templates.
