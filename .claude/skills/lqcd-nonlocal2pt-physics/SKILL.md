---
name: lqcd-nonlocal2pt-physics
description: >
  Lattice QCD physics reasoning skill for Nonlocal Two-Point (2pt) Correlators.
  Use whenever the user wants to calculate quasi-PDFs, pseudo-PDFs, or Distribution Amplitudes (DAs)
  using spatial displacements and Wilson lines. Covers: operator definition, nonlocal correlator formulation, Wick contraction, and the equivalence of using gauge covariant shifts (covDev) on propagators.
---

# LQCD Nonlocal 2pt Physics Reasoning

## Purpose
Given a mathematical definition of a nonlocal two-point correlator, the agent must derive the computational steps required to evaluate it on the lattice:
**Given Correlator → Wick contraction → Gauge-shifted Propagator(s) → Einsum(s)**
This allows the agent to seamlessly translate the mathematical formulation into an executable PyQUDA Python script involving spatial displacement, Wilson lines, and momentum phases.

## Core workflow

### Step 1: Analyze the Given Operators and Correlator
When presented with a nonlocal correlator formula (e.g., for quasi-PDFs or quasi-DA), first identify the source and sink interpolating operators. The target usually involves a spatial displacement $z$ (typically along $\hat{z}$) accompanied by a gauge link connecting the quark fields.

The typical local interpolating operator at the source (momentum $\vec{p}$, time $t=0$, position $y=0$ usually) is:
$$ \mathcal{O}_\text{src}(\vec{p}_{f}, t_i) = \sum_{\vec{y}} e^{-i\vec{p}_{f}\cdot\vec{y}}\mathcal{O}(\vec y,t) = \sum_{\vec{y}} e^{-i\vec{p}_{f}\cdot\vec{y}} \bar{q}_1(\vec{y},t_i) \Gamma_\text{src} q_2(\vec{y},t_i) $$

And a typical nonlocal operator at the sink (momentum $\vec{p}$, time $t$, displacement vector $\vec{z}$) is:
$$ \mathcal{O}_\text{snk}(\vec{p}_{i}, \vec{z}, t_f) =\sum_{\vec{x}} e^{-i\vec{p}_{i}\cdot\vec{x}} \left[ \bar{q}_2(\vec{x}+\vec{z}, t_f) \Gamma_\text{snk} W(\vec{x}+\vec{z},t_f;\vec{x},t_f) q_1(\vec{x}, t_f) \right] $$
Where $W(\vec{x}+\vec{z},t_f;\vec{x},t_f) = \prod_{i=0}^{z/\text{spacing}-1} U_{\hat{z}}(\vec{x}+i\hat{z}, t_f)$ is the straight Wilson line connecting $\vec{x}$ and $\vec{x}+\vec{z}$. More complicated off-axis displacements can also be used depending on the kinematic constraints.

**Convention**: Same as local 2pt, we use the DeGrand-Rossi basis for the $\gamma$ matrices. The correspondence between gamma matrices is as follows:

$$
\begin{array}{cccc}\text{ Chroma/PyQUDA }&\text{ binary-system }&\mathrm{~Euclidean~}&\mathrm{~Minkowski~}\\\mathrm{~Gamma~}(0)&0000&(I_{4})_E&(I_{4})_M\\\mathrm{~Gamma~}(1)&0001&(\gamma^{x})_E&i(\gamma^{1})_M\\\mathrm{~Gamma~}(2)&0010&(\gamma^{y})_E&-i(\gamma^{2})_M\\\mathrm{~Gamma~}(3)&0011&(\gamma^{x}\gamma^{y})_E&(\gamma^{1}\gamma^{2})_M\\\mathrm{~Gamma~}(4)&0100&(\gamma^{z})_E&i(\gamma^{3})_M\\\mathrm{~Gamma~}(5)&0101&(\gamma^{x}\gamma^{z})_E&-(\gamma^{1}\gamma^{3})_M\\\mathrm{~Gamma~}(6)&0110&(\gamma^{y}\gamma^{z})_E&(\gamma^{2}\gamma^{3})_M\\\mathrm{~Gamma~}(7)&0111&(\gamma^{x}\gamma^{y}\gamma^{z})_E=(\gamma^{5}\gamma^{t})_E&i(\gamma^{1}\gamma^{2}\gamma^{3})_M=-(\gamma^{5}\gamma^{0})_M\\\mathrm{~Gamma~}(8)&1000&(\gamma^{t})_E&-(\gamma^{0})_M\\\mathrm{~Gamma~}(9)&1001&(\gamma^{x}\gamma^{t})_E&-i(\gamma^{1}\gamma^{0})_M\\\mathrm{~Gamma~}(10)&1010&(\gamma^{y}\gamma^{t})_E&i(\gamma^{2}\gamma^{0})_M\\\mathrm{~Gamma~}(11)&1011&(\gamma^{x}\gamma^{y}\gamma^{t})_E=(\gamma^{z}\gamma^{5})_E&-(\gamma^{1}\gamma^{2}\gamma^{0})_M=i(\gamma^{3}\gamma^{5})_M\\\mathrm{~Gamma~}(12)&1100&(\gamma^{z}\gamma^{t})_E&-i(\gamma^{3}\gamma^{0})_M\\\mathrm{~Gamma~}(13)&1101&(\gamma^{x}\gamma^{z}\gamma^{t})_E=-(\gamma^{y}\gamma^{5})_E&(\gamma^{1}\gamma^{3}\gamma^{0})_M=i(\gamma^{2}\gamma^{5})_M\\\mathrm{~Gamma~}(14)&1110&(\gamma^{y}\gamma^{z}\gamma^{t})_E=(\gamma^{x}\gamma^{5})_E&-(\gamma^{2}\gamma^{3}\gamma^{0})_M=i(\gamma^{1}\gamma^{5})_M\\\mathrm{~Gamma~}(15)&1111&(\gamma^{x}\gamma^{y}\gamma^{z}\gamma^{t})_E=(\gamma^{5})_E&i(\gamma^{0}\gamma^{1}\gamma^{2}\gamma^{3})_M=(\gamma^{5})_M\end{array}
$$

### Step 2: Deconstruct the Correlator
Identify the momentum projection and the exact structure of the given nonlocal constituent two-point correlator:
$$
\begin{aligned}
 C_2(\vec{p}_f, \vec{p}_i, t, \vec{z}) &= \langle \mathcal{O}_\text{snk}(\vec{p}_{f}, t_f, \vec{z}) \mathcal{O}^\dagger_\text{src}(\vec{p}_{i}, t_i) \rangle \\
 &= \sum_{\vec{x}, \vec{y}} e^{-i\vec{p}_{f}\cdot\vec{x}} e^{i\vec{p}_{i}\cdot\vec{y}} \langle \mathcal{O}_\text{snk}(\vec{x}, \vec{z}, t_f) \mathcal{O}^\dagger_\text{src}(\vec{y}, t_i) \rangle \\
 &= \sum_{\vec{x}, \vec{y}} e^{-i\vec{p}_f\cdot\vec{x}} e^{i\vec{p}_i\cdot\vec{y}} \langle \left[ \bar{q}_2(\vec{x}, t_f) \Gamma_\text{snk} W(\vec{x},\vec{x}+\vec{z}) q_1(\vec{x}+\vec{z}, t_f) \right] \left[ \bar{q}_1(\vec{y},t_i) \Gamma_\text{src} q_2(\vec{y},t_i) \right]^\dagger \rangle
\end{aligned}
$$
The goal is to express this correlator in terms of quark propagators and gauge links, which can then be computed on the lattice.

### Step 3: Wick contraction
Expand the correlator by contracting all quark-antiquark pairs into propagators $S(\vec{x},t; \vec{y},t_i)$. First, take the Hermitian adjoint of the source operator. As Grassmann variables anti-commute, swapping the fermionic operators during the dagger operation introduces an overall **negative sign ($-1$)**:
$$ \left[ \bar{q}_1(\vec{y},t_i) \Gamma_\text{src} q_2(\vec{y},t_i) \right]^\dagger = - \bar{q}_2(\vec{y},t_i) \gamma_4 \Gamma_\text{src}^\dagger \gamma_4 q_1(\vec{y},t_i) \equiv - \bar{q}_2(\vec{y},t_i) \bar{\Gamma}_\text{src} q_1(\vec{y},t_i) $$

Now perform the contractions. The Grassmann trace grouping preserves this negative sign:
$$
\begin{aligned}
 C_2(\vec{p}_f, \vec{p}_i, t, \vec{z}) &= \sum_{\vec{x}, \vec{y}} e^{-i\vec{p}_{f}\cdot\vec{x}} e^{i\vec{p}_{i}\cdot\vec{y}} \langle \bar{q}_2(\vec{x}) \Gamma_\text{snk} W q_1(\vec{x}+\vec{z}) \left( - \bar{q}_2(\vec{y}) \bar{\Gamma}_\text{src} q_1(\vec{y}) \right) \rangle \\
 &= - \sum_{\vec{x}, \vec{y}} e^{-i\vec{p}_{f}\cdot\vec{x}} e^{i\vec{p}_{i}\cdot\vec{y}} \text{Tr} \left[ S_2(\vec{y},t_i; \vec{x},t_f) \Gamma_\text{snk} W(\vec{x}, \vec{x}+\vec{z}) S_1(\vec{x}+\vec{z},t_f; \vec{y},t_i) \bar{\Gamma}_\text{src} \right] 
\end{aligned}
$$

When using extended momentum sources (such as Momentum Wall sources or Smeared sources), the phase $e^{i\vec{p}_{i}\cdot\vec{y}}$ and the summation over $\vec{y}$ are handled implicitly during the Dirac inversion (e.g., quarks are inverted with specific momentum shifts $\vec{p}_1$ and $\vec{p}_2$ such that $\vec{p}_i = \vec{p}_1 + \vec{p}_2$). Thus, computing the correlator simplifies to explicitly applying the sink momentum phase $\vec{p}_f$ (which does not necessarily equal $\vec{p}_i$ depending on the initial conditions of the wall sources) over the sink coordinate $\vec{x}$. Abstracting the generalized source summation as `src`, we evaluate at $t_f = t$:
$$ C_2(\vec{p}_f, t, \vec{z}) = -\sum_{\vec{x}} e^{-i \vec{p}_f \cdot \vec{x}} \text{Tr} \left[ S_2(\text{src}; \vec{x},t) \Gamma_\text{snk} W(\vec{x}, \vec{x}+\vec{z}) S_1(\vec{x}+\vec{z},t; \text{src}) \bar{\Gamma}_\text{src} \right] $$

Applying $\gamma_5$-hermiticity to flip the backward propagator $S_2(\text{src}; \vec{x},t) = \gamma_5 S_2^\dagger(\vec{x},t; \text{src}) \gamma_5$:
$$ C_2(\vec{p}_f, t, \vec{z}) = -\sum_{\vec{x}} e^{-i \vec{p}_f \cdot \vec{x}} \text{Tr} \left[ \gamma_5 S_2^\dagger(\vec{x},t; \text{src}) \gamma_5 \Gamma_\text{snk} W(\vec{x}, \vec{x}+\vec{z}) S_1(\vec{x}+\vec{z},t; \text{src}) \bar{\Gamma}_\text{src} \right] $$

<!-- Using cyclic rules of the trace to group terms for lattice contraction, keeping the negative sign from the previous step:
$$ C_2(\vec{p}_f, t, \vec{z}) = - \sum_{\vec{x}} e^{-i \vec{p}_f \cdot \vec{x}} \text{Tr} \left[ S_2^\dagger(\vec{x},t; \text{src}) \left(\gamma_5 \Gamma_\text{snk}\right) \Big( W(\vec{x}, \vec{x}+\vec{z}) S_1(\vec{x}+\vec{z},t; \text{src}) \Big) \left(\bar{\Gamma}_\text{src} \gamma_5\right) \right] $$ -->

This leaves four distinct blocks that must be computed by the generated script on the lattice:
1. $\gamma_5 S_2^\dagger(\vec{x},t; \text{src}) \gamma_5$: The complex conjugate of the local propagator (Sink to Source).
2. $\Gamma_\text{snk}$: Sink spin-projection.
3. $W(\vec{x}, \vec{x}+\vec{z}) S_1(\vec{x}+\vec{z},t; \text{src})$: The gauge-shifted forward propagator!
4. $\bar{\Gamma}_\text{src}$: Source spin-projection.

### Step 4: The "Shifted Propagator" Equivalence
Calculating $W(\vec{x},\vec{x}+\vec{z})$ explicitly and then performing generic matrix multiplication is computationally demanding. Alternatively, lattice software like PyQUDA provides covariant derivative functions `covDev`.
Define a gauge-shifted fermion field $\psi_{\vec{z}}(\vec{x})$:
$$ \psi_{\vec{z}}(\vec{x}) = W(\vec{x}, \vec{x}+\vec{z}) S_1(\vec{x}+\vec{z},t; \text{src}) $$
The PyQUDA script computes this iteratively. If $\vec{z} = z \hat{\mu}$:
1. Base case $z=0$: $\psi_0(\vec{x}) = S_1(\vec{x},t; \text{src})$
2. Iterative shift: $\psi_{n+1}(\vec{x}) = U_\mu(\vec{x}) \psi_n(\vec{x}+\hat{\mu})$

The operation $U_\mu(\vec{x}) \psi_n(\vec{x}+\hat{\mu})$ is literally the action of `covDev(psi, mu)` from the pure gauge field. 
By applying `covDev` $Z_{max}$ times on the propagator, the agent seamlessly and intrinsically generates code for the Wilson line block without ever manifesting $W(\vec{x}, \vec{x}+\vec{z})$ globally! This enables calculating the $z$-dependent correlator inside a simple `z` loop dynamically.

### Step 5: Determine Einsum to correlators
Assume PyQUDA layout for shifted source-to-sink propagator `propag_shift` and ordinary sink-to-source `propag`: `[parity][t][z][y][x][spin_snk][spin_src][color_snk][color_src]`.
The script must perform a summation over spatial sites (`w, z, y, x`) and matrix multiplications on spin/color domains (`i, j, m, n, k, l, a, b`):
```python
numpy.opt_einsum.contract(
    "wtzyx, wtzyxjiba, mjk, wtzyxklba, nli -> mnt",
    mom_phase,            # e^{-i \vec{p} \cdot \vec{x}}
    gamma.gamma(15) @ propag_2.data.conj() @ gamma.gamma(15),   #\gamma_5 S_2^\dagger \gamma_5
    gamma_snk_list,       # Gamma_snk
    propag_shift.data,    # W(x, x+z) S_1(x+z; 0)
    gamma_src_list        # \bar{Gamma}_src
)
```

---

## Worked example: Drafting Python Specification from a Correlator

**Goal**: Given a meson DA correlator with injected wall source momenta, $m_{heavy}$ heavy quark. The agent's objective is to convert it into computable components for a PyQUDA script.
**Operator**: $\mathcal{O}_\text{snk}(\vec{x}, z) = \bar{q}(\vec{x}) \gamma_4\gamma_5 W(\vec{x}, \vec{x}+z\hat{z}) q(\vec{x}+z\hat{z})$

**Wick Contraction & Preparation**:
1. Gamma matrices for Source and Sink (e.g., using $\gamma_4\gamma_5$):
   `gamma_src = gamma.gamma(7)`
   `gamma_snk = gamma.gamma(7)`
2. Phase array preparation: The sink projection momentum is $\vec{p}_f$, computed from the kinematics. E.g., `phase = MomentumPhase(latt).getPhase([0, 0, -p_f_z])`.
3. Propagators:
   - For light/heavy quark 1: Solve `propag1 = core.invert(dirac, wall_source1)` (implicitly contains source momentum phase)
   - For light/heavy quark 2: Solve `propag2 = core.invert(dirac, wall_source2)`
4. Iterative $z$ Loop in script:
   - At $z=0$: Contract `propag2.conj()`, `gamma_snk`, `propag1`, `gamma_src`. Keep result.
   - Run `propag1_shift = Shift_prop(propag1_shift, 2)` where $2$ refers to $+z$ direction.
   - At $z=1$: Contract `propag2.conj()`, `gamma_snk`, `propag1_shift`, `gamma_src`. Keep result.
   - Repeat applying `Shift_prop` to the shifted propagator and contracting until $z = z_\text{max}$.

## Common Pitfalls

1. **Phase Sign Conventions**: The agent must be extremely careful about the sign of the phase. Since the analytical Wilson line starts at $\vec{x}$ (the anti-quark coordinate $\bar{q}(\vec{x})$), the Fourier phase $e^{-i\vec{p}\cdot\vec{x}}$ applies to $\vec{x}$. In the generated `opt_einsum`, ensure the sign of `mom_list` matches the analytical expectation. E.g., PyQUDA's momentum phase gives $e^{i p\cdot x}$, so the script must pass `-mom_1` or take its conjugate.
2. **Gauge Field Context for shifts**: When performing `covDev()`, the script relies on the pure gauge field object `gauge.pure_gauge`. The gauge field MUST NOT be the stout-smeared one if the intent is to measure the Wilson line using the unsmeared (or original) fields. The script typically inverts the Dirac operator using `gauge_stout` but computers the `Shift_prop` using the original `gauge` field to maintain proper renormalization matching properties.
3. **Array Wraparounds**: As $z$ increases, $x+z$ will wrap around the lattice boundaries. PyQUDA's `covDev` handles internal boundaries matching natively through QIO/MPI communications if set up correctly. The agent should ensure the code handles topological wrapping correctly if $z_\text{max} \sim L_z/2$.
