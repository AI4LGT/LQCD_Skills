# Theory — Λ→p three-point function

## Step 1: Operator definitions

The interpolating operators for Λ and proton:

$$ \mathcal{O}_{\Lambda} = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) s^c $$
$$ \mathcal{O}_{p} = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) u^c $$

Relevant weak currents (vector + axial):

$$ J_V^\mu = \bar{u} \gamma_\mu s,\qquad J_A^\mu = \bar{u} \gamma_\mu \gamma_5 s $$

Or combined into the charged weak current:

$$ J_W^\mu = \bar{u} \gamma_\mu (1 - \gamma_5) s = J_V^\mu - J_A^\mu $$

## Step 2: Three-point correlator

For vector or axial insertion, define:

$$ C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) = \mathrm{Tr}\!\left[T \left\langle \mathcal{O}_{p}(\vec{p}_f,t_f) J_\Gamma^\mu(\vec{q},\tau) \bar{\mathcal{O}}_{\Lambda}(\vec{p}_i,0) \right\rangle \right] $$

where:
- $J_\Gamma^\mu = \bar{u}\gamma_\mu \Gamma s$, $\Gamma = 1$ (vector) or $\Gamma = \gamma_5$ (axial)
- $T = \Gamma_A,\ A=0,\dots,15$, $\Gamma_A \in \{1,\gamma_\mu,\sigma_{\mu\nu},\gamma_\mu\gamma_5,\gamma_5\}$
- $\vec{q} = \vec{p}_i - \vec{p}_f$

## Step 3a: Quark field expansion

Fix the source at the origin and absorb the adjoint $\gamma_4$ factors into the source spin structure:

$$
C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) = \sum_{\vec{x},\vec{z}} e^{-i\vec{p}_f \cdot \vec{x}} e^{+i\vec{q}\cdot\vec{z}}
\epsilon^{abc} \epsilon^{def} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta_1\alpha_1} T_{\lambda\gamma}
\left\langle u^a_\alpha(x) d^b_\beta(x) u^c_\gamma(x)
\bar{u}^r_\rho(z) (\gamma_\mu \Gamma)_{\rho\sigma} s^r_\sigma(z)
\bar{s}^{f}_\lambda(0) \bar{d}^{e}_{\beta_1}(0) \bar{u}^{d}_{\alpha_1}(0) \right\rangle
$$

with $x = (\vec{x}, t_f)$ and $z = (\vec{z}, \tau)$.

## Step 3b: Wick contraction

The $\bar{u}$ in the weak current can attach to either of the two source $u$ quarks, yielding two contraction terms:

$$
\begin{aligned}
C_{3,\Gamma}^\mu &= \sum_{\vec{x},\vec{z}} e^{-i\vec{p}_f \cdot \vec{x}} e^{+i\vec{q}\cdot\vec{z}}
\epsilon^{abc} \epsilon^{def} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta_1\alpha_1} T_{\lambda\gamma} (\gamma_\mu \Gamma)_{\rho\sigma} \\
&\quad\Big[ S_{u\,\alpha\alpha_1}^{ad}(x,0) S_{d\,\beta\beta_1}^{be}(x,0) S_{u\,\gamma\rho}^{cr}(x,z) S_{s\,\sigma\lambda}^{rf}(z,0) \\
&\qquad - S_{u\,\alpha\rho}^{ar}(x,z) S_{d\,\beta\beta_1}^{be}(x,0) S_{u\,\gamma\alpha_1}^{cd}(x,0) S_{s\,\sigma\lambda}^{rf}(z,0) \Big]
\end{aligned}
$$

These are the direct and exchange attachments of the current to the proton sink.

## Step 3c: Simplification and sequential-source construction

**1. Isospin symmetry:** $S_u = S_d = S_l$

**2. Proton sink structure:** Collect into a sequential-source object at fixed $(\vec{p}_f, t_f, T)$

$$
\begin{aligned}
B_{\rho\lambda}^{\,rf}(0) &= e^{-i\vec{p}_f \cdot \vec{x}} \epsilon^{abc} \epsilon^{def}
(C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta_1\alpha_1} T_{\lambda\gamma} \\
&\quad\Big[ S_{u\,\alpha\alpha_1}^{ad}(x,0) S_{d\,\beta\beta_1}^{be}(x,0) \delta^{cr}\delta_{\gamma\rho}
- \delta^{ar}\delta_{\alpha\rho} S_{d\,\beta\beta_1}^{be}(x,0) S_{u\,\gamma\alpha_1}^{cd}(x,0) \Big]
\end{aligned}
$$

**3. Sequential source (two-dagger convention):**

First dagger — construct the sequential source:

$$ \eta^{\text{seq}}(0) = \gamma_5 B^\dagger(0) \gamma_5 $$

Solve the sequential propagator:

$$ D_u\, G^{\text{seq}} = \eta^{\text{seq}} $$

**4. Final three-point function:**

$$ C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) = \sum_{\vec{z}} e^{+i\vec{q}\cdot\vec{z}} \mathrm{Tr}\!\left[ G_u^{\text{seq}}(z,0) \gamma_\mu \Gamma S_s(z,0) \right] $$

In the actual code, the sequential propagator after the second dagger (conjugation/reordering) is denoted `G_l_seq_dag`, which is equivalent to $G_l^{\text{seq}}$ =$G_u^{\text{seq}}$  in the formula under the index convention.

## Step 4: Required propagators

For the Λ→p three-point function, the required propagators are:

- **One forward light propagator** $S_l(x;0)$ — reused for source $u/d$ lines and the proton sink construction
- **One forward strange propagator** $S_s(x;0)$ — from the same source position
- **One light sequential propagator** $G_l^{\text{seq}}$ — one for each fixed $(\vec{p}_f, t_f, \text{sink smearing}, \text{spin projector})$ combination

With a point source, the estimator takes the form:

$$ C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) \approx \sum_{\vec{z}} e^{+i\vec{q}\cdot\vec{z}} \mathrm{Tr}\!\left[ G_{l,\text{seq}(\vec{p}_f,t_f,T)}(\vec{z},\tau;\vec{x}_0,0)\, \gamma_\mu \Gamma\, S_{s,\text{point}(\vec{x}_0,0)}(\vec{z},\tau) \right] $$
