## Example: Lambda to proton semi-leptonic form factors

**Goal**: Extract the vector and axial form factors for the decay $\Lambda \to p \ell \bar{\nu}_\ell$. The relevant quark currents are
  $$J_V^\mu = \bar{u} \gamma_\mu s,\qquad J_A^\mu = \bar{u} \gamma_\mu \gamma_5 s,$$
or, equivalently, the charged weak current
  $$J_W^\mu = \bar{u} \gamma_\mu (1 - \gamma_5) s = J_V^\mu - J_A^\mu.$$
In practice, the three-point function below is combined with proton and Lambda two-point functions to isolate the form factors $f_i(q^2)$ and $g_i(q^2)$.

**Step 1 — Operators**:
  $$\mathcal{O}_{\Lambda} = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) s^c,\qquad \mathcal{O}_{p} = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) u^c$$

**Step 2 — Correlator**: For either the vector or axial insertion, define
  $$C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) = \mathrm{Tr}\!\left[P^+ \left\langle \mathcal{O}_{p}(\vec{p}_f,t_f) J_\Gamma^\mu(\vec{q},\tau) \bar{\mathcal{O}}_{\Lambda}(\vec{p}_i,0) \right\rangle \right],\quad J_\Gamma^\mu = \bar{u}\gamma_\mu \Gamma s$$
with
  $$\Gamma = 1 \text{ (vector)},\qquad \Gamma = \gamma_5 \text{ (axial)},\qquad P^+ = \frac{1 + \gamma_4}{2},\qquad \vec{q} = \vec{p}_i - \vec{p}_f.$$
The new ingredient for this process is the three-point function; the required proton and Lambda two-point functions follow the same logic as the proton mass example and a straightforward $\Lambda$ two-point analogue.

**Step 3a — Quark fields**: Fix the source at the origin by translation invariance and absorb the conventional adjoint $\gamma_4$ factors into the source spin structure. The three-point function becomes
  $$C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) = \sum_{\vec{x},\vec{z}} e^{-i\vec{p}_f \cdot \vec{x}} e^{+i\vec{q}\cdot\vec{z}} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta'\alpha'} P^+_{\lambda\gamma} \left\langle u^a_\alpha(x) d^b_\beta(x) u^c_\gamma(x) \bar{u}^r_\rho(z) (\gamma_\mu \Gamma)_{\rho\sigma} s^r_\sigma(z) \bar{s}^{c'}_\lambda(0) \bar{d}^{b'}_{\beta'}(0) \bar{u}^{a'}_{\alpha'}(0) \right\rangle$$
with $x = (\vec{x}, t_f)$ and $z = (\vec{z}, \tau)$.

**Step 3b — Wick contraction**:
Two connected contractions contribute, because the $\bar{u}$ in the weak current can attach to either of the two sink $u$ quarks:

  $$C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) = \sum_{\vec{x},\vec{z}} e^{-i\vec{p}_f \cdot \vec{x}} e^{+i\vec{q}\cdot\vec{z}} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta'\alpha'} P^+_{\lambda\gamma} (\gamma_\mu \Gamma)_{\rho\sigma} \\
  \Big[ S_{u\,\alpha\alpha'}^{aa'}(x,0) S_{d\,\beta\beta'}^{bb'}(x,0) S_{u\,\gamma\rho}^{cr}(x,z) S_{s\,\sigma\lambda}^{rc'}(z,0) \\
  - S_{u\,\alpha\rho}^{ar}(x,z) S_{d\,\beta\beta'}^{bb'}(x,0) S_{u\,\gamma\alpha'}^{ca'}(x,0) S_{s\,\sigma\lambda}^{rc'}(z,0) \Big]$$

The two terms are the direct and exchange attachments of the current to the proton sink.

**Step 3c — Simplification**:
  1. Apply isospin symmetry for the light quarks:
  $$S_u = S_d = S_l.$$
  2. Collect the entire proton sink structure into a sequential-source object at fixed $(\vec{p}_f, t_f, P^+)$:
  $$B_{\rho\lambda}^{\,rc'}(x;0) = e^{-i\vec{p}_f \cdot \vec{x}} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta'\alpha'} P^+_{\lambda\gamma} \Big[ S_{l\,\alpha\alpha'}^{aa'}(x,0) S_{l\,\beta\beta'}^{bb'}(x,0) \delta^{cr}\delta_{\gamma\rho} - \delta^{ar}\delta_{\alpha\rho} S_{l\,\beta\beta'}^{bb'}(x,0) S_{l\,\gamma\alpha'}^{ca'}(x,0) \Big]$$
  3. Use $\gamma_5$-hermiticity to turn this sink object into a light-quark sequential source,
  $$\eta^{\text{seq}}(x) = \gamma_5 B^\dagger(x;0) \gamma_5,$$
and solve
  $$D_l\, G_l^{\text{seq}} = \eta^{\text{seq}}.$$
Then the three-point function can be written in the compact form
  $$C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) = \sum_{\vec{z}} e^{+i\vec{q}\cdot\vec{z}} \mathrm{Tr}\!\left[ G_l^{\text{seq}}(z,0) \gamma_\mu \Gamma S_s(z,0) \right].$$

**Step 4 — Propagators needed**:
As in the proton case, baryon correlators do not naturally admit a simple wall-source treatment. In practice we use point or smeared-point sources, with all three source quark lines originating from the same source location $(\vec{x}_0,0)$.

For the $\Lambda \to p$ three-point function, the required propagators are:

- One forward light propagator $S_l(x;0)$, reused for the source $u/d$ lines and the proton sink construction
- One forward strange propagator $S_s(x;0)$ from the same source position
- One light sequential propagator $G_l^{\text{seq}}$ for each fixed sink momentum $\vec{p}_f$, sink time $t_f$, sink smearing choice, and spin projector

With a point source, the estimator takes the form
  $$C_{3,\Gamma}^\mu(\vec{p}_f,\vec{p}_i; t_f,\tau,0) \approx \sum_{\vec{z}} e^{+i\vec{q}\cdot\vec{z}} \mathrm{Tr}\!\left[ G_{l,\text{seq}(\vec{p}_f,t_f,P^+)}(\vec{z},\tau;\vec{x}_0,0)\, \gamma_\mu \Gamma\, S_{s,\text{point}(\vec{x}_0,0)}(\vec{z},\tau) \right].$$

To extract the full set of form factors, repeat the calculation for the needed current directions $\mu$, sink/projector choices, and momentum combinations, and combine with the proton and Lambda two-point correlators in a standard ratio or simultaneous fit analysis.

**Step 5 — Einsum** (see layout conventions above):
```python
# G_seq_dag denotes the sequential light propagator written with the same
# conjugated spin-color index ordering used in the reference conventions.
threept = numpy.einsum(
    "wtzyx,wtzyxjiba,jk,wtzyxkiab->t",
    phase_q,
    G_seq_dag,
    gamma_mu @ Gamma,
    S_s,
)
```
Here `Gamma = I` gives the vector insertion and `Gamma = gamma_5` gives the axial insertion. As in the proton two-point case, the baryon sink block is expensive and should be assembled through smaller intermediate contractions rather than one giant flat einsum.
