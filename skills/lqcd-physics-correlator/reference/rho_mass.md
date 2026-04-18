## Example: Rho meson mass (ρ⁺ channel)

**Goal**: Extract $m_\rho$. We need to calculate the two-point correlation function of $\rho^+$.

**Step 1 — Operator**:
  $$\mathcal{O}_{\rho^+_i} = \bar{d} \gamma_i u \quad (i = 1, 2, 3 \text{ for the three polarizations})$$
We have the corresponding Dirac conjugate operator (creation operator):
  $$\mathcal{O}_{\rho^+_i}^\dagger = \bar{u} \gamma_4\gamma_i^\dagger\gamma_4 d \quad (i = 1, 2, 3 \text{ for the three polarizations})$$

**Step 2 — Correlator**: Average over polarizations for better statistics:
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \langle \mathcal{O}_{\rho^+_i}(\vec{p},t) \mathcal{O}^\dagger_{\rho^+_i}(\vec{p},0) \rangle$$

**Step 3a — Quark fields**: Expand the operator in terms of quark fields and Fourier transform:
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \langle \bar{d}(\vec{x},t) \gamma_i u(\vec{x},t) \bar{u}(\vec{y},0) \gamma_4 \gamma_i^\dagger \gamma_4 d(\vec{y},0) \rangle$$

**Step 3b — Wick contraction**:
Same topology as the pion, just replace $\gamma_5 \to \gamma_i$:

  $$C_\rho(\vec{p}; t,0) = -\frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_d(\vec{y},0; \vec{x},t) \gamma_i S_u(\vec{x},t; \vec{y},0) \gamma_4 \gamma_i^\dagger \gamma_4 ]$$

**Step 3c — Simplification**:
  1. Apply the $\gamma_5$-hermiticity and the flavor symmetry:
  $$C_\rho(\vec{p}; t,0) = -\frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ \gamma_5 S_l^\dagger(\vec{x},t; \vec{y},0) \gamma_5 \gamma_i S_l(\vec{x},t; \vec{y},0) \gamma_4 \gamma_i^\dagger \gamma_4 ]$$
  2. Apply the cyclic property to simplify the gamma matrix structure. Unlike the pion case, $\gamma_5 \gamma_i$ does not reduce to the identity — you must explicitly carry the spin-color matrix through the trace:
  $$C_\rho(\vec{p}; t,0) = \frac{1}{3} \sum_i \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_l^\dagger(\vec{x},t; \vec{y},0) (\gamma_5 \gamma_i) S_l(\vec{x},t; \vec{y},0) (\gamma_i \gamma_5) ]$$

**Step 4 — Propagators needed**:
As in the pion case, the full source sum is intractable, so we estimate the correlator with point or wall source propagators.

For point source propagator
$$C_\rho(\vec{p}; t,0) \approx \frac{1}{3}\sum_i\sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{point}(\vec{x}_0,0)}^\dagger(\vec{x},t) (\gamma_5 \gamma_i) S_{l,\text{point}(\vec{x}_0,0)}(\vec{x},t) (\gamma_i \gamma_5) ]$$

For wall source propagator
$$C_\rho(\vec{p}; t,0) \approx \frac{1}{3}\sum_i\sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{wall}(-\vec{p}_2,0)}^\dagger(\vec{x},t) (\gamma_5 \gamma_i) S_{l,\text{wall}(\vec{p}_1,0)}(\vec{x},t) (\gamma_i \gamma_5) ]$$

**Step 5 — Einsum** (see layout conventions above):

For point source propagator
```python
from opt_einsum import contract

twopt = 0
for gamma_i in [gamma_1, gamma_2, gamma_3]:
  twopt += contract('wtzyx,wtzyxjiba,jk,wtzyxklba,li->t', phase, S_l.conj(), gamma_5 @ gamma_i, S_l, gamma_i @ gamma_5)
twopt /= 3
```

For wall source propagator
```python
from opt_einsum import contract

twopt = 0
for gamma_i in [gamma_1, gamma_2, gamma_3]:
  twopt += contract('wtzyx,wtzyxjiba,jk,wtzyxklba,li->t', phase, S_l_np2.conj(), gamma_5 @ gamma_i, S_l_p1, gamma_i @ gamma_5)
twopt /= 3
```
