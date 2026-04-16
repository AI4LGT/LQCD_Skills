## Example: Pion meson mass (π⁺ channel)

**Goal**: Extract $m_\pi$. We need to calculate the two-point correlation function of $\pi^+$.

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
It's impossible to calculate propagator from all source points, and we can use point or wall source propagators to estimate the correlator.

For point source propagator
$$C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{point}(\vec{x}_0,0)}^\dagger(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)}(\vec{x},t) ]$$

For wall source propagator
$$C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{wall}(-\vec{p}_2,0)}^\dagger(\vec{x},t) S_{l,\text{wall}(\vec{p}_1,0)}(\vec{x},t) ]$$

**Step 5 — Einsum** (see conventions above):
```python
twopt = numpy.einsum('wtzyx,wtzyxjiba,wtzyxijab->t', phase, S_l.conj(), S_l)
```