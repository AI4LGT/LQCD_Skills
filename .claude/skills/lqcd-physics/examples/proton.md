## Example 3: Nucleon mass (proton)

**Goal**: Extract $m_p$

**Step 1 — Operator**:
  $$\mathcal{O}_{p} = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) u^c$$

**Step 2 — Correlator**: A positive-parity projector is needed to isolate the ground-state nucleon:
  $$C_p(\vec{p}; t,0) = \mathrm{Tr}\big[P^+ \langle \mathcal{O}_{p}(\vec{p},t) \mathcal{O}^\dagger_{p}(\vec{p},0) \rangle\big],\quad P^+ = \frac{1 + \gamma_4}{2}$$

**Step 3a — Quark fields**: Expand in quark fields and Fourier transform, writing out all spin indices explicitly:
  $$C_p(\vec{p}; t,0) = P^+_{\gamma''\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} u^a_\alpha(\vec{x},t) (C\gamma_5)_{\alpha\beta} d^b_\beta(\vec{x},t) u^c_\gamma (\vec{x},t) \epsilon^{a'b'c'} \bar{u}^{c'}_{\gamma'}(\vec{y},0) (\gamma_4)_{\gamma'\gamma''} \bar{d}^{b'}_{\beta'}(\vec{y},0) (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} \bar{u}^{a'}_{\alpha'}(\vec{y},0)$$

**Step 3b — Wick contraction**:
Two contraction paths contribute, corresponding to the two ways of pairing the sink $u$ quarks with the source $\bar{u}$ quarks:

  $$C_p(\vec{p}; t,0) = P^+_{\gamma''\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} \\
  [ S_{u\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4)_{\gamma'\gamma''} (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} \\
  -S_{u\alpha\gamma'}^{ac'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\alpha'}^{ca'} (\vec{x},t;\vec{y},0) (\gamma_4)_{\gamma'\gamma''} (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} ]$$

**Step 3c — Simplification**:
  1. Relabel dummy color indices ($a' \leftrightarrow c'$) on the second term; the epsilon antisymmetry cancels the minus sign:
  $$C_p(\vec{p}; t,0) = P^+_{\gamma''\gamma} \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} \\
  [ S_{u\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4)_{\gamma'\gamma''} (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} \\
  + S_{u\alpha\gamma'}^{aa'}(\vec{x},t;\vec{y},0) (C\gamma_5)_{\alpha\beta} S_{d\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{u\gamma\alpha'}^{cc'} (\vec{x},t;\vec{y},0) (\gamma_4)_{\gamma'\gamma''} (\gamma_4 \gamma_5 C \gamma_4)_{\beta'\alpha'} ]$$
  2. Apply the flavor symmetry ($S_u = S_d = S_l$) and fold the $\gamma_4$ factors into the parity projector and the diquark structure:
  $$C_p(\vec{p}; t,0) = \sum_{\vec{x}, \vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta'\alpha'} P^+_{\gamma'\gamma} \\
  [ S_{l\alpha\alpha'}^{aa'}(\vec{x},t;\vec{y},0) S_{l\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{l\gamma\gamma'}^{cc'} (\vec{x},t;\vec{y},0) \\
  + S_{l\alpha\gamma'}^{aa'}(\vec{x},t;\vec{y},0) S_{l\beta\beta'}^{bb'}(\vec{x},t;\vec{y},0) S_{l\gamma\alpha'}^{cc'} (\vec{x},t;\vec{y},0) ]$$

**Step 4 — Propagators needed**:
Baryons do not naturally admit wall sources, because a single wall cannot be shared among three quark lines with a well-defined momentum partition. In practice we use point (or smeared-point) source propagators, where all three quark lines originate from the same source point $\vec{x}_0$ at $t = 0$. Only a single light-quark propagator is required — the three quark lines reuse it:

$$C_p(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (\gamma_5 C)_{\beta'\alpha'} P^+_{\gamma'\gamma} \\
  [ S_{l,\text{point}(\vec{x}_0,0)\,\alpha\alpha'}^{aa'}(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)\,\beta\beta'}^{bb'}(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)\,\gamma\gamma'}^{cc'}(\vec{x},t) \\
  + S_{l,\text{point}(\vec{x}_0,0)\,\alpha\gamma'}^{aa'}(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)\,\beta\beta'}^{bb'}(\vec{x},t) S_{l,\text{point}(\vec{x}_0,0)\,\gamma\alpha'}^{cc'}(\vec{x},t) ]$$

**Step 5 — Einsum** (see conventions above):
```python
numpy.einsum('wtzyx,abc,def,ij,kl,mn,wtzyxikad,wtzyxjlbe,wtzyxnmcf,li->t', phase, epsilon, epsilon, C @ gamma_5, C @ gamma_5, P_plus, S_l, S_l, S_l) \
+ numpy.einsum('wtzyx,abc,def,ij,kl,mn,wtzyximad,wtzyxjlbe,wtzyxnkcf,li->t', phase, epsilon, epsilon, C @ gamma_5, C @ gamma_5, P_plus, S_l, S_l, S_l)
```
Note: this expression may carry an extra minus sign depending on the transpose convention of the gamma matrices. The einsum is expensive for baryons — in practice, break it into smaller intermediate contractions to reduce cost.
