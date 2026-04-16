## Example: Lambda baryon to proton baryon semi-leptonic decay form factor

**Goal**: Extract $f_i(q^2)$ and $g_i(q^2)$. We need to calculate the matrix element of a flavor-changing current $J_\mu=\bar{u}\gamma_5\gamma_\mu s$ during the semi-leptonic process $\Lambda\to pl\bar{\nu}_l$
c
### Step 1: Operator

$$\mathcal{O}_\Lambda = \epsilon^{abc}(u^{Ta}C\gamma_5\, d^{b})s^{c}$$
$$\mathcal{O}_p = \epsilon^{abc}(u^{Ta}C\gamma_5\,d^{b})u^{c}$$

For the flavor-changing vector current:
$$J_V^\mu=\bar{u}\gamma_\mu s$$
For the flavor-changing axial vector current:
$$J_A^\mu=\bar{u}\gamma_5\gamma_\mu s$$

### Step 2: Correlator

$$
C_3^\mu(\vec{p}_f,\vec{q},\vec{p}_i; t_f,\tau,t_i) = \mathrm{Tr}[P^+ \langle \mathcal{O}_p(\vec{p}_f,t_f)J^\mu(\vec{q},\tau)\bar{\mathcal{O}}_\Lambda(\vec{p}_i,t_i) \rangle]
$$
where $\vec{q} = \vec{p}_i - \vec{p}_f$.

### Step 3: Wick contraction and propagator determination

#### Step 3a - Expand the operators in quark fields

Insert the explicit quark fields. For brevity, we suppress some Dirac and colour labels initially:

$$
\begin{aligned}
C_3 T &= -\sum_{\vec{x},\vec{y}} e^{-i\vec{p}_f\cdot\vec{x}_2} e^{-i\vec{q}\cdot\vec{x}} \,
\epsilon^{ijk}\epsilon^{lmn} \\
&\quad \langle \big( u^{i,T}_A(x_2) (C\gamma_5)_{AB} d^{j}_B(x_2) \big) u^{k}_C(x_2) \\
&\qquad \cdot \bar{u}^{o}_E(t_{seq}) (\Gamma^{\mu})_{EF} s^{o}_F(t_{seq}) \\
&\qquad \cdot \big( \bar{u}^{l}_G(0) (C\gamma_5)_{GH} \bar{d}^{m,T}_H(0) \big) \bar{s}^{n}_I(0) \rangle T_{IC}
\end{aligned}
$$

where $\Gamma^{\mu} = \gamma^\mu$ (vector) or $\gamma^\mu\gamma_5$ (axial-vector).

#### Step 3b - Perform the Wick contraction

Contract all quark fields using Wick's theorem. Because the current contains a $\bar{s}u$ pair, the strange quark from the current contracts with the $\bar{s}$ in $\bar{\mathcal{O}}_\Lambda$, and the $u$ quark from the current contracts either with the $u$ in the diquark of $\mathcal{O}_p$ or with the single $u$ in $\mathcal{O}_p$. Two topologically distinct diagrams arise:

- **Diagram 1 (direct)**: the $u$ from the current contracts with the $u$ inside the diquark of the proton. The other $u$ (the single one) contracts with the $\bar{u}$ from $\bar{\mathcal{O}}_\Lambda$.
- **Diagram 2 (exchange)**: the $u$ from the current contracts with the single $u$ of the proton, while the diquark $u$ contracts with the $\bar{u}$ from $\bar{\mathcal{O}}_\Lambda$.

After summing over colour with the epsilon tensors and applying fermion anti-commutation signs, we obtain:

$$
\begin{aligned}
C_3^{\Gamma}T &=-\sum_{\vec{x},\vec{y}} e^{-i\vec{p}_f\cdot\vec{x}_2} e^{-i\vec{q}\cdot\vec{x}} \,
\epsilon_{ijk}\epsilon_{lmn} (C\gamma_5)_{AB} \Gamma_{EF} (C\gamma_5)_{GH} (T)_{IC} \\
&\quad \times \Big[
S^{u,il}_{AG}(x_2,0) S^{d,ko}_{CE}(x_2,x) S^{u,jm}_{BH}(x_2,0) S^{s,on}_{FI}(x,0) \\
&\qquad -
S^{u,io}_{AE}(x_2,x) S^{d,kl}_{CG}(x_2,0) S^{u,jm}_{BH}(x_2,0) S^{s,on}_{FI}(x,0) \Big]
\end{aligned}
$$

$S^u$, $S^d$, $S^s$ are the quark propagators.

#### Step 3c - Simplify using $\gamma_5$-hermiticity and flavour symmetry

We use $\gamma_5$-hermiticity:
$$
S(x,y) = \gamma_5 S^\dagger(y,x) \gamma_5.
$$
For degenerate light quarks ($m_u = m_d$), we have $S^u = S^d = S_l$.

The three-point function can be rewritten as:
$$
C_3^{\Gamma}(t,t_{seq})T = \sum_{\vec{x}} e^{-i\vec{q}\cdot\vec{x}} \,
\operatorname{Tr}\big[ G^{\text{seq}}(x,0) \, \Gamma \, S^{s}(x,0) \big]
$$

where $G^{\text{seq}}$ is a sequential propagator encoding the sink-side light-quark structure and sink projection.

### Step 4: Sequential source method - propagator requirements

Instead of storing all sink-time slices, use the sequential source technique:

1. Compute light and strange propagators from the same source position $(\vec{x}_0,t_i)$:
   - $S_l(x;0)$ for degenerate $u/d$
   - $S_s(x;0)$ for strange

2. Construct the sink-time object $X(x_2,0)$:
$$
X(x_2,0) = \sum_{\vec{x}_2} e^{-i\vec{p}_f\cdot\vec{x}_2} \epsilon_{ijk}\epsilon_{lmn} (C\gamma_5)_{AB} (T)_{ID} (C\gamma_5)_{GH}
\big[ S_l(x_2,0)_{AG}^{il} S_l(x_2,0)_{BH}^{jm} - S^{d,kl}_{CG}(x_2,0) S^{u,jm}_{BH}(x_2,0) \big]
$$

3. Apply hermiticity to define the sequential source:
$$
J^{\text{seq}}(x_2) = \gamma_5 X^\dagger(x_2,0) \gamma_5
$$

4. Invert Dirac operator:
$$
D\,G^{\text{seq}} = J^{\text{seq}} \quad \Rightarrow \quad G^{\text{seq}} = D^{-1}J^{\text{seq}}
$$

5. Contract with strange propagator and current matrix:
$$
C_3^{\Gamma}(t,t_{seq}) = \sum_{\vec{x}} e^{-i\vec{q}\cdot\vec{x}}
\operatorname{Tr}\big[G^{\text{seq},ij}_{AB}(x,0)\,\Gamma_{BC}\,S^{s,ji}_{CA}(x,0)\big]
$$

Required propagators per configuration and source position:

- $S_l$: light ($u/d$), all-to-all from one source
- $S_s$: strange, all-to-all from one source
- $G^{\text{seq}}$: light sequential propagator at fixed sink setup

### Step 5: Einsum implementation (PyQUDA-compatible)

Assume propagator layout:

`[parity][t][z][y][x][spin_snk][spin_src][color_snk][color_src]`

Sequential source construction and 3pt contraction pattern:

```python
# Sequential source construction
X_Lambda_to_proton.data = (
    contract(
        "wtzyx, ijk, lmn, AB, GH, ID, wtzyxDGkl, wtzyxBHjm -> wtzyxIAni",
        mom_phase_final, epsilon, epsilon, Cg5, Cg5, gamma.gamma(T),
        prop_l.data, prop_l.data,
    )
    - contract(
        "wtzyx, ijk, lmn, AB, GH, ID, wtzyxAGil, wtzyxBHjm -> wtzyxIDnk",
        mom_phase_final, epsilon, epsilon, Cg5, Cg5, gamma.gamma(T),
        prop_l.data, prop_l.data,
    )
)

X_Lambda_to_proton.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5,
    X_Lambda_to_proton.data.conj(),
    G5,
)
src_seq_Lambda_to_proton = source.sequential12(X_Lambda_to_proton, (tsrc + tseq))

dirac_l.loadGauge(gauge_stout)
propag_seq_Lambda_to_proton = core.invertPropagator(dirac_l, src_seq_Lambda_to_proton)

three_pt_tmp_V = pycontract.mesonTwoPoint(
    prop_s,
    propag_seq_Lambda_to_proton,
    gamma.Gamma(/Gamma),
    gamma.Gamma(0),
)
three_pt_tmp_A = pycontract.mesonTwoPoint(
    prop_s,
    propag_seq_Lambda_to_proton,
    gamma.Gamma(/Gamma) @ gamma.Gamma(15),
    gamma.Gamma(0),
)
```
