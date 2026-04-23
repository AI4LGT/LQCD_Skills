# Einsum contraction code for Lambda to proton

Convention: the sequential propagator after the two-dagger procedure is denoted `G_l_seq_dag`, which is equivalent to $G_l^{\text{seq}}$ in the formulas under the index convention.

## Build sink block B

```python
# B_{rho lambda}^{r c'}(x;0) at fixed (p_f, t_f, T)
B_Lambda_to_proton.data = (
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
```

## First dagger: construct the sequential source

```python
# eta_seq(x) = gamma5 * B^dagger(x;0) * gamma5
eta_seq.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5,
    B_Lambda_to_proton.data.conj(),
    G5,
)
src_seq_l = source.sequential12(eta_seq, tsrc + tseq)
```

## Solve the sequential propagator

```python
dirac_l.loadGauge(gauge_stout)
G_l_seq = core.invertPropagator(dirac_l, src_seq_l)
# D_l G_l^seq = eta_seq
```

## Second dagger: conjugation / reordering

```python
# Convert G_l_seq to the conjugated/reordered layout
# used by the final three-point contraction.
G_l_seq_dag.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5,
    G_l_seq.data.conj(),
    G5,
)
```

## Final contraction

Vector (Γ = I) and axial (Γ = γ₅) insertions:

```python
# C_{3,Gamma}^mu = sum_z e^{+i q.z} Tr[ G_l_seq_dag * gamma_mu * Gamma * S_s ]
three_pt_V = numpy.einsum(
    "wtzyx,wtzyxijba,jk,wtzyxkiab->t",
    phase_q,
    G_l_seq_dag,
    gamma_mu @ I,
    S_s,
)
three_pt_A = numpy.einsum(
    "wtzyx,wtzyxijba,jk,wtzyxkiab->t",
    phase_q,
    G_l_seq_dag,
    gamma_mu @ G5,
    S_s,
)
```

Here `Gamma = I` gives the vector insertion and `Gamma = G5` gives the axial insertion.
