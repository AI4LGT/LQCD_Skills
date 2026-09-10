# Blending estimator physics contract

## Status and evidence ceiling

- TODO: `SPEC-BLD-01`
- Status: `PASS_ESTIMATOR_SPEC_IMPLEMENTATION_PARTIAL`
- Highest supported evidence: E1 for weights, elementals and synthetic
  contraction limits; runtime/physical estimator remains E0.

This document defines the complete estimator that a production workflow must
implement. The bundled script implements only selected algebraic layers and
must not be described as a complete unbiased all-to-all pipeline.

## Spaces, random basis and normalization

On each time slice let

$$
\mathcal L=\mathbb C^{N_cV_3},\qquad
D=N_cV_3=3V_3.
$$

The exact low-mode space $\mathcal L_1$ is spanned by $N_{\rm ev}$ orthonormal
gauge-covariant Laplacian eigenvectors $v_i$. Its orthogonal complement is

$$
\mathcal L_2=(I-P_1)\mathcal L,\qquad
P_1=\sum_{i=1}^{N_{\rm ev}}v_iv_i^\dagger,qquad
D_2=D-N_{\rm ev}.
$$

A noise realization consists of $N_{\rm st}\le D_2$ mutually orthonormal
vectors $\eta_r$ sampled isotropically in $\mathcal L_2$. Noise at distinct
time slices/configurations must follow the declared independence policy. The
combined basis is $\phi=(v_1,\ldots,v_{N_{\rm ev}},\eta_1,\ldots,
\eta_{N_{\rm st}})$.

The array kernels consume an already constructed array `(N,t,z,y,x,3)`. They
do not prove orthogonality, isotropy, low-mode projection, normalization or
seed independence. The scheduler-free E2E adapter requires hash-bound LapH
and perambulator producer metadata, but that declaration remains provenance
validation rather than a numerical proof of these properties.

## Reweighting estimator

Following arXiv:2505.01719v2, define

$$
\omega_n=\frac{D_2-n}{N_{\rm st}-n},\qquad 0\le n<N_{\rm st}.
$$

For a mode tuple, let $r$ be the number of distinct stochastic labels in that
tuple. The bundled weight is

$$
\Omega=\prod_{n=0}^{r-1}\omega_n.
$$

Thus two equal stochastic labels contribute one factor, while two distinct
labels contribute $\omega_0\omega_1$. This implements the paper's two-index
and three-index cases, including the double-counting correction. The full
unbiased statement is an expectation over the declared noise ensemble, e.g.

$$
\mathbb E_\eta\left[
\sum_{ij}\Omega_{ij}
|\phi_i\rangle|\phi_j\rangle
\langle\phi_i|\langle\phi_j|
\right]=I\otimes I,
$$

under the paper's orthonormal/isotropic sampling assumptions. A deterministic
weight test alone does not prove this expectation.

## Required dilution extension

If dilution is used, projectors $P^{[d]}$ must satisfy

$$
P^{[d]}P^{[d']}=\delta_{dd'}P^{[d]},\qquad
\sum_dP^{[d]}=I_{\mathcal L_2}.
$$

Every diluted noise source, solve and contraction must retain `(seed,time,
noise_label,dilution_label)` provenance. The array API has no dilution object
and therefore cannot claim a diluted estimator. The E2E metadata validator
checks declared dilution labels and post-run solve coverage, but does not
construct projectors or prove their expectation. Any future numerical adapter
must state whether reweighting is applied before or after dilution and prove
the corresponding expectation without reusing the implementation as its
oracle.

## Limits

- `N_st=0`: only low indices are legal and every weight is exactly one. This
  is standard distillation in the chosen $N_{\rm ev}$ subspace, not an exact
  full-space identity unless $N_{\rm ev}=D$.
- `N_st=D_2` with a complete orthonormal complement: every $\omega_n=1$ and
  the combined basis spans $\mathcal L$; projected all-to-all quantities are
  exact up to solve/numerical error.
- Complete dilution is exact only when the dilution projectors resolve the
  whole stated stochastic subspace and all corresponding solves are included.

These limits are algebraic checks. They do not establish unbiasedness for an
incomplete, incorrectly sampled or correlated ensemble.

## Elementals, perambulators and contractions

The meson and baryon elementals implement, respectively,

$$
\Phi_{ij}(t,\mathbf p)=
\Omega_{ij}\sum_{\mathbf x,c}
\phi_i^*(\mathbf x,t,c)e^{-i\mathbf p\cdot\mathbf x}
\phi_j(\mathbf x,t,c),
$$

and

$$
\Phi_{ijk}(t,\mathbf p)=
\Omega_{ijk}\sum_{\mathbf x,abc}
\epsilon_{abc}\phi_i^a\phi_j^b\phi_k^c
e^{-i\mathbf p\cdot\mathbf x}.
$$

Phases use global coordinates and global $3V_3$. A same-timeslice communicator
must sum spatial pieces exactly once.

`generate_perambulator` orchestrates callbacks for

$$
\tau_{ij}(t_1,t_2)=\phi_i^\dagger(t_1)D^{-1}\phi_j(t_2).
$$

The callbacks own source construction, action, boundary conditions, solver,
true residual, MRHS strategy and sink projection. The current routine does not
batch solves, generate noise, write I/O or define configuration averaging.
Meson/baryon two-point routines implement one local Wick convention, including
$\gamma_5$ backward-line and direct-minus-exchange algebra; that convention
still requires comparison with an independent full workflow.

## Smearing and cross-skill ownership

Blending's internal phased-link helper acts only during basis construction.
Its basis artifact is not a quark propagator line from
`pyquda_momentum_smear`. The two skills may meet only through an explicit
basis-source callback, solve or observable-assembly layer with separate
artifact IDs. No completed smeared propagator may be silently passed as
`eigvecs`.

## Unbiasedness and variance acceptance

A production test must predeclare an exact small-system or complete-
distillation reference $O_{\rm ref}$ and run multiple independent seeds. It
must report

$$
\Delta=\bar O-O_{\rm ref},\qquad
z=\Delta/\sigma_{\bar O},
$$

confidence intervals, seed count, failure rate and variance versus
$N_{\rm st},N_{\rm ev}$ and dilution. A single seed or agreement within an
unlabeled tolerance is not unbiasedness evidence. Setup, solves, contraction,
communication and I/O costs must be separated.

## Sources

- arXiv:2505.01719v2, Eqs. (1)-(4), (31)-(32), (37), and supplemental
  unbiasedness proof Eqs. (12)-(24).
- arXiv:2009.10691v1 Sec. II.A Eq. (4): momentum-phased links in a
  distillation context.
- `scripts/Def_blending.py`: implemented subset and ownership gates.
