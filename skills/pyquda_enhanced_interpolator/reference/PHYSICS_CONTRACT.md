# Enhanced interpolator and gamma-basis physics contract

## Status and evidence ceiling

- TODO: `SPEC-ENH-01`
- Status: `PASS_OPERATOR_CONTRACT`
- Highest supported evidence: local E1 gamma and Wick algebra.
- Interacting overlap/SNR claim: E0 and explicitly unverified.

The word “enhanced” identifies a paper-motivated operator family. It does not
mean that any array returned by the module has demonstrated better overlap or
signal-to-noise on the caller's ensemble.

## Euclidean gamma basis

The caller supplies one four-dimensional Hermitian Euclidean basis satisfying

$$
\gamma_\mu^\dagger=\gamma_\mu,
\qquad
\{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu}I,
$$

and identifies `gamma_t`, `gamma_parallel`, $\gamma_5$ and charge
conjugation $C$. The parallel direction must equal the hadron boost direction.
With the Euclidean continuation used by the bundled code,

$$
\gamma_\pm=\frac{\gamma_t\pm i\gamma_\parallel}{\sqrt2},
\qquad \gamma_\pm^2=0.
$$

The paper-normalized quark component is

$$
Q_+=\frac{\gamma_-\gamma_+}{\sqrt2}.
$$

With these conventions $Q_+^2=\sqrt2Q_+$; it is a scaled component map, not
an idempotent projector. Code and documentation must preserve this
normalization rather than silently replace it by $\gamma_-\gamma_+/2$.

## Euclidean adjoints and source kernels

Two operations are intentionally distinct:

$$
\overline M=\gamma_tM^\dagger\gamma_t
\quad\text{for a bilinear Dirac bar},
$$

and

$$
M_{\rm source}=\gamma_tM^*\gamma_t
\quad\text{for the two source-spin indices of the diquark kernel}.
$$

The latter has no transpose. Replacing `conj()` by `conj().T` changes the
baryon contraction and is a critical mutation.

## Meson operator

`enhanced_meson_kernel` consumes the matrix $\Gamma$ in $\bar q\Gamma q$ and
returns

$$
K=(\gamma_tQ_L^\dagger\gamma_t)\Gamma Q_R.
$$

The generic function is algebraic; callers must identify a paper-supported
bilinear. For the canonical pion identity in arXiv:2501.00729v2,

$$
u_+^\dagger\gamma_5d_+=\sqrt2\,\bar u\gamma_+\gamma_5d.
$$

Because the API expects a Dirac-bar matrix, the wrapper passes
$\Gamma=\gamma_t\gamma_5$. Passing bare $\gamma_5$ into the generic helper is
identically zero under this Clifford convention and is not the paper's pion
operator.

## Baryon operator

For the proton family in arXiv:2606.02447v2 Eq. (2.6), the sink diquark kernel
is

$$
D_{\rm sink}=C\gamma_5\Gamma,
\qquad
\Gamma\in\{I,\gamma_t,\gamma_+\},
$$

the free-quark kernel is

$$
\mathcal T\in\left\{\frac{I+\gamma_t}{2},\gamma_t,\gamma_+\right\},
$$

and the source kernel is $D_{\rm source}=\gamma_tD_{\rm sink}^*\gamma_t$.
The local `uud` contraction includes both color epsilons and direct-minus-
exchange of the identical $u$ lines. Propagator axes are exactly
`(sink_spin,source_spin,sink_color,source_color)`.

The paper explicitly notes that $\gamma_+$ itself is not an idempotent
projector and derives positivity restrictions for a general spin kernel in
Eqs. (2.9)-(2.11). A caller choosing a kernel outside the implemented paper
set must provide a new operator derivation and cannot inherit the enhanced
claim by name.

## Basis covariance

Under a unitary spin-basis change $S$,

$$
\gamma_\mu\to S\gamma_\mu S^\dagger,
\quad
D\to SDS^T\ \text{for the stated diquark index convention},
$$

and every propagator/source/sink index must be transformed consistently. The
fully contracted scalar is basis invariant only if $C$, $\gamma_5$,
$\gamma_t$, all kernels and all propagator legs use the same transformation.
Transforming gamma matrices alone is not a valid covariance test.

## Momentum projection and MPI

The contraction kernel returns a local density. `momentum_project` consumes a
separately constructed global-coordinate phase with exact checkerboard
`(e,t,z,y,xh)`, $e=2$, and reduces only over ranks sharing the same global
timeslices. It does not construct the phase, propagator, smearing or solver.
CUDA-aware reduction requires direct runtime proof.

## Meaning of “enhanced”

The papers motivate kinematic enhancement and report ensemble-specific gains.
For this skill, a physical enhancement claim requires a correlated comparison
against a conventional operator using identical configurations, propagators,
source count, precision, momentum, fit rule and cost. Report overlap,
effective-mass plateau, SNR and fit stability with uncertainties. Before
`E2E-ENH-01`, outputs are `enhanced_operator_candidate` or
`paper_normalized_kernel`, not “improved signal”.

## Sources

- arXiv:2501.00729v2, PDF pp. 1-2: projected-field pion identity and
  Euclidean $\gamma_\pm$ convention.
- arXiv:2606.02447v2 Secs. 2.1-2.2, Eqs. (2.6), (2.9)-(2.13): nucleon kernels
  and source/sink contractions.
- `scripts/Def_enhanced_interpolator.py`: exact local normalization and axis
  contract.

