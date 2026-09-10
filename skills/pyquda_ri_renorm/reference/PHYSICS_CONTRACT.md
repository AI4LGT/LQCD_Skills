# RI kinematics, projector and scheme contract

## Status and evidence ceiling

- TODO: `SPEC-RI-01`
- Status: `PASS_FAIL_CLOSED_CUSTOM_SCHEME`
- Highest supported evidence: E1 for tree-level CPU algebra and kinematic
  validation.
- Interacting gauge-fixed NPR, continuum conversion and GPU/MPI remain E0.

This module returns raw tree-normalized factors. It deliberately does not infer
a standard continuum RI/SMOM identity from $\omega$ or a projector-family name.

## Kinematics

The local Euclidean convention is

$$
q=p_{\rm out}-p_{\rm in},\qquad
p_{\rm in}^2=p_{\rm out}^2=\mu^2>0,\qquad
q^2=\omega\mu^2.
$$

Every four-vector uses the same component order as the caller-supplied gamma
list. `momentum_definition` is mandatory and distinguishes, for example,
$ap_\mu=2\pi n_\mu/L_\mu$ from $\sin(ap_\mu)$ or an improved momentum.

- $\omega=0$: exceptional kinematics, `RIprime/MOM` family label and
  `gamma_mu` projector family.
- $\omega=1$: symmetric nonexceptional family.
- other $\omega>0$: generalized nonexceptional family.

These labels are not complete schemes. A complete identity also contains the
gauge, gamma basis, momentum definition, incoming/outgoing sign, $Z_q$
definition and every projector/normalization equation.

## External-leg algebra

Fields end in
`(sink_spin,source_spin,sink_color,source_color)=(4,4,3,3)` and map to compound
$(s,c)$ matrices. The amputated bilinear vertex is

$$
\Lambda_O(p_{\rm out},p_{\rm in})=
S^{-1}(p_{\rm out})G_O(p_{\rm out},p_{\rm in})S^{-1}(p_{\rm in}).
$$

This is supported by arXiv:0901.2599 Eq. (1) and arXiv:2310.00814v2 Eq. (34),
after accounting for each paper's momentum-flow notation. The two propagator
legs are independent inputs; gamma5 Hermiticity is not used to manufacture a
general nonexceptional leg. Exact batch-shape equality forbids silent
broadcasting.

## Tree-normalized projectors

For one channel the implementation computes

$$
\lambda_O=
\frac{\operatorname{Tr}[P_O^\dagger\Lambda_O]}
     {\operatorname{Tr}[P_O^\dagger\Lambda_O^{(0)}]}.
$$

For vector, axial and tensor multiplets it sums all numerators and all
denominators before division; it does not average already normalized
components. This distinction matters if component tree norms differ.

The propagator projector is

$$
Z_q^{\rm prop}(p)=
-\frac{i}{12p^2}\operatorname{Tr}_{s,c}
[(\slashed p\otimes I_c)S^{-1}(p)],
$$

as in arXiv:2310.00814v2 Eq. (32), subject to the caller's Euclidean Fourier
and propagator normalization. The code offers `incoming`, `outgoing`,
arithmetic mean and principal-branch geometric mean. The latter two are
implementation/statistical conventions, not standard scheme definitions.
The geometric mean is exactly $\sqrt{Z_q^{in}Z_q^{out}}$ on the backend's
principal branch; no real part or absolute value is permitted.

For nonexceptional vertices:

- `gamma_mu` uses $\gamma_\mu$ and $\gamma_\mu\gamma_5$ component projectors;
- `qslash` uses $q_\mu\slashed q$ and
  $q_\mu\slashed q\gamma_5$, with tree vertices still supplied in
  $\gamma_\mu$/$\gamma_\mu\gamma_5$ order.

arXiv:0901.2599 Eqs. (9)-(12) show that standard RI/SMOM projector and $Z_q$
definitions are linked by Ward-Takahashi identities. The local custom
combinations have not all been mapped to those exact conditions. Therefore
every `RIConstants` result records

```text
scheme_identity = <kinematics>__Zq_propagator_<leg rule>__custom_unmatched
continuum_matching_status = unmapped_fail_closed
```

and must not dispatch an $\overline{\mathrm{MS}}$ conversion automatically.

## Raw factors and conditional identities

The local factors are

$$
Z_O=\frac{Z_q}{\lambda_O},\qquad
O\in\{V,A,S,P,T\}.
$$

Tree-level identities are algebra tests only. Interacting claims such as
$Z_V=Z_A$, $Z_mZ_P=1$ or Ward identities require their stated chiral limit,
regularization symmetry, current normalization, gauge and exact scheme
conditions. arXiv:0901.2599 explicitly notes that lattice regularizations may
break the bare identities. A failed interacting identity is therefore not
interpreted without checking those assumptions.

## Required interacting provenance

Before using physical NPR data, record gauge-fixing functional/tolerance,
configuration and source IDs, gamma basis, Fourier sign, boundary conditions,
momentum definition, hypercubic orbit, incoming/outgoing convention, quark
mass/chiral treatment, all projector matrices, $Z_q$ rule, covariance and
continuum matching step. Raw RI factors and converted factors must be stored as
separate artifacts.

## Sources

- arXiv:0901.2599: nonexceptional RI/SMOM scheme, Eqs. (1), (5), (9)-(12).
- arXiv:2310.00814v2 Appendix B.1, PDF pp. 13-14, Eqs. (32), (34)-(42):
  propagator $Z_q$, independent-leg amputation and example projectors.
- `scripts/Def_ri_renorm.py`: exact local custom scheme behavior.

