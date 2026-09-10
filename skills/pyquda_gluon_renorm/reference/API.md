# pyquda_gluon_renorm API

Directions are `0=x,1=y,2=z,3=t`; checkerboard field axes begin
`(e,t,z,y,xh)`. These functions construct bare observables unless a separate
renormalization prescription is supplied.
Fourier entrypoints validate exact `e=2` checkerboard geometry, even local
`Lx`, positive extents, and explicit field layout before contraction.
Real coupling/cut parameters must be finite and non-Boolean; selector flags,
Lorentz/link directions, and orientation flags are non-Boolean integers in
their documented finite sets. These checks run before gauge shifts, loops,
FFTs, or momentum-mask allocation.

## Contents

- Gauge coupling and local tensors
- Wilson-line field-strength products
- Fourier transforms
- EMT and gauge-dependent current candidates
- Momentum setup

## pyquda_gluon_renorm.g_0

`g_0(beta, u_0=None, normalization='wilson')` supports `beta=6/g0^2` or the
explicit legacy relation `beta=10/(g0^2*u0^4)`. The Wilson branch requires
`u_0` to be omitted; the legacy branch requires an explicit `u_0`. Therefore
the ambiguous positional call `g_0(beta,u_0)` fails closed instead of silently
discarding `u_0`.

- Returns: CuPy scalar.
- Example: `coupling = g_0(beta,normalization='wilson')`.
- Boundary: the gauge action, not DWF/clover fermions, selects normalization.
- Domain: `beta` and a legacy-branch `u_0` are finite positive real scalars;
  Boolean, complex, NaN, and infinity inputs fail closed before the square
  root.

## pyquda_gluon_renorm.plaq_munu

`plaq_munu(gauge)` constructs oriented plaquettes with PyQUDA link shifts.

- Returns: direction-pair checkerboard color matrices.
- Example: `P = plaq_munu(gauge)`.

## pyquda_gluon_renorm.A_mu

`A_mu(gauge, g_0, half_flag=1)` builds a traceless link-centered gauge
potential. It is gauge dependent.

- Returns: CuPy `(4,e,t,z,y,xh,3,3)`.
- Example: `A = A_mu(gauge,g0)`.
- Domain: `g_0` is finite positive real and `half_flag` is exactly integer 0/1.

## pyquda_gluon_renorm.F_munu

`F_munu(gauge, g_0, save_flag=1)` builds clover field strengths.

- Returns: `(F_ti,F_ij)` triplets when `save_flag=1`, with
  `F_ij=(F01,F02,F12)=(Bz,-By,Bx)` in `0,1,2,3=x,y,z,t`; otherwise full
  antisymmetric `(4,4,e,t,z,y,xh,3,3)`.
- Example: `F = F_munu(gauge,g0,save_flag=0)`.
- Reference: arXiv:hep-lat/0203008 for clover/improved field strength.
- Domain: `g_0` is finite positive real and `save_flag` is exactly integer 0/1.

## pyquda_gluon_renorm.F_and_tildeF

`F_and_tildeF(gauge,g_0,mu1,nu1,mu2,nu2)` applies the `1/2` dual-tensor
normalization.

- Returns: selected `(F,tildeF)` checkerboard fields.
- Example: `Fxt,tFxt = F_and_tildeF(gauge,g0,0,3,0,3)`.
- Domain: every Lorentz index is a non-Boolean integer in `0..3`.

## pyquda_gluon_renorm.FW_tildeFW

`FW_tildeFW(info,gauge,g_0,mu1,nu1,mu2,nu2,link_dir=2,link_length=0)` contracts
a transported `F-W-tildeF-W^dag` product.

- Returns: root host time series; requires `Gt=1`.
- Example: `ct = FW_tildeFW(info,gauge,g0,0,3,0,3,2,8)`.
- Domain: `link_dir` is in `0..3`; `link_length` is a nonnegative
  non-Boolean integer.
- Ordering: link controls, Lorentz indices, coupling, exact equality of explicit
  and `gauge.latt_info` global/local extents, process grid/rank coordinate, and
  `Gt=1` are all validated before `F_munu` can call `gauge.loop`.

## pyquda_gluon_renorm.FW_FW

Same contract for `F-W-F-W^dag`; `forward_flag` selects orientation. For
identical fields, orientation, communicator, and reduction convention, the
bundled normalization obeys `FW_FW(link_length=0) = -0.5 * FF`.

- Returns: root host time series; requires `Gt=1`.
- Example: `ct = FW_FW(info,gauge,g0,0,3,0,3,2,8,1)`.
- Domain: `forward_flag` is exactly integer 0/1; link and Lorentz inputs obey
  the same fail-closed contracts above.
- Ordering: invalid link/orientation inputs cannot trigger field-strength
  construction or a gauge-loop side effect first. Lattice-signature mismatch
  obeys the same pre-field-strength gate.

## pyquda_gluon_renorm.FF

`FF(info,gauge,g_0,mu1,nu1,mu2,nu2)` contracts a local field product.

- Returns: root host time series; requires `Gt=1`.
- Example: `ct = FF(info,gauge,g0,0,3,0,3)`.
- Domain: the explicit and gauge-owned lattice signatures must match exactly;
  coupling and time decomposition are validated before `F_munu` or reduction.

## pyquda_gluon_renorm.FT_Phase

`FT_Phase(p,latt_info)` constructs a global-coordinate checkerboard phase for
four integer modes ordered `(px,py,pz,pt)`. Labels are canonicalized to
`np.fft.fftfreq`-compatible bins; on an even extent, both `+L/2` and `-L/2`
select the canonical Nyquist representative `-L/2`.

- Returns: CuPy `(e,t,z,y,xh)` phase.
- Example: `phase = FT_Phase([1,0,0,0],info)`.

## pyquda_gluon_renorm.FT_Gauge_1mom

`FT_Gauge_1mom(gauge_data,p,latt_info,half_flag=1)` contracts one global
momentum. With the default link-centered contract, direction `mu` receives
`exp(-i*p_mu/2)`; pass `half_flag=0` only for an explicitly site-centered
four-direction field.
The input must have exact leading layout
`(4,2,Lt,Lz,Ly,Lx/2,Nc,Nc)` with square color matrices.

- Returns: normalized root host `(4,3,3)`; non-root returns `None`.
- Example: `Ap = FT_Gauge_1mom(A,[1,0,0,0],info,half_flag=1)`.

## pyquda_gluon_renorm.FT_Prop_1mom

`FT_Prop_1mom(prop_data,p,latt_info)` performs the analogous propagator
transform. The input must have exact
`(2,Lt,Lz,Ly,Lx/2,4,4,3,3)` layout.

- Returns: normalized root host spin-color propagator; every non-root rank
  returns `None` through the reduction contract.
- Example: `Sp = FT_Prop_1mom(prop.data,p,info)`.

## pyquda_gluon_renorm.FFT_Gauge_Allmom_MPI

`FFT_Gauge_Allmom_MPI(gauge_data,latt_info,comm,half_flag=1)` converts to
lexicographic storage and runs a single-rank CuPy FFT. For a link-centered
field, direction `mu` receives `exp(-i*p_mu/2)`.
It requires the same exact gauge layout as `FT_Gauge_1mom` and rejects
nontrivial `grid_size` or communicator size before the FFT.

- Returns: CuPy `(mu,t,z,y,x,3,3)` all-momentum array.
- Example: `allp = FFT_Gauge_Allmom_MPI(A,info,comm)`.
- Boundary: rejects any nontrivial grid/communicator.
- Domain: `half_flag` is exactly integer 0/1; Boolean aliases are rejected.

## pyquda_gluon_renorm.gaugeEMT_munu

`gaugeEMT_munu(gauge,g_0)` constructs off-diagonal/contracted gluonic EMT
candidates; `gaugeEMT_mumu(...,def_type=1)` produces diagonal components and
`gaugeEMT_mumu_tzyx` preserves sites.

- Returns: small contracted array, time/site array according to the exact name.
- Example: `T = gaugeEMT_munu(gauge,g0)`.
- Site ownership: `gaugeEMT_mumu_tzyx` returns four local CuPy fields on one
  rank; on multiple ranks, rank 0 receives four global NumPy fields and every
  non-root rank receives `(None,None,None,None)`.
- Provenance: exact normalization and trace prescription remain unverified in
  the copied source and must be mapped before physical use.
- Domain: `def_type` is a non-Boolean selector in `{0,1,2}` for the contracted
  API and `{1,2}` for the site-resolved API.
- Ordering: `gaugeEMT_munu` validates finite positive non-Boolean `g_0` before
  device allocation or field-strength construction.

## pyquda_gluon_renorm.ExA_t

`ExA_t(gauge,g_0,half_flag=1)` computes the time-resolved gauge-dependent
`E x A` candidate.

- Returns: root-owned host `(3,Lt)` array; every non-root rank returns `None`.
- Example: `spin_t = ExA_t(gauge,g0)`.
- Reference context: arXiv:1310.4263; gauge fixing is mandatory.

## pyquda_gluon_renorm.Topological_current_Kmu

`Topological_current_Kmu`, `Topological_current_Kmu_t`, and
`Topological_current_Kmu_tzyx` return progressively contracted/site-resolved
legacy Chern-Simons-current candidates.

- Returns: four-vector, time-resolved data, or four site fields according to
  the exact name.
- Example: `Ksite = Topological_current_Kmu_tzyx(gauge,g0)`.
- Site ownership: `Topological_current_Kmu_tzyx` follows the same one-rank
  CuPy / multi-rank root NumPy / multi-rank non-root four-`None` contract as
  `gaugeEMT_mumu_tzyx`.
- Boundary: exact component completeness/normalization is not independently
  verified; require gauge-fixing and equation-level provenance.

## pyquda_gluon_renorm.generate_pselect_mask

`generate_pselect_mask(info,mommin,mommax,cut_list,mode_list)` creates global
dense host masks using continuum-like `ap_mu=2*pi*n_mu/L_mu`. Every generated
integer candidate is canonicalized per direction to the NumPy FFT fundamental
bin before the cut and mask comparison, so wrapped labels and even-lattice
`+L/2` select the same bin as the canonical `-L/2` representative.

- Returns: list of Boolean global four-dimensional masks.
- Example: `masks = generate_pselect_mask(info,0,6,cuts,modes)`.
- Boundary: setup memory is `O(V_global)`.
- Domain: bounds and modes are non-Boolean integers; every cut is finite real
  with `0 < cut <= 1`. Invalid cuts fail before dense-mask allocation.

## Executable example registry

The quality contract maps every public symbol below to one real AST-direct
local test. CPU/stub/structural/rejection examples support at most E1; they
do not establish GPU, MPI, QUDA, runtime, interacting-physics, or production
evidence.

| Public symbol | Example ID | Test ID | Mode |
|---|---|---|---|
| `A_mu` | `API-EX-001` | `test_adversarial_public_inputs_fail_closed_before_numerical_kernels` | `rejection-only-e1` |
| `ExA_t` | `API-EX-022` | `test_gluon_exa_and_topological_current_match_independent_epsilon_sum` | `cpu-synthetic` |
| `FF` | `API-EX-029` | `test_gluon_zero_length_wilson_product_matches_local_normalization` | `cpu-synthetic` |
| `FFT_Gauge_Allmom_MPI` | `API-EX-002` | `test_all_eight_skills_match_independent_synthetic_golden` | `cpu-synthetic` |
| `FT_Gauge_1mom` | `API-EX-027` | `test_gluon_single_momentum_canonicalizes_even_nyquist_label` | `cpu-synthetic` |
| `FT_Phase` | `API-EX-026` | `test_gluon_public_api_invalid_domains_fail_before_kernels` | `rejection-only-e1` |
| `FT_Prop_1mom` | `API-EX-024` | `test_gluon_fourier_entrypoints_reject_malformed_checkerboards` | `rejection-only-e1` |
| `FW_FW` | `API-EX-029` | `test_gluon_zero_length_wilson_product_matches_local_normalization` | `cpu-synthetic` |
| `FW_tildeFW` | `API-EX-025` | `test_gluon_lattice_metadata_matches_gauge_before_field_strength` | `cpu-synthetic` |
| `F_and_tildeF` | `API-EX-001` | `test_adversarial_public_inputs_fail_closed_before_numerical_kernels` | `rejection-only-e1` |
| `F_munu` | `API-EX-023` | `test_gluon_f_munu_triplets_match_full_tensor_pair_order` | `cpu-synthetic` |
| `Topological_current_Kmu` | `API-EX-022` | `test_gluon_exa_and_topological_current_match_independent_epsilon_sum` | `cpu-synthetic` |
| `Topological_current_Kmu_t` | `API-EX-026` | `test_gluon_public_api_invalid_domains_fail_before_kernels` | `rejection-only-e1` |
| `Topological_current_Kmu_tzyx` | `API-EX-022` | `test_gluon_exa_and_topological_current_match_independent_epsilon_sum` | `cpu-synthetic` |
| `g_0` | `API-EX-020` | `test_gluon_coupling_uses_gauge_action_normalization` | `cpu-synthetic` |
| `gaugeEMT_mumu` | `API-EX-021` | `test_gluon_emt_matches_explicit_color_and_lorentz_loops` | `cpu-synthetic` |
| `gaugeEMT_mumu_tzyx` | `API-EX-001` | `test_adversarial_public_inputs_fail_closed_before_numerical_kernels` | `rejection-only-e1` |
| `gaugeEMT_munu` | `API-EX-021` | `test_gluon_emt_matches_explicit_color_and_lorentz_loops` | `cpu-synthetic` |
| `generate_pselect_mask` | `API-EX-019` | `test_gluon_anisotropic_momentum_mask_uses_each_axis_extent` | `cpu-synthetic` |
| `plaq_munu` | `API-EX-028` | `test_gluon_uses_pyquda_field_shift_signature` | `cpu-synthetic` |
