# LQCD_Skills

A collection of agent skills for lattice QCD research, covering the full chain
from a physics observable to a numerical result: correlator design, propagator
production with PyQUDA/QUDA (GPU/MPI), Wick contractions and dedicated array
kernels, and statistical fitting analysis. The skills work with both Codex and
Claude Code and are discovered automatically through symlinks when this
repository is opened as the project root.

## Repository layout

```
LQCD_Skills/
├── skills/                  # 12 skill directories (actual content)
│   ├── lqcd-physics-correlator/   # pipeline: observable -> operators/correlators/contractions
│   ├── lqcd-physics-spectrum/     # pipeline: correlator -> spectral decomposition/fit templates
│   ├── pyquda-tool/               # pipeline: propagator spec -> PyQUDA production script
│   ├── lqcd-analysis/             # pipeline: correlator data -> physics results
│   ├── pyquda_momentum_smear/     # kernel: momentum smearing
│   ├── pyquda_blending/           # kernel: blended all-to-all
│   ├── pyquda_enhanced_interpolator/  # kernel: kinematically enhanced interpolators
│   ├── pyquda_hisq_recover/       # kernel: HISQ -> naive four-spin reconstruction
│   ├── pyquda_ri_renorm/          # renormalization: RI'/MOM projection and amputation
│   ├── pyquda_quark_renorm/       # renormalization: RI/SMOM -> MS-bar conversion and running
│   ├── pyquda_gluon_renorm/       # renormalization: bare gluon observable kernels
│   └── pyquda_helicity_renorm/    # renormalization: helicity/EMT matching (legacy)
├── ensemble_registry.yaml   # ensemble parameter data source (NOT a skill)
├── TODO.md                  # planned physics/software/analysis roadmap
├── .codex/skills -> ../skills    # symlink: Codex auto-discovery
└── .claude/skills -> ../skills   # symlink: Claude Code auto-discovery
```

## Skill overview

### Pipeline skills (lightweight: SKILL.md only, plus worked reference examples)

Organized in data-flow order; the output of each stage feeds the next:

| Order | Skill | Input -> Output |
|---|---|---|
| 1 | `lqcd-physics-correlator` | Physics observable -> interpolating operators, two-/three-point correlators, Wick contractions (gamma5-hermiticity, flavor symmetry), propagator requirements, einsum expressions; includes worked examples for pion/rho/proton masses and Lambda semi-leptonic decay |
| 2 | `lqcd-physics-spectrum` | Correlator definition -> spectral decomposition, overlap factors, backward-propagating state structure, fit function templates |
| 3 | `pyquda-tool` | Propagator specification -> single-file PyQUDA data production script for MPI submission; ensemble parameters read from `ensemble_registry.yaml` |
| 4 | `lqcd-analysis` | Per-configuration C(t) data -> jackknife/bootstrap resampling, effective mass, multi-state correlated fits (lsqfit), ratio/summation matrix elements, dispersion relation, chi2/dof/Q-value/AIC diagnostics, conversion to physical units |

### Computational kernels (implementation skills with full audit structure)

| Skill | Scope | Key references |
|---|---|---|
| `pyquda_momentum_smear` | Link-phased Wuppertal momentum smearing: source/sink/sequential source, S-to-P, S-to-S, nonforward modes, MRHS, Fourier phase signs | arXiv:1602.05525 |
| `pyquda_blending` | Blended all-to-all: low Laplacian modes + explicit stochastic complement, blending weights, meson/baryon elementals, projected perambulators | arXiv:2505.01719 |
| `pyquda_enhanced_interpolator` | Kinematically enhanced boosted interpolators: Euclidean gamma-plus/minus kernels, enhanced meson bilinears, nucleon diquark source/sink, direct-minus-exchange contractions | arXiv:2606.02447, arXiv:2501.00729 |
| `pyquda_hisq_recover` | Staggered/HISQ -> naive four-spin reconstruction in Wilson-like spin-color storage, arbitrary global source coordinates, MPI-local parity, host gather | — |

### Renormalization (implementation skills)

| Skill | Scope | Boundary |
|---|---|---|
| `pyquda_ri_renorm` | Gauge-fixed off-shell RI'/MOM and nonexceptional bilinear kinematics, spin-color inversion, vertex amputation, raw Zq/ZA/ZV/ZS/ZP/ZT | No MS-bar conversion; the bundled arrays are not a complete production NPR pipeline |
| `pyquda_quark_renorm` | RI/MOM, RI'/MOM, RI/SMOM -> MS-bar conversion/running tables, legacy diagonal-error host fits | Never silently trust an unmapped coefficient table; use `pyquda_ri_renorm` for raw projection |
| `pyquda_gluon_renorm` | Bare pure-gauge observable kernels: plaquette, link-centered A_mu, clover F_mn and dual tensors, F-W-F, gluon EMT candidates, Chern-Simons/topological current K_mu, Fourier transforms and momentum masks | A bare observable is not a renormalized one |
| `pyquda_helicity_renorm` | Review/cautious adaptation of legacy nf=3 helicity/EMT perturbative matching and two-operator RG code (R12/R21) | Not a production coefficient source unless the exact paper, operator basis, scheme, and equation mapping are supplied |

## Typical workflow

```
Physics goal (mass, decay constant, form factor, PDF/GPD matrix element, ...)
   |  lqcd-physics-correlator        operators + correlators + contractions + propagator list
   |  lqcd-physics-spectrum          spectral decomposition + fit templates
   v
Data production
   |  pyquda-tool                    propagator inversion script (MPI submission)
   |  pyquda_momentum_smear          boosted source/sink
   |  pyquda_blending                all-to-all propagators
   |  pyquda_hisq_recover            HISQ propagator spin reconstruction
   |  pyquda_ri_renorm / gluon_renorm  NPR and bare gluon observables
   v
Analysis                    (renormalization post-processing: quark_renorm / helicity_renorm)
   |  lqcd-analysis                  resampling + fitting + scale conversion
   v
Physics results (with statistical and systematic errors)
```

## Skill internal structure conventions

Implementation skills follow progressive disclosure, so routine calls do not
load the full documentation:

- `SKILL.md`: routing description (triggers/anti-triggers) plus the minimal
  information needed for ordinary code generation or review.
- `USER_GUIDE.md`: read only when the user asks for principles, formula
  derivations, a complete workflow, or validation boundaries.
- `reference/API.md`: function signatures and calling conventions of the
  array kernels.
- `reference/PHYSICS_CONTRACT.md`: physics conventions (gamma basis, phases,
  normalization, index ordering, etc.).
- `scripts/Def_*.py`: reference implementations; `e2e_validation.py` /
  `gpu_validation.py`: validation scripts.
- `VALIDATION.md`: release label, directly verified claims, current blockers,
  and evidence boundaries.
- `agents/openai.yaml`: agent interface metadata (display name, default
  prompt, implicit invocation policy).

## ensemble_registry.yaml

This is **not a skill** — it is a data source that skills (mainly
`pyquda-tool`) query. One entry per ensemble records: dimensions, boundary
conditions, gauge action and beta, scale setting (w0), configuration
paths/format/numbering, and per-flavor quark action/mass/csw/smearing/solver
tolerance/multigrid parameters plus the critical mass. The current example is
C24P29 (N_f=2+1, 24^3 x 72, a ~ 0.1053 fm); replace it with your actual
ensemble parameters.

## Usage

- Open this repository root in Codex or Claude Code; the symlinks make all
  skills discoverable.
- Invoke via the trigger phrases in each skill description (e.g. "compute
  propagators", "momentum smearing", "effective mass"), or explicitly with
  `$skill-name`.
- Each description also states anti-triggers ("Do not use for ..."); when
  routing is ambiguous, the anti-triggers take precedence.

## Evidence and validation boundaries

The `VALIDATION.md` of each implementation skill is the authoritative record
of its release status. When reading conclusions, distinguish:

- Local static/CPU checks and independent host reference comparisons are not
  GPU/MPI/QUDA runtime validation.
- A single-GPU deterministic case is not evidence for multi-rank,
  CUDA-aware MPI, stochastic estimator unbiasedness, or physics correlators.
- Release labels (e.g. `experimental-static`) and E0/E1/E2 tiers hold only
  within their stated prerequisites; an end-to-end closure marked
  `BLOCKED_BY_PREREQUISITES` must not be promoted to a runtime or physics
  conclusion.

See `TODO.md` for the roadmap.
