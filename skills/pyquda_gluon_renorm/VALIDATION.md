# `pyquda_gluon_renorm` 验证与发布状态

## 当前状态

- Release label：`experimental-static`。
- generic Fourier-kernel function evidence 最高为 case-limited `E2-selected-single-DCU-Fourier-kernels`；另有 Run-15 selected C24P29 four-rank bare E2E evidence `E4-selected-four-rank-bare-gluon`，两者都不构成 gauge-fixed 或 renormalized function claim。
- 调用策略：允许自动路由；历史名称中的 `renorm` 不构成 renormalization/mixing 证明。
- 物理 contract：[reference/PHYSICS_CONTRACT.md](reference/PHYSICS_CONTRACT.md)。
- 物理闭环：[E2E-GLU-01](../../docs/pyquda_skill_evidence/manifests/E2E-GLU-01.json)，当前为 `PARTIAL_RUN15_SELECTED_BARE_GLUON_E4_PASS_GAUGE_FIXED_RENORMALIZATION_BLOCKED`：Run-15 只闭合 selected four-rank bare Clover/Fourier 与 global-SU(3) control，不构成 gauge-fixed、mixing、renormalized 或 physics PASS。
- target-runtime ledger：[KUNSHAN-SINGLE-DCU-01](../../benchmark/pyquda_skills/gpu_validation/evidence.json)；effective result 为 job `119734287`。

## 已直接验证

- checkerboard/layout gate、direction-specific half-link Fourier phase 与 momentum canonicalization；
- plaquette coupling convention、clover `F_munu` triplets、zero-length Wilson products；
- literal EMT color/Lorentz loops、epsilon sums 与 selected topological-current arrays；
- malformed metadata、nonfinite/complex-real contract 和 silent host fallback rejection。
- gfx906/CuPy 12.3.0 单 rank 上 `FT_Gauge_1mom` 与 `FFT_Gauge_Allmom_MPI` 对独立 checkerboard map/显式 host DFT 的 relative max errors 为 `5.23e-15` 与 `2.26e-15`；公共 half-link phase mutant 分离度为 `2.00`。
- Run-15（job `119914950`）的独立 sealed C24P29 four-rank bare phase 写出 rank-zero `gluon.json`：independent Clover max absolute error 为 `0`，one-momentum Fourier replay 的 max absolute/relative errors 为 `4.375934525892162e-18` / `1.7659178799952492e-14`，global-SU(3) mutant separation 为 `3.4297675881667187e-4 > 1e-8`。这是 selected bare case 的 E4，不能外推为 gauge-fixed、mixing、continuum matching、statistics、physics 或 performance evidence。
- versioned independent forward evaluation `2026-08-24-current-v5` 已在冻结
  snapshot 上完成 32/32 PASS（run-wide 9 fresh blind + 3 hash-reused）；该
  E1 gate 只验证冻结请求上的 behavior 与 claim boundary，不升级 runtime 或
  physics evidence。

这些结果只支持 bare array algebra，不证明 continuum normalization、gauge prescription、operator mixing 或 renormalized observable。

## 当前 blockers

- `A_mu`、`E x A` 与 `K_mu` 需要指定 gauge fixing；
- EMT/current candidates 缺逐式 `a/g0/trace/volume`、mixing 与 renormalization closure；
- Run-15 已有 selected four-rank bare gauge-invariant/global-SU(3) control；仍没有 checksum-bound gauge-fixed configuration、gauge-fixing functional/residual/Gribov policy、ordered mixing basis 或 independent gauge-fixed/renormalized component oracle；

## 本地复核

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v \
  tests.test_pyquda_skills tests.test_pyquda_phase_d_quality
PYTHONDONTWRITEBYTECODE=1 python -B utils/pyquda_upstream_contract.py --strict-hash
SKILL_VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
python "$SKILL_VALIDATOR" skills/pyquda_gluon_renorm
```

通用 `quick_validate.py` 当前会因仓库既有的 underscore identifier 报 naming
error；不要单独重命名。结构、YAML 与加载契约以本仓库 tests 和
`run.py --list-skills` 为准。

## 升级证据

冻结 ensemble/gauge action/beta/a/volume、gauge fixing、smoothing、Fourier/link-midpoint/epsilon/trace conventions、input checksum 与 independent clover/FFT/EMT reference。先做 gauge-invariant transformation control，再测试 gauge-dependent quantities；bare、gauge-fixed、mixed 和 renormalized outputs 必须分开。只有 interacting comparison 通过后才可考虑 E4。

统一 release matrix：[docs/pyquda_skill_release_matrix.md](../../docs/pyquda_skill_release_matrix.md)。
