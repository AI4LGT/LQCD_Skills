# `pyquda_quark_renorm` 验证与发布状态

## 当前状态

- Release label：`experimental-static`。
- 最高按函数直接证据：`E2-selected-single-DCU-spin-color-kernels`；legacy series/provenance 仍停留在 E0/E1。
- 调用策略：`explicit-only`；必须显式写 `$pyquda_quark_renorm`。
- 物理 contract：[reference/PHYSICS_CONTRACT.md](reference/PHYSICS_CONTRACT.md)。
- reference reproduction：[E2E-QRK-01](../../docs/pyquda_skill_evidence/manifests/E2E-QRK-01.json)，当前为 `PARTIAL_SELECTED_REFERENCE_E1_REMAINDER_BLOCKED`；仅四个 source-mapped gates 达到 selected E1。
- target-runtime ledger：[KUNSHAN-SINGLE-DCU-01](../../benchmark/pyquda_skills/gpu_validation/evidence.json)；effective result 为 job `119734287`。

## 已直接验证

- spin-color inversion、gamma5 adjoint、amputation shape/backend 与 covariance-preserving selected host algebra；
- Chetyrkin--Rétey Eqs. (34), (36), (37) mapped subset 和 Eq. (41) independent checkpoint；
- Gracey arXiv:hep-ph/0304113v1 Eq. (4.11) 的 RI-prime tensor fixed-order
  inverse：第三阶变量转换为 $a_s^3/64$，不再使用无来源的 `/6`；
- Baikov--Chetyrkin--Kühn arXiv:1402.6611 的 MS mass coefficients 与
  $c(a_s)$ running algebra；equal-scale/composition regression 已覆盖；
- coupling/domain/nonfinite/Boolean gates、conditioned inverse scan 与 singular rejection；
- selected Padé/fit implementation contracts，不将其提升为 production uncertainty model。
- gfx906/CuPy 12.3.0 上 complex64/complex128 compound-index inverse errors 为 `5.96e-8`/`1.94e-16`，完整 spin+color gamma5 adjoint 与 involution errors 为 `0`；spin-only transpose mutant separation 约为 `5.52e-2`。
- local CPU/NumPy/SciPy selected-reference artifact 对 CR2000 Eq. (41)、
  Eqs. (36)/(34)、Gracey Eq. (4.11) 与 BCK mass-running algebra 完成 4/4 PASS；
  其 source/test/provenance hashes 与预冻结 tolerances 由 manifest 校验。
- `MUT-01=PASS_9_KILLED`：9/9 预声明 mutations 均被区分，包括 tensor
  denominator 的 obsolete `/6` 与 source-required `/64`；没有 survivor。
- versioned independent forward evaluation `2026-08-24-current-v5` 已在冻结
  snapshot 上完成 32/32 PASS（run-wide 9 fresh blind + 3 hash-reused）；该
  E1 gate 只验证冻结请求上的 behavior 与 claim boundary，不补齐 coefficient
  provenance，也不升级 runtime 或 physics evidence。

未映射 SMOM、general-xi、Padé、field/tensor running tables 仍只有 literal
source evidence；RI-prime tensor Eq. (4.11) 的 `/64` 修正不为这些相邻表提供
provenance。

## 当前 blockers

- 多数 coefficients 缺逐 equation operator/scheme/gauge/normalization map；
- interacting NPR、correlated continuum fits 与完整 statistical/fit/scale/truncation error budget 未闭合。

## 本地复核

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v \
  tests.test_pyquda_skills tests.test_pyquda_physics_contracts \
  tests.test_pyquda_phase_d_quality
SKILL_VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
python "$SKILL_VALIDATOR" skills/pyquda_quark_renorm
```

通用 `quick_validate.py` 当前会因仓库既有的 underscore identifier 报 naming
error；不要单独重命名。结构、YAML 与加载契约以本仓库 tests 和
`run.py --list-skills` 为准。

## 升级证据

每个请求 channel 都要冻结 primary source/equation、operator、input/output scheme、gauge、`N_f`、expansion variable、loop order、tensor/scalar normalization、fit window/covariance 与 uncertainty decomposition。独立实现必须复现 reference points；raw RI、continuum extrapolation 和 MS-bar conversion 分阶段保存。未映射 channels 继续 fail closed，且 explicit-only 只有 provenance、independent oracle、current reviewer 同时通过才能解除。

统一 release matrix：[docs/pyquda_skill_release_matrix.md](../../docs/pyquda_skill_release_matrix.md)。
