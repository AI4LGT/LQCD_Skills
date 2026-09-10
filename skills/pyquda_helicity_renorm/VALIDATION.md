# `pyquda_helicity_renorm` 验证与发布状态

## 当前状态

- Release label：`experimental-static`。
- 最高按函数直接证据：`E1-target-cluster-CPU-algebra`；该模块 `gpu_applicable=false`，占用 DCU allocation 不构成 GPU 证据。
- 调用策略：`explicit-only`；必须显式写 `$pyquda_helicity_renorm`。
- 物理 contract：[reference/PHYSICS_CONTRACT.md](reference/PHYSICS_CONTRACT.md)。
- source-verification ledger：[reference/SOURCE_VERIFICATION.md](reference/SOURCE_VERIFICATION.md)；
  five-loop beta function 有直接原始式映射；已定位的 candidate primary source 虽能
  term-match matching/running 的部分结构，却在 $R_{12}/R_{21}$ tree terms 与 local
  branches 矛盾。因此 matching 与 $\Gamma$ tables 仍为 non-promoting
  `UNVERIFIED_LEGACY`/`PRIMARY_SOURCE_CONTRADICTION`。
- reference reproduction：[E2E-HEL-01](../../docs/pyquda_skill_evidence/manifests/E2E-HEL-01.json)，当前为 `BLOCKED_BY_PREREQUISITES`。
- target-runtime ledger：[KUNSHAN-SINGLE-DCU-01](../../benchmark/pyquda_skills/gpu_validation/evidence.json)；effective result 为 job `119734287` 的 CPU/SciPy row。

## 已直接验证

- literal output order `(R11,R12,R21,R22)`；
- one-loop placement `R12=8a`、`R21=-6a` 与 equal-scale algebra；
- code-defined Padé construction、singular-domain rejection 与 ODE orientation；
- scale/flavor/loop/nonfinite/Boolean 输入 contract。
- Python 3.8.20/NumPy 1.24.4/SciPy 1.10.1 上 one-loop `R12=8a`、`R21=-6a` error 为 `0`；equal-scale error 为 `0`，noncommuting-basis 与 small-step checks 分别为 `1.29e-10`、`5.48e-8`。
- versioned independent forward evaluation `2026-08-24-current-v5` 已在冻结
  snapshot 上完成 32/32 PASS（run-wide 9 fresh blind + 3 hash-reused）；该
  E1 gate 只验证冻结请求上的 behavior 与 claim boundary，不补齐 coefficient
  provenance，也不升级 runtime 或 physics evidence。

这些检查不能给 legacy matching/anomalous-dimension decimals 提供缺失的 operator、scheme、gauge、basis 或 equation provenance。

## 当前 blockers

- candidate primary source 已定位，但其 $R_{12}/R_{21}$ tree-level ones 与 local
  matching branches 矛盾；
- two-operator basis orientation、scheme、gauge 和逐 coefficient map 未闭合；
- 没有独立 reference matrix/running points 与 truncation/coupling uncertainty。

## 本地复核

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v \
  tests.test_pyquda_skills tests.test_pyquda_physics_contracts \
  tests.test_pyquda_phase_d_quality
SKILL_VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
python "$SKILL_VALIDATOR" \
  skills/pyquda_helicity_renorm
```

通用 `quick_validate.py` 当前会因仓库既有的 underscore identifier 报 naming
error；不要单独重命名。结构、YAML 与加载契约以本仓库 tests 和
`run.py --list-skills` 为准。

## 升级证据

先提供 exact primary source，并为每个 matrix element/`gamma_ij` 冻结 paper version、equation、ordered basis、matrix orientation、scheme、gauge、`N_f`、expansion variable、loop order 与 scale domain；再由独立实现复现 reference points、composition 和 matching/running order，并做独立 review。provenance、independent oracle、current-snapshot reviewer 三者未同时通过前，不解除 explicit-only。

统一 release matrix：[docs/pyquda_skill_release_matrix.md](../../docs/pyquda_skill_release_matrix.md)。
