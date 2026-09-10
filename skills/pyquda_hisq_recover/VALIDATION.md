# `pyquda_hisq_recover` 验证与发布状态

## 当前状态

- Release label：`experimental-static`。
- 最高按函数直接证据：`E2-selected-single-DCU-generic-reconstruction`；producer-specific evidence 仍为 E0。
- 调用策略：允许自动路由；输出只能称 naive four-spin reconstruction，不是 Wilson-action conversion。
- 物理 contract：[reference/PHYSICS_CONTRACT.md](reference/PHYSICS_CONTRACT.md)。
- 物理闭环：[E2E-HISQ-01](../../docs/pyquda_skill_evidence/manifests/E2E-HISQ-01.json)，当前为 `BLOCKED_BY_PREREQUISITES`。
- target-runtime ledger：[KUNSHAN-SINGLE-DCU-01](../../benchmark/pyquda_skills/gpu_validation/evidence.json)；effective result 为 job `119735957`，首次退化 mutant failure 保留在 job `119734287`。

## 已直接验证

- `Omega(x) G_chi Omega(x0)^dagger` 的 gamma order、global source coordinate 与 dtype preservation；
- nonzero source parity、two-odd-coordinate order 与 vectorized local reconstruction；
- custom `gamma_ops` 的 exact keys、`(4,4)`、numeric、finite gate；
- `gather` strict Boolean、local/root ownership contract 与 no-Python-site-loop property。
- gfx906/CuPy 12.3.0 上 complex64/complex128、three-odd gamma-order 与 two-odd source-dagger 四个 cases 的 production/oracle relative error 均为 `0`，对应 mutants separation 均为 `2`。
- versioned independent forward evaluation `2026-08-24-current-v5` 已在冻结
  snapshot 上完成 32/32 PASS（run-wide 9 fresh blind + 3 hash-reused）；该
  E1 gate 只验证冻结请求上的 behavior 与 claim boundary，不升级 runtime 或
  physics evidence。

这些验证不识别某个 producer 的 taste、eta phases、origin、boundary wrap、normalization 或 storage convention。

## 当前 blockers

- 没有 named/versioned producer adapter 与真实 producer data checksum；
- 没有跨 boundary、不同 rank offset 的 multi-rank reference comparison；
- named producer compatibility、MPI gather、production I/O 与 interacting correlator 仍未验证；

## 本地复核

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v \
  tests.test_pyquda_skills tests.test_pyquda_phase_d_quality
PYTHONDONTWRITEBYTECODE=1 python -B utils/pyquda_upstream_contract.py --strict-hash
SKILL_VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
python "$SKILL_VALIDATOR" skills/pyquda_hisq_recover
```

通用 `quick_validate.py` 当前会因仓库既有的 underscore identifier 报 naming
error；不要单独重命名。结构、YAML 与加载契约以本仓库 tests 和
`run.py --list-skills` 为准。

## 升级证据

冻结 producer name/version/file/dataset/checksum、taste/eta/gamma basis、source normalization/origin、boundary handling、global/local `tzyx` layout、even-odd storage、rank grid/offset 和 dtype。以 producer 自身 spin-lift 或独立 reference 对照 global/local outputs；metadata 缺失必须 fail closed。通过指定 producer 的 multi-rank comparison 后才可考虑 E3/E4。

统一 release matrix：[docs/pyquda_skill_release_matrix.md](../../docs/pyquda_skill_release_matrix.md)。
