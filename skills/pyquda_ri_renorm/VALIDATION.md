# `pyquda_ri_renorm` 验证与发布状态

## 当前状态

- Release label：`experimental-static`。
- 最高按函数直接证据：`E2-selected-single-DCU-inverse-amputation-tree-algebra`；standard interacting scheme evidence 仍为 E0/E1。
- 调用策略：允许自动路由；nonexceptional custom combinations 不自动获得标准 RI/SMOM scheme identity。
- 物理 contract：[reference/PHYSICS_CONTRACT.md](reference/PHYSICS_CONTRACT.md)。
- 物理闭环：[E2E-RI-01](../../docs/pyquda_skill_evidence/manifests/E2E-RI-01.json)，当前为 `BLOCKED_BY_PREREQUISITES`。
- target-runtime ledger：[KUNSHAN-SINGLE-DCU-01](../../benchmark/pyquda_skills/gpu_validation/evidence.json)；effective result 为 job `119734287`。

## 已直接验证

- `p_in/p_out/q/omega` kinematics 与 explicit `gamma_mu/qslash` projector family；
- independent-leg spin-color inversion/amputation 和 no-batch-broadcast contract；
- tree-normalized `Zq/ZA/ZV/ZS/ZP/ZT` algebra 与 incoming/outgoing/mean leg ownership；
- gamma/projector/tree matrices 的 exact `(4,4)`、same-backend、numeric、finite fail-closed gate。
- gfx906/CuPy 12.3.0 上 complex64/complex128 inverse errors 为 `8.94e-8`/`1.94e-16`，amputation errors 为 `4.67e-9`/`9.34e-18`；四种 Zq leg ownership 的 tree-level deviation 均为 `0`，swapped-leg mutant separation 约为 `1.02e-3`。
- versioned independent forward evaluation `2026-08-24-current-v5` 已在冻结
  snapshot 上完成 32/32 PASS（run-wide 9 fresh blind + 3 hash-reused）；该
  E1 gate 只验证冻结请求上的 behavior 与 claim boundary，不升级 runtime 或
  physics evidence。

tree-level agreement 不替代 gauge-fixed interacting NPR、Ward identities 或 continuum matching。

## 当前 blockers

- custom qslash/axial/Zq/leg combinations 缺标准 continuum scheme mapping；
- 没有 gauge-fixing functional/residual、interacting vertices 或 independent NPR table；
- 没有 production I/O、QUDA solve、multi-rank 或 interacting NPR evidence；

## 本地复核

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v \
  tests.test_pyquda_skills tests.test_pyquda_cross_skill_contract \
  tests.test_pyquda_phase_d_quality
SKILL_VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
python "$SKILL_VALIDATOR" skills/pyquda_ri_renorm
```

通用 `quick_validate.py` 当前会因仓库既有的 underscore identifier 报 naming
error；不要单独重命名。结构、YAML 与加载契约以本仓库 tests 和
`run.py --list-skills` 为准。

## 升级证据

冻结 gauge-fixing functional/residual/algorithm、action/mass/volume/a、momentum definition、`p_in/p_out/q/mu`、operator/projector/tree vertex、Zq leg definition、scheme identity、precision、checksums 和 complex tolerance。比较 independent-leg amputation、适用 Ward identities 与同 ensemble/scheme 的独立 NPR code；raw RI factors 与 MS-bar conversion 必须分阶段保存，才能考虑 E4。

统一 release matrix：[docs/pyquda_skill_release_matrix.md](../../docs/pyquda_skill_release_matrix.md)。
