# `pyquda_blending` 验证与发布状态

## 当前状态

- Release label：`experimental-static`。
- 最高按函数直接证据：`E2-selected-single-DCU-meson-elemental`；skill-wide release 仍受 E0/E1 blockers 限制。
- 调用策略：允许自动路由；与 momentum smearing 组合时必须保持 basis 与 quark-line artifact ownership 分离。
- 物理 contract：[reference/PHYSICS_CONTRACT.md](reference/PHYSICS_CONTRACT.md)。
- 物理闭环：[E2E-BLD-01](../../docs/pyquda_skill_evidence/manifests/E2E-BLD-01.json)，当前为 `BLOCKED_BY_PREREQUISITES`。
- target-runtime ledger：[KUNSHAN-SINGLE-DCU-01](../../benchmark/pyquda_skills/gpu_validation/evidence.json)；effective result 为 job `119734287`。

## 已直接验证

- blending tuple weights、`N_st=0` distillation limit 与重复 stochastic label contract；
- meson/baryon elementals、projected perambulator axes 和局部两点函数代数；
- global-coordinate spatial Fourier phase、volume normalization 与 buffer collective stub；
- malformed shape、Boolean index/count、nonfinite phase 输入的 fail-closed 路径。
- gfx906/CuPy 12.3.0 上 `meson_elemental` 对独立 host loops 的 relative max error 为 `5.16e-16`；Fourier 反号与忽略 global offset mutants 分离度分别为 `1.16`、`1.98`。
- versioned independent forward evaluation `2026-08-24-current-v5` 已在冻结
  snapshot 上完成 32/32 PASS（run-wide 9 fresh blind + 3 hash-reused）；该
  E1 gate 只验证冻结请求上的 behavior 与 claim boundary，不升级 runtime 或
  physics evidence。

E2 只覆盖上述 deterministic single-DCU case；它不是 stochastic estimator unbiasedness、真实 CUDA-aware MPI 或 production correlator 证据。

## 当前 blockers

- noise、orthogonal complement、dilution、Dirac solves、I/O 与配置平均 pipeline 未闭合；
- 没有多个预声明 seeds 对 exact all-to-all reference 的统计检验；
- multi-rank MPI、CUDA-aware collective、QUDA solves 与 interacting data 仍未验证；

## 本地复核

从仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v \
  tests.test_pyquda_skills tests.test_pyquda_cross_skill_contract \
  tests.test_pyquda_phase_d_quality
SKILL_VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
python "$SKILL_VALIDATOR" skills/pyquda_blending
```

`quick_validate.py` 只检查 skill 结构；数值证据以测试和 evidence manifest 为准。
该通用 validator 当前会因仓库既有的 underscore identifier 报 naming error；
LQCD_Master registry/config 正是使用该名称。不要单独重命名一个 skill，结构、
YAML 与加载契约以本仓库 tests 和 `run.py --list-skills` 为准。

## 升级证据

先冻结 noise/dilution、`N_ev/N_st`、solver residual、rank/grid、输入 checksum、seeds、exact reference 与统计 tolerance；再由用户明确授权执行 E2E manifest。必须报告 setup/solve/contraction/communication/I/O/total cost、失败率和置信区间，才能考虑 E4/E5；单 seed 或作业退出码 0 不足以升级。

统一 release matrix：[docs/pyquda_skill_release_matrix.md](../../docs/pyquda_skill_release_matrix.md)。
