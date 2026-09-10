# `pyquda_momentum_smear` 验证与发布状态

## 当前状态

- Release label：`experimental-static`。
- 最高按函数直接证据：`E3-selected-unit-gauge-QUDA-Wuppertal-smoke`；其余 selected phase helpers 为 E2，skill-wide release 仍受 E0/E1 blockers 限制。
- 调用策略：允许自动路由；smearing mode、physical hadron momentum 与 Fourier momentum 必须分别记录。
- 物理 contract：[reference/PHYSICS_CONTRACT.md](reference/PHYSICS_CONTRACT.md)。
- 物理闭环：[E2E-MOM-01](../../docs/pyquda_skill_evidence/manifests/E2E-MOM-01.json)，当前为 `BLOCKED_BY_PREREQUISITES`。
- target-runtime ledger：[KUNSHAN-SINGLE-DCU-01](../../benchmark/pyquda_skills/gpu_validation/evidence.json)；effective result 为 job `119734287`。

## 已直接验证

- 每个 Wuppertal iteration 的 forward/backward phased-link ownership；
- physical `+P_f` sink projection `exp(-i P_f x)` 与 conditional sequential candidate `exp(+i P_f x)`；
- source/sink/one-smear orchestration、`mrhs/restart` forwarding 与 metadata fail-closed；
- global-coordinate Fourier phase、periodicity、sign-flip-sensitive local oracles。
- gfx906 上两个真实 QUDA Wuppertal calls 的 unit-gauge plane-wave eigenvalue-ratio error 为 `1.07e-17`；spatial-link phases/time-link control 为 `0`，sink/sequential Fourier errors 为 `1.79e-15`。目标 PyQUDA 的 `phase_v2` 先在 host 构造 phase 再传到 backend，此边界未被写成 GPU phase generation。
- versioned independent forward evaluation `2026-08-24-current-v5` 已在冻结
  snapshot 上完成 32/32 PASS（run-wide 9 fresh blind + 3 hash-reused）；该
  E1 gate 只验证冻结请求上的 behavior 与 claim boundary，不升级 runtime 或
  physics evidence。

这些验证不闭合真实 downstream dagger/conjugation、QUDA true residual、multi-rank ownership 或 interacting 2pt/3pt correlator。

## 当前 blockers

- 没有 QUDA inversion/true residual、multi-rank ownership 或完整 downstream dagger closure；
- 没有 interacting gauge configuration、independent contraction 或 complex correlator tolerance；
- source/sink/Fourier/sequential sign chain 尚未在完整 nonforward pipeline 运行；

## 本地复核

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v \
  tests.test_pyquda_skills tests.test_pyquda_cross_skill_contract \
  tests.test_pyquda_phase_d_quality
PYTHONDONTWRITEBYTECODE=1 python -B utils/pyquda_upstream_contract.py --strict-hash
SKILL_VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
python "$SKILL_VALIDATOR" skills/pyquda_momentum_smear
```

通用 `quick_validate.py` 当前会因仓库既有的 underscore identifier 报 naming
error；不要单独重命名。结构、YAML 与加载契约以本仓库 tests 和
`run.py --list-skills` 为准。

## 升级证据

冻结 ensemble/action/mass/volume/a/boundary、`P_i/P_f/q`、source/sink smearing modes、sink Fourier sign、sequential dagger owner、solver precision/residual、contraction checksum 和复数 tolerance。运行 ±P、unsmeared/smeared 及三个独立 sign-flip controls，并与独立 complex 2pt/3pt reference 比较；只比较绝对值或峰位置不能升级到 E4。

统一 release matrix：[docs/pyquda_skill_release_matrix.md](../../docs/pyquda_skill_release_matrix.md)。
