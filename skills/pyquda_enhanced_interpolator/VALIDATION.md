# `pyquda_enhanced_interpolator` 验证与发布状态

## 当前状态

- Release label：`experimental-static`。
- 最高按函数直接证据：`E2-selected-single-DCU-gamma-and-Wick-algebra`；skill-wide release 仍受 E0/E1 blockers 限制。
- 调用策略：允许自动路由；本 skill 不拥有 propagator inversion、I/O 或 ensemble fit。
- 物理 contract：[reference/PHYSICS_CONTRACT.md](reference/PHYSICS_CONTRACT.md)。
- 物理闭环：[E2E-ENH-01](../../docs/pyquda_skill_evidence/manifests/E2E-ENH-01.json)，当前为 `BLOCKED_BY_PREREQUISITES`。
- target-runtime ledger：[KUNSHAN-SINGLE-DCU-01](../../benchmark/pyquda_skills/gpu_validation/evidence.json)；effective result 为 job `119735957`。

## 已直接验证

- Euclidean `gamma_plus/gamma_minus`、paper-normalized scaled `Q_plus` 与 canonical pion identity；
- bilinear adjoint 和 nucleon source complex-conjugation ownership；
- meson/baryon explicit spin-color Wick loops 与 momentum-reduction layout；
- 所有 public Dirac matrices 的 exact `(4,4)`、numeric、finite fail-closed gate。
- gfx906/CuPy 12.3.0 上 baryon explicit-loop relative error 为 `3.50e-14`；source conjugation、lightcone nilpotency、scaled-projector checks 为 `0--4.44e-16`，四个预声明 mutants 均被区分。
- versioned independent forward evaluation `2026-08-24-current-v5` 已在冻结
  snapshot 上完成 32/32 PASS（run-wide 9 fresh blind + 3 hash-reused）；该
  E1 gate 只验证冻结请求上的 behavior 与 claim boundary，不升级 runtime 或
  physics evidence。

这些验证不证明目标 propagator gamma basis、interacting overlap、effective-mass plateau 或 SNR 改善。

## 当前 blockers

- 没有与 conventional operator 共用 propagators/configurations 的相关统计比较；
- source/sink basis transform 尚未在真实 PyQUDA propagator 上闭合；
- device-buffer collective、QUDA inversion 与 interacting data 仍未验证；

## 本地复核

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v \
  tests.test_pyquda_skills tests.test_pyquda_cross_skill_contract \
  tests.test_pyquda_phase_d_quality
SKILL_VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
python "$SKILL_VALIDATOR" \
  skills/pyquda_enhanced_interpolator
```

通用 `quick_validate.py` 当前会因仓库既有的 underscore identifier 报 naming
error；不要单独重命名。结构、YAML 与加载契约以本仓库 tests 和
`run.py --list-skills` 为准。

## 升级证据

在运行前冻结 gamma representation、boost direction、operator normalization、ensemble/action/mass、source count、precision、cost、fit-window rule 与 covariance。用相同 propagators 对 enhanced/conventional operators 做 correlated comparison；无改善也必须保留。只有指定 observable/ensemble 的复 correlator、overlap、SNR 和 fit stability 均达到预注册 tolerance，才可考虑 E4/E5。

统一 release matrix：[docs/pyquda_skill_release_matrix.md](../../docs/pyquda_skill_release_matrix.md)。
