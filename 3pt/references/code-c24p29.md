# Full pyquda code — just for C24P29 point source, point sink for Lambda to proton

Ensemble parameters: `beta6.20_mu-0.2770_ms-0.2400_L24x72`

This code computes only zero momentum, `Gamma_cur_list=[1]`, `T_list=[1]`, `tseq_list=[8]`. Modify the corresponding lists to obtain more data points. Source position is `(0,0,0,tsrc)`.

```python
import sys
import gc
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source, phase
from pyquda.field import LatticePropagator

# ===========================
# configuration index and source position
# ===========================

cfg = int(sys.argv[1])
ix, iy, iz = 0, 0, 0

l_mass = "-0.2770"
s_mass = "-0.2356"
v = 0.951479
clover = 1 / (v**3)
print("clover= ", clover)

core.init([1, 1, 1, 4], resource_path=".cache")
latt_info = core.LatticeInfo([24, 24, 24, 72], -1, 1.0)
dirac_l = core.getDirac(
    latt_info, float(l_mass), 1e-10, 1000, 1.0, clover, clover,
    [[6, 6, 6, 3], [4, 4, 4, 6]]
)
dirac_s = core.getDirac(
    latt_info, float(s_mass), 1e-10, 1000, 1.0, clover, clover,
    [[6, 6, 6, 3], [4, 4, 4, 6]]
)

gauge_default = io.readChromaQIOGauge(
    config path
)
gauge_default.toDevice()

# ======================= new smear =========================
gauge_stout = gauge_default.copy()
gauge_stout.stoutSmear(1, 0.125, 4)

C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
Gx = gamma.gamma(1)
Gy = gamma.gamma(2)
Gz = gamma.gamma(4)
Gt = gamma.gamma(8)
G5 = gamma.gamma(15)
Cg5 = C @ G5
P_plus = (G0 + Gt) / 2

epsilon = cp.zeros((3, 3, 3))
for a in range(3):
    b = (a + 1) % 3
    c = (a + 2) % 3
    epsilon[a, b, c] = 1
    epsilon[a, c, b] = -1

# ===========================
# three-point function parameters
# ===========================
tsrc_list = list(range(0, 1, 1))  # = 0, 1
T_list = [1]       # projector matrices
Gamma_cur_list = [1]  # current matrices
tseq_list = [8]
mom_list = [[0, 0, 0]]

Lambda_to_proton_Gamma_curV = cp.zeros(
    (len(tsrc_list), len(mom_list), len(T_list),
     len(Gamma_cur_list), len(tseq_list), latt_info.Lt),
    "<c16"
)
Lambda_to_proton_Gamma_curA = cp.zeros(
    (len(tsrc_list), len(mom_list), len(T_list),
     len(Gamma_cur_list), len(tseq_list), latt_info.Lt),
    "<c16"
)

for tsrc_idx, tsrc in enumerate(tsrc_list):
    point_source = source.source12(latt_info, "point", [ix, iy, iz, tsrc])

    dirac_l.loadGauge(gauge_stout, True)
    prop_l = core.invertPropagator(dirac_l, point_source)

    dirac_s.loadGauge(gauge_stout, True)
    prop_s = core.invertPropagator(dirac_s, point_source)

    for mom_idx, mom in enumerate(mom_list):
        px, py, pz = mom

        mom_phase_current = phase.MomentumPhase(latt_info).getPhase(
            [px, py, pz, 0], [ix, iy, iz, 0]
        )
        mom_phase_final = phase.MomentumPhase(latt_info).getPhase(
            [-px, -py, -pz, 0], [ix, iy, iz, 0]
        )

        for tseq_idx, tseq in enumerate(tseq_list):
            for T_idx, T in enumerate(T_list):
                X_Lambda_to_proton = core.LatticePropagator(latt_info)
                src_seq_Lambda_to_proton = core.LatticePropagator(latt_info)

                X_Lambda_to_proton.data = (
                    contract(
                        "wtzyx, ijk, lmn, AB, GH, ID, wtzyxDGkl, wtzyxBHjm -> wtzyxIAni",
                        mom_phase_final, epsilon, epsilon, Cg5, Cg5,
                        gamma.gamma(T),
                        prop_l.data, prop_l.data,
                    )
                    - contract(
                        "wtzyx, ijk, lmn, AB, GH, ID, wtzyxAGil, wtzyxBHjm -> wtzyxIDnk",
                        mom_phase_final, epsilon, epsilon, Cg5, Cg5,
                        gamma.gamma(T),
                        prop_l.data, prop_l.data,
                    )
                )

                # First dagger: construct the sequential source
                X_Lambda_to_proton.data = contract(
                    "AB, wtzyxCBji, CD -> wtzyxADij",
                    G5, X_Lambda_to_proton.data.conj(), G5,
                )
                src_seq_Lambda_to_proton = source.sequential12(
                    X_Lambda_to_proton, (tsrc + tseq)
                )
                dirac_l.loadGauge(gauge_stout)
                propag_seq_Lambda_to_proton = core.invertPropagator(
                    dirac_l, src_seq_Lambda_to_proton
                )

                # Second dagger: conjugation / reordering
                propag_seq_dag = core.LatticePropagator(latt_info)
                propag_seq_dag.data = contract(
                    "AB, wtzyxCBji, CD -> wtzyxADij",
                    G5, propag_seq_Lambda_to_proton.data.conj(), G5,
                )

                for Gamma_cur_idx, Gamma_cur in enumerate(Gamma_cur_list):
                    gamma_v = gamma.gamma(Gamma_cur)
                    gamma_a = gamma.gamma(Gamma_cur) @ G5

                    three_pt_v = contract(
                        "wtzyxijba,jk,wtzyxkiab->wtzyx",
                        propag_seq_dag.data,
                        gamma_v,
                        prop_s.data,
                    )
                    Lambda_to_proton_Gamma_curV[
                        tsrc_idx, mom_idx, T_idx, Gamma_cur_idx, tseq_idx, :
                    ] += contract(
                        "wtzyx, wtzyx->t",
                        mom_phase_current,
                        three_pt_v,
                    )

                    three_pt_a = contract(
                        "wtzyxijba,jk,wtzyxkiab->wtzyx",
                        propag_seq_dag.data,
                        gamma_a,
                        prop_s.data,
                    )
                    Lambda_to_proton_Gamma_curA[
                        tsrc_idx, mom_idx, T_idx, Gamma_cur_idx, tseq_idx, :
                    ] += contract(
                        "wtzyx, wtzyx->t",
                        mom_phase_current,
                        three_pt_a,
                    )

                del X_Lambda_to_proton
                del src_seq_Lambda_to_proton
                del propag_seq_Lambda_to_proton
                del propag_seq_dag

tmp_Lambda_to_proton_Gamma_curV = core.gatherLattice(
    Lambda_to_proton_Gamma_curV.get(), [5, -1, -1, -1]
)
tmp_Lambda_to_proton_Gamma_curA = core.gatherLattice(
    Lambda_to_proton_Gamma_curA.get(), [5, -1, -1, -1]
)

if latt_info.mpi_rank == 0:
    np.save(
        f"./3pt_Lambda_to_proton/Lambda_to_proton_Gamma_curV_{cfg}.npy",
        tmp_Lambda_to_proton_Gamma_curV,
    )
    np.save(
        f"./3pt_Lambda_to_proton/Lambda_to_proton_Gamma_curA_{cfg}.npy",
        tmp_Lambda_to_proton_Gamma_curA,
    )
```
