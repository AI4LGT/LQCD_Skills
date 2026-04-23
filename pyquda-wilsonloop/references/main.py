import sys
import numpy as np

from pyquda_utils import core, io
from pyquda_utils.core import X, T

# Run (example): mpirun -n 4 python3 main.py /public/home/wlhxzwa/LQCD_Master/runs/.cache 10000

# -----------------------------
# 1) Parameter definitions
# -----------------------------
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

ensemble_preset = "C24P29"

latt_size = [24, 24, 24, 72]
process_grid = [1, 1, 1, 4]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
cfg_file = cfg_path_template.format(n_cfg=n_cfg)

# Observable: 3x4 rectangle in (X,T) plane
# Path explicitly expanded: [X,X,X,T,T,T,T,-X,-X,-X,-T,-T,-T,-T]

# QUDA / PyQUDA runtime
backend = "cupy"
enable_tuning = True

# -----------------------------
# 2) Initialize QUDA + read gauge configuration
# -----------------------------
core.init(process_grid, latt_size=latt_size, backend=backend, resource_path=resource_path, enable_tuning=enable_tuning)

gauge = io.readChromaQIOGauge(cfg_file)

# -----------------------------
# 3) Wilson loop evaluation using gauge.loop(...)
# -----------------------------
res = gauge.loop(
    [
        [[X, X, X, T, T, T, T, -X, -X, -X, -T, -T, -T, -T]],
        [[X, X, X, T, T, T, T, -X, -X, -X, -T, -T, -T, -T]],
        [[X, X, X, T, T, T, T, -X, -X, -X, -T, -T, -T, -T]],
        [[X, X, X, T, T, T, T, -X, -X, -X, -T, -T, -T, -T]],
    ],
    [1, 1, 1, 1],
)

Nc = gauge.latt_info.Nc

for i, obj in enumerate(res):
    U = obj.getHost()
    U = U.reshape(-1, Nc, Nc)

    tr = np.trace(U, axis1=-2, axis2=-1)
    local_sum = tr.real.sum()
    local_n = U.shape[0]

    comm = core.getMPIComm()
    global_sum = comm.allreduce(float(local_sum))
    global_n = comm.allreduce(int(local_n))

    global_mean = global_sum / (global_n * Nc)

    if core.getMPIRank() == 0:
        print(f"preset = {ensemble_preset}")
        print(f"cfg = {n_cfg}")
        print(f"cfg_file = {cfg_file}")
        print(f"loop = {i} (wl_rect_3x4_xt)")
        print(f"global_sum = {global_sum}")
        print(f"global_n = {global_n}")
        print(f"global_mean = {global_mean}")
