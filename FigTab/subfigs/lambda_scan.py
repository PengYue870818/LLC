"""
Gauge-robustness test: recomputes s_eps while sweeping the matching
parameter lambda_* over 0.20-0.35 (the anchor scan of Sec. 11.5), to
show that the measured slope does not depend on where the anchor S is
chosen on the horizon.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Gauge-robustness test: does the Q_K slope depend on the matching parameter lambda_*?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '.')
import os
DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent))

import Yue_Du as YD

MF = 1.17088762
DX = 1.076743307270465
BASE = "zoom_base_h5e4b_v2"
PERT = "zoom_pert_h5e4b_v2"


def load(d):
    r = YD.load_solution(DATA / d / "single_solution.npz", DATA / d / "single_diagnostics.json",
                         DATA / d / "single_horizon.npz")
    r.mf = MF
    h = r.horizon
    lamv = h.m / MF
    be = h.ell / MF
    tr = YD.horizon_transversality(lamv, be)
    r.horizon = YD.HorizonData(h.u, h.v, h.r, h.m, h.alpha, h.ell, lamv, tr)
    return r


rb = load(BASE)
rp = load(PERT)
print(f"{'lam*':>5} {'n':>5} {'pred':>10} {'meas':>10} {'ratio':>7} {'r2':>6}")
for lam in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
    try:
        cb = YD.extract_curve(rb, lam, savgol_window=7)
        cp = YD.extract_curve(rp, lam, savgol_window=7)
        mm = YD.match_curves_equal_rho(cp, cb, 0.04, 0.08, n=160)
        x = np.abs(np.log(mm["rho"]))
        y = (mm["logk1"] - mm["logk0"]) / DX
        f = YD.linear_fit(x, y)
        c1 = float(np.nanmedian(YD.interp_curve_field(cp, "cpsi", mm["rho"])))
        c0 = float(np.nanmedian(YD.interp_curve_field(cb, "cpsi", mm["rho"])))
        pred = 4.0 * (c1 ** 2 - c0 ** 2) / DX
        print(f"{lam:5.2f} {x.size:5d} {pred:10.5f} {f['slope']:10.5f} "
              f"{f['slope']/pred:7.3f} {f['r2']:6.3f}")
    except Exception as exc:
        print(f"{lam:5.2f}  failed: {exc}")
