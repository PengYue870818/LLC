"""
Independent extraction path: predicts the slope from C_m instead of
C_psi, as a consistency check on the measured s_eps across the three
resolution pairs (h = 1e-3, 5e-4, 2.5e-4).

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Predict the slope from C_m instead of C_psi (independent extraction path)."""
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
PAIRS = {
    "1e-3": ("zoom_base_h1e3_v2", "zoom_pert_h1e3_v2"),
    "5e-4": ("zoom_base_h5e4b_v2", "zoom_pert_h5e4b_v2"),
    "2.5e-4": ("h4_v5_x2", "h4_v5_x2_pert"),
}


def curve(d):
    p = DATA / d
    r = YD.load_solution(p / "single_solution.npz", p / "single_diagnostics.json",
                         p / "single_horizon.npz")
    r.mf = MF
    h = r.horizon
    lv = h.m / MF
    r.horizon = YD.HorizonData(h.u, h.v, h.r, h.m, h.alpha, h.ell, lv,
                               YD.horizon_transversality(lv, h.ell / MF))
    return YD.extract_curve(r, 0.3, savgol_window=7)


print(f"{'h':>7} {'field':>6} {'C0':>9} {'C1':>9} {'pred':>11} {'meas':>11} {'ratio':>7}")
for tag, (b, pt) in PAIRS.items():
    cb, cp = curve(b), curve(pt)
    mm = YD.match_curves_equal_rho(cp, cb, 0.04, 0.08, n=160)
    x = np.abs(np.log(mm["rho"]))
    y = (mm["logk1"] - mm["logk0"]) / DX
    f = YD.linear_fit(x, y)
    for field in ("cpsi", "cm2"):
        c1 = float(np.nanmedian(YD.interp_curve_field(cp, field, mm["rho"])))
        c0 = float(np.nanmedian(YD.interp_curve_field(cb, field, mm["rho"])))
        if field == "cm2":
            c1, c0 = np.sqrt(max(c1, 0.0)), np.sqrt(max(c0, 0.0))
        pred = 4.0 * (c1 ** 2 - c0 ** 2) / DX
        print(f"{tag:>7} {field:>6} {c0:9.5f} {c1:9.5f} {pred:11.6f} "
              f"{f['slope']:11.6f} {f['slope']/pred:7.4f}")
