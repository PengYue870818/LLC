"""
Error budget: how much the resolution change, smoothing-window choice,
and mass mismatch contribute to the uncertainty on the measured slope
s_eps. Prints a per-component breakdown used in the paper's error bar.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Error budget: how much do resolution and constraint residuals affect ln K and the slope?"""
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


rho_grid = np.linspace(0.04, 0.08, 60)
logk = {}
slopes = {}
for tag, (b, pt) in PAIRS.items():
    cb, cp = curve(b), curve(pt)
    mm = YD.match_curves_equal_rho(cp, cb, 0.04, 0.08, n=160)
    x = np.abs(np.log(mm["rho"]))
    y = (mm["logk1"] - mm["logk0"]) / DX
    f = YD.linear_fit(x, y)
    slopes[tag] = f["slope"]
    # ln K of the reference solution on the common rho grid
    k0 = YD.interp_curve_field(cb, "logk", rho_grid)
    logk[tag] = k0
    print(f"{tag}: slope={f['slope']:.6f}  sigma={f.get('slope_sigma', float('nan')):.2e}")

tags = list(logk)
ref = logk["2.5e-4"]
for tag in tags:
    d = logk[tag] - ref
    print(f"lnK({tag}) - lnK(2.5e-4): max |d| = {np.nanmax(np.abs(d)):.3e}, "
          f"rms = {np.nanstd(d):.3e}")

spread = max(slopes.values()) - min(slopes.values())
signal = slopes["2.5e-4"] * (np.log(1 / 0.04) - np.log(1 / 0.08))
print()
print(f"slope spread across resolutions : {spread:.3e}")
print(f"mean slope                       : {np.mean(list(slopes.values())):.6f}")
print(f"relative spread                  : {spread/np.mean(list(slopes.values())):.3%}")
print(f"Q_K change over the window       : {signal:.3e}")
print(f"spread / signal                  : {spread/signal:.2%}")
