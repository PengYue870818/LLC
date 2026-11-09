"""
Main results table: for each epsilon in {-1, 0.5, 1, 2} and each
resolution pair, lists the compensating amplitude a(eps), the H^2
distance d_X, the extracted C_psi values, the measured slope s_eps,
the analytic prediction 4 (C_1^2 - C_0^2)/d_X, and their ratio.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Collect the main parameter table: epsilon, a(eps), d_X, C values, slopes, ratios."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '.')
import os
DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent))

import Yue_Du as YD

MF = 1.17088762
W = 7
H = "5e-4"

BASE = {"1e-3": "zoom_base_h5e4b_v2", "x": "zoom_base_h5e4b_v2"}
PAIRS = {
    -1.0: ("zoom_base_h5e4b_v2", "z5_eps-1_v2", "scan_eps-1_v2"),
    0.5: ("zoom_bump_eps0.5_h5e4_v2", "z5_eps0.5_v2", "scan_eps0.5_v2"),
    1.0: ("zoom_base_h5e4b_v2", "zoom_pert_h5e4b_v2", "eqzoom_v2"),
    2.0: ("zoom_bump_eps0.5_h5e4_v2", "z5_eps2_v2", "scan_eps2_v2"),
}
DX = {-1.0: 1.0697, 0.5: 0.5375, 1.0: 1.076743307270465, 2.0: 2.1916917906310833}
MF_REF = 1.1707698893511003


def curve(d):
    p = DATA / d
    r = YD.load_solution(p / "single_solution.npz", p / "single_diagnostics.json",
                         p / "single_horizon.npz")
    r.mf = MF
    h = r.horizon
    lv = h.m / MF
    r.horizon = YD.HorizonData(h.u, h.v, h.r, h.m, h.alpha, h.ell, lv,
                               YD.horizon_transversality(lv, h.ell / MF))
    return YD.extract_curve(r, 0.3, savgol_window=W)


def coarse_mf(d, tag):
    j = json.load(open(DATA / d / f"{tag}_diagnostics.json", encoding="utf-8"))
    return float(j.get("mf_ah", float("nan")))


rows = []
for eps, (b, pt, coarse) in PAIRS.items():
    cb, cp = curve(b), curve(pt)
    mm = YD.match_curves_equal_rho(cp, cb, 0.04, 0.08, n=160)
    x = np.abs(np.log(mm["rho"]))
    y = (mm["logk1"] - mm["logk0"]) / DX[eps]
    f = YD.linear_fit(x, y)
    c1 = float(np.nanmedian(YD.interp_curve_field(cp, "cpsi", mm["rho"])))
    c0 = float(np.nanmedian(YD.interp_curve_field(cb, "cpsi", mm["rho"])))
    pred = 4.0 * (c1 ** 2 - c0 ** 2) / DX[eps]
    amp = json.load(open(DATA / coarse / "pert_coarse_provenance.json",
                         encoding="utf-8"))["config"]["initial"]["amplitude"]
    mf_p = coarse_mf(coarse, "pert_coarse")
    rows.append(dict(eps=eps, amp=amp, dX=DX[eps], C0=c0, C1=c1,
                     dC2=c1 ** 2 - c0 ** 2, pred=pred, meas=f["slope"],
                     ratio=f["slope"] / pred, dm=(mf_p - MF_REF) / MF_REF))
    print(f"eps={eps:+.1f} a={amp:.7f} dX={DX[eps]:.4f} C0={c0:+.5f} C1={c1:+.5f} "
          f"pred={pred:+.6f} meas={f['slope']:+.6f} ratio={f['slope']/pred:.4f} "
          f"dm={rows[-1]['dm']:+.2e}")

json.dump(rows, open("main_table.json", "w"), indent=1)
print("saved main_table.json")
