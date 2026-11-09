"""
Fig. 2. Identification of the resolved Kasner window on the matched
curve: plots C_psi^2, C_m^2, curvature slope and the analytic prediction
6 + 4 C_psi^2, and marks the rho-interval where the Kasner constraints
are satisfied.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Fig. 2: identification of the resolved Kasner window."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 9,
    "savefig.dpi": 220, "lines.linewidth": 1.8, "lines.markersize": 5,
    "axes.grid": True, "grid.alpha": 0.3,
})

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '.')
import os
DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent))

import Yue_Du as YD

MF = 1.17088762
LAM = 0.3
LO, HI = 0.04, 0.08          # analysis window

p = DATA / "zoom_base_h5e4b_v2"
res = YD.load_solution(p / "single_solution.npz", p / "single_diagnostics.json",
                       p / "single_horizon.npz")
h = res.horizon
lamv = h.m / MF
be = h.ell / MF
tr = YD.horizon_transversality(lamv, be)
res.horizon = YD.HorizonData(h.u, h.v, h.r, h.m, h.alpha, h.ell, lamv, tr)
res.mf = MF
c = YD.extract_curve(res, LAM, savgol_window=7)
sel = c.rho >= 0.03
cc = YD.CurveData(**{k: (getattr(c, k)[sel] if getattr(c, k).ndim == 1
                         and getattr(c, k).shape[0] == sel.size else getattr(c, k))
                     for k in c.__dataclass_fields__})
c_direct = YD.extract_curve(res, LAM, savgol_window=None)
x = np.log(1.0 / cc.rho)
xd = np.log(1.0 / c_direct.rho[c_direct.rho >= 0.03])
xw = [np.log(1.0 / HI), np.log(1.0 / LO)]

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.9))
for a in ax:
    a.axvspan(xw[0], xw[1], color="0.85", alpha=0.6, zorder=0,
              label="analysis window")

ax[0].plot(x, cc.cpsi ** 2, "k-", lw=2.0, label=r"$C_\Psi^2$ (Savitzky fit)")
ax[0].plot(x, np.maximum(cc.cm2, 0.0), "--", color="tab:orange", lw=1.6,
           label=r"$C_m^2$ (Savitzky fit)")
ax[0].plot(xd, c_direct.cpsi_direct[c_direct.rho >= 0.03] ** 2, ":",
           color="tab:blue", lw=1.0, label=r"$C_\Psi^2$ (pointwise)")
ax[0].set_xlabel(r"$\ln(1/\rho)$")
ax[0].set_ylabel("Kasner parameter squared")
ax[0].set_title(r"(a) independent extraction of $C$")
ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_ylim(0, 0.62)

ax[1].plot(x, cc.curvature_slope, "k-", lw=2.0,
           label=r"$-\mathrm{d}\ln K/\mathrm{d}\ln\rho$")
ax[1].plot(x, 6.0 + 4.0 * cc.cpsi ** 2, "--", color="tab:red", lw=1.6,
           label=r"$6+4C^2$")
ax[1].set_xlabel(r"$\ln(1/\rho)$")
ax[1].set_ylabel("curvature exponent")
ax[1].set_title("(b) measured exponent vs Kasner prediction")
ax[1].legend(fontsize=8, loc="upper left")

fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.13, wspace=0.26)
fig.savefig("paper_app_fig2_kasner.png", dpi=220)
fig.savefig("paper_app_fig2_kasner.pdf")
print("saved paper_app_fig2_kasner.png/.pdf")
print("C_psi^2 at window:", float(np.nanmedian(cc.cpsi[
    (cc.rho > LO) & (cc.rho < HI)]) ** 2))
print("C_m^2   at window:", float(np.nanmedian(cc.cm2[
    (cc.rho > LO) & (cc.rho < HI)])))
