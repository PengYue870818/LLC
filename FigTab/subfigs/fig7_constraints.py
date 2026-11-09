"""
Fig. 7. Hamiltonian / momentum constraint residuals along the matched
curve (or in the strong-collapse runs) for the two zoom pairs
h = 1e-3 and h = 5e-4, demonstrating the residual level relative to the
resolved scale.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
""""""
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
RUNS = [("1e-3", "zoom_base_h1e3_v2", "single"),
        ("5e-4", "zoom_base_h5e4b_v2", "single")]


def residuals_along_curve(d, tag):
    p = DATA / d
    r = YD.load_solution(p / f"{tag}_solution.npz", p / f"{tag}_diagnostics.json",
                         p / f"{tag}_horizon.npz")
    r.mf = MF
    h = r.horizon
    lamv = h.m / MF
    be = h.ell / MF
    tr = YD.horizon_transversality(lamv, be)
    r.horizon = YD.HorizonData(h.u, h.v, h.r, h.m, h.alpha, h.ell, lamv, tr)
    c = YD.extract_curve(r, LAM)
    der = YD.compute_unified_derivatives(r)
    a = np.exp(r.logalpha)
    m = 0.5 * r.r * (1.0 + 4.0 * der["ru"] * der["rv"] / a ** 2)
    du = float(r.u[1] - r.u[0])
    dv = float(r.v[1] - r.v[0])
    m_u = np.gradient(m, du, axis=0)
    m_v = np.gradient(m, dv, axis=1)
    a2 = a ** 2
    cu = (m_u + (r.r ** 2 / a2) * der["rv"] * der["pu"] ** 2
          - (2.0 * r.r * der["rv"] / a2)
          * (der["ruu"] - 2.0 * (der["au"] / a) * der["ru"]
             + 0.5 * r.r * der["pu"] ** 2))
    cv = (m_v + (r.r ** 2 / a2) * der["ru"] * der["pv"] ** 2
          - (2.0 * r.r * der["ru"] / a2)
          * (der["rvv"] - 2.0 * (der["av"] / a) * der["rv"]
             + 0.5 * r.r * der["pv"] ** 2))
    iu = np.clip(np.round((c.u - r.u[0]) / du).astype(int), 0, r.u.size - 1)
    jv = int(np.clip(np.round((c.v[0] - r.v[0]) / dv), 0, r.v.size - 1))
    return c, cu[iu, jv], cv[iu, jv]


fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.8))
for label, d, tag in RUNS:
    c, cu, cv = residuals_along_curve(d, tag)
    sel = (c.rho > 0.02) & (c.rho < 0.6)
    good = np.isfinite(cu) & np.isfinite(cv) & np.isfinite(c.m)
    good &= (np.abs(cu) < 1e3) & (np.abs(cv) < 1e3) & (np.abs(c.m) < 1e3)
    sel &= good
    ax[0].semilogy(np.log(1.0 / c.rho[sel]), np.abs(cu[sel]), lw=1.2,
                   label=f"$h={label}$")
    ax[1].semilogy(np.log(1.0 / c.rho[sel]), np.abs(cv[sel]), lw=1.2,
                   label=f"$h={label}$")
    print(label, "median |Cu|", float(np.nanmedian(np.abs(cu[sel]))),
          "median |Cv|", float(np.nanmedian(np.abs(cv[sel]))),
          "m median", float(np.nanmedian(c.m[sel])))

ax[0].set_xlabel(r"$\ln(1/\rho)$")
ax[0].set_ylabel(r"$|C_u|$")
ax[0].set_title("(a) u-constraint along the curve")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[1].set_xlabel(r"$\ln(1/\rho)$")
ax[1].set_ylabel(r"$|C_v|$")
ax[1].set_title("(b) v-constraint along the curve")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.12, wspace=0.28)
fig.savefig("paper_app_fig7_constraints.png", dpi=220)
fig.savefig("paper_app_fig7_constraints.pdf")
print("saved paper_app_fig7_constraints.png")
