"""
Fig. 8. Equal-final-mass construction: (a) Misner-Sharp mass m along the
trapping horizon for the base and four epsilon-perturbed runs; (b) the
difference from the base, showing the equal-mass matching to within
5e-4.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Fig. 8: equal final mass. (a) mass along the horizon; (b) difference from the base."""
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

RUNS = [("base", "eqzoom_v2", "base_coarse"),
        (r"$\epsilon=-1$", "scan_eps-1_v2", "pert_coarse"),
        (r"$\epsilon=0.5$", "scan_eps0.5_v2", "pert_coarse"),
        (r"$\epsilon=1$", "eqzoom_v2", "pert_coarse"),
        (r"$\epsilon=2$", "scan_eps2_v2", "pert_coarse")]

data = {}
for label, d, tag in RUNS:
    p = DATA / d
    r = YD.load_solution(p / f"{tag}_solution.npz", p / f"{tag}_diagnostics.json",
                         p / f"{tag}_horizon.npz")
    h = r.horizon
    good = np.isfinite(h.m) & (h.m > 0) & (h.m < 5)
    data[label] = (h.u[good], h.m[good])
    print(label, "M_f", float(np.nanmax(h.m[good])), "n", int(good.sum()))
    del r

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.6))

for label, (u, m) in data.items():
    if label == "base":
        ax[0].plot(u, m, "-", lw=2.8, color="k", alpha=0.30, label="base")
    else:
        ax[0].plot(u, m, "--", lw=1.4, label=label)
ax[0].set_xlabel(r"$u$")
ax[0].set_ylabel(r"$M$ on the horizon")
ax[0].set_title("(a) mass along the trapping horizon")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_ylim(0, 1.25)

lam0, m0 = None, None
for label, (u, m) in data.items():
    lam = m / np.nanmax(m)
    order = np.argsort(lam)
    lam, m = lam[order], m[order]
    if label == "base":
        lam0, m0 = lam, m
        continue
    common = (lam0 > lam.min()) & (lam0 < lam.max())
    rel = (np.interp(lam0[common], lam, m) - m0[common]) / m0[common]
    ax[1].semilogy(lam0[common], np.abs(rel), lw=1.4, label=label)
ax[1].set_xlabel(r"$\lambda = m/M_{\mathrm{f}}$")
ax[1].set_ylabel(r"$|M_\epsilon - M_0| / M_0$")
ax[1].set_title("(b) relative deviation from the base")
ax[1].legend(fontsize=8, loc="upper left")

fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.13, wspace=0.30)
fig.savefig("paper_app_fig8_mass.png", dpi=220)
fig.savefig("paper_app_fig8_mass.pdf")
print("saved paper_app_fig8_mass.png/.pdf")
