"""
Fig. 6. Initial-data profiles: the scalar pulse Psi(r) for the base run
and the four epsilon perturbations (-1, 0.5, 1, 2) used throughout the
paper.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""


#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Fig. 6: initial-data profiles for the base and the four epsilon perturbations."""
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

RUNS = [
    ("base", "eqzoom_v2", "base_coarse"),
    (r"$\epsilon=-1$", "scan_eps-1_v2", "pert_coarse"),
    (r"$\epsilon=0.5$", "scan_eps0.5_v2", "pert_coarse"),
    (r"$\epsilon=1$", "eqzoom_v2", "pert_coarse"),
    (r"$\epsilon=2$", "scan_eps2_v2", "pert_coarse"),
]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.7))
profiles = {}
for label, d, tag in RUNS:
    p = DATA / d
    r = YD.load_solution(p / f"{tag}_solution.npz", p / f"{tag}_diagnostics.json",
                         p / f"{tag}_horizon.npz")
    i = 0
    js = np.where(r.valid[i])[0]
    rr = r.r[i, js]
    pp = r.psi[i, js]
    ax[0].plot(rr, pp, lw=1.4, label=label)
    ax[1].plot(r.v[js], pp, lw=1.4, label=label)
    profiles[label] = (rr.copy(), pp.copy())
    del r

rr0, pp0 = profiles["base"]
for label, (rr, pp) in profiles.items():
    if label == "base":
        continue
    d = np.interp(rr0, rr, pp) - pp0
    ax[2].plot(rr0, d, lw=1.4, label=label)
ax[2].set_xlabel(r"$r(0,v)$")
ax[2].set_ylabel(r"$\Psi_\epsilon-\Psi_0$")
ax[2].set_title("(c) perturbation profile")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)
ax[2].set_xlim(1.0, 3.5)

ax[0].set_xlabel(r"$r(0,v)$")
ax[0].set_ylabel(r"$\Psi(0,v)$")
ax[0].set_title("(a) initial data in areal radius")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[0].set_xlim(0, 4)
r_end = float(rr0.max())
ax[0].axvline(r_end, color="0.5", ls=":", lw=1.2)
ax[0].annotate(r"$r_v\to0$", xy=(r_end, 0.45), xytext=(r_end + 0.15, 0.45),
               fontsize=10, color="0.35",
               arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
ax[1].set_xlabel(r"$v$")
ax[1].set_ylabel(r"$\Psi(0,v)$")
ax[1].set_title("(b) initial data on the null slice")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
ax[1].set_xlim(2.5, 10)
fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.14, wspace=0.36)
fig.savefig("paper_app_fig6_initial_data.png", dpi=220)
fig.savefig("paper_app_fig6_initial_data.pdf")
print("saved paper_app_fig6_initial_data.png")
