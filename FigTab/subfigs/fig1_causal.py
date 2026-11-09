"""
Appendix Fig. 1. Causal (u,v) diagram of the matching prescription:
regular centre v=u, trapping horizon r=2m, spacelike singular boundary,
and the matched null curve gamma_{lambda*} with the anchor point at
lambda* = 0.3.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Appendix Fig. 1: numerical causal diagram of the matching prescription."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '.')
import os
DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent))

import Yue_Du as YD

MF = 1.17088762
LAM = 0.3

p = DATA / "zoom_base_h5e4b_v2"
res = YD.load_solution(p / "single_solution.npz", p / "single_diagnostics.json",
                       p / "single_horizon.npz")
h = res.horizon
lamv = h.m / MF
be = h.ell / MF
tr = YD.horizon_transversality(lamv, be)
res.horizon = YD.HorizonData(h.u, h.v, h.r, h.m, h.alpha, h.ell, lamv, tr)
res.mf = MF
c = YD.extract_curve(res, LAM)
ua, va, _, _ = YD.anchor_from_lambda(res, LAM)

us, vs = [], []
for i in range(res.u.size):
    js = np.where(res.valid[i])[0]
    if js.size < 200:
        continue
    j1 = js[-1]
    if res.v[j1] >= res.v[-1] - 1e-9:
        continue
    if res.r[i, j1] < 0.05:
        us.append(float(res.u[i]))
        vs.append(float(res.v[j1]))

keep = c.r > 1e-6
# the matched curve must stop where it meets the spacelike singularity
if len(us) > 5:
    us_a = np.asarray(us)
    vs_a = np.asarray(vs)
    order = np.argsort(vs_a)
    u_sing = float(np.interp(va, vs_a[order], us_a[order]))
    keep = keep & (c.u < u_sing)
    print(f"matched curve clipped at u < {u_sing:.3f}")

fig, ax = plt.subplots(figsize=(7.0, 6.0))
ax.plot(res.u, res.u, "k-", lw=1.2, label="regular centre  $v=u$")
ax.plot(h.u, h.v, "-", color="tab:blue", lw=2.2, label=r"trapping horizon  $r=2m$")
ax.plot(us, vs, "-", color="tab:red", lw=2.2, label="spacelike singular boundary")
ax.plot(c.u[keep], c.v[keep], "-", color="tab:green", lw=1.8,
        label=r"matched null curve $\gamma_{\lambda_*}$")
ax.plot([ua], [va], "o", ms=11, mfc="white", mec="tab:purple", mew=2.2,
        label=fr"horizon sphere $S_{{\lambda_*}}$ ($\lambda_*={LAM}$)")
ax.set_xlabel(r"$u$")
ax.set_ylabel(r"$v$")
ax.set_xlim(1.20, 3.25)
ax.set_ylim(2.55, 5.0)
ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
ax.grid(alpha=0.25)
fig.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.10)
fig.savefig("paper_fig1_causal_diagram.png", dpi=220)
fig.savefig("paper_fig1_causal_diagram.pdf")
print("saved paper_fig1_causal_diagram.png")
print(f"anchor u={ua:.3f} v={va:.3f}; singular points={len(us)}; "
      f"curve u range [{c.u[keep].min():.3f},{c.u[keep].max():.3f}]")
