"""
Appendix Figs. 3, 4, 5. Finite-data conditioning test on the equal-mass
base/perturbed pairs: the raw quotient Q_K = |ln K1 - ln K0| / d_X, the
signed slope versus |ln rho| with the linear fit and the analytic
prediction, and the nested-window slope stability across three grid
resolutions.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Appendix Figs. 3, 4, 5 from the saved equal-mass pairs."""
import gc
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
DX = {0.5: 0.5375, 1.0: 1.076743307270465, 2.0: 2.1916917906310833,
      -1.0: 1.0697}

PAIRS = {
    ("1e-3", 1.0): ("zoom_base_h1e3_v2", "zoom_pert_h1e3_v2"),
    ("1e-3", -1.0): ("zoom_base_h1e3_v2", "zoom_eps-1_pert_v3"),
    ("1e-3", 0.5): ("zoom_bump_eps0.5_base_v3", "zoom_eps0.5_pert_v3"),
    ("1e-3", 2.0): ("zoom_bump_eps0.5_base_v3", "zoom_eps2_pert_v3"),
    ("5e-4", 1.0): ("zoom_base_h5e4b_v2", "zoom_pert_h5e4b_v2"),
    ("5e-4", -1.0): ("zoom_base_h5e4b_v2", "z5_eps-1_v2"),
    ("5e-4", 0.5): ("zoom_bump_eps0.5_h5e4_v2", "z5_eps0.5_v2"),
    ("5e-4", 2.0): ("zoom_bump_eps0.5_h5e4_v2", "z5_eps2_v2"),
    ("2.5e-4", 1.0): ("h4_v5_x2", "h4_v5_x2_pert"),
    ("2.5e-4", -1.0): ("h4_v5_x2", "h25_eps-1"),
    ("2.5e-4", 0.5): ("zoom_bump_eps0.5_h25e5_v2", "h25_eps0.5"),
    ("2.5e-4", 2.0): ("zoom_bump_eps0.5_h25e5_v2", "h25_eps2"),
}


def curve(d):
    r = YD.load_solution(DATA / d / "single_solution.npz", DATA / d / "single_diagnostics.json",
                         DATA / d / "single_horizon.npz")
    r.mf = MF
    h = r.horizon
    lamv = h.m / MF
    be = h.ell / MF
    tr = YD.horizon_transversality(lamv, be)
    r.horizon = YD.HorizonData(h.u, h.v, h.r, h.m, h.alpha, h.ell, lamv, tr)
    c = YD.extract_curve(r, LAM, savgol_window=7)
    del r
    gc.collect()
    return c


cache = {}
rows = []
for (htag, eps), (bt, pt) in PAIRS.items():
    if not all((DATA / t / "single_solution.npz").exists() for t in (bt, pt)):
        print(f"skip h={htag} eps={eps}: missing run")
        continue
    for tag in (bt, pt):
        if tag not in cache:
            cache[tag] = curve(tag)
    cb, cp = cache[bt], cache[pt]
    mm = YD.match_curves_equal_rho(cp, cb, 0.04, 0.08, n=160)
    x = np.abs(np.log(mm["rho"]))
    y = (mm["logk1"] - mm["logk0"]) / DX[eps]
    f = YD.linear_fit(x, y)
    c1 = float(np.nanmedian(YD.interp_curve_field(cp, "cpsi", mm["rho"])))
    c0 = float(np.nanmedian(YD.interp_curve_field(cb, "cpsi", mm["rho"])))
    pred = 4.0 * (c1 ** 2 - c0 ** 2) / DX[eps]
    rows.append(dict(h=htag, eps=eps, x=x, y=y, pred=pred, meas=f["slope"],
                     sigma=f["slope_sigma"], r2=f["r2"],
                     ratio=f["slope"] / pred if pred else float("nan")))
    print(f"h={htag} eps={eps:+.1f}: pred={pred:+.5f} meas={f['slope']:+.5f} "
          f"r2={f['r2']:.3f} ratio={f['slope']/pred:+.3f}")

# ------------------------------------------------------------------ Fig. 3
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6),
                       gridspec_kw={"width_ratios": [1.15, 1.0]})
cols = {"1e-3": "#1f77b4", "5e-4": "#d62728", "2.5e-4": "#2ca02c"}
sel = [r for r in rows if r["eps"] == 1.0]
common_slope = float(np.mean([r["meas"] for r in sel]))
for r in sel:
    ax[0].plot(r["x"], r["y"], ".", ms=3.5, color=cols[r["h"]],
               label=fr"$h={r['h']}$  (slope {r['meas']:.4f})")
    ax[0].plot(r["x"], common_slope * r["x"] +
               (np.mean(r["y"]) - common_slope * np.mean(r["x"])),
               "-", color=cols[r["h"]], lw=1.1)
    ax[1].plot(r["x"], r["y"] - (common_slope * r["x"] +
               (np.mean(r["y"]) - common_slope * np.mean(r["x"]))),
               "-", color=cols[r["h"]], lw=1.6)
ax[0].set_xlabel(r"$\ln(1/\rho)$")
ax[0].set_ylabel(r"$Q_K(\rho)=\Delta\ln\hat{K}/d_X$")
ax[0].set_title("(a) finite-data conditioning quotient")
ax[0].legend(fontsize=9, loc="upper left")
ax[0].grid(alpha=0.3)
ax[1].axhline(0.0, color="k", lw=1.0, ls=":")
ax[1].set_xlabel(r"$\ln(1/\rho)$")
ax[1].set_ylabel("residual from the common linear fit")
ax[1].set_title("(b) residuals (offset by the mean)")
ax[1].legend([fr"$h={r['h']}$" for r in sel], fontsize=9, loc="upper left")
ax[1].grid(alpha=0.3)
fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.13, wspace=0.28)
fig.savefig("paper_app_fig3_QK.png", dpi=220)
fig.savefig("paper_app_fig3_QK.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Fig. 4
fig, ax = plt.subplots(figsize=(6.6, 6.0))
mark = {0.5: "s", 1.0: "o", 2.0: "^", -1.0: "v"}
for r in rows:
    ax.errorbar(r["pred"], r["meas"], yerr=[2.0 * r["sigma"]], fmt=mark[r["eps"]],
                ms=8, color=cols[r["h"]], capsize=3, lw=1.2,
                mfc="none" if r["h"] != "1e-3" else cols[r["h"]])
lo = min(min(r["pred"], r["meas"]) for r in rows)
hi = max(max(r["pred"], r["meas"]) for r in rows)
ax.plot([lo * 1.05, hi * 1.05], [lo * 1.05, hi * 1.05], "k--", lw=1.2)
ax.set_xlabel(r"$4(C_1^2-C_0^2)/d_X$  (prediction)")
ax.set_ylabel(r"measured signed slope $s_\epsilon^{\rm num}$")
ax.set_xlim(lo * 1.06, hi * 1.06)
ax.set_ylim(lo * 1.06, hi * 1.06)
ax.grid(alpha=0.3)
ax.text(0.03, 0.97, "prediction = measurement", transform=ax.transAxes,
        fontsize=9, va="top")
ax.text(0.03, 0.90, r"$s_\epsilon/s_\epsilon^{\rm pred}=0.98$--$1.03$",
        transform=ax.transAxes, fontsize=9, va="top")

# inset: the positive cluster
axin = ax.inset_axes([0.54, 0.08, 0.42, 0.34])
pos = [r for r in rows if r["pred"] > 0]
for r in pos:
    axin.errorbar(r["pred"], r["meas"], yerr=[2.0 * r["sigma"]], fmt=mark[r["eps"]],
                  ms=7, color=cols[r["h"]], capsize=2, lw=1.0,
                  mfc="none" if r["h"] != "1e-3" else cols[r["h"]])
plo = min(min(r["pred"], r["meas"]) for r in pos)
phi = max(max(r["pred"], r["meas"]) for r in pos)
axin.plot([plo * 0.995, phi * 1.005], [plo * 0.995, phi * 1.005], "k--", lw=1.0)
axin.set_xlim(plo * 0.995, phi * 1.005)
axin.set_ylim(plo * 0.995, phi * 1.005)
axin.tick_params(labelsize=7)
axin.grid(alpha=0.3)


# compact legend: colour = resolution, marker = epsilon
handles = [Line2D([], [], color=cols[h], marker="o", ls="-", ms=7,
                  label=fr"$h={h}$") for h in ["1e-3", "5e-4", "2.5e-4"]]
handles += [Line2D([], [], color="0.35", marker=mark[e], ls="none", ms=7,
                   label=fr"$\epsilon={e:g}$") for e in [-1.0, 0.5, 1.0, 2.0]]
ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2,
          bbox_to_anchor=(0.0, 0.86))
fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.11)
fig.savefig("paper_app_fig4_scatter.png", dpi=220)
fig.savefig("paper_app_fig4_scatter.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Fig. 5
fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.9))
linestyles = {-1.0: ":", 0.5: "--", 1.0: "-", 2.0: "-."}
for r in rows:                      # all twelve (h, epsilon) pairs
    ax[0].plot(r["x"], np.abs(r["y"]), ls=linestyles[r["eps"]], lw=1.2,
               color=cols[r["h"]], alpha=0.9)
ax[0].set_xlabel(r"$\ln(1/\rho)$")
ax[0].set_ylabel(r"$|Q_K(\rho)|$")
ax[0].set_title("(a) all $(h,\\epsilon)$ pairs collapse")

# two keys: colour = resolution, line style = perturbation amplitude
key_h = [Line2D([], [], color=cols[h], lw=1.8, label=fr"$h={h}$")
         for h in ["1e-3", "5e-4", "2.5e-4"]]
leg1 = ax[0].legend(handles=key_h, fontsize=7, loc="upper left",
                    title="resolution", title_fontsize=7, framealpha=0.95)
ax[0].add_artist(leg1)
key_e = [Line2D([], [], color="0.25", lw=1.6, ls=linestyles[e],
                label=fr"$\epsilon={e:g}$") for e in [-1.0, 0.5, 1.0, 2.0]]
ax[0].legend(handles=key_e, fontsize=7, loc="lower right",
             title=r"amplitude", title_fontsize=7, framealpha=0.95)
ax[0].grid(alpha=0.3)

for h in ["1e-3", "5e-4", "2.5e-4"]:
    sub = sorted([r for r in rows if r["h"] == h], key=lambda r: r["eps"])
    ax[1].errorbar([r["eps"] for r in sub], [r["ratio"] for r in sub],
                   yerr=[r["sigma"] / abs(r["pred"]) for r in sub],
                   fmt="o-", capsize=3, label=fr"$h={h}$")
ax[1].axhline(1.0, color="k", ls="--", lw=1.0)
ax[1].set_xlabel(r"$\epsilon$")
ax[1].set_ylabel("measured / predicted")
ax[1].set_title(r"(b) plateau in $(\epsilon,h)$")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.subplots_adjust(left=0.1, right=0.99, top=0.94, bottom=0.12, wspace=0.25)
fig.savefig("paper_app_fig5_hvseps.png", dpi=220)
fig.savefig("paper_app_fig5_hvseps.pdf")
plt.close(fig)

print("saved paper_app_fig3_QK.png, paper_app_fig4_scatter.png, paper_app_fig5_hvseps.png")
