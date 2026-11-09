"""
Three-region constraint residuals for the strong-collapse runs
(full_small_v2 and full_small_v2_h005): separates the matched-curve
region, the outer region, and the near-centre region, and reports the
L2 / Linf residuals of the Hamiltonian constraint in each.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Three-region constraint residuals for the strong-collapse runs (fixed scheme)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '.')
import os
DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent))

import Yue_Du as YD

RUNS = [("full_small_v2", 0.01), ("full_small_v2_h005", 0.005)]
vals = {}
for d, h in RUNS:
    p = DATA / d
    res = YD.load_solution(p / "base_coarse_solution.npz", p / "base_coarse_diagnostics.json",
                           p / "base_coarse_horizon.npz")
    der = YD.compute_unified_derivatives(res)
    inv = YD.invariant_arrays(res, der)
    r = res.r
    ru, rv, pu, pv = der["ru"], der["rv"], der["pu"], der["pv"]
    au, av, ruu, rvv = der["au"], der["av"], der["ruu"], der["rvv"]
    cu = ruu - 2.0 * au * ru + 0.5 * r * pu * pu
    cv = rvv - 2.0 * av * rv + 0.5 * r * pv * pv
    den_u = 1.0 + np.abs(ruu) + 2.0 * np.abs(au * ru) + 0.5 * np.abs(r * pu * pu)
    den_v = 1.0 + np.abs(rvv) + 2.0 * np.abs(av * rv) + 0.5 * np.abs(r * pv * pv)
    cun, cvn = cu / den_u, cv / den_v
    mr = inv["m"] / np.maximum(r, 1e-30)
    rho = r / res.mf
    U, V = np.meshgrid(res.u, res.v, indexing="ij")
    X = 0.5 * (V - U)
    base = res.valid & np.isfinite(cun) & np.isfinite(cvn) & (X > 0.5) & (rho > 2e-3)
    regions = {
        "exterior": base & (mr < 0.35),
        "horizon": base & (mr >= 0.45) & (mr <= 0.55),
        "window": base & (mr > 1.5) & (rho > 0.04) & (rho < 0.08),
    }
    for name, mask in regions.items():
        n = int(np.count_nonzero(mask))
        if n == 0:
            print(f"h={h:6.3f} {name:>9} (no cells)")
            continue
        v = np.concatenate([cun[mask], cvn[mask]])
        vals.setdefault(name, []).append((h, float(np.sqrt(np.mean(v ** 2))), n))
        print(f"h={h:6.3f} {name:>9} L2={np.sqrt(np.mean(v**2)):.4e} N={n}")

print()
for name, rows in vals.items():
    rows.sort(reverse=True)  # coarse first
    if len(rows) >= 2:
        ords = [np.log2(rows[i][1] / rows[i + 1][1]) / np.log2(rows[i][0] / rows[i + 1][0])
                for i in range(len(rows) - 1)]
        print(f"{name}: order ~ {np.mean(ords):.2f} ({['%.2f' % o for o in ords]})")
