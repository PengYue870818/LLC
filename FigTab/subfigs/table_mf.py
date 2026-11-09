"""
Table 1. Final mass M_f versus grid spacing h, read directly from the
saved coarse and h = 5e-3 diagnostics. Reports the relative change with
respect to the finest available resolution (and with respect to the
coarse run for reference).

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Table 1: final mass versus grid spacing.

Table 1 of the paper is read straight out of the two saved reference runs, so
no new evolution is needed.  The caption defines the last column as the
relative change with respect to the finest available resolution; the value
relative to the coarse run is printed as well because the two differ by about
one percent.
"""
import json
import os
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent))

RUNS = [("1e-2", "eqzoom_v2", "base_coarse"),
        ("5e-3", "full_small_v2_h005", "base_coarse")]

rows = []
for htag, run, tag in RUNS:
    p = DATA / run / f"{tag}_diagnostics.json"
    if not p.exists():
        print(f"skip h={htag}: {p} missing")
        continue
    with open(p, "r", encoding="utf-8") as f:
        mf = float(json.load(f)["mf_ah"])
    rows.append((float(htag), mf))

if len(rows) < 2:
    print("need both reference runs (eqzoom_v2, full_small_v2_h005)")
    sys.exit(1)

rows.sort(key=lambda r: r[0], reverse=True)      # coarse first
mf_fine = rows[-1][1]
mf_coarse = rows[0][1]

print(f"{'h':>8} {'M_f':>18} {'dM_f/M_f (vs finest)':>22} {'(vs coarsest)':>16}")
for h, mf in rows:
    print(f"{h:>8.0e} {mf:18.10f} {abs(mf - mf_fine) / mf_fine:22.4e} "
          f"{abs(mf - mf_coarse) / mf_coarse:16.4e}")
print()
print(f"M_f(h=1e-2) = {mf_coarse:.10f}   (paper 1.1707698894)")
print(f"M_f(h=5e-3) = {mf_fine:.10f}   (paper 1.1813098597)")
print("Table 1 quotes 9.1e-3 for the coarse row; from these two files the")
print("change is 8.9e-3 relative to the fine run and 9.0e-3 relative to the")
print("coarse run, so the published entry is off in the second digit.")
