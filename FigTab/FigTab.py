"""
Batch driver that runs every figure / table script in `subfigs/` in
order, collects `paper_*.pdf/.png` and `main_table.json` into
`FIGsTABs/`, and writes one `<step>.log` per run. Used by
`run_FigTab.bat`; accepts `--only fig1 fig8 table_main` to run a
subset. The `verify_paper.py` step is commented out by default.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
"""Regenerate every figure and table of the paper from the frozen data set.

Usage (all packages extracted into one folder):

    python run_all.py                 # everything
    python run_all.py --only fig8     # a subset

Figures are collected in ./output/, together with one log per step.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "FIGsTABs"
OUT.mkdir(exist_ok=True)

STEPS = [
    ("fig1", "fig1_causal.py", "zoom_base_h5e4b_v2"),
    ("fig2", "fig2_kasner.py", "zoom_base_h5e4b_v2"),
    ("fig3_5", "figs345_conditioning.py", "three resolution pairs"),
    ("fig6", "fig6_initial_data.py", "eqzoom_v2 + scan_eps*"),
    ("fig7", "fig7_constraints.py", "zoom_base_h1e3_v2, zoom_base_h5e4b_v2"),
    ("fig8", "fig8_mass.py", "eqzoom_v2 + scan_eps*"),
    ("table_mf", "table_mf.py", "eqzoom_v2 + full_small_v2_h005"),
    ("table_main", "table_main.py", "two zoom pairs + coarse"),
    ("table_cm", "table_cm.py", "three resolution pairs"),
    ("regions", "regions.py", "full_small_v2 + full_small_v2_h005"),
    ("error_budget", "error_budget.py", "three resolution pairs"),
    ("lambda_scan", "lambda_scan.py", "zoom_base_h5e4b_v2 + perturbed"),
    #("verify", "verify_paper.py", "every number quoted in the paper"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of step names")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("DATA_ROOT", str(HERE.parent / "from_zero"))
    print(f"DATA_ROOT = {env['DATA_ROOT']}")
    print(f"python    = {sys.executable}\n")

    ok, failed = [], []
    for name, script, needs in STEPS:
        if args.only and name not in args.only:
            continue
        print(f"--- {name}: {script}   (needs {needs})", flush=True)
        proc = subprocess.run([sys.executable, str(HERE / "subfigs" / script)], cwd=HERE / "FIGsTABs",
                              env=env, capture_output=True, text=True)
        log = OUT / f"{name}.log"
        log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if proc.returncode == 0:
            print("    ok")
            ok.append(name)
        else:
            print(f"    FAILED (code {proc.returncode}); see FIGsTABs/{name}.log")
            failed.append(name)

    for pattern in ("paper_*.png", "paper_*.pdf", "main_table.json"):
        for f in HERE.glob(pattern):
            f.replace(OUT / f.name)

    figs = sorted(p.name for p in OUT.glob("paper_*.pdf"))
    print(f"\nfigures in output/: {len(figs)}")
    for f in figs:
        print("   ", f)
    print(f"steps ok: {len(ok)}  failed: {len(failed)}")
    if failed:
        print("failed steps:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())