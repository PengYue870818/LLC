"""
Driver that re-runs a saved production case with the current solver.
Reads the grid / initial-data / analysis configuration from a
`<tag>_provenance.json` in the seed_configs folder, applies optional
command-line overrides (`grid.h=...`, `initial.amplitude=...`,
`grid.restart_file=...`, etc.), invokes `Yue_Du.run_single`, and writes
the `.npz` / `.json` / horizon products into the target output directory.

    python rerun.py <src_dir> <tag> <dst_dir> [key.path=value ...]

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# Companion code for the paper.  All production figures use the matching
# anchor lambda_star = 0.3 (the anchor scan of Sec. 11.5 uses 0.20-0.35).
"""Re-run a saved production run with the current solver, reproducing its config.

Usage:
  python rerun.py <src_dir> <tag> <dst_dir> [key.path=value ...]

Example:
  python rerun.py eqzoom base_coarse eqzoom_v2
  python rerun.py zoom_base_h1e3 single zoom_base_h1e3_v2 \
      grid.restart_file=eqzoom_v2 grid.restart_u=1.28 grid.restart_tag=base_coarse
"""
import json
import sys
from dataclasses import fields
from pathlib import Path

import Yue_Du as YD


def build(cls, d):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def main():
    src, tag, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    over = {}
    for arg in sys.argv[4:]:
        k, v = arg.split("=", 1)
        over[k] = v
    prov = json.load(open(Path(src) / f"{tag}_provenance.json", encoding="utf-8"))["config"]
    grid = build(YD.GridConfig, prov["grid"])
    init = build(YD.InitialDataConfig, prov["initial"])
    anal = build(YD.AnalysisConfig, prov["analysis"])
    for k, v in over.items():
        sect, name = k.split(".", 1)
        obj = {"grid": grid, "initial": init, "analysis": anal}[sect]
        cur = getattr(obj, name)
        if isinstance(cur, bool):
            setattr(obj, name, v.lower() in ("1", "true", "yes"))
        elif isinstance(cur, int) and not isinstance(cur, bool):
            setattr(obj, name, int(v))
        elif isinstance(cur, float):
            setattr(obj, name, float(v))
        else:
            setattr(obj, name, v)
    cfg = YD.RunConfig(grid, init, anal, dst)
    print(f"running {src}/{tag} -> {dst} with h={grid.h} u0={grid.u0} "
          f"umax={grid.umax} vmax={grid.vmax} restart={grid.restart_file}",
          flush=True)
    res = YD.run_single(cfg)
    YD.save_result(res, Path(dst), tag)
    print(f"done {dst}: valid cells {int(res.diagnostics['num_valid_cells'])}",
          flush=True)


if __name__ == "__main__":
    main()
