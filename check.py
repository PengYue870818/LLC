"""
Environment self-check. Verifies that numpy >= 2.0, scipy >= 1.13,
numba >= 0.59, matplotlib >= 3.8 are importable at the required
versions, and that the local `core_numba` acceleration module actually
loads (this is the silent-fallback trap: if `core_numba` fails to
import, the solver still runs but produces numbers that do not match
the paper). Prints PASS / FAIL before any heavy computation.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

import sys
import importlib
from pathlib import Path

def check_deps():
    reqs = [
        ("numpy", "2.0"),
        ("scipy", "1.13"),
        ("numba", "0.59"),
        ("matplotlib", "3.8"),
    ]
    ok = True
    print("===== Third-party dependency version check =====")
    for pkg, min_ver in reqs:
        try:
            mod = importlib.import_module(pkg)
            ver = mod.__version__
            ver_tuple = tuple(map(int, ver.split(".")[:3]))
            min_tuple = tuple(map(int, min_ver.split(".")))
            if ver_tuple >= min_tuple:
                print(f"✅ {pkg}: {ver} （requires ≥ {min_ver}）")
            else:
                print(f"❌ {pkg}: {ver} （requires ≥ {min_ver};  version too old!）")
                ok = False
        except ImportError:
            print(f"❌ {pkg}: not installed")
            ok = False

    print()

    print("===== Project acceleration kernel check =====")
    try:
        import core_numba as _cn
        print(f"✅ core_numba: loaded")
        print(f"   file: {_cn.__file__}")
        print(f"   -> numba fast path will be used")
    except Exception as e:
        print(f"❌ core_numba: failed to load — {type(e).__name__}: {e}")
        print(f"   -> solver will silently fall back to slow Python; numbers will not match the paper!")
        ok = False

    print()

    print("=" * 40)
    if ok:
        print("✅ All checks passed; you can run rerun.py / run_from_zero.bat")
    else:
        print("⚠️ Some checks failed; fix the ❌ items above first.")
    return ok

if __name__ == "__main__":
    check_deps()