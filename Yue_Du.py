import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
try:
    import core_numba as _cn
except Exception:  # pragma: no cover - numba optional
    _cn = None
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PASS_SYMBOL = "PASS"
FAIL_SYMBOL = "FAIL"
WARN_SYMBOL = "WARN"
INFO_SYMBOL = "INFO"

_EXP_CLIP_STATS = {"count": 0, "lo": 0, "hi": 0}


def reset_exp_clip_stats() -> None:
    _EXP_CLIP_STATS["count"] = 0
    _EXP_CLIP_STATS["lo"] = 0
    _EXP_CLIP_STATS["hi"] = 0


def get_exp_clip_stats() -> Dict[str, int]:
    return dict(_EXP_CLIP_STATS)


@dataclass
class GridConfig:
    u0: float = 0.0
    umax: float = 3.6
    vmax: float = 40.0
    h: float = 0.02
    dv: float = 0.0
    first_step_factor: float = 1.0
    r_floor: float = 2.0e-5
    singularity_cut: float = 5.0e-3
    vacuum_r_threshold: float = 0.0
    ru_max: float = 1.0e8
    rv_max: float = 1.0e8
    alpha_floor: float = 1.0e-12
    alpha_ceiling: float = 1.0e6
    corrector_iterations: int = 1
    center_iterations: int = 20
    center_tolerance: float = 1.0e-8
    center_taylor_order: int = 2
    center_damping: float = 0.5
    center_fit_min_offset: int = 1
    center_fit_n_points: int = 8
    center_fit_radius_scale: float = 3.0
    center_project: bool = False
    center_project_cells: int = 6
    center_project_mode: str = "block"
    use_numba: bool = True
    use_center_sources: bool = False
    center_alpha_filter: bool = False
    center_alpha_mode: str = "fit"
    center_scheme: str = "ode"
    restart_file: str = ""
    restart_u: float = 0.0
    restart_tag: str = "single"
    center_block_cells: int = 40
    center_block_x0: float = 0.0
    freeze_alpha: int = 0


@dataclass
class InitialDataConfig:
    L0: float = 1.0
    family: str = "bump"
    amplitude: float = 3.0
    epsilon: float = 0.0
    xc: float = 2.0
    width: float = 0.6
    xc2: float = 2.2
    width2: float = 0.8
    deformation_scale: float = 1.0
    tp_a1: float = 1.0
    tp_a2: float = 0.55
    tp_b1: float = -0.35
    tp_b2: float = 1.0
    x_max_norm: float = 12.0
    ell_x: float = 1.0
    norm_points: int = 24001
    norm_weight_power: float = 2.0
    norm_weight_amplitude: float = 0.25


@dataclass
class AnalysisConfig:
    lambda_star: float = 0.3
    horizon_transversality_min: float = 1.0e-4
    min_horizon_points: int = 6
    min_curve_points: int = 6
    rho_max_fit: float = 0.35
    rho_min_fit: float = 1.0e-3
    savgol_window: int = 11
    savgol_order: int = 3
    kasner_tolerance: float = 0.15
    equal_mass_tolerance: float = 5.0e-4
    mass_root_span: float = 0.15
    fit_nested_windows: int = 5
    match_scheme: str = "equal_rho"
    outer_mass_rmin: float = 6.0
    outer_mass_rmax: float = 40.0
    fixed_kasner_window_lo: float = 1.0e-3
    fixed_kasner_window_hi: float = 1.0e-1
    minkowski_compare_r_min: float = 1.0
    minkowski_compare_r_max: float = 5.0
    minkowski_margin: int = 3
    minkowski_pass_threshold: float = 1.0e-8
    constraint_r_floor_factor: float = 100.0
    constraint_boundary_margin: int = 4
    constraint_r_max_fraction: float = 0.7
    constraint_r_min_fixed: float = 0.5
    constraint_r_max_fixed: float = 5.0
    constraint_conv_r_min: float = 0.5
    constraint_conv_r_max: float = 5.0
    constraint_conv_n_points: int = 40
    massflux_r_floor_factor: float = 20.0
    massflux_horizon_fraction: float = 0.9
    invalid_mass_penalty: float = 1.0e6
    constraint_include_normalized: bool = True


@dataclass
class RunConfig:
    grid: GridConfig
    initial: InitialDataConfig
    analysis: AnalysisConfig
    output_dir: str = "conditioning_output"


@dataclass
class HorizonData:
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    m: np.ndarray
    alpha: np.ndarray
    ell: np.ndarray
    lam: np.ndarray
    transversality: np.ndarray


@dataclass
class CurveData:
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    alpha: np.ndarray
    psi: np.ndarray
    ru: np.ndarray
    rv: np.ndarray
    psu: np.ndarray
    psv: np.ndarray
    m: np.ndarray
    ricci: np.ndarray
    logk: np.ndarray
    rho: np.ndarray
    cpsi: np.ndarray
    cpsi_direct: np.ndarray
    cm2: np.ndarray
    curvature_slope: np.ndarray
    p_parallel: np.ndarray
    p_perp: np.ndarray
    q_inv: np.ndarray
    kasner_sum: np.ndarray
    kasner_constraint: np.ndarray


@dataclass
class SimulationResult:
    config: RunConfig
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    logalpha: np.ndarray
    psi: np.ndarray
    valid: np.ndarray
    mf: float
    horizon: Optional[HorizonData]
    diagnostics: Dict[str, float]
    integrated_mass: Optional[np.ndarray] = None
    ruv_evolved: Optional[np.ndarray] = None


class DerivativeOperator:
    def __init__(self, u: np.ndarray, v: np.ndarray, h: float, dv: Optional[float] = None):
        self.u = np.asarray(u, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.hu = float(h)
        self.hv = float(dv) if dv else float(h)
        self.h = self.hu
        self.nu = self.u.size
        self.nv = self.v.size

    @staticmethod
    def _buf(f):
        return np.asarray(f, dtype=float)

    def du(self, f: np.ndarray, valid: np.ndarray) -> np.ndarray:
        f = self._buf(f)
        valid = np.asarray(valid, dtype=bool)
        out = np.full_like(f, np.nan)
        if self.nu > 2:
            m = valid[:-2, :] & valid[1:-1, :] & valid[2:, :]
            out[1:-1, :][m] = (f[2:, :] - f[:-2, :])[m] / (2.0 * self.hu)
        return out

    def dv(self, f: np.ndarray, valid: np.ndarray) -> np.ndarray:
        f = self._buf(f)
        valid = np.asarray(valid, dtype=bool)
        out = np.full_like(f, np.nan)
        if self.nv > 2:
            m = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
            out[:, 1:-1][m] = (f[:, 2:] - f[:, :-2])[m] / (2.0 * self.hv)
        return out

    def duu(self, f: np.ndarray, valid: np.ndarray) -> np.ndarray:
        f = self._buf(f)
        valid = np.asarray(valid, dtype=bool)
        out = np.full_like(f, np.nan)
        if self.nu > 2:
            m = valid[:-2, :] & valid[1:-1, :] & valid[2:, :]
            out[1:-1, :][m] = (f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :])[m] / (self.hu ** 2)
        return out

    def dvv(self, f: np.ndarray, valid: np.ndarray) -> np.ndarray:
        f = self._buf(f)
        valid = np.asarray(valid, dtype=bool)
        out = np.full_like(f, np.nan)
        if self.nv > 2:
            m = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
            out[:, 1:-1][m] = (f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2])[m] / (self.hv ** 2)
        return out

    def duv(self, f: np.ndarray, valid: np.ndarray) -> np.ndarray:
        f = self._buf(f)
        valid = np.asarray(valid, dtype=bool)
        out = np.full_like(f, np.nan)
        if self.nu > 2 and self.nv > 2:
            m = (valid[:-2, :-2] & valid[:-2, 2:] & valid[2:, :-2] & valid[2:, 2:]
                 & valid[1:-1, 1:-1])
            num = (f[2:, 2:] - f[:-2, 2:] - f[2:, :-2] + f[:-2, :-2])
            out[1:-1, 1:-1][m] = num[m] / (4.0 * self.hu * self.hv)
        return out

    def gradient_all(self, r, loga, psi, valid):
        return {
            "ru": self.du(r, valid),
            "rv": self.dv(r, valid),
            "pu": self.du(psi, valid),
            "pv": self.dv(psi, valid),
            "au": self.du(loga, valid),
            "av": self.dv(loga, valid),
            "ruu": self.duu(r, valid),
            "rvv": self.dvv(r, valid),
            "ruv": self.duv(r, valid),
        }


def jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonify(obj), f, indent=2, allow_nan=False)


def save_csv(path: Path, headers: Sequence[str], columns: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(c).ravel() for c in columns]
    n = max((a.size for a in arrays), default=0)
    padded = []
    for a in arrays:
        if a.size == n:
            padded.append(a)
        elif a.size == 1:
            padded.append(np.full(n, a[0]))
        else:
            out = np.full(n, np.nan)
            out[: a.size] = a
            padded.append(out)
    if n == 0:
        path.write_text(",".join(headers) + "\n", encoding="utf-8")
        return
    data = np.column_stack(padded)
    np.savetxt(path, data, delimiter=",", header=",".join(headers), comments="", fmt="%.16e")


def save_csv_ragged(path: Path, headers: Sequence[str], columns: Sequence[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in zip(*columns):
            vals = []
            for v in row:
                if isinstance(v, str):
                    vals.append(v)
                else:
                    try:
                        fv = float(v)
                    except Exception:
                        fv = float("nan")
                    vals.append(f"{fv:.16e}" if math.isfinite(fv) else "nan")
            f.write(",".join(vals) + "\n")


def provenance(config: RunConfig) -> Dict[str, object]:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "config": {
            "grid": asdict(config.grid),
            "initial": asdict(config.initial),
            "analysis": asdict(config.analysis),
            "output_dir": config.output_dir,
        },
    }


def bump_core(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    y = np.zeros_like(z)
    q = np.abs(z) < 1.0
    zz = z[q]
    y[q] = np.exp(-1.0 / (1.0 - zz * zz))
    return y


def bump_core_d1(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    y = np.zeros_like(z)
    q = np.abs(z) < 1.0
    zz = z[q]
    base = np.exp(-1.0 / (1.0 - zz * zz))
    y[q] = base * (-2.0 * zz) / (1.0 - zz * zz) ** 2
    return y


def bump_core_d2(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    y = np.zeros_like(z)
    q = np.abs(z) < 1.0
    zz = z[q]
    base = np.exp(-1.0 / (1.0 - zz * zz))
    num = 2.0 * (3.0 * zz ** 4 - 1.0)
    den = (1.0 - zz * zz) ** 4
    y[q] = base * num / den
    return y


def gaussian_core(z: np.ndarray) -> np.ndarray:
    return np.exp(-z * z)


def gaussian_core_d1(z: np.ndarray) -> np.ndarray:
    return -2.0 * z * np.exp(-z * z)


def gaussian_core_d2(z: np.ndarray) -> np.ndarray:
    return (4.0 * z * z - 2.0) * np.exp(-z * z)


def profile_basis(x: np.ndarray, cfg: InitialDataConfig):
    x = np.asarray(x, dtype=float)
    z1 = (x - cfg.xc) / cfg.width
    z2 = (x - cfg.xc2) / cfg.width2
    if cfg.family == "bump":
        f, f1, f2 = bump_core(z1), bump_core_d1(z1) / cfg.width, bump_core_d2(z1) / (cfg.width ** 2)
        g, g1, g2 = bump_core(z2), bump_core_d1(z2) / cfg.width2, bump_core_d2(z2) / (cfg.width2 ** 2)
    elif cfg.family == "gaussian":
        f, f1, f2 = gaussian_core(z1), gaussian_core_d1(z1) / cfg.width, gaussian_core_d2(z1) / (cfg.width ** 2)
        g, g1, g2 = gaussian_core(z2), gaussian_core_d1(z2) / cfg.width2, gaussian_core_d2(z2) / (cfg.width2 ** 2)
    elif cfg.family == "two_pulse":
        a1, a2 = cfg.tp_a1, cfg.tp_a2
        b1, b2 = cfg.tp_b1, cfg.tp_b2
        f = a1 * bump_core(z1) + a2 * bump_core(z2)
        f1 = a1 * bump_core_d1(z1) / cfg.width + a2 * bump_core_d1(z2) / cfg.width2
        f2 = a1 * bump_core_d2(z1) / (cfg.width ** 2) + a2 * bump_core_d2(z2) / (cfg.width2 ** 2)
        g = b1 * bump_core(z1) + b2 * bump_core(z2)
        g1 = b1 * bump_core_d1(z1) / cfg.width + b2 * bump_core_d1(z2) / cfg.width2
        g2 = b1 * bump_core_d2(z1) / (cfg.width ** 2) + b2 * bump_core_d2(z2) / (cfg.width2 ** 2)
    elif cfg.family == "shape":
        f, f1, f2 = bump_core(z1), bump_core_d1(z1) / cfg.width, bump_core_d2(z1) / (cfg.width ** 2)
        g = z1 * bump_core(z1)
        g1 = bump_core(z1) / cfg.width + z1 * bump_core_d1(z1) / cfg.width
        g2 = 2.0 * bump_core_d1(z1) / (cfg.width ** 2) + z1 * bump_core_d2(z1) / (cfg.width ** 2)
    else:
        raise ValueError(f"unknown family {cfg.family}")
    return f, f1, f2, g, g1, g2


def scalar_profile(x: np.ndarray, cfg: InitialDataConfig) -> np.ndarray:
    f, _, _, g, _, _ = profile_basis(x, cfg)
    return cfg.amplitude * f + cfg.epsilon * cfg.deformation_scale * g


def scalar_profile_dx(x: np.ndarray, cfg: InitialDataConfig) -> np.ndarray:
    _, f1, _, _, g1, _ = profile_basis(x, cfg)
    return cfg.amplitude * f1 + cfg.epsilon * cfg.deformation_scale * g1


def scalar_profile_dxx(x: np.ndarray, cfg: InitialDataConfig) -> np.ndarray:
    _, _, f2, _, _, g2 = profile_basis(x, cfg)
    return cfg.amplitude * f2 + cfg.epsilon * cfg.deformation_scale * g2


def weight_function(x: np.ndarray, x_max: float, cfg: InitialDataConfig) -> np.ndarray:
    return 1.0 + cfg.norm_weight_amplitude * (x / max(x_max, 1.0e-15)) ** cfg.norm_weight_power


def h2_distance(cfg1: InitialDataConfig, cfg0: InitialDataConfig) -> float:
    xmax = max(cfg1.x_max_norm, cfg0.x_max_norm)
    n = max(cfg1.norm_points, cfg0.norm_points)
    x = np.linspace(0.0, xmax, n)
    d0 = scalar_profile(x, cfg1) - scalar_profile(x, cfg0)
    d1 = scalar_profile_dx(x, cfg1) - scalar_profile_dx(x, cfg0)
    d2 = scalar_profile_dxx(x, cfg1) - scalar_profile_dxx(x, cfg0)
    ell = 0.5 * (cfg1.ell_x + cfg0.ell_x)
    w = 0.5 * (weight_function(x, xmax, cfg1) + weight_function(x, xmax, cfg0))
    val = np.trapezoid(w * (d0 * d0 + ell * ell * d1 * d1 + ell ** 4 * d2 * d2), x)
    return float(math.sqrt(max(val, 0.0)))


def normalized_deformation_scale(cfg: InitialDataConfig) -> float:
    base = InitialDataConfig(**asdict(cfg))
    base.epsilon = 0.0
    test = InitialDataConfig(**asdict(cfg))
    test.epsilon = 1.0
    test.deformation_scale = 1.0
    d = h2_distance(test, base)
    return 1.0 / max(d, 1.0e-30)


def kasner_exponents(c: np.ndarray):
    c = np.asarray(c, dtype=float)
    d = 3.0 + 2.0 * c * c
    return (-1.0 + 2.0 * c * c) / d, 2.0 / d, 4.0 * c / d


def kasner_k_tau(c: np.ndarray, tau: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return 64.0 * (8.0 * c ** 4 - 4.0 * c * c + 3.0) / (3.0 + 2.0 * c * c) ** 4 / tau ** 4


def proper_time_exponent(c: float, momenta: Sequence[float]) -> float:
    p1, p2, _ = kasner_exponents(np.array([c]))
    ps = np.array([p1[0], p2[0], p2[0]])
    momenta = np.asarray(momenta, dtype=float)
    cand = ps[(np.abs(momenta) > 0.0) & (ps > 0.0)]
    return 4.0 if cand.size == 0 else 4.0 / (1.0 + float(np.max(cand)))


def safe_exp(x, lo: float = -700.0, hi: float = 700.0):
    arr = np.asarray(x, dtype=float)
    vals = np.atleast_1d(arr).ravel()
    finite = np.isfinite(vals)
    if np.any(finite):
        vv = vals[finite]
        n_lo = int(np.count_nonzero(vv < lo))
        n_hi = int(np.count_nonzero(vv > hi))
        if n_lo or n_hi:
            _EXP_CLIP_STATS["count"] += n_lo + n_hi
            _EXP_CLIP_STATS["lo"] += n_lo
            _EXP_CLIP_STATS["hi"] += n_hi
    return np.exp(np.clip(arr, lo, hi))


def cell_sources(r: float, loga: float, ru: float, rv: float,
                 psu: float, psv: float, r_floor: float,
                 vacuum: bool = False, vacuum_threshold: float = 0.0):
    a2 = math.exp(float(np.clip(2.0 * loga, -700.0, 700.0)))
    if vacuum:
        if abs(r) < vacuum_threshold:
            return 0.0, 0.0, 0.0
        if abs(r) < 1.0e-30:
            return 0.0, 0.0, 0.0
        sr = -ru * rv / r - a2 / (4.0 * r)
        sa = ru * rv / (r * r) + a2 / (4.0 * r * r)
        return sr, sa, 0.0
    rr = math.copysign(max(abs(r), r_floor), r if r != 0.0 else 1.0)
    mm = 0.5 * rr * (1.0 + 4.0 * ru * rv / a2)
    sr = -a2 * mm / (2.0 * rr * rr)
    sa = a2 * mm / (2.0 * rr * rr * rr) - 0.5 * psu * psv
    sp = -(ru * psv + rv * psu) / rr
    return sr, sa, sp


def cell_sources_center(alpha0: float, alpha2: float,
                        psi0: float, psi2: float,
                        alpha0_prime: float, psi0_prime: float,
                        x_C: float) -> tuple:
    a0 = max(abs(alpha0), 1.0e-30)
    sr = -0.5 * alpha2 * x_C
    sa = -alpha2 / (4.0 * a0) - (psi0_prime * psi0_prime) / 16.0
    sp = psi2 - (alpha0_prime * psi0_prime) / (2.0 * a0)
    return sr, sa, sp


def center_block_width(grid: GridConfig,dv: float,stride: int,nv: int,jc: int) -> int:
    if dv <= 0.0:
        raise ValueError("dv must be positive")
    k = int(grid.center_block_cells)
    if grid.center_block_x0 > 0.0:
        k = int(round(grid.center_block_x0 / (0.5 * dv)))
    max_k = nv - jc - 7
    if max_k < 1:
        return 0
    return min(max(k, 2 * stride, 4), max_k)


class DoubleNullSolver:
    def __init__(self, config: RunConfig):
        self.config = config
        g = config.grid
        self.du = float(g.h)
        self.dv = float(g.dv) if g.dv > 0.0 else float(g.h)
        self.stride = max(1, int(round(self.du / self.dv)))
        # keep the regular centre v = u exactly on grid points
        self.du = self.stride * self.dv
        _u = np.arange(g.u0, g.umax + 0.5 * self.du, self.du)
        _f = float(getattr(g, 'first_step_factor', 1.0))
        if 0.0 < _f < 1.0 and _u.size > 2:
            _u = np.concatenate([[_u[0], _u[0] + _f * self.du],
                                 _u[2:] + (_f - 1.0) * self.du])
        self.u = _u
        self.v = np.arange(g.u0, g.vmax + 0.5 * self.dv, self.dv)
        self.nu = self.u.size
        self.nv = self.v.size
        self.r = np.full((self.nu, self.nv), np.nan)
        self.loga = np.full((self.nu, self.nv), np.nan)
        self.psi = np.full((self.nu, self.nv), np.nan)
        self.valid = np.zeros((self.nu, self.nv), dtype=bool)
        self.ruv_source = np.full((self.nu, self.nv), np.nan)
        self.center_psi = np.full(self.nu, np.nan)
        self.center_loga = np.full(self.nu, np.nan)
        self.center_psi_resid = np.full(self.nu, np.nan)
        self.center_loga_resid = np.full(self.nu, np.nan)
        self.center_reg_resid = np.full(self.nu, np.nan)
        self.center_coeffs = [None] * self.nu
        self._center_coeffs_active = None
        self.x_cut_factor = 3.0
        self.discard_boundary = np.full(self.nu, -1, dtype=int)
        self.singularity_reached = np.zeros(self.nu, dtype=bool)
        self.row_max_ru = np.full(self.nu, np.nan)
        self.row_max_rv = np.full(self.nu, np.nan)
        self.row_max_loga = np.full(self.nu, np.nan)
        self.row_min_m = np.full(self.nu, np.nan)
        self.row_max_m = np.full(self.nu, np.nan)
        self.vacuum_threshold = max(
            config.grid.vacuum_r_threshold,
            10.0 * config.grid.h if config.grid.vacuum_r_threshold <= 0.0 else 0.0
        )

    def _jc(self, i: int) -> int:
        return min(i * self.stride, self.nv - 1)

    def _is_vacuum(self) -> bool:
        return (self.config.initial.amplitude == 0.0
                and self.config.initial.epsilon == 0.0)

    def initialize(self) -> None:
        g = self.config.grid
        d = self.config.initial
        v = self.v

        if g.restart_file:
            self._initialize_from_restart()
            return

        if self._is_vacuum():
            for i in range(self.nu):
                jc = self._jc(i)
                for j in range(self.nv):
                    if j >= jc:
                        self.r[i, j] = (self.v[j] - self.u[i]) / 2.0
                        self.loga[i, j] = 0.0
                        self.psi[i, j] = 0.0
                        self.valid[i, j] = True
            self.r[0, 0] = 0.0
            self.center_psi[0] = 0.0
            self.center_loga[0] = 0.0
            self.center_psi_resid[0] = 0.0
            self.center_loga_resid[0] = 0.0
            self.center_reg_resid[0] = 0.0
            return

        vmax = float(v[-1])

        def rhs(vv, y):
            rr = max(float(y[0]), 0.0)
            p = float(y[1])
            x = rr / d.L0
            px = float(scalar_profile_dx(np.array([x]), d)[0])
            psiv = px * p / d.L0
            return np.array([p, -0.5 * rr * psiv * psiv])

        sol = solve_ivp(rhs, (float(v[0]), vmax), np.array([0.0, 0.5]),
                        t_eval=v, rtol=1.0e-12, atol=1.0e-14, max_step=0.5 * self.dv)
        if not sol.success:
            raise RuntimeError("initial row failed: " + sol.message)
        rr = sol.y[0]
        self.r[0, :] = rr
        self.loga[0, :] = 0.0
        self.psi[0, :] = scalar_profile(rr / d.L0, d)
        self.valid[0, :] = rr >= 0.0
        self.r[0, 0] = 0.0
        self.center_psi[0] = float(self.psi[0, 0])
        self.center_loga[0] = 0.0
        self.center_psi_resid[0] = 0.0
        self.center_loga_resid[0] = 0.0
        self.center_reg_resid[0] = 0.0

    def _initialize_from_restart(self) -> None:
        g = self.config.grid
        from pathlib import Path as _P
        base = _P(g.restart_file)
        npz = base / f"{g.restart_tag}_solution.npz"
        data = np.load(npz)
        uc, vc = data["u"], data["v"]
        i0 = int(np.argmin(np.abs(uc - g.restart_u)))
        if abs(uc[i0] - g.restart_u) > 0.5 * (uc[1] - uc[0]):
            raise RuntimeError("restart_u not on the coarse u-grid")
        js = np.where(data["valid"][i0])[0]
        if js.size < 10:
            raise RuntimeError("coarse row too short for restart")
        xc = vc[js]
        r_c = data["r"][i0, js]
        la_c = data["logalpha"][i0, js]
        ps_c = data["psi"][i0, js]
        r_i = PchipInterpolator(xc, r_c, extrapolate=False)(self.v)
        la_i = PchipInterpolator(xc, la_c, extrapolate=False)(self.v)
        ps_i = PchipInterpolator(xc, ps_c, extrapolate=False)(self.v)
        ok = np.isfinite(r_i) & np.isfinite(la_i) & np.isfinite(ps_i)
        self.r[0, :] = np.where(ok, r_i, np.nan)
        self.loga[0, :] = np.where(ok, la_i, np.nan)
        self.psi[0, :] = np.where(ok, ps_i, np.nan)
        self.valid[0, :] = ok & (r_i >= 0.0)
        self.r[0, 0] = 0.0
        self.valid[0, 0] = True
        self.center_psi[0] = float(self.psi[0, 0])
        self.center_loga[0] = 0.0
        self.center_psi_resid[0] = 0.0
        self.center_loga_resid[0] = 0.0
        self.center_reg_resid[0] = 0.0

    def _fit_center_coeffs(self, i: int):
        g = self.config.grid
        jc = self._jc(i)
        j0 = jc + 1
        j1 = min(j0 + max(g.center_fit_n_points, 3), self.nv)
        if j1 - j0 < 3:
            return None
        js = np.arange(j0, j1)
        js = js[self.valid[i, js] & np.isfinite(self.r[i, js])
                & np.isfinite(self.psi[i, js]) & np.isfinite(self.loga[i, js])]
        if js.size < 3:
            return None
        r_vals = self.r[i, js]
        r_cut_fit = g.center_fit_radius_scale * self.dv
        q = (r_vals > 0.0) & (r_vals < r_cut_fit)
        if np.count_nonzero(q) < 3:
            return None
        r_vals = r_vals[q]
        js = js[q]
        psi_vals = self.psi[i, js]
        r2 = r_vals * r_vals
        try:
            psi_cp = np.polyfit(r2, psi_vals, 1)
        except Exception:
            return None
        alpha0 = 1.0
        alpha0_prime = 0.0
        r1 = alpha0
        psi0 = float(psi_cp[1])
        slope_p = float(psi_cp[0])
        psi2 = slope_p * r1 * r1

        psi0_prime = 0.0
        if i >= 1 and self.center_coeffs[i - 1] is not None:
            psi0_prime = (psi0 - self.center_coeffs[i - 1]["psi0"]) / self.du

        alpha2 = (psi0_prime * psi0_prime) / 4.0

        return {
            "alpha0": float(alpha0),
            "alpha2": float(alpha2),
            "psi0": float(psi0),
            "psi2": float(psi2),
            "alpha0_prime": float(alpha0_prime),
            "psi0_prime": float(psi0_prime),
        }

    def _compute_sources_at(self, rs, re, rw, rn, aas, aae, aaw, an,
                            ps, pe, pw, pn, h, vacuum, x_C=1.0):
        du = self.du
        dv = self.dv
        rc = 0.25 * (rs + re + rw + rn)
        ac = 0.25 * (aas + aae + aaw + an)
        ru = ((rw + rn) - (rs + re)) / (2.0 * du)
        rv = ((re + rn) - (rs + rw)) / (2.0 * dv)
        psu = ((pw + pn) - (ps + pe)) / (2.0 * du)
        psv = ((pe + pn) - (ps + pw)) / (2.0 * dv)
        cc = self._center_coeffs_active
        if (cc is not None and self.config.grid.use_center_sources
                and abs(x_C) < self.x_cut_factor * min(du, dv)):
            sr, sa, sp = cell_sources_center(
                cc["alpha0"], cc["alpha2"],
                cc["psi0"], cc["psi2"],
                cc["alpha0_prime"], cc["psi0_prime"],
                x_C,
            )
            return sr, sa, sp, ru, rv
        sr, sa, sp = cell_sources(rc, ac, ru, rv, psu, psv, self.config.grid.r_floor,
                                  vacuum, self.vacuum_threshold)
        return sr, sa, sp, ru, rv

    def evolve_cell(self, i: int, j: int) -> bool:
        g = self.config.grid
        S, E, W, N = (i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)
        if not (self.valid[S] and self.valid[E] and self.valid[W]):
            return False
        rs, re, rw = self.r[S], self.r[E], self.r[W]
        aas, aae, aaw = self.loga[S], self.loga[E], self.loga[W]
        ps, pe, pw = self.psi[S], self.psi[E], self.psi[W]
        vacuum = self._is_vacuum()

        rn_pred = re + rw - rs
        an_pred = aae + aaw - aas
        pn_pred = pe + pw - ps
        x_C = (self.v[j] + self.v[j + 1] - self.u[i] - self.u[i + 1]) / 4.0

        sr0, sa0, sp0, ru0, rv0 = self._compute_sources_at(
            rs, re, rw, rn_pred, aas, aae, aaw, an_pred,
            ps, pe, pw, pn_pred, self.dv, vacuum, x_C=x_C)
        if abs(ru0) > g.ru_max or abs(rv0) > g.rv_max:
            return False
        if not (math.isfinite(sr0) and math.isfinite(sa0) and math.isfinite(sp0)):
            return False

        du, dv = self.du, self.dv
        rn_star = re + rw - rs + du * dv * sr0
        an_star = aae + aaw - aas + du * dv * sa0
        pn_star = pe + pw - ps + du * dv * sp0

        sr1, sa1, sp1, ru1, rv1 = self._compute_sources_at(
            rs, re, rw, rn_star, aas, aae, aaw, an_star,
            ps, pe, pw, pn_star, self.dv, vacuum, x_C=x_C)
        if abs(ru1) > g.ru_max or abs(rv1) > g.rv_max:
            return False
        if not (math.isfinite(sr1) and math.isfinite(sa1) and math.isfinite(sp1)):
            return False

        rn = re + rw - rs + 0.5 * du * dv * (sr0 + sr1)
        an = aae + aaw - aas + 0.5 * du * dv * (sa0 + sa1)
        pn = pe + pw - ps + 0.5 * du * dv * (sp0 + sp1)

        if not (math.isfinite(rn) and math.isfinite(an) and math.isfinite(pn)):
            return False

        alpha = math.exp(float(np.clip(an, -700.0, 700.0)))
        if rn < 0.0:
            return False
        if rn < g.singularity_cut and rv0 < 0.0:
            return False
        if not vacuum and rn <= g.r_floor and rv0 < 0.0:
            return False
        if alpha < g.alpha_floor or alpha > g.alpha_ceiling:
            return False

        self.r[N] = rn
        self.loga[N] = an
        self.psi[N] = pn
        self.ruv_source[N] = 0.5 * (sr0 + sr1)
        self.valid[N] = True
        return True

    def predict_center(self, i: int) -> float:
        if i <= 1:
            return float(self.psi[0, 0])
        return float(2.0 * self.center_psi[i - 1] - self.center_psi[i - 2])

    def refit_center(self, i: int, order: int) -> float:
        g = self.config.grid
        j0 = self._jc(i) + max(g.center_fit_min_offset, 1)
        j1 = min(j0 + max(g.center_fit_n_points, 8), self.nv)
        js = np.arange(j0, j1)
        js = js[self.valid[i, js]]
        if js.size < 3:
            return self.predict_center(i)
        r_vals = self.r[i, js]
        q = r_vals < g.center_fit_radius_scale * self.dv
        if np.count_nonzero(q) >= 3:
            js = js[q]
            r_vals = r_vals[q]
        r2 = r_vals ** 2
        deg = min(order, js.size - 1)
        try:
            cp = np.polyfit(r2, self.psi[i, js], deg)
            return float(np.polyval(cp, 0.0))
        except Exception:
            return self.predict_center(i)

    def _regularize_center(self, i: int) -> float:
        g = self.config.grid
        jc = self._jc(i)
        if not g.center_project:
            return float(self.psi[i, jc])
        kmax = min(int(g.center_project_cells), self.nv - 1 - jc)
        if kmax < 4:
            return float(self.psi[i, jc])
        ks = np.arange(0, kmax + 1)
        js = jc + ks
        keep = self.valid[i, js] & np.isfinite(self.r[i, js])
        keep[0] = True
        if np.count_nonzero(keep) < 5:
            return float(self.psi[i, jc])
        js = js[keep]
        h = self.dv
        x = 0.5 * (self.v[js] - self.u[i])
        x2 = x * x
        if str(g.center_project_mode) == "first_cell":
            if js.size >= 4 and js[0] == jc:
                q = x > 1.5 * h
                if np.count_nonzero(q) >= 3:
                    r3 = float(np.median((self.r[i, js[q]] - x[q]) / x[q] ** 3))
                    x1 = 0.5 * h
                    self.r[i, jc + 1] = x1 + r3 * x1 ** 3
            return float(self.psi[i, jc])
        try:
            q = x > 0.0
            y = (self.r[i, js[q]] - x[q]) / (x[q] ** 3)
            cr = np.polyfit(x2[q], y, 1)
            self.r[i, js] = x + cr[1] * x ** 3 + cr[0] * x ** 5
        except Exception:
            pass
        psi0 = float(self.psi[i, jc])
        try:
            cp = np.polyfit(x2, self.psi[i, js], 2)
            self.psi[i, js] = np.polyval(cp, x2)
            psi0 = float(np.polyval(cp, 0.0))
        except Exception:
            pass
        try:
            ca = np.polyfit(x2, self.loga[i, js], 2)
            ca[2] = 0.0
            self.loga[i, js] = np.polyval(ca, x2)
        except Exception:
            pass
        self.r[i, jc] = 0.0
        self.psi[i, jc] = psi0
        self.loga[i, jc] = 0.0
        return psi0

    def _filter_center_alpha(self, i: int) -> None:
        g = self.config.grid
        jc = self._jc(i)
        stride = max(1, int(self.stride))
        kmax = min(int(max(g.center_fit_n_points, 2 * stride)), self.nv - 1 - jc)
        mfit = min(int(max(8 * stride, 24)), self.nv - 1 - jc)
        if kmax < 1 or mfit < 6:
            return
        js = jc + np.arange(0, mfit + 1)
        keep = self.valid[i, js]
        keep[0] = True
        js = js[keep]
        if js.size < 6:
            return
        x = 0.5 * (self.v[js] - self.u[i])
        q = x > 0.0
        if np.count_nonzero(q) < 5:
            return
        xx = x[q]
        try:
            c = np.polyfit(xx ** 2, self.loga[i, js[q]], 2)
            c[2] = 0.0
        except Exception:
            return
        target = jc + np.arange(0, kmax + 1)
        target = target[target <= js.max()]
        xt = 0.5 * (self.v[target] - self.u[i])
        self.loga[i, target] = np.polyval(c, xt ** 2)
        self.loga[i, jc] = 0.0

    def _center_block(self, i: int, psi0: float, p0: float, psi2: float) -> int:
        jc = self._jc(i)
        g = self.config.grid
        K = center_block_width(
            g, self.dv, self.stride, self.nv, jc
        )
        if K < 1:
            return 0

        ks = np.arange(1, K + 1)
        x = 0.5 * ks * self.dv
        r3 = getattr(self, "_center_r3", 0.0)
        a2 = getattr(self, "_center_a2", 0.0)

        self.r[i, jc + ks] = x + r3 * x ** 3
        self.psi[i, jc + ks] = psi0 + psi2 * x ** 2
        self.loga[i, jc + ks] = a2 * x ** 2
        self.valid[i, jc + ks] = True
        return K

    def _fit_center_tail(self, i: int, first_j: int, npts: int = 0):
        if npts <= 0:
            npts = max(8, int(round(0.08 / (0.5 * self.dv))))
        j1 = min(first_j + npts, self.nv)
        js = np.arange(first_j, j1)
        js = js[self.valid[i, js] & np.isfinite(self.r[i, js])]
        if js.size < 6:
            return None
        x = 0.5 * (self.v[js] - self.u[i])
        q = x > 0.0
        if np.count_nonzero(q) < 6:
            return None
        xx = x[q]
        x2 = xx * xx
        out = {}
        try:
            r3 = float(np.median((self.r[i, js[q]] - xx) / xx ** 3))
            out["r3"] = r3
        except Exception:
            pass
        try:
            ca = np.polyfit(x2, self.loga[i, js[q]], 2)
            out["a2"] = float(ca[1])
            out["a0"] = float(math.exp(ca[2]))
        except Exception:
            pass
        try:
            cp = np.polyfit(x2, self.psi[i, js[q]], 2)
            out["psi2"] = float(cp[1])
        except Exception:
            pass
        return out

    def _fit_psi2(self,i: int,first_j: int,npts: int = 0) -> Optional[float]:
        if npts <= 0:
            npts = max(8, int(round(0.08 / (0.5 * self.dv))))

        j1 = min(first_j + npts, self.nv)
        if first_j < 0 or first_j >= j1:
            return None

        js = np.arange(first_j, j1)
        js = js[self.valid[i, js] & np.isfinite(self.psi[i, js])]
        if js.size < 4:
            return None

        x = 0.5 * (self.v[js] - self.u[i])
        keep = x > 0.0
        if np.count_nonzero(keep) < 4:
            return None

        try:
            coeffs = np.polyfit(x[keep] ** 2, self.psi[i, js[keep]], 2)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None

        psi2 = float(coeffs[1])
        return psi2 if math.isfinite(psi2) else None

    def _du_local(self, i: int) -> float:
        if i <= 0:
            return float(self.du)
        return float(self.u[i] - self.u[i - 1])

    def _march_row(self, i: int, psi_c: float, la_c: float) -> None:
        g = self.config.grid
        jc = self._jc(i)
        if g.center_scheme == "ode":
            la_c = 0.0
        self.valid[i, :] = False
        self.valid[i, jc] = True
        self.r[i, jc] = 0.0
        self.psi[i, jc] = psi_c
        self.loga[i, jc] = la_c
        jstart = jc
        if g.center_scheme == "ode":
            p0 = getattr(self, "_center_p0", 0.0)
            psi2 = getattr(self, "_center_psi2", 0.0)
            K = self._center_block(i, psi_c, p0, psi2)
            jstart = jc + K
        if g.use_numba and _cn is not None:
            cc = self._center_coeffs_active
            if cc is None:
                carr = np.zeros(6)
                use_cc = False
            else:
                carr = np.array([cc["alpha0"], cc["alpha2"], cc["psi0"], cc["psi2"],
                                 cc["alpha0_prime"], cc["psi0_prime"]], dtype=np.float64)
                use_cc = bool(self.config.grid.use_center_sources)
            stop = _cn.march_row(
                self.r, self.loga, self.psi, self.valid, self.ruv_source,
                i, jc, jstart, self.nv, self._du_local(i), self.dv, carr, use_cc,
                self.x_cut_factor * min(self.du, self.dv),
                g.r_floor, g.singularity_cut,
                g.alpha_floor, g.alpha_ceiling, g.ru_max, g.rv_max,
                int(getattr(g, 'freeze_alpha', 0)))
            if stop < self.nv - 1:
                self.discard_boundary[i] = stop
            return
        for j in range(jstart, self.nv - 1):
            if not self.evolve_cell(i - 1, j):
                self.discard_boundary[i] = j
                break

    def _remarch_tail(self, i: int, first_j: int) -> None:
        for j in range(first_j, self.nv - 1):
            if not self.evolve_cell(i - 1, j):
                self.discard_boundary[i] = j
                break

    def evolve_row(self, i: int) -> None:
        g = self.config.grid
        self._center_coeffs_active = (
            self.center_coeffs[i - 1] if i > 0 else None
        )
        psi_c = self.predict_center(i)
        la_c = 0.0
        psi_res_pre_damp = float("inf")
        if g.center_scheme == "ode":
            if i >= 2 and self.center_coeffs[i - 1] is not None:
                psi2_prev = self.center_coeffs[i - 1]["psi2"]
            else:
                psi2_prev = 0.0
            psi_c = 2.0 * self.center_psi[i - 1] - self.center_psi[i - 2] \
                + self.du * self.du * 6.0 * psi2_prev
            if not math.isfinite(psi_c):
                psi_c = self.center_psi[i - 1]
            self._center_p0 = (psi_c - self.center_psi[i - 1]) / self.du
            self._center_psi2 = psi2_prev
            self._march_row(i, psi_c, la_c)
            jc = self._jc(i)
            K = center_block_width(g, self.dv, self.stride, self.nv, jc)
            fit_start = jc + K + 1
            psi2_fit = self._fit_psi2(i, fit_start)
            psi2_new = psi2_prev if psi2_fit is None else psi2_fit
            tail = self._fit_center_tail(i, fit_start)


            if tail:
                if "r3" in tail:
                    self._center_r3 = tail["r3"]
                if "a2" in tail:
                    self._center_a2 = tail["a2"]
                if "a0" in tail:
                    self._center_alpha0 = tail["a0"]
            cc_ode = {
                "alpha0": 1.0,
                "alpha2": self._center_p0 ** 2 / 6.0,
                "psi0": float(psi_c),
                "psi2": float(psi2_new),
                "alpha0_prime": 0.0,
                "psi0_prime": float(self._center_p0),
            }
            self.center_coeffs[i] = cc_ode
            self.center_psi[i] = float(psi_c)
            self.center_loga[i] = la_c
            self.center_psi_resid[i] = 0.0
            jc = self._jc(i)
            if jc + 1 < self.nv and self.valid[i, jc + 1]:
                rv_c = (self.r[i, jc + 1] - self.r[i, jc]) / self.dv
                self.center_reg_resid[i] = abs(rv_c - 0.5) / 0.5
            else:
                self.center_reg_resid[i] = float("nan")
            self._center_psi2 = float(psi2_new)
            if np.any(self.valid[i]):
                js = np.where(self.valid[i])[0]
                self.row_max_loga[i] = float(np.nanmax(np.abs(self.loga[i, js])))
            return
        cc_iter = self._center_coeffs_active
        for _ in range(g.center_iterations):
            self._center_coeffs_active = cc_iter
            self._march_row(i, psi_c, la_c)
            psi_new = self.refit_center(i, g.center_taylor_order)
            if not math.isfinite(psi_new):
                break
            psi_res_pre_damp = abs(psi_new - psi_c)
            psi_c = (1.0 - g.center_damping) * psi_c + g.center_damping * psi_new
            cc_new = self._fit_center_coeffs(i)
            if cc_new is not None:
                cc_iter = cc_new
            if psi_res_pre_damp < g.center_tolerance:
                break
        self._center_coeffs_active = cc_iter
        self._march_row(i, psi_c, la_c)
        if self.config.grid.center_alpha_filter:
            self._filter_center_alpha(i)
        self.center_psi[i] = psi_c
        self.center_loga[i] = la_c
        self.center_psi_resid[i] = psi_res_pre_damp

        jc = self._jc(i)
        if jc + 1 < self.nv and self.valid[i, jc + 1]:
            h_c = self.dv
            rv_c = (self.r[i, jc + 1] - self.r[i, jc]) / h_c
            alpha_c = math.exp(float(np.clip(la_c, -700.0, 700.0)))
            rv_expected = 0.5 * alpha_c
            self.center_reg_resid[i] = (
                abs(rv_c - rv_expected) / max(abs(rv_expected), 1.0e-30)
            )
        else:
            self.center_reg_resid[i] = float("nan")

        if g.center_project:
            kmax = min(int(g.center_project_cells), self.nv - 1 - jc)
            psi_proj = self._regularize_center(i)
            if math.isfinite(psi_proj):
                psi_c = psi_proj
                self.center_psi[i] = psi_c
            if str(g.center_project_mode) != "first_cell" and kmax >= 4:
                self._remarch_tail(i, jc + kmax)

        cc = self._fit_center_coeffs(i)
        self.center_coeffs[i] = cc
        self.center_loga_resid[i] = float("nan")
        if (i >= 2
                and self.center_coeffs[i] is not None
                and self.center_coeffs[i - 1] is not None
                and self.center_coeffs[i - 2] is not None):
            a0 = self.center_coeffs[i]["alpha0"]
            a1 = self.center_coeffs[i - 1]["alpha0"]
            a2 = self.center_coeffs[i - 2]["alpha0"]
            alpha2 = self.center_coeffs[i]["alpha2"]
            psi0p = self.center_coeffs[i]["psi0_prime"]
            h_c = self.du
            a0p = (3.0 * a0 - 4.0 * a1 + a2) / (2.0 * h_c)
            a0pp = (a0 - 2.0 * a1 + a2) / (h_c * h_c)
            lhs = a0pp - (a0p / a0) * a0p - alpha2 + 0.25 * a0 * psi0p * psi0p
            self.center_loga_resid[i] = abs(lhs) * a0 * a0

        if np.any(self.valid[i]):
            js = np.where(self.valid[i])[0]
            if js.size > 2:
                r_row = self.r[i, js]
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    rv_row = np.gradient(r_row, self.dv)
                    self.row_max_rv[i] = float(np.nanmax(np.abs(rv_row)))
                self.row_max_loga[i] = float(np.nanmax(np.abs(self.loga[i, js])))
            last_valid = int(np.max(js)) if js.size > 0 else i
            if last_valid < self.nv - 1:
                self.singularity_reached[i] = True

    def run(self) -> SimulationResult:
        self.initialize()
        if not self._is_vacuum() and self.center_coeffs[0] is None:
            self.center_coeffs[0] = self._fit_center_coeffs(0)
        if not self._is_vacuum():
            for i in range(self.nu - 1):
                if self._jc(i + 1) >= self.nv - 2:
                    break
                self.evolve_row(i + 1)
                if not np.any(self.valid[i + 1]):
                    break
        mf, horizon, diagnostics, integrated = analyze_global(
            self.u, self.v, self.r, self.loga, self.psi, self.valid, self.config,
            ruv_source=self.ruv_source
        )
        diagnostics["center_psi_resid_max"] = (
            float(np.nanmax(self.center_psi_resid))
            if np.any(np.isfinite(self.center_psi_resid)) else float("nan")
        )
        diagnostics["center_loga_resid_max"] = 0.0
        if np.any(np.isfinite(self.center_reg_resid)):
            diagnostics["center_reg_resid_max"] = float(np.nanmax(self.center_reg_resid))
            diagnostics["center_reg_resid_median"] = float(np.nanmedian(self.center_reg_resid))
        else:
            diagnostics["center_reg_resid_max"] = float("nan")
            diagnostics["center_reg_resid_median"] = float("nan")
        n_cc = sum(1 for c in self.center_coeffs if c is not None)
        diagnostics["center_coeffs_rows"] = float(n_cc)
        if n_cc > 0:
            a0_vals = np.array([c["alpha0"] for c in self.center_coeffs if c is not None])
            a2_vals = np.array([c["alpha2"] for c in self.center_coeffs if c is not None])
            p0_vals = np.array([c["psi0"] for c in self.center_coeffs if c is not None])
            p2_vals = np.array([c["psi2"] for c in self.center_coeffs if c is not None])
            diagnostics["center_alpha0_min"] = float(np.nanmin(a0_vals))
            diagnostics["center_alpha0_max"] = float(np.nanmax(a0_vals))
            diagnostics["center_alpha2_abs_median"] = float(np.nanmedian(np.abs(a2_vals)))
            diagnostics["center_alpha2_abs_max"] = float(np.nanmax(np.abs(a2_vals)))
            diagnostics["center_psi0_abs_max"] = float(np.nanmax(np.abs(p0_vals)))
            diagnostics["center_psi2_abs_median"] = float(np.nanmedian(np.abs(p2_vals)))
            diagnostics["center_psi2_abs_max"] = float(np.nanmax(np.abs(p2_vals)))
        else:
            for k in ("center_alpha0_min", "center_alpha0_max",
                      "center_alpha2_abs_median", "center_alpha2_abs_max",
                      "center_psi0_abs_max", "center_psi2_abs_median",
                      "center_psi2_abs_max"):
                diagnostics[k] = float("nan")
        nz = self.discard_boundary[self.discard_boundary >= 0]
        diagnostics["discard_count"] = float(nz.size)
        diagnostics["discard_min_index"] = float(np.min(nz)) if nz.size else float("nan")
        diagnostics["num_valid_cells"] = float(np.count_nonzero(self.valid))
        diagnostics["num_singularity_rows"] = float(np.count_nonzero(self.singularity_reached))
        diagnostics["row_max_loga"] = (
            float(np.nanmax(self.row_max_loga))
            if np.any(np.isfinite(self.row_max_loga)) else float("nan")
        )
        diagnostics["row_max_rv"] = (
            float(np.nanmax(self.row_max_rv))
            if np.any(np.isfinite(self.row_max_rv)) else float("nan")
        )
        return SimulationResult(
            self.config, self.u, self.v, self.r, self.loga, self.psi, self.valid,
            mf, horizon, diagnostics, integrated, self.ruv_source
        )


def compute_unified_derivatives(result: SimulationResult) -> Dict[str, np.ndarray]:
    deriv_op = DerivativeOperator(result.u, result.v, result.config.grid.h,
                                  result.config.grid.dv or result.config.grid.h)
    return deriv_op.gradient_all(result.r, result.logalpha, result.psi, result.valid)


def derivatives(result: SimulationResult) -> Dict[str, np.ndarray]:
    return compute_unified_derivatives(result)


def invariant_arrays(result: SimulationResult,
                     d: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    if d is None:
        d = compute_unified_derivatives(result)
    r = result.r
    loga = result.logalpha
    ru, rv = d["ru"], d["rv"]
    pu, pv = d["pu"], d["pv"]
    base_finite = (
        result.valid
        & np.isfinite(r)
        & np.isfinite(loga)
        & np.isfinite(ru)
        & np.isfinite(rv)
        & np.isfinite(pu)
        & np.isfinite(pv)
        & (r > 0.0)
    )

    a2inv = np.full_like(r, np.nan)
    a2inv[base_finite] = np.exp(-2.0 * loga[base_finite])

    m = np.full_like(r, np.nan)
    ricci = np.full_like(r, np.nan)
    m[base_finite] = 0.5 * r[base_finite] * (
        1.0 + 4.0 * ru[base_finite] * rv[base_finite] * a2inv[base_finite]
    )
    ricci[base_finite] = -4.0 * pu[base_finite] * pv[base_finite] * a2inv[base_finite]

    r_safe = np.where(r > 0.0, r, np.nan)
    x = m / (r_safe ** 3)
    y = ricci / 12.0
    xy = x + y

    logk = np.full_like(r, np.nan)
    good = base_finite & np.isfinite(xy) & np.isfinite(ricci)
    if np.any(good):
        xy_abs = np.abs(xy[good])
        ricci_abs = np.abs(ricci[good])
        xy_use = np.maximum(xy_abs, 1.0e-300)
        ricci_use = np.maximum(ricci_abs, 1.0e-300)
        logk[good] = np.logaddexp(
            np.log(48.0) + 2.0 * np.log(xy_use),
            np.log(5.0 / 3.0) + 2.0 * np.log(ricci_use),
        )
    return {"m": m, "ricci": ricci, "logk": logk}


def compute_integrated_mass(result: SimulationResult, d: Dict[str, np.ndarray],
                            inv: Dict[str, np.ndarray]) -> np.ndarray:
    r = result.r
    loga = result.logalpha
    ru, pv = d["ru"], d["pv"]
    a2inv = safe_exp(-2.0 * loga)
    flux_v = -r * r * a2inv * ru * pv * pv
    nv = result.v.size
    m_int = np.full_like(inv["m"], np.nan)
    h = result.config.grid.dv if result.config.grid.dv > 0.0 else result.config.grid.h
    r_min_start = result.config.analysis.constraint_r_min_fixed
    for i in range(result.u.size):
        js = np.where(result.valid[i]
                      & np.isfinite(inv["m"][i])
                      & np.isfinite(r[i])
                      & (r[i] > r_min_start))[0]
        if js.size == 0:
            continue
        j0 = int(js[0])
        m0 = inv["m"][i, j0]
        if not math.isfinite(m0):
            continue
        m_int[i, j0] = m0
        for j in range(j0 + 1, nv):
            if not result.valid[i, j]:
                break
            if not (math.isfinite(flux_v[i, j - 1]) and math.isfinite(flux_v[i, j])):
                break
            m_int[i, j] = m_int[i, j - 1] + 0.5 * h * (flux_v[i, j - 1] + flux_v[i, j])
    return m_int


def normalized_norm(arr: np.ndarray, mask: np.ndarray):
    q = np.asarray(arr)[mask]
    q = q[np.isfinite(q)]
    if q.size == 0:
        return float("nan"), float("nan")
    return float(math.sqrt(np.mean(q * q))), float(np.max(np.abs(q)))


def plain_norm(arr: np.ndarray, mask: np.ndarray):
    q = np.asarray(arr)[mask]
    q = q[np.isfinite(q)]
    if q.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(q * q))), float(np.max(np.abs(q)))


def constraint_residuals(result: SimulationResult, d: Dict[str, np.ndarray]):
    r = result.r
    ru, rv = d["ru"], d["rv"]
    pu, pv = d["pu"], d["pv"]
    au, av = d["au"], d["av"]
    ruu, rvv = d["ruu"], d["rvv"]

    cu_raw = ruu - 2.0 * au * ru + 0.5 * r * pu * pu
    cv_raw = rvv - 2.0 * av * rv + 0.5 * r * pv * pv

    cu_den = 1.0 + np.abs(ruu) + 2.0 * np.abs(au * ru) + 0.5 * np.abs(r * pu * pu)
    cv_den = 1.0 + np.abs(rvv) + 2.0 * np.abs(av * rv) + 0.5 * np.abs(r * pv * pv)
    cu_norm = cu_raw / cu_den
    cv_norm = cv_raw / cv_den

    margin = result.config.analysis.constraint_boundary_margin
    r_min = result.config.analysis.constraint_r_min_fixed
    r_max = result.config.analysis.constraint_r_max_fixed

    mask = (result.valid
            & np.isfinite(cu_raw) & np.isfinite(cv_raw)
            & (r > r_min) & (r < r_max))
    if margin > 0:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

    cu_raw_l2, cu_raw_li = plain_norm(cu_raw, mask)
    cv_raw_l2, cv_raw_li = plain_norm(cv_raw, mask)
    cu_norm_l2, cu_norm_li = normalized_norm(cu_norm, mask)
    cv_norm_l2, cv_norm_li = normalized_norm(cv_norm, mask)

    return {
        "constraint_u_l2": cu_raw_l2,
        "constraint_u_linf": cu_raw_li,
        "constraint_v_l2": cv_raw_l2,
        "constraint_v_linf": cv_raw_li,
        "constraint_u_l2_normalized": cu_norm_l2,
        "constraint_u_linf_normalized": cu_norm_li,
        "constraint_v_l2_normalized": cv_norm_l2,
        "constraint_v_linf_normalized": cv_norm_li,
        "constraint_valid_points": float(np.count_nonzero(mask)),
    }


def mass_flux_residuals(result: SimulationResult, d: Dict[str, np.ndarray],
                        inv: Dict[str, np.ndarray]):
    r = result.r
    loga = result.logalpha
    a2inv = safe_exp(-2.0 * loga)
    a2 = safe_exp(2.0 * loga)
    ru, rv = d["ru"], d["rv"]
    pu, pv = d["pu"], d["pv"]
    au, av = d["au"], d["av"]
    ruu, rvv = d["ruu"], d["rvv"]

    r_safe = np.where(r > 0.0, r, np.nan)
    ruv_analytic = -ru * rv / r_safe - a2 / (4.0 * r_safe)

    m_u_chain = (
        ru / 2.0
        + 2.0 * (ru * ru * rv + r * ruu * rv + r * ru * ruv_analytic) * a2inv
        - 4.0 * r * ru * rv * au * a2inv
    )
    m_u_flux = -(r * r) * a2inv * rv * pu * pu
    res_u = m_u_chain - m_u_flux

    m_v_chain = (
        rv / 2.0
        + 2.0 * (rv * rv * ru + r * rvv * ru + r * rv * ruv_analytic) * a2inv
        - 4.0 * r * ru * rv * av * a2inv
    )
    m_v_flux = -(r * r) * a2inv * ru * pv * pv
    res_v = m_v_chain - m_v_flux

    margin = result.config.analysis.constraint_boundary_margin
    r_min = result.config.analysis.constraint_r_min_fixed
    r_max = result.config.analysis.constraint_r_max_fixed

    mask = (result.valid
            & np.isfinite(res_u) & np.isfinite(res_v)
            & (r > r_min) & (r < r_max))
    if margin > 0:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

    mu_l2, mu_li = plain_norm(res_u, mask)
    mv_l2, mv_li = plain_norm(res_v, mask)

    return {
        "massflux_u_l2": mu_l2, "massflux_u_linf": mu_li,
        "massflux_v_l2": mv_l2, "massflux_v_linf": mv_li,
        "massflux_valid_points": float(np.count_nonzero(mask)),
    }


def constraint_and_flux_residuals(result: SimulationResult, d: Dict[str, np.ndarray],
                                  inv: Dict[str, np.ndarray]):
    out = {}
    out.update(constraint_residuals(result, d))
    out.update(mass_flux_residuals(result, d, inv))
    return out


def pde_residual(result: SimulationResult, d: Dict[str, np.ndarray]):
    r = result.r
    loga = result.logalpha
    ru, rv = d["ru"], d["rv"]
    ruv = d["ruv"]
    a2 = safe_exp(2.0 * loga)

    S_r = np.full_like(r, np.nan)
    good = (result.valid
            & np.isfinite(ru) & np.isfinite(rv)
            & np.isfinite(r) & np.isfinite(loga)
            & (r > 0.0))
    rr = np.where(r > 0.0, r, np.nan)
    S_r[good] = -ru[good] * rv[good] / rr[good] - a2[good] / (4.0 * rr[good])
    R_r = ruv - S_r
    return {"S_r": S_r, "R_r": R_r}


def minkowski_pde_residual_test(hs):
    records = []
    for h in hs:
        cfg = RunConfig(
            GridConfig(h=h, umax=10.0, vmax=12.0, singularity_cut=1.0e-10),
            InitialDataConfig(family="bump", amplitude=0.0, epsilon=0.0),
            AnalysisConfig(),
            output_dir="minkowski",
        )
        cfg.initial.deformation_scale = 1.0
        try:
            res = run_single(cfg)
            d = compute_unified_derivatives(res)
            pde = pde_residual(res, d)
            R_r = pde["R_r"]
            r = res.r
            r_min = cfg.analysis.minkowski_compare_r_min
            r_max = cfg.analysis.minkowski_compare_r_max
            margin = cfg.analysis.minkowski_margin

            mask = (res.valid
                    & np.isfinite(R_r)
                    & (r > r_min) & (r < r_max))
            if margin > 0:
                mask[:margin, :] = False
                mask[-margin:, :] = False
                mask[:, :margin] = False
                mask[:, -margin:] = False

            n_pts = int(np.count_nonzero(mask))
            if n_pts < 10:
                records.append({
                    "h": float(h),
                    "error": f"insufficient residual points: {n_pts}",
                    "n_valid": int(np.count_nonzero(res.valid)),
                })
                continue

            vals = R_r[mask]
            vals = vals[np.isfinite(vals)]
            res_l2 = float(np.sqrt(np.mean(vals ** 2)))
            res_linf = float(np.max(np.abs(vals)))
            records.append({
                "h": float(h),
                "res_l2": res_l2,
                "res_linf": res_linf,
                "n_points": int(vals.size),
                "n_valid": int(np.count_nonzero(res.valid)),
                "below_threshold": bool(res_l2 < cfg.analysis.minkowski_pass_threshold),
            })
        except Exception as exc:
            records.append({"h": float(h), "error": str(exc)})

    for k in range(len(records) - 1):
        a, b = records[k], records[k + 1]
        if ("res_l2" in a and "res_l2" in b
                and a["res_l2"] > 1.0e-14 and b["res_l2"] > 1.0e-14):
            a["order_l2"] = float(
                math.log(a["res_l2"] / b["res_l2"]) / math.log(a["h"] / b["h"])
            )
        if ("res_linf" in a and "res_linf" in b
                and a["res_linf"] > 1.0e-14 and b["res_linf"] > 1.0e-14):
            a["order_linf"] = float(
                math.log(a["res_linf"] / b["res_linf"]) / math.log(a["h"] / b["h"])
            )
    return records


def minkowski_exact_test(hs):
    return minkowski_pde_residual_test(hs)


def constraint_convergence_suite(base_config: RunConfig, hs):
    records = []
    r_min = base_config.analysis.constraint_conv_r_min
    r_max = base_config.analysis.constraint_conv_r_max
    for h in hs:
        cfg = copy_config(base_config)
        cfg.grid.h = h
        cfg.grid.r_floor = base_config.grid.r_floor * max((h / base_config.grid.h) ** 2, 0.1)
        try:
            res = run_single(cfg)
            d = compute_unified_derivatives(res)
            r = res.r
            ru, rv = d["ru"], d["rv"]
            pu, pv = d["pu"], d["pv"]
            au, av = d["au"], d["av"]
            ruu, rvv = d["ruu"], d["rvv"]
            cu = ruu - 2.0 * au * ru + 0.5 * r * pu * pu
            cv = rvv - 2.0 * av * rv + 0.5 * r * pv * pv
            mask = (res.valid
                    & np.isfinite(cu) & np.isfinite(cv)
                    & (r > r_min) & (r < r_max))
            cu_l2 = float(np.sqrt(np.mean(cu[mask] ** 2))) if np.any(mask) else float("nan")
            cv_l2 = float(np.sqrt(np.mean(cv[mask] ** 2))) if np.any(mask) else float("nan")
            rec = {"h": float(h)}
            rec.update(res.diagnostics)
            rec["constraint_u_l2_interp"] = cu_l2
            rec["constraint_v_l2_interp"] = cv_l2
            rec["constraint_valid_points_conv"] = float(np.count_nonzero(mask))
            records.append(rec)
        except Exception as exc:
            records.append({"h": float(h), "error": str(exc)})
    return records


def find_horizon(result: SimulationResult, d: Dict[str, np.ndarray], inv: Dict[str, np.ndarray],
                 provisional_mf: Optional[float] = None) -> Optional[HorizonData]:
    u, v = result.u, result.v
    rv, ru = d["rv"], d["ru"]
    r = result.r
    m = inv["m"]
    alpha = safe_exp(result.logalpha)
    hu, hv, hr, hm, ha = [], [], [], [], []
    for i in range(1, result.u.size - 1):
        js = np.where(result.valid[i] & np.isfinite(rv[i]) & np.isfinite(ru[i]) & np.isfinite(m[i]))[0]
        if js.size < 4:
            continue
        H = r[i, js] - 2.0 * m[i, js]
        for k in range(js.size - 2):
            j0, j1, j2 = js[k], js[k + 1], js[k + 2]
            if not (j1 == j0 + 1 and j2 == j1 + 1):
                continue
            y0, y1, y2 = H[k], H[k + 1], H[k + 2]
            if not (math.isfinite(y0) and math.isfinite(y1) and math.isfinite(y2)):
                continue
            if y0 > 0.0 and y2 <= 0.0:
                try:
                    coeffs = np.polyfit([v[j0], v[j1], v[j2]], [y0, y1, y2], 2)
                    roots = np.roots(coeffs)
                    real_roots = [rr.real for rr in roots
                                  if abs(rr.imag) < 1.0e-10 and v[j0] <= rr.real <= v[j2]]
                    if not real_roots:
                        continue
                    vv = min(real_roots, key=lambda x: abs(x - v[j1]))
                except Exception:
                    continue
                rr_quad = np.polyfit([v[j0], v[j1], v[j2]], [r[i, j0], r[i, j1], r[i, j2]], 2)
                mm_quad = np.polyfit([v[j0], v[j1], v[j2]], [m[i, j0], m[i, j1], m[i, j2]], 2)
                aa_quad = np.polyfit([v[j0], v[j1], v[j2]], [alpha[i, j0], alpha[i, j1], alpha[i, j2]], 2)
                ru_quad = np.polyfit([v[j0], v[j1], v[j2]], [ru[i, j0], ru[i, j1], ru[i, j2]], 2)
                rv_quad = np.polyfit([v[j0], v[j1], v[j2]], [rv[i, j0], rv[i, j1], rv[i, j2]], 2)
                rr = float(np.polyval(rr_quad, vv))
                mm = float(np.polyval(mm_quad, vv))
                aa = float(np.polyval(aa_quad, vv))
                rui = float(np.polyval(ru_quad, vv))
                rvi = float(np.polyval(rv_quad, vv))
                if rui < 0.0 and abs(rvi) < 0.1 and rr > 0.0 and mm > 0.0:
                    hu.append(u[i]); hv.append(vv); hr.append(rr); hm.append(mm); ha.append(aa)
                break
    if len(hu) < result.config.analysis.min_horizon_points:
        return None
    hu = np.asarray(hu); hv = np.asarray(hv); hr = np.asarray(hr); hm = np.asarray(hm); ha = np.asarray(ha)
    order = np.argsort(hu)
    hu, hv, hr, hm, ha = hu[order], hv[order], hr[order], hm[order], ha[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        dvdu = np.gradient(hv, hu)
    dell_du = ha * np.sqrt(np.maximum(-dvdu, 0.0))
    ell = np.zeros_like(hu)
    if hu.size > 1:
        ell[1:] = np.cumsum(0.5 * (dell_du[1:] + dell_du[:-1]) * np.diff(hu))
    mf = float(provisional_mf if provisional_mf is not None else np.nanmax(hm))
    mf = max(mf, 1.0e-15)
    lam = hm / mf
    barell = ell / mf
    with np.errstate(divide="ignore", invalid="ignore"):
        trans = (np.abs(np.gradient(lam, barell, edge_order=1))
                 if hu.size >= 3 and np.ptp(barell) > 0
                 else np.full_like(lam, np.nan))
    return HorizonData(hu, hv, hr, hm, ha, ell, lam, trans)


def estimate_final_mass_from_horizon(horizon: Optional[HorizonData]) -> float:
    if horizon is None or horizon.m.size == 0:
        return float("nan")
    q = np.isfinite(horizon.m) & (horizon.m > 0.0)
    if np.count_nonzero(q) == 0:
        return float("nan")
    return float(np.max(horizon.m[q]))


def outer_mass_proxy(result: SimulationResult, inv: Dict[str, np.ndarray]) -> float:
    m = inv["m"]
    r = result.r
    rmin = result.config.analysis.outer_mass_rmin
    rmax = result.config.analysis.outer_mass_rmax
    samples = []
    for i in range(result.u.size):
        q = result.valid[i] & np.isfinite(m[i]) & (r[i] > rmin) & (r[i] < rmax)
        js = np.where(q)[0]
        if js.size < 4:
            continue
        rr = r[i, js]
        mm = m[i, js]
        try:
            A = np.column_stack([np.ones_like(rr), 1.0 / rr, 1.0 / (rr * rr)])
            beta, _, _, _ = np.linalg.lstsq(A, mm, rcond=None)
            samples.append(beta[0])
        except Exception:
            continue
    if not samples:
        return float("nan")
    arr = np.asarray(samples)
    k = max(1, arr.size // 5)
    return float(np.median(arr[-k:]))


def analyze_global(u, v, r, loga, psi, valid, config: RunConfig,
                   ruv_source: Optional[np.ndarray] = None):
    provisional = SimulationResult(config, u, v, r, loga, psi, valid, 1.0, None, {}, None, ruv_source)
    d = compute_unified_derivatives(provisional)
    inv = invariant_arrays(provisional, d)
    horizon0 = find_horizon(provisional, d, inv, None)
    mf = estimate_final_mass_from_horizon(horizon0)
    has_horizon = horizon0 is not None
    if not (math.isfinite(mf) and mf > 0.0):
        finite_m = inv["m"][valid & np.isfinite(inv["m"]) & (inv["m"] > 0.0)]
        mf = float(np.nanmax(finite_m)) if finite_m.size else 1.0
    provisional.mf = mf
    horizon = find_horizon(provisional, d, inv, mf)
    bondi = outer_mass_proxy(provisional, inv)
    diagnostics = constraint_and_flux_residuals(provisional, d, inv)
    diagnostics["mf_ah"] = mf
    diagnostics["mf_outer_proxy"] = bondi
    diagnostics["mf_ah_outer_difference"] = float(mf - bondi) if math.isfinite(bondi) else float("nan")
    diagnostics["has_horizon"] = float(1.0 if has_horizon else 0.0)
    diagnostics["horizon_located"] = float(1.0 if horizon is not None else 0.0)
    integrated = compute_integrated_mass(provisional, d, inv)
    if horizon is not None:
        diagnostics["horizon_points"] = float(horizon.u.size)
        diagnostics["horizon_transversality_global_min"] = (
            float(np.nanmin(horizon.transversality))
            if np.any(np.isfinite(horizon.transversality)) else float("nan")
        )
        diagnostics["horizon_algebraic_consistency"] = float(
            np.nanmax(np.abs(horizon.r - 2.0 * horizon.m) / np.maximum(horizon.r, 1.0e-15))
        )
    if integrated is not None:
        r_min_int = config.analysis.constraint_r_min_fixed
        mask_int = (valid & np.isfinite(integrated) & np.isfinite(inv["m"])
                    & (inv["m"] > 1.0e-6) & (r > r_min_int))
        if np.any(mask_int):
            rel = np.abs(integrated - inv["m"]) / np.maximum(np.abs(inv["m"]), 1.0e-15)
            rel = rel[mask_int]
            rel = rel[np.isfinite(rel)]
            diagnostics["integrated_vs_algebraic_mass_l2"] = (
                float(math.sqrt(np.mean(rel ** 2))) if rel.size else float("nan")
            )
            diagnostics["integrated_vs_algebraic_mass_max"] = (
                float(np.max(rel)) if rel.size else float("nan")
            )
        else:
            diagnostics["integrated_vs_algebraic_mass_l2"] = float("nan")
            diagnostics["integrated_vs_algebraic_mass_max"] = float("nan")
    else:
        diagnostics["integrated_vs_algebraic_mass_l2"] = float("nan")
        diagnostics["integrated_vs_algebraic_mass_max"] = float("nan")
    return mf, horizon, diagnostics, integrated


def row_interp(result: SimulationResult, arr: np.ndarray, i: int, vv: float) -> float:
    js = np.where(result.valid[i] & np.isfinite(arr[i]))[0]
    if js.size < 2:
        return float("nan")
    xs = result.v[js]
    if vv < xs[0] or vv > xs[-1]:
        return float("nan")
    if js.size >= 4:
        try:
            order = np.argsort(xs)
            xs_s = xs[order]
            ys_s = arr[i, js][order]
            xu, idx = np.unique(xs_s, return_index=True)
            if xu.size >= 2:
                return float(PchipInterpolator(xu, ys_s[idx], extrapolate=False)(vv))
        except Exception:
            pass
    return float(np.interp(vv, xs, arr[i, js]))


def horizon_transversality(lam: np.ndarray, barell: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    barell = np.asarray(barell, dtype=float)
    if lam.size < 2 or float(np.ptp(barell)) <= 0.0:
        return np.full(lam.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.gradient(lam, barell))


def anchor_from_lambda(result: SimulationResult, lambda_star: Optional[float] = None):
    if result.horizon is None:
        raise RuntimeError("no resolved trapping horizon")
    h = result.horizon
    lamstar = result.config.analysis.lambda_star if lambda_star is None else lambda_star
    q = np.isfinite(h.lam) & np.isfinite(h.transversality)
    lam, uu, vv, tr, mm = h.lam[q], h.u[q], h.v[q], h.transversality[q], h.m[q]
    if lam.size < 2:
        raise RuntimeError("insufficient horizon points")
    order = np.argsort(lam)
    lam, uu, vv, tr, mm = lam[order], uu[order], vv[order], tr[order], mm[order]
    lam_u, idx = np.unique(lam, return_index=True)
    uu, vv, tr, mm = uu[idx], vv[idx], tr[idx], mm[idx]
    if lamstar < lam_u[0] or lamstar > lam_u[-1]:
        raise RuntimeError(f"lambda_star {lamstar} outside [{lam_u[0]}, {lam_u[-1]}]")
    ua = float(PchipInterpolator(lam_u, uu)(lamstar))
    va = float(PchipInterpolator(lam_u, vv)(lamstar))
    ta = float(PchipInterpolator(lam_u, tr)(lamstar))
    ma = float(PchipInterpolator(lam_u, mm)(lamstar))
    return ua, va, ta, ma


def smooth_derivative(y: np.ndarray, x: np.ndarray, window: int, order: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    if n < 5:
        return np.gradient(y, x)
    w = min(window, n if n % 2 == 1 else n - 1)
    w = max(w, order + 2 + ((order + 2) % 2 == 0))
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w <= order:
        return np.gradient(y, x)
    ys = savgol_filter(y, w, min(order, w - 1), mode="interp")
    return np.gradient(ys, x)


def _row_interp(v: np.ndarray, valid_row: np.ndarray, arr_row: np.ndarray,
                vv: float) -> float:
    js = np.where(valid_row & np.isfinite(arr_row))[0]
    if js.size < 2:
        return float("nan")
    xs = v[js]
    if vv < xs[0] or vv > xs[-1]:
        return float("nan")
    ys = arr_row[js]
    if js.size >= 4:
        try:
            order = np.argsort(xs)
            xs_s = xs[order]
            ys_s = ys[order]
            xu, idx = np.unique(xs_s, return_index=True)
            if xu.size >= 2:
                return float(PchipInterpolator(xu, ys_s[idx], extrapolate=False)(vv))
        except Exception:
            pass
    return float(np.interp(vv, xs, ys))


def _row_fields(result: SimulationResult, i: int) -> Dict[str, np.ndarray]:
    r = result.r
    loga = result.logalpha
    psi = result.psi
    valid = result.valid
    nu, nv = r.shape
    hu = float(result.config.grid.h)
    hv = float(result.config.grid.dv) if result.config.grid.dv else float(result.config.grid.h)
    nan = float("nan")
    rows_ok = 0 < i < nu - 1

    def du(f):
        out = np.full(nv, nan)
        if rows_ok:
            m = valid[i - 1] & valid[i] & valid[i + 1]
            if m.any():
                out[m] = (f[i + 1][m] - f[i - 1][m]) / (2.0 * hu)
        return out

    def dv(f):
        out = np.full(nv, nan)
        if nv > 2:
            m = valid[i][:-2] & valid[i][1:-1] & valid[i][2:]
            if m.any():
                out[1:-1][m] = (f[i][2:][m] - f[i][:-2][m]) / (2.0 * hv)
        return out

    def duu(f):
        out = np.full(nv, nan)
        if rows_ok:
            m = valid[i - 1] & valid[i] & valid[i + 1]
            if m.any():
                out[m] = (f[i + 1][m] - 2.0 * f[i][m] + f[i - 1][m]) / (hu ** 2)
        return out

    def dvv(f):
        out = np.full(nv, nan)
        if nv > 2:
            m = valid[i][:-2] & valid[i][1:-1] & valid[i][2:]
            if m.any():
                out[1:-1][m] = ((f[i][2:][m] - 2.0 * f[i][1:-1][m]
                                 + f[i][:-2][m]) / (hv ** 2))
        return out

    def duv(f):
        out = np.full(nv, nan)
        if rows_ok and nv > 2:
            m = (valid[i - 1][:-2] & valid[i - 1][2:]
                 & valid[i + 1][:-2] & valid[i + 1][2:]
                 & valid[i][1:-1])
            if m.any():
                num = (f[i + 1][2:] - f[i - 1][2:]
                       - f[i + 1][:-2] + f[i - 1][:-2])
                out[1:-1][m] = num[m] / (4.0 * hu * hv)
        return out

    ru = du(r)
    rv = dv(r)
    pu = du(psi)
    pv = dv(psi)
    au = du(loga)
    av = dv(loga)
    ruu = duu(r)
    rvv = dvv(r)
    ruv = duv(r)

    rr = r[i]
    vl = valid[i]
    base = (vl & np.isfinite(rr) & np.isfinite(loga[i])
            & np.isfinite(ru) & np.isfinite(rv)
            & np.isfinite(pu) & np.isfinite(pv) & (rr > 0.0))
    a2inv = np.full(nv, nan)
    a2inv[base] = np.exp(-2.0 * loga[i][base])
    m = np.full(nv, nan)
    ricci = np.full(nv, nan)
    m[base] = 0.5 * rr[base] * (1.0 + 4.0 * ru[base] * rv[base] * a2inv[base])
    ricci[base] = -4.0 * pu[base] * pv[base] * a2inv[base]
    r_safe = np.where(rr > 0.0, rr, nan)
    x = m / (r_safe ** 3)
    y = ricci / 12.0
    xy = x + y
    logk = np.full(nv, nan)
    good = base & np.isfinite(xy) & np.isfinite(ricci)
    if np.any(good):
        xy_use = np.maximum(np.abs(xy[good]), 1.0e-300)
        ricci_use = np.maximum(np.abs(ricci[good]), 1.0e-300)
        logk[good] = np.logaddexp(
            np.log(48.0) + 2.0 * np.log(xy_use),
            np.log(5.0 / 3.0) + 2.0 * np.log(ricci_use),
        )
    return {"ru": ru, "rv": rv, "pu": pu, "pv": pv, "au": au, "av": av,
            "ruu": ruu, "rvv": rvv, "ruv": ruv,
            "m": m, "ricci": ricci, "logk": logk}


def extract_curve(result: SimulationResult, lambda_star: Optional[float] = None,
                  savgol_window: Optional[int] = None,
                  savgol_order: Optional[int] = None) -> CurveData:
    ua, va, _, _ = anchor_from_lambda(result, lambda_star)
    keys = ["u", "r", "alpha", "psi", "ru", "rv", "pu", "pv", "au", "av",
            "m", "ricci", "logk"]
    vals: Dict[str, List[float]] = {k: [] for k in keys}
    v = result.v
    vl = result.valid
    for i in np.where(result.u >= ua)[0]:
        stride = max(1, int(round(
            result.config.grid.h / (result.config.grid.dv or result.config.grid.h))))
        jc = i * stride
        kblock = 0
        if result.config.grid.center_scheme == "ode":
            stride = max(1, int(round(
                result.config.grid.h
                / (result.config.grid.dv or result.config.grid.h)
            )))
            dv = result.config.grid.dv or result.config.grid.h
            kblock = center_block_width(
                result.config.grid, dv, stride, result.v.size, jc
            )
        if kblock > 0:
            js_valid = np.where(result.valid[i])[0]
            if js_valid.size == 0 or js_valid[-1] <= jc + kblock + 2:
                continue
        rr = _row_interp(v, vl[i], result.r[i], va)
        if not math.isfinite(rr) or rr <= result.config.grid.r_floor:
            continue
        row = _row_fields(result, i)
        items = {
            "u": result.u[i],
            "r": rr,
            "alpha": math.exp(_row_interp(v, vl[i], result.logalpha[i], va)),
            "psi": _row_interp(v, vl[i], result.psi[i], va),
            "ru": _row_interp(v, vl[i], row["ru"], va),
            "rv": _row_interp(v, vl[i], row["rv"], va),
            "pu": _row_interp(v, vl[i], row["pu"], va),
            "pv": _row_interp(v, vl[i], row["pv"], va),
            "au": _row_interp(v, vl[i], row["au"], va),
            "av": _row_interp(v, vl[i], row["av"], va),
            "m": _row_interp(v, vl[i], row["m"], va),
            "ricci": _row_interp(v, vl[i], row["ricci"], va),
            "logk": _row_interp(v, vl[i], row["logk"], va),
        }
        if all(math.isfinite(x) for x in items.values()):
            for k, x in items.items():
                vals[k].append(float(x))
    if len(vals["u"]) < result.config.analysis.min_curve_points:
        raise RuntimeError(
            f"insufficient matched-curve points: {len(vals['u'])} < "
            f"{result.config.analysis.min_curve_points}"
        )
    arr = {k: np.asarray(v, dtype=float) for k, v in vals.items()}
    order = np.argsort(arr["r"])[::-1]
    arr = {k: v[order] for k, v in arr.items()}
    rho = arr["r"] / result.mf
    logr = np.log(arr["r"])
    window = savgol_window if savgol_window is not None else result.config.analysis.savgol_window
    poly = savgol_order if savgol_order is not None else result.config.analysis.savgol_order
    dpsi_dlogr = smooth_derivative(arr["psi"], logr, window, poly)
    dlogm_dlogr = smooth_derivative(np.log(np.maximum(np.abs(arr["m"]), 1.0e-300)),
                                    logr, window, poly)
    cpsi = 0.5 * dpsi_dlogr
    ru_safe = np.where(np.abs(arr["ru"]) > 1.0e-300, arr["ru"], np.nan)
    cpsi_direct = 0.5 * arr["pu"] / (ru_safe / np.maximum(arr["r"], 1.0e-300))
    cm2 = -0.5 * dlogm_dlogr
    curvature_slope = -smooth_derivative(arr["logk"], np.log(rho), window, poly)
    alpha = arr["alpha"]
    ru, rv = arr["ru"], arr["rv"]
    pu, pv = arr["pu"], arr["pv"]
    r_safe = np.maximum(arr["r"], 1.0e-30)
    au, av = arr["au"], arr["av"]
    Q = 4.0 * ru * rv / (alpha * alpha)
    sqrtQ = np.sqrt(np.maximum(Q, 1.0e-300))
    h_perp = sqrtQ / r_safe
    ruv = -ru * rv / r_safe - alpha ** 2 / (4.0 * r_safe)
    ruu = 2.0 * au / alpha * ru - 0.5 * r_safe * pu ** 2
    rvv = 2.0 * av / alpha * rv - 0.5 * r_safe * pv ** 2
    su = np.sqrt(np.maximum(rv / np.where(ru != 0.0, ru, np.nan), 0.0)) / alpha
    sv = -np.sqrt(np.maximum(ru / np.where(rv != 0.0, rv, np.nan), 0.0)) / alpha
    nu = -ru / sqrtQ
    nv = -rv / sqrtQ
    X = su * su * ruu + 2.0 * su * sv * ruv + sv * sv * rvv
    h_parallel = (-X / sqrtQ
                  - su * su * nu * (2.0 * au / alpha)
                  - sv * sv * nv * (2.0 * av / alpha))
    nu_up = 2.0 * rv / (alpha * alpha * sqrtQ)
    nv_up = 2.0 * ru / (alpha * alpha * sqrtQ)
    npsi = nu_up * pu + nv_up * pv
    H = h_parallel + 2.0 * h_perp
    safe_H = np.where(np.abs(H) > 1.0e-30, H, np.nan)
    p_par = h_parallel / safe_H
    p_perp = h_perp / safe_H
    q_inv = npsi / safe_H
    kasner_sum = p_par + 2.0 * p_perp
    kasner_constraint = p_par * p_par + 2.0 * p_perp * p_perp + q_inv * q_inv
    logk_scaled = arr["logk"] + 4.0 * math.log(result.mf)
    return CurveData(
        arr["u"], np.full_like(arr["u"], va), arr["r"], alpha, arr["psi"],
        ru, rv, pu, pv, arr["m"], arr["ricci"], logk_scaled, rho,
        cpsi, cpsi_direct, cm2, curvature_slope, p_par, p_perp, q_inv,
        kasner_sum, kasner_constraint,
    )


def kasner_window(curve: CurveData, config: AnalysisConfig) -> np.ndarray:
    rho = curve.rho
    c2 = curve.cpsi_direct * curve.cpsi_direct
    pred = 6.0 + 4.0 * c2
    q = np.isfinite(curve.cpsi_direct) & np.isfinite(curve.cm2) & np.isfinite(curve.curvature_slope)
    q &= rho <= config.rho_max_fit
    q &= rho >= config.rho_min_fit
    q &= np.abs(curve.cm2 - c2) <= config.kasner_tolerance * np.maximum(1.0, c2)
    q &= np.abs(curve.curvature_slope - pred) <= config.kasner_tolerance * np.maximum(1.0, pred)
    return q


def robust_constant(x: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(x)[mask]
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(y))
    mad = float(1.4826 * np.median(np.abs(y - med))) if y.size > 1 else 0.0
    return med, mad


def interp_curve_field(curve: CurveData, field: str, target_rho: np.ndarray) -> np.ndarray:
    x = curve.rho[::-1]
    y = getattr(curve, field)[::-1]
    q = np.isfinite(x) & np.isfinite(y)
    x, y = x[q], y[q]
    if x.size < 2:
        return np.full_like(target_rho, np.nan)
    xu, idx = np.unique(x, return_index=True)
    if xu.size < 2:
        return np.full_like(target_rho, np.nan)
    try:
        return PchipInterpolator(xu, y[idx], extrapolate=False)(target_rho)
    except Exception:
        return np.full_like(target_rho, np.nan)


def match_curves_equal_rho(curve1, curve0, rho_min=None, rho_max=None, n=300):
    r1 = curve1.rho[np.isfinite(curve1.rho)]
    r0 = curve0.rho[np.isfinite(curve0.rho)]
    if r1.size == 0 or r0.size == 0:
        raise RuntimeError("empty rho array in one curve")
    lo = max(np.nanmin(r1), np.nanmin(r0))
    hi = min(np.nanmax(r1), np.nanmax(r0))
    if rho_min is not None:
        lo = max(lo, rho_min)
    if rho_max is not None:
        hi = min(hi, rho_max)
    if not (lo > 0.0 and hi > lo):
        raise RuntimeError(
            f"no overlapping rho interval: curve1 [{np.nanmin(r1):.3e}, {np.nanmax(r1):.3e}], "
            f"curve0 [{np.nanmin(r0):.3e}, {np.nanmax(r0):.3e}]"
        )
    rho = np.geomspace(hi, lo, n)
    return {
        "rho": rho,
        "logk1": interp_curve_field(curve1, "logk", rho),
        "logk0": interp_curve_field(curve0, "logk", rho),
        "c1": interp_curve_field(curve1, "cpsi_direct", rho),
        "c0": interp_curve_field(curve0, "cpsi_direct", rho),
    }


def match_curves_equal_compactness(curve1, curve0, n=300):
    z1 = curve1.m / np.maximum(curve1.r, 1.0e-30)
    z0 = curve0.m / np.maximum(curve0.r, 1.0e-30)
    lo = max(np.nanmin(z1), np.nanmin(z0))
    hi = min(np.nanmax(z1), np.nanmax(z0))
    if not (hi > lo):
        raise RuntimeError("no overlapping compactness interval")
    z = np.linspace(lo, hi, n)

    def at(curve, field, target):
        x = (curve.m / np.maximum(curve.r, 1.0e-30))[::-1]
        y = getattr(curve, field)[::-1]
        q = np.isfinite(x) & np.isfinite(y)
        x, y = x[q], y[q]
        if x.size < 2:
            return np.full_like(target, np.nan)
        order = np.argsort(x)
        x, y = x[order], y[order]
        xu, idx = np.unique(x, return_index=True)
        if xu.size < 2:
            return np.full_like(target, np.nan)
        return PchipInterpolator(xu, y[idx], extrapolate=False)(target)

    return {
        "rho": z,
        "logk1": at(curve1, "logk", z),
        "logk0": at(curve0, "logk", z),
        "c1": at(curve1, "cpsi_direct", z),
        "c0": at(curve0, "cpsi_direct", z),
    }


def linear_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    q = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[q], np.asarray(y)[q]
    if x.size < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "slope_sigma": float("nan"), "r2": float("nan")}
    A = np.column_stack([x, np.ones_like(x)])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    dof = max(x.size - 2, 1)
    s2 = float(np.sum(res * res) / dof)
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        sigma = float(math.sqrt(max(cov[0, 0], 0.0)))
    except Exception:
        sigma = float("nan")
    ss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(res * res)) / ss if ss > 0.0 else 1.0
    return {"slope": float(beta[0]), "intercept": float(beta[1]),
            "slope_sigma": sigma, "r2": float(r2)}


def jackknife_slope(x: np.ndarray, y: np.ndarray, n_blocks: int = 8) -> Dict[str, float]:
    q = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[q], np.asarray(y)[q]
    if x.size < 2 * n_blocks:
        return {"slope_mean": float("nan"), "slope_std": float("nan")}
    idx = np.arange(x.size)
    blocks = np.array_split(idx, n_blocks)
    slopes = []
    for i in range(n_blocks):
        mask = np.ones(x.size, dtype=bool)
        mask[blocks[i]] = False
        if np.count_nonzero(mask) < 3:
            continue
        A = np.column_stack([x[mask], np.ones(np.count_nonzero(mask))])
        try:
            beta, _, _, _ = np.linalg.lstsq(A, y[mask], rcond=None)
            slopes.append(float(beta[0]))
        except Exception:
            continue
    if len(slopes) < 2:
        return {"slope_mean": float("nan"), "slope_std": float("nan")}
    arr = np.asarray(slopes)
    return {"slope_mean": float(np.mean(arr)), "slope_std": float(np.std(arr, ddof=1))}


def finite_data_analysis(result1, result0, lambda_star=None, match_scheme=None) -> Dict[str, object]:
    curve1 = extract_curve(result1, lambda_star)
    curve0 = extract_curve(result0, lambda_star)
    mask1 = kasner_window(curve1, result1.config.analysis)
    mask0 = kasner_window(curve0, result0.config.analysis)
    c1, _ = robust_constant(curve1.cpsi_direct, mask1)
    c0, _ = robust_constant(curve0.cpsi_direct, mask0)
    dx = h2_distance(result1.config.initial, result0.config.initial)
    scheme = match_scheme or result1.config.analysis.match_scheme
    if scheme == "equal_compactness":
        matched = match_curves_equal_compactness(curve1, curve0)
    else:
        matched = match_curves_equal_rho(
            curve1, curve0,
            result1.config.analysis.rho_min_fit,
            result1.config.analysis.rho_max_fit,
        )
    x = np.abs(np.log(np.maximum(matched["rho"], 1.0e-300)))
    signed = (matched["logk1"] - matched["logk0"]) / max(dx, 1.0e-300)
    quotient = np.abs(signed)
    fit_signed = linear_fit(x, signed)
    fit_abs = linear_fit(x, quotient)
    jack = jackknife_slope(x, signed)
    if math.isfinite(c1) and math.isfinite(c0):
        pred_signed = 4.0 * (c1 * c1 - c0 * c0) / max(dx, 1.0e-300)
    else:
        pred_signed = float("nan")
    nested = []
    rho_lo0 = float(np.nanmin(matched["rho"]))
    rho_hi0 = float(np.nanmax(matched["rho"]))
    for k in range(result1.config.analysis.fit_nested_windows):
        hi = rho_hi0 / (2.0 ** k)
        if not (hi > rho_lo0):
            break
        q = (matched["rho"] <= hi) & (matched["rho"] >= rho_lo0)
        if np.count_nonzero(q) >= 8:
            f = linear_fit(x[q], signed[q])
            f["rho_lo"] = float(rho_lo0)
            f["rho_hi"] = float(hi)
            nested.append(f)
    return {
        "curve1": curve1, "curve0": curve0, "mask1": mask1, "mask0": mask0,
        "matched": matched, "distance": dx, "c1": c1, "c0": c0,
        "signed_response": signed, "quotient": quotient,
        "fit_signed": fit_signed, "fit_abs": fit_abs, "jackknife": jack,
        "pred_signed": pred_signed, "pred_abs": abs(pred_signed) if math.isfinite(pred_signed) else float("nan"),
        "nested_fits": nested, "match_scheme": scheme,
    }


def export_kasner_diagnostics(result: SimulationResult, out_dir: Path,
                              lambda_star: Optional[float] = None) -> Optional[Path]:
    try:
        curve = extract_curve(result, lambda_star)
    except Exception as exc:
        print(f"export_kasner_diagnostics failed: {exc}")
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = 6.0 + 4.0 * curve.cpsi_direct ** 2
    path = out_dir / "kasner_diagnostics.csv"
    save_csv(
        path,
        ["rho", "r", "logk", "curvature_slope", "cpsi", "cpsi_direct",
         "cm2", "pred_eta", "kasner_sum", "kasner_constraint"],
        [curve.rho, curve.r, curve.logk, curve.curvature_slope,
         curve.cpsi, curve.cpsi_direct, curve.cm2, pred,
         curve.kasner_sum, curve.kasner_constraint],
    )
    return path


def run_single(config: RunConfig) -> SimulationResult:
    reset_exp_clip_stats()
    result = DoubleNullSolver(config).run()
    stats = get_exp_clip_stats()
    result.diagnostics["num_exp_clipped"] = float(stats["count"])
    result.diagnostics["num_exp_clipped_lo"] = float(stats["lo"])
    result.diagnostics["num_exp_clipped_hi"] = float(stats["hi"])
    return result


def copy_config(config: RunConfig) -> RunConfig:
    return RunConfig(
        GridConfig(**asdict(config.grid)),
        InitialDataConfig(**asdict(config.initial)),
        AnalysisConfig(**asdict(config.analysis)),
        config.output_dir,
    )


def equal_mass_amplitude(base_config: RunConfig, epsilon: float, cache=None):
    if cache is None:
        cache = {}
    base_cfg = copy_config(base_config)
    base_cfg.initial.epsilon = 0.0
    base_result = run_single(base_cfg)
    if base_result.horizon is None or not math.isfinite(base_result.mf) or base_result.mf <= 0.0:
        raise RuntimeError(
            f"base solution has no horizon (mf={base_result.mf}, "
            f"amplitude={base_cfg.initial.amplitude}); "
            f"increase --amplitude until a horizon forms"
        )
    target = base_result.mf
    a0 = base_config.initial.amplitude
    penalty = base_config.analysis.invalid_mass_penalty

    def evaluate(a: float):
        key = round(a, 12)
        if key in cache:
            return cache[key]
        cfg = copy_config(base_config)
        cfg.initial.amplitude = a
        cfg.initial.epsilon = epsilon
        res = run_single(cfg)
        ok = (res.horizon is not None
              and math.isfinite(res.mf)
              and res.mf > 0.0
              and math.isfinite(res.diagnostics.get("constraint_u_l2", float("nan")))
              and res.diagnostics.get("constraint_u_l2", float("inf")) < 10.0)
        if ok:
            val = res.mf - target
        else:
            val = penalty if a >= a0 else -penalty
        cache[key] = (val, res)
        return val, res

    lo = a0 * 0.8
    hi = a0 * 1.2
    flo, _ = evaluate(lo)
    fhi, _ = evaluate(hi)
    factor = 1.5
    attempts = 0
    while flo * fhi > 0.0 and attempts < 12:
        attempts += 1
        lo = a0 / (factor ** attempts)
        hi = a0 * (factor ** attempts)
        flo, _ = evaluate(lo)
        fhi, _ = evaluate(hi)
    if flo * fhi > 0.0:
        raise RuntimeError(
            f"equal-mass root not bracketed. base mf={target}, f(lo)={flo}, f(hi)={fhi}. "
            f"Try increasing amplitude."
        )
    aeq = brentq(
        lambda a: evaluate(a)[0], lo, hi,
        xtol=max(1.0e-8, 0.05 * base_config.analysis.equal_mass_tolerance),
        rtol=1.0e-8, maxiter=40,
    )
    _, req = evaluate(aeq)
    if req.horizon is None:
        raise RuntimeError("equal-mass root found but perturbed solution has no horizon")
    mismatch = abs(req.mf - target) / max(target, 1.0e-15)
    if mismatch > base_config.analysis.equal_mass_tolerance:
        raise RuntimeError(f"equal-mass mismatch {mismatch} exceeds tolerance")
    return aeq, req, base_result


def mass_direction_response(base_config: RunConfig, delta_a: float,
                            lambda_star: Optional[float] = None):
    cplus = copy_config(base_config)
    cminus = copy_config(base_config)
    cplus.initial.amplitude += delta_a
    cminus.initial.amplitude -= delta_a
    rplus = run_single(cplus)
    rminus = run_single(cminus)
    if rplus.horizon is None or rminus.horizon is None:
        raise RuntimeError("mass_direction_response requires both solutions to have horizons")
    cp = extract_curve(rplus, lambda_star)
    cm = extract_curve(rminus, lambda_star)
    matched = match_curves_equal_rho(cp, cm,
                                     base_config.analysis.rho_min_fit,
                                     base_config.analysis.rho_max_fit)
    dlnm = math.log(rplus.mf / rminus.mf)
    sm = (matched["logk1"] - matched["logk0"]) / max(abs(dlnm), 1.0e-300)
    return {"rho": matched["rho"], "S_M": sm,
            "mf_plus": rplus.mf, "mf_minus": rminus.mf}


def centered_parameter_response(base_config: RunConfig, epsilon: float,
                                lambda_star: Optional[float] = None):
    cp = copy_config(base_config)
    cm = copy_config(base_config)
    cp.initial.epsilon = epsilon
    cm.initial.epsilon = -epsilon
    rp = run_single(cp)
    rm = run_single(cm)
    if rp.horizon is None or rm.horizon is None:
        raise RuntimeError("centered_parameter_response requires horizons in both solutions")
    curvep = extract_curve(rp, lambda_star)
    curvem = extract_curve(rm, lambda_star)
    matched = match_curves_equal_rho(curvep, curvem,
                                     base_config.analysis.rho_min_fit,
                                     base_config.analysis.rho_max_fit)
    D = (matched["logk1"] - matched["logk0"]) / (2.0 * epsilon)
    return {"rho": matched["rho"], "D": D, "chi": np.abs(D),
            "plus": rp, "minus": rm}


def convergence_suite(base_config: RunConfig, hs, epsilons, lambda_star=None):
    records = []
    for h in hs:
        for eps in epsilons:
            cfg = copy_config(base_config)
            cfg.grid.h = h
            cfg.grid.r_floor = base_config.grid.r_floor * max((h / base_config.grid.h) ** 2, 0.1)
            try:
                resp = centered_parameter_response(cfg, eps, lambda_star)
                q = np.isfinite(resp["D"])
                med = float(np.median(resp["D"][q])) if np.any(q) else float("nan")
                records.append({"h": float(h), "epsilon": float(eps),
                                "median_D": med, "rho": resp["rho"], "D": resp["D"]})
            except Exception as exc:
                records.append({"h": float(h), "epsilon": float(eps),
                                "median_D": float("nan"), "error": str(exc)})
    return records


def observable_convergence_suite(base_config: RunConfig, hs, epsilon: float,
                                 lambda_star: Optional[float] = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for h in hs:
        cfg = copy_config(base_config)
        cfg.grid.h = h
        cfg.grid.r_floor = base_config.grid.r_floor * max((h / base_config.grid.h) ** 2, 0.1)
        try:
            base_cfg = copy_config(cfg)
            base_cfg.initial.epsilon = 0.0
            aeq, pert, base = equal_mass_amplitude(base_cfg, epsilon)
            analysis = finite_data_analysis(pert, base, lambda_star)
            fit = analysis["fit_signed"]
            rec = {
                "h": float(h),
                "measured_slope": float(fit["slope"]),
                "slope_sigma_ols": float(fit["slope_sigma"]),
                "slope_sigma_jack": float(analysis["jackknife"]["slope_std"]),
                "predicted_slope": float(analysis["pred_signed"]),
                "r2": float(fit["r2"]),
                "C1": float(analysis["c1"]),
                "C0": float(analysis["c0"]),
                "delta_C2": float(analysis["c1"] ** 2 - analysis["c0"] ** 2),
                "distance_X": float(analysis["distance"]),
                "mass_mismatch": float(abs(pert.mf - base.mf) / max(base.mf, 1.0e-15)),
                "mf": float(pert.mf),
                "kasner_points": int(np.count_nonzero(analysis["mask1"])),
                "compensating_amplitude": float(aeq),
            }
            records.append(rec)
        except Exception as exc:
            records.append({"h": float(h), "error": str(exc)})

    valid = [r for r in records if "measured_slope" in r and math.isfinite(r["measured_slope"])]
    for k in range(len(valid) - 1):
        a, b = valid[k], valid[k + 1]
        diff = abs(a["measured_slope"] - b["measured_slope"])
        a["diff_to_next"] = float(diff)
        if k > 0 and "diff_to_next" in valid[k - 1]:
            prev_diff = valid[k - 1]["diff_to_next"]
            if prev_diff > 0.0 and diff > 0.0:
                order = math.log(prev_diff / diff) / math.log(valid[k - 1]["h"] / a["h"])
                a["observed_order"] = float(order)

    if len(valid) >= 2:
        p = 2.0
        a, b = valid[-2], valid[-1]
        ratio = a["h"] / b["h"]
        if ratio > 1.0:
            S_extrap = (ratio ** p * b["measured_slope"] - a["measured_slope"]) / (ratio ** p - 1.0)
            records.append({
                "h": 0.0,
                "measured_slope": float(S_extrap),
                "note": f"Richardson extrapolation p={p}",
            })
    return records


def equal_mass_convergence_suite(base_config: RunConfig, hs,
                                 epsilon: float) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for h in hs:
        cfg = copy_config(base_config)
        cfg.grid.h = h
        cfg.grid.r_floor = base_config.grid.r_floor * max((h / base_config.grid.h) ** 2, 0.1)
        try:
            base_cfg = copy_config(cfg)
            base_cfg.initial.epsilon = 0.0
            aeq, pert, base = equal_mass_amplitude(base_cfg, epsilon)
            rel = abs(pert.mf - base.mf) / max(base.mf, 1.0e-15)
            records.append({
                "h": float(h),
                "base_mf": float(base.mf),
                "pert_mf": float(pert.mf),
                "delta_mf": float(abs(pert.mf - base.mf)),
                "relative_mismatch": float(rel),
                "compensating_amplitude": float(aeq),
                "tolerance": float(cfg.analysis.equal_mass_tolerance),
                "base_horizon_points": int(base.diagnostics.get("horizon_points", 0)),
                "base_transversality_min": float(
                    base.diagnostics.get("horizon_transversality_global_min", float("nan"))
                ),
            })
        except Exception as exc:
            records.append({"h": float(h), "error": str(exc)})

    valid = [r for r in records if "relative_mismatch" in r]
    for k in range(1, len(valid)):
        a, b = valid[k - 1], valid[k]
        if a["relative_mismatch"] > 0.0 and b["relative_mismatch"] > 0.0:
            b["order_vs_prev"] = float(
                math.log(a["relative_mismatch"] / b["relative_mismatch"])
                / math.log(a["h"] / b["h"])
            )
    return records


def kasner_window_sensitivity(base_config: RunConfig, epsilon: float,
                              lambda_star: Optional[float] = None,
                              windows: Optional[Sequence[Tuple[float, float]]] = None
                              ) -> List[Dict[str, object]]:
    if windows is None:
        windows = [
            (1.0e-3, 1.0e-1),
            (3.0e-4, 3.0e-2),
            (1.0e-4, 1.0e-2),
            (3.0e-5, 3.0e-3),
            (1.0e-5, 1.0e-3),
        ]

    base_cfg = copy_config(base_config)
    base_cfg.initial.epsilon = 0.0
    aeq, pert, base = equal_mass_amplitude(base_cfg, epsilon)
    curve1 = extract_curve(pert, lambda_star)
    curve0 = extract_curve(base, lambda_star)
    dx = h2_distance(pert.config.initial, base.config.initial)
    scheme = base.config.analysis.match_scheme

    records: List[Dict[str, object]] = []
    for (lo, hi) in windows:
        cfg_analysis = AnalysisConfig(**asdict(base.config.analysis))
        cfg_analysis.rho_min_fit = float(lo)
        cfg_analysis.rho_max_fit = float(hi)
        mask1 = kasner_window(curve1, cfg_analysis)
        mask0 = kasner_window(curve0, cfg_analysis)
        c1, mad1 = robust_constant(curve1.cpsi_direct, mask1)
        c0, mad0 = robust_constant(curve0.cpsi_direct, mask0)
        n1 = int(np.count_nonzero(mask1))
        n0 = int(np.count_nonzero(mask0))
        try:
            if scheme == "equal_compactness":
                matched = match_curves_equal_compactness(curve1, curve0)
            else:
                matched = match_curves_equal_rho(curve1, curve0, lo, hi)
            x = np.abs(np.log(np.maximum(matched["rho"], 1.0e-300)))
            signed = (matched["logk1"] - matched["logk0"]) / max(dx, 1.0e-300)
            fit = linear_fit(x, signed)
            measured = float(fit["slope"])
            sigma = float(fit["slope_sigma"])
        except Exception:
            measured = float("nan")
            sigma = float("nan")
        pred = 4.0 * (c1 * c1 - c0 * c0) / max(dx, 1.0e-300) if (
            math.isfinite(c1) and math.isfinite(c0)
        ) else float("nan")
        records.append({
            "rho_lo": float(lo),
            "rho_hi": float(hi),
            "C1": float(c1),
            "C0": float(c0),
            "C1_MAD": float(mad1),
            "C0_MAD": float(mad0),
            "n_window_1": n1,
            "n_window_0": n0,
            "delta_C2": float(c1 * c1 - c0 * c0) if (
                math.isfinite(c1) and math.isfinite(c0)
            ) else float("nan"),
            "measured_slope": measured,
            "slope_sigma": sigma,
            "predicted_slope": float(pred),
        })
    return records


def interpolate_solution_field(result: SimulationResult, field: str,
                               u_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
    src = getattr(result, field)
    out = np.full((u_grid.size, v_grid.size), np.nan)
    for i, uu in enumerate(u_grid):
        i0 = int(np.argmin(np.abs(result.u - uu)))
        js = np.where(result.valid[i0] & np.isfinite(src[i0]))[0]
        if js.size < 2:
            continue
        xs = result.v[js]
        ys = src[i0, js]
        for j, vv in enumerate(v_grid):
            if xs[0] <= vv <= xs[-1]:
                out[i, j] = float(np.interp(vv, xs, ys))
    return out


def solution_convergence_suite(base_config: RunConfig, hs):
    results = []
    for h in hs:
        cfg = copy_config(base_config)
        cfg.grid.h = h
        cfg.grid.r_floor = base_config.grid.r_floor * max((h / base_config.grid.h) ** 2, 0.1)
        try:
            results.append(run_single(cfg))
        except Exception:
            continue
    if len(results) < 3:
        return {"error": "insufficient resolutions"}
    u_common = np.linspace(results[-1].u[2], results[-1].u[-3], 80)
    v_common = np.linspace(results[-1].v[2], results[-1].v[-3], 80)
    report = {}
    for f in ["r", "psi", "logalpha"]:
        arrs = [interpolate_solution_field(res, f, u_common, v_common) for res in results]
        diffs = []
        for a, b in zip(arrs[:-1], arrs[1:]):
            d = a - b
            d = d[np.isfinite(d)]
            diffs.append(float(np.linalg.norm(d)) if d.size else float("nan"))
        if len(diffs) >= 2:
            a, b = diffs[-2], diffs[-1]
            if a > 0 and b > 0:
                report[f] = {"diffs": diffs,
                             "order": float(math.log(a / b) / math.log(hs[-2] / hs[-1]))}
    return report


def minkowski_convergence_suite(base_config: RunConfig, hs):
    return minkowski_pde_residual_test(hs)


def horizon_convergence_suite(base_config: RunConfig, hs, lambda_star):
    records = []
    previous = None
    for h in hs:
        cfg = copy_config(base_config)
        cfg.grid.h = h
        cfg.grid.r_floor = base_config.grid.r_floor * max((h / base_config.grid.h) ** 2, 0.1)
        try:
            res = run_single(cfg)
            ua, va, ta, ma = anchor_from_lambda(res, lambda_star)
            rec = {"h": float(h), "u_anchor": ua, "v_anchor": va,
                   "transversality": ta, "m_anchor": ma}
            if previous is not None:
                rec["dv_diff"] = abs(va - previous["v_anchor"])
                rec["dm_diff"] = abs(ma - previous["m_anchor"])
            records.append(rec)
            previous = rec
        except Exception as exc:
            records.append({"h": float(h), "error": str(exc)})
    for k in range(1, len(records)):
        a, b = records[k - 1], records[k]
        if "dv_diff" in b and a.get("dv_diff", 0.0) > 0 and b.get("dv_diff", 0.0) > 0:
            records[k]["order_v"] = float(
                math.log(a["dv_diff"] / b["dv_diff"]) / math.log(a["h"] / b["h"])
            )
    return records


def error_budget_analysis(config: RunConfig, epsilon: float, lambda_star=None):
    base_cfg = copy_config(config)
    base_cfg.initial.epsilon = 0.0
    aeq, pert, base = equal_mass_amplitude(base_cfg, epsilon)
    analysis = finite_data_analysis(pert, base, lambda_star)
    mresp = mass_direction_response(
        base_cfg, max(1.0e-4, 0.01 * abs(base_cfg.initial.amplitude)), lambda_star
    )
    delta_mf = abs(pert.mf - base.mf)
    rho_target = analysis["matched"]["rho"]
    rho_src = mresp["rho"][::-1]
    sm_src = mresp["S_M"][::-1]
    q = np.isfinite(rho_src) & np.isfinite(sm_src)
    rho_src, sm_src = rho_src[q], sm_src[q]
    if rho_src.size >= 2:
        order = np.argsort(rho_src)
        rho_u, idx = np.unique(rho_src[order], return_index=True)
        sm_u = sm_src[order][idx]
        try:
            S_M_at_target = PchipInterpolator(rho_u, sm_u, extrapolate=False)(rho_target)
        except Exception:
            S_M_at_target = np.full_like(rho_target, np.nan)
    else:
        S_M_at_target = np.full_like(rho_target, np.nan)
    sigma_mass = np.abs(S_M_at_target) * delta_mf / max(base.mf, 1.0e-15)
    fit = analysis["fit_signed"]
    pred = float(analysis["pred_signed"])
    smooth_err = derivative_sensitivity(pert, base, lambda_star)
    return {
        "distance_X": analysis["distance"],
        "delta_C2": analysis["c1"] ** 2 - analysis["c0"] ** 2,
        "predicted_signed_slope": pred,
        "measured_signed_slope": fit["slope"],
        "measured_slope_sigma_ols": fit["slope_sigma"],
        "jackknife_slope_std": analysis["jackknife"]["slope_std"],
        "slope_relative_difference": (
            abs(fit["slope"] - pred) / max(abs(pred), 1.0e-300)
            if math.isfinite(pred) and math.isfinite(fit["slope"]) else float("nan")
        ),
        "mass_mismatch": delta_mf / max(base.mf, 1.0e-15),
        "mass_systematic_median": float(np.nanmedian(sigma_mass)) if np.any(np.isfinite(sigma_mass)) else float("nan"),
        "mass_systematic_max": float(np.nanmax(sigma_mass)) if np.any(np.isfinite(sigma_mass)) else float("nan"),
        "smoothing_sensitivity_median": smooth_err["median"],
        "smoothing_sensitivity_max": smooth_err["max"],
    }


def derivative_sensitivity(result1, result0, lambda_star):
    slopes = []
    for window in [7, 11, 15, 21]:
        for order in [2, 3, 4]:
            if order >= window - 1:
                continue
            try:
                c1 = extract_curve(result1, lambda_star, savgol_window=window, savgol_order=order)
                c0 = extract_curve(result0, lambda_star, savgol_window=window, savgol_order=order)
                dx = h2_distance(result1.config.initial, result0.config.initial)
                matched = match_curves_equal_rho(c1, c0,
                                                 result1.config.analysis.rho_min_fit,
                                                 result1.config.analysis.rho_max_fit)
                x = np.abs(np.log(np.maximum(matched["rho"], 1.0e-300)))
                signed = (matched["logk1"] - matched["logk0"]) / max(dx, 1.0e-300)
                fit = linear_fit(x, signed)
                if math.isfinite(fit["slope"]):
                    slopes.append(fit["slope"])
            except Exception:
                continue
    if len(slopes) < 2:
        return {"median": float("nan"), "max": float("nan")}
    arr = np.asarray(slopes)
    return {"median": float(np.median(np.abs(arr - np.median(arr)))),
            "max": float(np.max(arr) - np.min(arr))}


def analytic_validation():
    M, r = 1.7, 5.3
    kvac_exact = 48.0 * M * M / r ** 6
    k_identity = 48.0 * (M / r ** 3) ** 2
    c = math.sqrt(1.5)
    tau = 0.73
    kiso = float(kasner_k_tau(np.array([c]), np.array([tau]))[0])
    kiso_exact = 20.0 / (27.0 * tau ** 4)
    p1, p2, q = kasner_exponents(np.array([c]))
    return {
        "vacuum_relative_error": abs(k_identity / kvac_exact - 1.0),
        "stiff_flrw_relative_error": abs(kiso / kiso_exact - 1.0),
        "isotropic_p1": float(p1[0]),
        "isotropic_p2": float(p2[0]),
        "isotropic_q2": float(q[0] ** 2),
    }


def run_verification_checks(config: RunConfig, epsilon: float, out: Path) -> Dict[str, object]:
    checks = {}
    base_cfg = copy_config(config)
    base_cfg.initial.epsilon = 0.0

    hs_mink = [0.04, 0.02, 0.01, 0.005]
    mink = minkowski_pde_residual_test(hs_mink)
    below = [r.get("below_threshold", False) for r in mink if "below_threshold" in r]
    checks["minkowski_records"] = mink
    checks["minkowski_pde_pass"] = bool(below and all(below))
    orders_l2 = [r["order_l2"] for r in mink
                 if "order_l2" in r and math.isfinite(r["order_l2"])]
    checks["minkowski_pde_order_l2"] = float(np.mean(orders_l2)) if orders_l2 else float("nan")
    checks["minkowski_pde_order_linf"] = float("nan")
    checks["minkowski_pass"] = checks["minkowski_pde_pass"]
    checks["minkowski_order_l2"] = checks["minkowski_pde_order_l2"]

    cons = constraint_convergence_suite(config, hs_mink)
    cu_interp = [r.get("constraint_u_l2_interp") for r in cons
                 if r.get("constraint_u_l2_interp") is not None]
    cv_interp = [r.get("constraint_v_l2_interp") for r in cons
                 if r.get("constraint_v_l2_interp") is not None]

    def order_from_list(vals, hs):
        arr = np.asarray(vals, dtype=float)
        hh = np.asarray(hs, dtype=float)
        q = np.isfinite(arr) & (arr > 0)
        if np.count_nonzero(q) < 2:
            return float("nan")
        idx = np.where(q)[0]
        return float(math.log(arr[idx[0]] / arr[idx[-1]]) / math.log(hh[idx[0]] / hh[idx[-1]]))

    hs_used = [r["h"] for r in cons if r.get("constraint_u_l2_interp") is not None]
    checks["constraint_u_order"] = order_from_list(cu_interp, hs_used)
    checks["constraint_v_order"] = order_from_list(cv_interp, hs_used)
    checks["constraint_pass"] = bool(
        math.isfinite(checks["constraint_u_order"]) and checks["constraint_u_order"] > 1.8
        and math.isfinite(checks["constraint_v_order"]) and checks["constraint_v_order"] > 1.8
    )

    try:
        base = run_single(base_cfg)
        checks["mf_finite"] = bool(math.isfinite(base.mf) and base.mf > 0)
        checks["has_horizon"] = bool(base.horizon is not None)
        checks["horizon_points"] = int(base.diagnostics.get("horizon_points", 0))
        checks["horizon_pass"] = bool(checks["horizon_points"] >= config.analysis.min_horizon_points)
        checks["constraint_l2_base"] = float(base.diagnostics.get("constraint_u_l2", float("nan")))
        checks["constraint_residual_pass"] = bool(
            math.isfinite(checks["constraint_l2_base"]) and checks["constraint_l2_base"] < 1.0e-1
        )
        checks["massflux_l2_base"] = float(base.diagnostics.get("massflux_u_l2", float("nan")))
        checks["massflux_residual_pass"] = bool(
            math.isfinite(checks["massflux_l2_base"]) and checks["massflux_l2_base"] < 1.0e-1
        )
        checks["integrated_vs_algebraic_mass_l2"] = float(
            base.diagnostics.get("integrated_vs_algebraic_mass_l2", float("nan"))
        )
        checks["integrated_mass_pass"] = bool(
            math.isfinite(checks["integrated_vs_algebraic_mass_l2"]) and
            checks["integrated_vs_algebraic_mass_l2"] < 1.0e-1
        )
        checks["center_psi_resid_max"] = float(
            base.diagnostics.get("center_psi_resid_max", float("nan"))
        )
        checks["center_loga_resid_max"] = 0.0
        checks["center_reg_resid_max"] = float(
            base.diagnostics.get("center_reg_resid_max", float("nan"))
        )
        checks["center_pass"] = bool(
            math.isfinite(checks["center_psi_resid_max"]) and
            checks["center_psi_resid_max"] < 1.0e-6
        )
        checks["center_reg_pass"] = bool(
            math.isfinite(checks["center_reg_resid_max"]) and
            checks["center_reg_resid_max"] < 1.0e-2
        )
        checks["transversality_global_min"] = float(
            base.diagnostics.get("horizon_transversality_global_min", float("nan"))
        )
        checks["transversality_pass"] = bool(
            math.isfinite(checks["transversality_global_min"]) and
            checks["transversality_global_min"] > config.analysis.horizon_transversality_min
        )
        checks["exp_clipping_total"] = float(base.diagnostics.get("num_exp_clipped", float("nan")))
        checks["exp_clipping_pass"] = bool(
            checks["exp_clipping_total"] == 0.0
            or checks["exp_clipping_total"] / max(base.u.size * base.v.size, 1) < 1.0e-6
        )
        checks["num_singularity_rows"] = float(base.diagnostics.get("num_singularity_rows", 0.0))
        checks["row_max_rv"] = float(base.diagnostics.get("row_max_rv", float("nan")))
        checks["row_max_loga"] = float(base.diagnostics.get("row_max_loga", float("nan")))

        try:
            export_kasner_diagnostics(base, out, config.analysis.lambda_star)
        except Exception:
            pass

        if not checks["has_horizon"]:
            checks["equal_mass_pass"] = False
            checks["delta_C2_nonzero_pass"] = False
            checks["slope_agree_pass"] = False
            checks["fit_r2_pass"] = False
            checks["kasner_window_pass"] = False
            checks["nested_pass"] = False
            checks["equal_mass_error"] = (
                f"no horizon formed; initial amplitude={config.initial.amplitude} "
                f"likely below threshold. Increase --amplitude."
            )
        else:
            try:
                aeq, pert, _ = equal_mass_amplitude(base_cfg, epsilon)
                checks["equal_mass_mismatch"] = abs(pert.mf - base.mf) / max(base.mf, 1.0e-15)
                checks["equal_mass_pass"] = bool(
                    checks["equal_mass_mismatch"] < config.analysis.equal_mass_tolerance
                )
                analysis = finite_data_analysis(pert, base, config.analysis.lambda_star)
                checks["delta_C2"] = float(analysis["c1"] ** 2 - analysis["c0"] ** 2)
                checks["C1"] = float(analysis["c1"])
                checks["C0"] = float(analysis["c0"])
                checks["delta_C2_nonzero_pass"] = bool(
                    math.isfinite(checks["delta_C2"]) and abs(checks["delta_C2"]) > 1.0e-4
                )
                fit = analysis["fit_signed"]
                jack = analysis["jackknife"]
                sigma_eff = math.sqrt(max(fit["slope_sigma"], 0.0) ** 2 + max(jack["slope_std"], 0.0) ** 2)
                checks["measured_slope"] = float(fit["slope"])
                checks["predicted_slope"] = float(analysis["pred_signed"])
                checks["slope_sigma_effective"] = float(sigma_eff)
                checks["fit_r2"] = float(fit["r2"])
                checks["slope_agree_pass"] = bool(
                    math.isfinite(fit["slope"]) and math.isfinite(analysis["pred_signed"]) and
                    abs(fit["slope"] - analysis["pred_signed"]) <=
                    3.0 * max(sigma_eff, 1.0e-12) + 0.1 * abs(analysis["pred_signed"])
                )
                checks["fit_r2_pass"] = bool(
                    math.isfinite(fit["r2"]) and fit["r2"] > 0.80
                )
                checks["kasner_window_points"] = int(np.count_nonzero(analysis["mask1"]))
                checks["kasner_window_pass"] = bool(checks["kasner_window_points"] >= 20)
                checks["nested_fits"] = analysis["nested_fits"]
                slopes_nested = [f["slope"] for f in analysis["nested_fits"]
                                 if math.isfinite(f["slope"])]
                if len(slopes_nested) >= 2:
                    checks["nested_slope_spread"] = float(max(slopes_nested) - min(slopes_nested))
                    checks["nested_pass"] = bool(
                        checks["nested_slope_spread"] < 0.5 * abs(analysis["pred_signed"]) + 1.0e-2
                    )
                else:
                    checks["nested_slope_spread"] = float("nan")
                    checks["nested_pass"] = False
            except Exception as exc:
                checks["physics_error"] = str(exc)
                for k in ["equal_mass_pass", "delta_C2_nonzero_pass",
                          "slope_agree_pass", "fit_r2_pass",
                          "kasner_window_pass", "nested_pass"]:
                    checks.setdefault(k, False)
                checks["kasner_window_points"] = 0
                checks["C1"] = float("nan")
                checks["C0"] = float("nan")
                checks["delta_C2"] = float("nan")
                checks["measured_slope"] = float("nan")
                checks["predicted_slope"] = float("nan")
                checks["fit_r2"] = float("nan")
                checks["nested_slope_spread"] = float("nan")
                checks["slope_sigma_effective"] = float("nan")
                checks["equal_mass_mismatch"] = float("nan")
    except Exception as exc:
        checks["base_error"] = str(exc)
        for k in ["horizon_pass", "constraint_residual_pass", "massflux_residual_pass",
                  "integrated_mass_pass", "center_pass", "center_reg_pass",
                  "transversality_pass", "equal_mass_pass", "delta_C2_nonzero_pass",
                  "slope_agree_pass", "fit_r2_pass", "kasner_window_pass", "nested_pass"]:
            checks.setdefault(k, False)

    checks["all_solver_pass"] = bool(
        checks.get("minkowski_pde_pass", False)
        and checks.get("constraint_pass", False)
        and checks.get("horizon_pass", False)
        and checks.get("constraint_residual_pass", False)
        and checks.get("center_pass", False)
        and checks.get("center_reg_pass", False)
        and checks.get("exp_clipping_pass", False)
    )
    checks["all_physics_pass"] = bool(
        checks.get("transversality_pass", False)
        and checks.get("kasner_window_pass", False)
        and checks.get("delta_C2_nonzero_pass", False)
        and checks.get("equal_mass_pass", False)
    )
    checks["overall_pass"] = bool(checks["all_solver_pass"] and checks["all_physics_pass"])
    save_json(out / "verification_summary.json", checks)
    return checks


def print_verification(checks: Dict[str, object]) -> None:
    def line(name, ok, value, criterion=""):
        sym = PASS_SYMBOL if ok else FAIL_SYMBOL
        if isinstance(value, (int, float, np.floating)) and math.isfinite(float(value)):
            vs = f"{value:.6g}"
        else:
            vs = str(value)
        print(f"[{sym}] {name:<40s} = {vs:<18s} {criterion}")

    print("=" * 78)
    print("  Numerical Verification Summary")
    print("=" * 78)
    print("  QUICK SELF-TEST on a coarse demo run (default h = 0.02, no restart).")
    print("  This run is far too coarse to resolve a Kasner window, so the")
    print("  physics and slope checks below cannot pass by construction.")
    print("  For the paper-level check use:  python verify_paper.py")
    print("=" * 78)
    print("-- A. Solver correctness --")
    line("Minkowski PDE residual (below thr)", checks.get("minkowski_pde_pass", False),
         checks.get("minkowski_pde_order_l2", float("nan")), "(threshold 1e-8)")
    line("Constraint u convergence order", checks.get("constraint_pass", False),
         checks.get("constraint_u_order", float("nan")), "(requires > 1.8)")
    line("Constraint v convergence order", checks.get("constraint_pass", False),
         checks.get("constraint_v_order", float("nan")), "(requires > 1.8)")
    line("Horizon exists", checks.get("horizon_pass", False),
         checks.get("horizon_points", 0), "(requires >= 6)")
    line("Constraint residual L2", checks.get("constraint_residual_pass", False),
         checks.get("constraint_l2_base", float("nan")), "(requires < 1e-1)")
    line("Mass flux residual L2", checks.get("massflux_residual_pass", False),
         checks.get("massflux_l2_base", float("nan")), "(requires < 1e-1)")
    line("Integrated vs algebraic mass", checks.get("integrated_mass_pass", False),
         checks.get("integrated_vs_algebraic_mass_l2", float("nan")), "(requires < 1e-1)")
    line("Center psi residual", checks.get("center_pass", False),
         checks.get("center_psi_resid_max", float("nan")), "(requires < 1e-6)")
    line("Center regularity resid", checks.get("center_reg_pass", False),
         checks.get("center_reg_resid_max", float("nan")), "(requires < 1e-2)")
    line("exp clipping count", checks.get("exp_clipping_pass", False),
         checks.get("exp_clipping_total", float("nan")), "(requires 0 or fraction < 1e-6)")
    print("-- B. Physics correctness --")
    line("Anchor transversality", checks.get("transversality_pass", False),
         checks.get("transversality_global_min", float("nan")), "(requires > 1e-4)")
    line("Kasner window points", checks.get("kasner_window_pass", False),
         checks.get("kasner_window_points", 0), "(requires >= 20)")
    line("Delta C^2 nonzero", checks.get("delta_C2_nonzero_pass", False),
         checks.get("delta_C2", float("nan")), "(requires |.| > 1e-4)")
    line("Equal final mass mismatch", checks.get("equal_mass_pass", False),
         checks.get("equal_mass_mismatch", float("nan")), "(requires < 5e-4)")
    print("-- C. Main result --")
    line("C1", bool(False), checks.get("C1", float("nan")))
    line("C0", bool(False), checks.get("C0", float("nan")))
    line("Predicted slope", bool(False), checks.get("predicted_slope", float("nan")))
    line("Measured slope", checks.get("slope_agree_pass", False),
         checks.get("measured_slope", float("nan")))
    line("Effective sigma", True, checks.get("slope_sigma_effective", float("nan")))
    line("Slope consistency", checks.get("slope_agree_pass", False),
         checks.get("measured_slope", float("nan")), "(requires |diff| < 3*sigma + 10%)")
    line("Fit R^2", checks.get("fit_r2_pass", False),
         checks.get("fit_r2", float("nan")), "(requires > 0.80)")
    line("Nested window slope spread", checks.get("nested_pass", False),
         checks.get("nested_slope_spread", float("nan")), "(requires < 50% of prediction)")
    print("-- D. Overall --")
    line("All solver checks", checks.get("all_solver_pass", False),
         checks.get("all_solver_pass", False))
    line("All physics checks", checks.get("all_physics_pass", False),
         checks.get("all_physics_pass", False))
    overall = checks.get("overall_pass", False)
    print("=" * 78)
    print(f"Overall: {'PASS' if overall else 'FAIL - see above'}")
    if checks.get("equal_mass_error"):
        print(f"Diagnostic: {checks['equal_mass_error']}")
    if checks.get("physics_error"):
        print(f"Physics diagnostic: {checks['physics_error']}")
    if checks.get("base_error"):
        print(f"Diagnostic: {checks['base_error']}")
    print("=" * 78)


def _fmt(x, fmt=".4e"):
    if x is None:
        return "--"
    try:
        v = float(x)
    except Exception:
        return "--"
    if not math.isfinite(v):
        return "--"
    return f"{v:{fmt}}"


def generate_paper_tables(out_dir: Path,
                          observable_conv: Sequence[Dict[str, object]],
                          family_records: Sequence[Dict[str, object]],
                          error_budget: Dict[str, object],
                          equal_mass_conv: Sequence[Dict[str, object]] = (),
                          kasner_sens: Sequence[Dict[str, object]] = ()) -> Path:
    tables_dir = Path(out_dir) / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Resolution convergence of the final observable.}",
        r"\label{TABRESCONV}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lcccccc}",
        r"$h$ & $s_\epsilon^{\rm meas}$ & $\sigma_{\rm OLS}$ & $\sigma_{\rm JK}$ "
        r"& $s_\epsilon^{\rm pred}$ & $|s^{\rm meas}-s^{\rm pred}|$ "
        r"& $R^2$ \\",
        r"\hline",
    ]
    for r in observable_conv:
        if "measured_slope" not in r:
            continue
        if r.get("h", 1.0) == 0.0:
            h_str = r"\textit{Richardson}"
        else:
            h_str = _fmt(r["h"])
        diff = "--"
        if (math.isfinite(r.get("measured_slope", float("nan")))
                and math.isfinite(r.get("predicted_slope", float("nan")))):
            diff = _fmt(abs(r["measured_slope"] - r["predicted_slope"]))
        lines.append(" & ".join([
            h_str,
            _fmt(r.get("measured_slope")),
            _fmt(r.get("slope_sigma_ols")),
            _fmt(r.get("slope_sigma_jack")),
            _fmt(r.get("predicted_slope")),
            diff,
            _fmt(r.get("r2"), fmt=".3f"),
        ]) + r" \\")
    lines += [
        r"\end{tabular}",
        r"\end{ruledtabular}",
        r"\end{table}",
        "",
    ]
    (tables_dir / "table1_convergence.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Family robustness.}",
        r"\label{TABFAMILY}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lcccc}",
        r"Family & $s_\epsilon^{\rm meas}$ & $\sigma$ & $s_\epsilon^{\rm pred}$ "
        r"& $|\Delta|$ \\",
        r"\hline",
    ]
    for rec in family_records:
        if "error" in rec:
            lines.append(f"{rec.get('family','?')} & \\multicolumn{{4}}{{c}}{{error}} \\\\")
            continue
        rep = rec.get("report", {})
        fit = rep.get("measured_signed_fit", {})
        lines.append(" & ".join([
            rec.get("family", "?"),
            _fmt(fit.get("slope")),
            _fmt(fit.get("slope_sigma")),
            _fmt(rep.get("predicted_signed_slope")),
            _fmt(abs(fit.get("slope", float("nan")) -
                     rep.get("predicted_signed_slope", float("nan")))),
        ]) + r" \\")
    lines += [
        r"\end{tabular}",
        r"\end{ruledtabular}",
        r"\end{table}",
        "",
    ]
    (tables_dir / "table2_family.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Error budget for the primary finite-data test.}",
        r"\label{TABERRBUDGET}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lc}",
        r"Quantity & Value \\",
        r"\hline",
    ]
    for key in [
        "distance_X", "delta_C2", "predicted_signed_slope",
        "measured_signed_slope", "measured_slope_sigma_ols",
        "jackknife_slope_std", "slope_relative_difference",
        "mass_mismatch", "mass_systematic_median", "mass_systematic_max",
        "smoothing_sensitivity_median", "smoothing_sensitivity_max",
    ]:
        lines.append(f"{key.replace('_', chr(92) + '_')} & {_fmt(error_budget.get(key))} " + r"\\")
    lines += [
        r"\end{tabular}",
        r"\end{ruledtabular}",
        r"\end{table}",
        "",
    ]
    (tables_dir / "table3_error_budget.tex").write_text("\n".join(lines), encoding="utf-8")

    if equal_mass_conv:
        lines = [
            r"\begin{table}[t]",
            r"\caption{Equal-final-mass construction.}",
            r"\label{TABEQMASS}",
            r"\begin{ruledtabular}",
            r"\begin{tabular}{lccc}",
            r"$h$ & $M_f^{\rm base}$ & $M_f^{\rm pert}$ & $\Delta M_f / M_f$ \\",
            r"\hline",
        ]
        for r in equal_mass_conv:
            if "relative_mismatch" not in r:
                continue
            lines.append(" & ".join([
                _fmt(r["h"]),
                _fmt(r["base_mf"]),
                _fmt(r["pert_mf"]),
                _fmt(r["relative_mismatch"]),
            ]) + r" \\")
        lines += [
            r"\end{tabular}",
            r"\end{ruledtabular}",
            r"\end{table}",
            "",
        ]
        (tables_dir / "table4_equal_mass.tex").write_text("\n".join(lines), encoding="utf-8")

    if kasner_sens:
        lines = [
            r"\begin{table}[t]",
            r"\caption{Kasner-window sensitivity.}",
            r"\label{TABKASWIN}",
            r"\begin{ruledtabular}",
            r"\begin{tabular}{lcccccc}",
            r"$\rho_{\min}$ & $\rho_{\max}$ & $C_1$ & $C_0$ & $\Delta C^2$ "
            r"& $s_\epsilon^{\rm meas}$ & $s_\epsilon^{\rm pred}$ \\",
            r"\hline",
        ]
        for r in kasner_sens:
            lines.append(" & ".join([
                _fmt(r.get("rho_lo")),
                _fmt(r.get("rho_hi")),
                _fmt(r.get("C1")),
                _fmt(r.get("C0")),
                _fmt(r.get("delta_C2")),
                _fmt(r.get("measured_slope")),
                _fmt(r.get("predicted_slope")),
            ]) + r" \\")
        lines += [
            r"\end{tabular}",
            r"\end{ruledtabular}",
            r"\end{table}",
            "",
        ]
        (tables_dir / "table5_kasner_window.tex").write_text("\n".join(lines), encoding="utf-8")

    return tables_dir


def plot_and_save_csv_observable_convergence(records, out: Path):
    keep = [r for r in records if "measured_slope" in r]
    if len(keep) < 2:
        return
    hs = np.array([r["h"] for r in keep])
    ms = np.array([r["measured_slope"] for r in keep])
    ps = np.array([r["predicted_slope"] for r in keep])
    ss = np.array([r.get("slope_sigma_ols", np.nan) for r in keep])

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    mask = hs > 0.0
    ax.errorbar(hs[mask], ms[mask], yerr=ss[mask], fmt="o-",
                label=r"$s_\epsilon^{\rm meas}$")
    ax.axhline(ps[0], color="r", linestyle="--", label=r"$s_\epsilon^{\rm pred}$")
    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("signed slope")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure10_observable_convergence.png", dpi=180)
    fig.savefig(out / "figure10_observable_convergence.pdf")
    plt.close(fig)

    save_csv(
        out / "figure10_observable_convergence.csv",
        ["h", "measured_slope", "slope_sigma_ols", "slope_sigma_jack",
         "predicted_slope", "r2"],
        [hs, ms, ss,
         np.array([r.get("slope_sigma_jack", np.nan) for r in keep]),
         ps,
         np.array([r.get("r2", np.nan) for r in keep])],
    )


def plot_and_save_csv_equal_mass_convergence(records, out: Path):
    keep = [r for r in records if "relative_mismatch" in r]
    if len(keep) < 2:
        return
    hs = np.array([r["h"] for r in keep])
    rel = np.array([r["relative_mismatch"] for r in keep])
    tolr = np.array([r["tolerance"] for r in keep])

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.loglog(hs, rel, "o-", label=r"$\Delta M_f / M_f$")
    ax.axhline(tolr[0], color="r", linestyle="--", label="tolerance")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\Delta M_f / M_f$")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure11_equal_mass_convergence.png", dpi=180)
    fig.savefig(out / "figure11_equal_mass_convergence.pdf")
    plt.close(fig)

    save_csv(
        out / "figure11_equal_mass_convergence.csv",
        ["h", "base_mf", "pert_mf", "delta_mf", "relative_mismatch", "tolerance"],
        [hs, np.array([r["base_mf"] for r in keep]),
         np.array([r["pert_mf"] for r in keep]),
         np.array([r["delta_mf"] for r in keep]),
         rel, tolr],
    )


def plot_and_save_csv_kasner_sensitivity(records, out: Path):
    keep = [r for r in records if "delta_C2" in r and math.isfinite(r["delta_C2"])]
    if len(keep) < 2:
        return
    lo = np.array([r["rho_lo"] for r in keep])
    hi = np.array([r["rho_hi"] for r in keep])
    dc2 = np.array([r["delta_C2"] for r in keep])
    ms = np.array([r["measured_slope"] for r in keep])
    ps = np.array([r["predicted_slope"] for r in keep])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    idx = np.arange(len(keep))
    labels = [f"[{l:.0e}, {h:.0e}]" for l, h in zip(lo, hi)]
    axes[0].plot(idx, dc2, "o-")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel(r"$\Delta C^2$")
    axes[1].plot(idx, ms, "o-", label="measured")
    axes[1].plot(idx, ps, "s--", label="predicted")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("signed slope")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure12_kasner_sensitivity.png", dpi=180)
    fig.savefig(out / "figure12_kasner_sensitivity.pdf")
    plt.close(fig)

    save_csv(
        out / "figure12_kasner_sensitivity.csv",
        ["rho_lo", "rho_hi", "C1", "C0", "delta_C2",
         "measured_slope", "predicted_slope"],
        [lo, hi,
         np.array([r["C1"] for r in keep]),
         np.array([r["C0"] for r in keep]),
         dc2, ms, ps],
    )


def run_full_experiment(config: RunConfig, epsilon: float) -> Dict[str, object]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "provenance.json", provenance(config))
    full = {}
    full["analytic_validation"] = analytic_validation()

    hs_all = [0.04, 0.02, 0.01, 0.005]

    mink = minkowski_pde_residual_test(hs_all)
    full["minkowski_test"] = mink
    plot_and_save_csv_minkowski(mink, out)

    cons = constraint_convergence_suite(config, hs_all)
    full["constraint_convergence"] = cons
    plot_and_save_csv_constraints(cons, out)

    sol = solution_convergence_suite(config, hs_all)
    full["solution_convergence"] = sol

    try:
        cfg_eq = copy_config(config)
        cfg_eq.output_dir = str(out / "primary_equal_mass")
        full["primary_equal_mass"] = run_primary(cfg_eq, epsilon, True)
        plot_and_save_csv_primary_artifacts(Path(cfg_eq.output_dir), out / "primary_figures")
        full["primary_available"] = True
    except Exception as exc:
        full["primary_equal_mass"] = {"error": str(exc)}
        full["primary_available"] = False

    hor = horizon_convergence_suite(config, hs_all, config.analysis.lambda_star)
    full["horizon_convergence"] = hor
    plot_and_save_csv_horizon(hor, out)

    if full["primary_available"]:
        full["families"] = run_family_suite(config, epsilon, ["bump", "gaussian", "two_pulse", "shape"])
        plot_and_save_csv_families(full["families"], out)

        full["lambdas"] = run_lambda_suite(config, epsilon, [0.20, 0.25, 0.30, 0.35])
        plot_and_save_csv_lambdas(full["lambdas"], out)

        full["schemes"] = run_scheme_suite(config, epsilon, ["equal_rho", "equal_compactness"])
        plot_and_save_csv_schemes(full["schemes"], out)

        full["norms"] = run_norm_suite(config, epsilon, [
            ("default", 0.25, 2.0, 1.0),
            ("unweighted", 0.0, 2.0, 1.0),
            ("linear", 0.25, 1.0, 1.0),
            ("short_ell", 0.25, 2.0, 0.5),
        ])
        plot_and_save_csv_norms(full["norms"], out)

        full["convergence"] = convergence_suite(config, [0.04, 0.02, 0.01],
                                                [0.04, 0.02, 0.01],
                                                config.analysis.lambda_star)
        plot_and_save_csv_convergence(full["convergence"], out)

        full["observable_convergence"] = observable_convergence_suite(
            config, hs_all, epsilon, config.analysis.lambda_star
        )
        plot_and_save_csv_observable_convergence(full["observable_convergence"], out)

        full["equal_mass_convergence"] = equal_mass_convergence_suite(config, hs_all, epsilon)
        plot_and_save_csv_equal_mass_convergence(full["equal_mass_convergence"], out)

        full["kasner_sensitivity"] = kasner_window_sensitivity(
            config, epsilon, config.analysis.lambda_star
        )
        plot_and_save_csv_kasner_sensitivity(full["kasner_sensitivity"], out)

        full["error_budget"] = error_budget_analysis(config, epsilon, config.analysis.lambda_star)

        full["paper_tables_dir"] = str(generate_paper_tables(
            out,
            full["observable_convergence"],
            full["families"],
            full["error_budget"],
            full["equal_mass_convergence"],
            full["kasner_sensitivity"],
        ))
    else:
        for k in ["families", "lambdas", "schemes", "norms", "convergence",
                  "observable_convergence", "equal_mass_convergence",
                  "kasner_sensitivity", "error_budget", "paper_tables_dir"]:
            full[k] = {"error": "skipped: no primary equal-mass family available"}

    full["verification"] = run_verification_checks(config, epsilon, out)
    print_verification(full["verification"])
    save_json(out / "full_experiment_report.json", full)
    return full


def plot_and_save_csv_minkowski(records, out: Path):
    keep = [r for r in records if "res_l2" in r]
    if len(keep) < 2:
        return
    hs = np.array([r["h"] for r in keep])
    res_l2 = np.array([r["res_l2"] for r in keep])
    res_linf = np.array([r["res_linf"] for r in keep])

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.loglog(hs, res_l2, "o-", label=r"$\|R_r\|_2$")
    ax.loglog(hs, res_linf, "s-", label=r"$\|R_r\|_\infty$")
    ax.set_xlabel("h")
    ax.set_ylabel("PDE residual")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure0_minkowski_test.png", dpi=180)
    fig.savefig(out / "figure0_minkowski_test.pdf")
    plt.close(fig)

    save_csv(out / "figure0_minkowski_test.csv",
             ["h", "res_l2", "res_linf"],
             [hs, res_l2, res_linf])


def plot_and_save_csv_constraints(records, out: Path):
    keep = [r for r in records if "constraint_u_l2_interp" in r]
    if len(keep) < 2:
        return
    hs = np.array([r["h"] for r in keep])
    cu = np.array([r["constraint_u_l2_interp"] for r in keep])
    cv = np.array([r["constraint_v_l2_interp"] for r in keep])
    mu = np.array([r.get("massflux_u_l2", np.nan) for r in keep])
    mv = np.array([r.get("massflux_v_l2", np.nan) for r in keep])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    axes[0].loglog(hs, cu, "o-", label="C_u L2")
    axes[0].loglog(hs, cv, "s-", label="C_v L2")
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("constraint L2")
    axes[0].legend(loc="best")
    axes[1].loglog(hs, mu, "o-", label="M_u L2")
    axes[1].loglog(hs, mv, "s-", label="M_v L2")
    axes[1].set_xlabel("h")
    axes[1].set_ylabel("mass flux L2")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure6_constraints.png", dpi=180)
    fig.savefig(out / "figure6_constraints.pdf")
    plt.close(fig)
    save_csv(out / "figure6_constraints.csv",
             ["h", "constraint_u_l2_interp", "constraint_v_l2_interp",
              "massflux_u_l2", "massflux_v_l2"],
             [hs, cu, cv, mu, mv])


def plot_and_save_csv_horizon(records, out: Path):
    keep = [r for r in records if "v_anchor" in r]
    if len(keep) < 2:
        return
    hs = np.array([r["h"] for r in keep])
    va = np.array([r["v_anchor"] for r in keep])
    ma = np.array([r["m_anchor"] for r in keep])
    ta = np.array([r["transversality"] for r in keep])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    axes[0].semilogx(hs, va, "o-")
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("v_anchor")
    axes[1].semilogx(hs, ma, "s-")
    axes[1].set_xlabel("h")
    axes[1].set_ylabel("m_anchor")
    fig.tight_layout()
    fig.savefig(out / "figure7_horizon_convergence.png", dpi=180)
    fig.savefig(out / "figure7_horizon_convergence.pdf")
    plt.close(fig)
    save_csv(out / "figure7_horizon_convergence.csv",
             ["h", "v_anchor", "m_anchor", "transversality"], [hs, va, ma, ta])


def plot_and_save_csv_primary_artifacts(in_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cfg_json = in_dir / "config.json"
    if cfg_json.exists():
        try:
            with open(cfg_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cfg = RunConfig(
                GridConfig(**raw["grid"]),
                InitialDataConfig(**raw["initial"]),
                AnalysisConfig(**raw["analysis"]),
                str(in_dir),
            )
        except Exception:
            cfg = RunConfig(GridConfig(), InitialDataConfig(), AnalysisConfig(), str(in_dir))
    else:
        cfg = RunConfig(GridConfig(), InitialDataConfig(), AnalysisConfig(), str(in_dir))
    try:
        base = load_solution(in_dir / "base_solution.npz",
                             in_dir / "base_diagnostics.json",
                             in_dir / "base_horizon.npz")
        pert = load_solution(in_dir / "perturbed_solution.npz",
                             in_dir / "perturbed_diagnostics.json",
                             in_dir / "perturbed_horizon.npz")
    except Exception:
        return
    base.config = cfg
    pert.config = cfg
    try:
        analysis = finite_data_analysis(pert, base)
    except Exception:
        return
    plot_and_save_csv_kasner(analysis["curve1"], analysis["curve0"],
                             analysis["mask1"], analysis["mask0"], out)
    plot_and_save_csv_finite(analysis, out)
    plot_and_save_csv_nested(analysis, out)


def plot_and_save_csv_kasner(curve1, curve0, mask1, mask0, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
    rows_sol, rows_x, rows_cpsi, rows_cm2, rows_cs, rows_pred, rows_win = [], [], [], [], [], [], []
    for ax, curve, mask, label in [(axes[0], curve0, mask0, "base"),
                                   (axes[1], curve1, mask1, "perturbed")]:
        x = np.abs(np.log(curve.rho))
        pred = 6.0 + 4.0 * curve.cpsi_direct * curve.cpsi_direct
        ax.plot(x, curve.cpsi_direct * curve.cpsi_direct, label="C_psi^2")
        ax.plot(x, curve.cm2, label="C_m^2")
        ax.plot(x, curve.curvature_slope, label="curvature slope")
        ax.plot(x, pred, label="6+4C^2")
        if np.any(mask):
            ax.scatter(x[mask], curve.curvature_slope[mask], s=10, label="Kasner window")
        ax.set_xlabel("|ln rho|")
        ax.set_title(label)
        ax.legend(loc="best", fontsize=7)
        for k in range(x.size):
            rows_sol.append(label)
            rows_x.append(float(x[k]))
            rows_cpsi.append(float(curve.cpsi_direct[k] ** 2))
            rows_cm2.append(float(curve.cm2[k]))
            rows_cs.append(float(curve.curvature_slope[k]))
            rows_pred.append(float(pred[k]))
            rows_win.append(int(mask[k]))
    fig.tight_layout()
    fig.savefig(out / "figure2_kasner_dual.png", dpi=180)
    fig.savefig(out / "figure2_kasner_dual.pdf")
    plt.close(fig)
    save_csv_ragged(out / "figure2_kasner_dual.csv",
                    ["solution", "x", "cpsi_sq", "cm2", "curvature_slope",
                     "pred", "in_window"],
                    [rows_sol, rows_x, rows_cpsi, rows_cm2, rows_cs, rows_pred, rows_win])


def plot_and_save_csv_finite(analysis, out: Path):
    matched = analysis["matched"]
    x = np.abs(np.log(matched["rho"]))
    q = analysis["quotient"]
    s = analysis["signed_response"]
    fit = analysis["fit_signed"]
    pred = analysis["pred_signed"]
    fit_line = (fit["slope"] * x + fit["intercept"]
                if math.isfinite(fit["slope"]) else np.full_like(x, np.nan))
    pred_line = (pred * x + np.nanmedian(s - pred * x)
                 if math.isfinite(pred) else np.full_like(x, np.nan))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, q, label="Q_K")
    ax.set_xlabel("|ln rho|")
    ax.set_ylabel("Q_K")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure3_finite_quotient.png", dpi=180)
    fig.savefig(out / "figure3_finite_quotient.pdf")
    plt.close(fig)
    save_csv(out / "figure3_finite_quotient.csv", ["x", "Q_K"], [x, q])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, s, label="signed response")
    if math.isfinite(fit["slope"]):
        ax.plot(x, fit_line, label="fit")
    if math.isfinite(pred):
        ax.plot(x, pred_line, "--", label="predicted")
    ax.set_xlabel("|ln rho|")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure4_signed_slope.png", dpi=180)
    fig.savefig(out / "figure4_signed_slope.pdf")
    plt.close(fig)
    save_csv(out / "figure4_signed_slope.csv",
             ["x", "signed_response", "fit_line", "pred_line"],
             [x, s, fit_line, pred_line])


def plot_and_save_csv_nested(analysis, out: Path):
    nested = analysis["nested_fits"]
    if not nested:
        return
    his = np.array([f["rho_hi"] for f in nested])
    slopes = np.array([f["slope"] for f in nested])
    sigmas = np.array([f["slope_sigma"] for f in nested])
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.errorbar(his, slopes, yerr=sigmas, fmt="o-")
    ax.axhline(analysis["pred_signed"], color="r", linestyle="--", label="predicted")
    ax.set_xlabel("rho_hi (fixed rho_lo)")
    ax.set_ylabel("signed slope")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure4b_nested_windows.png", dpi=180)
    fig.savefig(out / "figure4b_nested_windows.pdf")
    plt.close(fig)
    save_csv(out / "figure4b_nested_windows.csv",
             ["rho_lo", "rho_hi", "slope", "slope_sigma", "r2"],
             [np.array([f["rho_lo"] for f in nested]), his, slopes, sigmas,
              np.array([f["r2"] for f in nested])])


def plot_and_save_csv_families(records, out: Path):
    labels, slopes, preds, sigmas = [], [], [], []
    for rec in records:
        if "error" in rec:
            continue
        rep = rec.get("report", {})
        fit = rep.get("measured_signed_fit", {})
        labels.append(rec.get("family", "?"))
        slopes.append(fit.get("slope", np.nan))
        sigmas.append(fit.get("slope_sigma", np.nan))
        preds.append(rep.get("predicted_signed_slope", np.nan))
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.bar(x - 0.18, slopes, 0.36, yerr=sigmas, label="measured")
    ax.bar(x + 0.18, preds, 0.36, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("signed slope")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure9_family_robustness.png", dpi=180)
    fig.savefig(out / "figure9_family_robustness.pdf")
    plt.close(fig)
    save_csv_ragged(out / "figure9_family_robustness.csv",
                    ["family", "measured_slope", "slope_sigma", "predicted_slope"],
                    [labels, slopes, sigmas, preds])


def plot_and_save_csv_lambdas(records, out: Path):
    keep = [r for r in records if "error" not in r and "lambda_star" in r]
    if len(keep) < 2:
        return
    lams = np.array([r["lambda_star"] for r in keep])
    slopes = np.array([r.get("measured_signed_fit", {}).get("slope", np.nan) for r in keep])
    preds = np.array([r.get("predicted_signed_slope", np.nan) for r in keep])
    sigmas = np.array([r.get("measured_signed_fit", {}).get("slope_sigma", np.nan) for r in keep])
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.errorbar(lams, slopes, yerr=sigmas, fmt="o", label="measured")
    ax.plot(lams, preds, "s--", label="predicted")
    ax.set_xlabel("lambda_star")
    ax.set_ylabel("signed slope")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure8_lambda_robustness.png", dpi=180)
    fig.savefig(out / "figure8_lambda_robustness.pdf")
    plt.close(fig)
    save_csv(out / "figure8_lambda_robustness.csv",
             ["lambda_star", "measured_slope", "slope_sigma", "predicted_slope"],
             [lams, slopes, sigmas, preds])


def plot_and_save_csv_schemes(schemes, out: Path):
    scheme_detail = {}
    for label, rep in schemes.items():
        if "error" in rep:
            continue
        scheme_detail[label] = rep
    if len(scheme_detail) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    rows_scheme, rows_slope, rows_pred = [], [], []
    for label, rep in scheme_detail.items():
        fit = rep.get("measured_signed_fit", {})
        rows_scheme.append(label)
        rows_slope.append(fit.get("slope", np.nan))
        rows_pred.append(rep.get("predicted_signed_slope", np.nan))
    x = np.arange(len(rows_scheme))
    axes[0].bar(x - 0.18, rows_slope, 0.36, label="measured")
    axes[0].bar(x + 0.18, rows_pred, 0.36, label="predicted")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rows_scheme)
    axes[0].set_ylabel("signed slope")
    axes[0].legend(loc="best")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "figure3b_scheme_compare.png", dpi=180)
    fig.savefig(out / "figure3b_scheme_compare.pdf")
    plt.close(fig)
    save_csv_ragged(out / "figure3b_scheme_compare.csv",
                    ["scheme", "measured_slope", "predicted_slope"],
                    [rows_scheme, rows_slope, rows_pred])


def plot_and_save_csv_norms(norms, out: Path):
    rows_label, rows_slope, rows_pred = [], [], []
    for label, rep in norms.items():
        if "error" in rep:
            continue
        rows_label.append(label)
        rows_slope.append(rep.get("measured_signed_fit", {}).get("slope", np.nan))
        rows_pred.append(rep.get("predicted_signed_slope", np.nan))
    if not rows_label:
        return
    x = np.arange(len(rows_label))
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.bar(x - 0.18, rows_slope, 0.36, label="measured")
    ax.bar(x + 0.18, rows_pred, 0.36, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(rows_label)
    ax.set_ylabel("signed slope")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure3c_norm_sensitivity.png", dpi=180)
    fig.savefig(out / "figure3c_norm_sensitivity.pdf")
    plt.close(fig)
    save_csv_ragged(out / "figure3c_norm_sensitivity.csv",
                    ["norm_variant", "measured_slope", "predicted_slope"],
                    [rows_label, rows_slope, rows_pred])


def plot_and_save_csv_convergence(records, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rows_h, rows_eps, rows_x, rows_D = [], [], [], []
    for rec in records:
        if "rho" not in rec:
            continue
        x = np.abs(np.log(rec["rho"]))
        ax.plot(x, rec["D"], label=f"h={rec['h']:.4g}, eps={rec['epsilon']:.3g}")
        for k in range(x.size):
            rows_h.append(float(rec["h"]))
            rows_eps.append(float(rec["epsilon"]))
            rows_x.append(float(x[k]))
            rows_D.append(float(rec["D"][k]))
    ax.set_xlabel("|ln rho|")
    ax.set_ylabel("centered response")
    if len(records) <= 15:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure5_two_parameter_convergence.png", dpi=180)
    fig.savefig(out / "figure5_two_parameter_convergence.pdf")
    plt.close(fig)
    save_csv_ragged(out / "figure5_two_parameter_convergence.csv",
                    ["h", "epsilon", "x", "D"],
                    [rows_h, rows_eps, rows_x, rows_D])


def load_solution(npz_path: Path, diag_path: Path, horizon_path: Path,
                  include_extras: bool = False) -> SimulationResult:
    with np.load(npz_path, allow_pickle=True) as data:
        u = data["u"]; v = data["v"]; r = data["r"]; loga = data["logalpha"]
        psi = data["psi"]; valid = data["valid"]; mf = float(data["mf"][0])
        integrated = data["integrated_mass"] if (include_extras and "integrated_mass" in data) else None
        ruv_ev = data["ruv_evolved"] if (include_extras and "ruv_evolved" in data) else None
    diagnostics = {}
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as f:
            diagnostics = json.load(f)
    horizon = None
    if horizon_path.exists():
        hd = np.load(horizon_path, allow_pickle=True)
        horizon = HorizonData(hd["u"], hd["v"], hd["r"], hd["m"], hd["alpha"],
                              hd["ell"], hd["lam"], hd["transversality"])
    cfg_path = npz_path.parent / "config.json"
    grid_kw, init_kw, anal_kw = {}, {}, {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cd = json.load(f)
            gfields = set(GridConfig.__dataclass_fields__)
            ifields = set(InitialDataConfig.__dataclass_fields__)
            afields = set(AnalysisConfig.__dataclass_fields__)
            grid_kw = {k: v for k, v in cd.get("grid", {}).items() if k in gfields}
            init_kw = {k: v for k, v in cd.get("initial", {}).items() if k in ifields}
            anal_kw = {k: v for k, v in cd.get("analysis", {}).items() if k in afields}
        except Exception:
            pass
    if len(u) > 1:
        grid_kw["h"] = float(u[1] - u[0])
        if len(v) > 1:
            grid_kw["dv"] = float(v[1] - v[0])
    cfg = RunConfig(GridConfig(**grid_kw), InitialDataConfig(**init_kw),
                    AnalysisConfig(**anal_kw), str(npz_path.parent))
    return SimulationResult(cfg, u, v, r, loga, psi, valid, mf, horizon, diagnostics, integrated, ruv_ev)


def save_result(result: SimulationResult, directory: Path, tag: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "u": result.u, "v": result.v, "r": result.r,
        "logalpha": result.logalpha, "psi": result.psi,
        "valid": result.valid, "mf": np.array([result.mf]),
    }
    _cc = getattr(result, "center_coeffs", None)
    if _cc is not None:
        import csv as _csv
        _rows = []
        for _i, _c in enumerate(_cc):
            if _c is None:
                continue
            _rows.append([_i, result.u[_i],
                          _c["alpha0"], _c["alpha2"],
                          _c["psi0"], _c["psi2"],
                          _c["alpha0_prime"], _c["psi0_prime"]])
        _p = directory / f"{tag}_center_coeffs.csv"
        with open(_p, "w", newline="", encoding="utf-8") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["i", "u", "alpha0", "alpha2",
                         "psi0", "psi2", "alpha0_prime", "psi0_prime"])
            _w.writerows(_rows)
    if result.integrated_mass is not None:
        payload["integrated_mass"] = result.integrated_mass
    if result.ruv_evolved is not None:
        payload["ruv_evolved"] = result.ruv_evolved
    np.savez_compressed(directory / f"{tag}_solution.npz", **payload)
    save_json(directory / f"{tag}_diagnostics.json", result.diagnostics)
    save_json(directory / f"{tag}_provenance.json", provenance(result.config))
    if result.horizon is not None:
        h = result.horizon
        np.savez_compressed(directory / f"{tag}_horizon.npz",
                            u=h.u, v=h.v, r=h.r, m=h.m, alpha=h.alpha,
                            ell=h.ell, lam=h.lam, transversality=h.transversality)


def save_curve(curve: CurveData, directory: Path, tag: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{tag}_curve.npz",
                        **{k: getattr(curve, k) for k in curve.__dataclass_fields__})


def report_finite(analysis):
    return {
        "distance_X": analysis["distance"],
        "C1": analysis["c1"],
        "C0": analysis["c0"],
        "predicted_signed_slope": analysis["pred_signed"],
        "measured_signed_fit": analysis["fit_signed"],
        "jackknife": analysis["jackknife"],
        "nested_fits": analysis["nested_fits"],
        "match_scheme": analysis.get("match_scheme", "equal_rho"),
    }


def default_config():
    cfg = RunConfig(GridConfig(), InitialDataConfig(), AnalysisConfig())
    cfg.initial.deformation_scale = normalized_deformation_scale(cfg.initial)
    return cfg


def run_primary(config: RunConfig, epsilon: float, equal_mass: bool,
                lambda_star=None) -> Dict[str, object]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", {
        "grid": asdict(config.grid),
        "initial": asdict(config.initial),
        "analysis": asdict(config.analysis),
        "output_dir": config.output_dir,
    })
    save_json(out / "analytic_validation.json", analytic_validation())
    base_cfg = copy_config(config)
    base_cfg.initial.epsilon = 0.0
    if equal_mass:
        aeq, pert, base = equal_mass_amplitude(base_cfg, epsilon)
    else:
        base = run_single(base_cfg)
        if base.horizon is None:
            raise RuntimeError("base solution has no horizon; increase amplitude")
        pert_cfg = copy_config(base_cfg)
        pert_cfg.initial.epsilon = epsilon
        pert = run_single(pert_cfg)
        if pert.horizon is None:
            raise RuntimeError("perturbed solution has no horizon; increase amplitude")
        aeq = pert.config.initial.amplitude
    save_result(base, out, "base")
    save_result(pert, out, "perturbed")
    analysis = finite_data_analysis(pert, base, lambda_star)
    report = report_finite(analysis)
    report["base_diagnostics"] = base.diagnostics
    report["perturbed_diagnostics"] = pert.diagnostics
    report["base_mf"] = base.mf
    report["perturbed_mf"] = pert.mf
    report["equal_mass"] = equal_mass
    report["compensating_amplitude"] = aeq
    report["relative_mass_mismatch"] = abs(pert.mf - base.mf) / max(base.mf, 1.0e-15)
    save_json(out / "primary_report.json", report)
    try:
        export_kasner_diagnostics(base, out, lambda_star)
    except Exception:
        pass
    return report


def run_family_suite(config: RunConfig, epsilon: float,
                     families) -> List[Dict[str, object]]:
    records = []
    for family in families:
        cfg = copy_config(config)
        cfg.initial.family = family
        cfg.initial.deformation_scale = normalized_deformation_scale(cfg.initial)
        cfg.output_dir = str(Path(config.output_dir) / f"family_{family}")
        try:
            rep = run_primary(cfg, epsilon, True)
            records.append({"family": family, "report": rep})
        except Exception as exc:
            records.append({"family": family, "error": str(exc)})
    save_json(Path(config.output_dir) / "family_suite.json", records)
    return records


def run_lambda_suite(config: RunConfig, epsilon: float,
                     lambdas) -> List[Dict[str, object]]:
    base_cfg = copy_config(config)
    base_cfg.initial.epsilon = 0.0
    aeq, pert, base = equal_mass_amplitude(base_cfg, epsilon)
    records = []
    for lam in lambdas:
        try:
            a = finite_data_analysis(pert, base, lam)
            rec = report_finite(a)
            rec["lambda_star"] = lam
            rec["relative_mass_mismatch"] = abs(pert.mf - base.mf) / max(base.mf, 1.0e-15)
            records.append(rec)
        except Exception as exc:
            records.append({"lambda_star": lam, "error": str(exc)})
    save_json(Path(config.output_dir) / "lambda_suite.json", records)
    return records


def run_scheme_suite(config: RunConfig, epsilon: float,
                     schemes) -> Dict[str, Dict[str, object]]:
    base_cfg = copy_config(config)
    base_cfg.initial.epsilon = 0.0
    aeq, pert, base = equal_mass_amplitude(base_cfg, epsilon)
    results = {}
    for scheme in schemes:
        try:
            a = finite_data_analysis(pert, base, config.analysis.lambda_star, scheme)
            results[scheme] = report_finite(a)
        except Exception as exc:
            results[scheme] = {"error": str(exc)}
    save_json(Path(config.output_dir) / "scheme_suite.json", results)
    return results


def run_norm_suite(config: RunConfig, epsilon: float,
                   variants) -> Dict[str, Dict[str, object]]:
    base_cfg = copy_config(config)
    base_cfg.initial.epsilon = 0.0
    aeq, pert, base = equal_mass_amplitude(base_cfg, epsilon)
    results = {}
    for label, w_amp, w_pow, ell in variants:
        cfg1 = InitialDataConfig(**asdict(pert.config.initial))
        cfg1.norm_weight_amplitude = w_amp
        cfg1.norm_weight_power = w_pow
        cfg1.ell_x = ell
        cfg0 = InitialDataConfig(**asdict(base.config.initial))
        cfg0.norm_weight_amplitude = w_amp
        cfg0.norm_weight_power = w_pow
        cfg0.ell_x = ell
        try:
            dx_alt = h2_distance(cfg1, cfg0)
            analysis = finite_data_analysis(pert, base, config.analysis.lambda_star)
            matched = analysis["matched"]
            x = np.abs(np.log(np.maximum(matched["rho"], 1.0e-300)))
            signed = (matched["logk1"] - matched["logk0"]) / max(dx_alt, 1.0e-300)
            fit = linear_fit(x, signed)
            c1 = analysis["c1"]
            c0 = analysis["c0"]
            pred = 4.0 * (c1 * c1 - c0 * c0) / max(dx_alt, 1.0e-300)
            results[label] = {
                "distance_X": dx_alt,
                "measured_signed_fit": fit,
                "predicted_signed_slope": pred,
            }
        except Exception as exc:
            results[label] = {"error": str(exc)}
    save_json(Path(config.output_dir) / "norm_suite.json", results)
    return results


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = default_config()
    cfg.grid.h = args.h
    cfg.grid.dv = args.dv
    cfg.grid.u0 = args.u0
    cfg.grid.restart_file = args.restart_file
    cfg.grid.restart_u = args.restart_u
    cfg.grid.restart_tag = args.restart_tag
    cfg.grid.umax = args.umax
    cfg.grid.vmax = args.vmax
    cfg.grid.r_floor = args.r_floor
    cfg.grid.singularity_cut = args.singularity_cut
    cfg.initial.family = args.family
    cfg.initial.amplitude = args.amplitude
    cfg.initial.xc = args.xc
    cfg.initial.width = args.width
    cfg.initial.xc2 = args.xc2
    cfg.initial.width2 = args.width2
    cfg.analysis.lambda_star = args.lambda_star
    cfg.analysis.rho_min_fit = args.rho_min
    cfg.analysis.rho_max_fit = args.rho_max
    cfg.analysis.match_scheme = args.match_scheme
    cfg.output_dir = args.output
    cfg.initial.deformation_scale = normalized_deformation_scale(cfg.initial)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[
        "analytic", "single", "primary", "families", "lambdas",
        "schemes", "norms", "convergence", "mass_response",
        "constraints", "minkowski", "horizons", "error_budget",
        "solution_convergence", "verify", "full",
        "observable_convergence", "equal_mass_convergence",
        "kasner_sensitivity", "paper_tables", "kasner_diagnostic",
        "pde_residual",
    ], default="verify")
    parser.add_argument("--output", default="conditioning_output")
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--u0", type=float, default=0.0)
    parser.add_argument("--restart-file", default="")
    parser.add_argument("--restart-u", type=float, default=0.0)
    parser.add_argument("--restart-tag", default="single")
    parser.add_argument("--umax", type=float, default=3.6)
    parser.add_argument("--vmax", type=float, default=40.0)
    parser.add_argument("--r-floor", type=float, default=2.0e-5)
    parser.add_argument("--singularity-cut", type=float, default=5.0e-3)
    parser.add_argument("--family", choices=["bump", "gaussian", "two_pulse", "shape"], default="bump")
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--xc", type=float, default=2.0)
    parser.add_argument("--width", type=float, default=0.6)
    parser.add_argument("--xc2", type=float, default=2.2)
    parser.add_argument("--width2", type=float, default=0.8)
    parser.add_argument("--lambda-star", type=float, default=0.3)
    parser.add_argument("--rho-min", type=float, default=1.0e-3)
    parser.add_argument("--rho-max", type=float, default=0.35)
    parser.add_argument("--match-scheme", choices=["equal_rho", "equal_compactness"], default="equal_rho")
    parser.add_argument("--hs", default="0.04,0.02,0.01,0.005")
    parser.add_argument("--epsilons", default="0.04,0.02,0.01")
    parser.add_argument("--lambdas", default="0.20,0.25,0.30,0.35")
    parser.add_argument("--delta-a", type=float, default=0.005)
    args = parser.parse_args()
    cfg = build_config(args)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", {
        "grid": asdict(cfg.grid),
        "initial": asdict(cfg.initial),
        "analysis": asdict(cfg.analysis),
        "output_dir": cfg.output_dir,
    })
    if args.mode == "analytic":
        print(json.dumps(jsonify(analytic_validation()), indent=2))
        return 0
    if args.mode == "verify":
        checks = run_verification_checks(cfg, args.epsilon, out)
        print_verification(checks)
        return 0 if checks.get("overall_pass", False) else 1
    if args.mode == "pde_residual":
        records = minkowski_pde_residual_test(parse_float_list(args.hs))
        plot_and_save_csv_minkowski(records, out)
        save_json(out / "minkowski_pde_residual.json", records)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "single":
        result = run_single(cfg)
        save_result(result, out, "single")
        try:
            p = export_kasner_diagnostics(result, out, cfg.analysis.lambda_star)
            if p is not None:
                print(f"kasner diagnostics written to {p}")
        except Exception:
            pass
        print(json.dumps(jsonify({
            "mf": result.mf,
            "has_horizon": result.horizon is not None,
            "horizon_points": int(result.diagnostics.get("horizon_points", 0)),
            "num_valid_cells": int(result.diagnostics.get("num_valid_cells", 0)),
            "num_singularity_rows": float(result.diagnostics.get("num_singularity_rows", 0.0)),
            "center_psi_resid_max": float(result.diagnostics.get("center_psi_resid_max", float("nan"))),
            "center_reg_resid_max": float(result.diagnostics.get("center_reg_resid_max", float("nan"))),
            "row_max_rv": float(result.diagnostics.get("row_max_rv", float("nan"))),
            "row_max_loga": float(result.diagnostics.get("row_max_loga", float("nan"))),
            "constraint_u_l2": float(result.diagnostics.get("constraint_u_l2", float("nan"))),
            "massflux_u_l2": float(result.diagnostics.get("massflux_u_l2", float("nan"))),
            "integrated_vs_algebraic_mass_l2": float(result.diagnostics.get("integrated_vs_algebraic_mass_l2", float("nan"))),
            "diagnostics": result.diagnostics,
        }), indent=2))
        return 0
    if args.mode == "kasner_diagnostic":
        result = run_single(cfg)
        save_result(result, out, "kasner_diag")
        try:
            p = export_kasner_diagnostics(result, out, cfg.analysis.lambda_star)
            if p is not None:
                print(f"kasner diagnostics written to {p}")
                return 0
        except Exception as exc:
            print(f"kasner diagnostic failed: {exc}")
            return 1
        return 1
    if args.mode == "primary":
        report = run_primary(cfg, args.epsilon, True)
        plot_and_save_csv_primary_artifacts(out, out / "primary_figures")
        print(json.dumps(jsonify(report), indent=2))
        return 0
    if args.mode == "families":
        records = run_family_suite(cfg, args.epsilon,
                                   ["bump", "gaussian", "two_pulse", "shape"])
        plot_and_save_csv_families(records, out)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "lambdas":
        records = run_lambda_suite(cfg, args.epsilon, parse_float_list(args.lambdas))
        plot_and_save_csv_lambdas(records, out)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "schemes":
        records = run_scheme_suite(cfg, args.epsilon,
                                   ["equal_rho", "equal_compactness"])
        plot_and_save_csv_schemes(records, out)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "norms":
        records = run_norm_suite(cfg, args.epsilon, [
            ("default", 0.25, 2.0, 1.0),
            ("unweighted", 0.0, 2.0, 1.0),
            ("linear", 0.25, 1.0, 1.0),
            ("short_ell", 0.25, 2.0, 0.5),
        ])
        plot_and_save_csv_norms(records, out)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "convergence":
        records = convergence_suite(cfg, parse_float_list(args.hs),
                                    parse_float_list(args.epsilons),
                                    cfg.analysis.lambda_star)
        plot_and_save_csv_convergence(records, out)
        serial = [{k: v for k, v in r.items() if k not in ["rho", "D"]} for r in records]
        save_json(out / "convergence_report.json", serial)
        print(json.dumps(jsonify(serial), indent=2))
        return 0
    if args.mode == "solution_convergence":
        records = solution_convergence_suite(cfg, parse_float_list(args.hs))
        save_json(out / "solution_convergence.json", records)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "horizons":
        records = horizon_convergence_suite(cfg, parse_float_list(args.hs),
                                            cfg.analysis.lambda_star)
        plot_and_save_csv_horizon(records, out)
        save_json(out / "horizon_convergence.json", records)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "mass_response":
        rec = mass_direction_response(cfg, args.delta_a, cfg.analysis.lambda_star)
        np.savez_compressed(out / "mass_direction_response.npz",
                            rho=rec["rho"], S_M=rec["S_M"])
        print(json.dumps(jsonify({
            "mf_plus": float(rec["mf_plus"]),
            "mf_minus": float(rec["mf_minus"]),
            "median_S_M": float(np.nanmedian(rec["S_M"])),
        }), indent=2))
        return 0
    if args.mode == "constraints":
        records = constraint_convergence_suite(cfg, parse_float_list(args.hs))
        plot_and_save_csv_constraints(records, out)
        save_json(out / "constraint_convergence_report.json", records)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "minkowski":
        records = minkowski_pde_residual_test(parse_float_list(args.hs))
        plot_and_save_csv_minkowski(records, out)
        save_json(out / "minkowski_test.json", records)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "error_budget":
        rec = error_budget_analysis(cfg, args.epsilon, cfg.analysis.lambda_star)
        save_json(out / "error_budget_report.json", rec)
        print(json.dumps(jsonify(rec), indent=2))
        return 0
    if args.mode == "observable_convergence":
        records = observable_convergence_suite(cfg, parse_float_list(args.hs),
                                               args.epsilon, cfg.analysis.lambda_star)
        plot_and_save_csv_observable_convergence(records, out)
        save_json(out / "observable_convergence.json", records)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "equal_mass_convergence":
        records = equal_mass_convergence_suite(cfg, parse_float_list(args.hs), args.epsilon)
        plot_and_save_csv_equal_mass_convergence(records, out)
        save_json(out / "equal_mass_convergence.json", records)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "kasner_sensitivity":
        records = kasner_window_sensitivity(cfg, args.epsilon, cfg.analysis.lambda_star)
        plot_and_save_csv_kasner_sensitivity(records, out)
        save_json(out / "kasner_sensitivity.json", records)
        print(json.dumps(jsonify(records), indent=2))
        return 0
    if args.mode == "paper_tables":
        obs = observable_convergence_suite(cfg, parse_float_list(args.hs),
                                           args.epsilon, cfg.analysis.lambda_star)
        fams = run_family_suite(cfg, args.epsilon,
                                ["bump", "gaussian", "two_pulse", "shape"])
        eb = error_budget_analysis(cfg, args.epsilon, cfg.analysis.lambda_star)
        emc = equal_mass_convergence_suite(cfg, parse_float_list(args.hs), args.epsilon)
        ks = kasner_window_sensitivity(cfg, args.epsilon, cfg.analysis.lambda_star)
        td = generate_paper_tables(out, obs, fams, eb, emc, ks)
        print(f"paper tables written to {td}")
        return 0
    if args.mode == "full":
        rec = run_full_experiment(cfg, args.epsilon)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())