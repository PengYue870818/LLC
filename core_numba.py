"""
Optional numba-JIT acceleration kernels for the double-null row march
(equations for r, ln a, and Psi on each null diamond). Imported lazily
by `Yue_Du`; if numba is unavailable the solver falls back to a
pure-Python loop that is numerically equivalent but ~100x slower and
produces slightly different numbers near the singularity.

Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
All rights reserved.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numba kernels for the double-null Einstein-scalar solver.

The null-diamond row march is a sequential scalar recurrence; a jitted
implementation is much faster than the pure-python/numpy version, which
makes the fine v-resolution needed near the spacelike singularity feasible.

Conventions (identical to Yue_Du.py):

    metric      ds^2 = -alpha^2 du dv + r^2 dOmega^2
    Eq. (10)    r_{uv}      = - r_u r_v / r - alpha^2/(4r)
    Eq. (11)    (ln a)_{uv} =   r_u r_v / r^2 + alpha^2/(4r^2) - psi_u psi_v/2
    Eq. (12)    psi_{uv}    = - (r_u psi_v + r_v psi_u)/r
"""
import math

from numba import njit


@njit(cache=True, inline="always")
def rv_is_negative(rs, re, rw, rn_pred):
    return ((re + rn_pred) - (rs + rw)) < 0.0


@njit(cache=True, inline="always")
def sources(r, loga, ru, rv, psu, psv, r_floor, freeze_alpha=0):
    a2 = math.exp(min(max(2.0 * loga, -700.0), 700.0))
    rr = abs(r)
    if rr < r_floor:
        rr = r_floor
    # horizon-regular form: use the Misner-Sharp mass to cancel the 1/r^2
    # terms analytically
    mm = 0.5 * rr * (1.0 + 4.0 * ru * rv / a2)
    sr = -a2 * mm / (2.0 * rr * rr)
    sa = a2 * mm / (2.0 * rr * rr * rr) - 0.5 * psu * psv
    if freeze_alpha == 1:
        sa = 0.0
    sp = -(ru * psv + rv * psu) / rr
    return sr, sa, sp


@njit(cache=True, inline="always")
def sources_center(alpha0, alpha2, psi0, psi2, alpha0p, psi0p, x_C):
    a0 = abs(alpha0)
    if a0 < 1.0e-30:
        a0 = 1.0e-30
    sr = -0.5 * alpha2 * x_C
    sa = -alpha2 / (4.0 * a0) - (psi0p * psi0p) / 16.0
    sp = psi2 - (alpha0p * psi0p) / (2.0 * a0)
    return sr, sa, sp


@njit(cache=True)
def march_row(r, loga, psi, valid, ruv, i, jc, jstart, nv, du, dv,
              cc, use_cc, cut, r_floor, sing_cut,
              alpha_floor, alpha_ceiling, ru_max, rv_max, freeze_alpha=0):
    """Fill row i from row i-1, marching from column jstart.

    The centre column jc is fixed (r=0, loga=0); columns jc+1..jstart-1 are
    assumed to have been filled by the regular-centre Taylor block.
    Returns the first column j at which the update failed (nv if complete).
    """
    r[i, jc] = 0.0
    loga[i, jc] = 0.0
    valid[i, jc] = True
    for j in range(jstart, nv - 1):
        if not (valid[i - 1, j] and valid[i - 1, j + 1] and valid[i, j]):
            return j
        rs = r[i - 1, j]
        re = r[i - 1, j + 1]
        rw = r[i, j]
        aas = loga[i - 1, j]
        aae = loga[i - 1, j + 1]
        aaw = loga[i, j]
        ps = psi[i - 1, j]
        pe = psi[i - 1, j + 1]
        pw = psi[i, j]
        rn_pred = re + rw - rs
        an_pred = aae + aaw - aas
        pn_pred = pe + pw - ps
        x_C = 0.25 * dv * (2.0 * j + 1.0) - 0.25 * du * (2.0 * i + 1.0)
        ru = ((rw + rn_pred) - (rs + re)) / (2.0 * du)
        rv = ((re + rn_pred) - (rs + rw)) / (2.0 * dv)
        psu = ((pw + pn_pred) - (ps + pe)) / (2.0 * du)
        psv = ((pe + pn_pred) - (ps + pw)) / (2.0 * dv)
        if abs(ru) > ru_max or abs(rv) > rv_max:
            return j
        rc = 0.25 * (rs + re + rw + rn_pred)
        ac = 0.25 * (aas + aae + aaw + an_pred)
        if use_cc and abs(x_C) < cut:
            sr0, sa0, sp0 = sources_center(cc[0], cc[1], cc[2], cc[3],
                                           cc[4], cc[5], x_C)
        else:
            sr0, sa0, sp0 = sources(rc, ac, ru, rv, psu, psv, r_floor, freeze_alpha)
        if not (math.isfinite(sr0) and math.isfinite(sa0) and math.isfinite(sp0)):
            return j
        rn_star = re + rw - rs + du * dv * sr0
        an_star = aae + aaw - aas + du * dv * sa0
        pn_star = pe + pw - ps + du * dv * sp0
        ru1 = ((rw + rn_star) - (rs + re)) / (2.0 * du)
        rv1 = ((re + rn_star) - (rs + rw)) / (2.0 * dv)
        psu1 = ((pw + pn_star) - (ps + pe)) / (2.0 * du)
        psv1 = ((pe + pn_star) - (ps + pw)) / (2.0 * dv)
        if abs(ru1) > ru_max or abs(rv1) > rv_max:
            return j
        rc1 = 0.25 * (rs + re + rw + rn_star)
        ac1 = 0.25 * (aas + aae + aaw + an_star)
        if use_cc and abs(x_C) < cut:
            sr1, sa1, sp1 = sources_center(cc[0], cc[1], cc[2], cc[3],
                                           cc[4], cc[5], x_C)
        else:
            sr1, sa1, sp1 = sources(rc1, ac1, ru1, rv1, psu1, psv1, r_floor)
        if not (math.isfinite(sr1) and math.isfinite(sa1) and math.isfinite(sp1)):
            return j
        rn = re + rw - rs + du * dv * sr1
        an = aae + aaw - aas + du * dv * sa1
        pn = pe + pw - ps + du * dv * sp1
        if not (math.isfinite(rn) and math.isfinite(an) and math.isfinite(pn)):
            return j
        alpha = math.exp(min(max(an, -700.0), 700.0))
        if rn < 0.0:
            return j
        if rn < sing_cut and rv_is_negative(rs, re, rw, rn_pred):
            return j
        if alpha < alpha_floor or alpha > alpha_ceiling:
            return j
        r[i, j + 1] = rn
        loga[i, j + 1] = an
        psi[i, j + 1] = pn
        ruv[i, j + 1] = 0.5 * (sr0 + sr1)
        valid[i, j + 1] = True
    return nv
