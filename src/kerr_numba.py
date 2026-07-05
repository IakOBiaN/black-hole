"""Numba-accelerated Kerr ray tracer.

Same physics as src/kerr_tracer.py, but each ray is integrated by a compiled
per-ray kernel and the rays are spread across CPU cores with prange, instead
of marching the whole batch in lockstep with numpy. The Carter constant q does
not enter the analytic right-hand side (it cancels in R + Delta*Theta), so the
kernel only needs the axial angular momentum b.
"""

import numpy as np
from numba import njit, prange

from .kerr import (delta, rho2, alpha, omega, varpi, horizon_radius)

_HALF_PI = np.pi / 2.0
_TWO_PI = 2.0 * np.pi


@njit(cache=True, fastmath=True, inline="always")
def _rhs(r, theta, p_r, p_theta, b, a):
    ct = np.cos(theta)
    stn = np.sin(theta)
    ct2 = ct * ct
    st2 = stn * stn + 1.0e-9

    rr2 = r * r + a * a * ct2
    d = r * r - 2.0 * r + a * a
    P = r * r + a * a - a * b
    bb = (b - a) ** 2 + ct2 * (b * b / st2 - a * a)
    n = P * P - d * bb

    dr = d / rr2 * p_r
    dtheta = p_theta / rr2

    db_db = 2.0 * (b - a) + 2.0 * b * ct2 / st2
    dn_db = -2.0 * a * P - d * db_db
    dphi = -dn_db / (2.0 * d * rr2)

    drr2_dr = 2.0 * r
    dd_dr = 2.0 * r - 2.0
    dn_dr = 2.0 * P * (2.0 * r) - dd_dr * bb
    t1 = -0.5 * p_r * p_r * (dd_dr * rr2 - d * drr2_dr) / (rr2 * rr2)
    t2 = 0.5 * p_theta * p_theta * drr2_dr / (rr2 * rr2)
    drrho = d * rr2
    ddrrho_dr = dd_dr * rr2 + d * drr2_dr
    t3 = 0.5 * (dn_dr * drrho - n * ddrrho_dr) / (drrho * drrho)
    dp_r = t1 + t2 + t3

    drr2_dth = -2.0 * a * a * stn * ct
    db_dth = -2.0 * b * b * ct / (st2 * stn) + 2.0 * a * a * stn * ct
    dn_dth = -d * db_dth
    s1 = 0.5 * d * p_r * p_r * drr2_dth / (rr2 * rr2)
    s2 = 0.5 * p_theta * p_theta * drr2_dth / (rr2 * rr2)
    s3 = 0.5 / d * (dn_dth * rr2 - n * drr2_dth) / (rr2 * rr2)
    dp_theta = s1 + s2 + s3

    return dr, dtheta, dphi, dp_r, dp_theta


@njit(cache=True, fastmath=True)
def _trace_ray(r, theta, p_r, p_theta, b, a, inner, outer, r_h, r_escape,
               dzeta, max_steps):
    phi = 0.0
    r_prev, theta_prev, phi_prev = r, theta, phi
    for _ in range(max_steps):
        pole_dist = min(theta_prev, np.pi - theta_prev)
        delta_prev = r_prev * r_prev - 2.0 * r_prev + a * a
        dz = (dzeta * min(max(r_prev / 5.0, 0.5), 4.0)
              * min(max((pole_dist / 0.4) ** 2, 0.01), 1.0)
              * min(max(delta_prev / 0.3, 0.05), 1.0))

        k1 = _rhs(r, theta, p_r, p_theta, b, a)
        k2 = _rhs(r + 0.5 * dz * k1[0], theta + 0.5 * dz * k1[1],
                  p_r + 0.5 * dz * k1[3], p_theta + 0.5 * dz * k1[4], b, a)
        k3 = _rhs(r + 0.5 * dz * k2[0], theta + 0.5 * dz * k2[1],
                  p_r + 0.5 * dz * k2[3], p_theta + 0.5 * dz * k2[4], b, a)
        k4 = _rhs(r + dz * k3[0], theta + dz * k3[1],
                  p_r + dz * k3[3], p_theta + dz * k3[4], b, a)

        rn = r + dz / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        tn = theta + dz / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        pn = phi + dz / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        prn = p_r + dz / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        ptn = p_theta + dz / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])

        if tn < 0.0:
            tn, ptn, pn = -tn, -ptn, pn + np.pi
        elif tn > np.pi:
            tn, ptn, pn = _TWO_PI - tn, -ptn, pn + np.pi

        delta_new = rn * rn - 2.0 * rn + a * a
        if rn <= r_h or delta_new <= 1.0e-2 or rn < 0.0:
            return np.nan, np.nan, 1        # captured
        if rn >= r_escape:
            return np.nan, np.nan, 0        # escaped

        if (theta_prev - _HALF_PI) * (tn - _HALF_PI) < 0.0:
            frac = (_HALF_PI - theta_prev) / (tn - theta_prev)
            r_cross = r_prev + frac * (rn - r_prev)
            if inner <= r_cross <= outer:
                az = phi_prev + frac * (pn - phi_prev)
                return r_cross, az, 0       # disk hit

        r, theta, phi, p_r, p_theta = rn, tn, pn, prn, ptn
        r_prev, theta_prev, phi_prev = r, theta, phi

    return np.nan, np.nan, 0


@njit(cache=True, parallel=True, fastmath=True)
def _trace_all(r_c, theta_c, p_r, p_theta, b, a, inner, outer, r_h, r_escape,
               dzeta, max_steps, radius, azimuth, captured):
    for i in prange(p_r.shape[0]):
        rc, az, cap = _trace_ray(r_c, theta_c, p_r[i], p_theta[i], b[i], a,
                                  inner, outer, r_h, r_escape, dzeta, max_steps)
        radius[i] = rc
        azimuth[i] = az
        captured[i] = cap


def trace_batch_kerr(camera, a, disk, dzeta=0.1, max_steps=4000, r_escape=None):
    """Trace all rays with the Numba kernel. Returns (radius, b, azimuth,
    captured), matching src.kerr_tracer.trace_batch_kerr."""
    pos = camera.position
    r_c = float(np.linalg.norm(pos))
    theta_c = float(np.arccos(pos[2] / r_c))
    if r_escape is None:
        r_escape = 1.2 * r_c

    dirs = camera.ray_directions()
    h, w, _ = dirs.shape
    d = dirs.reshape(-1, 3)

    st, ct = np.sin(theta_c), np.cos(theta_c)
    n_r = d @ np.array([st, 0.0, ct])
    n_theta = d @ np.array([ct, 0.0, -st])
    n_phi = d[:, 1]

    rho_c = np.sqrt(rho2(r_c, theta_c, a))
    e_f = 1.0 / (alpha(r_c, theta_c, a)
                 + omega(r_c, theta_c, a) * varpi(r_c, theta_c, a) * n_phi)
    p_r = e_f * rho_c / np.sqrt(delta(r_c, a)) * n_r
    p_theta = e_f * rho_c * n_theta
    b = e_f * varpi(r_c, theta_c, a) * n_phi

    n = d.shape[0]
    radius = np.empty(n)
    azimuth = np.empty(n)
    captured = np.empty(n, dtype=np.int8)
    _trace_all(r_c, theta_c, p_r, p_theta, b, a, disk.inner, disk.outer,
               horizon_radius(a) * 1.002, r_escape, dzeta, max_steps,
               radius, azimuth, captured)

    return (radius.reshape(h, w), b.reshape(h, w), azimuth.reshape(h, w),
            captured.reshape(h, w).astype(bool))
