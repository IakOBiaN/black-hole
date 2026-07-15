"""Numba-accelerated Kerr ray tracer.

Backward ray tracing per James et al. (2015), Appendix A.1: for each pixel we
take the momentum of the *received* photon -- its propagation direction n_F
points from the scene into the camera, so n_F = -d for a camera ray direction
d -- with FIDO-measured energy E_F = 1/(alpha + omega varpi n_F_phi), and
integrate the super-Hamiltonian ray equations backward in time (negative
affine steps) from the camera into the scene. This gets both the trajectory
and the sign of the photon's axial angular momentum b right; b feeds the
Doppler shift, so the approaching side of the disk blueshifts.

Each ray is integrated by a compiled per-ray kernel spread across CPU cores
with prange. Rays record up to `max_hits` crossings of the equatorial plane
inside [r_min, r_max] and keep going, so a semi-transparent disk can be
composited from several lensed layers (the disk in the article is only
marginally optically thick).
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
    # max() keeps the exact-pole limit finite: at theta = 0 the numerator
    # vanishes with b^2 (a pole-on camera has b = 0), and 0/0 would raise.
    db_dth = (-2.0 * b * b * ct / (st2 * max(stn, 1.0e-12))
              + 2.0 * a * a * stn * ct)
    dn_dth = -d * db_dth
    s1 = 0.5 * d * p_r * p_r * drr2_dth / (rr2 * rr2)
    s2 = 0.5 * p_theta * p_theta * drr2_dth / (rr2 * rr2)
    s3 = 0.5 / d * (dn_dth * rr2 - n * drr2_dth) / (rr2 * rr2)
    dp_theta = s1 + s2 + s3

    return dr, dtheta, dphi, dp_r, dp_theta


@njit(cache=True, fastmath=True)
def _trace_ray(r, theta, p_r, p_theta, b, a, phi0, r_min, r_max, r_h,
               r_escape, dzeta, max_steps, hit_r, hit_phi):
    """March one ray backward in time; record equatorial crossings with
    radius in [r_min, r_max] into hit_r/hit_phi (length = max hits). Returns
    (number of hits, captured flag). phi0 is the ray's starting azimuth
    (nonzero only for a camera on the spin axis, where each ray carries its
    own meridian)."""
    max_hits = hit_r.shape[0]
    n_hits = 0
    phi = phi0
    r_prev, theta_prev, phi_prev = r, theta, phi
    for _ in range(max_steps):
        pole_dist = min(theta_prev, np.pi - theta_prev)
        delta_prev = r_prev * r_prev - 2.0 * r_prev + a * a
        # Backward in time: negative affine step, shrunk near the horizon
        # (Delta -> 0) and the spin axis where Boyer-Lindquist is singular.
        # Near-axis rays need many small steps; pole-on cameras should pass
        # a generous max_steps budget.
        dz = -(dzeta * min(max(r_prev / 5.0, 0.5), 4.0)
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
            return n_hits, 1                # fell to the horizon
        if rn >= r_escape:
            return n_hits, 0                # left the scene

        if (theta_prev - _HALF_PI) * (tn - _HALF_PI) < 0.0:
            frac = (_HALF_PI - theta_prev) / (tn - theta_prev)
            r_cross = r_prev + frac * (rn - r_prev)
            if r_min <= r_cross <= r_max:
                hit_r[n_hits] = r_cross
                # If the ray was reflected through the pole this step, phi
                # jumped by pi and interpolating across the jump smears the
                # azimuth; take the endpoint instead.
                dphi_step = pn - phi_prev
                if -_HALF_PI < dphi_step < _HALF_PI:
                    hit_phi[n_hits] = phi_prev + frac * dphi_step
                else:
                    hit_phi[n_hits] = pn
                n_hits += 1
                if n_hits >= max_hits:
                    return n_hits, 0

        r, theta, phi, p_r, p_theta = rn, tn, pn, prn, ptn
        r_prev, theta_prev, phi_prev = r, theta, phi

    return n_hits, 0


@njit(cache=True, parallel=True, fastmath=True)
def _trace_all(r_c, theta_c, p_r, p_theta, b, a, phi0, r_min, r_max, r_h,
               r_escape, dzeta, max_steps, hits_r, hits_phi, n_hits,
               captured):
    for i in prange(p_r.shape[0]):
        n, cap = _trace_ray(r_c, theta_c, p_r[i], p_theta[i], b[i], a,
                            phi0[i], r_min, r_max, r_h, r_escape, dzeta,
                            max_steps, hits_r[i], hits_phi[i])
        n_hits[i] = n
        captured[i] = cap


def _camera_momenta(camera, a):
    """Conserved quantities of the received photon for every pixel: canonical
    p_r, p_theta and axial angular momentum b = p_phi, with -p_t = 1, plus
    each ray's starting azimuth phi0. The camera is treated as a FIDO
    (article eq. A.9 with beta = 0)."""
    pos = camera.position
    r_c = float(np.linalg.norm(pos))
    theta_c = float(np.arccos(np.clip(pos[2] / r_c, -1.0, 1.0)))

    dirs = camera.ray_directions()
    h, w, _ = dirs.shape
    d = dirs.reshape(-1, 3)

    st, ct = np.sin(theta_c), np.cos(theta_c)
    rho_c = np.sqrt(rho2(r_c, theta_c, a))

    if st < 1.0e-10:
        # Camera on the spin axis, where the phi coordinate degenerates:
        # every ray lives in its own meridian plane. Give each ray the
        # azimuth of its transverse direction; there b = 0 exactly (zero
        # axial angular momentum through the axis), so the singular
        # b^2/sin^2 terms vanish along the whole path.
        n_f = -d
        n_r = n_f[:, 2] * np.sign(ct if ct != 0.0 else 1.0)
        n_theta = np.hypot(n_f[:, 0], n_f[:, 1])
        phi0 = np.arctan2(n_f[:, 1], n_f[:, 0])
        e_f = 1.0 / alpha(r_c, theta_c, a)
        p_r = e_f * rho_c / np.sqrt(delta(r_c, a)) * n_r
        p_theta = e_f * rho_c * n_theta
        b = np.zeros(d.shape[0])
        return r_c, theta_c, (h, w), p_r, p_theta, b, phi0

    # n_F points along the incoming photon's propagation: opposite to the
    # camera's look direction d.
    n_r = -(d @ np.array([st, 0.0, ct]))
    n_theta = -(d @ np.array([ct, 0.0, -st]))
    n_phi = -d[:, 1]

    e_f = 1.0 / (alpha(r_c, theta_c, a)
                 + omega(r_c, theta_c, a) * varpi(r_c, theta_c, a) * n_phi)
    p_r = e_f * rho_c / np.sqrt(delta(r_c, a)) * n_r
    p_theta = e_f * rho_c * n_theta
    b = e_f * varpi(r_c, theta_c, a) * n_phi
    return r_c, theta_c, (h, w), p_r, p_theta, b, np.zeros(d.shape[0])


def trace_batch_kerr_multi(camera, a, r_min, r_max, max_hits=8, dzeta=0.1,
                           max_steps=6000, r_escape=None):
    """Trace all rays, recording up to max_hits equatorial crossings each.

    Returns (hits_r, hits_phi, n_hits, b, captured): hits_r and hits_phi have
    shape (H, W, max_hits) ordered front (camera side) to back, n_hits the
    crossing count per pixel, b the photon axial angular momentum, captured a
    mask of rays that end on the horizon."""
    r_c, theta_c, (h, w), p_r, p_theta, b, phi0 = _camera_momenta(camera, a)
    if r_escape is None:
        r_escape = 1.2 * r_c

    # Nudge an exactly polar camera off theta = 0: the kernel state itself
    # must not start on the coordinate singularity.
    theta_c = max(theta_c, 1.0e-7)

    n = p_r.shape[0]
    hits_r = np.empty((n, max_hits))
    hits_phi = np.empty((n, max_hits))
    n_hits = np.empty(n, dtype=np.int64)
    captured = np.empty(n, dtype=np.int8)
    _trace_all(r_c, theta_c, p_r, p_theta, b, a, phi0, r_min, r_max,
               horizon_radius(a) * 1.002, r_escape, dzeta, max_steps,
               hits_r, hits_phi, n_hits, captured)

    return (hits_r.reshape(h, w, max_hits), hits_phi.reshape(h, w, max_hits),
            n_hits.reshape(h, w), b.reshape(h, w),
            captured.reshape(h, w).astype(bool))


def trace_batch_kerr(camera, a, disk, dzeta=0.1, max_steps=4000,
                     r_escape=None):
    """Single-hit opaque-disk trace. Returns (radius, b, azimuth, captured),
    matching src.kerr_tracer.trace_batch_kerr: radius/azimuth of the first
    disk crossing (NaN if none), captured True only for rays that reach the
    horizon without hitting the disk."""
    hits_r, hits_phi, n_hits, b, captured = trace_batch_kerr_multi(
        camera, a, disk.inner, disk.outer, max_hits=1, dzeta=dzeta,
        max_steps=max_steps, r_escape=r_escape)

    hit = n_hits > 0
    radius = np.where(hit, hits_r[..., 0], np.nan)
    azimuth = np.where(hit, hits_phi[..., 0], np.nan)
    return radius, b, azimuth, captured & ~hit
