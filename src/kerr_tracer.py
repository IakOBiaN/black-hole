"""Vectorized ray tracer for a Kerr black hole.

Each camera ray is converted into the conserved quantities (axial angular
momentum b and Carter constant q) of the *received* photon using the FIDO
orthonormal frame of James et al. (2015), Appendix A.1: the photon's
propagation direction n_F points into the camera, so n_F = -d for camera ray
direction d, and the null geodesic is marched backward in time (negative
affine steps) with a fixed-step RK4. The camera is treated as a FIDO. Rays
are classified as hitting the equatorial disk, the horizon, or escaping to
infinity.
"""

import numpy as np

from .kerr import (delta, rho2, sigma, alpha, omega, varpi, horizon_radius,
                   rhs)


def _rk4_step(y, b, q, a, dz):
    """RK4 step with a per-ray affine step dz (shape matches the ray axis)."""
    dzb = dz[None, :]
    k1 = rhs(y, b, q, a)
    k2 = rhs(y + 0.5 * dzb * k1, b, q, a)
    k3 = rhs(y + 0.5 * dzb * k2, b, q, a)
    k4 = rhs(y + dzb * k3, b, q, a)
    return y + dzb / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def trace_batch_kerr(camera, a, disk, dzeta=0.1, max_steps=4000,
                     r_escape=None):
    """Trace all rays. Returns (radius, b, azimuth, captured): radius is the
    equatorial disk hit radius (NaN if none), b the photon axial angular
    momentum (for the Doppler shift), azimuth the disk-plane angle of the hit,
    captured a mask of rays that fell through the horizon.

    dzeta must stay small (~0.1) or the fixed-step march misclassifies rays
    near the shadow boundary into speckle."""
    pos = camera.position
    r_c = float(np.linalg.norm(pos))
    theta_c = float(np.arccos(pos[2] / r_c))
    if r_escape is None:
        r_escape = 1.2 * r_c

    dirs = camera.ray_directions()
    h, w, _ = dirs.shape
    d = dirs.reshape(-1, 3)
    n = d.shape[0]

    st, ct = np.sin(theta_c), np.cos(theta_c)
    r_hat = np.array([st, 0.0, ct])
    theta_hat = np.array([ct, 0.0, -st])
    # The received photon propagates opposite to the look direction d.
    n_r = -(d @ r_hat)
    n_theta = -(d @ theta_hat)
    n_phi = -d[:, 1]  # phi_hat = (0, 1, 0) at phi = 0

    rho_c = np.sqrt(rho2(r_c, theta_c, a))
    delta_c = delta(r_c, a)
    e_f = 1.0 / (alpha(r_c, theta_c, a)
                 + omega(r_c, theta_c, a) * varpi(r_c, theta_c, a) * n_phi)
    p_r = e_f * rho_c / np.sqrt(delta_c) * n_r
    p_theta = e_f * rho_c * n_theta
    b = e_f * varpi(r_c, theta_c, a) * n_phi
    q = p_theta ** 2 + ct * ct * (b * b / (st * st) - a * a)

    y = np.stack([np.full(n, r_c), np.full(n, theta_c), np.zeros(n),
                  p_r, p_theta])
    radius = np.full(n, np.nan)
    azimuth = np.full(n, np.nan)
    captured = np.zeros(n, dtype=bool)
    r_h = horizon_radius(a) * 1.002
    half_pi = np.pi / 2.0

    # Active rays are kept compacted so finished rays stop costing anything.
    idx = np.arange(n)
    bw, qw = b.copy(), q.copy()
    r_prev, theta_prev, phi_prev = y[0].copy(), y[1].copy(), y[2].copy()
    for _ in range(max_steps):
        if idx.size == 0:
            break
        # Shrink the step near the coordinate singularities so the ray never
        # overshoots them: near the horizon (Delta -> 0, the 1/Delta terms in
        # the potential blow up) and near the spin axis (sin(theta) -> 0).
        pole_dist = np.minimum(theta_prev, np.pi - theta_prev)
        delta_prev = delta(r_prev, a)
        # Negative affine step: the ray is integrated backward in time.
        dz = -(dzeta * np.clip(r_prev / 5.0, 0.5, 4.0)
               * np.clip((pole_dist / 0.4) ** 2, 0.01, 1.0)
               * np.clip(delta_prev / 0.3, 0.05, 1.0))
        y = _rk4_step(y, bw, qw, a, dz)

        # Reflect rays that step over a pole: theta stays in [0, pi], p_theta
        # flips, and phi jumps by pi (Boyer-Lindquist is singular at the axis).
        below = y[1] < 0.0
        y[1, below], y[4, below], y[2, below] = (-y[1, below], -y[4, below],
                                                 y[2, below] + np.pi)
        above = y[1] > np.pi
        y[1, above], y[4, above], y[2, above] = (2.0 * np.pi - y[1, above],
                                                 -y[4, above], y[2, above] + np.pi)

        r, theta, phi = y[0], y[1], y[2]
        done = np.zeros(idx.size, dtype=bool)

        # A ray inside the near-horizon danger zone is captured; a crossing
        # recorded during that plunge is spurious (the 1/Delta blow-up) and is
        # suppressed. delta < 1e-2 stays well below Delta(ISCO), so the real
        # inner disk edge is untouched.
        plunged = (r <= r_h) | (delta(r, a) <= 1.0e-2) | (r < 0.0)
        captured[idx[plunged]] = True
        done |= plunged

        crossed = (~plunged) & ((theta_prev - half_pi) * (theta - half_pi) < 0.0)
        if crossed.any():
            step = theta - theta_prev
            frac = np.where(step != 0.0, (half_pi - theta_prev) / step, 0.0)
            r_cross = r_prev + frac * (r - r_prev)
            hit = crossed & (r_cross >= disk.inner) & (r_cross <= disk.outer)
            # Across a pole reflection phi jumps by pi; interpolating over
            # the jump smears the azimuth, so take the endpoint there.
            dphi_step = phi - phi_prev
            az = np.where(np.abs(dphi_step) < half_pi,
                          phi_prev + frac * dphi_step, phi)
            radius[idx[hit]] = r_cross[hit]
            azimuth[idx[hit]] = az[hit]
            done |= hit

        done |= (r >= r_escape)

        keep = ~done
        if not keep.all():
            idx, y, bw, qw = idx[keep], y[:, keep], bw[keep], qw[keep]
            r, theta, phi = r[keep], theta[keep], phi[keep]
        r_prev, theta_prev, phi_prev = r, theta, phi

    return (radius.reshape(h, w), b.reshape(h, w), azimuth.reshape(h, w),
            captured.reshape(h, w))
