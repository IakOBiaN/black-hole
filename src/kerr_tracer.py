"""Vectorized ray tracer for a Kerr black hole.

Each camera ray is converted into the photon's conserved quantities (axial
angular momentum b and Carter constant q) using the FIDO orthonormal frame
of James et al. (2015), then the 3D null geodesic is marched with a
fixed-step RK4. The camera is assumed far enough that it is essentially a
FIDO (frame dragging is negligible at the camera radius). Rays are classified
as hitting the equatorial disk, the horizon, or escaping to infinity.
"""

import numpy as np

from .kerr import (delta, rho2, sigma, alpha, omega, varpi, horizon_radius,
                   _rk4)


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
    n_r = d @ r_hat
    n_theta = d @ theta_hat
    n_phi = d[:, 1]  # phi_hat = (0, 1, 0) at phi = 0

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
    active = np.ones(n, dtype=bool)
    radius = np.full(n, np.nan)
    azimuth = np.full(n, np.nan)
    captured = np.zeros(n, dtype=bool)
    r_h = horizon_radius(a) * 1.002
    half_pi = np.pi / 2.0

    r_prev, theta_prev, phi_prev = y[0].copy(), y[1].copy(), y[2].copy()
    for _ in range(max_steps):
        if not active.any():
            break
        y = np.where(active[None, :], _rk4(y, b, q, a, dzeta), y)
        r, theta, phi = y[0], y[1], y[2]

        crossed = active & ((theta_prev - half_pi) * (theta - half_pi) < 0.0)
        if crossed.any():
            step = theta - theta_prev
            frac = np.where(step != 0.0, (half_pi - theta_prev) / step, 0.0)
            r_cross = r_prev + frac * (r - r_prev)
            hit = crossed & (r_cross >= disk.inner) & (r_cross <= disk.outer)
            radius[hit] = r_cross[hit]
            azimuth[hit] = (phi_prev + frac * (phi - phi_prev))[hit]
            active[hit] = False

        fell_in = active & (r <= r_h)
        captured[fell_in] = True
        active &= ~fell_in
        active &= ~(r >= r_escape)

        r_prev = np.where(active, r, r_prev)
        theta_prev = np.where(active, theta, theta_prev)
        phi_prev = np.where(active, phi, phi_prev)

    return (radius.reshape(h, w), b.reshape(h, w), azimuth.reshape(h, w),
            captured.reshape(h, w))
