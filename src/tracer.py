"""Vectorized batch ray tracer.

Integrates every camera ray at once with a fixed-step RK4 in the orbital
angle phi, instead of calling an adaptive solver per ray. All rays share the
camera radius r0, so u0 = 1/r0, e1 = pos/|pos| and the ODE are common; only
direction-dependent quantities vary per ray. Disk crossings are detected from
the sign change of cos(phi) e1.z + sin(phi) e2.z between steps.
"""

import numpy as np

from .metric import schwarzschild_radius, lapse_squared


def _rk4_step(u, du, dphi, mass):
    def acc(uu):
        return 3.0 * mass * uu * uu - uu
    k1u, k1d = du, acc(u)
    k2u, k2d = du + 0.5 * dphi * k1d, acc(u + 0.5 * dphi * k1u)
    k3u, k3d = du + 0.5 * dphi * k2d, acc(u + 0.5 * dphi * k2u)
    k4u, k4d = du + dphi * k3d, acc(u + dphi * k3u)
    u_next = u + dphi / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    du_next = du + dphi / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
    return u_next, du_next


def trace_batch(camera, mass, disk, dphi=0.02, max_steps=1500,
                r_escape_factor=1.1):
    """Trace all rays. Returns (radius, bz, azimuth) buffers of shape (H, W):
    radius is the disk hit radius (NaN where no hit), bz the photon z angular
    momentum per energy, azimuth the disk-plane angle of the hit (NaN if none).
    """
    pos = camera.position
    r0 = float(np.linalg.norm(pos))
    e1 = pos / r0
    dirs = camera.ray_directions()
    h, w, _ = dirs.shape
    d = dirs.reshape(-1, 3)

    rs = schwarzschild_radius(mass)
    lapse0 = lapse_squared(r0, mass)

    cos_psi = d @ e1
    tangential = d - cos_psi[:, None] * e1[None, :]
    sin_psi = np.linalg.norm(tangential, axis=1)
    radial = sin_psi < 1.0e-12
    safe_sin = np.where(radial, 1.0, sin_psi)
    e2 = tangential / safe_sin[:, None]

    u0 = 1.0 / r0
    b = r0 * sin_psi / np.sqrt(lapse0)
    disc = 1.0 / np.where(radial, 1.0, b) ** 2 - u0 * u0 * (1.0 - rs * u0)
    du0 = np.where(cos_psi < 0.0, 1.0, -1.0) * np.sqrt(np.clip(disc, 0.0, None))
    bz = np.cross(np.broadcast_to(pos, d.shape), d)[:, 2] / np.sqrt(lapse0)

    cz1, cz2 = e1[2], e2[:, 2]
    u_horizon = 1.0 / rs
    u_escape = 1.0 / (r_escape_factor * r0)

    n = d.shape[0]
    u = np.full(n, u0)
    du = du0.copy()
    active = ~radial
    radius = np.full(n, np.nan)
    azimuth = np.full(n, np.nan)

    phi = 0.0
    f_prev = np.full(n, cz1)
    for _ in range(max_steps):
        if not active.any():
            break
        phi_next = phi + dphi
        u_next, du_next = _rk4_step(u, du, dphi, mass)
        f_next = np.cos(phi_next) * cz1 + np.sin(phi_next) * cz2

        crossed = active & (f_prev * f_next < 0.0)
        if crossed.any():
            frac = f_prev / (f_prev - f_next)
            r_cross = 1.0 / (u + frac * (u_next - u))
            phi_cross = phi + frac * dphi
            hit = crossed & (r_cross >= disk.inner) & (r_cross <= disk.outer)
            if hit.any():
                cx = np.cos(phi_cross) * e1[0] + np.sin(phi_cross) * e2[:, 0]
                cy = np.cos(phi_cross) * e1[1] + np.sin(phi_cross) * e2[:, 1]
                radius[hit] = r_cross[hit]
                azimuth[hit] = np.arctan2(r_cross * cy, r_cross * cx)[hit]
                active[hit] = False

        active[active & (u_next >= u_horizon)] = False
        active[active & (u_next <= u_escape)] = False

        u = np.where(active, u_next, u)
        du = np.where(active, du_next, du)
        f_prev = f_next
        phi = phi_next

    return (radius.reshape(h, w), bz.reshape(h, w), azimuth.reshape(h, w))
