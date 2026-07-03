"""Volumetric accretion disk rendering.

The disk is a thin slab of emitting and absorbing gas. Each ray is marched
through curved spacetime; inside the slab we accumulate emission with
absorption, front-to-back:

    color += transmittance * emission * density * ds
    transmittance *= exp(-kappa * density * ds)

Density is the flow-aligned turbulent texture times a Gaussian vertical
profile, so the gas is semi-transparent and, viewed near edge-on, the long
line-of-sight chord integrates through varied clouds and gains depth. The
result is a linear HDR image; bloom and tone mapping are applied afterwards.
"""

import numpy as np

from .metric import schwarzschild_radius, lapse_squared
from .tracer import _rk4_step
from .disk import redshift_factor
from .temperature import disk_temperature
from .texture import disk_pattern
from .color import blackbody_color


def _blackbody_lut(t_peak, n=512):
    samples = np.linspace(1000.0, 4.0 * t_peak, n)
    lut = np.array([blackbody_color(t) for t in samples])
    return samples, lut


def render_linear(camera, mass, disk, t_peak=5000.0, mode="beautiful",
                  thickness=0.25, density_scale=8.0, kappa=25.0,
                  emission_scale=10.0, density_power=2.5, doppler_strength=None,
                  z_shear=0.6, texture_kwargs=None, dphi=0.006, max_steps=2200,
                  r_escape_factor=1.1):
    """March all rays through the disk volume and return a linear HDR image."""
    if doppler_strength is None:
        doppler_strength = 0.6 if mode == "beautiful" else 1.0
    if texture_kwargs is None:
        texture_kwargs = dict(octaves=5)

    pos = camera.position
    r0 = float(np.linalg.norm(pos))
    e1 = pos / r0
    dirs = camera.ray_directions()
    h, w, _ = dirs.shape
    d = dirs.reshape(-1, 3)
    n = d.shape[0]

    rs = schwarzschild_radius(mass)
    lapse0 = lapse_squared(r0, mass)
    cos_psi = d @ e1
    tangential = d - cos_psi[:, None] * e1[None, :]
    sin_psi = np.linalg.norm(tangential, axis=1)
    radial = sin_psi < 1.0e-12
    e2 = tangential / np.where(radial, 1.0, sin_psi)[:, None]

    u0 = 1.0 / r0
    b = r0 * sin_psi / np.sqrt(lapse0)
    disc = 1.0 / np.where(radial, 1.0, b) ** 2 - u0 * u0 * (1.0 - rs * u0)
    du0 = np.where(cos_psi < 0.0, 1.0, -1.0) * np.sqrt(np.clip(disc, 0.0, None))
    bz = np.cross(np.broadcast_to(pos, d.shape), d)[:, 2] / np.sqrt(lapse0)

    u = np.full(n, u0)
    du = du0.copy()
    active = ~radial
    color = np.zeros((n, 3))
    trans = np.ones(n)

    samples, lut = _blackbody_lut(t_peak)
    z_cut = 4.0 * thickness
    u_horizon = 1.0 / rs
    u_escape = 1.0 / (r_escape_factor * r0)

    x_prev = np.broadcast_to(r0 * e1, (n, 3)).copy()
    phi = 0.0
    for _ in range(max_steps):
        if not active.any():
            break
        phi += dphi
        u_next, du_next = _rk4_step(u, du, dphi, mass)
        u = np.where(active, u_next, u)
        du = np.where(active, du_next, du)

        r = 1.0 / u
        cph, sph = np.cos(phi), np.sin(phi)
        x = r * (cph * e1[0] + sph * e2[:, 0])
        y = r * (cph * e1[1] + sph * e2[:, 1])
        z = r * (cph * e1[2] + sph * e2[:, 2])
        ds = np.sqrt((x - x_prev[:, 0]) ** 2 + (y - x_prev[:, 1]) ** 2
                     + (z - x_prev[:, 2]) ** 2)
        r_cyl = np.sqrt(x * x + y * y)

        inside = (active & (r_cyl >= disk.inner) & (r_cyl <= disk.outer)
                  & (np.abs(z) < z_cut))
        if inside.any():
            rc = r_cyl[inside]
            zc = z[inside]
            az = np.arctan2(y[inside], x[inside]) + z_shear * zc
            vert = np.exp(-0.5 * (zc / thickness) ** 2)
            pattern = disk_pattern(rc, az, disk.inner, **texture_kwargs)
            density = density_scale * vert * pattern ** density_power

            g = redshift_factor(rc, bz[inside], mass, doppler_strength)
            temp = g * disk_temperature(rc, disk.inner, t_peak)
            clamped = np.clip(temp, samples[0], samples[-1])
            hue = np.stack([np.interp(clamped, samples, lut[:, c])
                            for c in range(3)], axis=1)
            emission = hue * ((temp / t_peak) ** 4 * emission_scale)[:, None]

            seg = density * ds[inside]
            color[inside] += (trans[inside] * seg)[:, None] * emission
            trans[inside] *= np.exp(-kappa * seg)

        active &= u < u_horizon
        active &= u > u_escape
        active &= trans > 0.01
        x_prev = np.stack([x, y, z], axis=1)

    return color.reshape(h, w, 3)
