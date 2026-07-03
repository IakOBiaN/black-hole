"""Procedural gas texture for the disk.

The texture is built in (radius, azimuth) space so it follows the orbital
flow: features are stretched along the azimuth (the direction of motion) and
sheared by a radius-dependent twist into trailing spirals, like differentially
rotating gas. Domain-warped fractal noise adds turbulent, fibrous detail on
top. The angular coordinate uses periodic value noise, so the texture is
seamless across the phi wrap.
"""

import numpy as np

_TWO_PI = 2.0 * np.pi


def _hash01(ix, iy):
    n = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263) & 0x7fffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0x7fffffff
    n = (n ^ (n >> 16)) & 0x7fffffff
    return n / 0x7fffffff


def _value_noise(x, y, period):
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    y0p, y1p = y0 % period, (y0 + 1) % period
    v00 = _hash01(x0, y0p)
    v10 = _hash01(x0 + 1, y0p)
    v01 = _hash01(x0, y1p)
    v11 = _hash01(x0 + 1, y1p)
    a = v00 + ux * (v10 - v00)
    b = v01 + ux * (v11 - v01)
    return a + uy * (b - a)


def _fbm(x, y, period, octaves=6, gain=0.5):
    value, amp, freq, per, total = 0.0, 0.5, 1.0, period, 0.0
    for _ in range(octaves):
        value = value + amp * _value_noise(x * freq, y * freq, per)
        total += amp
        amp *= gain
        freq *= 2.0
        per *= 2
    return value / total


def disk_pattern(r, azimuth, r_inner, radial_freq=11.0, angular_cells=5,
                 twist=1.8, warp_radial=3.0, warp_angular=1.1, octaves=7):
    """Cloud density in [0, 1] over the disk, flowing along the orbit."""
    lr = np.log(r / r_inner)
    u = radial_freq * lr
    v = angular_cells * (azimuth + twist * lr) / _TWO_PI

    wu = _fbm(u, v, angular_cells, octaves)
    wv = _fbm(u + 3.1, v + 1.7, angular_cells, octaves)
    u2 = u + warp_radial * (wu - 0.5)
    v2 = v + warp_angular * (wv - 0.5)
    return _fbm(u2, v2, angular_cells, octaves)
