"""Procedural gas texture for the disk.

Fractal value noise sampled in (log r, phi) with an angular period, so it is
seamless across the phi wrap. The angular coordinate is sheared by log(r)
(twist), which winds the streaks into spirals like differentially rotating
gas. Radial frequency is higher than angular, so features stretch along the
flow.
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


def disk_pattern(r, azimuth, r_inner, base_angular=10, octaves=6,
                 radial_freq=4.0, twist=6.0, lacunarity=2.0, gain=0.6):
    lr = np.log(r / r_inner)
    value = np.zeros_like(r, dtype=float)
    amp, total, freq, period = 1.0, 0.0, 1.0, base_angular
    for _ in range(octaves):
        x = radial_freq * freq * lr
        y = period * (azimuth + twist * lr) / _TWO_PI
        value += amp * _value_noise(x, y, period)
        total += amp
        amp *= gain
        freq *= lacunarity
        period = int(round(period * lacunarity))
    return value / total
