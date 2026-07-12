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


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def debris_extent(r_inner, r_outer, debris_ratio=0.16):
    """Outermost radius reached by the sparse debris beyond the nominal
    disk edge (the tracer must record crossings out to here)."""
    return r_inner * (r_outer / r_inner) ** (1.0 + debris_ratio)


def disk_material(r, azimuth, r_inner, r_outer,
                  n_filaments=42.0, streak_cells=6, twist=1.0,
                  gap_freq=6.0, gap_depth=0.8,
                  edge_ratio=0.17, debris_ratio=0.16,
                  envelope_scale=1.2, tau=4.0):
    """Emission weight and opacity of the artist-style accretion disk.

    Modeled on the Double Negative Interstellar disk (James et al. 2015,
    Figs. 14-15): a physically thin, marginally optically thick sheet with
    fine filaments stretched along the orbital flow, darker ring gaps, a
    ragged noise-modulated outer edge and sparse debris flung beyond it.

    r, azimuth are arrays over disk samples. Returns (emission, alpha):
    emission is a dimensionless brightness weight (order unity in the bright
    inner region) and alpha the optical opacity in [0, 1] used to composite
    the lensed disk layers front to back.
    """
    r = np.asarray(r, dtype=float)
    # Normalized log-radius: 0 at the inner edge, 1 at the nominal outer
    # edge; the debris zone extends beyond 1.
    x = np.log(np.maximum(r, 1.0e-9) / r_inner) / np.log(r_outer / r_inner)
    phi_n = (np.asarray(azimuth) / _TWO_PI) % 1.0

    # Fine filaments: high radial frequency, few azimuthal cells, so noise
    # features become thin arcs stretched along the orbital flow, sheared
    # into trailing spirals by the differential-rotation twist.
    u = n_filaments * x
    v = streak_cells * (phi_n + twist * x)
    fil = _fbm(u, v, streak_cells, octaves=5)
    fil = fil * fil * (3.0 - 2.0 * fil)          # sharpen toward threads
    fine = _fbm(2.7 * u + 11.3, 2.0 * v + 4.7, 2 * streak_cells, octaves=4)
    fil = np.clip(0.62 * fil + 0.58 * fine * fine, 0.0, 1.2) ** 1.35

    # Slow radial modulation carving darker, slightly wavy ring gaps.
    gp = _fbm(gap_freq * x + 0.37, 2.0 * phi_n, 2, octaves=3)
    gaps = (1.0 - gap_depth) + gap_depth * _smoothstep(0.40, 0.60, gp)

    # Ragged outer edge: its radius wanders with azimuth.
    edge_noise = _fbm(np.full_like(x, 0.29), 5.0 * phi_n, 5, octaves=4)
    edge_noise = _smoothstep(0.15, 0.85, edge_noise)
    x_edge = 1.0 - edge_ratio * edge_noise
    body_out = 1.0 - _smoothstep(x_edge - 0.05, x_edge + 0.015, x)
    body_in = _smoothstep(0.0, 0.02, x)

    # Filaments modulate a continuous sheet rather than slicing it to
    # ribbons: the disk body stays mostly optically thick with bright
    # threads on top, as in the article's artist disk.
    body = (0.32 + 0.68 * fil) * gaps * body_in * body_out

    # Sparse debris flung beyond the ragged edge: fine noise stretched into
    # long trailing wisps by a strong differential-rotation shear, fading
    # with distance and gone entirely past the debris extent. Seen nearly
    # edge-on these read as the chaotic gas swirls at the far left and right
    # of the disk in the article's figures.
    spec = _fbm(1.3 * n_filaments * x + 31.7,
                2 * streak_cells * (phi_n + 1.4 * x) + 17.1,
                2 * streak_cells, octaves=4)
    spec = np.clip((spec - 0.40) / 0.60, 0.0, 1.0) ** 1.4
    beyond = np.clip((x - x_edge) / 0.10, 0.0, None)
    debris = spec * np.exp(-beyond) * _smoothstep(-0.01, 0.02, x - x_edge)
    debris = np.where(x < 1.0 + debris_ratio, debris, 0.0)

    # Radial brightness envelope: hot bright inner region fading outward.
    envelope = np.exp(-envelope_scale * np.clip(x, 0.0, None))

    emission = envelope * (2.2 * body + 2.4 * debris)
    alpha = 1.0 - np.exp(-(tau * body + 1.2 * debris))
    return emission, np.clip(alpha, 0.0, 1.0)
