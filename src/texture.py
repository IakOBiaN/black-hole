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


def debris_extent(r_inner, r_outer, debris_ratio=0.22):
    """Outermost radius reached by the sparse debris beyond the nominal
    disk edge (the tracer must record crossings out to here)."""
    return r_inner * (r_outer / r_inner) ** (1.0 + debris_ratio)


def disk_material(r, azimuth, r_inner, r_outer, time=0.0,
                  shear_ages=(30.0, 110.0, 380.0),
                  gap_freq=3.5, gap_depth=0.45,
                  edge_ratio=0.17, debris_ratio=0.22,
                  envelope_scale=0.9, envelope_anchor=3.83, tau=4.0):
    """Emission weight and opacity of the artist-style accretion disk.

    Modeled on the Double Negative Interstellar disk (James et al. 2015,
    Figs. 14-15): a milky, marginally optically thick sheet of gas drawn
    out by the orbital shear, a few shallow darker lanes in its outer half,
    and ragged edges where the gas thins into feathery streaks.

    The gas field is built by differential-rotation advection: noise is
    sampled at the azimuth phi - Omega(r) * age with the Keplerian
    Omega ~ r^(-3/2), for several ages at once. An initially roundish blob
    is sheared into a trailing spiral streak whose winding grows with age
    and with Omega's gradient -- so the inner disk shows tightly wound
    threads while the outer edge keeps open, freshly curled swirls and
    clumps, like real turbulent gas rather than concentric circles. Past
    the noisy outer edge a rising survival threshold keeps only the densest
    gas, which tapers away by the debris extent.

    r, azimuth are arrays over disk samples. Returns (emission, alpha):
    emission is a dimensionless brightness weight (order unity in the bright
    inner region) and alpha the optical opacity in [0, 1] used to composite
    the lensed disk layers front to back.
    """
    r = np.asarray(r, dtype=float)
    # Normalized log-radius: 0 at the inner edge, 1 at the nominal outer
    # edge; the debris zone extends beyond 1.
    x = np.log(np.maximum(r, 1.0e-9) / r_inner) / np.log(r_outer / r_inner)

    # Keplerian angular velocity (geometrized units, M = 1).
    # Absolute floor at r = 3M: below it the pattern rotates rigidly. This
    # keeps the shear windings bounded when the disk reaches a near-extremal
    # ISCO (Omega -> huge there would wind the fibres into aliasing mush).
    omega_k = np.maximum(r, 3.0) ** -1.5

    # Time evolution: every gas element orbits at its own Keplerian rate, so
    # the whole pattern (fibres, billows, lanes, edge silhouette) is sampled
    # at the advected azimuth and keeps shearing as time runs.
    phi = np.asarray(azimuth, dtype=float) - omega_k * time
    phi_n = (phi / _TWO_PI) % 1.0

    # Layered shear advection, young (coarse, barely wound: visible swirls)
    # to old (fine, tightly wound threads). The youngest layer's domain is
    # additionally curled by a smooth warp that strengthens outward, so the
    # edge breaks into eddies instead of arcs.
    layer_cells = (10, 6, 3)
    layer_scale = (12.0, 34.0, 80.0)
    layer_gamma = (1.6, 1.8, 2.0)
    weight_out = (1.5, 0.5, 0.3)       # relative weight at the outer edge
    weight_in = (0.25, 0.6, 1.0)       # relative weight at the inner edge
    blend = _smoothstep(0.15, 0.95, x)

    thread = np.zeros_like(x)
    total = np.zeros_like(x)
    for k, age in enumerate(shear_ages):
        cells = layer_cells[k]
        pn = ((phi - omega_k * age) / _TWO_PI) % 1.0
        u = layer_scale[k] * x + 17.3 * k
        v = cells * pn
        if k == 0:
            curl = _smoothstep(0.25, 1.05, x)
            half = max(cells // 2, 1)
            vw = half * pn
            wu = _fbm(2.1 * x + 8.3, vw + 4.1, half, octaves=3) - 0.5
            wv = _fbm(2.6 * x + 2.9, vw + 9.7, half, octaves=3) - 0.5
            u = u + 5.0 * curl * wu
            v = v + 2.2 * curl * wv
        n = _fbm(u, v, cells, octaves=5) ** layer_gamma[k]
        w = weight_in[k] + (weight_out[k] - weight_in[k]) * blend
        thread += w * n
        total += w * 0.55 ** layer_gamma[k]   # rough fbm mean^gamma
    thread = np.clip(thread / np.maximum(total, 1.0e-9) * 0.55, 0.0, 1.3)
    thread = thread ** 1.7

    # A few shallow, slightly wavy darker lanes, deepening outward.
    lane = _fbm(gap_freq * x + 0.37, 2.0 * phi_n, 2, octaves=2)
    lane_w = gap_depth * _smoothstep(0.10, 0.60, x)
    lanes = 1.0 - lane_w * (1.0 - _smoothstep(0.42, 0.58, lane))

    # Ragged edges: the edge radius wanders with azimuth, and beyond it a
    # rising survival threshold thins the fibre field into feathery streaks
    # that continue coherently outward instead of banding off.
    edge_noise = _fbm(np.full_like(x, 0.29), 7.0 * phi_n, 7, octaves=5)
    edge_noise = _smoothstep(0.10, 0.90, edge_noise)
    x_edge = 1.0 - edge_ratio * edge_noise
    x_max = 1.0 + debris_ratio
    over = np.clip((x - x_edge) / np.maximum(x_max - x_edge, 1.0e-6),
                   0.0, 1.0)
    survive = _smoothstep(1.3 * over - 0.02, 1.3 * over + 0.20, thread)
    body_out = np.where(x <= x_edge, 1.0,
                        survive * np.exp(-1.6 * over))
    body_out = np.where(x < x_max, body_out, 0.0)
    # Feathered inner edge: the material thins toward the ISCO rather than
    # cutting off, softening the lensed dark gap against the photon ring.
    body_in = _smoothstep(-0.02, 0.06, x)

    # Fibres modulate a continuous milky sheet inside; toward the edge the
    # milky base drains away and large turbulent billows take over, so the
    # rim dissolves into amplified cloud chaos as in the article's disk.
    base = 0.30 - 0.14 * _smoothstep(0.5, 1.0, x)
    pn_b = ((phi - omega_k * shear_ages[0]) / _TWO_PI) % 1.0
    bil = _fbm(6.0 * x + 3.3, 4.0 * pn_b + 1.9, 4, octaves=4)
    chaos = 1.0 + _smoothstep(0.45, 1.0, x) * 1.9 * (bil ** 1.5 - 0.32)
    chaos = np.maximum(chaos, 0.15)
    body = (base + 0.85 * thread) * chaos * lanes * body_in * body_out

    # Radial brightness envelope: a power law in radius, like the steep
    # emissivity falloff of a real disk -- a blazing inner rim dropping by
    # orders of magnitude toward dim, translucent outskirts. This gradient,
    # not the texture, is what makes the lensed disk read as glowing gas
    # around the hole instead of a uniformly bright sheet.
    # Anchored at an absolute radius (the a=0.6 ISCO), not at r_inner: the
    # disk must not dim wholesale when a higher spin drags the inner edge
    # toward the horizon. Capped so the sub-anchor rim stays finite.
    envelope = np.minimum(
        np.maximum(r / envelope_anchor, 1.0e-9) ** -envelope_scale, 3.0)

    # The gas also thins outward: the outer disk becomes a translucent haze
    # rather than an opaque sheet.
    tau_eff = tau * (0.25 + 0.75 * envelope ** 0.4)

    emission = envelope * 2.0 * body
    alpha = 1.0 - np.exp(-tau_eff * body)
    return emission, np.clip(alpha, 0.0, 1.0)
