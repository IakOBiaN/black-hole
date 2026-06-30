"""Thin-disk effective temperature profile (Shakura-Sunyaev shape).

With a zero-torque inner boundary the effective temperature follows

    T(r) ~ (r_in/r)^(3/4) * (1 - sqrt(r_in/r))^(1/4),

which vanishes at the inner edge, peaks at r = (49/36) r_in, then falls off
as r^(-3/4). The profile is normalized so its maximum equals t_peak.
"""

import numpy as np

_PEAK_SHAPE = (36.0 / 49.0) ** 0.75 * (1.0 / 7.0) ** 0.25


def disk_temperature(r, r_inner, t_peak):
    r = np.asarray(r, dtype=float)
    x = r_inner / r
    shape = np.where(r > r_inner,
                     x ** 0.75 * np.clip(1.0 - np.sqrt(x), 0.0, None) ** 0.25,
                     0.0)
    return t_peak * shape / _PEAK_SHAPE
