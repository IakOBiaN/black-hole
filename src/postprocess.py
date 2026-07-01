"""Post-processing on the linear HDR image, before tone mapping."""

import numpy as np
from scipy.ndimage import gaussian_filter


def add_bloom(linear, strength, scales=(0.004, 0.013, 0.033),
              weights=(1.0, 0.6, 0.3)):
    """Add a multi-scale bloom: bright regions bleed a soft halo. Works in
    linear light so the glow adds physically. scales are fractions of the
    image height."""
    if strength <= 0.0:
        return linear

    height = linear.shape[0]
    accum = np.zeros_like(linear)
    total = float(sum(weights))
    for scale, weight in zip(scales, weights):
        sigma = max(scale * height, 0.8)
        for c in range(3):
            accum[..., c] += weight * gaussian_filter(
                linear[..., c], sigma, mode="constant")

    return linear + strength * accum / total
