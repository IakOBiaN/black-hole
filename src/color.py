"""Blackbody color and tone mapping.

The CIE 1931 color matching functions are approximated by the multi-lobe
Gaussian fit of Wyman, Sloan and Shirley (JCGT 2013), so no data tables are
needed. A Planck spectrum is integrated against them to get XYZ, then
converted to linear sRGB.
"""

import numpy as np

_LAMBDA_NM = np.arange(380.0, 781.0, 5.0)
_HC_OVER_K = 1.43877688e-2  # m * K


def _lobe(lam, mu, s1, s2):
    t = (lam - mu) * np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * t * t)


def _cie_xyz_bars(lam):
    x = (0.362 * _lobe(lam, 442.0, 0.0624, 0.0374)
         + 1.056 * _lobe(lam, 599.8, 0.0264, 0.0323)
         - 0.065 * _lobe(lam, 501.1, 0.0490, 0.0382))
    y = (0.821 * _lobe(lam, 568.8, 0.0213, 0.0247)
         + 0.286 * _lobe(lam, 530.9, 0.0613, 0.0322))
    z = (1.217 * _lobe(lam, 437.0, 0.0845, 0.0278)
         + 0.681 * _lobe(lam, 459.0, 0.0385, 0.0725))
    return x, y, z


_XBAR, _YBAR, _ZBAR = _cie_xyz_bars(_LAMBDA_NM)

_XYZ_TO_RGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])


def _planck(lam_nm, temperature):
    lam_m = lam_nm * 1.0e-9
    with np.errstate(over="ignore"):
        return 1.0 / (lam_m ** 5 * (np.exp(_HC_OVER_K / (lam_m * temperature)) - 1.0))


def blackbody_color(temperature):
    """Linear sRGB chromaticity of a blackbody, normalized so the brightest
    channel is 1 (hue only, brightness applied separately)."""
    spectrum = _planck(_LAMBDA_NM, temperature)
    xyz = np.array([np.sum(spectrum * _XBAR),
                    np.sum(spectrum * _YBAR),
                    np.sum(spectrum * _ZBAR)])
    rgb = _XYZ_TO_RGB @ xyz
    rgb = np.clip(rgb, 0.0, None)
    peak = rgb.max()
    return rgb / peak if peak > 0.0 else rgb


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _luminance(c):
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def tonemap(linear, mode, exposure=None, saturation=None, highlight_desat=None,
            desat_start=0.55):
    """Convert a linear-light HDR image to 8-bit sRGB. The Reinhard curve is
    applied to luminance and the color ratios are preserved (so mid-tones keep
    their hue), while the brightest regions are pushed toward white for a
    film-like white-hot highlight response."""
    if exposure is None:
        exposure = 1.8 if mode == "beautiful" else 1.2
    if saturation is None:
        # The film's disk is a pale pinkish cream: keep chroma well below
        # the raw 4500 K blackbody saturation.
        saturation = 0.8 if mode == "beautiful" else 1.0
    if highlight_desat is None:
        highlight_desat = 0.9 if mode == "beautiful" else 0.5

    c = linear * exposure
    lum = _luminance(c)
    lum_safe = np.where(lum > 0.0, lum, 1.0)
    toned = lum / (1.0 + lum)

    graded = c * (toned / lum_safe)[..., None]
    gray = toned[..., None]
    graded = gray + saturation * (graded - gray)

    weight = np.clip((toned - desat_start) / (1.0 - desat_start), 0.0, 1.0) ** 2
    graded = graded + (highlight_desat * weight)[..., None] * (gray - graded)
    return (linear_to_srgb(graded) * 255.0 + 0.5).astype(np.uint8)
