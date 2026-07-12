"""Tests of the article's three frequency-shift treatments (Fig. 15a/b/c)."""

import numpy as np

from src.renderer import _blackbody_hue_lut, _shifted_disk_color, _LUMA_FILM

T = 4500.0
HUE_OF = _blackbody_hue_lut(6.0 * T)


def test_none_mode_ignores_g():
    g = np.array([0.5, 1.0, 1.5])
    color = _shifted_disk_color(T, g, T, "none", HUE_OF)
    assert np.allclose(color[0], color[1])
    assert np.allclose(color[1], color[2])


def test_hue_mode_shifts_colour_but_preserves_film_luma():
    g = np.array([0.6, 1.0, 1.5])
    color = _shifted_disk_color(T, g, T, "hue", HUE_OF)
    luma = color @ _LUMA_FILM
    # Same perceived brightness on both sides of the disk...
    assert np.allclose(luma, luma[1], rtol=1.0e-6)
    # ...but the blueshifted sample is bluer and the redshifted one redder.
    blue_ratio = color[:, 2] / color[:, 0]
    assert blue_ratio[2] > blue_ratio[1] > blue_ratio[0]


def test_full_mode_scales_intensity_as_g_fourth():
    # Liouville: I_nu ~ nu^3, so the frequency-integrated intensity of the
    # shifted blackbody scales as g^4 -- the brightness factor applied on
    # top of the observed-temperature hue must be (g T / T_ref)^4.
    g = np.array([0.5, 1.0, 1.4])
    color = _shifted_disk_color(T, g, T, "full", HUE_OF)
    expected = HUE_OF(g * T) * (g[:, None]) ** 4
    assert np.allclose(color, expected, rtol=1.0e-9)

    # Net effect: the approaching side far outshines the receding side.
    luma = color @ _LUMA_FILM
    assert luma[2] > luma[1] > luma[0]
    assert luma[0] / luma[1] < 0.1
