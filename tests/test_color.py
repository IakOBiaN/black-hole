import numpy as np

from src.color import blackbody_color, linear_to_srgb


def test_cool_blackbody_is_reddish():
    rgb = blackbody_color(3000.0)
    assert rgb[0] > rgb[1] > rgb[2]


def test_hot_blackbody_is_bluish():
    rgb = blackbody_color(20000.0)
    assert rgb[2] >= rgb[0]


def test_color_is_normalized():
    for t in (2000.0, 6500.0, 15000.0):
        rgb = blackbody_color(t)
        assert np.isclose(rgb.max(), 1.0)
        assert np.all(rgb >= 0.0)


def test_srgb_endpoints():
    assert np.isclose(linear_to_srgb(np.array(0.0)), 0.0)
    assert np.isclose(linear_to_srgb(np.array(1.0)), 1.0)
