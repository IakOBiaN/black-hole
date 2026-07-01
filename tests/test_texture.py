import numpy as np

from src.texture import disk_pattern

R_IN = 6.0


def test_pattern_in_unit_range():
    r = np.linspace(6.1, 13.9, 500)
    az = np.linspace(-np.pi, np.pi, 500)
    p = disk_pattern(r, az, R_IN)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_pattern_seamless_across_azimuth_wrap():
    r = np.linspace(6.1, 13.9, 200)
    low = disk_pattern(r, np.full_like(r, -np.pi), R_IN)
    high = disk_pattern(r, np.full_like(r, np.pi), R_IN)
    assert np.allclose(low, high)


def test_pattern_is_deterministic():
    r = np.linspace(6.1, 13.9, 50)
    az = np.linspace(-1.0, 1.0, 50)
    assert np.array_equal(disk_pattern(r, az, R_IN), disk_pattern(r, az, R_IN))
