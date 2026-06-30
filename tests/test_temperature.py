import numpy as np

from src.temperature import disk_temperature

R_IN = 6.0
T_PEAK = 10000.0


def test_temperature_vanishes_at_inner_edge():
    assert disk_temperature(R_IN, R_IN, T_PEAK) == 0.0


def test_peak_value_and_location():
    r = np.linspace(R_IN * 1.0001, 40.0, 20000)
    temp = disk_temperature(r, R_IN, T_PEAK)
    assert np.isclose(temp.max(), T_PEAK, rtol=1.0e-3)
    assert np.isclose(r[np.argmax(temp)], 49.0 / 36.0 * R_IN, rtol=1.0e-2)


def test_falls_off_outward():
    r = np.array([10.0, 20.0, 40.0])
    temp = disk_temperature(r, R_IN, T_PEAK)
    assert temp[0] > temp[1] > temp[2]
