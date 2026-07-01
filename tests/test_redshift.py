import numpy as np

from src.disk import redshift_factor, orbital_angular_velocity

M = 1.0


def test_pure_gravitational_redshift_when_no_doppler():
    # bz = 0 removes the orbital Doppler term, leaving sqrt(1 - 3M/r) < 1.
    r = 10.0
    assert np.isclose(redshift_factor(r, 0.0, M), np.sqrt(1.0 - 3.0 * M / r))
    assert redshift_factor(r, 0.0, M) < 1.0


def test_isco_gravitational_factor():
    assert np.isclose(redshift_factor(6.0, 0.0, M), np.sqrt(0.5))


def test_approaching_side_blueshifted_relative_to_receding():
    r = 10.0
    omega = orbital_angular_velocity(r, M)
    bz = 0.5 / omega  # gives a moderate Doppler term
    approaching = redshift_factor(r, bz, M)
    receding = redshift_factor(r, -bz, M)
    assert approaching > receding
    assert approaching > redshift_factor(r, 0.0, M) > receding


def test_doppler_strength_zero_removes_asymmetry():
    r = 10.0
    bz = 3.0
    assert np.isclose(redshift_factor(r, bz, M, doppler_strength=0.0),
                      redshift_factor(r, -bz, M, doppler_strength=0.0))
