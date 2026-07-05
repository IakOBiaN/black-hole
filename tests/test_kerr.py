import numpy as np

from src.kerr import (horizon_radius, photon_orbit_radii, photon_orbit_b,
                      equatorial_ray_status, rhs, rhs_numerical)


def test_analytic_rhs_matches_numerical():
    rng = np.random.default_rng(0)
    for a in (0.0, 0.9, 0.998):
        for _ in range(300):
            y = np.array([rng.uniform(1.3, 30.0), rng.uniform(0.3, np.pi - 0.3),
                          0.0, rng.uniform(-3, 3), rng.uniform(-3, 3)])
            b, q = rng.uniform(-8, 8), rng.uniform(0, 20)
            analytic = rhs(y, b, q, a)
            numerical = rhs_numerical(y, b, q, a)
            assert np.allclose(analytic, numerical, rtol=1.0e-3, atol=1.0e-6)


def test_horizon_radius():
    assert horizon_radius(0.0) == 2.0
    assert np.isclose(horizon_radius(0.9), 1.0 + np.sqrt(1.0 - 0.81))
    assert horizon_radius(0.998) < 1.1


def test_photon_orbits_reduce_to_photon_sphere_at_zero_spin():
    r1, r2 = photon_orbit_radii(0.0)
    assert np.isclose(r1, 3.0) and np.isclose(r2, 3.0)


def test_frame_dragging_splits_photon_orbits():
    r1, r2 = photon_orbit_radii(0.9)
    assert r1 < 3.0 < r2  # prograde orbit closer in, retrograde farther out


def test_schwarzschild_capture_threshold_at_zero_spin():
    bc = 3.0 * np.sqrt(3.0)
    assert equatorial_ray_status(bc - 0.2, 0.0) == "captured"
    assert equatorial_ray_status(bc + 0.2, 0.0) == "escaped"


def test_frame_dragging_capture_asymmetry():
    a = 0.9
    r1, r2 = photon_orbit_radii(a)
    b_prograde = photon_orbit_b(r1, a)
    b_retrograde = photon_orbit_b(r2, a)

    # Co-rotating photons are captured with a smaller impact parameter.
    assert abs(b_prograde) < abs(b_retrograde)

    assert equatorial_ray_status(b_prograde - 0.2, a) == "captured"
    assert equatorial_ray_status(b_prograde + 0.2, a) == "escaped"
    assert equatorial_ray_status(b_retrograde + 0.2, a) == "captured"
    assert equatorial_ray_status(b_retrograde - 0.2, a) == "escaped"
