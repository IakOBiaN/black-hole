import numpy as np

from src.texture import disk_pattern, disk_material, debris_extent

R_IN = 6.0
INNER, OUTER = 9.26, 18.7


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


def test_material_ranges():
    rng = np.random.default_rng(7)
    r = rng.uniform(INNER * 0.98, debris_extent(INNER, OUTER), 4000)
    az = rng.uniform(-np.pi, np.pi, 4000)
    emission, alpha = disk_material(r, az, INNER, OUTER)
    assert np.all(emission >= 0.0)
    assert np.all((alpha >= 0.0) & (alpha <= 1.0))


def test_material_seamless_across_azimuth_wrap():
    r = np.linspace(INNER, OUTER, 300)
    em_lo, al_lo = disk_material(r, np.full_like(r, -np.pi), INNER, OUTER)
    em_hi, al_hi = disk_material(r, np.full_like(r, np.pi), INNER, OUTER)
    assert np.allclose(em_lo, em_hi)
    assert np.allclose(al_lo, al_hi)


def test_material_vanishes_outside_debris_extent():
    r_far = np.linspace(debris_extent(INNER, OUTER) * 1.001, OUTER * 2.0, 200)
    emission, alpha = disk_material(r_far, np.zeros_like(r_far), INNER, OUTER)
    assert np.allclose(emission, 0.0)
    assert np.allclose(alpha, 0.0)


def test_material_vanishes_below_the_inner_feather():
    # The inner edge feathers in from x = -0.02 (in normalized log-radius),
    # i.e. from (OUTER/INNER)**-0.02 * INNER; below that there is nothing.
    feather_start = INNER * (OUTER / INNER) ** -0.02
    r_in = np.linspace(INNER * 0.8, feather_start * 0.999, 100)
    emission, _ = disk_material(r_in, np.zeros_like(r_in), INNER, OUTER)
    assert np.allclose(emission, 0.0)


def test_material_time_is_differential_rotation():
    # Advancing time by t must equal rotating each ring by its own
    # Keplerian angle Omega(r) * t: the gas orbits differentially.
    rng = np.random.default_rng(3)
    r = rng.uniform(INNER * 1.1, OUTER * 0.95, 500)
    az = rng.uniform(-np.pi, np.pi, 500)
    t = 137.0
    em_t, al_t = disk_material(r, az, INNER, OUTER, time=t)
    em_rot, al_rot = disk_material(r, az - r ** -1.5 * t, INNER, OUTER)
    assert np.allclose(em_t, em_rot)
    assert np.allclose(al_t, al_rot)


def test_material_outer_edge_is_ragged():
    # The nominal-edge neighbourhood must mix opaque and clear samples as
    # the edge radius wanders with azimuth.
    az = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
    r = np.full_like(az, OUTER * 0.97)
    _, alpha = disk_material(r, az, INNER, OUTER)
    assert alpha.max() > 0.3
    assert alpha.min() < 0.05
