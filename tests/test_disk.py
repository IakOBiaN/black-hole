import numpy as np

from src.disk import trace, Disk

DISK = Disk(2.0, 20.0)


def test_crossing_matches_flat_space_when_mass_negligible():
    # With negligible mass the geodesic is a straight line, so the equatorial
    # crossing must match the Euclidean intersection.
    mass = 1.0e-6
    inc = np.radians(20.0)
    pos = 30.0 * np.array([np.cos(inc), 0.0, np.sin(inc)])
    target = np.array([10.0, 3.0, 0.0])
    direction = target - pos
    direction /= np.linalg.norm(direction)

    kind, r, _ = trace(pos, direction, mass, DISK)
    assert kind == "disk"
    assert np.isclose(r, np.linalg.norm(target), rtol=1.0e-4)


def test_crossing_outside_annulus_is_background():
    mass = 1.0e-6
    inc = np.radians(20.0)
    pos = 30.0 * np.array([np.cos(inc), 0.0, np.sin(inc)])
    target = np.array([25.0, 0.0, 0.0])
    direction = target - pos
    direction /= np.linalg.norm(direction)

    assert trace(pos, direction, mass, DISK)[0] == "background"


def test_central_ray_hits_horizon():
    pos = np.array([30.0, 0.0, 0.0])
    assert trace(pos, np.array([-1.0, 0.0, 0.0]), 1.0, DISK)[0] == "horizon"


def test_outward_ray_is_background():
    pos = np.array([30.0, 0.0, 0.0])
    assert trace(pos, np.array([1.0, 0.0, 0.0]), 1.0, DISK)[0] == "background"
