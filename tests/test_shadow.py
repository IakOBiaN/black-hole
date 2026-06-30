import numpy as np

from src.metric import lapse_squared, critical_impact_parameter
from src.renderer import ray_status

M = 1.0


def _shadow_edge_angle(r0):
    """Predicted apparent angular radius of the shadow seen from radius r0:
    the ray angle alpha from the line of sight (toward the hole) at which the
    impact parameter equals the critical one."""
    return np.arcsin(critical_impact_parameter(M) * np.sqrt(lapse_squared(r0, M)) / r0)


def test_shadow_boundary_matches_critical_impact_parameter():
    r0 = 30.0
    edge = _shadow_edge_angle(r0)

    # alpha is the angle from the line of sight (the -radial direction), so
    # the angle from the outward radial direction is pi - alpha.
    inside = np.pi - (edge - 1.0e-3)
    outside = np.pi - (edge + 1.0e-3)

    assert ray_status(r0, np.cos(inside), np.sin(inside), M) == "captured"
    assert ray_status(r0, np.cos(outside), np.sin(outside), M) == "escaped"


def test_rays_pointing_away_escape():
    r0 = 30.0
    # Looking radially outward: never captured.
    assert ray_status(r0, 1.0, 0.0, M) == "escaped"
