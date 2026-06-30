import numpy as np
import pytest

from src.metric import (
    schwarzschild_radius,
    photon_sphere_radius,
    critical_impact_parameter,
)
from src.geodesic import integrate_ray, deflection_angle

M = 1.0


def test_characteristic_radii():
    assert schwarzschild_radius(M) == pytest.approx(2.0 * M)
    assert photon_sphere_radius(M) == pytest.approx(3.0 * M)
    assert critical_impact_parameter(M) == pytest.approx(3.0 * np.sqrt(3.0) * M)


def test_capture_below_critical():
    b = 0.99 * critical_impact_parameter(M)
    assert integrate_ray(50.0, b, M, ingoing=True)["status"] == "captured"


def test_escape_above_critical():
    b = 1.01 * critical_impact_parameter(M)
    assert integrate_ray(50.0, b, M, ingoing=True)["status"] == "escaped"


@pytest.mark.parametrize("b", [1.0e4, 1.0e3, 1.0e2])
def test_weak_field_deflection(b):
    # Deflection series: 4M/b + (15pi/4)(M/b)^2 + (128/3)(M/b)^3 + ...
    x = M / b
    expected = 4.0 * x + 15.0 * np.pi / 4.0 * x**2 + 128.0 / 3.0 * x**3
    assert deflection_angle(b, M) == pytest.approx(expected, rel=1.0e-3)


def test_deflection_diverges_toward_critical():
    bc = critical_impact_parameter(M)
    near = deflection_angle(1.001 * bc, M)
    far = deflection_angle(2.0 * bc, M)
    assert near > far > 0.0


def test_captured_ray_has_infinite_deflection():
    assert deflection_angle(0.5 * critical_impact_parameter(M), M) == np.inf
