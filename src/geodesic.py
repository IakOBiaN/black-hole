"""Null geodesic integration around a Schwarzschild black hole.

Each photon path lies in a plane through the center, so we work in polar
coordinates (r, phi) within that plane and integrate the light-bending
equation for u = 1/r:

    d^2u/dphi^2 + u = 3 M u^2

The impact parameter b = L/E fixes the trajectory shape through the first
integral (du/dphi)^2 = 1/b^2 - u^2 (1 - 2 M u).
"""

import numpy as np
from scipy.integrate import solve_ivp

from .metric import schwarzschild_radius, critical_impact_parameter


def _rhs(phi, y, mass):
    u, dudphi = y
    return [dudphi, 3.0 * mass * u * u - u]


def _initial_slope(r0, b, mass, ingoing):
    rs = schwarzschild_radius(mass)
    u0 = 1.0 / r0
    disc = 1.0 / (b * b) - u0 * u0 * (1.0 - rs * u0)
    disc = max(disc, 0.0)
    return (1.0 if ingoing else -1.0) * np.sqrt(disc)


def integrate_ray(r0, b, mass, ingoing=True, r_escape=1.0e7,
                  phi_span=50.0, rtol=1.0e-10, atol=1.0e-12):
    """Trace a null geodesic from radius r0 with impact parameter b.

    Returns a dict with the trajectory (phi, r) and a status:
    'captured' (crossed the horizon), 'escaped' (reached r_escape), or
    'incomplete' (phi_span exhausted)."""
    rs = schwarzschild_radius(mass)
    u0 = 1.0 / r0
    du0 = _initial_slope(r0, b, mass, ingoing)

    u_horizon = 1.0 / rs
    u_escape = 1.0 / r_escape

    def hit_horizon(phi, y, m):
        return y[0] - u_horizon
    hit_horizon.terminal = True
    hit_horizon.direction = 1.0

    def reach_escape(phi, y, m):
        return y[0] - u_escape
    reach_escape.terminal = True
    reach_escape.direction = -1.0

    sol = solve_ivp(_rhs, (0.0, phi_span), [u0, du0], args=(mass,),
                    events=(hit_horizon, reach_escape),
                    rtol=rtol, atol=atol, dense_output=False)

    if sol.t_events[0].size > 0:
        status = "captured"
    elif sol.t_events[1].size > 0:
        status = "escaped"
    else:
        status = "incomplete"

    return {"status": status, "phi": sol.t, "r": 1.0 / sol.y[0]}


def deflection_angle(b, mass, r_start=1.0e6):
    """Total light deflection for a ray coming from infinity with impact
    parameter b. Returns np.inf if the ray is captured."""
    if b <= critical_impact_parameter(mass):
        return np.inf
    res = integrate_ray(r_start, b, mass, ingoing=True,
                        r_escape=r_start, phi_span=200.0)
    if res["status"] != "escaped":
        return np.inf
    return res["phi"][-1] - 2.0 * np.arccos(b / r_start)
