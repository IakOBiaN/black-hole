"""Thin accretion disk in the equatorial plane, and ray-disk intersection.

A photon path lies in the plane spanned by the camera position and the ray
direction. Within that plane the 3D position at polar angle phi is

    X(phi) = r(phi) * (cos(phi) * e1 + sin(phi) * e2),

so the ray crosses the equatorial plane (z = 0) where

    cos(phi) * (e1.z) + sin(phi) * (e2.z) = 0.

We integrate r(phi) along the geodesic and pick the first crossing whose
radius falls inside the disk annulus; that is the visible (opaque) surface.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .geodesic import _rhs, _initial_slope
from .metric import schwarzschild_radius, lapse_squared


class Disk:
    def __init__(self, inner, outer):
        self.inner = inner
        self.outer = outer

    def contains(self, r):
        return self.inner <= r <= self.outer


def _orbital_basis(pos, direction):
    r0 = np.linalg.norm(pos)
    e1 = pos / r0
    tangential = direction - np.dot(direction, e1) * e1
    norm = np.linalg.norm(tangential)
    if norm < 1.0e-12:
        return r0, e1, None
    return r0, e1, tangential / norm


def orbital_angular_velocity(r, mass):
    """Keplerian (circular geodesic) angular velocity, prograde about +z."""
    return np.sqrt(mass / r ** 3)


def redshift_factor(r, bz, mass, doppler_strength=1.0):
    """Frequency shift g = nu_obs/nu_emit for a photon (z angular momentum
    per energy bz) emitted by disk material on a circular orbit at radius r.
    doppler_strength scales the orbital Doppler term (1 = physical)."""
    grav = np.sqrt(1.0 - 3.0 * mass / r)
    denom = 1.0 - doppler_strength * bz * orbital_angular_velocity(r, mass)
    denom = np.clip(denom, 1.0e-3, None)
    return grav / denom


def trace(pos, direction, mass, disk, r_escape_factor=1.1,
          phi_span=50.0, rtol=1.0e-7, atol=1.0e-9):
    """Trace a ray and return (kind, radius, bz): kind is 'disk', 'horizon'
    or 'background'; radius is the disk hit radius (or None); bz is the
    photon z angular momentum per energy (for the Doppler shift)."""
    r0, e1, e2 = _orbital_basis(pos, direction)
    cos_psi = float(np.dot(direction, e1))
    bz = float(np.cross(pos, direction)[2]) / np.sqrt(lapse_squared(r0, mass))
    if e2 is None:
        return ("horizon" if cos_psi < 0.0 else "background", None, bz)

    rs = schwarzschild_radius(mass)
    sin_psi = float(np.linalg.norm(direction - cos_psi * e1))
    b = r0 * sin_psi / np.sqrt(lapse_squared(r0, mass))
    u0 = 1.0 / r0
    du0 = _initial_slope(r0, b, mass, ingoing=cos_psi < 0.0)

    cz1, cz2 = float(e1[2]), float(e2[2])
    r_escape = r_escape_factor * r0

    def hit_horizon(phi, y, m):
        return y[0] - 1.0 / rs
    hit_horizon.terminal = True
    hit_horizon.direction = 1.0

    def reach_escape(phi, y, m):
        return y[0] - 1.0 / r_escape
    reach_escape.terminal = True
    reach_escape.direction = -1.0

    def cross_equator(phi, y, m):
        return np.cos(phi) * cz1 + np.sin(phi) * cz2

    sol = solve_ivp(_rhs, (0.0, phi_span), [u0, du0], args=(mass,),
                    events=(hit_horizon, reach_escape, cross_equator),
                    rtol=rtol, atol=atol, dense_output=True)

    for phi in sol.t_events[2]:
        r = 1.0 / sol.sol(phi)[0]
        if disk.contains(r):
            return ("disk", r, bz)

    if sol.t_events[0].size > 0:
        return ("horizon", None, bz)
    return ("background", None, bz)
