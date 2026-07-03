"""Kerr (rotating) black hole geodesics in Boyer-Lindquist coordinates.

Units are geometrized with M = 1, so the spin parameter a runs in [0, 1). The
null geodesic equations use the super-Hamiltonian form of James et al. (2015),
eq. (Rays2), which stays well behaved at radial and polar turning points. The
ray's conserved quantities are its axial angular momentum b = p_phi (energy
-p_t set to 1) and its Carter constant q.

Spatial and momentum derivatives of the potential are taken numerically for
now; analytic derivatives are an optimization for later.
"""

import numpy as np


def horizon_radius(a):
    return 1.0 + np.sqrt(1.0 - a * a)


def delta(r, a):
    return r * r - 2.0 * r + a * a


def rho2(r, theta, a):
    return r * r + a * a * np.cos(theta) ** 2


def photon_orbit_b(r_o, a):
    """Axial angular momentum of an unstable spherical photon orbit at r_o."""
    return -(r_o ** 3 - 3.0 * r_o ** 2 + a * a * r_o + a * a) / (a * (r_o - 1.0))


def photon_orbit_q(r_o, a):
    """Carter constant of an unstable spherical photon orbit at r_o."""
    return -(r_o ** 3 * (r_o ** 3 - 6.0 * r_o ** 2 + 9.0 * r_o - 4.0 * a * a)) \
        / (a * a * (r_o - 1.0) ** 2)


def photon_orbit_radii(a):
    """Radii of the prograde (r1) and retrograde (r2) equatorial photon
    orbits, which bound the spherical photon orbits."""
    r1 = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
    r2 = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(a)))
    return r1, r2


def _potential(r, theta, b, q, a):
    """U = (R + Delta*Theta) / (2 Delta rho^2); the q term cancels in R+DT."""
    ct2 = np.cos(theta) ** 2
    st2 = np.sin(theta) ** 2
    d = r * r - 2.0 * r + a * a
    rr2 = r * r + a * a * ct2
    P = r * r + a * a - a * b
    R = P * P - d * ((b - a) ** 2 + q)
    Theta = q - ct2 * (b * b / st2 - a * a)
    return (R + d * Theta) / (2.0 * d * rr2)


def rhs(y, b, q, a, h=1.0e-6):
    """Right-hand side of the null geodesic equations for state
    y = [r, theta, phi, p_r, p_theta]."""
    r, theta, phi, p_r, p_theta = y
    d = delta(r, a)
    rr2 = rho2(r, theta, a)

    dr = d / rr2 * p_r
    dtheta = p_theta / rr2
    dphi = -(_potential(r, theta, b + h, q, a)
             - _potential(r, theta, b - h, q, a)) / (2.0 * h)

    def hamiltonian(rr, th):
        dd = delta(rr, a)
        r2 = rho2(rr, th, a)
        return (-dd / (2.0 * r2) * p_r * p_r - 1.0 / (2.0 * r2) * p_theta * p_theta
                + _potential(rr, th, b, q, a))

    dp_r = (hamiltonian(r + h, theta) - hamiltonian(r - h, theta)) / (2.0 * h)
    dp_theta = (hamiltonian(r, theta + h) - hamiltonian(r, theta - h)) / (2.0 * h)
    return np.array([dr, dtheta, dphi, dp_r, dp_theta])


def _rk4(y, b, q, a, dzeta):
    k1 = rhs(y, b, q, a)
    k2 = rhs(y + 0.5 * dzeta * k1, b, q, a)
    k3 = rhs(y + 0.5 * dzeta * k2, b, q, a)
    k4 = rhs(y + dzeta * k3, b, q, a)
    return y + dzeta / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def equatorial_ray_status(b, a, r0=500.0, dzeta=0.5, max_steps=40000):
    """Trace an equatorial (q = 0) photon inward from r0 with the given axial
    angular momentum and report 'captured' or 'escaped'."""
    d0 = delta(r0, a)
    P0 = r0 * r0 + a * a - a * b
    R0 = P0 * P0 - d0 * (b - a) ** 2
    p_r = -np.sqrt(max(R0, 0.0)) / d0
    y = np.array([r0, np.pi / 2.0, 0.0, p_r, 0.0])
    r_h = horizon_radius(a)

    for _ in range(max_steps):
        y = _rk4(y, b, 0.0, a, dzeta)
        if y[0] <= r_h * 1.001:
            return "captured"
        if y[0] >= r0:
            return "escaped"
    return "incomplete"
