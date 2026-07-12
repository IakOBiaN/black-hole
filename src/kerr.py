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


def sigma(r, theta, a):
    return np.sqrt((r * r + a * a) ** 2 - a * a * delta(r, a) * np.sin(theta) ** 2)


def alpha(r, theta, a):
    """FIDO lapse function."""
    return np.sqrt(rho2(r, theta, a) * delta(r, a)) / sigma(r, theta, a)


def omega(r, theta, a):
    """Frame-dragging angular velocity of the FIDO."""
    return 2.0 * a * r / sigma(r, theta, a) ** 2


def varpi(r, theta, a):
    """Cylindrical radius factor (proper circumference / 2 pi)."""
    return sigma(r, theta, a) * np.sin(theta) / np.sqrt(rho2(r, theta, a))


def isco(a, prograde=True):
    """Innermost stable circular orbit radius (Bardeen et al.)."""
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = np.sqrt(3.0 * a * a + z1 * z1)
    sign = -1.0 if prograde else 1.0
    return 3.0 + z2 + sign * np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


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


def kerr_redshift_factor(r, b, a, doppler_strength=1.0, r_cam=None,
                         theta_cam=np.pi / 2.0):
    """Frequency shift g = nu_obs/nu_emit for a photon (axial angular momentum
    b, energy 1) emitted by disk material on a prograde circular geodesic at
    equatorial radius r. Reduces to sqrt(1-3/r)/(1-Omega b) when a = 0.

    By default the observer sits at infinity (nu_obs = 1). Pass the camera's
    Boyer-Lindquist r_cam (and theta_cam) to observe with a FIDO there
    instead: nu_obs = (1 - b omega_c)/alpha_c, cf. the blue-shift formula in
    James et al. (2015), Appendix A.1."""
    omega_orb = 1.0 / (r ** 1.5 + a)
    g_tt = -(1.0 - 2.0 / r)
    g_tphi = -2.0 * a / r
    g_phiphi = r * r + a * a + 2.0 * a * a / r
    dt2 = -(g_tt + 2.0 * omega_orb * g_tphi + omega_orb * omega_orb * g_phiphi)
    u_t = 1.0 / np.sqrt(dt2)
    denom = np.clip(1.0 - doppler_strength * omega_orb * b, 1.0e-3, None)
    nu_obs = 1.0
    if r_cam is not None:
        nu_obs = ((1.0 - b * omega(r_cam, theta_cam, a))
                  / alpha(r_cam, theta_cam, a))
    return nu_obs / (u_t * denom)


def _potential(r, theta, b, q, a):
    """U = (R + Delta*Theta) / (2 Delta rho^2); the q term cancels in R+DT."""
    ct2 = np.cos(theta) ** 2
    # Softly regularize sin^2(theta) so the b^2/sin^2 term and its derivative
    # stay finite and smooth at the spin axis, where Boyer-Lindquist
    # coordinates are singular; only a hair-thin polar region is affected.
    st2 = np.sin(theta) ** 2 + 1.0e-9
    d = r * r - 2.0 * r + a * a
    rr2 = r * r + a * a * ct2
    P = r * r + a * a - a * b
    R = P * P - d * ((b - a) ** 2 + q)
    Theta = q - ct2 * (b * b / st2 - a * a)
    return (R + d * Theta) / (2.0 * d * rr2)


def rhs_numerical(y, b, q, a, h=1.0e-6):
    """Right-hand side via finite-difference derivatives; kept as the
    validated reference for the analytic rhs."""
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


def rhs(y, b, q, a):
    """Right-hand side of the null geodesic equations for state
    y = [r, theta, phi, p_r, p_theta], with analytic derivatives.

    G = -Delta/(2 rho^2) p_r^2 - 1/(2 rho^2) p_theta^2 + U, where the potential
    U = N/(2 Delta rho^2) and N = R + Delta*Theta = P^2 - Delta*B with
    P = r^2+a^2-a b and B = (b-a)^2 + cos^2(theta)(b^2/sin^2(theta) - a^2)."""
    r, theta, phi, p_r, p_theta = y
    ct, stn = np.cos(theta), np.sin(theta)
    ct2 = ct * ct
    st2 = stn * stn + 1.0e-9

    rr2 = r * r + a * a * ct2
    d = r * r - 2.0 * r + a * a
    P = r * r + a * a - a * b
    bb = (b - a) ** 2 + ct2 * (b * b / st2 - a * a)
    n = P * P - d * bb

    dr = d / rr2 * p_r
    dtheta = p_theta / rr2

    # dphi = -dU/db
    db_db = 2.0 * (b - a) + 2.0 * b * ct2 / st2
    dn_db = -2.0 * a * P - d * db_db
    dphi = -dn_db / (2.0 * d * rr2)

    # dp_r = dG/dr
    drr2_dr = 2.0 * r
    dd_dr = 2.0 * r - 2.0
    dn_dr = 2.0 * P * (2.0 * r) - dd_dr * bb
    t1 = -0.5 * p_r * p_r * (dd_dr * rr2 - d * drr2_dr) / (rr2 * rr2)
    t2 = 0.5 * p_theta * p_theta * drr2_dr / (rr2 * rr2)
    drrho = d * rr2
    ddrrho_dr = dd_dr * rr2 + d * drr2_dr
    t3 = 0.5 * (dn_dr * drrho - n * ddrrho_dr) / (drrho * drrho)
    dp_r = t1 + t2 + t3

    # dp_theta = dG/dtheta
    drr2_dth = -2.0 * a * a * stn * ct
    db_dth = -2.0 * b * b * ct / (st2 * stn) + 2.0 * a * a * stn * ct
    dn_dth = -d * db_dth
    s1 = 0.5 * d * p_r * p_r * drr2_dth / (rr2 * rr2)
    s2 = 0.5 * p_theta * p_theta * drr2_dth / (rr2 * rr2)
    s3 = 0.5 / d * (dn_dth * rr2 - n * drr2_dth) / (rr2 * rr2)
    dp_theta = s1 + s2 + s3

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
