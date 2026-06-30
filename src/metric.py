"""Schwarzschild metric in geometrized units (G = c = 1)."""

import numpy as np


def schwarzschild_radius(mass):
    return 2.0 * mass


def photon_sphere_radius(mass):
    return 3.0 * mass


def critical_impact_parameter(mass):
    return 3.0 * np.sqrt(3.0) * mass


def lapse_squared(r, mass):
    return 1.0 - 2.0 * mass / r


def impact_parameter_from_angle(r, psi, mass):
    """Impact parameter b = L/E for a ray emitted at angle psi from the
    outward radial direction, measured in the local static frame."""
    return r * np.sin(psi) / np.sqrt(lapse_squared(r, mass))
