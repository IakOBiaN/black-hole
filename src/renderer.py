"""Backward ray tracing of the camera image."""

import numpy as np
from PIL import Image

from .metric import lapse_squared
from .geodesic import integrate_ray


def ray_status(r0, cos_psi, sin_psi, mass, r_escape_factor=3.0,
               rtol=1.0e-6, atol=1.0e-9):
    """Classify a single ray as 'captured' or 'escaped'. cos_psi/sin_psi
    describe the angle between the ray and the outward radial direction."""
    if sin_psi <= 1.0e-12:
        return "captured" if cos_psi < 0.0 else "escaped"
    b = r0 * sin_psi / np.sqrt(lapse_squared(r0, mass))
    res = integrate_ray(r0, b, mass, ingoing=cos_psi < 0.0,
                        r_escape=r_escape_factor * r0, rtol=rtol, atol=atol)
    return res["status"]


def render(camera, mass, shadow_color=(0, 0, 0), miss_color=(0, 0, 0)):
    """Render the scene. Captured rays take shadow_color; rays that escape
    to infinity take miss_color (the background)."""
    pos = camera.position
    r0 = np.linalg.norm(pos)
    radial = pos / r0
    dirs = camera.ray_directions()
    h, w, _ = dirs.shape

    shadow = np.array(shadow_color, dtype=np.uint8)
    miss = np.array(miss_color, dtype=np.uint8)
    img = np.empty((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            d = dirs[i, j]
            cos_psi = float(np.dot(d, radial))
            sin_psi = np.sqrt(max(0.0, 1.0 - cos_psi * cos_psi))
            status = ray_status(r0, cos_psi, sin_psi, mass)
            img[i, j] = shadow if status == "captured" else miss

    return img


def save_png(image, path):
    Image.fromarray(image, mode="RGB").save(path)
