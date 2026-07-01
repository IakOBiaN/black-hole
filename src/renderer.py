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


def render_buffers(camera, mass, disk):
    """Trace every pixel. Returns (radius, bz) buffers: radius is the disk hit
    radius (NaN where no hit), bz the photon z angular momentum per energy.
    This is the expensive geometry pass; shading is applied separately."""
    from .disk import trace

    pos = camera.position
    dirs = camera.ray_directions()
    h, w, _ = dirs.shape

    radius = np.full((h, w), np.nan)
    bz = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            kind, r, b = trace(pos, dirs[i, j], mass, disk)
            bz[i, j] = b
            if kind == "disk":
                radius[i, j] = r
    return radius, bz


def shade_disk(radius_buffer, bz_buffer, azimuth_buffer, disk, mass,
               t_peak=4000.0, mode="accurate", doppler_strength=None,
               texture_contrast=0.8, bloom_strength=None):
    """Color a radius buffer including relativistic shifts: the observed
    temperature is g * T_emit, which carries Doppler shift, Doppler beaming
    (via brightness ~ T^4) and gravitational redshift together. A procedural
    gas texture modulates the brightness, and a bloom halo is added in linear
    light before tone mapping."""
    from .temperature import disk_temperature
    from .disk import redshift_factor
    from .color import blackbody_color, tonemap
    from .texture import disk_pattern
    from .postprocess import add_bloom

    if doppler_strength is None:
        doppler_strength = 0.6 if mode == "beautiful" else 1.0
    if bloom_strength is None:
        bloom_strength = 0.9 if mode == "beautiful" else 0.4

    h, w = radius_buffer.shape
    linear = np.zeros((h, w, 3))
    mask = ~np.isnan(radius_buffer)

    if mask.any():
        r = radius_buffer[mask]
        g = redshift_factor(r, bz_buffer[mask], mass, doppler_strength)
        temp = g * disk_temperature(r, disk.inner, t_peak)

        samples = np.linspace(1000.0, 4.0 * t_peak, 512)
        lut = np.array([blackbody_color(t) for t in samples])
        clamped = np.clip(temp, samples[0], samples[-1])
        hue = np.stack([np.interp(clamped, samples, lut[:, c])
                        for c in range(3)], axis=1)

        pattern = disk_pattern(r, azimuth_buffer[mask], disk.inner)
        texture = 1.0 + texture_contrast * (2.0 * pattern - 1.0)

        brightness = (temp / t_peak) ** 4 * texture
        linear[mask] = hue * brightness[:, None]

    linear = add_bloom(linear, bloom_strength)
    return tonemap(linear, mode)


def save_png(image, path):
    Image.fromarray(image, mode="RGB").save(path)
