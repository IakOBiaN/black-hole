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


def _disk_emission(r, g, azimuth, disk_inner, t_peak, texture_contrast,
                   texture_kwargs):
    """Linear HDR emission for masked disk samples given their radius, redshift
    factor g and azimuth. Observed temperature is g * T_emit, so brightness
    ~ T_obs^4 carries the Doppler beaming and gravitational dimming."""
    from .temperature import disk_temperature
    from .color import blackbody_color
    from .texture import disk_pattern

    temp = g * disk_temperature(r, disk_inner, t_peak)
    samples = np.linspace(1000.0, 4.0 * t_peak, 512)
    lut = np.array([blackbody_color(t) for t in samples])
    clamped = np.clip(temp, samples[0], samples[-1])
    hue = np.stack([np.interp(clamped, samples, lut[:, c])
                    for c in range(3)], axis=1)

    pattern = disk_pattern(r, azimuth, disk_inner, **texture_kwargs)
    texture = 1.0 + texture_contrast * (2.0 * pattern - 1.0)
    brightness = (temp / t_peak) ** 4 * texture
    return hue * brightness[:, None]


def shade_disk_linear(radius_buffer, bz_buffer, azimuth_buffer, disk, mass,
                      t_peak=5000.0, mode="accurate", doppler_strength=None,
                      texture_contrast=1.0, texture_kwargs=None):
    """Linear HDR emission of the Schwarzschild disk (no bloom, no tone map)."""
    from .disk import redshift_factor

    if doppler_strength is None:
        doppler_strength = 0.6 if mode == "beautiful" else 1.0
    if texture_kwargs is None:
        texture_kwargs = {}

    linear = np.zeros(radius_buffer.shape + (3,))
    mask = ~np.isnan(radius_buffer)
    if mask.any():
        r = radius_buffer[mask]
        g = redshift_factor(r, bz_buffer[mask], mass, doppler_strength)
        linear[mask] = _disk_emission(r, g, azimuth_buffer[mask], disk.inner,
                                      t_peak, texture_contrast, texture_kwargs)
    return linear


def _downsample(image, factor):
    """Average factor x factor blocks (supersampled anti-aliasing)."""
    h, w, c = image.shape
    return image.reshape(h // factor, factor, w // factor, factor, c).mean((1, 3))


def render_disk_image(camera, mass, disk, mode="beautiful", t_peak=5000.0,
                      supersample=2, bloom_strength=None, texture_contrast=1.0,
                      doppler_strength=None, texture_kwargs=None):
    """Full thin-disk render: supersample, shade in linear light, downsample,
    add bloom, tone map to an 8-bit sRGB frame."""
    from .tracer import trace_batch
    from .postprocess import add_bloom
    from .color import tonemap

    hi = camera.supersampled(supersample) if supersample > 1 else camera
    radii, bz, azimuth = trace_batch(hi, mass, disk)
    linear = shade_disk_linear(radii, bz, azimuth, disk, mass, t_peak, mode,
                               doppler_strength, texture_contrast, texture_kwargs)
    if supersample > 1:
        linear = _downsample(linear, supersample)

    if bloom_strength is None:
        bloom_strength = 0.9 if mode == "beautiful" else 0.4
    return tonemap(add_bloom(linear, bloom_strength), mode)


def render_kerr_image(camera, spin, disk, mode="beautiful", t_peak=5000.0,
                      supersample=1, bloom_strength=None, texture_contrast=1.0,
                      doppler_strength=None, texture_kwargs=None):
    """Full Kerr render: trace the rotating black hole, shade the equatorial
    disk with the Kerr circular-orbit redshift, add bloom, tone map."""
    from .kerr_tracer import trace_batch_kerr
    from .kerr import kerr_redshift_factor
    from .postprocess import add_bloom
    from .color import tonemap

    if doppler_strength is None:
        doppler_strength = 0.6 if mode == "beautiful" else 1.0
    if texture_kwargs is None:
        texture_kwargs = {}

    hi = camera.supersampled(supersample) if supersample > 1 else camera
    radius, b, azimuth, _ = trace_batch_kerr(hi, spin, disk)

    linear = np.zeros(radius.shape + (3,))
    mask = ~np.isnan(radius)
    if mask.any():
        g = kerr_redshift_factor(radius[mask], b[mask], spin, doppler_strength)
        linear[mask] = _disk_emission(radius[mask], g, azimuth[mask], disk.inner,
                                      t_peak, texture_contrast, texture_kwargs)
    if supersample > 1:
        linear = _downsample(linear, supersample)

    if bloom_strength is None:
        bloom_strength = 0.9 if mode == "beautiful" else 0.4
    return tonemap(add_bloom(linear, bloom_strength), mode)


def save_png(image, path):
    Image.fromarray(image, mode="RGB").save(path)
