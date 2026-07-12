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


# Luma weights used by James et al. (2015), Appendix "DNGR modelling of
# accretion disks", to strip the Doppler-induced intensity change while
# keeping the perceived colour: {R,G,B}/(0.30R + 0.59G + 0.11B).
_LUMA_FILM = np.array([0.30, 0.59, 0.11])


def _blackbody_hue_lut(t_max, n=768):
    from .color import blackbody_color

    samples = np.linspace(500.0, t_max, n)
    lut = np.array([blackbody_color(t) for t in samples])

    def sample(temps):
        clamped = np.clip(temps, samples[0], samples[-1])
        return np.stack([np.interp(clamped, samples, lut[:, c])
                         for c in range(3)], axis=-1)
    return sample


def _shifted_disk_color(t_emit, g, t_ref, shift_mode, hue_of):
    """Linear RGB of blackbody disk samples under the article's three
    treatments of the Doppler + gravitational frequency shift (Fig. 15):

    - "none": no shift at all -- the disk as painted (Fig. 15a);
    - "hue":  colours shifted (T_obs = g T), but the intensity change removed
      by normalizing with the film-luma weighted mean (Fig. 15b);
    - "full": colours shifted and specific intensity transported per
      Liouville's theorem, I ~ g^4 for a blackbody (Fig. 15c).
    """
    t_emit = np.broadcast_to(np.asarray(t_emit, dtype=float), np.shape(g))
    if shift_mode == "none":
        hue = hue_of(t_emit)
        brightness = (t_emit / t_ref) ** 4
    elif shift_mode == "hue":
        hue = hue_of(g * t_emit)
        luma = hue @ _LUMA_FILM
        luma_ref = hue_of(t_emit) @ _LUMA_FILM
        hue = hue * (luma_ref / np.maximum(luma, 1.0e-9))[:, None]
        brightness = (t_emit / t_ref) ** 4
    elif shift_mode == "full":
        hue = hue_of(g * t_emit)
        brightness = (g * t_emit / t_ref) ** 4
    else:
        raise ValueError(f"unknown shift_mode {shift_mode!r}")
    return hue * brightness[:, None]


def trace_kerr_scene(camera, spin, disk, supersample=2, max_hits=8,
                     dzeta=0.1, max_steps=6000, texture_kwargs=None):
    """Geometry pass: trace every (supersampled) ray once and keep all
    equatorial crossings. The Kerr metric is axisymmetric and static, so the
    same traced scene can be re-shaded for any disk time and any camera
    azimuth -- animations pay for tracing only once."""
    from .kerr_numba import trace_batch_kerr_multi
    from .texture import debris_extent

    if texture_kwargs is None:
        texture_kwargs = {}
    hi = camera.supersampled(supersample) if supersample > 1 else camera
    pos = hi.position
    r_cam = float(np.linalg.norm(pos))
    theta_cam = float(np.arccos(pos[2] / r_cam))

    debris_ratio = texture_kwargs.get("debris_ratio", 0.22)
    r_min = disk.inner * 0.98
    r_max = debris_extent(disk.inner, disk.outer, debris_ratio)
    hits_r, hits_phi, n_hits, b, _ = trace_batch_kerr_multi(
        hi, spin, r_min, r_max, max_hits=max_hits, dzeta=dzeta,
        max_steps=max_steps)

    return {"hits_r": hits_r, "hits_phi": hits_phi, "n_hits": n_hits,
            "b": b, "r_cam": r_cam, "theta_cam": theta_cam,
            "max_hits": max_hits, "supersample": supersample}


def shade_kerr_scene(scene, spin, disk, mode="beautiful", t_peak=4500.0,
                     bloom_strength=None, shift_mode=None,
                     temp_profile="powerlaw", temp_exponent=0.45,
                     doppler_strength=1.0, time=0.0, camera_azimuth=0.0,
                     texture_kwargs=None, exposure=None, saturation=None):
    """Shading pass: composite the lensed disk layers front to back with the
    material's opacity, apply bloom and tone map.

    time is coordinate time in M: the gas orbits differentially at its
    Keplerian rate. camera_azimuth (radians) swings the camera around the
    spin axis (by axisymmetry this only re-phases the disk pattern).

    shift_mode is "none" (movie look, default for mode="beautiful"), "hue"
    or "full" (physically complete, default for mode="accurate").
    temp_profile: "powerlaw" (T = t_peak (r/r_in)^-temp_exponent -- the disk
    dims and reddens toward its edge), "constant" (the article's uniform
    4500 K sheet) or "shakura" (zero-torque profile peaking at t_peak).
    """
    from .kerr import kerr_redshift_factor
    from .temperature import disk_temperature
    from .texture import disk_material
    from .postprocess import add_bloom
    from .color import tonemap

    if shift_mode is None:
        shift_mode = "none" if mode == "beautiful" else "full"
    if texture_kwargs is None:
        texture_kwargs = {}

    hits_r, hits_phi = scene["hits_r"], scene["hits_phi"]
    n_hits, b = scene["n_hits"], scene["b"]

    hue_of = _blackbody_hue_lut(6.0 * t_peak)
    linear = np.zeros(n_hits.shape + (3,))
    transmit = np.ones(n_hits.shape)
    for k in range(scene["max_hits"]):
        mask = (n_hits > k) & (transmit > 1.0e-3)
        if not mask.any():
            break
        r = hits_r[..., k][mask]
        phi = hits_phi[..., k][mask] + camera_azimuth
        emission, alpha = disk_material(r, phi, disk.inner, disk.outer,
                                        time=time, **texture_kwargs)
        if temp_profile == "constant":
            t_emit = np.full_like(r, t_peak)
        elif temp_profile == "powerlaw":
            t_emit = t_peak * (r / disk.inner) ** -temp_exponent
        else:
            t_emit = disk_temperature(r, disk.inner, t_peak)
        g = kerr_redshift_factor(r, b[mask], spin, doppler_strength,
                                 r_cam=scene["r_cam"],
                                 theta_cam=scene["theta_cam"])
        color = _shifted_disk_color(t_emit, g, t_peak, shift_mode, hue_of)
        weight = transmit[mask] * alpha * emission
        linear[mask] += weight[:, None] * color
        transmit[mask] *= 1.0 - alpha

    if scene["supersample"] > 1:
        linear = _downsample(linear, scene["supersample"])

    if bloom_strength is None:
        bloom_strength = 0.75 if mode == "beautiful" else 0.4
    return tonemap(add_bloom(linear, bloom_strength), mode,
                   exposure=exposure, saturation=saturation)


def render_kerr_image(camera, spin, disk, mode="beautiful", t_peak=4500.0,
                      supersample=2, bloom_strength=None, shift_mode=None,
                      temp_profile="powerlaw", temp_exponent=0.45,
                      doppler_strength=1.0, time=0.0, camera_azimuth=0.0,
                      max_hits=8, dzeta=0.1, max_steps=6000,
                      texture_kwargs=None, exposure=None, saturation=None):
    """Full Kerr render of the semi-transparent artist disk: trace the scene
    once and shade it. See trace_kerr_scene / shade_kerr_scene."""
    scene = trace_kerr_scene(camera, spin, disk, supersample=supersample,
                             max_hits=max_hits, dzeta=dzeta,
                             max_steps=max_steps,
                             texture_kwargs=texture_kwargs)
    return shade_kerr_scene(scene, spin, disk, mode=mode, t_peak=t_peak,
                            bloom_strength=bloom_strength,
                            shift_mode=shift_mode, temp_profile=temp_profile,
                            temp_exponent=temp_exponent,
                            doppler_strength=doppler_strength, time=time,
                            camera_azimuth=camera_azimuth,
                            texture_kwargs=texture_kwargs,
                            exposure=exposure, saturation=saturation)


def save_png(image, path):
    Image.fromarray(image, mode="RGB").save(path)
