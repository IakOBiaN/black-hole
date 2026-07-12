"""End-to-end check of the Doppler sign through the whole tracing pipeline.

Geometry ground truth: the camera sits on the +x axis and the disk orbits
prograde about +z (Omega = dphi/dt > 0, counterclockwise seen from above).
Material at -y therefore moves toward the camera and material at +y away
from it. The camera's screen-right basis vector is +y, so the approaching
side is screen-LEFT and must be blueshifted (g > 1).
"""

import numpy as np

from src.camera import Camera
from src.disk import Disk, redshift_factor
from src.kerr import isco, kerr_redshift_factor
from src.kerr_numba import trace_batch_kerr
from src.tracer import trace_batch

M = 1.0


def test_kerr_approaching_side_is_screen_left():
    a = 0.6
    cam = Camera(distance=40.0, resolution=(80, 50), fov_deg=40.0,
                 inclination_deg=15.0)
    disk = Disk(isco(a), 14.0)
    radius, b, _, _ = trace_batch_kerr(cam, a, disk)

    mask = ~np.isnan(radius)
    g = np.full(radius.shape, np.nan)
    g[mask] = kerr_redshift_factor(radius[mask], b[mask], a)

    w = radius.shape[1]
    assert np.nanmean(g[:, : w // 2]) > 1.0      # approaching: blueshift
    assert np.nanmean(g[:, w // 2:]) < 1.0       # receding: redshift


def test_schwarzschild_approaching_side_is_screen_left():
    cam = Camera(distance=30.0, resolution=(80, 50), fov_deg=45.0,
                 inclination_deg=12.0)
    disk = Disk(6.0, 14.0)
    radius, bz, _ = trace_batch(cam, M, disk)

    mask = ~np.isnan(radius)
    g = np.full(radius.shape, np.nan)
    g[mask] = redshift_factor(radius[mask], bz[mask], M)

    w = radius.shape[1]
    assert np.nanmean(g[:, : w // 2]) > np.nanmean(g[:, w // 2:])


def test_camera_position_factor_blueshifts_slightly():
    # A FIDO at finite radius sees a slightly higher frequency than an
    # observer at infinity (1/alpha > 1), for a b = 0 photon.
    g_inf = kerr_redshift_factor(10.0, 0.0, 0.6)
    g_cam = kerr_redshift_factor(10.0, 0.0, 0.6, r_cam=74.1)
    assert g_cam > g_inf
    assert np.isclose(g_cam / g_inf, 1.0, atol=0.05)
