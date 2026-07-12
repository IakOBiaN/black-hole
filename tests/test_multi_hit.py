"""Tests for the multi-crossing Kerr tracer used by the transparent disk."""

import numpy as np

from src.camera import Camera
from src.disk import Disk
from src.kerr import isco
from src.kerr_numba import trace_batch_kerr, trace_batch_kerr_multi

A = 0.6
CAM = Camera(distance=40.0, resolution=(64, 40), fov_deg=40.0,
             inclination_deg=10.0)
DISK = Disk(isco(A), 14.0)


def test_first_hit_matches_single_hit_tracer():
    hits_r, hits_phi, n_hits, b_multi, _ = trace_batch_kerr_multi(
        CAM, A, DISK.inner, DISK.outer, max_hits=6)
    radius, b_single, azimuth, _ = trace_batch_kerr(CAM, A, DISK)

    hit = n_hits > 0
    assert np.array_equal(hit, ~np.isnan(radius))
    assert np.allclose(hits_r[hit, 0], radius[hit], atol=1.0e-9)
    assert np.allclose(hits_phi[hit, 0], azimuth[hit], atol=1.0e-9)
    assert np.allclose(b_multi, b_single)


def test_rays_near_the_hole_cross_the_plane_more_than_once():
    # Rays lensed around the hole pierce the equatorial plane repeatedly;
    # a camera near the disk plane must see secondary images somewhere.
    _, _, n_hits, _, _ = trace_batch_kerr_multi(
        CAM, A, DISK.inner, DISK.outer, max_hits=6)
    assert n_hits.max() >= 2
    assert (n_hits >= 2).sum() > 5


def test_hits_ordered_front_to_back_in_bounds():
    hits_r, _, n_hits, _, _ = trace_batch_kerr_multi(
        CAM, A, DISK.inner, DISK.outer, max_hits=6)
    for k in range(6):
        sel = n_hits > k
        assert np.all(hits_r[sel, k] >= DISK.inner - 1.0e-9)
        assert np.all(hits_r[sel, k] <= DISK.outer + 1.0e-9)
