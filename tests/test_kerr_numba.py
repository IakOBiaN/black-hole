import numpy as np

from black_hole.camera import Camera
from black_hole.disk import Disk
from black_hole.kerr import isco
from black_hole.kerr_tracer import trace_batch_kerr as trace_numpy
from black_hole.kerr_numba import trace_batch_kerr as trace_numba


def test_numba_tracer_matches_numpy_reference():
    a = 0.9
    cam = Camera(distance=25.0, resolution=(48, 36), fov_deg=40.0,
                 inclination_deg=15.0)
    disk = Disk(isco(a), 14.0)

    r_nb, _, az_nb, cap_nb = trace_numba(cam, a, disk)
    r_np, _, az_np, cap_np = trace_numpy(cam, a, disk)

    assert np.array_equal(cap_nb, cap_np)
    assert np.array_equal(np.isnan(r_nb), np.isnan(r_np))

    common = ~np.isnan(r_nb)
    assert np.abs(r_nb[common] - r_np[common]).max() < 1.0e-6
    assert np.abs(az_nb[common] - az_np[common]).max() < 1.0e-6
