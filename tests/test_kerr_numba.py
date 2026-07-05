import numpy as np

from src.camera import Camera
from src.disk import Disk
from src.kerr import isco
from src.kerr_tracer import trace_batch_kerr as trace_numpy
from src.kerr_numba import trace_batch_kerr as trace_numba


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
