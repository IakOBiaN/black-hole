import numpy as np

from src.camera import Camera
from src.disk import Disk
from src.tracer import trace_batch
from src.renderer import render_buffers

CAM = Camera(distance=30.0, resolution=(48, 48), fov_deg=52.0, inclination_deg=10.0)
DISK = Disk(6.0, 14.0)


def test_batch_matches_reference_tracer():
    r, bz, _ = trace_batch(CAM, 1.0, DISK)
    ref_r, ref_bz = render_buffers(CAM, 1.0, DISK)

    hit = ~np.isnan(r)
    ref_hit = ~np.isnan(ref_r)
    assert (hit == ref_hit).mean() > 0.999

    common = hit & ref_hit
    assert np.abs(r[common] - ref_r[common]).max() < 0.01
    assert np.allclose(bz, ref_bz)


def test_azimuth_defined_on_disk_hits():
    r, _, az = trace_batch(CAM, 1.0, DISK)
    hit = ~np.isnan(r)
    assert np.all(~np.isnan(az[hit]))
    assert np.all((az[hit] >= -np.pi) & (az[hit] <= np.pi))
