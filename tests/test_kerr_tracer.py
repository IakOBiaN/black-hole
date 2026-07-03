import numpy as np

from src.camera import Camera
from src.disk import Disk
from src.kerr_tracer import trace_batch_kerr

NO_DISK = Disk(1000.0, 1001.0)  # inner radius never reached -> pure shadow map


def _shadow_centroid_col(a):
    cam = Camera(distance=25.0, resolution=(64, 48), fov_deg=34.0,
                 inclination_deg=12.0)
    _, _, _, captured = trace_batch_kerr(cam, a, NO_DISK, dzeta=0.15,
                                         max_steps=3000)
    cols = np.where(captured)[1]
    return cols.mean()


def test_frame_dragging_shifts_the_shadow():
    center = (64 - 1) / 2.0
    centroid_schwarzschild = _shadow_centroid_col(0.0)
    centroid_kerr = _shadow_centroid_col(0.9)

    # Non-spinning shadow is centered; frame dragging displaces the spinning
    # shadow sideways.
    assert abs(centroid_schwarzschild - center) < 2.0
    assert centroid_kerr < centroid_schwarzschild - 3.0
