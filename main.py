"""Render the current scene to out/.

A black hole with a thin accretion disk on a black background: blackbody color
from the temperature profile, relativistic Doppler and gravitational shifts, a
procedural gas texture, and bloom. Set SPIN = 0 for a Schwarzschild hole (fast
path) or SPIN in (0, 1) for a rotating Kerr hole -- the real Gargantua, with an
asymmetric shadow, frame dragging and a disk reaching down to the ISCO.
"""

from src.camera import Camera
from src.renderer import render_disk_image, render_kerr_image, save_png
from src.disk import Disk
from src.kerr import isco

MASS = 1.0
SPIN = 0.999          # 0 = Schwarzschild, up to ~0.999 = near-extremal Kerr
MODE = "beautiful"
SUPERSAMPLE = 2


def main():
    if SPIN <= 0.0:
        camera = Camera(distance=33.0, resolution=(720, 480),
                        fov_deg=52.0, inclination_deg=10.0)
        disk = Disk(inner=6.0 * MASS, outer=14.0 * MASS)
        image = render_disk_image(camera, MASS, disk, mode=MODE,
                                  t_peak=5000.0, supersample=SUPERSAMPLE)
    else:
        camera = Camera(distance=33.0, resolution=(720, 480),
                        fov_deg=40.0, inclination_deg=15.0)
        disk = Disk(inner=isco(SPIN), outer=14.0)
        image = render_kerr_image(camera, SPIN, disk, mode=MODE,
                                  t_peak=5000.0, supersample=SUPERSAMPLE)
    save_png(image, "out/disk.png")


if __name__ == "__main__":
    main()
