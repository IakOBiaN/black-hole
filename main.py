"""Render the current scene to out/.

A Schwarzschild black hole with a thin accretion disk on a black background:
blackbody color from the temperature profile, relativistic Doppler and
gravitational shifts, a procedural gas texture, and bloom.
"""

from src.camera import Camera
from src.renderer import render_disk_image, save_png
from src.disk import Disk

MASS = 1.0
MODE = "beautiful"
SUPERSAMPLE = 2


def main():
    camera = Camera(distance=33.0, resolution=(720, 480),
                    fov_deg=52.0, inclination_deg=10.0)
    disk = Disk(inner=6.0 * MASS, outer=14.0 * MASS)

    image = render_disk_image(camera, MASS, disk, mode=MODE,
                              t_peak=4000.0, supersample=SUPERSAMPLE)
    save_png(image, "out/disk.png")


if __name__ == "__main__":
    main()
