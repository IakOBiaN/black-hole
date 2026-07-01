"""Render the current scene to out/.

Phase 2: a Schwarzschild black hole with a thin accretion disk on a black
background. The disk now glows with a blackbody color set by its radial
temperature profile.
"""

from src.camera import Camera
from src.renderer import render_buffers, shade_disk, save_png
from src.disk import Disk

MASS = 1.0
MODE = "accurate"


def main():
    camera = Camera(distance=30.0, resolution=(320, 320),
                    fov_deg=52.0, inclination_deg=10.0)
    disk = Disk(inner=6.0 * MASS, outer=14.0 * MASS)

    radii, bz = render_buffers(camera, MASS, disk)
    image = shade_disk(radii, bz, disk, MASS, t_peak=6500.0, mode=MODE)
    save_png(image, "out/disk.png")


if __name__ == "__main__":
    main()
