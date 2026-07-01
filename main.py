"""Render the current scene to out/.

Phase 2: a Schwarzschild black hole with a thin accretion disk on a black
background. The disk now glows with a blackbody color set by its radial
temperature profile.
"""

from src.camera import Camera
from src.tracer import trace_batch
from src.renderer import shade_disk, save_png
from src.disk import Disk

MASS = 1.0
MODE = "accurate"


def main():
    camera = Camera(distance=35.0, resolution=(1920, 1080),
                    fov_deg=52.0, inclination_deg=10.0)
    disk = Disk(inner=6.0 * MASS, outer=14.0 * MASS)

    radii, bz, azimuth = trace_batch(camera, MASS, disk)
    image = shade_disk(radii, bz, disk, MASS, t_peak=6500.0, mode=MODE)
    save_png(image, "out/disk.png")


if __name__ == "__main__":
    main()
