"""Render the current scene to out/.

Phase 2: a Schwarzschild black hole with a thin accretion disk on a black
background. The disk is drawn in a flat color for now; physical temperature,
color and relativistic effects come next.
"""

from src.camera import Camera
from src.renderer import render_disk, save_png
from src.disk import Disk

MASS = 1.0


def main():
    camera = Camera(distance=30.0, resolution=(320, 320),
                    fov_deg=52.0, inclination_deg=10.0)
    disk = Disk(inner=6.0 * MASS, outer=14.0 * MASS)
    image = render_disk(camera, MASS, disk)
    save_png(image, "out/disk.png")


if __name__ == "__main__":
    main()
