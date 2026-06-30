"""Render the current scene to out/.

Phase 1: a Schwarzschild black hole shadow. The production background is
black; here a neutral gray is used so the shadow is visible for inspection.
"""

from src.camera import Camera
from src.renderer import render, save_png

MASS = 1.0


def main():
    camera = Camera(distance=30.0, resolution=(240, 240), fov_deg=28.0)
    image = render(camera, MASS, shadow_color=(0, 0, 0), miss_color=(40, 40, 48))
    save_png(image, "out/shadow.png")


if __name__ == "__main__":
    main()
