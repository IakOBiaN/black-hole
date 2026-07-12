"""Render the current scene to out/.

The Interstellar shot: a Kerr black hole with the artist-style accretion
disk of James et al. (2015) -- physically thin, marginally optically thick,
filamentary, with a ragged outer edge -- lensed by backward ray tracing
through the Kerr metric. Geometry follows the paper's Fig. 15: spin
a/M = 0.6 (Nolan and Franklin slowed Gargantua's spin for visual clarity),
camera at r = 74.1M, theta = 86.56 degrees (3.44 degrees above the disk
plane), disk at a constant 4500 K reaching in to the ISCO (3.83M). The
inner radius is fitted to the paper's Fig. 15a: its lensed dome and front
band only match if the disk extends to near-ISCO radii (the 9.26M-18.7M
annulus quoted in Fig. 13's caption is the pedagogical paint-swatch disk,
not this one), and the ISCO orbital speed ~0.55c is exactly the disk speed
the paper quotes.

MODE "beautiful" reproduces the movie treatment: no Doppler/gravitational
colour or brightness shifts (Fig. 15a) plus a soft veiling glow standing in
for IMAX lens flare (Fig. 16). MODE "accurate" turns on the physically
complete frequency shifts (Fig. 15c): blue and bright on the approaching
side, red and dim on the receding side.
"""

from src.camera import Camera
from src.renderer import render_kerr_image, save_png
from src.disk import Disk
from src.kerr import isco

SPIN = 0.6
DISK = Disk(inner=isco(SPIN), outer=18.7)
CAMERA = Camera(distance=74.1, resolution=(1100, 500), fov_deg=19.0,
                inclination_deg=3.44)

MODE = "beautiful"          # "beautiful" (movie look) or "accurate"
SUPERSAMPLE = 3


def main():
    image = render_kerr_image(CAMERA, SPIN, DISK, mode=MODE,
                              supersample=SUPERSAMPLE, dzeta=0.07,
                              max_steps=9000)
    save_png(image, "out/disk.png")


if __name__ == "__main__":
    main()
