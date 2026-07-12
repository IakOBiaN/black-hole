"""Reproduce Figures 15a-c of James et al. (2015) for physics validation.

All three frames share the lensing geometry of main.py (a/M = 0.6, disk
9.26M-18.7M at 4500 K, camera at 74.1M, 3.44 degrees above the disk plane)
and differ only in the treatment of the Doppler + gravitational frequency
shift, exactly as in the paper:

  fig15a  no shifts                     -> out/fig15a.png
  fig15b  colour (hue) shift only       -> out/fig15b.png
  fig15c  colour and brightness shifts  -> out/fig15c.png
"""

from main import CAMERA, DISK, SPIN, SUPERSAMPLE
from src.renderer import render_kerr_image, save_png

for name, shift in (("fig15a", "none"), ("fig15b", "hue"), ("fig15c", "full")):
    image = render_kerr_image(CAMERA, SPIN, DISK, mode="accurate",
                              shift_mode=shift, supersample=SUPERSAMPLE,
                              bloom_strength=0.25)
    save_png(image, f"out/{name}.png")
    print(f"saved out/{name}.png")
