"""Render a snapshot of the black hole to a PNG.

The Interstellar shot by default: a Kerr black hole (a/M = 0.6, the spin
Nolan and Franklin chose for visual clarity) with a thin, marginally
optically thick gas disk reaching in to the ISCO, seen from r = 74.1M just
3.44 degrees above the disk plane -- the geometry of Fig. 15a of James,
von Tunzelmann, Franklin & Thorne (2015). The disk dims and reddens toward
its edge (T ~ r^-0.45 blackbody) and its gas orbits differentially, so
--time moves the material around the hole.

Every parameter is a command-line flag:

    python main.py                             # the default frame
    python main.py --spin 0.998 --fov 24       # near-extremal Gargantua
    python main.py --time 150 --azimuth 90     # later, camera swung 90 deg
    python main.py --mode accurate             # full Doppler physics

Output goes to --out (default out/disk.png).
"""

import argparse

import numpy as np

from src.camera import Camera
from src.disk import Disk
from src.kerr import isco
from src.renderer import render_kerr_image, save_png


def build_parser():
    p = argparse.ArgumentParser(
        description="Render a Kerr black hole with its accretion disk.")
    p.add_argument("--spin", type=float, default=0.6,
                   help="black hole spin a/M in [0, 0.999] (default 0.6)")
    p.add_argument("--distance", type=float, default=74.1,
                   help="camera Boyer-Lindquist radius in M (default 74.1)")
    p.add_argument("--inclination", type=float, default=3.44,
                   help="camera elevation above the disk plane, degrees")
    p.add_argument("--fov", type=float, default=19.0,
                   help="vertical field of view, degrees (default 19)")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=500)
    p.add_argument("--supersample", type=int, default=3,
                   help="anti-aliasing factor (default 3)")
    p.add_argument("--mode", choices=("beautiful", "accurate"),
                   default="beautiful",
                   help="beautiful = movie look (no frequency shifts, "
                        "veiling glow); accurate = full Doppler physics")
    p.add_argument("--time", type=float, default=0.0,
                   help="disk time in M: the gas orbits differentially")
    p.add_argument("--azimuth", type=float, default=0.0,
                   help="camera azimuth around the spin axis, degrees")
    p.add_argument("--inner", type=float, default=None,
                   help="disk inner radius in M (default: the ISCO)")
    p.add_argument("--outer", type=float, default=18.7,
                   help="disk outer radius in M (default 18.7)")
    p.add_argument("--t-peak", type=float, default=4500.0,
                   help="disk temperature at the inner edge, K")
    p.add_argument("--flat-temp", action="store_true",
                   help="uniform-temperature disk (the article's model) "
                        "instead of the r^-0.45 profile")
    p.add_argument("--out", default="out/disk.png")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    inner = args.inner if args.inner is not None else isco(args.spin)
    disk = Disk(inner=inner, outer=args.outer)
    camera = Camera(distance=args.distance,
                    resolution=(args.width, args.height),
                    fov_deg=args.fov, inclination_deg=args.inclination)
    image = render_kerr_image(
        camera, args.spin, disk, mode=args.mode, t_peak=args.t_peak,
        supersample=args.supersample, time=args.time,
        camera_azimuth=np.radians(args.azimuth),
        temp_profile="constant" if args.flat_temp else "powerlaw",
        dzeta=0.07, max_steps=9000)
    save_png(image, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
