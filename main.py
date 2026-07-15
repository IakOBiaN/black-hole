"""Render a snapshot of the black hole to a PNG.

The Interstellar shot by default: a Kerr black hole (a/M = 0.6, the spin
Nolan and Franklin chose for visual clarity) with a thin, marginally
optically thick gas disk reaching in to the ISCO, seen from r = 74.1M just
3.44 degrees above the disk plane -- the geometry of Fig. 15a of James,
von Tunzelmann, Franklin & Thorne (2015). The disk dims and reddens toward
its edge (T ~ r^-0.45 blackbody) and its gas orbits differentially, so
--time moves the material around the hole.

Every parameter is a command-line flag, and the same keys work in
animate.py, so any framing found here can be animated as is:

    python main.py                             # the default frame
    python main.py --spin 0.998 --fov 24       # near-extremal Gargantua
    python main.py --time 150 --azimuth 90     # later, camera swung 90 deg
    python main.py --mode accurate             # full Doppler physics
    python main.py --fov 6.2 --aim-x -4 --roll 13   # close-up framing

Output goes to --out (default out/disk.png).
"""

import argparse

import numpy as np

from src.cli import (add_camera_args, add_look_args, camera_from_args,
                     look_kwargs_from_args)
from src.disk import Disk
from src.kerr import isco
from src.renderer import render_kerr_image, save_png


def build_parser():
    p = argparse.ArgumentParser(
        description="Render a Kerr black hole with its accretion disk.")
    p.add_argument("--spin", type=float, default=0.6,
                   help="black hole spin a/M in [0, 0.999] (default 0.6)")
    p.add_argument("--time", type=float, default=0.0,
                   help="disk time in M: the gas orbits differentially")
    p.add_argument("--azimuth", type=float, default=0.0,
                   help="camera azimuth around the spin axis, degrees")
    add_camera_args(p)
    add_look_args(p)
    p.add_argument("--out", default="out/disk.png")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    inner = args.inner if args.inner is not None else isco(args.spin)
    disk = Disk(inner=inner, outer=args.outer)
    image = render_kerr_image(
        camera_from_args(args), args.spin, disk, time=args.time,
        camera_azimuth=np.radians(args.azimuth),
        supersample=args.supersample, dzeta=0.07, max_steps=args.max_steps,
        **look_kwargs_from_args(args))
    save_png(image, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
