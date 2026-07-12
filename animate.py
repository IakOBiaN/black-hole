"""Render GIF animations of the black hole.

Three modes, and in every one of them time runs: the disk gas orbits the
hole differentially (inner gas laps the outer gas, shearing the pattern),
because the texture is sampled at the advected azimuth phi - Omega(r) t.

    time    fixed camera, the gas orbits            (default)
    orbit   the camera circles the hole while time runs
    spin    the spin ramps up while time runs

The Kerr metric is axisymmetric and static, so for "time" and "orbit" the
geometry is traced once and every frame is just a re-shade -- only "spin"
retraces per frame (the geometry itself changes with a).

    python animate.py --anim time  --frames 60 --step 4
    python animate.py --anim orbit --frames 72
    python animate.py --anim spin  --frames 40 --spin-from 0.0 --spin-to 0.998
"""

import argparse

import numpy as np
from PIL import Image

from src.camera import Camera
from src.disk import Disk
from src.kerr import isco
from src.renderer import trace_kerr_scene, shade_kerr_scene


def build_parser():
    p = argparse.ArgumentParser(description="Animate the black hole.")
    p.add_argument("--anim", choices=("time", "orbit", "spin"),
                   default="time")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--step", type=float, default=4.0,
                   help="coordinate time per frame, in M (default 4: the "
                        "ISCO gas at a=0.6 completes an orbit in ~13 frames)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--spin", type=float, default=0.6)
    p.add_argument("--spin-from", type=float, default=0.0,
                   help="spin mode: starting a/M")
    p.add_argument("--spin-to", type=float, default=0.998,
                   help="spin mode: final a/M")
    p.add_argument("--orbit-degrees", type=float, default=360.0,
                   help="orbit mode: total camera sweep")
    p.add_argument("--distance", type=float, default=74.1)
    p.add_argument("--inclination", type=float, default=3.44)
    p.add_argument("--fov", type=float, default=19.0)
    p.add_argument("--width", type=int, default=660)
    p.add_argument("--height", type=int, default=300)
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--mode", choices=("beautiful", "accurate"),
                   default="beautiful")
    p.add_argument("--outer", type=float, default=18.7)
    p.add_argument("--out", default=None,
                   help="output GIF (default out/anim_<mode>.gif)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.out or f"out/anim_{args.anim}.gif"
    camera = Camera(distance=args.distance,
                    resolution=(args.width, args.height),
                    fov_deg=args.fov, inclination_deg=args.inclination)

    frames = []
    if args.anim in ("time", "orbit"):
        disk = Disk(isco(args.spin), args.outer)
        scene = trace_kerr_scene(camera, args.spin, disk,
                                 supersample=args.supersample,
                                 dzeta=0.07, max_steps=9000)
        sweep = np.radians(args.orbit_degrees) if args.anim == "orbit" else 0.0
        for i in range(args.frames):
            image = shade_kerr_scene(
                scene, args.spin, disk, mode=args.mode,
                time=i * args.step,
                camera_azimuth=sweep * i / max(args.frames, 1))
            frames.append(Image.fromarray(image, "RGB"))
            print(f"frame {i + 1}/{args.frames}", end="\r")
    else:
        spins = np.linspace(args.spin_from, args.spin_to, args.frames)
        for i, a in enumerate(spins):
            disk = Disk(isco(a), args.outer)
            scene = trace_kerr_scene(camera, a, disk,
                                     supersample=args.supersample,
                                     dzeta=0.07, max_steps=9000)
            image = shade_kerr_scene(scene, a, disk, mode=args.mode,
                                     time=i * args.step)
            frames.append(Image.fromarray(image, "RGB"))
            print(f"frame {i + 1}/{args.frames}", end="\r")
    print()

    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
