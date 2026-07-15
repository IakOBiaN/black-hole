"""Render GIF animations of the black hole.

Three modes, and in every one of them time runs: the disk gas orbits the
hole differentially (inner gas laps the outer gas, shearing the pattern),
because the texture is sampled at the advected azimuth phi - Omega(r) t.

    time    fixed camera, the gas orbits            (default)
    orbit   the camera circles the hole while time runs
    spin    the spin ramps up while time runs

Camera and look keys are the same as in main.py, so any framing found with
a still can be animated unchanged, e.g.:

    python animate.py --anim time  --frames 60 --step 4
    python animate.py --anim time  --fov 6.2 --aim-x -4 --roll 13
    python animate.py --anim spin  --track-isco --spin-to 0.6

The Kerr metric is axisymmetric and static, so for "time" and "orbit" the
geometry is traced once and every frame is just a re-shade -- only "spin"
retraces per frame (the geometry itself changes with a).
"""

import argparse

import numpy as np
from PIL import Image

from src.cli import (add_camera_args, add_look_args, camera_from_args,
                     look_kwargs_from_args)
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
    p.add_argument("--spin-to", type=float, default=0.6,
                   help="spin mode: final a/M")
    p.add_argument("--orbit-degrees", type=float, default=360.0,
                   help="orbit mode: total camera sweep")
    p.add_argument("--track-isco", action="store_true",
                   help="spin mode: let the inner edge follow the shrinking "
                        "ISCO (near a/M=1 the blazing inner gas veils the "
                        "shadow)")
    add_camera_args(p)
    add_look_args(p)
    p.set_defaults(width=1000, height=450, supersample=2)
    p.add_argument("--out", default=None,
                   help="output file; .gif or .mp4 by extension (default "
                        "out/anim_<mode>.gif). MP4 is H.264: an order of "
                        "magnitude smaller than GIF and free of palette "
                        "banding")
    p.add_argument("--crf", type=int, default=18,
                   help="MP4 quality, lower = better/larger (default 18, "
                        "visually lossless ~17-20)")
    return p


def save_animation(frames, out, fps, crf):
    """Write frames to .mp4 (H.264 via imageio-ffmpeg) or .gif (dithered
    255-colour palette) depending on the extension."""
    if out.lower().endswith(".mp4"):
        import imageio

        with imageio.get_writer(out, fps=fps, codec="libx264",
                                quality=None, output_params=[
                                    "-crf", str(crf), "-preset", "slow",
                                    "-pix_fmt", "yuv420p"]) as writer:
            for f in frames:
                writer.append_data(np.asarray(f))
        return

    # Quantize with dithering: without it the GIF's 256-colour palette bands
    # the smooth dark glow into visible steps.
    frames = [f.quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                         dither=Image.Dither.FLOYDSTEINBERG)
              for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.out or f"out/anim_{args.anim}.gif"
    camera = camera_from_args(args)
    look = look_kwargs_from_args(args)

    frames = []
    if args.anim in ("time", "orbit"):
        inner = args.inner if args.inner is not None else isco(args.spin)
        disk = Disk(inner, args.outer)
        scene = trace_kerr_scene(camera, args.spin, disk,
                                 supersample=args.supersample,
                                 dzeta=0.07, max_steps=args.max_steps)
        sweep = np.radians(args.orbit_degrees) if args.anim == "orbit" else 0.0
        for i in range(args.frames):
            image = shade_kerr_scene(
                scene, args.spin, disk, time=i * args.step,
                camera_azimuth=sweep * i / max(args.frames, 1), **look)
            frames.append(Image.fromarray(image, "RGB"))
            print(f"frame {i + 1}/{args.frames}", end="\r")
    else:
        # Same disk throughout the sweep unless --track-isco pins the inner
        # edge to the shrinking ISCO.
        spins = np.linspace(args.spin_from, args.spin_to, args.frames)
        inner_fixed = (args.inner if args.inner is not None
                       else isco(min(args.spin_from, args.spin_to)))
        for i, a in enumerate(spins):
            disk = Disk(isco(a) if args.track_isco
                        else max(inner_fixed, isco(a)), args.outer)
            scene = trace_kerr_scene(camera, a, disk,
                                     supersample=args.supersample,
                                     dzeta=0.07, max_steps=args.max_steps)
            image = shade_kerr_scene(scene, a, disk, time=i * args.step,
                                     **look)
            frames.append(Image.fromarray(image, "RGB"))
            print(f"frame {i + 1}/{args.frames}", end="\r")
    print()

    save_animation(frames, out, args.fps, args.crf)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
