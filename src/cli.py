"""Command-line arguments shared by main.py (snapshots) and animate.py
(animations), so any framing that works for a still can be animated with
the same keys."""

from .camera import Camera


def add_camera_args(p):
    p.add_argument("--distance", type=float, default=74.1,
                   help="camera Boyer-Lindquist radius in M (default 74.1)")
    p.add_argument("--inclination", type=float, default=3.44,
                   help="camera elevation above the disk plane, degrees "
                        "(90 = looking straight down the spin axis)")
    p.add_argument("--fov", type=float, default=19.0,
                   help="vertical field of view, degrees (default 19)")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=500)
    p.add_argument("--supersample", type=int, default=3,
                   help="anti-aliasing factor")
    p.add_argument("--aim-x", type=float, default=0.0,
                   help="aim the view this many degrees right of the hole")
    p.add_argument("--aim-y", type=float, default=0.0,
                   help="aim the view this many degrees above the hole")
    p.add_argument("--roll", type=float, default=0.0,
                   help="camera roll about the line of sight, degrees")


def add_look_args(p):
    p.add_argument("--mode", choices=("beautiful", "accurate"),
                   default="beautiful",
                   help="beautiful = movie look (no frequency shifts, "
                        "veiling glow); accurate = full Doppler physics")
    p.add_argument("--t-peak", type=float, default=4500.0,
                   help="disk temperature at the anchor radius, K")
    p.add_argument("--flat-temp", action="store_true",
                   help="uniform-temperature disk (the article's model) "
                        "instead of the r^-0.45 profile")
    p.add_argument("--saturation", type=float, default=None,
                   help="tone-map colour saturation (default: 0.8 beautiful,"
                        " 1.0 accurate)")
    p.add_argument("--bloom", type=float, default=None,
                   help="glow strength (default: 0.75 beautiful, 0.4 "
                        "accurate)")
    p.add_argument("--outer", type=float, default=18.7,
                   help="disk outer radius in M (default 18.7)")
    p.add_argument("--inner", type=float, default=None,
                   help="disk inner radius in M (default: the ISCO)")
    p.add_argument("--max-steps", type=int, default=9000,
                   help="integration step budget per ray (pole-on views "
                        "need ~40000: near-axis rays step very finely)")


def camera_from_args(args):
    return Camera(distance=args.distance,
                  resolution=(args.width, args.height),
                  fov_deg=args.fov, inclination_deg=args.inclination,
                  aim_deg=(args.aim_x, args.aim_y), roll_deg=args.roll)


def look_kwargs_from_args(args):
    """Keyword arguments for shade_kerr_scene / render_kerr_image."""
    return dict(mode=args.mode, t_peak=args.t_peak,
                temp_profile="constant" if args.flat_temp else "powerlaw",
                saturation=args.saturation, bloom_strength=args.bloom)
