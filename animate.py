"""Compatibility wrapper for running animations from a source checkout."""

import _bootstrap  # noqa: F401  (binds black_hole to ./src)
from black_hole.animation import build_parser, main, save_animation


if __name__ == "__main__":
    main()
