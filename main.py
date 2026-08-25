"""Compatibility wrapper for running snapshots from a source checkout."""

import _bootstrap  # noqa: F401  (binds black_hole to ./src)
from black_hole.snapshot import PRESETS, build_parser, main, parse_args


if __name__ == "__main__":
    main()
