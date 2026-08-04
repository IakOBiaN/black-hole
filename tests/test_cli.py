from black_hole.snapshot import parse_args


def test_default_render_settings_are_unchanged():
    args = parse_args([])
    assert args.preset is None
    assert (args.width, args.height) == (1100, 500)
    assert args.supersample == 3
    assert args.max_steps == 9000
    assert args.out == "out/disk.png"


def test_preview_preset_uses_fast_render_settings():
    args = parse_args(["--preset", "preview"])
    assert (args.width, args.height) == (720, 480)
    assert args.supersample == 1
    assert args.max_steps == 6000
    assert args.out == "out/preview.png"


def test_explicit_flags_override_preview_preset():
    args = parse_args([
        "--preset", "preview",
        "--width", "800",
        "--supersample", "2",
        "--out", "out/custom.png",
    ])
    assert args.width == 800
    assert args.height == 480
    assert args.supersample == 2
    assert args.out == "out/custom.png"
