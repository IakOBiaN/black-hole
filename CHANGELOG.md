# Changelog

This file records the user-visible changes in each release.

## [0.1.0] - 2026-08-04

The first public release of the cinematic Kerr black hole renderer.

### Rendering

- Trace light through Kerr spacetime for rotating black holes, with the
  Schwarzschild case available by setting the spin to zero.
- Resolve multiple crossings of the equatorial disk to form higher-order
  lensed images of its far side and underside.
- Render a procedural accretion disk with differential orbital motion,
  radial temperature variation, fine filaments, gaps and sparse debris.
- Choose between the cinematic `beautiful` mode and the frequency-shifted,
  relativistically beamed `accurate` mode.
- Apply supersampling, bloom and filmic tone mapping to finished frames.

### Images and animation

- Render configurable PNG snapshots with camera, disk and appearance controls.
- Create time, camera-orbit and spin animations as GIF or H.264 MP4 files.
- Reuse traced geometry across animation frames when the spacetime is fixed.
- Start quickly with the 720x480 `preview` preset.

### Installation and exploration

- Install the project as the `black-hole-renderer` Python package.
- Use the `black-hole` and `black-hole-animate` console commands.
- Explore an interactive Google Colab notebook with a saved example result.

### Verification and scope

- Validate capture boundaries, Kerr equations, the Schwarzschild frequency
  shift limit, Numba and NumPy tracer parity, step convergence and formation
  of multiple images.
- Run the automated test and validation suites on every push and pull request.
- Document the numerical accuracy and the limits of the idealized thin-disk
  model in `ACCURACY.md`.

[0.1.0]: https://github.com/IakOBiaN/black-hole/releases/tag/v0.1.0
