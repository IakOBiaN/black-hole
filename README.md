# black-hole

A physically accurate, modern, and visually compelling black hole renderer.

The goal is an image on par with the black hole in *Interstellar*, built from
first principles: tracing light rays through curved spacetime and rendering the
gravitationally lensed view a camera near the hole would actually see.

Two rendering modes are available throughout:

- **accurate** — physically faithful (no artistic adjustments)
- **beautiful** — the movie treatment: no frequency shifts (as chosen by
  Nolan and Franklin), plus a soft veiling glow standing in for IMAX lens
  flare

The reference is James, von Tunzelmann, Franklin & Thorne (2015), *Gravitational
lensing by spinning black holes in astrophysics, and in the movie Interstellar*
(the DNGR paper, see `article/`). `main.py` renders the paper's Fig. 15/16
geometry: spin a/M = 0.6, disk from 9.26M to 18.7M at a constant 4500 K,
camera at r = 74.1M, 3.44° above the disk plane. `render_article_figs.py`
reproduces Figs. 15a–c (no shifts / hue shift only / full shifts) for
physics validation.

## What is implemented

- Backward ray tracing of the received photon's null geodesic through the
  Kerr metric in Boyer–Lindquist coordinates (super-Hamiltonian form,
  integrated backward in time per the DNGR prescription), Numba-compiled
  and parallelized.
- A physically thin, marginally optically thick artist-style disk: fine
  filaments stretched along the orbital flow, ring gaps, a ragged outer
  edge with sparse debris beyond it. Rays record every crossing of the
  equatorial plane and the lensed layers are composited front to back with
  the material's opacity.
- Blackbody shading with Doppler + gravitational frequency shift of disk
  material on prograde circular geodesics: colour shift and relativistic
  beaming (I ∝ g⁴), with the paper's three treatments selectable
  (`shift_mode = "none" | "hue" | "full"`).
- Multi-scale bloom and a filmic tone map.
- A Schwarzschild fast path (`src/tracer.py`) kept from the earlier
  roadmap stages.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

```
src/      source code
out/      rendered images (git-ignored)
```
