# black-hole

A physically accurate, modern, and visually compelling black hole renderer.

The goal is an image on par with the black hole in *Interstellar*, built from
first principles: tracing light rays through curved spacetime and rendering the
gravitationally lensed view a camera near the hole would actually see.

Two rendering modes are planned throughout:

- **accurate** — physically faithful (no artistic adjustments)
- **beautiful** — cinematic tuning on top of the accurate result

## Roadmap

0. Geodesic prototype — integrate null geodesics in the Schwarzschild metric.
1. Schwarzschild image — backward ray tracing with a lensed starfield.
2. Accretion disk — temperature profile, blackbody color, Doppler beaming,
   gravitational and Doppler redshift.
3. GPU port for speed.
4. Kerr (rotating) black hole — Gargantua.
5. Cinematic finish — bloom/glare, lensed background.

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
