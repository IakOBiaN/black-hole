# Accuracy and model scope

This renderer is physically grounded where light propagation and frequency
shift are concerned, but it is not a complete accretion-flow simulation. This
document separates equations taken from general relativity from numerical
approximations and deliberate artistic choices.

The short version is:

> The Kerr spacetime, null-geodesic lensing, circular-orbit frequency shifts,
> and relativistic beaming are physical. The accretion-disk material and the
> final cinematic grade are artist-directed.

## What is model-derived

- **Kerr spacetime.** The spin parameter is dimensionless, with geometrized
  units `G = c = M = 1`. The horizon, frame-dragging angular velocity and
  prograde ISCO are evaluated from the standard Kerr expressions.
- **Null geodesics.** Camera rays are converted to photon momenta in a FIDO
  orthonormal frame and integrated backward through the Kerr metric using the
  super-Hamiltonian equations in James et al. (2015), Appendix A.
- **Multiple images.** A ray can record up to eight crossings of the
  equatorial plane. This produces the direct view and higher-order lensed
  images of the disk instead of painting a halo around the shadow.
- **Orbital motion.** Disk material follows the angular velocity of prograde
  circular equatorial Kerr geodesics, `Omega = 1 / (r^(3/2) + a)`.
- **Frequency shift.** The observed/emitted frequency ratio includes the
  emitter's orbital motion, gravitational shift and the finite-radius FIDO
  observer. In `full` shift mode, specific intensity follows the Liouville
  scaling `I_obs = g^4 I_emit`.
- **Blackbody colour.** A Planck spectrum is integrated against an
  approximation to the CIE 1931 colour-matching functions and converted to
  linear sRGB.

These statements describe the model implemented by the code. They do not mean
that the rendered disk is a prediction of what a particular astrophysical
black hole must look like.

## Numerical approximations

- Rays are point samples from a pinhole camera. Anti-aliasing uses regular
  supersampling; the renderer does not propagate the elliptical ray bundles
  used by DNGR for IMAX-quality filtering.
- Geodesics use fourth-order Runge-Kutta integration. The affine step is
  reduced heuristically near the horizon and spin axis, but it is not an
  embedded, error-controlled adaptive RK method.
- Boyer-Lindquist coordinates are singular on the spin axis and at the
  horizon. The tracer reflects coordinates across a pole, shortens the step
  near singular regions and terminates just outside the horizon.
- Equatorial crossing positions are linearly interpolated between integration
  steps.
- The camera is a stationary FIDO. Camera velocity, relativistic aberration,
  lens distortion and exposure time are not modelled.
- Numerical precision is IEEE 754 double precision on the CPU. Numba uses
  `fastmath=True`, so insignificant last-bit differences may occur across
  platforms.

The default `dzeta=0.07` is a quality/performance choice. Pole-on views require
many more steps because rays pass close to the coordinate axis.

## Artist-directed choices

- The disk is a geometrically thin, procedurally textured sheet, not the
  result of a general-relativistic magnetohydrodynamic simulation.
- Filaments, ring gaps, debris, the ragged outer edge and material opacity are
  procedural art controls.
- The default radial temperature falloff used for the gallery is an artistic
  power law. A thin-disk-shaped profile is also implemented, but neither mode
  performs full plasma radiative transfer.
- `beautiful` mode deliberately suppresses relativistic frequency shifts and
  adds stronger bloom, following the aesthetic choice discussed for
  *Interstellar*. `accurate` mode restores the implemented shift and beaming
  physics, but retains the procedural disk.
- Bloom, exposure, saturation, highlight desaturation and tone mapping are
  display transforms rather than spacetime physics.
- The current scene has a black background. Light from a celestial sphere or
  star field is not yet traced.

## Not currently modelled

- volumetric emission, absorption and scattering;
- magnetohydrodynamics, turbulence derived from plasma evolution or jets;
- synchrotron spectra and polarization transport;
- returning radiation that changes the disk state;
- arbitrary moving observers and finite shutter-time ray bundles;
- lensing of a background sky;
- black-hole charge, non-Kerr metrics or dynamical spacetime.

Research GRRT tools such as GYOTO, Odyssey, AART and Mahakala cover some of
these areas. This project instead aims to be a compact, readable CPU renderer
for reproducible Kerr-lensing images and educational experiments.

## Reproducing the validation

Install the normal project dependencies and run:

```bash
python validate.py
pytest tests -q
```

`validate.py` exits with a non-zero status if a measured error exceeds its
documented tolerance. It currently checks:

1. numerical capture boundaries against analytic equatorial photon orbits for
   spins `a = 0, 0.6, 0.9, 0.998`;
2. analytic Hamiltonian derivatives against finite differences;
3. the Kerr redshift expression against its analytic Schwarzschild limit;
4. pixel-level agreement between the Numba tracer and the NumPy reference;
5. shadow-mask convergence when the integration step is reduced;
6. the presence of rays that form multiple lensed disk images.

### Reference result

The following baseline was produced on 2026-08-02 with Python 3.12,
NumPy 2.4.6 and Numba 0.66.0:

| Check | Measured value | Acceptance limit |
|---|---:|---:|
| Maximum equatorial capture-boundary error | `1.2207e-5 M` | `<= 5e-4 M` |
| Analytic vs finite-difference Kerr RHS | `2.4662e-5` relative | `<= 2e-3` |
| Schwarzschild-limit redshift error | `3.5117e-16` relative | `<= 1e-12` |
| Numba vs NumPy hit-coordinate difference | `4.9738e-14` | `<= 1e-6` |
| Shadow-mask disagreement, `dzeta=0.07` vs `0.04` | `0 / 6144` pixels | `<= 0.2%` |
| Pixels with two or more disk crossings | `18 / 2560` pixels | `>= 5` |

These are regression baselines at deliberately modest image resolutions, not
a claim of global error bounds for every camera, spin and integration budget.
Near-critical rays and near-extremal spins should be checked separately when
changing the integrator.

Use `python validate.py --json` to record machine-readable results in a CI job
or benchmark archive. The test suite additionally checks the Schwarzschild
critical impact parameter, Kerr frame-dragging direction, Doppler sign,
`g^4` intensity scaling, disk temperature, texture continuity and rendering
post-processing.

## Interpretation

Passing these checks supports the narrower claim that the implemented Kerr
geodesics and relativistic shading behave consistently with their analytic
limits at the renderer's working resolution. It is not evidence that the
procedural disk is an astrophysically complete model, nor does it establish
the renderer as a replacement for a research-grade GRRT code.

## References

- O. James, E. von Tunzelmann, P. Franklin, K. S. Thorne, *Gravitational
  lensing by spinning black holes in astrophysics, and in the movie
  Interstellar*, Classical and Quantum Gravity **32**, 065001 (2015),
  [doi:10.1088/0264-9381/32/6/065001](https://doi.org/10.1088/0264-9381/32/6/065001),
  [arXiv:1502.03808](https://arxiv.org/abs/1502.03808).
- J. M. Bardeen, W. H. Press, S. A. Teukolsky, *Rotating Black Holes:
  Locally Nonrotating Frames, Energy Extraction, and Scalar Synchrotron
  Radiation*, The Astrophysical Journal **178**, 347 (1972),
  [doi:10.1086/151796](https://doi.org/10.1086/151796).
