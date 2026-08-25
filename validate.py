"""Run quantitative validation checks for the renderer.

Unlike the unit test suite, this command prints the measured numerical error
for the main physics claims.  It is intentionally deterministic so reports
from different machines and future versions can be compared directly.

Usage:

    python validate.py
    python validate.py --json
"""

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

import _bootstrap  # noqa: F401  (binds black_hole to ./src)
from black_hole.camera import Camera
from black_hole.disk import Disk
from black_hole.kerr import (
    equatorial_ray_status,
    isco,
    kerr_redshift_factor,
    photon_orbit_b,
    photon_orbit_radii,
    rhs,
    rhs_numerical,
)
from black_hole.kerr_numba import trace_batch_kerr_multi
from black_hole.kerr_numba import trace_batch_kerr as trace_numba
from black_hole.kerr_tracer import trace_batch_kerr as trace_numpy


@dataclass
class Result:
    check: str
    measurement: str
    value: float
    limit: float
    unit: str
    operator: str
    passed: bool


def result(check, measurement, value, limit, unit="", lower_is_better=True):
    passed = value <= limit if lower_is_better else value >= limit
    operator = "<=" if lower_is_better else ">="
    return Result(check, measurement, float(value), float(limit), unit,
                  operator, bool(passed))


def _capture_transition(expected, spin):
    """Locate an equatorial capture/escape boundary around an exact value."""
    # A narrow bracket avoids unrelated turning-point structure farther from
    # the critical orbit. The small step is important for the retrograde
    # boundary of a near-extremal hole.
    lo, hi = expected - 0.05, expected + 0.05
    kwargs = dict(r0=100.0, dzeta=0.02, max_steps=200000)
    lo_status = equatorial_ray_status(lo, spin, **kwargs)
    hi_status = equatorial_ray_status(hi, spin, **kwargs)
    if lo_status == hi_status:
        raise RuntimeError(f"capture boundary not bracketed near b={expected}")

    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if equatorial_ray_status(mid, spin, **kwargs) == lo_status:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def validate_capture_boundaries():
    errors = []
    exact_schwarzschild = 3.0 * np.sqrt(3.0)
    measured = _capture_transition(exact_schwarzschild, 0.0)
    errors.append(abs(measured - exact_schwarzschild))

    for spin in (0.6, 0.9, 0.998):
        prograde_r, retrograde_r = photon_orbit_radii(spin)
        for radius in (prograde_r, retrograde_r):
            exact = photon_orbit_b(radius, spin)
            errors.append(abs(_capture_transition(exact, spin) - exact))

    return result(
        "Kerr geodesics",
        "maximum equatorial capture-boundary error",
        max(errors),
        5.0e-4,
        "M",
    )


def validate_hamiltonian_rhs():
    rng = np.random.default_rng(20260802)
    worst = 0.0
    for spin in (0.0, 0.6, 0.9, 0.998):
        for _ in range(100):
            state = np.array([
                rng.uniform(1.3, 30.0),
                rng.uniform(0.3, np.pi - 0.3),
                0.0,
                rng.uniform(-3.0, 3.0),
                rng.uniform(-3.0, 3.0),
            ])
            b = rng.uniform(-8.0, 8.0)
            q = rng.uniform(0.0, 20.0)
            analytic = rhs(state, b, q, spin)
            finite_difference = rhs_numerical(state, b, q, spin)
            scale = np.maximum(np.abs(finite_difference), 1.0e-6)
            worst = max(worst, float(np.max(np.abs(analytic - finite_difference) / scale)))

    return result(
        "Kerr equations",
        "maximum relative analytic-vs-finite-difference RHS error",
        worst,
        2.0e-3,
    )


def validate_redshift_limit():
    radii = np.linspace(6.1, 30.0, 80)
    b = np.linspace(-4.0, 4.0, radii.size)
    measured = kerr_redshift_factor(radii, b, 0.0)
    exact = np.sqrt(1.0 - 3.0 / radii) / (1.0 - b / radii ** 1.5)
    relative_error = np.max(np.abs(measured - exact) / np.abs(exact))
    return result(
        "Frequency shift",
        "maximum error in the analytic Schwarzschild limit",
        relative_error,
        1.0e-12,
    )


def validate_numba_parity():
    spin = 0.9
    camera = Camera(distance=25.0, resolution=(48, 36), fov_deg=40.0,
                    inclination_deg=15.0)
    disk = Disk(isco(spin), 14.0)
    r_numba, _, phi_numba, captured_numba = trace_numba(camera, spin, disk)
    r_numpy, _, phi_numpy, captured_numpy = trace_numpy(camera, spin, disk)

    if not np.array_equal(captured_numba, captured_numpy):
        mismatch = np.count_nonzero(captured_numba != captured_numpy)
        return result("Implementation parity", "Numba/Numpy mask mismatches",
                      mismatch, 0.0, "pixels")

    common = ~np.isnan(r_numba) & ~np.isnan(r_numpy)
    error = max(
        float(np.max(np.abs(r_numba[common] - r_numpy[common]))),
        float(np.max(np.abs(phi_numba[common] - phi_numpy[common]))),
    )
    return result(
        "Implementation parity",
        "maximum Numba/Numpy hit-coordinate difference",
        error,
        1.0e-6,
    )


def validate_step_convergence():
    spin = 0.9
    camera = Camera(distance=25.0, resolution=(96, 64), fov_deg=34.0,
                    inclination_deg=12.0)
    no_disk = Disk(1000.0, 1001.0)
    reference = trace_numba(camera, spin, no_disk, dzeta=0.04,
                            max_steps=12000)[3]
    candidate = trace_numba(camera, spin, no_disk, dzeta=0.07,
                            max_steps=9000)[3]
    disagreement = np.count_nonzero(reference != candidate) / reference.size
    return result(
        "Step convergence",
        "shadow-mask disagreement for dzeta 0.07 vs 0.04",
        disagreement,
        2.0e-3,
        "fraction of pixels",
    )


def validate_multiple_images():
    spin = 0.6
    camera = Camera(distance=40.0, resolution=(64, 40), fov_deg=40.0,
                    inclination_deg=10.0)
    disk = Disk(isco(spin), 14.0)
    _, _, hit_count, _, _ = trace_batch_kerr_multi(
        camera, spin, disk.inner, disk.outer, max_hits=6)
    multi_hit_pixels = np.count_nonzero(hit_count >= 2)
    return result(
        "Multiple images",
        "pixels with two or more equatorial disk crossings",
        multi_hit_pixels,
        5.0,
        "pixels",
        lower_is_better=False,
    )


def run_validation():
    return [
        validate_capture_boundaries(),
        validate_hamiltonian_rhs(),
        validate_redshift_limit(),
        validate_numba_parity(),
        validate_step_convergence(),
        validate_multiple_images(),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure the numerical accuracy of the Kerr renderer.")
    parser.add_argument("--json", action="store_true",
                        help="write machine-readable JSON")
    args = parser.parse_args(argv)

    results = run_validation()
    if args.json:
        print(json.dumps([asdict(item) for item in results], indent=2))
    else:
        print("Black-hole renderer validation")
        print("=" * 30)
        for item in results:
            status = "PASS" if item.passed else "FAIL"
            suffix = f" {item.unit}" if item.unit else ""
            print(f"[{status}] {item.check}: {item.measurement}")
            print(f"       {item.value:.8g}{suffix} (limit {item.operator} "
                  f"{item.limit:.8g}{suffix})")

    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
