from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from scipy.optimize import minimize

DEFAULT_WAVELENGTH = "CuKa"


def normalize_0_100(y):
    y = np.asarray(y, dtype=float)
    y = y - y.min()
    return 100.0 * y / np.clip(y.max(), 1e-12, None)


def load_experimental_profile(xy_file, min_angle=10.0, max_angle=80.0, baseline_percentile=5.0):
    data = np.loadtxt(xy_file)
    two_theta = data[:, 0]
    intensity = data[:, 1]

    keep = (two_theta >= min_angle) & (two_theta <= max_angle)
    two_theta = two_theta[keep]
    intensity = intensity[keep]

    intensity = np.clip(intensity - np.percentile(intensity, baseline_percentile), 0.0, None)
    intensity = normalize_0_100(intensity)
    return two_theta, intensity


def load_reference_structures(cif_files):
    return {Path(cif).stem: Structure.from_file(cif) for cif in cif_files}


def make_calculator(wavelength=DEFAULT_WAVELENGTH):
    return XRDCalculator(wavelength=wavelength)


def apply_lattice_scales(structure, scales):
    strained = structure.copy()
    strained.apply_strain(np.asarray(scales, dtype=float) - 1.0)
    return strained


def get_stick_pattern(structure, calculator, min_angle=10.0, max_angle=80.0, intensity_threshold=1.0):
    pattern = calculator.get_pattern(structure, two_theta_range=(min_angle, max_angle))
    peak_pos = np.asarray(pattern.x, dtype=float)
    peak_int = np.asarray(pattern.y, dtype=float)
    keep = peak_int >= intensity_threshold
    return peak_pos[keep], normalize_0_100(peak_int[keep])


def simulate_profile(two_theta_grid, peak_pos, peak_intensity, fwhm=0.30, gauss_frac=0.2):
    if len(peak_pos) == 0:
        return np.zeros_like(two_theta_grid)

    dx = two_theta_grid[:, None] - peak_pos[None, :]
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0

    gauss = np.exp(-0.5 * (dx / sigma) ** 2)
    lorentz = (gamma**2) / (dx**2 + gamma**2)

    profile = ((1.0 - gauss_frac) * gauss + gauss_frac * lorentz) @ peak_intensity
    return normalize_0_100(profile)


def fit_scale_only(y_obs, y_phase, y_bg):
    y_model = y_phase + y_bg
    scale = float(np.dot(y_obs, y_model) / np.clip(np.dot(y_model, y_model), 1e-12, None))
    return scale, np.clip(scale * y_phase + y_bg, 0.0, None)


def fit_background_and_scale(y_obs, y_phase, x_scaled, degree=6):
    X = np.column_stack([x_scaled**k for k in range(degree + 1)] + [y_phase])
    coeffs, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
    bg_coeffs = coeffs[: degree + 1]
    phase_scale = coeffs[-1]

    y_bg = np.clip(sum(bg_coeffs[k] * x_scaled**k for k in range(degree + 1)), 0.0, None)
    y_fit = np.clip(phase_scale * y_phase + y_bg, 0.0, None)
    return phase_scale, bg_coeffs, y_bg, y_fit


def compute_rwp(y_obs, y_calc):
    w = 1.0 / np.clip(y_obs, 1.0, None)
    numer = np.sum(w * (y_obs - y_calc) ** 2)
    denom = np.sum(w * y_obs**2)
    return 100.0 * np.sqrt(numer / np.clip(denom, 1e-12, None))


def pearson_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def refine_phase_sequential(
    two_theta,
    y_obs,
    base_structure,
    calculator,
    *,
    min_angle=10.0,
    max_angle=80.0,
    intensity_threshold=1.0,
    background_degree=6,
    fwhm_init=0.30,
    gauss_frac=0.2,
    lattice_scale_bounds=(0.98, 1.02),
    fwhm_bounds=(0.05, 1.20),
    lattice_maxiter=60,
    width_maxiter=50,
):
    x_scaled = 2.0 * (two_theta - min_angle) / (max_angle - min_angle) - 1.0

    peak_pos_0, peak_int_0 = get_stick_pattern(
        base_structure,
        calculator,
        min_angle=min_angle,
        max_angle=max_angle,
        intensity_threshold=intensity_threshold,
    )
    y_phase_0 = simulate_profile(two_theta, peak_pos_0, peak_int_0, fwhm=fwhm_init, gauss_frac=gauss_frac)
    _, bg_coeffs, y_bg, y_fit_1 = fit_background_and_scale(y_obs, y_phase_0, x_scaled, degree=background_degree)

    def lattice_objective(scales):
        s = np.clip(np.asarray(scales, dtype=float), *lattice_scale_bounds)
        refined_structure = apply_lattice_scales(base_structure, s)
        peak_pos, peak_int = get_stick_pattern(
            refined_structure,
            calculator,
            min_angle=min_angle,
            max_angle=max_angle,
            intensity_threshold=intensity_threshold,
        )
        y_phase = simulate_profile(two_theta, peak_pos, peak_int, fwhm=fwhm_init, gauss_frac=gauss_frac)
        _, y_fit = fit_scale_only(y_obs, y_phase, y_bg)
        return np.mean((y_obs - y_fit) ** 2)

    res_lat = minimize(
        lattice_objective,
        x0=np.array([1.0, 1.0, 1.0]),
        method="Powell",
        bounds=[lattice_scale_bounds, lattice_scale_bounds, lattice_scale_bounds],
        options={"maxiter": lattice_maxiter, "xtol": 1e-3, "ftol": 1e-3},
    )
    best_scales = np.clip(np.asarray(res_lat.x, dtype=float), *lattice_scale_bounds)

    structure_2 = apply_lattice_scales(base_structure, best_scales)
    peak_pos_2, peak_int_2 = get_stick_pattern(
        structure_2,
        calculator,
        min_angle=min_angle,
        max_angle=max_angle,
        intensity_threshold=intensity_threshold,
    )
    y_phase_2 = simulate_profile(two_theta, peak_pos_2, peak_int_2, fwhm=fwhm_init, gauss_frac=gauss_frac)
    _, y_fit_2 = fit_scale_only(y_obs, y_phase_2, y_bg)

    def width_objective(width):
        w = float(np.clip(width[0], *fwhm_bounds))
        y_phase = simulate_profile(two_theta, peak_pos_2, peak_int_2, fwhm=w, gauss_frac=gauss_frac)
        _, y_fit = fit_scale_only(y_obs, y_phase, y_bg)
        return np.mean((y_obs - y_fit) ** 2)

    res_w = minimize(
        width_objective,
        x0=np.array([fwhm_init]),
        method="Powell",
        bounds=[fwhm_bounds],
        options={"maxiter": width_maxiter, "xtol": 1e-3, "ftol": 1e-3},
    )
    best_fwhm = float(np.clip(res_w.x[0], *fwhm_bounds))

    y_phase_3 = simulate_profile(two_theta, peak_pos_2, peak_int_2, fwhm=best_fwhm, gauss_frac=gauss_frac)
    scale_3, y_fit_3 = fit_scale_only(y_obs, y_phase_3, y_bg)

    return {
        "bg_coeffs": bg_coeffs,
        "scales": best_scales,
        "fwhm": best_fwhm,
        "scale": scale_3,
        "y_bg": y_bg,
        "y_fit_step1": y_fit_1,
        "y_fit_step2": y_fit_2,
        "y_fit_final": y_fit_3,
        "rwp": compute_rwp(y_obs, y_fit_3),
        "pearson": pearson_corr(y_obs, y_fit_3),
    }
