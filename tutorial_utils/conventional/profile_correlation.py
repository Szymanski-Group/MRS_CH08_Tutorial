from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure


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


def load_reference_stick_library(
    cif_files,
    min_angle=10.0,
    max_angle=80.0,
    wavelength="CuKa",
    intensity_threshold=1.0,
):
    calculator = XRDCalculator(wavelength=wavelength)
    refs = {}

    for cif_file in cif_files:
        pattern = calculator.get_pattern(Structure.from_file(cif_file), two_theta_range=(min_angle, max_angle))
        peak_pos = np.asarray(pattern.x, dtype=float)
        peak_intensity = np.asarray(pattern.y, dtype=float)

        keep = peak_intensity >= intensity_threshold
        peak_pos = peak_pos[keep]
        peak_intensity = normalize_0_100(peak_intensity[keep])

        refs[Path(cif_file).stem] = (peak_pos, peak_intensity)

    return refs


def simulate_continuous_profile(two_theta_grid, peak_pos, peak_intensity, fwhm=0.30, gauss_frac=0.2):
    if len(peak_pos) == 0:
        return np.zeros_like(two_theta_grid)

    dx = two_theta_grid[:, None] - peak_pos[None, :]

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0

    gauss = np.exp(-0.5 * (dx / sigma) ** 2)
    lorentz = (gamma**2) / (dx**2 + gamma**2)

    profile = ((1.0 - gauss_frac) * gauss + gauss_frac * lorentz) @ peak_intensity
    return normalize_0_100(profile)


def pearson_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_phases(exp_profile, exp_two_theta, reference_library, fwhm=0.30, gauss_frac=0.2):
    rows = []
    simulated_profiles = {}

    for phase, (peak_pos, peak_intensity) in reference_library.items():
        sim_profile = simulate_continuous_profile(
            exp_two_theta,
            peak_pos,
            peak_intensity,
            fwhm=fwhm,
            gauss_frac=gauss_frac,
        )
        rows.append(
            {
                "phase": phase,
                "pearson": pearson_corr(exp_profile, sim_profile),
                "cosine": cosine_similarity(exp_profile, sim_profile),
            }
        )
        simulated_profiles[phase] = sim_profile

    by_pearson = sorted(rows, key=lambda r: r["pearson"], reverse=True)
    by_cosine = sorted(rows, key=lambda r: r["cosine"], reverse=True)
    return by_pearson, by_cosine, simulated_profiles
