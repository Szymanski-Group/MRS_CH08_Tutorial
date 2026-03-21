from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from scipy.signal import find_peaks


def normalize_0_100(y):
    y = np.asarray(y, dtype=float)
    y = y - y.min()
    return 100.0 * y / np.clip(y.max(), 1e-12, None)


def load_pattern(xy_file, min_angle=10.0, max_angle=100.0):
    data = np.loadtxt(xy_file)
    tt, intensity = data[:, 0], data[:, 1]
    keep = (tt >= min_angle) & (tt <= max_angle)
    return tt[keep], normalize_0_100(intensity[keep])


def detect_peaks(
    tt,
    intensity,
    baseline_percentile=5.0,
    prominence_fraction=0.02,
    min_peak_distance_deg=0.22,
    max_detected_peaks=30,
):
    corrected = np.clip(intensity - np.percentile(intensity, baseline_percentile), 0.0, None)

    step = np.median(np.diff(tt))
    min_distance_pts = max(1, int(round(min_peak_distance_deg / step)))
    prominence = prominence_fraction * (corrected.max() - corrected.min())

    peak_idx, _ = find_peaks(corrected, prominence=prominence, distance=min_distance_pts)
    if len(peak_idx) > max_detected_peaks:
        keep = np.argsort(corrected[peak_idx])[::-1][:max_detected_peaks]
        peak_idx = peak_idx[keep]

    peak_idx = np.sort(peak_idx)
    return peak_idx, tt[peak_idx]


def load_reference_library(
    cif_files,
    min_angle=10.0,
    max_angle=100.0,
    wavelength="CuKa",
    intensity_threshold=1.0,
):
    calc = XRDCalculator(wavelength=wavelength)
    refs = {}
    for cif in cif_files:
        pattern = calc.get_pattern(Structure.from_file(cif), two_theta_range=(min_angle, max_angle))
        tt = np.asarray(pattern.x, dtype=float)
        inten = np.asarray(pattern.y, dtype=float)
        keep = inten >= intensity_threshold
        refs[Path(cif).stem] = (tt[keep], inten[keep])
    return refs


def q_from_two_theta(tt, wavelength_angstrom=1.5406):
    theta = np.deg2rad(np.asarray(tt, dtype=float) / 2.0)
    return (2.0 * np.sin(theta) / wavelength_angstrom) ** 2


def greedy_match(obs_tt, ref_tt, match_tolerance_deg=0.25):
    candidates = []
    for i_obs, obs in enumerate(obs_tt):
        j0 = np.searchsorted(ref_tt, obs - match_tolerance_deg, side="left")
        j1 = np.searchsorted(ref_tt, obs + match_tolerance_deg, side="right")
        candidates.extend((abs(obs - ref_tt[j]), i_obs, j) for j in range(j0, j1))

    candidates.sort(key=lambda t: t[0])
    used_obs, used_ref, pairs = set(), set(), []
    for delta, i_obs, j_ref in candidates:
        if i_obs in used_obs or j_ref in used_ref:
            continue
        used_obs.add(i_obs)
        used_ref.add(j_ref)
        pairs.append((i_obs, j_ref, delta))
    return pairs


def score_phase(
    obs_peaks,
    ref_peaks,
    num_obs_lines_for_fom=20,
    match_tolerance_deg=0.25,
    min_matched_lines_for_score=6,
    wavelength_angstrom=1.5406,
):
    n_used = min(num_obs_lines_for_fom, len(obs_peaks))
    if n_used == 0 or len(ref_peaks) == 0:
        return {
            "de_wolff": 0.0,
            "smith_snyder": 0.0,
            "n_used": 0,
            "n_match": 0,
            "n_possible": 0,
            "mean_delta_2theta": np.inf,
        }

    obs_used = obs_peaks[:n_used]
    pairs = greedy_match(obs_used, ref_peaks, match_tolerance_deg=match_tolerance_deg)

    n_possible = max(int(np.sum(ref_peaks <= obs_used[-1])), 1)
    n_match = len(pairs)
    if n_match < min_matched_lines_for_score:
        return {
            "de_wolff": 0.0,
            "smith_snyder": 0.0,
            "n_used": int(n_used),
            "n_match": int(n_match),
            "n_possible": int(n_possible),
            "mean_delta_2theta": np.inf,
        }

    deltas = np.array([d for _, _, d in pairs], dtype=float)
    mean_delta = float(np.mean(deltas))
    completeness = (n_match / n_possible) * (n_match / n_used)

    smith_snyder = completeness / max(mean_delta, 1e-8)

    q_obs = q_from_two_theta(obs_used, wavelength_angstrom=wavelength_angstrom)
    q_ref = q_from_two_theta(ref_peaks, wavelength_angstrom=wavelength_angstrom)
    q_limit = q_obs[-1]
    n20 = max(int(np.sum(q_ref <= q_limit)), 1)
    dq = np.array([abs(q_obs[i] - q_ref[j]) for i, j, _ in pairs], dtype=float)
    de_wolff = (q_limit / (2.0 * n20 * max(float(np.mean(dq)), 1e-12))) * completeness

    return {
        "de_wolff": float(de_wolff),
        "smith_snyder": float(smith_snyder),
        "n_used": int(n_used),
        "n_match": int(n_match),
        "n_possible": int(n_possible),
        "mean_delta_2theta": mean_delta,
    }


def rank_phases(
    obs_peaks,
    refs,
    num_obs_lines_for_fom=20,
    match_tolerance_deg=0.25,
    min_matched_lines_for_score=6,
    wavelength_angstrom=1.5406,
):
    rows = []
    for name, (tt, _) in refs.items():
        rows.append(
            {
                "phase": name,
                **score_phase(
                    obs_peaks,
                    tt,
                    num_obs_lines_for_fom=num_obs_lines_for_fom,
                    match_tolerance_deg=match_tolerance_deg,
                    min_matched_lines_for_score=min_matched_lines_for_score,
                    wavelength_angstrom=wavelength_angstrom,
                ),
            }
        )
    return (
        sorted(rows, key=lambda r: r["de_wolff"], reverse=True),
        sorted(rows, key=lambda r: r["smith_snyder"], reverse=True),
    )
