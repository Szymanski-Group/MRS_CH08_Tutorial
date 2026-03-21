# Auto-extracted module from notebook code cell.
# Detailed implementation for tutorial sections.

import glob
from IPython.display import Image, display

# Inline tutorial script
from pathlib import Path
import csv

# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD stick patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


# Input/output
EXPERIMENT_DIR = Path("data/exp_patterns/one_phase")
REFERENCE_DIR = Path("data/reference_structures")
OUTPUT_DIR = Path("outputs/conventional/profile_correlation")

# Pattern and simulation settings
MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
WAVELENGTH = "CuKa"
REFERENCE_INTENSITY_THRESHOLD = 1.0

# Continuous-profile broadening (same spirit as Slide-21)
FWHM = 0.30
GAUSS_FRAC = 0.2

# Experimental preprocessing
BASELINE_PERCENTILE = 5.0

# Reporting
MAX_EXPERIMENT_PATTERNS = None
TOP_K_TO_PRINT = 5

# Plot style (matching Slide-27 sizing)
FIGSIZE = (8, 7)
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
TITLE_SIZE = 15


def normalize_0_100(y):
    y = np.asarray(y, dtype=float)
    y = y - y.min()
    return 100.0 * y / np.clip(y.max(), 1e-12, None)


def phase_title_label(phase_name):
    # Convert "Li2MnO3_15" -> "Li2MnO3 (s.g. 15)"
    formula, sg = phase_name.rsplit("_", 1)
    return f"{formula} (s.g. {sg})"


def load_experimental_profile(xy_file):
    data = np.loadtxt(xy_file)
    two_theta = data[:, 0]
    intensity = data[:, 1]

    keep = (two_theta >= MIN_ANGLE) & (two_theta <= MAX_ANGLE)
    two_theta = two_theta[keep]
    intensity = intensity[keep]

    # Simple baseline removal so correlations focus on pattern shape.
    intensity = np.clip(intensity - np.percentile(intensity, BASELINE_PERCENTILE), 0.0, None)
    intensity = normalize_0_100(intensity)

    return two_theta, intensity


def load_reference_stick_library(cif_files):
    calculator = XRDCalculator(wavelength=WAVELENGTH)
    refs = {}

    for cif_file in cif_files:
        pattern = calculator.get_pattern(
            Structure.from_file(cif_file),
            two_theta_range=(MIN_ANGLE, MAX_ANGLE),
        )
        peak_pos = np.asarray(pattern.x, dtype=float)
        peak_intensity = np.asarray(pattern.y, dtype=float)

        keep = peak_intensity >= REFERENCE_INTENSITY_THRESHOLD
        peak_pos = peak_pos[keep]
        peak_intensity = normalize_0_100(peak_intensity[keep])

        refs[cif_file.stem] = (peak_pos, peak_intensity)

    return refs


def simulate_continuous_profile(two_theta_grid, peak_pos, peak_intensity):
    if len(peak_pos) == 0:
        return np.zeros_like(two_theta_grid)

    dx = two_theta_grid[:, None] - peak_pos[None, :]

    sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = FWHM / 2.0

    gauss = np.exp(-0.5 * (dx / sigma) ** 2)
    lorentz = (gamma**2) / (dx**2 + gamma**2)

    profile = ((1.0 - GAUSS_FRAC) * gauss + GAUSS_FRAC * lorentz) @ peak_intensity
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


def rank_phases(exp_profile, exp_two_theta, reference_library):
    rows = []
    simulated_profiles = {}

    for phase, (peak_pos, peak_intensity) in reference_library.items():
        sim_profile = simulate_continuous_profile(exp_two_theta, peak_pos, peak_intensity)

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


def print_rank_table(pattern_name, rows, score_key, label):
    print(f"\n{pattern_name}: top {TOP_K_TO_PRINT} phases by {label}")
    print("rank  phase             score")
    print("----  ----------------  ------")
    for i, row in enumerate(rows[:TOP_K_TO_PRINT], start=1):
        print(f"{i:>4}  {row['phase']:<16}  {row[score_key]:>6.3f}")


def plot_summary(pattern_name, two_theta, exp_profile, best_pearson, best_cosine, simulated_profiles):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=FIGSIZE, sharex=True)

    # 1) Experimental profile
    axes[0].fill_between(two_theta, 0, exp_profile, color="black", alpha=0.15)
    axes[0].plot(two_theta, exp_profile, color="black", linewidth=2.2)
    axes[0].set_title(f"{pattern_name}: Experimental profile", fontsize=TITLE_SIZE, pad=4)

    # 2) Best by Pearson
    pearson_phase = best_pearson["phase"]
    axes[1].fill_between(two_theta, 0, exp_profile, color="black", alpha=0.10)
    axes[1].plot(two_theta, exp_profile, color="black", linewidth=2.0, label="Experimental")
    axes[1].plot(two_theta, simulated_profiles[pearson_phase], color="#1f4ed8", linewidth=2.0, label="Simulated")
    axes[1].set_title(f"Best by Pearson: {phase_title_label(pearson_phase)}", fontsize=TITLE_SIZE, pad=4)
    axes[1].legend(fontsize=11, loc="upper right")

    # 3) Best by Cosine
    cosine_phase = best_cosine["phase"]
    axes[2].fill_between(two_theta, 0, exp_profile, color="black", alpha=0.10)
    axes[2].plot(two_theta, exp_profile, color="black", linewidth=2.0, label="Experimental")
    axes[2].plot(two_theta, simulated_profiles[cosine_phase], color="#dc2626", linewidth=2.0, label="Simulated")
    axes[2].set_title(f"Best by Cosine: {phase_title_label(cosine_phase)}", fontsize=TITLE_SIZE, pad=4)
    axes[2].legend(fontsize=11, loc="upper right")

    for ax in axes:
        ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Intensity", fontsize=AXIS_LABEL_SIZE, labelpad=12)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    axes[-1].set_xlabel("2θ", fontsize=AXIS_LABEL_SIZE, labelpad=12)

    out_file = OUTPUT_DIR / f"{pattern_name}_profile-correlation.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"  Saved plot: {out_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp_files = sorted(EXPERIMENT_DIR.glob("*.xy"))
    if MAX_EXPERIMENT_PATTERNS is not None:
        exp_files = exp_files[:MAX_EXPERIMENT_PATTERNS]

    reference_library = load_reference_stick_library(sorted(REFERENCE_DIR.glob("*.cif")))

    all_rows = []
    print("\n=== Full-Profile Correlation Demo (Pearson + Cosine) ===")
    print(f"Experimental patterns: {len(exp_files)}")
    print(f"Reference phases:      {len(reference_library)}")

    for exp_file in exp_files:
        pattern_name = exp_file.stem
        two_theta, exp_profile = load_experimental_profile(exp_file)

        by_pearson, by_cosine, simulated_profiles = rank_phases(
            exp_profile,
            two_theta,
            reference_library,
        )

        print(f"\n--- {pattern_name} ---")
        print_rank_table(pattern_name, by_pearson, "pearson", "Pearson")
        print_rank_table(pattern_name, by_cosine, "cosine", "Cosine")

        plot_summary(
            pattern_name,
            two_theta,
            exp_profile,
            by_pearson[0],
            by_cosine[0],
            simulated_profiles,
        )

        for row in by_pearson:
            all_rows.append(
                {
                    "pattern": pattern_name,
                    "phase": row["phase"],
                    "pearson": row["pearson"],
                    "cosine": row["cosine"],
                }
            )

    csv_file = OUTPUT_DIR / "all_pattern_profile-correlations.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pattern", "phase", "pearson", "cosine"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved ranking table: {csv_file}")
