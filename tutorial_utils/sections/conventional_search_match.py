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

# For peak detection
from scipy.signal import find_peaks

# To load structures and compute XRD stick patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


# Input/output
EXPERIMENT_DIR = Path("data/exp_patterns/one_phase")
REFERENCE_DIR = Path("data/reference_structures")
OUTPUT_DIR = Path("outputs/conventional/search_match")

# Pattern and FoM settings
MIN_ANGLE = 10.0
MAX_ANGLE = 100.0
PLOT_MIN_ANGLE = 10.0
PLOT_MAX_ANGLE = 80.0
WAVELENGTH = "CuKa"
WAVELENGTH_ANGSTROM = 1.5406
MAX_EXPERIMENT_PATTERNS = None
TOP_K_TO_PRINT = 5

# Peak detection
BASELINE_PERCENTILE = 5.0
PEAK_PROMINENCE_FRACTION = 0.02
MIN_PEAK_DISTANCE_DEG = 0.22
MAX_DETECTED_PEAKS = 30

# Search-match
REFERENCE_INTENSITY_THRESHOLD = 1.0
NUM_OBS_LINES_FOR_FOM = 20
MATCH_TOLERANCE_DEG = 0.25
MIN_MATCHED_LINES_FOR_SCORE = 6

# Plot style
FIGSIZE = (8, 5)
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
TITLE_SIZE = 15


def normalize_0_100(y):
    y = np.asarray(y, dtype=float)
    y = y - y.min()
    return 100.0 * y / np.clip(y.max(), 1e-12, None)


def load_pattern(xy_file):
    data = np.loadtxt(xy_file)
    tt, intensity = data[:, 0], data[:, 1]
    keep = (tt >= MIN_ANGLE) & (tt <= MAX_ANGLE)
    return tt[keep], normalize_0_100(intensity[keep])


def detect_peaks(tt, intensity):
    corrected = np.clip(intensity - np.percentile(intensity, BASELINE_PERCENTILE), 0.0, None)

    step = np.median(np.diff(tt))
    min_distance_pts = max(1, int(round(MIN_PEAK_DISTANCE_DEG / step)))
    prominence = PEAK_PROMINENCE_FRACTION * (corrected.max() - corrected.min())

    peak_idx, _ = find_peaks(corrected, prominence=prominence, distance=min_distance_pts)
    if len(peak_idx) > MAX_DETECTED_PEAKS:
        keep = np.argsort(corrected[peak_idx])[::-1][:MAX_DETECTED_PEAKS]
        peak_idx = peak_idx[keep]

    peak_idx = np.sort(peak_idx)
    return peak_idx, tt[peak_idx]


def load_reference_library(cif_files):
    calc = XRDCalculator(wavelength=WAVELENGTH)
    refs = {}
    for cif in cif_files:
        pat = calc.get_pattern(Structure.from_file(cif), two_theta_range=(MIN_ANGLE, MAX_ANGLE))
        tt = np.asarray(pat.x, dtype=float)
        inten = np.asarray(pat.y, dtype=float)
        keep = inten >= REFERENCE_INTENSITY_THRESHOLD
        refs[cif.stem] = (tt[keep], inten[keep])
    return refs


def q_from_two_theta(tt):
    theta = np.deg2rad(np.asarray(tt, dtype=float) / 2.0)
    return (2.0 * np.sin(theta) / WAVELENGTH_ANGSTROM) ** 2


def greedy_match(obs_tt, ref_tt):
    candidates = []
    for i_obs, obs in enumerate(obs_tt):
        j0 = np.searchsorted(ref_tt, obs - MATCH_TOLERANCE_DEG, side="left")
        j1 = np.searchsorted(ref_tt, obs + MATCH_TOLERANCE_DEG, side="right")
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


def score_phase(obs_peaks, ref_peaks):
    n_used = min(NUM_OBS_LINES_FOR_FOM, len(obs_peaks))
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
    pairs = greedy_match(obs_used, ref_peaks)

    n_possible = max(int(np.sum(ref_peaks <= obs_used[-1])), 1)
    n_match = len(pairs)

    if n_match < MIN_MATCHED_LINES_FOR_SCORE:
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

    # Smith-Snyder style FoM (adapted for search-match)
    smith_snyder = completeness / max(mean_delta, 1e-8)

    # de Wolff style FoM (adapted for search-match)
    q_obs = q_from_two_theta(obs_used)
    q_ref = q_from_two_theta(ref_peaks)
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


def rank_phases(obs_peaks, refs):
    rows = [{"phase": name, **score_phase(obs_peaks, tt)} for name, (tt, _) in refs.items()]
    return (
        sorted(rows, key=lambda r: r["de_wolff"], reverse=True),
        sorted(rows, key=lambda r: r["smith_snyder"], reverse=True),
    )


def format_topk_summary(rows, score_key):
    filtered = [r for r in rows if r[score_key] > 0.0] or rows
    top_rows = filtered[:TOP_K_TO_PRINT]
    return ", ".join(f"{row['phase']} ({row[score_key]:.3f})" for row in top_rows)


def print_rank_table(pattern_name, rows, score_key, label):
    print(f"  {label}: {format_topk_summary(rows, score_key)}")


def phase_title_label(phase_name):
    # Convert "Li2MnO3_15" -> "Li2MnO3 (s.g. 15)"
    formula, sg = phase_name.rsplit("_", 1)
    return f"{formula} (s.g. {sg})"


def plot_summary(pattern_name, tt, intensity, obs_peaks, best_m, best_f, refs, show_plot=True):
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)

    # Experimental profile + detected peaks
    axes[0].plot(tt, intensity, color="black", linewidth=2.2,
                 label="Experimental")
    axes[0].scatter(obs_peaks, np.interp(obs_peaks, tt, intensity),
                    s=30, color="#dc2626", zorder=5, label="Detected")
    axes[0].set_title(f"{pattern_name}: Experimental + detected peaks",
                      fontsize=TITLE_SIZE, pad=4)
    axes[0].legend(fontsize=11, loc="upper right")

    # Best by de Wolff
    phase_m = best_m["phase"]
    ref_tt_m, ref_i_m = refs[phase_m]
    axes[1].fill_between(tt, 0, intensity, color="black", alpha=0.10)
    axes[1].plot(tt, intensity, color="black", linewidth=2.0)
    axes[1].vlines(ref_tt_m, 0, 0.8 * ref_i_m, color="#1f4ed8", linewidth=1.7)
    axes[1].set_title(
        f"Best by de Wolff: {phase_title_label(phase_m)}",
        fontsize=TITLE_SIZE,
        pad=4,
    )

    # Best by Smith-Snyder
    phase_f = best_f["phase"]
    ref_tt_f, ref_i_f = refs[phase_f]
    axes[2].fill_between(tt, 0, intensity, color="black", alpha=0.10)
    axes[2].plot(tt, intensity, color="black", linewidth=2.0)
    axes[2].vlines(ref_tt_f, 0, 0.8 * ref_i_f, color="#dc2626", linewidth=1.7)
    axes[2].set_title(
        f"Best by Smith-Snyder: {phase_title_label(phase_f)}",
        fontsize=TITLE_SIZE,
        pad=4,
    )

    for ax in axes:
        ax.set_xlim(PLOT_MIN_ANGLE, PLOT_MAX_ANGLE)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Intensity", fontsize=AXIS_LABEL_SIZE, labelpad=12)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    axes[-1].set_xlabel("2θ", fontsize=AXIS_LABEL_SIZE, labelpad=12)

    out = OUTPUT_DIR / f"{pattern_name}_summary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    if show_plot:
        display(Image(filename=str(out), width=400))
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp_files = sorted(EXPERIMENT_DIR.glob("*.xy"))
    if MAX_EXPERIMENT_PATTERNS is not None:
        exp_files = exp_files[:MAX_EXPERIMENT_PATTERNS]

    refs = load_reference_library(sorted(REFERENCE_DIR.glob("*.cif")))
    all_rows = []

    print(f"Search-match | patterns={len(exp_files)} refs={len(refs)}")

    for exp_file in exp_files:
        name = exp_file.stem
        tt, intensity = load_pattern(exp_file)
        _, obs_peaks = detect_peaks(tt, intensity)

        print(f"\n{name} | peaks={len(obs_peaks)}")

        by_m, by_f = rank_phases(obs_peaks, refs)
        print_rank_table(name, by_m, "de_wolff", "de Wolff")
        print_rank_table(name, by_f, "smith_snyder", "Smith-Snyder")

        plot_summary(name, tt, intensity, obs_peaks, by_m[0], by_f[0], refs)

        for row in by_m:
            all_rows.append({"pattern": name, "phase": row["phase"],
                             **{k: row[k] for k in row if k != "phase"}})

    csv_file = OUTPUT_DIR / "all_pattern_rankings.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pattern",
                "phase",
                "de_wolff",
                "smith_snyder",
                "n_used",
                "n_match",
                "n_possible",
                "mean_delta_2theta",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nRankings saved: {csv_file}")
