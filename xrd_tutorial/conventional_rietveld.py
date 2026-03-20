from pathlib import Path
import csv

# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# For optimization
from scipy.optimize import minimize

# To load structures and compute XRD stick patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


# Input/output
EXPERIMENT_DIR = Path("data/exp_patterns/one_phase")
REFERENCE_DIR = Path("data/reference_structures")
OUTPUT_DIR = Path("outputs/conventional/rietveld")

# Pattern settings
MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
WAVELENGTH = "CuKa"
REFERENCE_INTENSITY_THRESHOLD = 1.0

# Refinement settings
BACKGROUND_DEGREE = 6  # higher degree -> more flexible/curvy background
FWHM_INIT = 0.30
GAUSS_FRAC = 0.2

LATTICE_SCALE_BOUNDS = (0.98, 1.02)  # refine a,b,c scale factors
FWHM_BOUNDS = (0.05, 1.20)

LATTICE_MAXITER = 40
WIDTH_MAXITER = 40

# Reporting
MAX_EXPERIMENT_PATTERNS = None
TOP_K_TO_PRINT = 5
PATTERNS_TO_RUN = ["TiO2"]  # set to None to run all patterns

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

    # Keep observed profile scale simple and consistent.
    intensity = normalize_0_100(intensity)
    return two_theta, intensity


def load_reference_structures(cif_files):
    return {cif.stem: Structure.from_file(cif) for cif in cif_files}


def apply_lattice_scales(structure, scales):
    s = structure.copy()
    # scale = 1.00 means no change; strain is relative (scale - 1).
    s.apply_strain([scales[0] - 1.0, scales[1] - 1.0, scales[2] - 1.0])
    return s


def get_stick_pattern(structure, calculator):
    pattern = calculator.get_pattern(structure, two_theta_range=(MIN_ANGLE, MAX_ANGLE))
    peak_pos = np.asarray(pattern.x, dtype=float)
    peak_intensity = np.asarray(pattern.y, dtype=float)

    keep = peak_intensity >= REFERENCE_INTENSITY_THRESHOLD
    peak_pos = peak_pos[keep]
    peak_intensity = normalize_0_100(peak_intensity[keep])
    return peak_pos, peak_intensity


def simulate_profile(two_theta_grid, peak_pos, peak_intensity, fwhm):
    if len(peak_pos) == 0:
        return np.zeros_like(two_theta_grid)

    dx = two_theta_grid[:, None] - peak_pos[None, :]

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0

    gauss = np.exp(-0.5 * (dx / sigma) ** 2)
    lorentz = (gamma**2) / (dx**2 + gamma**2)

    profile = ((1.0 - GAUSS_FRAC) * gauss + GAUSS_FRAC * lorentz) @ peak_intensity
    return normalize_0_100(profile)


def fit_scale_only(y_obs, y_phase, y_bg):
    # Solve y_obs ≈ scale * y_phase + y_bg for scale (least squares, clipped positive).
    numer = np.dot(y_phase, (y_obs - y_bg))
    denom = np.dot(y_phase, y_phase)
    scale = 0.0 if denom < 1e-12 else max(0.0, numer / denom)
    y_fit = scale * y_phase + y_bg
    return scale, y_fit


def fit_background_and_scale(y_obs, y_phase, x_scaled, degree):
    # Linear model: y ≈ scale * phase + Chebyshev background.
    cheb_basis = np.polynomial.chebyshev.chebvander(x_scaled, degree)
    A = np.column_stack([y_phase, cheb_basis])
    params, *_ = np.linalg.lstsq(A, y_obs, rcond=None)

    scale = max(0.0, float(params[0]))
    bg_coeffs = np.asarray(params[1:], dtype=float)

    y_bg = np.polynomial.chebyshev.chebval(x_scaled, bg_coeffs)
    y_fit = scale * y_phase + y_bg
    return scale, bg_coeffs, y_bg, y_fit


def compute_rwp(y_obs, y_calc):
    # Simple weighted profile R-factor.
    w = 1.0 / np.clip(y_obs, 1e-3, None)
    numer = np.sum(w * (y_obs - y_calc) ** 2)
    denom = np.sum(w * y_obs**2)
    return 100.0 * np.sqrt(numer / np.clip(denom, 1e-12, None))


def pearson_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def refine_phase_sequential(two_theta, y_obs, base_structure, calculator):
    x_scaled = 2.0 * (two_theta - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE) - 1.0

    # ---------------------------------------------------------------------
    # Step 1: refine background (polynomial), with initial lattice/width.
    # ---------------------------------------------------------------------
    peak_pos_0, peak_int_0 = get_stick_pattern(base_structure, calculator)
    y_phase_0 = simulate_profile(two_theta, peak_pos_0, peak_int_0, FWHM_INIT)
    _, bg_coeffs, y_bg, y_fit_1 = fit_background_and_scale(y_obs, y_phase_0, x_scaled, BACKGROUND_DEGREE)

    # ---------------------------------------------------------------------
    # Step 2: refine lattice parameters (a,b,c scales), background fixed.
    # ---------------------------------------------------------------------
    def lattice_objective(scales):
        s = np.clip(np.asarray(scales, dtype=float), *LATTICE_SCALE_BOUNDS)
        refined_structure = apply_lattice_scales(base_structure, s)
        peak_pos, peak_int = get_stick_pattern(refined_structure, calculator)
        y_phase = simulate_profile(two_theta, peak_pos, peak_int, FWHM_INIT)
        _, y_fit = fit_scale_only(y_obs, y_phase, y_bg)
        return np.mean((y_obs - y_fit) ** 2)

    res_lat = minimize(
        lattice_objective,
        x0=np.array([1.0, 1.0, 1.0]),
        method="Powell",
        bounds=[LATTICE_SCALE_BOUNDS, LATTICE_SCALE_BOUNDS, LATTICE_SCALE_BOUNDS],
        options={"maxiter": LATTICE_MAXITER, "xtol": 1e-3, "ftol": 1e-3},
    )
    best_scales = np.clip(np.asarray(res_lat.x, dtype=float), *LATTICE_SCALE_BOUNDS)

    structure_2 = apply_lattice_scales(base_structure, best_scales)
    peak_pos_2, peak_int_2 = get_stick_pattern(structure_2, calculator)
    y_phase_2 = simulate_profile(two_theta, peak_pos_2, peak_int_2, FWHM_INIT)
    _, y_fit_2 = fit_scale_only(y_obs, y_phase_2, y_bg)

    # ---------------------------------------------------------------------
    # Step 3: refine peak width (FWHM), lattice/background fixed.
    # ---------------------------------------------------------------------
    def width_objective(width):
        w = float(np.clip(width[0], *FWHM_BOUNDS))
        y_phase = simulate_profile(two_theta, peak_pos_2, peak_int_2, w)
        _, y_fit = fit_scale_only(y_obs, y_phase, y_bg)
        return np.mean((y_obs - y_fit) ** 2)

    res_w = minimize(
        width_objective,
        x0=np.array([FWHM_INIT]),
        method="Powell",
        bounds=[FWHM_BOUNDS],
        options={"maxiter": WIDTH_MAXITER, "xtol": 1e-3, "ftol": 1e-3},
    )
    best_fwhm = float(np.clip(res_w.x[0], *FWHM_BOUNDS))

    y_phase_3 = simulate_profile(two_theta, peak_pos_2, peak_int_2, best_fwhm)
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


def print_rank_table(pattern_name, rows):
    print(f"\n{pattern_name}: top {TOP_K_TO_PRINT} phases by final Rwp (lower is better)")
    print("rank  phase             Rwp(%)  Pearson   a_scale  b_scale  c_scale  FWHM")
    print("----  ----------------  ------  -------  -------  -------  -------  -----")
    for i, row in enumerate(rows[:TOP_K_TO_PRINT], start=1):
        s = row["scales"]
        print(
            f"{i:>4}  {row['phase']:<16}  {row['rwp']:>6.2f}  {row['pearson']:>7.3f}  "
            f"{s[0]:>7.4f}  {s[1]:>7.4f}  {s[2]:>7.4f}  {row['fwhm']:>5.3f}"
        )


def plot_refinement_summary(pattern_name, two_theta, y_obs, best_row):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=FIGSIZE, sharex=True)

    # Step 1: background refinement
    axes[0].plot(two_theta, y_obs, color="black", linewidth=2.0, label="Experimental")
    axes[0].plot(two_theta, best_row["y_bg"], color="#6b7280", linewidth=1.8, label="Background")
    axes[0].plot(two_theta, best_row["y_fit_step1"], color="#1f4ed8", linewidth=1.8, label="Step 1 fit")
    axes[0].set_title("Step 1: Background refinement", fontsize=TITLE_SIZE, pad=4)
    axes[0].legend(fontsize=10, loc="upper right")

    # Step 2: lattice refinement
    axes[1].plot(two_theta, y_obs, color="black", linewidth=2.0, label="Experimental")
    axes[1].plot(two_theta, best_row["y_fit_step2"], color="#1f4ed8", linewidth=1.8, label="Step 2 fit")
    axes[1].set_title("Step 2: Lattice-parameter refinement", fontsize=TITLE_SIZE, pad=4)
    axes[1].legend(fontsize=10, loc="upper right")

    # Step 3: peak-width refinement (final)
    axes[2].plot(two_theta, y_obs, color="black", linewidth=2.0, label="Experimental")
    axes[2].plot(two_theta, best_row["y_fit_final"], color="#dc2626", linewidth=1.9, label="Final fit")
    axes[2].set_title(f"Step 3: Peak-width refinement ({phase_title_label(best_row['phase'])})", fontsize=TITLE_SIZE, pad=4)
    axes[2].legend(fontsize=10, loc="upper right")

    for ax in axes:
        ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Intensity", fontsize=AXIS_LABEL_SIZE, labelpad=12)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    axes[-1].set_xlabel("2θ", fontsize=AXIS_LABEL_SIZE, labelpad=12)

    out_file = OUTPUT_DIR / f"{pattern_name}_rietveld-sequential.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"  Saved plot: {out_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp_files = sorted(EXPERIMENT_DIR.glob("*.xy"))
    if PATTERNS_TO_RUN is not None:
        allowed = set(PATTERNS_TO_RUN)
        exp_files = [f for f in exp_files if f.stem in allowed]
    if MAX_EXPERIMENT_PATTERNS is not None:
        exp_files = exp_files[:MAX_EXPERIMENT_PATTERNS]

    structures = load_reference_structures(sorted(REFERENCE_DIR.glob("*.cif")))
    calculator = XRDCalculator(wavelength=WAVELENGTH)

    all_rows = []
    print("\n=== Sequential Rietveld-Style Refinement Demo ===")
    print("Refinement order: (1) background -> (2) lattice params -> (3) peak width")
    print(f"Experimental patterns: {len(exp_files)}")
    print(f"Reference phases:      {len(structures)}")

    for exp_file in exp_files:
        pattern_name = exp_file.stem
        two_theta, y_obs = load_experimental_profile(exp_file)

        rows = []
        for phase, structure in structures.items():
            result = refine_phase_sequential(two_theta, y_obs, structure, calculator)
            rows.append({"phase": phase, **result})

        rows = sorted(rows, key=lambda r: r["rwp"])
        best_row = rows[0]

        print(f"\n--- {pattern_name} ---")
        print_rank_table(pattern_name, rows)
        plot_refinement_summary(pattern_name, two_theta, y_obs, best_row)

        for row in rows:
            all_rows.append(
                {
                    "pattern": pattern_name,
                    "phase": row["phase"],
                    "rwp": row["rwp"],
                    "pearson": row["pearson"],
                    "a_scale": row["scales"][0],
                    "b_scale": row["scales"][1],
                    "c_scale": row["scales"][2],
                    "fwhm": row["fwhm"],
                }
            )

    csv_file = OUTPUT_DIR / "all_pattern_rietveld-rankings.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pattern", "phase", "rwp", "pearson", "a_scale", "b_scale", "c_scale", "fwhm"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved ranking table: {csv_file}")

