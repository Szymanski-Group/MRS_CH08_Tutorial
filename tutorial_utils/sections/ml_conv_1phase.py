from IPython.display import Image, display

from pathlib import Path
import csv

import numpy as np

import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

EXPERIMENT_DIR = Path("data/exp_patterns/one_phase")
REFERENCE_DIR = Path("data/reference_structures")
print(f"Reference CIF files: {len(list(REFERENCE_DIR.glob('*.cif')))}")
print(f"Resolved path: {REFERENCE_DIR.resolve()}")
OUTPUT_DIR = Path("outputs/ml/conv")

MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
NUM_POINTS = 1800
WAVELENGTH = "CuKa"
WAVELENGTH_ANGSTROM = 1.5406
REFERENCE_INTENSITY_THRESHOLD = 1.0

SYNTH_SAMPLES_PER_FORMULA = 80
RANDOM_SEED = 42

UNIFORM_SHIFT_RANGE = (-0.15, 0.15)
SAMPLE_DISPLACEMENT_RANGE_MM = (-0.20, 0.20)
GONIOMETER_RADIUS_MM = 240.0

U_RANGE = (0.01, 0.06)
V_RANGE = (-0.02, 0.01)
W_RANGE = (0.002, 0.010)
SIZE_NM_RANGE = (8.0, 80.0)
MICROSTRAIN_RANGE = (0.0, 0.003)

FWHM_RANGE = (0.08, 0.45)
ETA_RANGE = (0.10, 0.70)

BACKGROUND_SCALE_RANGE = (0.05, 0.30)
HUMP_SCALE_RANGE = (0.02, 0.20)
NOISE_SCALE_RANGE = (0.002, 0.020)

VAL_FRACTION = 0.20

def normalize_0_100(y):
    y = np.asarray(y, dtype=float)
    y = y - y.min()
    return 100.0 * y / np.clip(y.max(), 1e-12, None)

def sample_displacement_shift(two_theta_deg, displacement_mm):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    d_relative_change = displacement_mm / GONIOMETER_RADIUS_MM * np.cos(theta_rad) ** 2
    return np.rad2deg(-d_relative_change * np.tan(theta_rad))

def instrumental_fwhm(two_theta_deg, u, v, w):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    tan_theta = np.tan(theta_rad)
    fwhm_sq = u * tan_theta**2 + v * tan_theta + w
    return np.sqrt(np.clip(fwhm_sq, 1e-4, None))

def size_fwhm(two_theta_deg, size_nm):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    wavelength_nm = WAVELENGTH_ANGSTROM / 10.0
    beta_rad = 0.9 * wavelength_nm / (size_nm * np.cos(theta_rad))
    return np.rad2deg(beta_rad)

def strain_fwhm(two_theta_deg, microstrain):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    beta_rad = 4.0 * microstrain * np.tan(theta_rad)
    return np.rad2deg(beta_rad)

def pseudo_voigt_profile(two_theta_grid, centers, fwhm, eta):
    dx = two_theta_grid[:, None] - centers[None, :]
    sigma = np.clip(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-6, None)
    gamma = np.clip(fwhm / 2.0, 1e-6, None)
    gauss = np.exp(-0.5 * (dx / sigma[None, :]) ** 2)
    lorentz = (gamma[None, :] ** 2) / (dx**2 + gamma[None, :] ** 2)
    return (1.0 - eta) * gauss + eta * lorentz

def load_reference_sticks(cif_files):
    calc = XRDCalculator(wavelength=WAVELENGTH)
    refs = []

    for cif_file in cif_files:
        pattern = calc.get_pattern(Structure.from_file(cif_file), two_theta_range=(MIN_ANGLE, MAX_ANGLE))
        peak_pos = np.asarray(pattern.x, dtype=float)
        peak_int = np.asarray(pattern.y, dtype=float)

        keep = peak_int >= REFERENCE_INTENSITY_THRESHOLD
        peak_pos = peak_pos[keep]
        peak_int = normalize_0_100(peak_int[keep])

        formula = cif_file.stem.split("_", 1)[0]
        refs.append({"phase": cif_file.stem, "formula": formula, "peak_pos": peak_pos, "peak_int": peak_int})

    return refs

def simulate_artifact_profile(two_theta_grid, base_pos, base_int, rng):
    if len(base_pos) == 0:
        return np.zeros_like(two_theta_grid)

    peak_int = base_int * np.exp(rng.normal(0.0, 0.25, size=len(base_int)))

    uniform_shift = rng.uniform(*UNIFORM_SHIFT_RANGE)
    displacement = rng.uniform(*SAMPLE_DISPLACEMENT_RANGE_MM)
    peak_pos = base_pos + uniform_shift + sample_displacement_shift(base_pos, displacement)

    keep = (peak_pos >= MIN_ANGLE - 1.0) & (peak_pos <= MAX_ANGLE + 1.0)
    peak_pos = peak_pos[keep]
    peak_int = peak_int[keep]
    if len(peak_pos) == 0:
        return np.zeros_like(two_theta_grid)

    u = rng.uniform(*U_RANGE)
    v = rng.uniform(*V_RANGE)
    w = rng.uniform(*W_RANGE)
    size_nm = rng.uniform(*SIZE_NM_RANGE)
    microstrain = rng.uniform(*MICROSTRAIN_RANGE)

    fwhm = np.sqrt(
        instrumental_fwhm(peak_pos, u, v, w) ** 2
        + size_fwhm(peak_pos, size_nm) ** 2
        + strain_fwhm(peak_pos, microstrain) ** 2
    )
    fwhm += rng.uniform(*FWHM_RANGE)
    eta = rng.uniform(*ETA_RANGE)

    peaks = pseudo_voigt_profile(two_theta_grid, peak_pos, fwhm, eta) @ peak_int
    peaks = normalize_0_100(peaks)

    x_cheb = 2.0 * (two_theta_grid - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE) - 1.0
    coeffs = np.array(
        [
            1.0,
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.4, 0.4),
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.1, 0.1),
        ]
    )
    background = np.polynomial.chebyshev.chebval(x_cheb, coeffs)
    background -= background.min()
    background /= np.clip(background.max(), 1e-12, None)
    background *= rng.uniform(*BACKGROUND_SCALE_RANGE) * peaks.max()

    center = rng.uniform(18.0, 35.0)
    width = rng.uniform(5.0, 12.0)
    hump = rng.uniform(*HUMP_SCALE_RANGE) * peaks.max() * np.exp(-0.5 * ((two_theta_grid - center) / width) ** 2)

    noise_sigma = rng.uniform(*NOISE_SCALE_RANGE) * peaks.max()
    noise = rng.normal(0.0, noise_sigma, size=len(two_theta_grid))

    y = peaks + background + hump + noise
    y -= y.min()
    return normalize_0_100(y)

def build_synthetic_dataset(reference_sticks, two_theta_grid, rng):

    grouped = {}
    for ref in reference_sticks:
        grouped.setdefault(ref["formula"], []).append(ref)

    X = []
    y = []

    for formula, entries in sorted(grouped.items()):
        for _ in range(SYNTH_SAMPLES_PER_FORMULA):
            ref = entries[rng.integers(0, len(entries))]
            profile = simulate_artifact_profile(two_theta_grid, ref["peak_pos"], ref["peak_int"], rng)
            X.append(profile)
            y.append(formula)

    return np.asarray(X), np.asarray(y)

def preprocess_experimental_pattern(xy_file, two_theta_grid):
    data = np.loadtxt(xy_file)
    x = data[:, 0]
    y = data[:, 1]

    keep = (x >= MIN_ANGLE) & (x <= MAX_ANGLE)
    x = x[keep]
    y = y[keep]

    y_interp = np.interp(two_theta_grid, x, y)
    y_interp = np.clip(y_interp - np.percentile(y_interp, 5.0), 0.0, None)
    y_interp = normalize_0_100(y_interp)
    return y_interp

def tune_model(X_train, y_train, X_val, y_val, model_builder, param_grid):
    best = None
    for params in param_grid:
        model = model_builder(**params)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        if best is None or val_acc > best["val_acc"]:
            best = {"model": model, "val_acc": val_acc, "params": params}
    return best

def plot_accuracy_summary(model_results):
    model_names = list(model_results.keys())
    val_acc = [model_results[m]["val_acc"] for m in model_names]
    test_acc = [model_results[m]["test_acc"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        x - width / 2,
        val_acc,
        width=width,
        color="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        label="Validation",
    )
    ax.bar(
        x + width / 2,
        test_acc,
        width=width,
        color="tab:red",
        edgecolor="black",
        linewidth=1.0,
        label="Test",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=18, labelpad=12)
    ax.legend(fontsize=18, loc='lower right', framealpha=1)
    ax.grid(axis="y", alpha=0.25)

    out_file = OUTPUT_DIR / "model_accuracy_summary.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_file}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)

    reference_sticks = load_reference_sticks(sorted(REFERENCE_DIR.glob("*.cif")))
    X, y = build_synthetic_dataset(reference_sticks, two_theta_grid, rng)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    exp_files = sorted(EXPERIMENT_DIR.glob("*.xy"))
    X_exp = np.asarray([preprocess_experimental_pattern(f, two_theta_grid) for f in exp_files])
    y_exp = np.asarray([f.stem for f in exp_files])

    print("\n=== Conventional ML Phase-ID Demo (k-NN, Random Forest, SVM) ===")
    print(f"Synthetic training+validation samples: {len(X)}")
    print(f"Number of phase classes: {len(np.unique(y))}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test patterns (experimental): {len(X_exp)}")

    model_results = {
        "k-NN": tune_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model_builder=lambda n_neighbors: make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"),
            ),
            param_grid=[{"n_neighbors": k} for k in [1, 3, 5, 7, 11]],
        ),
        "Random Forest": tune_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model_builder=lambda n_estimators, max_depth, min_samples_leaf: RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            param_grid=[
                {"n_estimators": 300, "max_depth": d, "min_samples_leaf": l}
                for d in [None, 15, 30]
                for l in [1, 2, 4]
            ],
        ),
        "SVM": tune_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model_builder=lambda C, gamma: make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=C, gamma=gamma, probability=False),
            ),
            param_grid=[{"C": c, "gamma": g} for c in [1.0, 5.0, 10.0] for g in ["scale", 0.01, 0.001]],
        ),
    }

    prediction_rows = []

    for model_name, result in model_results.items():
        model = result["model"]
        y_pred_exp = model.predict(X_exp)
        test_acc = accuracy_score(y_exp, y_pred_exp)
        result["test_acc"] = test_acc

        print(f"\n{model_name}")
        print(f"  Best params: {result['params']}")
        print(f"  Validation accuracy: {result['val_acc']:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")

        print("  Test predictions:")
        for true_label, pred_label in zip(y_exp, y_pred_exp):
            correct = (true_label == pred_label)
            mark = "✓" if correct else "x"
            print(f"    {true_label:8s} -> {pred_label:8s} {mark}")
            prediction_rows.append(
                {
                    "model": model_name,
                    "pattern": true_label,
                    "true_phase": true_label,
                    "predicted_phase": pred_label,
                    "correct": int(correct),
                }
            )

    plot_accuracy_summary(model_results)

    metrics_file = OUTPUT_DIR / "model_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "validation_accuracy", "test_accuracy", "best_params"],
        )
        writer.writeheader()
        for name, result in model_results.items():
            writer.writerow(
                {
                    "model": name,
                    "validation_accuracy": result["val_acc"],
                    "test_accuracy": result["test_acc"],
                    "best_params": str(result["params"]),
                }
            )

    pred_file = OUTPUT_DIR / "experimental_predictions.csv"
    with open(pred_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "pattern", "true_phase", "predicted_phase", "correct"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"\nSaved metrics: {metrics_file}")
    print(f"Saved predictions: {pred_file}")
