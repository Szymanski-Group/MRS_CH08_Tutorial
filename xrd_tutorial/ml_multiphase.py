from pathlib import Path
import csv

# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD stick patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Conventional ML models
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score


# Input/output
EXPERIMENT_DIR = Path("data/exp_patterns/multi_phase")
REFERENCE_DIR = Path("data/reference_structures")
OUTPUT_DIR = Path("outputs/ml/multiphase")

# Pattern settings
MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
NUM_POINTS = 1400
WAVELENGTH = "CuKa"
WAVELENGTH_ANGSTROM = 1.5406
REFERENCE_INTENSITY_THRESHOLD = 1.0

# Synthetic-data settings
SYNTH_SAMPLES_PER_FORMULA = 60
RANDOM_SEED = 42
SINGLE_PHASE_FRACTION = 0.15

# Artifact ranges (same style as Slide-34)
UNIFORM_SHIFT_RANGE = (-0.12, 0.12)
SAMPLE_DISPLACEMENT_RANGE_MM = (-0.18, 0.18)
GONIOMETER_RADIUS_MM = 240.0

U_RANGE = (0.01, 0.06)
V_RANGE = (-0.02, 0.01)
W_RANGE = (0.002, 0.010)
SIZE_NM_RANGE = (8.0, 80.0)
MICROSTRAIN_RANGE = (0.0, 0.003)

FWHM_RANGE = (0.08, 0.45)
ETA_RANGE = (0.10, 0.70)

BACKGROUND_SCALE_RANGE = (0.05, 0.28)
HUMP_SCALE_RANGE = (0.02, 0.20)
NOISE_SCALE_RANGE = (0.002, 0.020)

# Split
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
    refs_by_formula = {}

    for cif_file in cif_files:
        pattern = calc.get_pattern(Structure.from_file(cif_file), two_theta_range=(MIN_ANGLE, MAX_ANGLE))
        peak_pos = np.asarray(pattern.x, dtype=float)
        peak_int = np.asarray(pattern.y, dtype=float)

        keep = peak_int >= REFERENCE_INTENSITY_THRESHOLD
        peak_pos = peak_pos[keep]
        peak_int = normalize_0_100(peak_int[keep])

        formula = cif_file.stem.split("_", 1)[0]
        refs_by_formula.setdefault(formula, []).append({"phase": cif_file.stem, "peak_pos": peak_pos, "peak_int": peak_int})

    return refs_by_formula


def simulate_component_profile(two_theta_grid, base_pos, base_int, rng):
    if len(base_pos) == 0:
        return np.zeros_like(two_theta_grid)

    # Intensity perturbation (texture-like random reweighting)
    peak_int = base_int * np.exp(rng.normal(0.0, 0.25, size=len(base_int)))

    # Peak-position perturbations
    uniform_shift = rng.uniform(*UNIFORM_SHIFT_RANGE)
    displacement = rng.uniform(*SAMPLE_DISPLACEMENT_RANGE_MM)
    peak_pos = base_pos + uniform_shift + sample_displacement_shift(base_pos, displacement)

    keep = (peak_pos >= MIN_ANGLE - 1.0) & (peak_pos <= MAX_ANGLE + 1.0)
    peak_pos = peak_pos[keep]
    peak_int = peak_int[keep]
    if len(peak_pos) == 0:
        return np.zeros_like(two_theta_grid)

    # Peak broadening model
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

    profile = pseudo_voigt_profile(two_theta_grid, peak_pos, fwhm, eta) @ peak_int
    return profile / np.clip(profile.max(), 1e-12, None)


def simulate_multiphase_profile(two_theta_grid, formula_list, refs_by_formula, rng):
    components = []

    for formula in formula_list:
        ref = refs_by_formula[formula][rng.integers(0, len(refs_by_formula[formula]))]
        comp = simulate_component_profile(two_theta_grid, ref["peak_pos"], ref["peak_int"], rng)
        components.append(comp)

    # Random phase fractions (sum to 1)
    weights = rng.dirichlet(np.ones(len(components)) * 1.5)
    peaks = np.zeros_like(two_theta_grid)
    for w, comp in zip(weights, components):
        peaks += w * comp
    peaks = normalize_0_100(peaks)

    # Global background
    x_cheb = 2.0 * (two_theta_grid - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE) - 1.0
    coeffs = np.array([1.0, rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1)])
    background = np.polynomial.chebyshev.chebval(x_cheb, coeffs)
    background -= background.min()
    background /= np.clip(background.max(), 1e-12, None)
    background *= rng.uniform(*BACKGROUND_SCALE_RANGE) * peaks.max()

    # Amorphous hump + Gaussian noise
    center = rng.uniform(18.0, 35.0)
    width = rng.uniform(5.0, 12.0)
    hump = rng.uniform(*HUMP_SCALE_RANGE) * peaks.max() * np.exp(-0.5 * ((two_theta_grid - center) / width) ** 2)

    noise_sigma = rng.uniform(*NOISE_SCALE_RANGE) * peaks.max()
    noise = rng.normal(0.0, noise_sigma, size=len(two_theta_grid))

    y = peaks + background + hump + noise
    y -= y.min()
    return normalize_0_100(y)


def build_synthetic_multiphase_dataset(refs_by_formula, two_theta_grid, rng):
    formulas = sorted(refs_by_formula.keys())
    X = []
    y_labels = []

    # Anchor each formula so every phase appears many times in training data.
    for anchor in formulas:
        others = [f for f in formulas if f != anchor]
        for _ in range(SYNTH_SAMPLES_PER_FORMULA):
            # Include a small fraction of pure (1-phase) samples.
            # Keep the original 2-phase vs 3-phase ratio for the remaining fraction.
            n_components = int(
                rng.choice(
                    [1, 2, 3],
                    p=[
                        SINGLE_PHASE_FRACTION,
                        (1.0 - SINGLE_PHASE_FRACTION) * 0.65,
                        (1.0 - SINGLE_PHASE_FRACTION) * 0.35,
                    ],
                )
            )
            chosen_others = list(rng.choice(others, size=n_components - 1, replace=False))
            labels = sorted([anchor] + chosen_others)

            profile = simulate_multiphase_profile(two_theta_grid, labels, refs_by_formula, rng)
            X.append(profile)
            y_labels.append(labels)

    return np.asarray(X), y_labels


def preprocess_experimental_pattern(xy_file, two_theta_grid):
    data = np.loadtxt(xy_file)
    x = data[:, 0]
    y = data[:, 1]

    keep = (x >= MIN_ANGLE) & (x <= MAX_ANGLE)
    x = x[keep]
    y = y[keep]

    y_interp = np.interp(two_theta_grid, x, y)
    y_interp = np.clip(y_interp - np.percentile(y_interp, 5.0), 0.0, None)
    return normalize_0_100(y_interp)


def get_label_scores(model, X):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = model.predict(X)

    if isinstance(scores, list):
        # Some wrappers can return list of per-label arrays.
        cols = []
        for s in scores:
            s = np.asarray(s)
            if s.ndim == 2 and s.shape[1] == 2:
                cols.append(s[:, 1])
            else:
                cols.append(s.ravel())
        return np.column_stack(cols)

    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores[:, None]
    return scores


def predict_with_threshold(scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    # Keep at least one phase prediction per pattern.
    empty = np.where(y_pred.sum(axis=1) == 0)[0]
    if len(empty) > 0:
        best_idx = np.argmax(scores[empty], axis=1)
        y_pred[empty, best_idx] = 1

    return y_pred


def threshold_metrics(y_true_bin, scores, threshold):
    y_pred_bin = predict_with_threshold(scores, threshold)
    precision_micro = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    recall_micro = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    return precision_micro, recall_micro, f1_micro, y_pred_bin


def pick_best_threshold(y_true_bin, scores):
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.10, 0.90, 41):
        _, _, f1_micro, _ = threshold_metrics(y_true_bin, scores, t)
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_t = float(t)
    return best_t, best_f1


def tune_model(X_train, y_train, X_val, y_val, model_builder, param_grid):
    best = None
    for params in param_grid:
        model = model_builder(**params)
        model.fit(X_train, y_train)

        val_scores = get_label_scores(model, X_val)
        threshold, f1_micro = pick_best_threshold(y_val, val_scores)

        if best is None or f1_micro > best["val_f1_micro"]:
            best = {
                "model": model,
                "threshold": threshold,
                "val_f1_micro": f1_micro,
                "params": params,
            }
    return best


def labels_from_filename(file_stem):
    # Example: Li2MnO3_MnO_TiO2 -> [Li2MnO3, MnO, TiO2]
    return file_stem.split("_")


def plot_test_metric_summary(model_results):
    model_names = list(model_results.keys())
    test_precision = [model_results[m]["test_precision_micro"] for m in model_names]
    test_recall = [model_results[m]["test_recall_micro"] for m in model_names]
    test_f1 = [model_results[m]["test_f1_micro"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width, test_precision, width=width, color="tab:blue", edgecolor="black", linewidth=1.0, label="Precision")
    ax.bar(x, test_recall, width=width, color="tab:green", edgecolor="black", linewidth=1.0, label="Recall")
    ax.bar(x + width, test_f1, width=width, color="tab:red", edgecolor="black", linewidth=1.0, label="F1-score")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=18, labelpad=12)
    ax.legend(fontsize=18, loc="lower right", framealpha=1)
    ax.grid(axis="y", alpha=0.25)

    out_file = OUTPUT_DIR / "model_test-metric_summary.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)

    refs_by_formula = load_reference_sticks(sorted(REFERENCE_DIR.glob("*.cif")))
    all_formulas = sorted(refs_by_formula.keys())

    X, y_label_lists = build_synthetic_multiphase_dataset(refs_by_formula, two_theta_grid, rng)

    mlb = MultiLabelBinarizer(classes=all_formulas)
    y_bin = mlb.fit_transform(y_label_lists)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_bin,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
    )

    exp_files = sorted(EXPERIMENT_DIR.glob("*.xy"))
    X_test = np.asarray([preprocess_experimental_pattern(f, two_theta_grid) for f in exp_files])
    y_test_labels = [labels_from_filename(f.stem) for f in exp_files]
    y_test = mlb.transform(y_test_labels)

    print("\n=== Multi-Phase Conventional ML Demo (k-NN, Random Forest, SVM) ===")
    print(f"Synthetic samples: {len(X)}")
    print(f"Unique formulas (labels): {len(all_formulas)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Experimental test patterns: {len(X_test)}")

    model_results = {
        "k-NN": tune_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model_builder=lambda n_neighbors: OneVsRestClassifier(
                make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"))
            ),
            param_grid=[{"n_neighbors": k} for k in [1, 3, 5, 7, 11]],
        ),
        "Random Forest": tune_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model_builder=lambda n_estimators, max_depth, min_samples_leaf: OneVsRestClassifier(
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )
            ),
            param_grid=[
                {"n_estimators": 250, "max_depth": d, "min_samples_leaf": l}
                for d in [None, 15, 30]
                for l in [1, 2, 4]
            ],
        ),
        "SVM": tune_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model_builder=lambda C, gamma: OneVsRestClassifier(
                make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma, probability=True))
            ),
            param_grid=[{"C": c, "gamma": g} for c in [1.0, 5.0, 10.0] for g in ["scale", 0.01, 0.001]],
        ),
    }

    prediction_rows = []

    for model_name, result in model_results.items():
        model = result["model"]
        threshold = result["threshold"]
        test_scores = get_label_scores(model, X_test)
        test_precision_micro, test_recall_micro, test_f1_micro, y_pred_test = threshold_metrics(y_test, test_scores, threshold)

        result["test_precision_micro"] = test_precision_micro
        result["test_recall_micro"] = test_recall_micro
        result["test_f1_micro"] = test_f1_micro

        print(f"\n{model_name}")
        print(f"  Best params: {result['params']}")
        print(f"  Validation-picked threshold: {threshold:.2f}")
        print(f"  Test precision (micro): {test_precision_micro:.3f}")
        print(f"  Test recall (micro): {test_recall_micro:.3f}")
        print(f"  Test F1 (micro): {test_f1_micro:.3f}")

        print("  Test predictions (threshold-based):")
        for i, exp_file in enumerate(exp_files):
            true_set = y_test_labels[i]
            pred_set = list(mlb.classes_[np.where(y_pred_test[i] == 1)[0]])
            inter = len(set(true_set) & set(pred_set))
            f1_pattern = 0.0 if (len(true_set) + len(pred_set)) == 0 else 2.0 * inter / (len(true_set) + len(pred_set))

            print(f"    {exp_file.stem:26s} -> {pred_set}")
            prediction_rows.append(
                {
                    "model": model_name,
                    "pattern": exp_file.stem,
                    "true_labels": ";".join(true_set),
                    "predicted_labels": ";".join(pred_set),
                    "n_predicted": len(pred_set),
                    "threshold": threshold,
                    "f1_pattern": f1_pattern,
                }
            )

    plot_test_metric_summary(model_results)

    metrics_file = OUTPUT_DIR / "model_test_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "threshold", "test_precision_micro", "test_recall_micro", "test_f1_micro", "best_params"],
        )
        writer.writeheader()
        for name, result in model_results.items():
            writer.writerow(
                {
                    "model": name,
                    "threshold": result["threshold"],
                    "test_precision_micro": result["test_precision_micro"],
                    "test_recall_micro": result["test_recall_micro"],
                    "test_f1_micro": result["test_f1_micro"],
                    "best_params": str(result["params"]),
                }
            )

    pred_file = OUTPUT_DIR / "test_predictions_threshold.csv"
    with open(pred_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "pattern", "true_labels", "predicted_labels", "n_predicted", "threshold", "f1_pattern"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"\nSaved metrics: {metrics_file}")
    print(f"Saved predictions: {pred_file}")

