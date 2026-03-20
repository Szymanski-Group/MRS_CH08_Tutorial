from pathlib import Path
import csv

# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD stick patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Neural-network model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Input/output
EXPERIMENT_DIR = Path("data/exp_patterns/one_phase")
REFERENCE_DIR = Path("data/reference_structures")
OUTPUT_DIR = Path("outputs/dl/nn_1phase")

# Pattern settings
MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
NUM_POINTS = 1800
WAVELENGTH = "CuKa"
WAVELENGTH_ANGSTROM = 1.5406
REFERENCE_INTENSITY_THRESHOLD = 1.0

# Synthetic-data settings
SYNTH_SAMPLES_PER_FORMULA = 80
RANDOM_SEED = 42

# Neural-net settings (fixed architecture; no tuning)
NN_HIDDEN_LAYER_SIZES = (128, 64)
NN_ALPHA = 1e-4
NN_LEARNING_RATE_INIT = 1e-3
NN_MAX_ITER = 220

# Artifact ranges (same style as Slide-34)
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

# Split
VAL_FRACTION = 0.20


# -----------------------------
# Pattern simulation utilities
# -----------------------------
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
    coeffs = np.array([1.0, rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1)])
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
    return normalize_0_100(y_interp)


# -----------------------------
# Neural-network training
# -----------------------------
def build_nn():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=NN_HIDDEN_LAYER_SIZES,
            activation="relu",
            solver="adam",
            alpha=NN_ALPHA,
            batch_size="auto",
            learning_rate_init=NN_LEARNING_RATE_INIT,
            max_iter=NN_MAX_ITER,
            random_state=RANDOM_SEED,
        ),
    )


def plot_loss_curve(train_loss_curve):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, len(train_loss_curve) + 1)

    ax.plot(epochs, train_loss_curve, color="tab:blue", linewidth=2.2, label="Train")

    ax.set_xlabel("Epoch", fontsize=18, labelpad=8)
    ax.set_ylabel("Loss", fontsize=18, labelpad=10)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16, loc="upper right", framealpha=1)
    ax.grid(alpha=0.25)

    out_file = OUTPUT_DIR / "nn_loss_curve.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_file}")


def plot_accuracy_summary(val_acc, test_acc):
    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    x = np.array([0])
    width = 0.36
    ax.bar(x - width / 2, [val_acc], width=width, color="tab:blue", edgecolor="black", linewidth=1.0, label="Validation")
    ax.bar(x + width / 2, [test_acc], width=width, color="tab:red", edgecolor="black", linewidth=1.0, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(["Neural Net"], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=18, labelpad=12)
    ax.legend(fontsize=16, loc="lower right", framealpha=1)
    ax.grid(axis="y", alpha=0.25)

    out_file = OUTPUT_DIR / "nn_accuracy_summary.png"
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
    X_test = np.asarray([preprocess_experimental_pattern(f, two_theta_grid) for f in exp_files])
    y_test = np.asarray([f.stem for f in exp_files])

    print("\n=== Neural-Net Phase-ID Demo (1-phase) ===")
    print(f"Synthetic training+validation samples: {len(X)}")
    print(f"Number of phase classes: {len(np.unique(y))}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test patterns (experimental): {len(X_test)}")

    model = build_nn()
    model.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, model.predict(X_val))
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_loss_curve = np.asarray(model.named_steps["mlpclassifier"].loss_curve_, dtype=float)

    print("\nNeural Net")
    print(f"  Fixed hidden layers: {NN_HIDDEN_LAYER_SIZES}")
    print(f"  Output nodes (classes): {len(np.unique(y))}")
    print(f"  Validation accuracy: {val_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")

    print("  Test predictions:")
    prediction_rows = []
    for true_label, pred_label in zip(y_test, y_pred_test):
        correct = true_label == pred_label
        mark = "\u2713" if correct else "x"
        print(f"    {true_label:8s} -> {pred_label:8s} {mark}")
        prediction_rows.append(
            {
                "pattern": true_label,
                "true_phase": true_label,
                "predicted_phase": pred_label,
                "correct": int(correct),
            }
        )

    if len(train_loss_curve) > 0:
        plot_loss_curve(train_loss_curve)
    plot_accuracy_summary(val_acc, test_acc)

    metrics_file = OUTPUT_DIR / "nn_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "hidden_layers", "n_outputs", "validation_accuracy", "test_accuracy"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "Neural Net",
                "hidden_layers": str(NN_HIDDEN_LAYER_SIZES),
                "n_outputs": len(np.unique(y)),
                "validation_accuracy": val_acc,
                "test_accuracy": test_acc,
            }
        )

    pred_file = OUTPUT_DIR / "experimental_predictions.csv"
    with open(pred_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pattern", "true_phase", "predicted_phase", "correct"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"\nSaved metrics: {metrics_file}")
    print(f"Saved predictions: {pred_file}")

