import os
from IPython.display import Image, display

from pathlib import Path
import csv

import numpy as np

import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_DIR = Path("data/exp_patterns/multi_phase")
REFERENCE_DIR = Path("data/reference_structures")
OUTPUT_DIR = Path("outputs/dl/cnn_multiphase")

MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
NUM_POINTS = 1400
WAVELENGTH = "CuKa"
WAVELENGTH_ANGSTROM = 1.5406
REFERENCE_INTENSITY_THRESHOLD = 1.0

SYNTH_SAMPLES_PER_FORMULA = 60
RANDOM_SEED = 42
SINGLE_PHASE_FRACTION = 0.15

CNN_CONV_CHANNELS = (16, 32)
CNN_KERNEL_SIZES = (7, 5)
CNN_POOL_KERNEL_SIZE = 2
NN_HIDDEN_LAYER_SIZES = (128, 64)
NN_ALPHA = 1e-4
NN_LEARNING_RATE_INIT = 1e-3
NN_BATCH_SIZE = 32
NN_MAX_ITER = 220
PREDICTION_THRESHOLD = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    profile = pseudo_voigt_profile(two_theta_grid, peak_pos, fwhm, eta) @ peak_int
    return profile / np.clip(profile.max(), 1e-12, None)

def simulate_multiphase_profile(two_theta_grid, formula_list, refs_by_formula, rng):
    components = []
    for formula in formula_list:
        ref = refs_by_formula[formula][rng.integers(0, len(refs_by_formula[formula]))]
        components.append(simulate_component_profile(two_theta_grid, ref["peak_pos"], ref["peak_int"], rng))

    weights = rng.dirichlet(np.ones(len(components)) * 1.5)
    peaks = np.zeros_like(two_theta_grid)
    for w, comp in zip(weights, components):
        peaks += w * comp
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

def build_synthetic_multiphase_dataset(refs_by_formula, two_theta_grid, rng):
    formulas = sorted(refs_by_formula.keys())
    X = []
    y_labels = []

    for anchor in formulas:
        others = [f for f in formulas if f != anchor]
        for _ in range(SYNTH_SAMPLES_PER_FORMULA):
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
    return normalize_0_100(y_interp)

def labels_from_filename(file_stem):
    return file_stem.split("_")

class SimpleConvNet(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, CNN_CONV_CHANNELS[0], kernel_size=CNN_KERNEL_SIZES[0], padding=CNN_KERNEL_SIZES[0] // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=CNN_POOL_KERNEL_SIZE),
            nn.Conv1d(
                CNN_CONV_CHANNELS[0],
                CNN_CONV_CHANNELS[1],
                kernel_size=CNN_KERNEL_SIZES[1],
                padding=CNN_KERNEL_SIZES[1] // 2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=CNN_POOL_KERNEL_SIZE),
        )

        downsample_factor = CNN_POOL_KERNEL_SIZE**2
        conv_output_points = NUM_POINTS // downsample_factor
        conv_output_size = CNN_CONV_CHANNELS[1] * conv_output_points

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, NN_HIDDEN_LAYER_SIZES[0]),
            nn.ReLU(),
            nn.Linear(NN_HIDDEN_LAYER_SIZES[0], NN_HIDDEN_LAYER_SIZES[1]),
            nn.ReLU(),
            nn.Linear(NN_HIDDEN_LAYER_SIZES[1], n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build_simple_nn(n_outputs):
    return SimpleConvNet(n_outputs).to(DEVICE)

def fit_simple_nn(model, X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    y_train = y_train.astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled[:, None, :]),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LEARNING_RATE_INIT, weight_decay=NN_ALPHA)

    loss_curve = []
    model.train()
    for _ in range(NN_MAX_ITER):
        epoch_loss = 0.0
        n_seen = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            n_seen += batch_size

        loss_curve.append(epoch_loss / np.clip(n_seen, 1, None))

    return scaler, np.asarray(loss_curve, dtype=float)

def get_label_scores(model, scaler, X):
    X_scaled = scaler.transform(X).astype(np.float32)
    X_tensor = torch.from_numpy(X_scaled[:, None, :]).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        scores = torch.sigmoid(logits).cpu().numpy()

    return scores

def predict_binary_labels(scores, threshold=PREDICTION_THRESHOLD):
    return (scores >= threshold).astype(int)

def compute_micro_metrics(y_true_bin, y_pred_bin):
    precision_micro = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    recall_micro = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    return precision_micro, recall_micro, f1_micro

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

def plot_test_metric_summary(precision_micro, recall_micro, f1_micro):
    fig, ax = plt.subplots(figsize=(6.6, 4.5))

    x = np.array([0])
    width = 0.24
    ax.bar(x - width, [precision_micro], width=width, color="tab:blue", edgecolor="black", linewidth=1.0, label="Precision")
    ax.bar(x, [recall_micro], width=width, color="tab:green", edgecolor="black", linewidth=1.0, label="Recall")
    ax.bar(x + width, [f1_micro], width=width, color="tab:red", edgecolor="black", linewidth=1.0, label="F1-score")

    ax.set_xticks(x)
    ax.set_xticklabels(["CNN"], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=18, labelpad=12)
    ax.legend(fontsize=16, loc="lower right", framealpha=1)
    ax.grid(axis="y", alpha=0.25)

    out_file = OUTPUT_DIR / "nn_test-metric_summary.png"
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

    print("\n=== CNN Phase-ID Demo (multi-phase) ===")
    print(f"Synthetic samples: {len(X)}")
    print(f"Unique formulas (labels): {len(all_formulas)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Experimental test patterns: {len(X_test)}")

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model = build_simple_nn(n_outputs=len(all_formulas))
    scaler, train_loss_curve = fit_simple_nn(model, X_train, y_train)

    val_scores = get_label_scores(model, scaler, X_val)
    y_pred_val = predict_binary_labels(val_scores, threshold=PREDICTION_THRESHOLD)
    val_precision_micro, val_recall_micro, val_f1_micro = compute_micro_metrics(y_val, y_pred_val)

    test_scores = get_label_scores(model, scaler, X_test)
    y_pred_test = predict_binary_labels(test_scores, threshold=PREDICTION_THRESHOLD)
    test_precision_micro, test_recall_micro, test_f1_micro = compute_micro_metrics(y_test, y_pred_test)

    print("\nNeural Net")
    print(f"  Conv layers: {CNN_CONV_CHANNELS} kernels={CNN_KERNEL_SIZES} pool={CNN_POOL_KERNEL_SIZE}")
    print(f"  Fixed hidden layers: {NN_HIDDEN_LAYER_SIZES}")
    print(f"  Output nodes (labels): {len(all_formulas)}")
    print(f"  Binary threshold: {PREDICTION_THRESHOLD:.2f}")
    print(f"  Validation precision (micro): {val_precision_micro:.3f}")
    print(f"  Validation recall (micro): {val_recall_micro:.3f}")
    print(f"  Validation F1 (micro): {val_f1_micro:.3f}")
    print(f"  Test precision (micro): {test_precision_micro:.3f}")
    print(f"  Test recall (micro): {test_recall_micro:.3f}")
    print(f"  Test F1 (micro): {test_f1_micro:.3f}")

    print("  Test predictions (binary outputs):")
    prediction_rows = []
    for i, exp_file in enumerate(exp_files):
        true_set = y_test_labels[i]
        pred_set = list(mlb.classes_[np.where(y_pred_test[i] == 1)[0]])

        true_binary = " ".join(map(str, y_test[i].astype(int).tolist()))
        pred_binary = " ".join(map(str, y_pred_test[i].astype(int).tolist()))
        inter = len(set(true_set) & set(pred_set))
        f1_pattern = 0.0 if (len(true_set) + len(pred_set)) == 0 else 2.0 * inter / (len(true_set) + len(pred_set))

        print(f"    {exp_file.stem:26s} -> [{pred_binary}] {pred_set}")
        prediction_rows.append(
            {
                "pattern": exp_file.stem,
                "true_labels": ";".join(true_set),
                "predicted_labels": ";".join(pred_set),
                "true_binary": true_binary,
                "predicted_binary": pred_binary,
                "n_predicted": len(pred_set),
                "f1_pattern": f1_pattern,
            }
        )

    if len(train_loss_curve) > 0:
        plot_loss_curve(train_loss_curve)
    plot_test_metric_summary(test_precision_micro, test_recall_micro, test_f1_micro)

    metrics_file = OUTPUT_DIR / "nn_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "hidden_layers",
                "n_outputs",
                "threshold",
                "validation_precision_micro",
                "validation_recall_micro",
                "validation_f1_micro",
                "test_precision_micro",
                "test_recall_micro",
                "test_f1_micro",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "CNN",
                "hidden_layers": str(NN_HIDDEN_LAYER_SIZES),
                "n_outputs": len(all_formulas),
                "threshold": PREDICTION_THRESHOLD,
                "validation_precision_micro": val_precision_micro,
                "validation_recall_micro": val_recall_micro,
                "validation_f1_micro": val_f1_micro,
                "test_precision_micro": test_precision_micro,
                "test_recall_micro": test_recall_micro,
                "test_f1_micro": test_f1_micro,
            }
        )

    pred_file = OUTPUT_DIR / "test_predictions_binary.csv"
    with open(pred_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pattern",
                "true_labels",
                "predicted_labels",
                "true_binary",
                "predicted_binary",
                "n_predicted",
                "f1_pattern",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"\nSaved metrics: {metrics_file}")
    print(f"Saved predictions: {pred_file}")
