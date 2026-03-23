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
OUTPUT_DIR = Path("outputs/dl/mixture_of_experts")

MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
NUM_POINTS = 1400
WAVELENGTH = "CuKa"
WAVELENGTH_ANGSTROM = 1.5406
REFERENCE_INTENSITY_THRESHOLD = 1.0

SYNTH_SAMPLES_PER_FORMULA = 60
RANDOM_SEED = 42
SINGLE_PHASE_FRACTION = 0.15

CNN_CONV_CHANNELS = (6, 12)
CNN_KERNEL_SIZES = (5, 3)
CNN_POOL_KERNEL_SIZE = 2
CNN_ADAPTIVE_POOL_POINTS = 32
NN_HIDDEN_LAYER_SIZES = (16, 8)
NN_ALPHA = 1e-4
NN_LEARNING_RATE_INIT = 1e-3
NN_BATCH_SIZE = 64
NN_MAX_ITER = 80
NN_EARLY_STOPPING_MIN_EPOCHS = 20
NN_EARLY_STOPPING_PATIENCE = 10
NN_EARLY_STOPPING_MIN_DELTA = 1e-4
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

class BinaryConvExpert(nn.Module):
    def __init__(self):
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
            nn.AdaptiveAvgPool1d(CNN_ADAPTIVE_POOL_POINTS),
        )

        conv_output_points = CNN_ADAPTIVE_POOL_POINTS
        conv_output_size = CNN_CONV_CHANNELS[1] * conv_output_points

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, NN_HIDDEN_LAYER_SIZES[0]),
            nn.ReLU(),
            nn.Linear(NN_HIDDEN_LAYER_SIZES[0], NN_HIDDEN_LAYER_SIZES[1]),
            nn.ReLU(),
            nn.Linear(NN_HIDDEN_LAYER_SIZES[1], 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build_binary_expert():
    return BinaryConvExpert().to(DEVICE)

def fit_binary_expert(model, X_train_scaled, y_train_bin, X_val_scaled, y_val_bin):
    y_train_bin = y_train_bin.astype(np.float32).reshape(-1, 1)
    y_val_bin = y_val_bin.astype(np.float32).reshape(-1, 1)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled[:, None, :]),
        torch.from_numpy(y_train_bin),
    )
    train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True)
    X_val_tensor = torch.from_numpy(X_val_scaled[:, None, :]).to(DEVICE)
    y_val_tensor = torch.from_numpy(y_val_bin).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LEARNING_RATE_INIT, weight_decay=NN_ALPHA)

    train_loss_curve = []
    val_loss_curve = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, NN_MAX_ITER + 1):
        model.train()
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

        train_loss = epoch_loss / np.clip(n_seen, 1, None)
        train_loss_curve.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
        val_loss_curve.append(val_loss)

        is_improved = val_loss < (best_val_loss - NN_EARLY_STOPPING_MIN_DELTA)
        if is_improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= NN_EARLY_STOPPING_MIN_EPOCHS and epochs_without_improvement >= NN_EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)

    return (
        np.asarray(train_loss_curve, dtype=float),
        np.asarray(val_loss_curve, dtype=float),
        int(best_epoch),
    )

def get_binary_scores(model, X_scaled):
    X_tensor = torch.from_numpy(X_scaled[:, None, :]).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        scores = torch.sigmoid(logits).cpu().numpy()

    return scores

def predict_binary_labels(scores, threshold=PREDICTION_THRESHOLD):
    return (scores >= threshold).astype(int)

def compute_micro_metrics(y_true_bin, y_pred_bin):
    precision_micro = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    recall_micro = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    return precision_micro, recall_micro, f1_micro

def plot_loss_curve(expert_train_loss_curves, expert_val_loss_curves):
    if len(expert_train_loss_curves) == 0:
        return

    def mean_curve_with_padding(curves_by_formula):
        formulas = sorted(curves_by_formula)
        max_len = max(len(curves_by_formula[f]) for f in formulas)
        matrix = np.full((len(formulas), max_len), np.nan, dtype=float)
        for i, f in enumerate(formulas):
            curve = curves_by_formula[f]
            matrix[i, : len(curve)] = curve
        return np.nanmean(matrix, axis=0), max_len

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for formula in sorted(expert_train_loss_curves):
        train_curve = expert_train_loss_curves[formula]
        val_curve = expert_val_loss_curves[formula]
        ax.plot(np.arange(1, len(train_curve) + 1), train_curve, color="tab:blue", alpha=0.12, linewidth=0.9)
        ax.plot(np.arange(1, len(val_curve) + 1), val_curve, color="tab:orange", alpha=0.10, linewidth=0.9)

    train_mean, train_len = mean_curve_with_padding(expert_train_loss_curves)
    val_mean, val_len = mean_curve_with_padding(expert_val_loss_curves)
    ax.plot(np.arange(1, train_len + 1), train_mean, color="tab:blue", linewidth=2.3, label="Train (mean expert)")
    ax.plot(np.arange(1, val_len + 1), val_mean, color="tab:orange", linewidth=2.1, label="Val (mean expert)")

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
    ax.set_xticklabels(["MoE-CNN"], fontsize=16)
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    exp_files = sorted(EXPERIMENT_DIR.glob("*.xy"))
    X_test = np.asarray([preprocess_experimental_pattern(f, two_theta_grid) for f in exp_files])
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    y_test_labels = [labels_from_filename(f.stem) for f in exp_files]
    y_test = mlb.transform(y_test_labels)

    print("\n=== CNN Mixture-of-Experts Demo (multi-phase) ===")
    print(f"Synthetic samples: {len(X)}")
    print(f"Unique formulas (labels): {len(all_formulas)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Experimental test patterns: {len(X_test)}")

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    expert_models = {}
    expert_train_loss_curves = {}
    expert_val_loss_curves = {}
    val_scores_by_formula = []
    test_scores_by_formula = []
    expert_metric_rows = []
    expert_epochs = []

    print("\nExperts")
    for i, formula in enumerate(all_formulas):
        torch.manual_seed(RANDOM_SEED + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED + i)

        model = build_binary_expert()
        train_loss_curve, val_loss_curve, best_epoch = fit_binary_expert(
            model,
            X_train_scaled,
            y_train[:, i],
            X_val_scaled,
            y_val[:, i],
        )
        epochs_ran = len(train_loss_curve)

        val_scores = get_binary_scores(model, X_val_scaled)
        val_preds = predict_binary_labels(val_scores, threshold=PREDICTION_THRESHOLD)

        test_scores = get_binary_scores(model, X_test_scaled)

        val_precision = precision_score(y_val[:, i], val_preds, zero_division=0)
        val_recall = recall_score(y_val[:, i], val_preds, zero_division=0)
        val_f1 = f1_score(y_val[:, i], val_preds, zero_division=0)

        print(
            f"  {formula:10s} -> output neuron in [0, 1], threshold={PREDICTION_THRESHOLD:.2f}, "
            f"epochs={epochs_ran}, best_epoch={best_epoch}, val F1={val_f1:.3f}"
        )

        expert_models[formula] = model
        expert_train_loss_curves[formula] = train_loss_curve
        expert_val_loss_curves[formula] = val_loss_curve
        val_scores_by_formula.append(val_scores)
        test_scores_by_formula.append(test_scores)
        expert_epochs.append(epochs_ran)
        expert_metric_rows.append(
            {
                "formula": formula,
                "threshold": PREDICTION_THRESHOLD,
                "epochs_ran": epochs_ran,
                "best_epoch": best_epoch,
                "validation_precision": val_precision,
                "validation_recall": val_recall,
                "validation_f1": val_f1,
            }
        )

    val_score_matrix = np.column_stack(val_scores_by_formula)
    test_score_matrix = np.column_stack(test_scores_by_formula)

    y_pred_val = predict_binary_labels(val_score_matrix, threshold=PREDICTION_THRESHOLD)
    y_pred_test = predict_binary_labels(test_score_matrix, threshold=PREDICTION_THRESHOLD)

    val_precision_micro, val_recall_micro, val_f1_micro = compute_micro_metrics(y_val, y_pred_val)
    test_precision_micro, test_recall_micro, test_f1_micro = compute_micro_metrics(y_test, y_pred_test)

    print("\nMixture of Experts")
    print(f"  Conv layers per expert: {CNN_CONV_CHANNELS} kernels={CNN_KERNEL_SIZES} pool={CNN_POOL_KERNEL_SIZE}")
    print(f"  Hidden layers per expert: {NN_HIDDEN_LAYER_SIZES}")
    print(f"  Experts (one per label): {len(all_formulas)}")
    print("  Output neuron per expert: 1 (sigmoid score in [0, 1])")
    print(f"  Binary threshold: {PREDICTION_THRESHOLD:.2f}")
    print(f"  Max epochs per expert: {NN_MAX_ITER}")
    print(
        "  Early stopping: "
        f"min_epochs={NN_EARLY_STOPPING_MIN_EPOCHS}, "
        f"patience={NN_EARLY_STOPPING_PATIENCE}, "
        f"min_delta={NN_EARLY_STOPPING_MIN_DELTA}"
    )
    print(f"  Mean epochs actually run: {np.mean(expert_epochs):.1f}")
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

    plot_loss_curve(expert_train_loss_curves, expert_val_loss_curves)
    plot_test_metric_summary(test_precision_micro, test_recall_micro, test_f1_micro)

    metrics_file = OUTPUT_DIR / "nn_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "hidden_layers",
                "n_experts",
                "threshold",
                "max_iter",
                "early_stopping_min_epochs",
                "early_stopping_patience",
                "early_stopping_min_delta",
                "mean_epochs_ran",
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
                "model": "CNN-MoE",
                "hidden_layers": str(NN_HIDDEN_LAYER_SIZES),
                "n_experts": len(all_formulas),
                "threshold": PREDICTION_THRESHOLD,
                "max_iter": NN_MAX_ITER,
                "early_stopping_min_epochs": NN_EARLY_STOPPING_MIN_EPOCHS,
                "early_stopping_patience": NN_EARLY_STOPPING_PATIENCE,
                "early_stopping_min_delta": NN_EARLY_STOPPING_MIN_DELTA,
                "mean_epochs_ran": float(np.mean(expert_epochs)),
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

    expert_metrics_file = OUTPUT_DIR / "expert_metrics.csv"
    with open(expert_metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "formula",
                "threshold",
                "epochs_ran",
                "best_epoch",
                "validation_precision",
                "validation_recall",
                "validation_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(expert_metric_rows)

    print(f"\nSaved metrics: {metrics_file}")
    print(f"Saved predictions: {pred_file}")
    print(f"Saved expert metrics: {expert_metrics_file}")
