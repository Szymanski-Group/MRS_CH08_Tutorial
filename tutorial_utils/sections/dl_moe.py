import os
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
    if len(base_pos) == 0: return np.zeros_like(two_theta_grid)
    peak_int = base_int * np.exp(rng.normal(0.0, 0.25, size=len(base_int)))
    uniform_shift = rng.uniform(*UNIFORM_SHIFT_RANGE)
    displacement = rng.uniform(*SAMPLE_DISPLACEMENT_RANGE_MM)
    peak_pos = base_pos + uniform_shift + sample_displacement_shift(base_pos, displacement)
    keep = (peak_pos >= MIN_ANGLE - 1.0) & (peak_pos <= MAX_ANGLE + 1.0)
    peak_pos, peak_int = peak_pos[keep], peak_int[keep]
    if len(peak_pos) == 0: return np.zeros_like(two_theta_grid)
    u, v, w = rng.uniform(*U_RANGE), rng.uniform(*V_RANGE), rng.uniform(*W_RANGE)
    size_nm, microstrain = rng.uniform(*SIZE_NM_RANGE), rng.uniform(*MICROSTRAIN_RANGE)
    fwhm = np.sqrt(instrumental_fwhm(peak_pos, u, v, w) ** 2 + size_fwhm(peak_pos, size_nm) ** 2 + strain_fwhm(peak_pos, microstrain) ** 2)
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
    for w, comp in zip(weights, components): peaks += w * comp
    peaks = normalize_0_100(peaks)
    x_cheb = 2.0 * (two_theta_grid - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE) - 1.0
    coeffs = np.array([1.0, rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1)])
    background = np.polynomial.chebyshev.chebval(x_cheb, coeffs)
    background -= background.min()
    background = (background / np.clip(background.max(), 1e-12, None)) * rng.uniform(*BACKGROUND_SCALE_RANGE) * peaks.max()
    hump = rng.uniform(*HUMP_SCALE_RANGE) * peaks.max() * np.exp(-0.5 * ((two_theta_grid - rng.uniform(18.0, 35.0)) / rng.uniform(5.0, 12.0)) ** 2)
    noise = rng.normal(0.0, rng.uniform(*NOISE_SCALE_RANGE) * peaks.max(), size=len(two_theta_grid))
    y = peaks + background + hump + noise
    return normalize_0_100(y - y.min())

def build_synthetic_multiphase_dataset(refs_by_formula, two_theta_grid, rng):
    formulas = sorted(refs_by_formula.keys())
    X, y_labels = [], []
    for anchor in formulas:
        others = [f for f in formulas if f != anchor]
        for _ in range(SYNTH_SAMPLES_PER_FORMULA):
            n_comp = int(rng.choice([1, 2, 3], p=[SINGLE_PHASE_FRACTION, (1.0 - SINGLE_PHASE_FRACTION) * 0.65, (1.0 - SINGLE_PHASE_FRACTION) * 0.35]))
            labels = sorted([anchor] + list(rng.choice(others, size=n_comp - 1, replace=False)))
            X.append(simulate_multiphase_profile(two_theta_grid, labels, refs_by_formula, rng))
            y_labels.append(labels)
    return np.asarray(X), y_labels

def preprocess_experimental_pattern(xy_file, two_theta_grid):
    data = np.loadtxt(xy_file)
    return normalize_0_100(np.interp(two_theta_grid, data[:, 0], data[:, 1]))

class BinaryConvExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, CNN_CONV_CHANNELS[0], kernel_size=CNN_KERNEL_SIZES[0], padding=CNN_KERNEL_SIZES[0] // 2),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(CNN_CONV_CHANNELS[0], CNN_CONV_CHANNELS[1], kernel_size=CNN_KERNEL_SIZES[1], padding=CNN_KERNEL_SIZES[1] // 2),
            nn.ReLU(), nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(CNN_CONV_CHANNELS[1] * (NUM_POINTS // 4), NN_HIDDEN_LAYER_SIZES[0]), nn.ReLU(),
                                       nn.Linear(NN_HIDDEN_LAYER_SIZES[0], NN_HIDDEN_LAYER_SIZES[1]), nn.ReLU(), nn.Linear(NN_HIDDEN_LAYER_SIZES[1], 1))
    def forward(self, x): return self.classifier(self.features(x))

def fit_binary_expert(model, X_tr, y_tr, X_vl, y_vl):
    t_ds = TensorDataset(torch.from_numpy(X_tr[:, None, :]), torch.from_numpy(y_tr.astype(np.float32).reshape(-1, 1)))
    v_ds = TensorDataset(torch.from_numpy(X_vl[:, None, :]), torch.from_numpy(y_vl.astype(np.float32).reshape(-1, 1)))
    t_ld = DataLoader(t_ds, batch_size=NN_BATCH_SIZE, shuffle=True)
    v_ld = DataLoader(v_ds, batch_size=NN_BATCH_SIZE)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=NN_LEARNING_RATE_INIT, weight_decay=NN_ALPHA)
    t_loss_c, v_loss_c = [], []
    for _ in range(NN_MAX_ITER):
        model.train()
        tl = 0
        for xb, yb in t_ld:
            opt.zero_grad()
            l = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            l.backward()
            opt.step()
            tl += l.item() * xb.size(0)
        t_loss_c.append(tl / len(X_tr))
        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in v_ld: vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * xb.size(0)
        v_loss_c.append(vl / len(X_vl))
    return np.array(t_loss_c), np.array(v_loss_c)

def plot_loss_curve(train_curves, val_curves):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tm = np.mean(list(train_curves.values()), axis=0)
    vm = np.mean(list(val_curves.values()), axis=0)
    epochs = np.arange(1, len(tm) + 1)
    ax.plot(epochs, tm, color="tab:blue", linewidth=2.2, label="Mean Train")
    ax.plot(epochs, vm, color="tab:orange", linestyle="--", linewidth=2.2, label="Mean Val")
    ax.set_xlabel("Epoch", fontsize=18, labelpad=8)
    ax.set_ylabel("Loss", fontsize=18, labelpad=10)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16, loc="upper right", framealpha=1)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nn_loss_curve.png", dpi=200)
    plt.close()

def plot_test_metric_summary(precision, recall, f1):
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    x, width = np.array([0]), 0.24
    ax.bar(x - width, [precision], width=width, color="tab:blue", edgecolor="black", label="Precision")
    ax.bar(x, [recall], width=width, color="tab:green", edgecolor="black", label="Recall")
    ax.bar(x + width, [f1], width=width, color="tab:red", edgecolor="black", label="F1-score")
    ax.set_xticks(x); ax.set_xticklabels(["CNN-MoE"], fontsize=16)
    ax.set_ylim(0.0, 1.05); ax.set_ylabel("Score", fontsize=18)
    ax.legend(fontsize=16, loc="lower right", framealpha=1); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "nn_test-metric_summary.png", dpi=200); plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)
    refs = load_reference_sticks(sorted(REFERENCE_DIR.glob("*.cif")))
    formulas = sorted(refs.keys())
    X, y_l = build_synthetic_multiphase_dataset(refs, grid, rng)
    mlb = MultiLabelBinarizer(classes=formulas)
    y_bin = mlb.fit_transform(y_l)
    Xt, Xv, yt, yv = train_test_split(X, y_bin, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt).astype(np.float32)
    Xv_s = scaler.transform(Xv).astype(np.float32)
    exp_f = sorted(EXPERIMENT_DIR.glob("*.xy"))
    Xte_s = scaler.transform(np.asarray([preprocess_experimental_pattern(f, grid) for f in exp_f])).astype(np.float32)
    yte_labels = [f.stem.split("_") for f in exp_f]
    yte = mlb.transform(yte_labels)
    t_curves, v_curves, v_scores, te_scores = {}, {}, [], []
    for i, f in enumerate(formulas):
        torch.manual_seed(RANDOM_SEED + i)
        m = BinaryConvExpert().to(DEVICE)
        tc, vc = fit_binary_expert(m, Xt_s, yt[:, i], Xv_s, yv[:, i])
        t_curves[f], v_curves[f] = tc, vc
        m.eval()
        with torch.no_grad():
            v_scores.append(torch.sigmoid(m(torch.from_numpy(Xv_s[:, None, :]).to(DEVICE))).cpu().numpy().flatten())
            te_scores.append(torch.sigmoid(m(torch.from_numpy(Xte_s[:, None, :]).to(DEVICE))).cpu().numpy().flatten())
    v_sm, te_sm = np.column_stack(v_scores), np.column_stack(te_scores)
    pv, pte = (v_sm >= PREDICTION_THRESHOLD).astype(int), (te_sm >= PREDICTION_THRESHOLD).astype(int)
    vp, vr, vf = precision_score(yv, pv, average="micro"), recall_score(yv, pv, average="micro"), f1_score(yv, pv, average="micro")
    tp, tr, tf = precision_score(yte, pte, average="micro"), recall_score(yte, pte, average="micro"), f1_score(yte, pte, average="micro")
    
    plot_loss_curve(t_curves, v_curves)
    plot_test_metric_summary(tp, tr, tf)
    
    with open(OUTPUT_DIR / "nn_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_experts", "threshold", "validation_precision_micro", "validation_recall_micro", "validation_f1_micro", "test_precision_micro", "test_recall_micro", "test_f1_micro"])
        w.writeheader()
        w.writerow({"model": "CNN-MoE", "n_experts": len(formulas), "threshold": PREDICTION_THRESHOLD, "validation_precision_micro": vp, "validation_recall_micro": vr, "validation_f1_micro": vf, "test_precision_micro": tp, "test_recall_micro": tr, "test_f1_micro": tf})

    prediction_rows = []
    for i, exp_file in enumerate(exp_f):
        true_set = yte_labels[i]
        pred_set = list(mlb.classes_[np.where(pte[i] == 1)[0]])
        inter = len(set(true_set) & set(pred_set))
        f1_p = 0.0 if (len(true_set) + len(pred_set)) == 0 else 2.0 * inter / (len(true_set) + len(pred_set))
        prediction_rows.append({"pattern": exp_file.stem, "true_labels": ";".join(true_set), "predicted_labels": ";".join(pred_set), "true_binary": " ".join(map(str, yte[i])), "predicted_binary": " ".join(map(str, pte[i])), "n_predicted": len(pred_set), "f1_pattern": f1_p})
    
    with open(OUTPUT_DIR / "test_predictions_binary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "true_labels", "predicted_labels", "true_binary", "predicted_binary", "n_predicted", "f1_pattern"])
        w.writeheader(); w.writerows(prediction_rows)

if __name__ == "__main__": main()