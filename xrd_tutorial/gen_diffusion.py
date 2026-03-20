from pathlib import Path
import csv

# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD stick patterns
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# For data split
from sklearn.model_selection import train_test_split

# Diffusion model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Input/output
REFERENCE_DIR = Path("data/vanadium_oxides")
EXPERIMENT_FILE = Path("data/exp_patterns/one_phase/V2O5.xy")
OUTPUT_DIR = Path("outputs/generative/diffusion")

# Pattern settings
MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
NUM_POINTS = 1800
WAVELENGTH = "CuKa"
WAVELENGTH_ANGSTROM = 1.5406
REFERENCE_INTENSITY_THRESHOLD = 1.0

# Synthetic-data settings
SYNTH_SAMPLES_PER_PHASE = 80
RANDOM_SEED = 42

# Diffusion settings (simple pedagogical version)
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-6
DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 2e-2

# Structure-vector settings
SITE_FEATURES = 5  # [present, is_V, frac_x, frac_y, frac_z]
MIN_SITES_FOR_DECODE = 2
GENERATED_SAMPLES = 64

# Artifact ranges (same style as Slide-58 script)
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


def canonical_sites(structure):
    def site_key(site):
        symbol = site.specie.symbol
        species_rank = 0 if symbol == "V" else 1
        frac = np.mod(site.frac_coords, 1.0)
        return (species_rank, symbol, float(frac[0]), float(frac[1]), float(frac[2]))

    return sorted(structure.sites, key=site_key)


def structure_to_vector(structure, max_sites):
    sites = canonical_sites(structure)
    vec = np.zeros(7 + SITE_FEATURES * max_sites, dtype=float)

    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    vec[0:6] = [a, b, c, alpha, beta, gamma]
    vec[6] = len(sites)

    for i, site in enumerate(sites[:max_sites]):
        base = 7 + i * SITE_FEATURES
        frac = np.mod(site.frac_coords, 1.0)
        vec[base + 0] = 1.0
        vec[base + 1] = 1.0 if site.specie.symbol == "V" else 0.0
        vec[base + 2 : base + 5] = frac

    return vec


def vector_to_structure(vec, max_sites):
    vec = np.asarray(vec, dtype=float)
    lengths = np.clip(vec[0:3], 2.0, 25.0)
    angles = np.clip(vec[3:6], 50.0, 130.0)

    requested_n = int(np.clip(np.rint(vec[6]), MIN_SITES_FOR_DECODE, max_sites))

    present_scores = np.asarray([vec[7 + i * SITE_FEATURES] for i in range(max_sites)])
    candidate_indices = np.where(present_scores > 0.5)[0]

    if len(candidate_indices) < MIN_SITES_FOR_DECODE:
        candidate_indices = np.argsort(present_scores)[-requested_n:]

    if len(candidate_indices) > requested_n:
        keep_order = np.argsort(present_scores[candidate_indices])[-requested_n:]
        candidate_indices = candidate_indices[keep_order]

    candidate_indices = np.sort(candidate_indices)

    species = []
    frac_coords = []

    for i in candidate_indices:
        base = 7 + i * SITE_FEATURES
        is_v = vec[base + 1] >= 0.5
        frac = np.mod(vec[base + 2 : base + 5], 1.0)

        species.append("V" if is_v else "O")
        frac_coords.append(frac)

    if len(species) < MIN_SITES_FOR_DECODE:
        species = ["V", "O"]
        frac_coords = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])]

    if "V" not in species:
        species[0] = "V"
    if "O" not in species:
        species[-1] = "O"

    lattice = Lattice.from_parameters(*lengths, *angles)
    return Structure(lattice, species, frac_coords, to_unit_cell=True, coords_are_cartesian=False)


def load_reference_data(cif_files, max_sites):
    calc = XRDCalculator(wavelength=WAVELENGTH)
    refs = []

    for cif_file in cif_files:
        structure = Structure.from_file(cif_file)
        pattern = calc.get_pattern(structure, two_theta_range=(MIN_ANGLE, MAX_ANGLE))

        peak_pos = np.asarray(pattern.x, dtype=float)
        peak_int = np.asarray(pattern.y, dtype=float)

        keep = peak_int >= REFERENCE_INTENSITY_THRESHOLD
        peak_pos = peak_pos[keep]
        peak_int = normalize_0_100(peak_int[keep])

        refs.append(
            {
                "phase": cif_file.stem,
                "formula": structure.composition.reduced_formula,
                "structure": structure,
                "structure_vec": structure_to_vector(structure, max_sites),
                "peak_pos": peak_pos,
                "peak_int": peak_int,
            }
        )

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


def build_synthetic_dataset(reference_data, two_theta_grid, rng):
    X = []
    y = []
    phases = []

    for ref in sorted(reference_data, key=lambda r: r["phase"]):
        for _ in range(SYNTH_SAMPLES_PER_PHASE):
            profile = simulate_artifact_profile(two_theta_grid, ref["peak_pos"], ref["peak_int"], rng)
            X.append(profile)
            y.append(ref["structure_vec"])
            phases.append(ref["phase"])

    return np.asarray(X), np.asarray(y), np.asarray(phases)


def preprocess_experimental_pattern(xy_file, two_theta_grid):
    data = np.loadtxt(xy_file)
    x = data[:, 0]
    y = data[:, 1]

    keep = (x >= MIN_ANGLE) & (x <= MAX_ANGLE)
    x = x[keep]
    y = y[keep]

    y_interp = np.interp(two_theta_grid, x, y)
    return normalize_0_100(y_interp)


def simulate_profile_from_structure(structure, two_theta_grid):
    calc = XRDCalculator(wavelength=WAVELENGTH)
    pattern = calc.get_pattern(structure, two_theta_range=(MIN_ANGLE, MAX_ANGLE))

    peak_pos = np.asarray(pattern.x, dtype=float)
    peak_int = np.asarray(pattern.y, dtype=float)

    keep = peak_int >= 0.1
    peak_pos = peak_pos[keep]
    peak_int = peak_int[keep]

    if len(peak_pos) == 0:
        return np.zeros_like(two_theta_grid)

    peak_int = normalize_0_100(peak_int)
    fwhm = np.full(len(peak_pos), 0.20)
    eta = 0.35
    y = pseudo_voigt_profile(two_theta_grid, peak_pos, fwhm, eta) @ peak_int
    return normalize_0_100(y)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# -----------------------------
# Torch data + model utilities
# -----------------------------
class XRDDiffusionDataset(Dataset):
    def __init__(self, patterns, targets):
        self.patterns = torch.tensor(patterns / 100.0, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        return self.patterns[idx], self.targets[idx]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half = self.embedding_dim // 2
        freq = torch.exp(-np.log(10000.0) * torch.arange(0, half, device=t.device) / max(half - 1, 1))
        phase = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class PatternEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, out_dim),
            nn.SiLU(),
        )

    def forward(self, x_pattern):
        x = x_pattern.unsqueeze(1)
        return self.net(x)


class ConditionalDenoiser(nn.Module):
    def __init__(self, target_dim, cond_dim=128, time_dim=64, hidden_dim=512):
        super().__init__()
        self.pattern_encoder = PatternEncoder(cond_dim)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        in_dim = target_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x_noisy, t, x_pattern):
        cond = self.pattern_encoder(x_pattern)
        t_emb = self.time_mlp(self.time_emb(t))
        h = torch.cat([x_noisy, cond, t_emb], dim=1)
        return self.net(h)


# -----------------------------
# Diffusion helpers
# -----------------------------
def make_diffusion_schedule(num_steps, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    alpha_bars_prev = torch.ones_like(alpha_bars)
    alpha_bars_prev[1:] = alpha_bars[:-1]

    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    posterior_var = betas * (1.0 - alpha_bars_prev) / torch.clamp(1.0 - alpha_bars, min=1e-12)
    posterior_var[0] = betas[0]

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "posterior_var": posterior_var,
        "num_steps": num_steps,
    }


def q_sample(x0, t, noise, schedule):
    sqrt_ab = schedule["sqrt_alpha_bars"][t].unsqueeze(1)
    sqrt_omb = schedule["sqrt_one_minus_alpha_bars"][t].unsqueeze(1)
    return sqrt_ab * x0 + sqrt_omb * noise


def p_sample_loop(model, cond_patterns, target_dim, schedule, device):
    model.eval()
    x = torch.randn((cond_patterns.shape[0], target_dim), device=device)

    for step in reversed(range(schedule["num_steps"])):
        t = torch.full((x.shape[0],), step, device=device, dtype=torch.long)

        beta_t = schedule["betas"][step]
        alpha_t = schedule["alphas"][step]
        alpha_bar_t = schedule["alpha_bars"][step]

        eps_theta = model(x, t, cond_patterns)
        mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta) / torch.sqrt(alpha_t)

        if step > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt(schedule["posterior_var"][step])
            x = mean + sigma * z
        else:
            x = mean

    return x


# -----------------------------
# Training + plotting
# -----------------------------
def fit_standardizer(y_train):
    mean = y_train.mean(axis=0)
    std = y_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize(y, mean, std):
    return (y - mean) / std


def unstandardize(y_scaled, mean, std):
    return y_scaled * std + mean


def train_epoch(model, dataloader, optimizer, schedule, device):
    model.train()
    running = 0.0

    for patterns, targets in dataloader:
        patterns = patterns.to(device)
        targets = targets.to(device)

        t = torch.randint(0, schedule["num_steps"], (targets.shape[0],), device=device)
        noise = torch.randn_like(targets)
        x_noisy = q_sample(targets, t, noise, schedule)

        pred_noise = model(x_noisy, t, patterns)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item() * targets.shape[0]

    return running / len(dataloader.dataset)


def evaluate_epoch(model, dataloader, schedule, device):
    model.eval()
    running = 0.0

    with torch.no_grad():
        for patterns, targets in dataloader:
            patterns = patterns.to(device)
            targets = targets.to(device)

            t = torch.randint(0, schedule["num_steps"], (targets.shape[0],), device=device)
            noise = torch.randn_like(targets)
            x_noisy = q_sample(targets, t, noise, schedule)
            pred_noise = model(x_noisy, t, patterns)

            loss = F.mse_loss(pred_noise, noise)
            running += loss.item() * targets.shape[0]

    return running / len(dataloader.dataset)


def plot_loss_curve(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, color="tab:blue", linewidth=2.2, label="Train")
    ax.plot(epochs, val_losses, color="tab:red", linewidth=2.2, label="Validation")

    ax.set_xlabel("Epoch", fontsize=18, labelpad=8)
    ax.set_ylabel("Noise-Prediction Loss", fontsize=16, labelpad=10)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14, loc="upper right", framealpha=1)
    ax.grid(alpha=0.25)

    out_file = OUTPUT_DIR / "diffusion_loss_curve.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_file}")


def plot_pattern_match(two_theta_grid, y_exp, y_pred):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    ax.plot(two_theta_grid, y_exp, color="tab:blue", linewidth=1.7, label="Experimental: V2O5.xy")
    ax.plot(two_theta_grid, y_pred, color="tab:green", linewidth=1.5, label="Generated structure (simulated XRD)")

    ax.set_xlabel(r"2$\theta$ (degrees)", fontsize=16)
    ax.set_ylabel("Intensity (normalized)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, framealpha=1, loc="upper right")

    out_file = OUTPUT_DIR / "v2o5_generated_vs_experimental.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close(fig)
    print(f"Saved plot: {out_file}")


# -----------------------------
# Main demo
# -----------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)

    cif_files = sorted(REFERENCE_DIR.glob("*.cif"))
    if len(cif_files) == 0:
        raise FileNotFoundError(f"No CIF files found in {REFERENCE_DIR}")

    structures = [Structure.from_file(f) for f in cif_files]
    max_sites = max(len(s) for s in structures)
    target_dim = 7 + SITE_FEATURES * max_sites

    reference_data = load_reference_data(cif_files, max_sites)
    X, y, phase_labels = build_synthetic_dataset(reference_data, two_theta_grid, rng)

    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X,
        y,
        phase_labels,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=phase_labels,
    )

    y_mean, y_std = fit_standardizer(y_train)
    y_train_scaled = standardize(y_train, y_mean, y_std)
    y_val_scaled = standardize(y_val, y_mean, y_std)

    train_dataset = XRDDiffusionDataset(X_train, y_train_scaled)
    val_dataset = XRDDiffusionDataset(X_val, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConditionalDenoiser(target_dim=target_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = make_diffusion_schedule(DIFFUSION_STEPS, BETA_START, BETA_END, device)

    print("\n=== Conditional Diffusion Demo (Vanadium Oxides) ===")
    print(f"CIF phases: {len(cif_files)}")
    print(f"Max sites in CIF set: {max_sites}")
    print(f"Structure vector dimension: {target_dim}")
    print(f"Synthetic training+validation samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Device: {device}")

    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, schedule, device)
        val_loss = evaluate_epoch(model, val_loader, schedule, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == NUM_EPOCHS:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train loss: {train_loss:.5f} | val loss: {val_loss:.5f}")

    plot_loss_curve(train_losses, val_losses)

    # -----------------------------
    # Inference on experimental V2O5 pattern
    # -----------------------------
    y_exp = preprocess_experimental_pattern(EXPERIMENT_FILE, two_theta_grid)
    cond_patterns = torch.tensor(np.repeat((y_exp / 100.0)[None, :], GENERATED_SAMPLES, axis=0), dtype=torch.float32, device=device)

    with torch.no_grad():
        sampled_scaled = p_sample_loop(model, cond_patterns, target_dim, schedule, device).cpu().numpy()

    sampled_vectors = unstandardize(sampled_scaled, y_mean[None, :], y_std[None, :])

    best = {
        "similarity": -1.0,
        "structure": None,
        "pattern": None,
        "index": -1,
    }

    for i in range(GENERATED_SAMPLES):
        try:
            struct_i = vector_to_structure(sampled_vectors[i], max_sites)
            y_pred_i = simulate_profile_from_structure(struct_i, two_theta_grid)
            sim_i = cosine_similarity(y_exp, y_pred_i)

            if sim_i > best["similarity"]:
                best = {
                    "similarity": sim_i,
                    "structure": struct_i,
                    "pattern": y_pred_i,
                    "index": i,
                }
        except Exception:
            continue

    if best["structure"] is None:
        raise RuntimeError("Could not decode any valid generated structure.")

    generated_cif = OUTPUT_DIR / "generated_structure_from_V2O5_xy.cif"
    best["structure"].to(filename=generated_cif)
    print(f"\nSaved generated structure: {generated_cif}")
    print(f"Best sample index: {best['index']}")
    print(f"Cosine similarity (exp vs generated simulated XRD): {best['similarity']:.4f}")

    plot_pattern_match(two_theta_grid, y_exp, best["pattern"])

    metrics_file = OUTPUT_DIR / "diffusion_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_cif_phases",
                "max_sites",
                "target_dim",
                "synthetic_samples_total",
                "train_samples",
                "val_samples",
                "epochs",
                "final_train_loss",
                "final_val_loss",
                "generated_samples",
                "best_sample_index",
                "best_cosine_similarity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "num_cif_phases": len(cif_files),
                "max_sites": max_sites,
                "target_dim": target_dim,
                "synthetic_samples_total": len(X),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "epochs": NUM_EPOCHS,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "generated_samples": GENERATED_SAMPLES,
                "best_sample_index": best["index"],
                "best_cosine_similarity": best["similarity"],
            }
        )

    print(f"Saved metrics: {metrics_file}")

