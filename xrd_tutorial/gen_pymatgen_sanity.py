# Keep plotting caches local so this runs cleanly in restricted environments
import os
from pathlib import Path
import csv
import warnings

os.environ.setdefault("XDG_CACHE_HOME", ".cache")
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

# For numerical work
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# Pymatgen tools for structure checks
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum

# Silence repetitive CrystalNN hints to keep tutorial output readable
warnings.filterwarnings("ignore", message="No oxidation states specified on sites!.*")
warnings.filterwarnings(
    "ignore",
    message="CrystalNN: cannot locate an appropriate radius.*",
)

# Input/output folders
INPUT_DIR = Path("data/reference_structures")
OUTPUT_DIR = Path("outputs/generative/sanity_check")
SUMMARY_CSV = OUTPUT_DIR / "sanity_summary.csv"

# Bond valence settings
BVS_NEIGHBOR_CUTOFF = 3.2  # Angstrom
BVS_SCALE_FACTOR = 1.015

# Simple plausibility thresholds for a quick tutorial check
MIN_REASONABLE_BOND = 1.1
MAX_REASONABLE_BOND = 3.2
BVS_MEAN_ABS_DIFF_THRESHOLD = 0.8

# Rough coordination-environment labels from CN
CN_TO_ENV = {
    2: "linear",
    3: "trigonal",
    4: "tetra / square",
    5: "tbp / sq pyramidal",
    6: "octahedral",
    7: "pentagonal bipyr",
    8: "cubic-like",
    12: "close-packed",
}


def get_ordered_approx_structure(structure):
    """
    Replace partially occupied sites with their dominant species.
    This keeps the script robust for simple tutorial workflows.
    """
    approx = structure.copy()
    used_ordered_approx = False

    for i, site in enumerate(list(approx)):
        if site.is_ordered:
            continue
        used_ordered_approx = True
        dominant_species = max(site.species.items(), key=lambda item: item[1])[0]
        approx.replace(i, dominant_species, site.frac_coords)

    return approx, used_ordered_approx


def get_cn_and_bonds(structure):
    """Use CrystalNN to get per-site CN and local bond lengths."""
    nn_finder = CrystalNN(weighted_cn=True)
    cn_values = []
    env_labels = []
    bond_lengths = []

    for site_index, _site in enumerate(structure):
        try:
            nn_data = nn_finder.get_nn_data(structure, site_index)
        except Exception:
            continue
        if not nn_data.cn_weights:
            continue

        best_cn = int(max(nn_data.cn_weights, key=nn_data.cn_weights.get))
        cn_values.append(best_cn)
        env_labels.append(CN_TO_ENV.get(best_cn, f"CN={best_cn}"))

        # Note: this counts bonds from each site's perspective (simple for teaching).
        for nn_info in nn_data.cn_nninfo[best_cn]:
            neighbor = nn_info["site"]
            bond_lengths.append(float(neighbor.nn_distance))

    return np.array(cn_values), env_labels, np.array(bond_lengths)


def assign_oxidation_states(structure):
    """Try BVAnalyzer first, then fall back to oxidation-state guessing."""
    analyzer = BVAnalyzer()
    try:
        return analyzer.get_oxi_state_decorated_structure(structure), "BVAnalyzer", True
    except Exception:
        guessed = structure.copy()
        try:
            guessed.add_oxidation_state_by_guess(max_sites=-1)
            return guessed, "add_oxidation_state_by_guess", True
        except Exception:
            return None, "failed", False


def get_bvs_vs_oxi(structure_with_oxi):
    """Compute bond valence sums and compare to assigned oxidation states."""
    oxi_values = []
    bvs_values = []
    skipped_sites = 0

    for site in structure_with_oxi:
        # calculate_bv_sum requires a single species at each site
        if not site.is_ordered:
            skipped_sites += 1
            continue

        neighbors = structure_with_oxi.get_neighbors(site, BVS_NEIGHBOR_CUTOFF)
        ordered_neighbors = [nn for nn in neighbors if nn.is_ordered]
        if not ordered_neighbors:
            skipped_sites += 1
            continue

        try:
            bvs = calculate_bv_sum(site, ordered_neighbors, scale_factor=BVS_SCALE_FACTOR)
            bvs_values.append(float(bvs))
            oxi_values.append(float(site.specie.oxi_state))
        except Exception:
            skipped_sites += 1

    return np.array(oxi_values), np.array(bvs_values), skipped_sites


def is_structure_reasonable(bond_lengths, cn_values, oxi_ok, oxi_values, bvs_values):
    """Very simple rule-based sanity check (good for tutorials, not strict validation)."""
    if bond_lengths.size == 0 or cn_values.size == 0 or not oxi_ok:
        return False, np.nan, np.nan

    outlier_fraction = np.mean(
        (bond_lengths < MIN_REASONABLE_BOND) | (bond_lengths > MAX_REASONABLE_BOND)
    )
    bond_ok = outlier_fraction < 0.10

    # Most sites should have a physically sensible CN range
    cn_ok = np.mean((cn_values >= 2) & (cn_values <= 12)) > 0.90

    if oxi_values.size > 0:
        bvs_mean_abs_diff = float(np.mean(np.abs(bvs_values - oxi_values)))
        bvs_ok = bvs_mean_abs_diff < BVS_MEAN_ABS_DIFF_THRESHOLD
    else:
        bvs_mean_abs_diff = np.nan
        bvs_ok = False

    return bond_ok and cn_ok and bvs_ok, outlier_fraction, bvs_mean_abs_diff


def make_plot(cif_name, cn_values, env_labels, bond_lengths, structure_oxi, oxi_values, bvs_values, status):
    """Create one 2x2 tutorial figure per structure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Bond-length distribution
    ax = axes[0, 0]
    if bond_lengths.size > 0:
        # Protect against near-constant data where numpy struggles with many bins.
        if np.ptp(bond_lengths) < 1e-6:
            center = float(bond_lengths[0])
            bins = np.linspace(center - 0.05, center + 0.05, 6)
        else:
            bins = 25
        ax.hist(bond_lengths, bins=bins, color="royalblue", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Bond length (A)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Bond length distribution", fontsize=12)

    # Coordination-number distribution
    ax = axes[0, 1]
    if cn_values.size > 0:
        unique_cn, cn_counts = np.unique(cn_values, return_counts=True)
        ax.bar(unique_cn, cn_counts, color="darkorange", alpha=0.85)
        ax.set_xticks(unique_cn)
    ax.set_xlabel("Coordination number (CrystalNN)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Coordination numbers", fontsize=12)

    # Coordination-environment labels (rough CN-based grouping)
    ax = axes[1, 0]
    if env_labels:
        env_unique, env_counts = np.unique(env_labels, return_counts=True)
        ax.barh(env_unique, env_counts, color="seagreen", alpha=0.85)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Environment labels (CN-based)", fontsize=12)

    # Bond valence sum vs oxidation state
    ax = axes[1, 1]
    if oxi_values.size > 0:
        ax.scatter(oxi_values, bvs_values, color="crimson", alpha=0.75, s=24)
        x_min = min(oxi_values.min(), bvs_values.min()) - 0.5
        x_max = max(oxi_values.max(), bvs_values.max()) + 0.5
        ax.plot([x_min, x_max], [x_min, x_max], "k--", linewidth=1.2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
    ax.set_xlabel("Assigned oxidation state", fontsize=11)
    ax.set_ylabel("Bond valence sum", fontsize=11)
    ax.set_title("BVS vs oxidation state", fontsize=12)

    # Add oxidation-state labels in title area for quick interpretation
    oxi_labels = sorted(
        {
            str(site.specie) if site.is_ordered else str(site.species)
            for site in structure_oxi
        }
    )
    fig.suptitle(f"{cif_name}  |  {status}\nOxidation states: {', '.join(oxi_labels)}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = OUTPUT_DIR / f"{Path(cif_name).stem}_sanity.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# Create output folder
OUTPUT_DIR.mkdir(exist_ok=True)

# Collect all CIFs
cif_files = sorted(INPUT_DIR.glob("*.cif"))
if not cif_files:
    raise FileNotFoundError(f"No CIF files found in: {INPUT_DIR}")

summary_rows = []
print("\nRunning sanity checks...\n")

for cif_path in cif_files:
    structure = Structure.from_file(cif_path)

    # 1) Oxidation-state assignment
    structure_oxi, oxi_method, oxi_ok = assign_oxidation_states(structure)

    # 2) Bond lengths + coordination numbers/environments
    structure_for_nn = structure_oxi if oxi_ok else structure
    structure_for_nn, used_ordered_approx = get_ordered_approx_structure(structure_for_nn)
    cn_values, env_labels, bond_lengths = get_cn_and_bonds(structure_for_nn)

    # 3) Bond valence sums
    if oxi_ok:
        structure_for_bvs, _ = get_ordered_approx_structure(structure_oxi)
        oxi_values, bvs_values, skipped_sites = get_bvs_vs_oxi(structure_for_bvs)
    else:
        used_ordered_approx = False
        oxi_values, bvs_values, skipped_sites = np.array([]), np.array([]), len(structure)

    # 4) Quick reasonableness decision
    reasonable, outlier_fraction, bvs_mean_abs_diff = is_structure_reasonable(
        bond_lengths,
        cn_values,
        oxi_ok,
        oxi_values,
        bvs_values,
    )
    status = "PASS (looks reasonable)" if reasonable else "REVIEW (possible issues)"

    plot_path = make_plot(
        cif_path.name,
        cn_values,
        env_labels,
        bond_lengths,
        structure_oxi if structure_oxi is not None else structure,
        oxi_values,
        bvs_values,
        status,
    )

    mean_cn = float(np.mean(cn_values)) if cn_values.size else np.nan
    mean_bond = float(np.mean(bond_lengths)) if bond_lengths.size else np.nan
    mean_cn_text = f"{mean_cn:.2f}" if np.isfinite(mean_cn) else "n/a"
    mean_bond_text = f"{mean_bond:.2f}" if np.isfinite(mean_bond) else "n/a"

    print(f"{cif_path.name}: {status}")
    print(f"  oxidation method: {oxi_method}")
    print(f"  mean CN: {mean_cn_text} | mean bond length: {mean_bond_text} A")
    print(f"  bond outlier fraction: {outlier_fraction:.3f}")
    print(f"  mean |BVS - oxidation|: {bvs_mean_abs_diff:.3f} (skipped sites: {skipped_sites})")
    print(f"  used ordered approximation for partial occupancy: {used_ordered_approx}")
    print(f"  saved plot: {plot_path}\n")

    summary_rows.append(
        {
            "cif_file": cif_path.name,
            "status": status,
            "oxidation_method": oxi_method,
            "mean_cn": mean_cn,
            "mean_bond_length_A": mean_bond,
            "bond_outlier_fraction": float(outlier_fraction),
            "mean_abs_bvs_minus_oxi": float(bvs_mean_abs_diff),
            "num_sites_skipped_in_bvs": int(skipped_sites),
            "used_ordered_approx_for_disorder": used_ordered_approx,
        }
    )

# Save a compact summary table
with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Saved summary table: {SUMMARY_CSV}")

"""
Try on your own:
- Change BVS_MEAN_ABS_DIFF_THRESHOLD and see how PASS/REVIEW changes
- Compare two polymorphs (for example, the TiO2 and ZrO2 files)
- Add a structure with known bad geometry and inspect the bond-length histogram
"""
