# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Input structure and plotting range
CIF_FILE = "data/cif/TiO2.cif"
MIN_ANGLE = 10
MAX_ANGLE = 80
NUM_POINTS = 4000

# Peak broadening settings
WAVELENGTH_ANGSTROM = 1.5406  # Cu Kalpha
K_FACTOR = 0.9
CRYSTALLITE_SIZE_NM = 17.5

# Tetragonal-preserving strain
# Keep a = b so symmetry is not broken.
STRAIN_LEVEL = -0.015
OUT_OF_PLANE_STRAIN = -0.010

PATTERNS = [
    ("TiO2", "#1f4ed8"),
    ("TiO2 strained", "#dc2626"),
]


def scherrer_fwhm_deg(two_theta_deg, crystallite_size_nm):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    wavelength_nm = WAVELENGTH_ANGSTROM / 10.0
    beta_rad = K_FACTOR * wavelength_nm / (crystallite_size_nm * np.cos(theta_rad))
    return np.rad2deg(beta_rad)


def gaussian_unit_area(two_theta_grid, centers, fwhm_deg):
    sigma = fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dx = two_theta_grid[:, None] - centers[None, :]
    return np.exp(-0.5 * (dx / sigma[None, :]) ** 2) / (
        sigma[None, :] * np.sqrt(2.0 * np.pi)
    )


def broaden_pattern(two_theta_grid, peak_pos, peak_intensity, size_nm):
    fwhm = scherrer_fwhm_deg(peak_pos, size_nm)
    profile = gaussian_unit_area(two_theta_grid, peak_pos, fwhm)
    intensity = profile @ peak_intensity
    return 100.0 * intensity / intensity.max()


# Load TiO2 once
tio2 = Structure.from_file(CIF_FILE)

# Build a strained copy directly from TiO2 (no second CIF needed)
tio2_strained = tio2.copy()
tio2_strained.apply_strain([STRAIN_LEVEL, STRAIN_LEVEL, OUT_OF_PLANE_STRAIN])

structures = [tio2, tio2_strained]

# Initialize XRD calculator and 2theta grid
calc = XRDCalculator(wavelength=WAVELENGTH_ANGSTROM)
two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 2.5))

# Compute and plot broadened profiles for original + strained TiO2
for structure, (label, color) in zip(structures, PATTERNS):
    pattern = calc.get_pattern(structure, two_theta_range=(MIN_ANGLE, MAX_ANGLE))
    peak_pos = np.array(pattern.x)
    peak_intensity = np.array(pattern.y)
    peak_intensity = 100.0 * peak_intensity / peak_intensity.max()

    continuous_intensity = broaden_pattern(
        two_theta_grid, peak_pos, peak_intensity, CRYSTALLITE_SIZE_NM
    )
    ax.fill_between(two_theta_grid, 0, continuous_intensity, color=color, alpha=0.25)
    ax.plot(two_theta_grid, continuous_intensity, color=color, linewidth=2.2, label=label)

# Formatting (kept same look as original figure)
ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
ax.set_ylim(0, 105)
ax.tick_params(axis="both", labelsize=15)
ax.set_yticks([])

# Save plot
output = "TiO2_strained-xrd.png"
plt.tight_layout()
plt.savefig(output, dpi=200)
print(f"\nLoaded CIF: {CIF_FILE}")
print(f"Saved plot: {output}")


"""
Try on your own:
- Change STRAIN_LEVEL and OUT_OF_PLANE_STRAIN and re-plot
- Keep the in-plane strain the same for a and b so tetragonal symmetry is preserved
"""
