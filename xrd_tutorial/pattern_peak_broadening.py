# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# To load structures and compute XRD patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Input structure and plotting range
CIF_FILE = "data/cif/TiO2.cif"
MIN_ANGLE = 10
MAX_ANGLE = 80
NUM_POINTS = 4000

# Scherrer broadening settings
WAVELENGTH_ANGSTROM = 1.5406  # Cu Kalpha
K_FACTOR = 0.9
PROFILES = [
    ("Large particles", 30.0),
    ("Moderate size", 10.0),
    ("Very small", 5.0),
]

# Color map used across the 3 particle-size profiles
PROFILE_CMAP = LinearSegmentedColormap.from_list(
    "blue_purple_red",
    ["#1f4ed8", "#7e22ce", "#dc2626"],
)


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


# Initialize XRD calculator on TiO2
pattern = XRDCalculator(wavelength=WAVELENGTH_ANGSTROM).get_pattern(
    Structure.from_file(CIF_FILE),
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)

# Extract discrete peak positions and intensities
peak_pos = np.array(pattern.x)
peak_intensity = np.array(pattern.y)
peak_intensity = 100.0 * peak_intensity / peak_intensity.max()

# Build a high-resolution 2theta grid for continuous profiles
two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)

# Initialize a 3-panel plot (one panel per crystallite size)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 7), sharex=True)
colors = [PROFILE_CMAP(v) for v in np.linspace(0.0, 1.0, len(PROFILES))]

# Broaden the same stick pattern using different crystallite sizes
for ax, color, (_, size_nm) in zip(axes, colors, PROFILES):
    continuous_intensity = broaden_pattern(two_theta_grid, peak_pos, peak_intensity, size_nm)
    ax.fill_between(two_theta_grid, 0, continuous_intensity, color=color, alpha=0.25)
    ax.plot(two_theta_grid, continuous_intensity, color=color, linewidth=2.2)
    ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Intensity", fontsize=18, labelpad=12)
    ax.tick_params(axis="both", labelsize=15)

axes[-1].set_xlabel("2θ", fontsize=18, labelpad=12)

# Save plot
output = "TiO2_peak_broadening.png"
plt.tight_layout()
plt.savefig(output, dpi=200)
print(f"\nLoaded CIF: {CIF_FILE}")
print(f"Saved plot: {output}")
print("Smaller crystallite size gives broader peaks (larger Scherrer FWHM).")

"""
Try on your own:
- Change the particle sizes in PROFILES and re-plot the XRD patterns
"""
