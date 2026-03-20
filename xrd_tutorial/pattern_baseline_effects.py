# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Reproducible random noise
np.random.seed(42)

# Input structure and plotting range
CIF_FILE = "data/cif/LiMnO2.cif"
MIN_ANGLE = 10
MAX_ANGLE = 80
NUM_POINTS = 4000

# Peak profile settings
ETA = 0.5  # pseudo-Voigt mixing (0 = Gaussian, 1 = Lorentzian)
CRYSTALLITE_SIZE_NM = 20.0
MICROSTRAIN = 0.001

# Instrument broadening (Caglioti; matches galaxi defaults)
U = 0.04
V = -0.01
W = 0.006

# Cu Kalpha split settings (galaxi defaults)
CU_KA1_WAVELENGTH = 1.5405929
CU_KA2_WAVELENGTH = 1.5444260
KA1_KA2_RATIO = 2.0  # Kalpha1:Kalpha2 intensity ratio

# Background and noise settings
CHEB_COEFFS = np.array([1.0, -0.3, 0.15, -0.05, 0.02])
BACKGROUND_SCALE = 0.24
NOISE_LEVEL = 0.005
BASELINE_OFFSET = 0.01


def instrumental_fwhm(two_theta_deg):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    tan_theta = np.tan(theta_rad)
    fwhm_sq = U * tan_theta**2 + V * tan_theta + W
    return np.sqrt(np.clip(fwhm_sq, 1e-4, None))


def size_fwhm(two_theta_deg):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    wavelength_nm = CU_KA1_WAVELENGTH / 10.0
    beta_rad = 0.9 * wavelength_nm / (CRYSTALLITE_SIZE_NM * np.cos(theta_rad))
    return np.rad2deg(beta_rad)


def strain_fwhm(two_theta_deg):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    beta_rad = 4.0 * MICROSTRAIN * np.tan(theta_rad)
    return np.rad2deg(beta_rad)


def pseudo_voigt_profile(two_theta_grid, centers, fwhm, eta):
    dx = two_theta_grid[:, None] - centers[None, :]
    sigma = np.clip(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-6, None)
    gamma = np.clip(fwhm / 2.0, 1e-6, None)
    gauss = np.exp(-0.5 * (dx / sigma[None, :]) ** 2)
    lorentz = (gamma[None, :] ** 2) / (dx**2 + gamma[None, :] ** 2)
    return (1.0 - eta) * gauss + eta * lorentz


# Load structure once
structure = Structure.from_file(CIF_FILE)

# Compute stick patterns separately for Cu Kalpha1 and Kalpha2
pattern_ka1 = XRDCalculator(wavelength=CU_KA1_WAVELENGTH).get_pattern(
    structure,
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)
pattern_ka2 = XRDCalculator(wavelength=CU_KA2_WAVELENGTH).get_pattern(
    structure,
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)

# Extract discrete peak positions and intensities
peak_pos_ka1 = np.array(pattern_ka1.x)
peak_intensity_ka1 = np.array(pattern_ka1.y)
peak_pos_ka2 = np.array(pattern_ka2.x)
peak_intensity_ka2 = np.array(pattern_ka2.y)

# Build a high-resolution 2theta grid for a continuous profile
two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)

# Combine instrument + size + strain broadening for each peak
fwhm_ka1 = np.sqrt(
    instrumental_fwhm(peak_pos_ka1) ** 2
    + size_fwhm(peak_pos_ka1) ** 2
    + strain_fwhm(peak_pos_ka1) ** 2
)
fwhm_ka2 = np.sqrt(
    instrumental_fwhm(peak_pos_ka2) ** 2
    + size_fwhm(peak_pos_ka2) ** 2
    + strain_fwhm(peak_pos_ka2) ** 2
)

# Kalpha1/Kalpha2 intensity weights
weight_ka1 = KA1_KA2_RATIO / (1.0 + KA1_KA2_RATIO)
weight_ka2 = 1.0 / (1.0 + KA1_KA2_RATIO)

# Broaden and combine Kalpha1 + Kalpha2 peaks
profile_ka1 = pseudo_voigt_profile(two_theta_grid, peak_pos_ka1, fwhm_ka1, ETA) @ peak_intensity_ka1
profile_ka2 = pseudo_voigt_profile(two_theta_grid, peak_pos_ka2, fwhm_ka2, ETA) @ peak_intensity_ka2
peaks = weight_ka1 * profile_ka1 + weight_ka2 * profile_ka2
peaks *= max(peak_intensity_ka1.max(), peak_intensity_ka2.max()) / peaks.max()

# Add smooth Chebyshev background
x_cheb = 2.0 * (two_theta_grid - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE) - 1.0
background_shape = np.polynomial.chebyshev.chebval(x_cheb, CHEB_COEFFS)
background_shape -= background_shape.min()
background_shape /= background_shape.max()
background = BACKGROUND_SCALE * peaks.max() * background_shape

# Add Gaussian counting noise
noise = np.random.randn(NUM_POINTS) * NOISE_LEVEL * peaks.max()

# Final intensity (keep positive baseline for plotting)
intensity = peaks + background + noise
intensity -= intensity.min()
intensity += BASELINE_OFFSET * peaks.max()

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot continuous profile as a filled curve with an outline
ax.fill_between(two_theta_grid, 0, intensity, color="black", alpha=0.15)
ax.plot(two_theta_grid, intensity, color="black", linewidth=2.2)

# Formatting
ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
ax.set_ylim(0, intensity.max() * 1.05)
ax.set_xlabel("2θ", fontsize=18, labelpad=12)
ax.set_ylabel("Intensity", fontsize=18, labelpad=12)
ax.tick_params(axis="both", labelsize=15)

# Save plot
plt.tight_layout()
plt.savefig("LiMnO2_xrd_with_background.png", dpi=200)
print("\nSaved plot: LiMnO2_xrd_with_background.png")

"""
Try on your own:
- Change CHEB_COEFFS to make the baseline flatter or more curved
- Increase/decrease BACKGROUND_SCALE to control baseline height
- Increase/decrease NOISE_LEVEL to simulate cleaner/noisier data
"""
