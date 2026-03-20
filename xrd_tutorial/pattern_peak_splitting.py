# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Plotting range and peak width
MIN_ANGLE = 10
MAX_ANGLE = 80

NUM_POINTS = 4000  # number of points in XRD pattern
FWHM = 0.1  # full width at half maximum
GAUSS_FRAC = 0.2  # fraction gaussian (vs. lorentzian)

# Cu Kalpha doublet values used in galaxi
CU_KA1_WAVELENGTH = 1.5405929
CU_KA2_WAVELENGTH = 1.5444260
CU_KA1_KA2_RATIO = 2.0  # Kalpha1:Kalpha2 intensity ratio


def gaussian(x, center, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, center, fwhm):
    gamma = fwhm / 2.0
    return (gamma**2) / ((x - center) ** 2 + gamma**2)


def pseudo_voigt(x, center, fwhm, eta):
    return (1.0 - eta) * gaussian(x, center, fwhm) + eta * lorentzian(x, center, fwhm)


# Load NaCl structure once
structure = Structure.from_file("data/cif/NaCl.cif")

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

# Kalpha1/Kalpha2 intensity weights
weight_ka1 = CU_KA1_KA2_RATIO / (1.0 + CU_KA1_KA2_RATIO)
weight_ka2 = 1.0 / (1.0 + CU_KA1_KA2_RATIO)

# Build a high-resolution 2theta grid for a continuous profile
two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)
continuous_intensity = np.zeros_like(two_theta_grid)

# Broaden Kalpha1 peaks into pseudo-Voigt line shapes
for t, i in zip(peak_pos_ka1, peak_intensity_ka1):
    continuous_intensity += weight_ka1 * i * pseudo_voigt(two_theta_grid, t, FWHM, GAUSS_FRAC)

# Broaden Kalpha2 peaks into pseudo-Voigt line shapes
for t, i in zip(peak_pos_ka2, peak_intensity_ka2):
    continuous_intensity += weight_ka2 * i * pseudo_voigt(two_theta_grid, t, FWHM, GAUSS_FRAC)

# Keep a similar intensity scale as previous examples.
continuous_intensity *= 100.0 / continuous_intensity.max()

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot continuous profile as a filled curve with an outline
ax.fill_between(two_theta_grid, 0, continuous_intensity, color="blue", alpha=0.25)
ax.plot(two_theta_grid, continuous_intensity, color="darkblue", linewidth=1.5)

# Formatting
ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
ax.set_ylim(0, continuous_intensity.max() * 1.05)
ax.set_xlabel("2θ", fontsize=18, labelpad=12)
ax.set_ylabel("Intensity", fontsize=18, labelpad=12)
ax.tick_params(axis="both", labelsize=15)

# Save plot
plt.tight_layout()
plt.savefig("NaCl_peak_splitting_pattern.png", dpi=200)
print("\nSaved plot: NaCl_peak_splitting_pattern.png")

"""
Let's zoom in on a peak at high two-theta
to better visualize the splitting effect
"""
# Initialize plot
fig, ax = plt.subplots(figsize=(6, 4))

# Plot continuous profile as a filled curve with an outline
ax.fill_between(two_theta_grid, 0, continuous_intensity, color="blue", alpha=0.25)
ax.plot(two_theta_grid, continuous_intensity, color="darkblue", linewidth=1.5)

# Zoomed in
ax.set_xlim(75, 77.5)
ax.set_ylim(0, 28)

# Save plot
plt.tight_layout()
plt.savefig("NaCl_zoomed.png", dpi=200)
print("\nSaved plot: NaCl_zoomed.png")

"""
Try on your own:
- Include higher two-theta to see more visible splitting
- Check how the peak width (larger or smaller FWHM) affects splitting
"""
