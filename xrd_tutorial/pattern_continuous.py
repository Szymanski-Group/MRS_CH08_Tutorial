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

NUM_POINTS = 4000 # number of points in XRD pattern
FWHM = 0.3 # full width at half maximum
GAUSS_FRAC = 0.2 # fraction gaussian (vs. lorentzian)


def gaussian(x, center, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, center, fwhm):
    gamma = fwhm / 2.0
    return (gamma**2) / ((x - center) ** 2 + gamma**2)


def pseudo_voigt(x, center, fwhm, eta):
    return (1.0 - eta) * gaussian(x, center, fwhm) + eta * lorentzian(x, center, fwhm)


# Initialize XRD calculator on NaCl
pattern = XRDCalculator(wavelength="CuKa").get_pattern(
    Structure.from_file("data/cif/NaCl.cif"),
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)

# Extract discrete peak positions and intensities
peak_pos = np.array(pattern.x)
peak_intensity = np.array(pattern.y)

# Build a high-resolution 2theta grid for a continuous profile
two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)
continuous_intensity = np.zeros_like(two_theta_grid)

# Broaden each stick peak into a pseudo-Voigt line shape
for t, i in zip(peak_pos, peak_intensity):
    continuous_intensity += i * pseudo_voigt(two_theta_grid, t, FWHM, GAUSS_FRAC)

# Keep peak scale similar to the original stick pattern.
continuous_intensity *= peak_intensity.max() / continuous_intensity.max()

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot continuous profile as a filled curve with an outline
ax.fill_between(two_theta_grid, 0, continuous_intensity, color="blue", alpha=0.25)
ax.plot(two_theta_grid, continuous_intensity, color="darkblue", linewidth=2.2)

# Formatting
ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
ax.set_ylim(0, continuous_intensity.max() * 1.05)
ax.set_xlabel("2θ", fontsize=18, labelpad=12)
ax.set_ylabel("Intensity", fontsize=18, labelpad=12)
ax.tick_params(axis="both", labelsize=15)

# Save plot
plt.tight_layout()
plt.savefig("NaCl_continuous_pattern.png", dpi=200)
print("\nSaved plot: NaCl_continuous_pattern.png")

"""
Try on your own:
- Use broader peaks (larger FWHM)
- Changing the Gaussian/Lorentzian fraction (GAUSS_FRAC)
- Loading other structures and plotting their continuous XRD patterns
"""
