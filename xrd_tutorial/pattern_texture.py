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

# Peak profile settings
FWHM = 0.4  # full width at half maximum
GAUSS_FRAC = 0.2  # fraction gaussian (vs. lorentzian)

# Texture settings (March-Dollase)
PREFERRED_ORIENTATION = (0, 0, 1)
MARCH_DOLLASE_R = 0.65  # 1.0 = random orientation (no texture)


def gaussian(x, center, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, center, fwhm):
    gamma = fwhm / 2.0
    return (gamma**2) / ((x - center) ** 2 + gamma**2)


def pseudo_voigt(x, center, fwhm, eta):
    return (1.0 - eta) * gaussian(x, center, fwhm) + eta * lorentzian(x, center, fwhm)


def march_dollase_factor(hkls, preferred_orientation, r):
    # Same core form used in galaxi:
    # P(alpha) = (r^2 cos^2(alpha) + sin^2(alpha)/r)^(-3/2)
    hkls = np.asarray(hkls, dtype=float)
    preferred = np.asarray(preferred_orientation, dtype=float)

    preferred_mag = np.linalg.norm(preferred)
    if preferred_mag < 1e-12:
        raise ValueError("Preferred orientation vector must be non-zero.")

    hkl_mag = np.linalg.norm(hkls, axis=1)
    hkl_mag = np.clip(hkl_mag, 1e-12, None)

    cos_alpha = (hkls @ preferred) / (hkl_mag * preferred_mag)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    sin_alpha_sq = 1.0 - cos_alpha**2

    r = max(float(r), 1e-6)
    numerator = r * r * cos_alpha**2 + sin_alpha_sq / r
    numerator = np.clip(numerator, 1e-12, None)
    return numerator ** (-1.5)


def continuous_profile(two_theta_grid, peak_pos, peak_intensity, fwhm, eta):
    intensity = np.zeros_like(two_theta_grid)
    for t, i in zip(peak_pos, peak_intensity):
        intensity += i * pseudo_voigt(two_theta_grid, t, fwhm, eta)
    return 100.0 * intensity / intensity.max()


# Load TiO2 structure and compute stick pattern
pattern = XRDCalculator(wavelength="CuKa").get_pattern(
    Structure.from_file(CIF_FILE),
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)

# Extract discrete peak positions, intensities, and hkls
peak_pos = np.array(pattern.x)
peak_intensity = np.array(pattern.y)
peak_hkls = np.array([h[0]["hkl"] for h in pattern.hkls], dtype=float)

# No-texture intensities (random orientation)
intensity_no_texture = peak_intensity.copy()

# Apply preferred orientation correction to stick intensities
texture_scale = march_dollase_factor(peak_hkls, PREFERRED_ORIENTATION, MARCH_DOLLASE_R)
intensity_with_texture = peak_intensity * texture_scale

# Build a high-resolution 2theta grid for continuous profiles
two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)
continuous_no_texture = continuous_profile(
    two_theta_grid, peak_pos, intensity_no_texture, FWHM, GAUSS_FRAC
)
continuous_with_texture = continuous_profile(
    two_theta_grid, peak_pos, intensity_with_texture, FWHM, GAUSS_FRAC
)

# Initialize a 2-panel plot: without texture (top), with texture (bottom)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharex=True)

axes[0].fill_between(two_theta_grid, 0, continuous_no_texture, color="#1f4ed8", alpha=0.25)
axes[0].plot(two_theta_grid, continuous_no_texture, color="#1f4ed8", linewidth=2.2)
axes[0].set_title("Without texture", fontsize=16, pad=4)

axes[1].fill_between(two_theta_grid, 0, continuous_with_texture, color="#dc2626", alpha=0.25)
axes[1].plot(two_theta_grid, continuous_with_texture, color="#dc2626", linewidth=2.2)
axes[1].set_title(
    f"With Texture",
    fontsize=16,
    pad=4,
)

# Formatting
for ax in axes:
    ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Intensity", fontsize=16, labelpad=10)
    ax.tick_params(axis="both", labelsize=13)

axes[-1].set_xlabel("2θ", fontsize=16, labelpad=10)

# Save plot
output = "TiO2_texture_comparison.png"
plt.tight_layout()
plt.savefig(output, dpi=200)
print(f"\nLoaded CIF: {CIF_FILE}")
print(f"Preferred orientation: {PREFERRED_ORIENTATION}")
print(f"March-Dollase r (textured panel): {MARCH_DOLLASE_R}")
print(f"Saved plot: {output}")


"""
Try on your own:
- Change PREFERRED_ORIENTATION, e.g., (1, 0, 0), (1, 1, 0), (1, 1, 1)
- Apply weaker/stronger texture by changing MARCH_DOLLASE_R (closer/farther from 1.0)
"""
