# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Input structure and plotting range
CIF_FILE = "data/cif/Li2CO3.cif"
OUTPUT_FILE = "Li2CO3_ideal_vs_artifacts.png"
MIN_ANGLE = 10
MAX_ANGLE = 80
NUM_POINTS = 4000

# Cu Kalpha split settings
CU_KA1_WAVELENGTH = 1.5405929
CU_KA2_WAVELENGTH = 1.5444260
KA1_KA2_RATIO = 2.0

# Idealized profile settings
IDEAL_FWHM = 0.30
IDEAL_ETA = 0.2

# Mixed-artifact settings
ETA = 0.55
CRYSTALLITE_SIZE_NM = 18.0
MICROSTRAIN = 0.0015
U, V, W = 0.018, -0.004, 0.004
UNIFORM_SHIFT_RANGE = (-0.04, 0.04)
SAMPLE_DISPLACEMENT_RANGE_MM = (-0.15, 0.15)
GONIOMETER_RADIUS_MM = 240.0
PREFERRED_ORIENTATION = (0, 0, 1)
MARCH_DOLLASE_R = 0.7
STRAIN_IN_PLANE = -0.010
STRAIN_OUT_OF_PLANE = -0.006
BACKGROUND_SCALE = 0.16
DIFFUSE_SCALE = 0.10
AMORPHOUS_SCALE = 0.12
NOISE_LEVEL = 0.012
BASELINE_OFFSET = 0.01

# Chebyshev background shape
CHEB_COEFFS = np.array([1.0, -0.3, 0.15, -0.05, 0.02], dtype=float)

# Reproducible artifact sampling
np.random.seed(42)


def gaussian(x, center, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, center, fwhm):
    gamma = fwhm / 2.0
    return (gamma**2) / ((x - center) ** 2 + gamma**2)


def pseudo_voigt(x, center, fwhm, eta):
    return (1.0 - eta) * gaussian(x, center, fwhm) + eta * lorentzian(x, center, fwhm)


def pseudo_voigt_profile(two_theta_grid, centers, fwhm, eta):
    dx = two_theta_grid[:, None] - centers[None, :]
    sigma = np.clip(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-6, None)
    gamma = np.clip(fwhm / 2.0, 1e-6, None)
    gauss = np.exp(-0.5 * (dx / sigma[None, :]) ** 2)
    lorentz = (gamma[None, :] ** 2) / (dx**2 + gamma[None, :] ** 2)
    return (1.0 - eta) * gauss + eta * lorentz


def instrumental_fwhm(two_theta_deg):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    tan_theta = np.tan(theta_rad)
    fwhm_sq = U * tan_theta**2 + V * tan_theta + W
    return np.sqrt(np.clip(fwhm_sq, 1e-4, None))


def size_fwhm(two_theta_deg, size_nm):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    wavelength_nm = CU_KA1_WAVELENGTH / 10.0
    beta_rad = 0.9 * wavelength_nm / (size_nm * np.cos(theta_rad))
    return np.rad2deg(beta_rad)


def strain_fwhm(two_theta_deg, microstrain):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    beta_rad = 4.0 * microstrain * np.tan(theta_rad)
    return np.rad2deg(beta_rad)


def sample_displacement_shift(two_theta_deg, displacement_mm):
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    d_relative_change = displacement_mm / GONIOMETER_RADIUS_MM * np.cos(theta_rad) ** 2
    return np.rad2deg(-d_relative_change * np.tan(theta_rad))


def march_dollase_factor(hkls, preferred_orientation, r):
    hkls = np.asarray(hkls, dtype=float)
    preferred = np.asarray(preferred_orientation, dtype=float)
    hkl_mag = np.clip(np.linalg.norm(hkls, axis=1), 1e-12, None)
    pref_mag = np.clip(np.linalg.norm(preferred), 1e-12, None)
    cos_alpha = np.clip((hkls @ preferred) / (hkl_mag * pref_mag), -1.0, 1.0)
    sin_alpha_sq = 1.0 - cos_alpha**2
    r = max(float(r), 1e-6)
    return np.clip(r * r * cos_alpha**2 + sin_alpha_sq / r, 1e-12, None) ** (-1.5)


def normalize_0_100(y):
    y = np.asarray(y).flatten()
    return 100.0 * y / np.clip(y.max(), 1e-12, None)


# Load Li2CO3 and make a strained copy for the artifact panel
structure = Structure.from_file(CIF_FILE)
structure_strained = structure.copy()
structure_strained.apply_strain([STRAIN_IN_PLANE, STRAIN_IN_PLANE, STRAIN_OUT_OF_PLANE])

# Build idealized continuous profile from a CuKa stick pattern
ideal_pattern = XRDCalculator(wavelength="CuKa").get_pattern(
    structure,
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)
ideal_peak_pos = np.array(ideal_pattern.x)
ideal_peak_intensity = np.array(ideal_pattern.y)
two_theta_grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NUM_POINTS)
ideal_intensity = np.zeros_like(two_theta_grid)
for t, i in zip(ideal_peak_pos, ideal_peak_intensity):
    ideal_intensity += i * pseudo_voigt(two_theta_grid, t, IDEAL_FWHM, IDEAL_ETA)
ideal_intensity *= ideal_peak_intensity.max() / np.clip(ideal_intensity.max(), 1e-12, None)
ideal_intensity = normalize_0_100(ideal_intensity)

# Build artifact-rich profile: splitting + strain + texture + broadening
pattern_ka1 = XRDCalculator(wavelength=CU_KA1_WAVELENGTH).get_pattern(
    structure_strained,
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)
pattern_ka2 = XRDCalculator(wavelength=CU_KA2_WAVELENGTH).get_pattern(
    structure_strained,
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)

pos_ka1 = np.array(pattern_ka1.x)
int_ka1 = np.array(pattern_ka1.y)
hkls_ka1 = np.array([h[0]["hkl"] for h in pattern_ka1.hkls], dtype=float)

pos_ka2 = np.array(pattern_ka2.x)
int_ka2 = np.array(pattern_ka2.y)
hkls_ka2 = np.array([h[0]["hkl"] for h in pattern_ka2.hkls], dtype=float)

int_ka1 *= march_dollase_factor(hkls_ka1, PREFERRED_ORIENTATION, MARCH_DOLLASE_R)
int_ka2 *= march_dollase_factor(hkls_ka2, PREFERRED_ORIENTATION, MARCH_DOLLASE_R)

uniform_shift = np.random.uniform(*UNIFORM_SHIFT_RANGE)
displacement = np.random.uniform(*SAMPLE_DISPLACEMENT_RANGE_MM)
pos_ka1 = pos_ka1 + uniform_shift + sample_displacement_shift(pos_ka1, displacement)
pos_ka2 = pos_ka2 + uniform_shift + sample_displacement_shift(pos_ka2, displacement)

fwhm_ka1 = np.sqrt(
    instrumental_fwhm(pos_ka1) ** 2
    + size_fwhm(pos_ka1, CRYSTALLITE_SIZE_NM) ** 2
    + strain_fwhm(pos_ka1, MICROSTRAIN) ** 2
)
fwhm_ka2 = np.sqrt(
    instrumental_fwhm(pos_ka2) ** 2
    + size_fwhm(pos_ka2, CRYSTALLITE_SIZE_NM) ** 2
    + strain_fwhm(pos_ka2, MICROSTRAIN) ** 2
)

profile_ka1 = pseudo_voigt_profile(two_theta_grid, pos_ka1, fwhm_ka1, ETA) @ int_ka1
profile_ka2 = pseudo_voigt_profile(two_theta_grid, pos_ka2, fwhm_ka2, ETA) @ int_ka2

weight_ka1 = KA1_KA2_RATIO / (1.0 + KA1_KA2_RATIO)
weight_ka2 = 1.0 / (1.0 + KA1_KA2_RATIO)
artifact_peaks = weight_ka1 * profile_ka1 + weight_ka2 * profile_ka2
artifact_peaks *= 100.0 / np.clip(artifact_peaks.max(), 1e-12, None)

# Add smooth background + diffuse + amorphous hump + noise
x_cheb = 2.0 * (two_theta_grid - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE) - 1.0
background = np.polynomial.chebyshev.chebval(x_cheb, CHEB_COEFFS)
background = background - background.min()
background = background / np.clip(background.max(), 1e-12, None)
background = BACKGROUND_SCALE * artifact_peaks.max() * background

theta = np.deg2rad(two_theta_grid / 2.0)
diffuse = DIFFUSE_SCALE * artifact_peaks.max() * np.exp(-2.0 * np.sin(theta) ** 2)
amorphous_hump = AMORPHOUS_SCALE * artifact_peaks.max() * np.exp(
    -0.5 * ((two_theta_grid - 25.0) / 7.5) ** 2
)
noise = np.random.randn(NUM_POINTS) * NOISE_LEVEL * artifact_peaks.max()

artifact_intensity = artifact_peaks + background + diffuse + amorphous_hump + noise
artifact_intensity = artifact_intensity - artifact_intensity.min()
artifact_intensity = artifact_intensity + BASELINE_OFFSET * artifact_peaks.max()
artifact_intensity = normalize_0_100(artifact_intensity)

# Initialize a 2-panel plot: idealized (top), mixed artifacts (bottom)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharex=True)

axes[0].fill_between(two_theta_grid, 0, ideal_intensity, color="#1f4ed8", alpha=0.25)
axes[0].plot(two_theta_grid, ideal_intensity, color="#1f4ed8", linewidth=2.2)
axes[0].set_title("Idealized", fontsize=16, pad=4)

axes[1].fill_between(two_theta_grid, 0, artifact_intensity, color="#dc2626", alpha=0.25)
axes[1].plot(two_theta_grid, artifact_intensity, color="#dc2626", linewidth=2.2)
axes[1].set_title("With all artifacts mixed in", fontsize=16, pad=4)

# Formatting
for ax in axes:
    ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Intensity", fontsize=16, labelpad=10)
    ax.tick_params(axis="both", labelsize=13)

axes[-1].set_xlabel("2θ", fontsize=16, labelpad=10)

# Save plot
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200)
print(f"\nLoaded CIF: {CIF_FILE.split('/')[-1]}")
print(f"Saved plot: {OUTPUT_FILE}")


"""
Try on your own:
- Change PREFERRED_ORIENTATION and MARCH_DOLLASE_R to explore texture strength
- Increase/decrease NOISE_LEVEL, BACKGROUND_SCALE, and AMORPHOUS_SCALE
- Change STRAIN_IN_PLANE / STRAIN_OUT_OF_PLANE and compare peak-position shifts
"""
