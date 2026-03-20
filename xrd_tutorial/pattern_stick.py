# For handling arrays
import numpy as np

# For plotting
import matplotlib.pyplot as plt

# To load structures and compute XRD patterns
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# User-facing controls for plot range
MIN_ANGLE = 10
MAX_ANGLE = 80

# Initialize XRD calculator on NaCl
pattern = XRDCalculator(wavelength="CuKa").get_pattern(
    Structure.from_file("data/cif/NaCl.cif"),
    two_theta_range=(MIN_ANGLE, MAX_ANGLE),
)

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot discrete peaks as vertical lines
ax.vlines(pattern.x, 0, pattern.y, color="black", linewidth=6.5)

# Formatting
ax.set_xlim(MIN_ANGLE, MAX_ANGLE)
ax.set_ylim(0, max(pattern.y) * 1.05)
ax.set_xlabel("2θ", fontsize=18, labelpad=12)
ax.set_ylabel("Intensity", fontsize=18, labelpad=12)
ax.tick_params(axis="both", labelsize=15)

# Save plot
plt.tight_layout()
plt.savefig("NaCl_stick_pattern.png", dpi=200)
print("\nSaved plot: NaCl_stick_pattern.png")


"""
Try on your own:
- Vary the wavelength (1.5406 Å is the CuKa value; you can specify others)
- Load other structures and plot their XRD stick patterns
- Check how peaks look at higher values of two-theta
"""
