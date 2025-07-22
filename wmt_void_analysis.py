import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load galaxy data
galaxies = pd.read_csv("sdss_galaxies.csv")
voids = fits.open("void_catalog.fits")[1].data

# Match and compute redshift residuals
residuals = []
for void in voids:
    mask = (
        (galaxies['ra'] > void['RA'] - 5) & (galaxies['ra'] < void['RA'] + 5) &
        (galaxies['dec'] > void['DEC'] - 5) & (galaxies['dec'] < void['DEC'] + 5)
    )
    local = galaxies[mask]
    if len(local) > 0:
        residual = local['z'].mean() - void['Z']
        residuals.append(residual * 1e6)

# Save results
with open("wmt_redshift_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Measured {len(residuals)} redshift residuals\n")
    f.write(f"Mean residual: {np.mean(residuals):.2f} μz\n")
    f.write(f"Standard deviation: {np.std(residuals):.2f} μz\n")

# Plot
plt.hist(residuals, bins=30, alpha=0.75, color="darkblue")
plt.xlabel("Redshift Residual (μz)")
plt.ylabel("Frequency")
plt.title("White Matter Theory: Void Redshift Residuals")
plt.tight_layout()
plt.savefig("wmt_redshift_plot.png")
