""" Plot with surface photometry comparison. """
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from photutils import CircularAnnulus
from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats

import context

if __name__ == "__main__":
    # Data from MUSE
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    tfile = os.path.join(wdir, "results.fits")
    table = Table.read(tfile)
    # V-band photometry
    phot_dir = os.path.join(context.data_dir, "FORS2")
    vband = fits.getdata(os.path.join(phot_dir, "hydra1.fits"))
    mask = fits.getdata(os.path.join(phot_dir, "mask.fits"))
    sky = 4764.883
    exptime = 479.9798
    ps = 0.252
    muv = -2.5 * np.log10((vband - sky) / exptime) + 27.2 + \
          2.5 * np.log10(ps**2)
    positions = [(573, 1255)]
    rs = np.arange(1, 250)
    mu_V = np.zeros(len(rs))
    for i, r in enumerate(rs):
        annulus_aperture = CircularAnnulus(positions, r_in=r, r_out=r+1)
        annulus_mask = annulus_aperture.to_mask(method='center')[0]
        data = annulus_mask.multiply(muv)
        data = data[data !=0]
        _, median_sigclip, _ = sigma_clipped_stats(data)
        mu_V[i] = np.mean(data)
    ax = plt.subplot(111)
    ax.plot(table["R"] , table["mu_r"], "o", label="$\mu_r$")
    ax.plot((rs + 0.5) * 0.25 * 0.26, mu_V - 0.8, label="$\mu_V$ - 0.8")
    ax.invert_yaxis()
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel("$\mu$ (mag arcsec$^{-2}$)")
    plt.legend()
    plt.savefig(os.path.join(wdir, "plots/sbp_comparison.png"))
    plt.show()
