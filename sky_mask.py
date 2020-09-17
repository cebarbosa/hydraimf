"""
Find residual emission lines from the fitting to be excluded in a second
fitting.
"""
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from spectres import spectres
import matplotlib.pyplot as plt

import context

def stack_residuals(wdir):
    """ Get residuals from fit. """
    table = Table.read(os.path.join(wdir, "results.fits"))
    # Producing model
    snstr = os.path.split(wdir)[1]
    for i, spec in enumerate(table[::-1]):
        s = spec["BIN"].replace("_", "_{}_".format(snstr))
        # Using observed data
        sci_table = os.path.join(wdir, "sci", "{}.fits".format(s))
        sci = Table.read(sci_table)
        wlin = sci["wave"].data
        flam = sci["flam"].data
        flamerr = sci["flamerr"].data
        # Getting models
        fit_table = os.path.join(wdir, "EMCEE", "{}_seds.fits".format(s))
        wave = fits.getdata(fit_table, ext=1)
        data = fits.getdata(fit_table)
        bestfit = np.nanmedian(data, axis=0)
        flam, flamerr = spectres(wave, wlin, flam, spec_errs=flamerr)
        norm = np.median(flam)
        flam /= norm
        flamerr /= norm
        res = flam - bestfit
        plt.plot(wave, flam, "-")
        plt.plot(wave, bestfit, "-")
        plt.plot(wave, res, "-", c="0.8")
        plt.show()




if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    stack_residuals(wdir)