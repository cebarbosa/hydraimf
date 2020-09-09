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
    sample_file = os.path.join(wdir, "EMCEE",
                               "fieldA_sn250_0001_seds.fits")
    snstr = os.path.split(wdir)[1]
    wave = fits.getdata(sample_file, ext=1)
    resid = np.zeros((len(table), len(wave)))
    for i, spec in enumerate(table):
        s = spec["BIN"].replace("_", "_{}_".format(snstr))
        # Using observed data
        sci_table = os.path.join(wdir, "sci", "{}.fits".format(s))
        sci = Table.read(sci_table)
        wlin = sci["wave"].data
        flam = sci["flam"].data
        flamerr = sci["flamerr"].data
        flam, flamerr = spectres(wave, wlin, flam, spec_errs=flamerr)
        norm = np.median(flam)
        flam /= norm
        flamerr /= norm
        # Getting models
        fit_table = os.path.join(wdir, "EMCEE", "{}_seds.fits".format(s))
        data = fits.getdata(fit_table)
        bestfit = np.nanmedian(data, axis=0)
        resid[i] = (flam - bestfit)
        plt.plot(wave, resid[i], "-", c="0.8")
    plt.plot(wave, resid.mean(axis=0), "r-")
    plt.show()




if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    residual = stack_residuals(wdir)