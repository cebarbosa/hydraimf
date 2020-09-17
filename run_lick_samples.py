# -*- coding: utf-8 -*-
"""

Created on 08/09/2020

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import copy
import yaml
import re

import numpy as np
import astropy.units as u
from astropy.table import Table, hstack, vstack
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter1d
from spectres import spectres
from tqdm import tqdm
try:
    import ppxf_util as util
except:
    import ppxf.ppxf_util as util
import emcee
from scipy.interpolate import interp1d

import context
from lick import Lick
import paintbox as pb
from run_paintbox import build_sed_model

import context

def run_lick_samples(wdir, dataset="MUSE", redo=False,
                     velscale=None, nsim=200, sigma=None):
    """ Calculates Lick indices on samples from full spectrum fitting. """
    velscale = context.velscale if velscale is None else velscale
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                "tables/spindex_CS.dat")
    bandsz0 = np.loadtxt(bandsfile, usecols=(3,4,1,2,5,6)) * u.AA
    units_bin = np.loadtxt(bandsfile, usecols=(7,))
    units = np.where(units_bin, u.Unit("mag"), u.Unit("angstrom"))
    names = np.loadtxt(bandsfile, usecols=(8,), dtype=str)
    names = [_.replace("_", "\_") for _ in names]
    # Setting input data
    emcee_dir = os.path.join(wdir, "EMCEE")
    tables = sorted([_ for _ in os.listdir(emcee_dir) if _.endswith(
        "results.fits")])
    lick = Lick(bandsz0, units=units)
    for tab in tables[::-1]:
        t = Table.read(os.path.join(emcee_dir, tab))
        vel = t["median"][np.where(t["param"]=="V")[0]].data * u.km / u.s
        # print(tab)
        # input()
        # s = spec["BIN"].replace("_", "_{}_".format(snstr))
        # vel = spec["V"] * u.km / u.s
        # emcee_db = os.path.join(wdir, "EMCEE", "{}.h5".format(s))
        # reader = emcee.backends.HDFBackend(emcee_db)
        # trace = reader.get_chain(discard=800, flat=True, thin=100)
        # sedcomp = []
        # for comp in comps:
        #     print(comp.parnames)
        #     idx = [sed.parnames.index(p) for p in comp.parnames]
        #     comp_sed = np.zeros((len(trace), len(wave)))
        #     for i, par in enumerate(trace[:, idx]):
        #         comp_sed[i] = comp(par)
        #     R, Ia, Im = lick(wave, comp_sed, vel=vel)
        #     plt.plot(names, np.median(Ia, axis=0), "o-")
        #     sedcomp.append(comp_sed)
        # Using observed data
        sci_table = os.path.join(wdir, "sci", tab.replace("_results", ""))
        sci = Table.read(sci_table)
        wave = sci["wave"].data
        flam = sci["flam"].data
        flamerr = sci["flamerr"].data
        # Masking wavelengths with sky lines/ residuals
        skylines = np.array([4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                             5895, 5888, 5990, 5895, 6300, 6363, 6386, 6562,
                             6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                             8747, 8757, 8767,
                             8777, 8787, 8797, 8827, 8836, 8919, 9310])
        goodpixels = np.arange(len(wave))
        for line in skylines:
            sky = np.argwhere((wave <= line - 5) | (wave >= line + 5)).ravel()
            goodpixels = np.intersect1d(goodpixels, sky)
        interpolator = interp1d(wave[goodpixels], flam[goodpixels])
        # Correcting for sky and emission lines
        R, Ia, Im = lick(wave, interpolator(wave), vel=vel)
        plt.plot(names, Ia.ravel(), "o-", label="Interpolated")
        R, Ia, Im = lick(wave, flam, vel=vel)
        plt.plot(names, Ia.ravel(), "o-", label="Original")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    run_lick_samples(wdir)