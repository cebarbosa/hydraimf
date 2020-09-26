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
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import cm
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

def get_colors(R, cmapname="Blues_r"):
    norm = Normalize(vmin=0, vmax=np.ceil(np.max(R)),
                                       clip=True)
    cmap = plt.get_cmap(cmapname)
    new_cmap = truncate_colormap(cmap, 0.0, 0.8)
    mapper = cm.ScalarMappable(norm=norm, cmap=new_cmap)
    return mapper, np.array([(mapper.to_rgba(v)) for v in R])

def truncate_colormap(cmap, minval=0.0, maxval=16, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval,
                                            b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def lick_tests(wdir, velscale=None):
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
    norm = Normalize(vmin=0, vmax=16, clip=True)
    cmap = plt.get_cmap('Blues_r')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    # sed = build_sed_model(sample="test")[0]
    mapper = cm.ScalarMappable(norm=norm, cmap=new_cmap)
    table = Table.read(os.path.join(wdir, "results.fits"))
    for i, gal in enumerate(table):
        vel = gal["V"] * u.km / u.s
        sci_table = os.path.join(wdir, "sci",
                        "{0[0]}_sn250_{0[1]}.fits".format(
                         gal["BIN"].split("_")))
        sci = Table.read(sci_table)
        if i == 0:
            wave = sci["wave"].data
            sed = build_sed_model(wave)[0]
            # Masking wavelengths with sky lines/ residuals
            skylines = np.array(
                [4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                 5895, 5888, 5990, 5895, 6300, 6363, 6386, 6562,
                 6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                 8747, 8757, 8767,
                 8777, 8787, 8797, 8827, 8836, 8919, 9310])
            goodpixels = np.arange(len(wave))
            for line in skylines:
                sky = np.argwhere(
                    (wave <= line - 2) | (wave >= line + 2)).ravel()
                goodpixels = np.intersect1d(goodpixels, sky)
        # Performing Lick measurements
        # Original data
        flam = sci["flam"].data
        ew_obs= lick(wave, flam, vel=vel)[1].ravel()
        # Interpolated, observed data
        interpolator = interp1d(wave[goodpixels], flam[goodpixels])
        ew_obs_interp = lick(wave, interpolator(wave), vel=vel)[1].ravel()
        # SSP model
        params = np.array([gal[p] for p in sed.parnames])
        ew_model= lick(wave, sed(params), vel=vel)[1].ravel()
        # SSP interpolated
        ew_model_interp = lick(wave[goodpixels], sed(params)[goodpixels],
                               vel=vel)[1].ravel()
        for j, ew in enumerate([ew_obs, ew_obs_interp, ew_model,
                               ew_model_interp]):
            plt.plot(names, ew, "o-", label="{}".format(j))
        plt.legend()
        plt.show()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    lick_tests(wdir)