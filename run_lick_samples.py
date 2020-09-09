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

import context
from lick import Lick
import paintbox as pb

import context

def build_sed_model(wave, w1=4500, w2=9400, velscale=200, sample=None,
                    fwhm=2.95, em_oversample=8, porder=30, nssps=1):
    """ Build model for NGC 3311"""
    # Preparing templates
    sample = "all" if sample is None else sample
    templates_file = os.path.join(context.home, "templates",
        "emiles_muse_vel{}_w{}_{}_{}_fwhm{}.fits".format(velscale, w1, w2,
                                        sample, fwhm))
    templates = fits.getdata(templates_file, ext=0)
    tnorm = np.median(templates, axis=1)
    templates /= tnorm[:, None]
    params = Table.read(templates_file, hdu=1)
    limits = {}
    for i, param in enumerate(params.colnames):
        vmin, vmax = params[param].min(), params[param].max()
        limits[param] = (vmin, vmax)
    logwave = Table.read(templates_file, hdu=2)["loglam"].data
    twave = np.exp(logwave)
    ssp = pb.StPopInterp(twave, params, templates)
    # Using interpolation routine to get normalization
    norm = pb.StPopInterp(np.ones(1), params, tnorm) if nssps > 1 else None
    if nssps > 1:
        for i in range(nssps):
            w = pb.Polynomial(twave, 0)
            w.parnames = ["w_{}".format(i+1)]
            s = copy.deepcopy(ssp)
            s.parnames = ["{}_{}".format(_, i+1) for _ in s.parnames]
            if i == 0:
                pop = w * s
            else:
                pop += (w * s)
    else:
        pop = ssp
    # Adding extinction to the stellar populations
    extinction = pb.CCM89(twave)
    stars = pb.Rebin(wave, pb.LOSVDConv(pop * extinction, velscale=velscale))
    # Loading templates for the emission lines
    velscale_gas = velscale / em_oversample
    logemwave = util.log_rebin([w1, w2], twave, velscale=velscale_gas)[1]
    emwave = np.exp(logemwave)
    emission_lines, line_names, line_wave = util.emission_lines(logemwave,
            [emwave[0], emwave[-1]], fwhm)
    line_names = [_.replace("[", "").replace("]", "").replace("_", "") for _ in
                  line_names]
    emission_lines = emission_lines.T
    emnorm = emission_lines.max(axis=1)
    emission_lines /= emnorm[:, None]
    emission = pb.Rebin(wave,
                         pb.LOSVDConv(pb.EmissionLines(emwave, emission_lines,
                         line_names), velscale=velscale_gas))
    emission.parnames[-1] = "sigma_gas"
    emission.parnames[-2] = "V_gas"
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    # Using skycalc model to model residuals
    sky_templates_file = os.path.join(context.data_dir,
                                      "sky/sky_templates.fits")
    wsky = Table.read(sky_templates_file, hdu=1)["wave"].data
    fsky = fits.getdata(sky_templates_file, hdu=1)
    snames = Table.read(sky_templates_file, hdu=2)["skylines"].data
    snames = ["sky" + re.sub(r"[\(\)$\-\_]", "", _.decode()) for _ in snames]
    sky = pb.Rebin(wave, pb.EmissionLines(wsky, fsky, snames))
    # Creating a model including LOSVD
    sed = (stars * poly) + sky + emission
    # Setting properties that may be useful later in modeling
    sed.ssppars = limits
    sed.sspcolnames = params.colnames
    sed.sspparams = params
    sed.line_names = line_names
    sed.porder = porder
    sed.nssps = nssps
    return stars, sed

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
    sigma_str = "" if sigma is None else "_sigma{}".format(sigma)
    # Setting input data
    tfile = os.path.join(wdir, "results.fits")
    snstr = os.path.split(wdir)[1]
    table = Table.read(os.path.join(wdir, "results.fits"))
    # Producing model
    sample_file = os.path.join(wdir, "EMCEE",
                               "fieldA_sn250_0001_seds.fits")
    wave = fits.getdata(sample_file, ext=1)
    comps = build_sed_model(wave)
    sed = comps[-1]
    lick = Lick(bandsz0, units=units)
    for spec in table:
        s = spec["BIN"].replace("_", "_{}_".format(snstr))
        vel = spec["V"] * u.km / u.s
        emcee_db = os.path.join(wdir, "EMCEE", "{}.h5".format(s))
        reader = emcee.backends.HDFBackend(emcee_db)
        trace = reader.get_chain(discard=800, flat=True, thin=100)
        sedcomp = []
        for comp in comps:
            print(comp.parnames)
            idx = [sed.parnames.index(p) for p in comp.parnames]
            comp_sed = np.zeros((len(trace), len(wave)))
            for i, par in enumerate(trace[:, idx]):
                comp_sed[i] = comp(par)
            R, Ia, Im = lick(wave, comp_sed, vel=vel)
            plt.plot(names, np.median(Ia, axis=0), "o-")
            sedcomp.append(comp_sed)
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
        # Correcting for sky and emission lines
        R, Ia, Im = lick(wave, flam, vel=vel)
        plt.plot(names, Ia.ravel(), "o-")
        plt.show()

if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    run_lick_samples(wdir)