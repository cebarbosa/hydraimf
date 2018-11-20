# -*- coding: utf-8 -*-
""" 

Created on 04/12/17

Author : Carlos Eduardo Barbosa

Prepare stellar population model as templates for pPXF and SSP fitting.

"""
from __future__ import print_function, division

import os
from itertools import product
from datetime import datetime

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io import fits

from specutils.io import read_fits

import ppxf.ppxf_util as util

import context
from muse_resolution import broad2res

class EMiles_models():
    """ Class to handle data from the EMILES SSP models. """
    def __init__(self, sample=None, path=None, w1=4500, w2=10000):
        if path is None:
            self.path = os.path.join(context.home,
                        "models/EMILES_BASTI_w{}_{}".format(w1, w2))
        else:
            self.path = path
        self.sample = "all" if sample is None else sample
        if self.sample not in ["all", "bsf", "salpeter", "minimal",
                               "kinematics"]:
            raise ValueError("EMILES sample not defined: {}".format(
                self.sample))
        self.values = self.Values(self.sample)

    class Values():
        """ Defines the possible values of the model parameters in different
        samples.
        """
        def __init__(self, sample):
            if sample == "all":
                self.exponents = np.array(
                    [0.3, 0.5, 0.8, 1.0, 1.3, 1.5,
                     1.8, 2.0, 2.3, 2.5, 2.8, 3.0,
                     3.3, 3.5])
                self.ZH = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                     0.15,  0.26,  0.4])
                self.age = np.linspace(1., 14., 27)
                self.alphaFe = np.array([0., 0.2, 0.4])
                self.NaFe = np.array([0., 0.3, 0.6])
            elif sample == "minimal":
                self.exponents = np.array([1.3, 1.5])
                self.ZH = np.array([0.06, 0.15])
                self.age = np.array([10., 14.])
                self.alphaFe = np.array([0., 0.2])
                self.NaFe = np.array([0., 0.3])
            elif sample == "salpeter":
                self.exponents = np.array([1.3])
                self.ZH = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                     0.15,  0.26,  0.4])
                self.age = np.linspace(1., 14., 14)
                self.alphaFe = np.array([0., 0.2, 0.4])
                self.NaFe = np.array([0., 0.3, 0.6])
            if sample == "bsf":
                self.exponents = np.array([0.3, 0.8, 1.3, 1.8, 2.3])
                self.ZH = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                     0.15,  0.26,  0.4])
                self.age = np.linspace(1., 14., 14)
                self.alphaFe = np.array([0., 0.2, 0.4])
                self.NaFe = np.array([0., 0.3, 0.6])
            if sample == "kinematics":
                self.exponents = np.array([1.3])
                self.ZH = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                     0.15,  0.26,  0.4])
                self.age = np.linspace(1., 14., 14)
                self.alphaFe = np.array([0., 0.4])
                self.NaFe = np.array([0., 0.6])


            return

    def get_filename(self, imf, metal, age, alpha, na):
        """ Returns the name of files for the EMILES library. """
        msign = "p" if metal >= 0. else "m"
        esign = "p" if alpha >= 0. else "m"
        azero = "0" if age < 10. else ""
        nasign = "p" if na >= 0. else "m"
        return "Ebi{0:.2f}Z{1}{2:.2f}T{3}{4:02.4f}_Afe{5}{6:2.1f}_NaFe{7}{" \
               "8:1.1f}.fits".format(imf, msign, abs(metal), azero, age, esign,
                                     abs(alpha), nasign, na)

def trim_templates(emiles, w1=4500, w2=10000, redo=False):
    """ Slice spectra from templates according to wavelength range. """
    newpath = os.path.join(context.home, "models/EMILES_BASTI_w{}_{}".format(
                           w1, w2))
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    for args in product(emiles.values.exponents, emiles.values.ZH,
                        emiles.values.age, emiles.values.alphaFe,
                        emiles.values.NaFe):
        filename = os.path.join(emiles.path, emiles.get_filename(*args))
        newfilename = os.path.join(newpath, emiles.get_filename(*args))
        if os.path.exists(newfilename):
            continue
        spec = read_fits.read_fits_spectrum1d(filename)
        idx = np.where(np.logical_and(spec.dispersion > w1, spec.dispersion
                                      < w2))
        tab = Table([spec.dispersion[idx] * u.AA, spec.flux[idx] * u.adu],
                    names=["wave", "flux"])
        tab.write(newfilename, format="fits")
        print("Created file ", newfilename)
    return

def prepare_templates_emiles_muse(w1, w2, velscale, sample="all", redo=False,
                                  fwhm=None, instrument=None):
    """ Pipeline for the preparation of the templates."""
    instrument = "muse" if instrument is None else instrument
    output = os.path.join(context.home, "templates",
            "emiles_{}_vel{}_w{}_{}_{}.fits".format(instrument, velscale, w1,
                                                    w2, sample))
    if os.path.exists(output) and not redo:
        return
    fwhm = 2.95 if fwhm is None else fwhm
    emiles_base = EMiles_models(path=os.path.join(context.home, "models",
                              "EMILES_BASTI_INTERPOLATED"), sample=sample)
    trim_templates(emiles_base, w1=w1, w2=w2, redo=False)
    emiles = EMiles_models(path=os.path.join(context.home, "models",
                           "EMILES_BASTI_w{}_{}".format(w1, w2)), sample=sample)
    # First part: getting SSP templates
    grid = np.array(np.meshgrid(emiles.values.exponents, emiles.values.ZH,
                             emiles.values.age, emiles.values.alphaFe,
                             emiles.values.NaFe)).T.reshape(-1, 5)

    filenames = []
    for args in grid:
        filenames.append(os.path.join(emiles.path, emiles.get_filename(*args)))
    dim = len(grid)
    params = np.zeros((dim, 5))
    # Using first spectrum to build arrays
    filename = os.path.join(emiles.path, emiles.get_filename(*grid[0]))
    spec = Table.read(filename, format="fits")
    wave = spec["wave"]
    wrange = [wave[0], wave[-1]]
    flux = spec["flux"]
    newflux, logLam, velscale = util.log_rebin(wrange, flux,
                                               velscale=velscale)
    ssps = np.zeros((dim, len(logLam)))
    # Iterate over all models
    newfolder = os.path.join(context.home, "templates", \
                             "vel{}_w{}_{}".format(velscale, w1, w2))
    if not os.path.exists(newfolder):
        os.mkdir(newfolder)
    ############################################################################
    # Sub routine to process a single spectrum
    def process_spec(filename, velscale, redo=False):
        outname = os.path.join(newfolder, os.path.split(filename)[1])
        if os.path.exists(outname) and not redo:
            return
        spec = Table.read(filename, format="fits")
        flux = spec["flux"].data
        if fwhm > 2.51:
            flux = broad2res(wave, flux.T, np.ones_like(wave) * 2.51,
                             res=fwhm)[0].T
        newflux, logLam, velscale = util.log_rebin(wrange, flux,
                                               velscale=velscale)
        hdu = fits.PrimaryHDU(newflux)
        hdu.writeto(outname, overwrite=True)
        return
    ############################################################################
    for i, fname in enumerate(filenames):
        print("Processing SSP {}".format(i+1))
        process_spec(fname, velscale)
    for i, args in enumerate(grid):
        print("Processing SSP {}".format(i+1))
        filename = os.path.join(newfolder, emiles.get_filename(*args))
        data = fits.getdata(filename)
        ssps[i] = data
        params[i] = args
    # Second part : emission line templates
    emlines = context.get_emission_lines()
    sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))
    wave = np.exp(logLam)
    emission= np.zeros((len(emlines), len(logLam)))
    for i, eml in enumerate(emlines):
        print("Processing emission line {}".format(i+1))
        lname, lwave = eml
        emission[i] = np.exp(-(wave - lwave) ** 2 / (2 * sigma * sigma))
    hdu1 = fits.PrimaryHDU(ssps)
    hdu1.header["EXTNAME"] = "SSPS"
    hdu2 = fits.ImageHDU(emission)
    hdu2.header["EXTNAME"] = "EMISSION_LINES"
    params = Table(params, names=["alpha", "[Z/H]", "age", "[alpha/Fe]",
                                  "[Na/Fe]"])
    hdu3 = fits.BinTableHDU(params)
    hdu3.header["EXTNAME"] = "PARAMS"
    hdu1.header["CRVAL1"] = logLam[0]
    hdu1.header["CD1_1"] = logLam[1] - logLam[0]
    hdu1.header["CRPIX1"] = 1.
    # Making wavelength array
    hdu4 = fits.BinTableHDU(Table([logLam], names=["loglam"]))
    hdu4.header["EXTNAME"] = "LOGLAM"
    hdulist = fits.HDUList([hdu1, hdu2, hdu3, hdu4])
    hdulist.writeto(output, overwrite=True)
    return

def prepare_muse():
    w1 = 4500
    w2 = 10000
    velscale = 30  # km / s
    starttime = datetime.now()
    prepare_templates_emiles_muse(w1, w2, velscale, sample="bsf",
                                  redo=True)
    endtime = datetime.now()
    print("The program took {} to run".format(endtime - starttime))

def prepare_wifis():
    w1 = 11600
    w2 = 12400
    velscale = 20 # km / s
    starttime = datetime.now()
    prepare_templates_emiles_muse(w1, w2, velscale, sample="kinematics",
                                  redo=True, fwhm=2.5, instrument="wifis")
    endtime = datetime.now()
    print("The program took {} to run".format(endtime - starttime))


if __name__ == "__main__":
    prepare_muse()
    # prepare_wifis()


