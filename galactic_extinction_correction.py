# -*- coding: utf-8 -*-
""" 

Created on 16/01/20

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os

import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import extinction

import context

def correct_spectra(input_dir, output_dir, filenames, unit=None):
    unit = 1. if unit is None else unit
    for fname in filenames:
        data = Table.read(os.path.join(input_dir, fname))
        wave = data["wave"].data.byteswap().newbyteorder() * u.angstrom
        flam = data["flux"].data * unit
        flamerr = data["fluxerr"].data * unit
        ############################################################################
        # Make extinction correction
        CCM89 = extinction.ccm89(wave, context.Av, context.Rv)
        flam = extinction.remove(CCM89, flam)
        flamerr = extinction.remove(CCM89, flamerr)
        table = Table([wave, flam, flamerr], names=["wave", "flam", "flamerr"])
        table.write(os.path.join(output_dir, fname), overwrite=True)

def run_ngc3311(targetSN=250):
    indir = os.path.join(context.get_data_dir("MUSE"), "fieldA",
                        "sn{}/spec1d_FWHM2.95".format(targetSN))
    outdir = os.path.join(context.get_data_dir("MUSE"), "fieldA",
                          "sn{}/sci".format(targetSN))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filenames = sorted([_ for _ in os.listdir(indir) if _.endswith(".fits")])
    unit = np.power(10., -20) * u.erg / u.s / u.cm / u.cm / u.AA
    correct_spectra(indir, outdir, filenames, unit=unit)

def run_m87(targetSN=500):
    imgname, cubename = context.get_img_cube_m87()
    wdir = os.path.split(imgname)[0]
    indir = os.path.join(wdir, "sn{}/spec1d_FWHM2.95".format(targetSN))
    outdir = os.path.join(wdir, "sn{}/sci".format(targetSN))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filenames = sorted([_ for _ in os.listdir(indir) if _.endswith(".fits")])
    unit = np.power(10., -20) * u.erg / u.s / u.cm / u.cm / u.AA
    correct_spectra(indir, outdir, filenames, unit=unit)

if __name__ == "__main__":
    # run_ngc3311()
    run_m87()