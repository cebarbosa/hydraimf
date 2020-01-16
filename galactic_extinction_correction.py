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

def main(targetSN=80):
    wdir = os.path.join(context.get_data_dir("MUSE-DEEP"), "fieldA",
                        "spec1d_FWHM2.95_sn{}".format(targetSN))
    os.chdir(wdir)
    outdir = os.path.join(context.get_data_dir("MUSE-DEEP"), "fieldA",
                          "sci_sn{}".format(targetSN))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filenames = sorted([_ for _ in os.listdir(wdir) if _.endswith(".fits")])
    unit = np.power(10., -20) * u.erg / u.s / u.cm / u.cm / u.AA
    for fname in filenames:
        data = Table.read(fname)
        wave = data["wave"].data.byteswap().newbyteorder() * u.angstrom
        flam = data["flux"].data * unit
        flamerr = data["fluxerr"].data * unit
        ############################################################################
        # Make extinction correction
        CCM89 = extinction.ccm89(wave, context.Av, context.Rv)
        flam = extinction.remove(CCM89, flam)
        flamerr = extinction.remove(CCM89, flamerr)
        table = Table([wave, flam, flamerr], names=["wave", "flam", "flamerr"])
        table.write(os.path.join(outdir, fname), overwrite=True)


if __name__ == "__main__":
    main()