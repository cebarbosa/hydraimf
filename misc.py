# -*- coding: utf-8 -*-
""" 

Created on 29/11/17

Author : Carlos Eduardo Barbosa

Miscelaneous routines

"""
import numpy as np
from astropy.io import fits

def array_from_header(filename, axis=3, extension=1):
    """ Produces array for wavelenght of a given array. """
    h = fits.getheader(filename, ext=extension)
    w0 = h["CRVAL{0}".format(axis)]
    dwkeys = ["CD{0}_{0}".format(axis), "CDELT{0}".format(axis)]
    for key in dwkeys:
        if key in h:
            dwkey = key
            break
    deltaw = h[dwkey]
    pix0 = h["CRPIX{0}".format(axis)]
    npix = h["NAXIS{0}".format(axis)]
    return w0 + deltaw * (np.arange(npix) + 1 - pix0)

def snr(flux, axis=0):
    """ Calculates the S/N ratio of a spectra.

    Translated from the IDL routine der_snr.pro """
    signal = np.nanmedian(flux, axis=axis)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.*flux - \
           np.roll(flux, 2, axis=axis) - np.roll(flux, -2, axis=axis)), \
           axis=axis)
    return signal, noise, signal / noise
