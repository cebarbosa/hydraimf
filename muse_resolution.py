# -*- coding: utf-8 -*-
"""

Created on 04/05/16

@author: Carlos Eduardo Barbosa

Calculates and plot the spectral resolution of MUSE.

"""
from __future__ import print_function
import os

import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

import context

def get_muse_fwhm():
    """ Returns the FWHM of the MUSE spectrograph as a function of the
    wavelength. """
    wave, R = np.loadtxt(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "tables/muse_wave_R.dat")).T
    wave = wave *u.nm
    fwhm = wave.to("angstrom") / R
    # First interpolation to obtain extrapolated values
    f1 = interp1d(wave.to("angstrom"), fwhm, kind="linear", bounds_error=False,
                 fill_value="extrapolate")
    # Second interpolation using spline
    wave = np.hstack((4000, wave.to("angstrom").value, 10000))
    f = interp1d(wave, f1(wave), kind="cubic", bounds_error=False)
    return f

def broad2res(w, flux, obsres, res=2.95, fluxerr=None):
    """ Broad resolution of observed spectra to a given resolution.

    Input Parameters
    ----------------
    w : np.array
        Wavelength array

    flux: np.array
        Spectrum to be broadened to the desired resolution.

    obsres : float or np.array
        Observed wavelength spectral resolution FWHM.

    res: float
        Resolution FWHM  of the spectra after the broadening.

    fluxerr: np.array
        Spectrum errors whose uncertainties are propagated

    Output parameters
    -----------------
    np.array:
        Broadened spectra.

    """
    dw = np.diff(w)[0]
    sigma_diff = np.sqrt(res**2 - obsres**2) / 2.3548 / dw
    diag = np.diag(flux)
    for j in range(len(w)):
        diag[j] = gaussian_filter1d(diag[j], sigma_diff[j], mode="constant",
                                 cval=0.0)
    newflux = diag.sum(axis=0)
    if fluxerr is None:
        return newflux
    errdiag = np.diag(fluxerr)
    for j in range(len(w)):
        errdiag[j] = gaussian_filter1d(errdiag[j]**2, sigma_diff[j],
                                       mode="constant", cval=0.0)
    newfluxerr = np.sqrt(errdiag.sum(axis=0))
    return newflux, newfluxerr

def plot_muse_fwhm():
    f = get_muse_fwhm()
    wave = np.linspace(4000, 10000, 1000)
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-paper")
    plt.figure(1)
    plt.minorticks_on()
    plt.plot(wave, f(wave), "-")
    plt.xlabel("$\lambda$ ($\AA$)")
    plt.ylabel(r"Spectral resolution $\alpha$ FWHM (Angstrom)")
    plt.show()

def plot_vel_resolution():
    f = get_muse_fwhm()
    wave = np.linspace(4000, 10000, 1000)
    from astropy.constants import c
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-paper")
    plt.figure(1)
    plt.minorticks_on()
    plt.plot(wave, c.to("km/s") * f(wave) / wave / 2.634, "-")
    plt.xlabel("$\lambda$ ($\AA$)")
    plt.ylabel(r"Velocity scale - sigma (km/s)")
    plt.show()

def broad_binned(FWHM, targetSN=250, dataset="MUSE", redo=False):
    """ Performs convolution to homogeneize the resolution. """
    wdir = os.path.join(context.data_dir, dataset, "voronoi/sn{}".format(
                     targetSN), "spec1d")
    outdir = "{}_FWHM{}".format(wdir, FWHM)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    specs = sorted([_ for _ in os.listdir(wdir) if _.endswith(
                    ".fits")])
    for i, filename in enumerate(specs):
        print("Convolving file {} ({} / {})".format(filename, i+1,
                                                    len(specs)))
        filepath = os.path.join(wdir, filename)
        output = os.path.join(outdir, filename)
        if os.path.exists(output) and not redo:
            continue
        data = Table.read(filepath, format="fits")
        wave = data["wave"]
        flux = data["flux"]
        fluxerr = data["fluxerr"]
        muse_fwhm = get_muse_fwhm()
        obsres = muse_fwhm(wave)
        newflux, newfluxerr = broad2res(wave, flux, obsres,
                                        FWHM, fluxerr=fluxerr)
        newtable = Table([wave, newflux, newfluxerr],
                         names=["wave", "flux", "fluxerr"])
        newtable.write(output, overwrite=True)

def homogeneize_resolution_m87(res=2.95, targetSN=500, dataset="MUSE"):
    """ Performs convolution to homogeneize the resolution. """
    imgname, cubename = context.get_img_cube_m87()
    input_dir = os.path.join(os.path.split(imgname)[0], "sn{}".format(targetSN),
                    "spec1d")
    output_dir = "{}_FWHM{}".format(input_dir, res)
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    specs = sorted([_ for _ in os.listdir(input_dir) if _.endswith(".fits")])
    for i, filename in enumerate(specs):
        print("Convolving file {} ({} / {})".format(filename, i+1,
                                                    len(specs)))
        filepath = os.path.join(input_dir, filename)
        output = os.path.join(output_dir, filename)
        data = Table.read(filepath, format="fits")
        wave = data["wave"]
        flux = data["flux"]
        fluxerr = data["fluxerr"]
        muse_fwhm = get_muse_fwhm()
        obsres = muse_fwhm(wave)
        newflux, newfluxerr = broad2res(wave, flux, obsres,
                                        res, fluxerr=fluxerr)
        newtable = Table([wave, newflux, newfluxerr],
                         names=["wave", "flux", "fluxerr"])
        newtable.write(output, overwrite=True)


if __name__ == "__main__":
    # plot_muse_fwhm()
    # plot_vel_resolution()
    broad_binned(2.95, targetSN=250, dataset="MUSE")
    # homogeneize_resolution_m87()