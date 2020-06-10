# -*- coding: utf-8 -*-
""" 

Created on 16/01/20

Author : Carlos Eduardo Barbosa

Test BSF in spectra provided by Chiara

"""
from __future__ import print_function, division

import os
import platform

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from numpy.polynomial import Legendre
import pymc3 as pm
import theano.tensor as tt
import context
import bsf_old

class SpecModel():
    def __init__(self, wave, velscale=None, test=False, nssps=1, porder=5):
        self.velscale = 50 * u.km / u.s if velscale is None else velscale
        self.wave = wave
        self.porder = porder
        # Templates have been already convolved to match the resolution of the
        # observations
        tempfile_extension = "bsf" if test is False else "test"
        templates_file = os.path.join(context.home, "templates",
                                 "emiles_muse_vel{}_w4500_10000_{"
                                 "}_fwhm2.5.fits".format(
                                  int(self.velscale.value), tempfile_extension))
        templates = fits.getdata(templates_file, ext=0)
        table = Table.read(templates_file, hdu=1)
        logwave = Table.read(templates_file, hdu=2)["loglam"].observed
        twave = np.exp(logwave) * u.angstrom
        self.spec = bsf_old.SEDModel(twave, table, templates, nssps=nssps,
                                     wave_out=self.wave, velscale=self.velscale)
        self.parnames = [_.split("_")[0] for _ in self.spec.parnames]
        # Making polynomial to slightly change continuum
        N = len(wave)
        self.x = np.linspace(-1, 1, N)
        self.poly = np.ones((porder, N))
        for i in range(porder):
            self.poly[i] = Legendre.basis(i+1)(self.x)
            self.parnames.append("a{}".format(i+1))
        self.grad_dim = (len(self.parnames), len(self.wave))
        self.nparams = self.spec.nparams + porder
        self.ssp_parameters = table.colnames

    def __call__(self, theta):
        p0, p1 = np.split(theta, [len(theta)- self.porder])
        return self.spec(p0) * (1. + np.dot(p1, self.poly))

    def gradient(self, theta):
        grad = np.zeros(self.grad_dim)
        p0, p1 = np.split(theta, [len(theta)- self.porder])
        spec = self.spec(p0)
        specgrad = self.spec.gradient(p0)
        poly = (1. + np.dot(p1, self.poly))
        grad[:len(theta)- self.porder] = specgrad * poly
        grad[len(theta)- self.porder:] = spec * self.poly
        return grad


def run_bsf(fname, test=False, redo=False, outdir=None):
    """ Routine to run BSF in a single spectrum"""
    name = fname.split(".")[0]
    outdir = os.path.join(os.getcwd(), "bsf") if outdir is None else outdir
    outdb = os.path.join(outdir, name)
    if os.path.exists(outdb) and not redo:
        return
    summary = "{}.csv".format(outdb)
    ############################################################################
    # Reading input data
    flux = fits.getdata(fname)
    fluxerr = np.ones_like(flux) * 0.01 * flux.mean()
    header = fits.getheader(fname)
    wcs = WCS(header)
    pix = np.arange(len(flux))+1
    wave = wcs.wcs_pix2world(pix, 1)[0]
    idx = np.where((wave > 4700) & (wave < 9200))[0]
    wave = wave[idx]
    flux = flux[idx]
    fluxerr = fluxerr[idx]
    ############################################################################
    # Building parametric model for fitting
    porder = 5
    sed = SpecModel(wave, test=test, porder=porder)
    # Estimating flux
    m0 = -2.5 * np.log10(np.median(flux) / np.median(sed.spec.templates))
    f0 = np.power(10, -0.4 * m0)
    # Making fitting
    model = pm.Model()
    with model:
        Av = pm.Uniform("Av", lower=0., upper=0.001)
        Rv = pm.Uniform("Rv", lower=3.095, upper=3.105)
        mag = pm.Normal("mag", mu=m0, sd=3., testval=m0)
        f0 = pm.Deterministic("flux",
                                pm.math.exp(-0.4 * mag * np.log(10)))
        theta = [Av, Rv, f0]
        # Setting limits given by stellar populations
        ########################################################################
        for param in sed.ssp_parameters:
            vmin = sed.spec.params[param].min()
            vmax = sed.spec.params[param].max()
            vmean = 0.5 * (vmin + vmax)
            p = pm.Uniform(param, lower=vmin, upper=vmax, testval=vmean)
            theta.append(p)
        V = pm.Normal("V", mu=0., sd=100., testval=3800.)
        theta.append(V)
        BoundHalfNormal = pm.Bound(pm.HalfNormal, lower=25)
        sigma = BoundHalfNormal("sigma", sd=np.sqrt(2) * 200)
        theta.append(sigma)
        for i in range(porder):
            a = pm.Normal("a{}".format(i+1), mu=0, sd=0.01)
            theta.append(a)
        nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
        theta.append(nu)
        theta = tt.as_tensor_variable(theta).T
        logl = bsf_old.LogLike(flux, wave, fluxerr, sed, loglike="studt")
        # use a DensityDist
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})
        trace = pm.sample()
        df = pm.stats.summary(trace, alpha=0.3173)
        df.to_csv(summary)
        pm.save_trace(trace, outdb, overwrite=True)
    return

def main(test=False):
    wdir = os.path.join(context.data_dir, "test")
    os.chdir(wdir)
    outdir = os.path.join(wdir, "bsf")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filenames = sorted([_ for _ in os.listdir(wdir) if _.endswith(".fits")])
    if platform.node() == "kadu-Inspiron-5557":
        for fname in filenames:
            run_bsf(fname, outdir=outdir, test=test)

if __name__ == "__main__":
    main()
