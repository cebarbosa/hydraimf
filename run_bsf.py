# -*- coding: utf-8 -*-
"""

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import platform

import numpy as np
from scipy.optimize import least_squares
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack
import matplotlib.pyplot as plt
from numpy.polynomial import Legendre
import pymc3 as pm
import theano.tensor as tt
from tqdm import tqdm

import context
import bsf

class SpecModel():
    def __init__(self, wave, velscale=None, test=False, nssps=1, porder=5):
        self.velscale = 100 * u.km / u.s if velscale is None else velscale
        self.wave = wave
        self.porder = porder
        # Templates have been already convolved to match the resolution of the
        # observations
        tempfile_extension = "bsf" if test is False else "test"
        templates_file = os.path.join(context.home, "templates",
                       "emiles_muse_vel{}_w4500_10000_{}_fwhm2.95.fits".format(
                        int(self.velscale.value), tempfile_extension))
        templates = fits.getdata(templates_file, ext=0)
        table = Table.read(templates_file, hdu=1)
        # for col in table.colnames:
        #     print(col, table[col].min(), table[col].max())
        logwave = Table.read(templates_file, hdu=2)["loglam"].data
        twave = np.exp(logwave) * u.angstrom
        self.spec = bsf.SEDModel(twave, table, templates, nssps=nssps,
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


def run_MAP(flam, flamerr, sed, output, redo=False):
    """ Routine to run BSF in a single spectrum"""
    if os.path.exists(output) and not redo:
        return
    # Estimating flux
    m0 = -2.5 * np.log10(np.median(flam) / np.median(sed.spec.templates))
    # Building model for the simplified fitting
    bounds = [[0., 0., 0, -0.96, 1., 3500., 100.],
              [1., 6, np.infty, 0.4, 14., 4100., 500.]]
    ptest = np.array([0.1, 4.1, np.power(10, -0.5 * m0), 0.2, 10.5, 3500.,
                      202.1])
    def resid(p):
        return (flam - sed(p)) / flamerr
    sol = least_squares(resid, ptest, bounds=bounds)
    tmap = Table(sol["x"], names=sed.parnames)
    tmap.write(output, format="fits", overwrite=True)
    return
    model0 = pm.Model()
    with model0:
        Av = pm.Exponential("Av", lam=1 / 0.2, testval=0.1)
        BNormal = pm.Bound(pm.Normal, lower=0)
        Rv = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
        mag = pm.Normal("mag", mu=m0, sd=3., testval=m0)
        flux = pm.Deterministic("flux",
                                pm.math.exp(-0.4 * mag * np.log(10)))
        theta = [Av, Rv, flux]
        # Setting limits given by stellar populations
        ########################################################################
        for param in sed0.ssp_parameters:
            vmin = sed0.spec.params[param].min()
            vmax = sed0.spec.params[param].max()
            vmean = 0.55 * (vmin + vmax)
            p0 = pm.Uniform(param, lower=vmin, upper=vmax, testval=vmean)
            theta.append(p0)
        V = pm.Normal("V", mu=3800., sd=100., testval=3800.)
        theta.append(V)
        BoundHalfNormal = pm.Bound(pm.HalfNormal, lower=25)
        sigma = BoundHalfNormal("sigma", sd=np.sqrt(2) * 200, testval=200)
        theta.append(sigma)
        nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
        theta.append(nu)
        theta = tt.as_tensor_variable(theta).T
        logl = bsf.LogLike(flam, wave, flamerr, sed0, loglike="studt")
        # use a DensityDist
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})
    # Performing MAP estimate
    with model0:
        sol = pm.find_MAP(progressbar=True)
        p0 = np.array([sol[_] for _ in sed0.parnames])
    for i, param in enumerate(sed0.parnames):
        print(param, p0[i])

    plt.plot(wave, sed0(p0))
    plt.show()
    return
    # Making fitting
    model = pm.Model()
    with model:
        Av = pm.Exponential("Av", lam=1 / 0.2, testval=0.1)
        BNormal = pm.Bound(pm.Normal, lower=0)
        Rv = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
        mag = pm.Normal("mag", mu=m0, sd=3., testval=m0)
        flux = pm.Deterministic("flux",
                                pm.math.exp(-0.4 * mag * np.log(10)))
        theta = [Av, Rv, flux]
        # Setting limits given by stellar populations
        ########################################################################
        for param in sed0.ssp_parameters:
            vmin = sed0.spec.params[param].min()
            vmax = sed0.spec.params[param].max()
            vmean = 0.5 * (vmin + vmax)
            p0 = pm.Uniform(param, lower=vmin, upper=vmax, testval=vmean)
            theta.append(p0)
        V = pm.Normal("V", mu=3800., sd=100., testval=3800.)
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
        logl = bsf.LogLike(flam, wave, flamerr, sed0, loglike="studt")
        # use a DensityDist
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})
        trace = pm.sample()
        df = pm.stats.summary(trace, alpha=0.3173)
        df.to_csv(summary)
        pm.save_trace(trace, outdb, overwrite=True)
    return

def setting_polynomials(targetSN=250, velscale=None, dataset=None):
    velscale = 200 * u.km / u.s if velscale is None else velscale
    dataset = "MUSE" if dataset is None else dataset
    wdir = os.path.join(context.get_data_dir(dataset), "fieldA",
                        "sn{}/sci".format(targetSN))
    filenames = sorted([_ for _ in os.listdir(wdir) if _.endswith(".fits")])
    data = Table.read(os.path.join(wdir, filenames[0]))
    wave = data["wave"]
    # Loading models
    map0dir = os.path.join(wdir, "MAP0")
    if not os.path.exists(map0dir):
        os.mkdir(map0dir)
    sed0 = SpecModel(wave, test=True, porder=0, velscale=velscale)
    resid = np.zeros((len(filenames), len(wave)))
    for i, fname in enumerate(tqdm(filenames)):
        spec = os.path.join(wdir, fname)
        # Reading input data
        data = Table.read(spec)
        flam = data["flam"].data
        flamerr = data["flamerr"].data
        idx = np.where(np.isfinite(flam * flamerr))
        map0_file = os.path.join(map0dir, fname.replace(".", "_MAP0."))
        run_MAP(flam[idx], flamerr[idx], sed0, map0_file)
        map0table = Table.read(map0_file)
        map0 = np.array([map0table[_].data[0] for _ in sed0.parnames])
        resid[i] = flam - sed0(map0)
        # plt.plot(wave, resid[i], "-", c="C0")
    y = resid[0]
    z = np.polyfit(wave.data, y, 75)
    p = np.poly1d(z)
    plt.plot(wave, y, "-", c = "C1")
    plt.plot(wave, p(wave.data), c="C2")
    plt.show()


if __name__ == "__main__":
    setting_polynomials()
