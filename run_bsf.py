# -*- coding: utf-8 -*-
"""

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
import matplotlib.pyplot as plt
from matplotlib import cm
import pymc3 as pm
import ppxf_util as util
import theano.tensor as tt
from spectres import spectres
import scipy.optimize as opt
import emcee
from tqdm import tqdm

import context
import bsf

def build_sed_model(wave, w1=4500, w2=9400, velscale=200, sample=None,
                    fwhm=2.95, em_oversample=4, porder=30):
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
    p0_ssp = np.zeros(len(params.colnames))
    for i, param in enumerate(params.colnames):
        vmin, vmax = params[param].min(), params[param].max()
        limits[param] = (vmin, vmax)
        p0_ssp[i] = vmin + (vmax - vmin) * np.random.rand(1)
    logwave = Table.read(templates_file, hdu=2)["loglam"].data
    twave = np.exp(logwave)
    ssp = bsf.StPopInterp(twave, params, templates)
    # Adding extinction to the stellar populations
    extinction = bsf.CCM89(twave)
    limits["Av"] = (0., 5)
    limits["Rv"] = (0., 5)
    stars = bsf.Rebin(wave, bsf.LOSVDConv(ssp * extinction, velscale=velscale))
    limits["V"] = (3600, 4100)
    limits["sigma"] = (50, 500)
    p0_ext = np.array([0.1, 3.8])
    p0_stars = np.hstack([p0_ssp, p0_ext, [3810, 189]])
    # Loading templates for the emission lines
    velscale_gas = velscale / em_oversample
    logemwave = util.log_rebin([w1, w2], wave,
                       velscale=velscale_gas, oversample=5)[1]
    emwave = np.exp(logemwave)
    emission_lines, line_names, line_wave = util.emission_lines(logemwave,
            [emwave[0], emwave[-1]], fwhm)
    line_names = [_.replace("[", "").replace("]", "").replace("_", "") for _ in
                  line_names]
    emission_lines = emission_lines.T
    emnorm = emission_lines.max(axis=1)
    emission_lines /= emnorm[:, None]
    emission = bsf.Rebin(wave,
                         bsf.LOSVDConv(bsf.EmissionLines(emwave, emission_lines,
                         line_names), velscale=velscale_gas))
    emission.parnames[-1] = "sigma_gas"
    emission.parnames[-2] = "V_gas"
    p0_em = np.ones(len(line_names), dtype=np.float)
    p0_em_losvd = np.array([3815, 60])
    for lname in line_names:
        limits[lname] = (0, 10.)
    limits["V_gas"] = (3600, 4100)
    limits["sigma_gas"] = (50, 100)
    # Adding a polynomial
    poly = bsf.Polynomial(wave, porder)
    p0_poly = np.zeros(porder + 1, dtype=np.float)
    p0_poly[0] = 1.
    limits["p0"] = (0, 10)
    for i in range(porder):
        limits["p{}".format(i+1)] = (-1, 1)
    ############################################################################
    # Creating a model including LOSVD
    sed = (stars + emission) * poly
    # Setting properties that may be useful later in modeling
    sed.limits = limits
    sed.bounds = np.array([limits[par] for par in sed.parnames])
    sed.x0 = np.hstack([p0_stars, p0_em, p0_em_losvd, p0_poly])
    sed.line_names = line_names
    sed.porder = porder
    sed.ssppars = params.colnames
    return sed

def build_pymc3_model(flux, sed, loglike=None, fluxerr=None):
    loglike = "normal" if loglike is None else loglike
    model = pm.Model()
    flux = flux.astype(np.float)
    fluxerr = np.ones_like(flux) if fluxerr is None else fluxerr
    ssp_testvals = {"imf": 1.85, "Z": 0.07, "T": 10.25, "alphaFe":0.15, "NaFe":
        0.22} # Reasonable values avoiding nodes
    with model:
        theta = []
        # Stellar population parameters
        for param in ["imf", "Z", "T", "alphaFe", "NaFe"]:
            vmin, vmax = sed.limits[param]
            v = pm.Uniform(param, lower=vmin, upper=vmax,
                           testval=ssp_testvals[param])
            theta.append(v)
        # Dust attenuation parameters
        Av = pm.Exponential("Av", lam=1 / 0.4, testval=0.1)
        theta.append(Av)
        BNormal = pm.Bound(pm.Normal, lower=0)
        Rv = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
        theta.append(Rv)
        # Stellar kinematics
        BoundedNormal = pm.Bound(pm.Normal, lower=3500, upper=4200)
        V = BoundedNormal("V", mu=3800., sigma=100., testval=3810.)
        theta.append(V)
        BoundedHalfNormal = pm.Bound(pm.HalfNormal, lower=100, upper=500)
        sigma = BoundedHalfNormal("sigma", sd=100., testval=190.)
        theta.append(sigma)
        # Emission lines
        for em in sed.em_names:
            v = pm.HalfNormal(em, sigma=1, testval=0.5)
            theta.append(v)
        Vgas = BoundedNormal("V_gas", mu=3800., sigma=100.)
        theta.append(Vgas)
        sigma_gas = pm.Uniform("sigma_gas", lower=50, upper=100)
        theta.append(sigma_gas)
        # Polynomial continuum
        p0 = pm.Normal("p0", mu=1., sd=0.5, testval=1.)
        theta.append(p0)
        for n in range(sed.porder):
            pn = pm.Normal("p{}".format(n+1), mu=0., sd=.05, testval=0.)
            theta.append(pn)
        if loglike == "studt":
            nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(nu)
        if loglike == "normal2":
            x = pm.Normal("x", mu=0, sd=1)
            s = pm.Deterministic("S", 1. + pm.math.exp(x))
            theta.append(s)
        theta = tt.as_tensor_variable(theta).T
        logl = bsf.TheanoLogLikeInterface(flux, sed, loglike=loglike,
                                          obserr=fluxerr)
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})
    return model

def run_model_nuts(model, db, draws=1000, redo=False):
    summary = "{}.csv".format(db)
    if os.path.exists(summary) and not redo:
        with model:
            trace = pm.load_trace(db)
        return trace
    with model:
        trace = pm.sample(draws, tune=draws, step=pm.Metropolis())
        df = pm.stats.summary(trace)
        df.to_csv(summary)
    pm.save_trace(trace, db, overwrite=True)
    return trace

class Uniform():
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def __call__(self, x):
        if self.xmin < x < self.xmax:
            return 0.
        return -np.inf

class Normal():
    def __init__(self, x0, sigma, xmin=-np.inf, xmax=np.inf):
        self.x0 = x0
        self.sigma = sigma
        self.xmin = xmin
        self.xmax = xmax

    def __call__(self, x):
        if self.xmin < x < self.xmax:
            return - 0.5 * np.power((x - self.x0) / self.sigma, 2.)
        return -np.inf

class Loglike():
    def __init__(self, observed, sed, obserr=None, ltype=None):
        self.observed = observed
        self.sed = sed
        self.obserr = np.ones_like(self.observed) if obserr is None else obserr

        self.ltype = "normal2" if ltype is None else ltype
        ltypes = {"normal": bsf.NormalLogLike, "normal2": bsf.Normal2LogLike,
                  "studt": bsf.StudTLogLike}
        self.parnames = self.sed.parnames
        if self.ltype == "normal2":
            self.parnames.append("S")
        elif self.ltype == "studt":
            self.parnames.append("nu")
        self.loglike = ltypes[self.ltype](self.observed, self.sed,
                                     obserr=self.obserr)
        polypars = ["p{}".format(i+1) for i in range(sed.porder)]
        self.priors = {}
        for param in sed.parnames:
            isssp = [param.startswith(p) for p in sed.ssppars]
            if any(isssp):
                idx = int(np.argwhere(isssp))
                vmin, vmax = sed.limits[sed.ssppars[idx]]
                self.priors[param] = Uniform(vmin, vmax)
            elif param == "Av":
                self.priors[param] = Normal(0, 1, xmin=0)
            elif param == "Rv":
                self.priors[param] = Normal(3.1, 1., xmin=0)
            elif param in ["V", "V_gas"]:
                self.priors[param] = Normal(3800, 100, xmin=3500., xmax=4100)
            elif param == "sigma":
                self.priors[param] = Normal(100, 100, xmin=100, xmax=500)
            elif param == "sigma_gas":
                self.priors[param] = Normal(40, 20, xmin=40, xmax=100)
            elif param in sed.line_names:
                self.priors[param] = Uniform(0, 10.)
            elif param == "V_gas":
                self.priors[param] = Normal(3800, 100, xmin=3500., xmax=4100)
            elif param == "p0":
                self.priors[param] = Normal(1, 1, xmin=0., xmax=100)
            elif param in polypars:
                self.priors[param] = Normal(0, 0.1)
            elif param == "S":
                norm = Normal(0, 1)
                self.priors[param] = lambda x: 1 + np.exp(norm(x))
            elif param == "nu":
                self.priors[param] = Uniform(2.01, 50)
            else:
                print("Missing parameter in priors: {}".format(param))

    def log_prior(self, theta):
        like = 0.
        for param, val in zip(self.parnames, theta):
            like += self.priors[param](val)
        return like

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike(theta)

def run_MAP(flam, flamerr, sed, outprefix, redo=False):
    """ Use duall_annelling to get MAP estimate. """
    output = "{}_params.fits".format(outprefix)
    if os.path.exists(output) and not redo:
        t = Table.read(output)
        x = np.array([t[param].data for param in sed.parnames]).ravel()
        return x
    name = os.path.split(outprefix)[1]
    tid = Table([[name]], names=["spec"])
    loglike = bsf.NormalLogLike(flam, sed, obserr=flamerr)
    bounds = [sed.limits[p] for p in sed.parnames]
    time1 = datetime.now()
    print("Starting MAP for {} in {}".format(name, time1.isoformat()))
    sol = opt.minimize(lambda p: -loglike(p), x0=sed.x0, bounds=bounds)
    time2 = datetime.now()
    print("Ellapsed time for fitting: {}.".format(time2 - time1))
    tpars = Table(sol.x, names=sed.parnames)
    tpars = hstack([tid, tpars])
    tpars.write(output, overwrite=True)
    return sol.x

def run_emcee(flam, flamerr, sed, p0, dbname, draws=1000, redo=False):
    if os.path.exists(dbname) and not redo:
        return
    backend = emcee.backends.HDFBackend(dbname)
    loglike = Loglike(flam, sed, obserr=flamerr)
    pos = p0 + 1e-4 * np.random.randn(2 * len(p0), len(p0))
    nwalkers, ndim = pos.shape
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike,
                                    backend=backend)
    max_n = 10000
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    # This will be useful to testing convergence
    old_tau = np.inf
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    sampler.run_mcmc(pos, draws, progress=True)
    return

def plot_fitting(wave, flux, fluxerr, sed, trace, outfig, redo=True):
    if os.path.exists("{}.png".format(outfig)) and not redo:
        return
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    traces = np.stack(trace[v] for v in sed.parnames).T
    spec = np.zeros((len(traces), len(wave)))
    loglike = bsf.NormalLogLike(flux, sed, obserr=fluxerr)
    llike = np.zeros(len(traces))
    for i in tqdm(range(len(traces)), desc="Loading indices for plots and "
                                           "table..."):
        spec[i] = sed(traces[i])
        llike[i] = loglike(traces[i])
    plt.plot(llike)
    plt.show()


    x = np.median(spec, axis=0)
    fig = plt.figure(figsize=(context.fig_width, 3.5))
    ax = plt.subplot(211)
    ax.errorbar(wave, flux, yerr=fluxerr, fmt="-", mec="w", mew=0.4,
                elinewidth=0.8)
    for c, per in zip(colors, percs):
        ax.fill_between(wave, np.percentile(spec, per, axis=0),
                         np.percentile(spec, per + 10, axis=0), color=c)
    ax.set_xlabel(r"$\lambda$ (\r{A})")
    ax = plt.subplot(212)
    for c, per in zip(colors, percs):
        ax.fill_between(wave, flux - np.percentile(spec, per, axis=0),
                         flux - np.percentile(spec, per + 10, axis=0), \
                              color=c)
    ax.axhline(y=0, ls="--", c="k", lw=0.8)
    plt.subplots_adjust(left=0.12, right=0.98, wspace=0.02, top=0.953,
                        bottom=0.095)
    plt.show()
    return

def run_ngc3311(targetSN=250, velscale=200, doMCMC=False, doEMCEE=True,
                ltype=None, sample=None, draws=1000):
    """ Run BSF full spectrum fitting. """
    ltype = "normal2" if ltype is None else ltype
    sample = "all" if sample is None else sample
    imgname, cubename = context.get_field_files("fieldA")
    wdir = os.path.join(os.path.split(cubename)[0], "sn{}/sci".format(targetSN))
    outdir = os.path.join(wdir, "BSF")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    specnames = [_ for _ in sorted(os.listdir(wdir)) if _.endswith(".fits")]
    # Read first spectrum to set the dispersion
    data = Table.read(os.path.join(wdir, specnames[0]))
    wave_lin = data["wave"].data
    flam = data["flam"].data
    _, logwave, velscale = util.log_rebin([wave_lin[0], wave_lin[-1]], flam,
                                    velscale=velscale)
    wave = np.exp(logwave)[1:-1]
    sed = build_sed_model(wave, sample=sample)
    methods = ["MCMC", "MAP", "EMCEE"]
    mdirs = {}
    for method in methods:
        mdir = os.path.join(outdir, method)
        if not os.path.exists(mdir):
            os.mkdir(mdir)
        mdirs[method] = mdir
    for specname in specnames:
        name = specname.split(".")[0]
        data = Table.read(os.path.join(wdir, specname))
        wlin = data["wave"].data
        flam = data["flam"].data
        flamerr = data["flamerr"].data
        flam, flamerr = spectres(wave, wlin, flam, spec_errs=flamerr)
        norm = np.median(flam)
        flam /= norm
        flamerr /= norm
        # Run MAP for initialization of both methods
        outprefix = os.path.join(mdirs["MAP"], name)
        p0 = run_MAP(flam, flamerr, sed, outprefix)
        if ltype == "normal2":
            p0 = np.hstack([p0, np.array([0])])
        if doMCMC:
            dbname = os.path.join(mdirs["MCMC"], name)
            model = build_pymc3_model(flam, sed, fluxerr=flamerr)
            trace = run_model_nuts(model, dbname)
            plot_fitting(wave, flam, flamerr, sed, trace, dbname)
        if doEMCEE:
            dbname = os.path.join(mdirs["EMCEE"], "{}.h5".format(name))
            run_emcee(flam, flamerr, sed, p0, dbname, draws=draws, redo=False)
            reader = emcee.backends.HDFBackend(dbname)
            samples = reader.get_chain(flat=True)
            pfit = np.median(samples, axis=0)
            plt.plot(wave, flam)
            plt.plot(wave, sed(pfit[:-1]))
            plt.show()
            input()


        input(404)



if __name__ == "__main__":
    run_ngc3311(sample="test", draws=100)
