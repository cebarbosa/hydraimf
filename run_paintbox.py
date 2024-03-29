# -*- coding: utf-8 -*-
"""

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import sys
import copy
import re
import getpass

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from matplotlib import cm
import ppxf.ppxf_util as util
from spectres import spectres
import pymc3 as pm
import theano.tensor as tt
from tqdm import tqdm
import seaborn as sns
import emcee
from scipy import stats

import context
import paintbox as pb
from paintbox.interfaces import TheanoLogLikeInterface

def build_sed_model(wave, w1=4500, w2=9400, velscale=200, sample=None,
                    fwhm=2.95, em_oversample=8, porder=30, nssps=1,
                    use_emission=True):
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
    ssp = pb.ParametricModel(twave, params, templates)
    # Using interpolation routine to get normalization
    norm = pb.ParametricModel(np.ones(1), params, tnorm) if nssps > 1 else None
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
    stars = pb.Resample(wave, pb.LOSVDConv(pop * extinction, velscale=velscale))
    if use_emission:
    # Loading templates for the emission lines
        velscale_gas = velscale / em_oversample
        logemwave = util.log_rebin([w1, w2], twave, velscale=velscale_gas)[1]
        emwave = np.exp(logemwave)
        emission_lines, line_names, line_wave = util.emission_lines(logemwave,
                [emwave[0], emwave[-1]], fwhm)
        ############################################################################
        # Adding [NI]
        line_wave = [5200.257]
        gauss = util.gaussian(logemwave, line_wave, fwhm, True)
        emission_lines = np.column_stack([emission_lines, gauss])
        line_names = np.append(line_names, 'NI5200')
        line_wave = np.append(line_wave, line_wave[0])
        ############################################################################
        line_names = [_.replace("[", "").replace("]", "").replace("_", "") for _ in
                      line_names]
        emission_lines = emission_lines.T
        emnorm = emission_lines.max(axis=1)
        emission_lines /= emnorm[:, None]
        emission = pb.Resample(wave, pb.LOSVDConv(pb.NonParametricModel(emwave,
                      emission_lines, line_names), velscale=velscale_gas))
        emission.parnames[-1] = "sigma_gas"
        emission.parnames[-2] = "V_gas"
    else:
        line_names = []
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    # Using skycalc model to model residuals
    sky_templates_file = os.path.join(context.data_dir,
                                      "sky/sky_templates.fits")
    wsky = Table.read(sky_templates_file, hdu=1)["wave"].data
    fsky = fits.getdata(sky_templates_file, hdu=1)
    snames = Table.read(sky_templates_file, hdu=2)["skylines"].data
    snames = ["sky" + re.sub(r"[\(\)$\-\_]", "", _.decode()) for _ in snames]
    if not use_emission:
        fsky = fsky[:-2]
        snames = snames[:-2]
    sky = pb.Resample(wave, pb.NonParametricModel(wsky, fsky, snames))
    # Creating a model including LOSVD
    sed = (stars * poly) + sky
    if use_emission:
        sed += emission
    # Setting properties that may be useful later in modeling
    sed.ssppars = limits
    sed.sspcolnames = params.colnames
    sed.sspparams = params
    sed.line_names = line_names
    sed.porder = porder
    sed.nssps = nssps
    return sed, norm, sky

def make_priors(sed):
    priors = {}
    polynames = ["p{}".format(i) for i in range(1,100)]
    stop=False
    nssps = np.maximum(1,
                       np.sum([1 for p in sed.parnames if p.startswith("w_")]))
    for param in sed.parnames:
        if len(param.split("_")) == 2:
            pname, npop = param.split("_")
        else:
            pname = param
        if pname in sed.ssppars:
            vmin, vmax = sed.ssppars[pname]
            priors[param] = stats.uniform(loc=vmin, scale=vmax - vmin)
        elif pname == "w":
            priors[param] = stats.halfnorm(
                            scale=1 / nssps * np.sqrt(0.5 * np.pi))
        elif param == "Av":
            priors[param] = stats.expon(scale=0.2)
        elif param == "Rv":
            mu, sd = 3.1, 0.8
            a, b = (0 - mu) / sd, (np.infty - mu) / sd
            priors[param] = stats.truncnorm(a, b, mu, sd)
        elif param in ["V", "V_gas"]:
            priors[param] = stats.norm(loc=3800, scale=50)
        elif param == "sigma":
            priors[param] = stats.uniform(loc=100, scale=400)
        elif param == "sigma_gas":
            priors[param] = stats.uniform(loc=60, scale=60)
        elif param == "p0":
            mu, sd = 1, 0.3
            a, b = (0 - mu) / sd, (np.infty - mu) / sd
            priors[param] = stats.truncnorm(a, b, mu, sd)
        elif param in polynames:
            priors[param] = stats.norm(0, 0.02)
        elif param.startswith("sky"):
            priors[param] = stats.norm(0, 0.1)
        elif param in sed.line_names:
            priors[param] = stats.halfnorm(scale=1)
        else:
            print("missing prior: {}".format(param))
            stop = True
    if stop:
        input()
    priors["eta"] = stats.uniform(loc=1, scale=10)
    priors["nu"] = stats.uniform(loc=2.01, scale=8)
    return priors

# Deterministic function for stick breaking
def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

def make_pymc3_model(sed, loglike):
    model = pm.Model()
    polynames = ["p{}".format(i + 1) for i in range(100)]
    with model:
        # alpha = pm.Gamma('alpha', 1., 1.)
        # beta = pm.Beta('beta', 1, alpha, shape=sed.nssps)
        # w = pm.Deterministic('w', stick_breaking(beta))
        theta = []
        for param in loglike.parnames:
            v = False
            if len(param.split("_")) == 2:
                pname, n = param.split("_")
            else:
                pname = param
                n = 1
            # Weight of the ssp
            # if pname == "w":
            #     theta.append(w[int(n)-1])
            # Stellar population parameters
            if pname in sed.ssppars:
                vmin, vmax = sed.ssppars[pname]
                vinit = float(0.5 * (vmin + vmax))
                v = pm.Uniform(param, lower=float(vmin), upper=float(vmax),
                               testval=vinit)
            # Dust attenuation parameters
            elif param == "Av":
                v = pm.Exponential("Av", lam=1 / 0.4, testval=0.1)
            elif param == "Rv":
                BNormal = pm.Bound(pm.Normal, lower=0)
                v = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
            elif param == "V":
                # Stellar kinematics
                v = pm.Normal("V", mu=3800., sd=50., testval=3805.)
            elif param == "sigma":
                v = pm.Uniform(param, lower=100, upper=500, testval=185.)
            # Emission lines
            elif param in sed.line_names:
                v = pm.HalfNormal(param, 1., testval=1.)
            elif param == "V_gas":
                v = pm.Uniform("V_gas", lower=3600., upper=4100.,
                                   testval=3805.)
            elif param == "sigma_gas":
                v = pm.Uniform("sigma_gas", lower=60., upper=120,
                                       testval=85.)
            # Polynomial parameters
            elif pname == "p0":
                v = pm.Normal("p0", mu=1, sd=0.1, testval=1.)
            elif param in polynames:
                v = pm.Normal(param, mu=0, sd=0.01, testval=0.)
            elif param.startswith("sky"):
                v = pm.Normal(param, mu=0, sd=1, testval=-0.1)
            elif param == "eta":
                v = pm.Uniform("eta", lower=1, upper=10)
            elif param == "nu":
                v = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(v)
            if v is False:
                print("Missing parameter prior for pymc3: {}".format(param))
                input()
        theta = tt.as_tensor_variable(theta).T
        logl = TheanoLogLikeInterface(loglike)
        pm.DensityDist('loglike', lambda v: logl(v), observed={'v': theta})
    return model

def run_mcmc(model, db, redo=False, method=None):
    summary = "{}.csv".format(db)
    method = "MCMC" if method is None else method
    if os.path.exists(summary) and not redo:
        return
    with model:
        if method == "MCMC":
            trace = pm.sample(draws=300, tune=300, step=pm.Metropolis(),
                              cores=1)
        elif method == "NUTS":
            trace = pm.sample()
        elif method == "SMC":
            trace = pm.sample_smc(draws=250, threshold=0.3,
                                      progressbar=True )
        pm.save_trace(trace, db, overwrite=True)
    return trace

# def run_emcee(loglike, sed, db, priors):
#     mcmc_db = db.replace("EMCEE", "MCMC").replace(".h5", "")
#     trace = load_traces(mcmc_db, loglike.parnames)
#     ndim = len(loglike.parnames)
#     nwalkers = 2 * ndim
#     pos = np.zeros((nwalkers, ndim))
#     priors = []
#     for i, param in enumerate(loglike.parnames):
#         if len(param.split("_")) == 2:
#             pname, n = param.split("_")
#         else:
#             pname = param
#         ########################################################################
#         # Setting first guess and limits of models
#         ########################################################################
#         # Stellar population parameters
#         if pname in sed.ssppars:
#             vmin, vmax = sed.ssppars[pname]
#         else:
#             vmin = np.percentile(trace[:,i], 1)
#             vmax = np.percentile(trace[:,i], 99)
#         prior = stats.uniform(vmin, vmax - vmin)
#         priors.append(prior.logpdf)
#         pos[:, i] = prior.rvs(nwalkers)
#     def log_probability(theta):
#         lp = np.sum([prior(val) for prior, val in zip(priors, theta)])
#         if not np.isfinite(lp) or np.isnan(lp):
#             return -np.inf
#         return loglike(theta)
#     backend = emcee.backends.HDFBackend(db)
#     backend.reset(nwalkers, ndim)
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
#                                     backend=backend)
#     sampler.run_mcmc(pos, 1000, progress=True)
#     return

def run_sampler(loglike, priors, outdb, nsteps=1000):
    ndim = len(loglike.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(loglike.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    def log_probability(theta):
        lp = np.sum([prior(val) for prior, val in zip(logpdf, theta)])
        if not np.isfinite(lp) or np.isnan(lp):
            return -np.inf
        return lp + loglike(theta)
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    backend=backend)
    sampler.run_mcmc(pos, nsteps, progress=True)
    return

def weighted_traces(trace, sed, weights, outtab, redo=False):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    if os.path.exists(outtab) and not redo:
        a = Table.read(outtab)
        return a
    ws, ps = [], []
    for i in range(sed.nssps):
        w = trace[:, sed.parnames.index("w_{}".format(i+1))]
        j = [sed.parnames.index("{}_{}".format(p, i+1)) for p in
             sed.sspcolnames]
        ssp = trace[:, j]
        n = np.array([weights(s)[0] for s in ssp])
        ws.append(w * n)
        ps.append(ssp)
    ws = np.stack(ws)
    ws /= ws.sum(axis=0)
    ps = np.stack(ps)
    a = np.zeros((len(sed.sspcolnames), len(ws[1])))
    for i, param in enumerate(sed.sspcolnames):
        a[i] = np.sum(ps[:,:,i] * ws, axis=0)
    a = Table(a.T, names=sed.sspcolnames)
    a.write(outtab, overwrite=True)
    return a

def make_table(trace, outtab):
    data = np.array([trace[p].data for p in trace.colnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = []
    for i, param in enumerate(trace.colnames):
        t = Table()
        t["param"] = [param]
        t["median"] = [round(v[i], 5)]
        t["lerr".format(param)] = [round(vlerr[i], 5)]
        t["uerr".format(param)] = [round(vuerr[i], 5)]
        tab.append(t)
    tab = vstack(tab)
    tab.write(outtab, overwrite=True)
    return tab

def plot_fitting(wave, flux, fluxerr, sed, traces, db, redo=True, sky=None,
                 norm=1, unit_norm=19, lw=1, skylines=[]):
    outfig = "{}_fitting".format(db.replace(".h5", ""))
    specnum = os.path.split(db)[1].replace(".h5", "").split("_")[-1]
    if os.path.exists("{}.png".format(outfig)) and not redo:
        return
    percs = np.array([2.2, 15.8, 84.1, 97.8])
    fracs = np.array([0.3, 0.6, 0.3])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    models = np.zeros((len(traces), len(wave)))
    loglike = pb.NormalLogLike(flux, sed, obserr=fluxerr)
    llike = np.zeros(len(traces))
    for i in tqdm(range(len(traces)), desc="Loading spectra for plots and "
                                           "table..."):
        models[i] = sed(traces[i])
        llike[i] = loglike(traces[i])
    outmodels = db.replace(".h5", "_seds.fits")
    hdu0 = fits.PrimaryHDU(models)
    hdu1 = fits.ImageHDU(wave)
    hdus = [hdu0, hdu1]
    skyspec = np.zeros((len(traces), len(wave)))
    if sky is not None:
        idx = [i for i,p in enumerate(sed.parnames) if p.startswith("sky")]
        skytrace = traces[:, idx]
        for i in tqdm(range(len(skytrace)), desc="Loading sky models"):
            skyspec[i] = sky(skytrace[i])
        hdu2 = fits.ImageHDU(skyspec)
        hdus.append(hdu2)
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(outmodels, overwrite=True)
    fig = plt.figure()
    plt.plot(llike)
    plt.savefig("{}_loglike.png".format(db))
    plt.close(fig)
    sspdict = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age",
               "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
    summary = []
    for i, param in enumerate(sed.ssppars):
        t = traces[:,i]
        m = np.median(t)
        lowerr = m - np.percentile(t, 16)
        uperr = np.percentile(t, 84) - m
        s = "{}=${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(sspdict[param], m,
                                                       uperr, lowerr)
        summary.append(s)
    ############################################################################
    # Normalizing spectra for plots
    skyspec *= norm * np.power(10., unit_norm)
    models *= norm * np.power(10., unit_norm)
    flux *= norm * np.power(10., unit_norm)
    fluxerr *= norm * np.power(10., unit_norm)
    ############################################################################
    skymed = np.median(skyspec, axis=0)
    bestfit = np.median(models, axis=0)
    flux0 = flux - skymed
    # Starting figure
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                            figsize=(2 * context.fig_width, 3))
    ax = plt.subplot(axs[0])
    for skyline in skylines:
        ax.axvspan(skyline - 3, skyline + 3, color="0.9", zorder=-100)
    ax.fill_between(wave, flux + fluxerr, flux - fluxerr, color="0.8")
    ax.fill_between(wave, flux0 + fluxerr, flux0 - fluxerr,
                    "-", label="Spectrum {}".format(specnum), color="C0")
    for i in [0, 2, 1]:
        c = colors[i]
        per = percs[i]
        label = "Model" if i == 1 else None
        ax.fill_between(wave, np.percentile(models, per, axis=0) - skymed,
                         np.percentile(models, percs[i+1], axis=0) - skymed,
                        color=c, label=label, lw=lw)
    ax.set_ylabel("$f_\lambda$ ($10^{{-{0}}}$ erg cm$^{{-2}}$ s$^{{"
                  "-1}}$ \\r{{A}}$^{{-1}}$)".format(unit_norm))
    ax.xaxis.set_ticklabels([])
    ax.text(0.03, 0.88, "   ".join(summary), transform=ax.transAxes, fontsize=6)
    plt.legend(loc=1)
    ax.set_xlim(4700, 9400)
    ylim = ax.get_ylim()
    ax.set_ylim(None, 1.1 * ylim[1])
    # Residual plot
    ax = plt.subplot(axs[1])
    for skyline in skylines:
        ax.axvspan(skyline - 3, skyline + 3, color="0.9", zorder=-100)
    p = flux - bestfit
    sigma_mad = 1.4826 * np.median(np.abs(p - np.median(p)))
    rmse = 100 * sigma_mad / np.median(flux)
    ax.fill_between(wave, skymed - fluxerr, skymed + fluxerr, color="0.8")
    ax.fill_between(wave, fluxerr, - fluxerr, "-", color="C0")
    for i in [0, 2, 1]:
        c = colors[i]
        per = percs[i]
        label = "RMSE={:.2f}\%".format(rmse) if i==1 \
                 else None
        ax.fill_between(wave, np.percentile(models, per, axis=0) - skymed -
                        flux0,
                         np.percentile(models, percs[i+1], axis=0) - skymed -
                        flux0,
                        color=c, lw=lw, label=label)
    # ax.set_ylim(-5 * sigma_mad, 5 * sigma_mad)
    ax.axhline(y=0, ls="--", c="k", lw=1, zorder=1000)
    ax.set_xlabel(r"$\lambda$ (\r{A})")
    ax.set_ylabel("$\Delta f_\lambda$")
    ax.set_xlim(4700, 9400)
    plt.legend(loc=3)
    plt.subplots_adjust(left=0.06, right=0.995, hspace=0.02, top=0.99,
                        bottom=0.11)
    fig.align_ylabels(axs)
    plt.savefig("{}.png".format(outfig), dpi=250)
    # plt.show()
    plt.close()
    return

def plot_corner(trace, outroot, title=None, redo=False):
    title = "" if title is None else title
    output = "{}_corner.png".format(outroot)
    if os.path.exists(output) and not redo:
        return
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
    N = len(trace.colnames)
    params = trace.colnames
    data = np.stack([trace[p] for p in params]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    title = [title]
    for i, param in enumerate(params):
        s = "{0}$={1:.2f}^{{+{2:.2f}}}_{{-{3:.2f}}}$".format(
            labels[param], v[i], vuerr[i], vlerr[i])
        title.append(s)
    fig, axs = plt.subplots(N, N, figsize=(context.fig_width, 3.5))
    grid = np.array(np.meshgrid(params, params)).reshape(2, -1).T
    for i, (p1, p2) in enumerate(grid):
        i1 = params.index(p1)
        i2 = params.index(p2)
        ax = axs[i // N, i % N]
        ax.tick_params(axis="both", which='major',
                       labelsize=4)
        if i // N < i % N:
            ax.set_visible(False)
            continue
        x = data[:,i1]
        if p1 == p2:
            sns.kdeplot(x, shade=True, ax=ax, color="C0")
        else:
            y = data[:, i2]
            r, p = stats.pearsonr(x, y)
            c = "r" if p <= 0.01 else "b"
            sns.kdeplot(x, y, shade=True, ax=ax, cmap="Blues")
            ax.axhline(np.median(y), ls="-", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 16), ls="--", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 84), ls="--", c="k", lw=0.5)
            props = dict(facecolor='white', alpha=0.95, edgecolor="none")
            signal = "+" if r >= 0 else ""
            ax.text(0.37, 0.80, "$r={}{:.2f}$".format(signal, r),
                    transform=ax.transAxes, fontsize=4, color=c, bbox=props)
        if i > N * (N - 1) - 1:
            ax.set_xlabel(labels[p1], size=7)
        else:
            ax.xaxis.set_ticklabels([])
        if i in np.arange(0, N * N, N)[1:]:
            ax.set_ylabel(labels[p2], size=7)
        else:
            ax.yaxis.set_ticklabels([])
        ax.axvline(np.median(x), ls="-", c="k", lw=0.5)
        ax.axvline(np.percentile(x, 16), ls="--", c="k", lw=0.5)
        ax.axvline(np.percentile(x, 84), ls="--", c="k", lw=0.5)
    plt.text(0.6, 0.7, "\n".join(title), transform=plt.gcf().transFigure,
             size=8)
    plt.subplots_adjust(left=0.11, right=0.99, top=0.98, bottom=0.08,
                        hspace=0.04, wspace=0.04)
    fig.align_ylabels()
    for fmt in ["png", "pdf"]:
        output = "{}_corner.{}".format(outroot, fmt)
        plt.savefig(output, dpi=300)
    plt.close(fig)
    return

def load_traces(db, params):
    if not os.path.exists(db):
        return None
    ntraces = len(os.listdir(db))
    data = [np.load(os.path.join(db, _, "samples.npz")) for _ in
            os.listdir(db)]
    traces = []
    # Reading weights
    for param in params:
        if param.startswith("w_"):
            w = np.vstack([data[num]["w"] for num in range(ntraces)])
            n = param.split("_")[1]
            v = w[:, int(n) - 1]
        else:
            v = np.vstack([data[num][param] for num in range(ntraces)]
                          ).flatten()
        traces.append(v)
    traces = np.column_stack(traces)
    return traces

def compare_traces(flux, fluxerr, sed, t1, t2):
    loglike = pb.NormalLogLike(flux, sed, obserr=fluxerr)
    names = ["EMCEE", "SMC"]
    for j, t in enumerate([t1, t2]):
        llike = np.zeros(len(t))
        for i in tqdm(range(len(t)),
                      desc="Loading spectra for trace {}".format(j+1)):
            llike[i] = loglike(t[i])
        x = np.linspace(0,1, len(t))
        plt.plot(x, llike, label=names[j])
    plt.legend()
    plt.show()
    return

def run_ngc3311(targetSN=250, velscale=200, ltype=None, sample=None,
                redo=False, nssps=1, porder=30, dataset="MUSE",
                postprocessing=True):
    """ Run pb full spectrum fitting. """
    # os.environ["OMP_NUM_THREADS"] = "8"
    ltype = "normal2" if ltype is None else ltype
    sample = "all" if sample is None else sample
    nssps_str = "" if nssps == 1 else "_{}ssps".format(nssps)
    # Setting up directories
    wdir = os.path.join(context.data_dir, dataset,
                        "voronoi/sn{}/sci".format(targetSN))
    emcee_dir = os.path.join(os.path.split(wdir)[0], "EMCEE{}_{}".format(
                             nssps_str, ltype))
    mcmc_dir = os.path.join(os.path.split(wdir)[0], "MCMC{}_{}".format(
                            nssps_str, ltype))
    for dir_ in [emcee_dir, mcmc_dir]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    specnames = sorted([_ for _ in sorted(os.listdir(wdir)) if _.endswith(
                        ".fits")])
    # Check int arguments in sys.argv to set bins
    idxs = []
    for argv in sys.argv:
        try:
            idxs.append(int(argv))
        except:
            continue
    idxs = [i for i in idxs if i < len(specnames)]
    if len(idxs) > 0:
        specnames = [specnames[i] for i in idxs]
    # Read first spectrum to set the dispersion
    data = Table.read(os.path.join(wdir, specnames[0]))
    wave_lin = data["wave"].data
    flam = data["flam"].data
    _, logwave, velscale = util.log_rebin([wave_lin[0], wave_lin[-1]], flam,
                                    velscale=velscale)
    wave = np.exp(logwave)[1:-1]
    # Masking wavelengths with sky lines/ residuals
    skylines = np.array([4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                         5895, 5888, 5990, 5895, 6300, 6363, 6386, 6562,
                         6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                         8747, 8757, 8767, 8777, 8787, 8797, 8827, 8836, 8919,
                         9310])
    goodpixels = np.arange(len(wave))
    for line in skylines:
        sky = np.argwhere((wave <= line - 5) | (wave >= line + 5)).ravel()
        goodpixels = np.intersect1d(goodpixels, sky)
    # wave = wave[goodpixels]
    print("Producing SED model...")
    use_emission = False if ltype.startswith("studt2") else True
    sed, mw, sky = build_sed_model(wave, sample=sample, nssps=nssps,
                                   porder=porder, use_emission=use_emission)
    priors = make_priors(sed)
    for specname in specnames:
        print("Processing spectrum {}".format(specname))
        name = specname.split(".")[0]
        binnum = name.split("_")[2]
        data = Table.read(os.path.join(wdir, specname))
        wlin = data["wave"].data
        flam = data["flam"].data
        flamerr = data["flamerr"].data
        flam, flamerr = spectres(wave, wlin, flam, spec_errs=flamerr)
        norm = np.median(flam)
        flam /= norm
        flamerr /= norm
        # Setting the loglikelyhood
        if ltype == "normal2":
            loglike = pb.Normal2LogLike(flam, sed, obserr=flamerr)
        elif ltype == "studt":
            loglike = pb.StudTLogLike(flam, sed, obserr=flamerr)
        elif ltype == "studt2":
            loglike = pb.StudT2LogLike(flam, sed, obserr=flamerr)
        # Start with an MCMC using Metropolis step
        # mcmc_db = os.path.join(mcmc_dir, "{}".format(name))
        # if not os.path.exists(mcmc_db) and postprocessing:
        #     continue
        # if not os.path.exists(mcmc_db):
        #     print("Compiling pymc3 model for MH model...")
        #     model = make_pymc3_model(sed, loglike)
        #     run_mcmc(model, mcmc_db, redo=False, method="MCMC")
        # Run second method using initial results from MH run
        emcee_db = os.path.join(emcee_dir, "{}.h5".format(name))
        if not os.path.exists(emcee_db) and postprocessing:
            continue
        if not os.path.exists(emcee_db) or redo:
            os.environ["OMP_NUM_THREADS"] = "8"
            print("Running EMCEE...")
            run_sampler(loglike, priors, emcee_db)
            # run_emcee(flam, flamerr, sed, emcee_db, priors, loglike=ltype)
            os.environ["OMP_NUM_THREADS"] = "2"
        reader = emcee.backends.HDFBackend(emcee_db)
        trace = reader.get_chain(discard=800, flat=True, thin=100)
        print("Using trace with {} samples".format(len(trace)))
        emcee_traces = trace[:, :len(sed.parnames)]
        idx = [sed.parnames.index(p) for p in sed.sspcolnames]
        ptrace_emcee = Table(emcee_traces[:, idx], names=sed.sspcolnames)
        if postprocessing:
            print("Producing corner plots...")
            title = "Spectrum {}".format(binnum)
            plot_corner(ptrace_emcee, emcee_db, title=title, redo=False)
            print("Producing fitting figure...")
            plot_fitting(wave, flam, flamerr, sed, emcee_traces, emcee_db,
                         redo=True , sky=sky, norm=norm, skylines=skylines)
            print("Making summary table...")
            outtab = os.path.join(emcee_db.replace(".h5", "_results.fits"))
            summary_trace = Table(trace, names=loglike.parnames)
            make_table(summary_trace, outtab)

if __name__ == "__main__":
    postprocessing = True if getpass.getuser() == "kadu" else False
    ltype = "normal2"
    porder= 45 if ltype == "studt2" else 30
    run_ngc3311(ltype=ltype, postprocessing=postprocessing, porder=porder,
                nssps=1, sample="all")