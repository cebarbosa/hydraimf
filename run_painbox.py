# -*- coding: utf-8 -*-
"""

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import copy

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import cm
try:
    import ppxf_util as util
except:
    import ppxf.ppxf_util as util
from spectres import spectres
import pymc3 as pm
import theano.tensor as tt
from tqdm import tqdm
import seaborn as sns
import emcee
from scipy import stats

import context
from paintbox import paintbox as pb

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
    # Creating a model including LOSVD
    sed = (stars + emission) * poly
    # Setting properties that may be useful later in modeling
    sed.ssppars = limits
    sed.sspcolnames = params.colnames
    sed.sspparams = params
    sed.line_names = line_names
    sed.porder = porder
    sed.nssps = nssps
    return sed, norm

def make_p0(sed):
    """ Produces an initial guess for the model. """
    polynames = ["p{}".format(i + 1) for i in range(sed.porder)]
    p0 = []
    for param in sed.parnames:
        if len(param.split("_")) == 2:
            pname, n = param.split("_")
        else:
            pname = param
        # Weight of the ssp
        if pname == "w":
            p0.append(1.)
        # Stellar population parameters
        elif pname in sed.ssppars:
            vmin, vmax = sed.ssppars[pname]
            p0.append(0.5 * (vmin + vmax))
        # Dust attenuation parameters
        elif param == "Av":
            p0.append(0.1)
        elif param == "Rv":
            p0.append(3.1)
        elif param == "V":
            p0.append(3800)
        elif param == "sigma":
            p0.append(200)
        # Emission lines
        elif param in sed.line_names:
            p0.append(1)
        elif param == "V_gas":
            p0.append(3850)
        elif param == "sigma_gas":
            p0.append(75.)
        # Polynomia parameters
        elif pname == "p0":
            p0.append(1.)
        elif param in polynames:
            p0.append(0.01)
    return np.array(p0)

# Deterministic function for stick breaking
def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

def make_pymc3_model(flux, sed, loglike=None, fluxerr=None):
    loglike = "normal2" if loglike is None else loglike
    flux = flux.astype(np.float)
    fluxerr = np.ones_like(flux) if fluxerr is None else fluxerr
    model = pm.Model()
    polynames = ["p{}".format(i + 1) for i in range(sed.porder)]
    with model:
        # alpha = pm.Gamma('alpha', 1., 1.)
        # beta = pm.Beta('beta', 1, alpha, shape=sed.nssps)
        # w = pm.Deterministic('w', stick_breaking(beta))
        theta = []
        for param in sed.parnames:
            if len(param.split("_")) == 2:
                pname, n = param.split("_")
            else:
                pname = param
                n = 1
            # Weight of the ssp
            if pname == "w":
                theta.append(w[int(n)-1])
            # Stellar population parameters
            if pname in sed.ssppars:
                vmin, vmax = sed.ssppars[pname]
                vinit = float(0.5 * (vmin + vmax))
                v = pm.Uniform(param, lower=float(vmin), upper=float(vmax),
                               testval=vinit)
                theta.append(v)
            # Dust attenuation parameters
            elif param == "Av":
                Av = pm.Exponential("Av", lam=1 / 0.4, testval=0.1)
                theta.append(Av)
            elif param == "Rv":
                BNormal = pm.Bound(pm.Normal, lower=0)
                Rv = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
                theta.append(Rv)
            elif param == "V":
                # Stellar kinematics
                V = pm.Normal("V", mu=3800., sigma=50., testval=3805.)
                theta.append(V)
            elif param == "sigma":
                sigma = pm.Uniform(param, lower=100, upper=500, testval=185.)
                theta.append(sigma)
            # Emission lines
            elif param in sed.line_names:
                v = pm.HalfNormal(param, 1., testval=1.)
                theta.append(v)
            elif param == "V_gas":
                V_gas = pm.Uniform("V_gas", lower=3600., upper=4100.,
                                   testval=3805.)
                theta.append(V_gas)
            elif param == "sigma_gas":
                sigma_gas = pm.Uniform("sigma_gas", lower=60., upper=120,
                                       testval=85.)
                theta.append(sigma_gas)
            # Polynomial parameters
            elif pname == "p0":
                p0 = pm.Normal("p0", mu=1, sd=0.1, testval=1.)
                theta.append(p0)
            elif param in polynames:
                pn = pm.Normal(param, mu=0, sd=0.01, testval=0.)
                theta.append(pn)
        if loglike == "studt":
            nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(nu)
        if loglike == "normal2":
            x = pm.Normal("x", mu=0, sd=1, testval=0.)
            s = pm.Deterministic("S", 1. + pm.math.exp(x))
            theta.append(s)
        theta = tt.as_tensor_variable(theta).T
        logl = pb.TheanoLogLikeInterface(flux, sed, loglike=loglike,
                                          obserr=fluxerr)
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})

    return model

def run_mcmc(model, db, redo=False, method=None):
    summary = "{}.csv".format(db)
    method = "MCMC" if method is None else method
    if os.path.exists(summary) and not redo:
        return
    with model:
        if method == "MCMC":
            trace = pm.sample(250, tune=250, step=pm.Metropolis())
        elif method == "NUTS":
            trace = pm.sample()
        elif method == "SMC":
            trace = pm.sample_smc(draws=250, threshold=0.3,
                                      progressbar=True )
        df = pm.stats.summary(trace)
        df.to_csv(summary)
    pm.save_trace(trace, db, overwrite=True)
    return trace

def run_emcee(flam, flamerr, sed, db, loglike="normal2"):
    pnames = copy.deepcopy(sed.parnames)
    if loglike == "normal2":
        pnames.append("S")
    if loglike == "studt":
        pnames.append("nu")
    ndim = len(pnames)
    nwalkers = 2 * ndim
    polynames = ["p{}".format(i+1) for i in range(sed.porder)]
    pos = np.zeros((nwalkers, ndim))
    priors = []
    for i, param in enumerate(pnames):
        if len(param.split("_")) == 2:
            pname, n = param.split("_")
        else:
            pname = param
        ########################################################################
        # Setting first guess and limits of models
        ########################################################################
        # Stellar population parameters
        if pname in sed.ssppars:
            vmin, vmax = sed.ssppars[pname]
            prior = stats.uniform(vmin, vmax - vmin)
        elif param == "Av":
            prior = stats.expon(0, 0.4)
        elif param== "Rv":
            vmin = 0
            vmax = np.infty
            mu = 3.1
            sd = 1.
            a, b = (vmin - mu) / sd, (vmax - mu) / sd
            prior = stats.truncnorm(a, b, mu, sd)
        elif param in ["V", "V_gas"]:
            prior = stats.norm(3800, 100)
        elif param == "sigma":
            prior = stats.uniform(100, 400)
        elif param == "sigma_gas":
            prior = stats.uniform(60, 60)
        elif param in sed.line_names:
            prior = stats.halfnorm(0, 1)
        elif param == "p0":
            prior = stats.norm(1, 0.1)
        elif param in polynames:
            prior = stats.norm(0, 0.02)
        elif param == "S":
            prior = stats.lognorm(1, 1)
        else:
            raise("Error: parameter not known: {}".format(param))
        priors.append(prior.logpdf)
        pos[:, i] = prior.rvs(nwalkers)
    if loglike == "normal2":
        log_likelihood = pb.Normal2LogLike(flam, sed, obserr=flamerr)
    def log_probability(theta):
        lp = np.sum([prior(val) for prior, val in zip(priors, theta)])
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    backend = emcee.backends.HDFBackend(db)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    backend=backend)
    sampler.run_mcmc(pos, 500, progress=True)
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

def make_table(trace, binnum, outtab):
    data = np.array([trace[p].data for p in trace.colnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = Table()
    tab["BIN"] = [binnum]
    for i, param in enumerate(trace.colnames):
        tab[param] = [round(v[i], 5)]
        tab["{}_lerr".format(param)] = [round(vlerr[i], 5)]
        tab["{}_uerr".format(param)] = [round(vuerr[i], 5)]
    tab.write(outtab, overwrite=True)
    return tab

def plot_fitting(wave, flux, fluxerr, sed, traces, db, redo=True):
    outfig = "{}_fitting".format(db.replace(".h5", ""))
    specnum = os.path.split(db)[1].replace(".h5", "").split("_")[-1]
    if os.path.exists("{}.png".format(outfig)) and not redo:
        return
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    spec = np.zeros((len(traces), len(wave)))
    loglike = pb.NormalLogLike(flux, sed, obserr=fluxerr)
    llike = np.zeros(len(traces))
    for i in tqdm(range(len(traces)), desc="Loading spectra for plots and "
                                           "table..."):
        spec[i] = sed(traces[i])
        llike[i] = loglike(traces[i])
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
    y = np.median(spec, axis=0)
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                            figsize=(2 * context.fig_width, 3.5))
    ax = plt.subplot(axs[0])
    ax.errorbar(wave, flux, yerr=fluxerr, fmt="-", mec="w", mew=0.4,
                elinewidth=0.8, label="Spectrum {}".format(specnum))
    for c, per in zip(colors, percs):
        ax.fill_between(wave, np.percentile(spec, per, axis=0),
                         np.percentile(spec, per + 10, axis=0), color=c)
    ax.errorbar(wave, y, fmt="-", mec="w", mew=0.4,
                elinewidth=0.8, label="Model")

    ax.set_ylabel("Normalized flux")
    ax.xaxis.set_ticklabels([])
    ax.text(0.03, 0.88, "   ".join(summary), transform=ax.transAxes, fontsize=6)
    plt.legend()
    ax = plt.subplot(axs[1])
    for c, per in zip(colors, percs):
        ax.fill_between(wave, flux - np.percentile(spec, per, axis=0),
                         flux - np.percentile(spec, per + 10, axis=0), \
                              color=c)
    ax.errorbar(wave, flux - y, fmt="-", mec="w", mew=0.4,
                elinewidth=0.8, c="C1")
    ax.axhline(y=0, ls="--", c="k", lw=0.8)
    ax.set_xlabel(r"$\lambda$ (\r{A})")
    ax.set_ylabel("Residue")
    plt.subplots_adjust(left=0.08, right=0.99, hspace=0.02, top=0.953,
                        bottom=0.095)
    plt.savefig("{}.png".format(outfig), dpi=250)
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
            sns.kdeplot(x, y, shade=True, ax=ax, cmap="Blues")
            ax.axhline(np.median(y), ls="-", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 16), ls="--", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 84), ls="--", c="k", lw=0.5)
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
    plt.subplots_adjust(left=0.12, right=0.995, top=0.98, bottom=0.08,
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
                redo=False, nssps=1, porder=30):
    """ Run pb full spectrum fitting. """
    # os.environ["OMP_NUM_THREADS"] = "8"
    ltype = "normal2" if ltype is None else ltype
    sample = "all" if sample is None else sample
    nssps_str = "" if nssps == 1 else "_{}ssps".format(nssps)
    # Setting up directories
    imgname, cubename = context.get_field_files("fieldA")
    wdir = os.path.join(os.path.split(cubename)[0], "sn{}/sci".format(targetSN))
    emcee_dir = os.path.join(os.path.split(wdir)[0], "EMCEE{}".format(nssps_str))
    mcmc_dir = os.path.join(os.path.split(wdir)[0], "MCMC{}".format(nssps_str))
    smc_dir =os.path.join(os.path.split(wdir)[0], "SMC{}".format(nssps_str))
    for dir_ in [emcee_dir, mcmc_dir, smc_dir]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    specnames = [_ for _ in sorted(os.listdir(wdir)) if _.endswith(".fits")]
    # Read first spectrum to set the dispersion
    data = Table.read(os.path.join(wdir, specnames[0]))
    wave_lin = data["wave"].data
    flam = data["flam"].data
    _, logwave, velscale = util.log_rebin([wave_lin[0], wave_lin[-1]], flam,
                                    velscale=velscale)
    wave = np.exp(logwave)[1:-1]
    sed, mw = build_sed_model(wave, sample=sample, nssps=nssps, porder=porder)
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
        # Start with an MCMC using Metropolis step
        mcmc_db = os.path.join(mcmc_dir, "{}".format(name))
        if not os.path.exists(mcmc_db):
            print("Compiling pymc3 model for MH model...")
            model = make_pymc3_model(flam, sed, fluxerr=flamerr,
                                     loglike=ltype)
            run_mcmc(model, mcmc_db, redo=False, method="MCMC")
        # Run second method using initial results from MH run
        emcee_db = os.path.join(emcee_dir, "{}.h5".format(name))
        if not os.path.exists(emcee_db) or redo:
            run_emcee(flam, flamerr, sed, emcee_db)
        reader = emcee.backends.HDFBackend(emcee_db)
        samples = reader.get_chain(discard=100, flat=True, thin=50)
        emcee_traces = samples[:, :len(sed.parnames)]
        idx = [sed.parnames.index(p) for p in sed.sspcolnames]
        ptrace_emcee = Table(emcee_traces[:, idx], names=sed.sspcolnames)
        ########################################################################
        # Testing Sequential Monte Carlo
        # smc_db = os.path.join(smc_dir, name)
        # if not os.path.exists(smc_db):
        #     print("Compiling pymc3 model for SMC model...")
        #     model = make_pymc3_model(flam, sed, fluxerr=flamerr,
        #                              loglike=ltype)
        #     run_mcmc(model, smc_db, redo=False, method="SMC")
        # smc_traces = load_traces(smc_db, sed.parnames)
        # ptrace_smc = Table(smc_traces[:, idx], names=sed.sspcolnames)
        # plot_corner(ptrace_smc, smc_db, title=title, redo=redo)
        # plot_fitting(wave, flam, flamerr, sed, smc_traces, smc_db, redo=redo)
        ########################################################################
        print("Producing corner plots...")
        title = "Spectrum {}".format(binnum)
        plot_corner(ptrace_emcee, emcee_db, title=title, redo=redo)
        print("Producing fitting figure...")
        plot_fitting(wave, flam, flamerr, sed, emcee_traces, emcee_db, redo=redo)
        print("Making summary table...")
        outtab = os.path.join(emcee_db.replace(".h5", "_results.fits"))
        make_table(ptrace_emcee, binnum, outtab)

if __name__ == "__main__":
    run_ngc3311(sample="all")