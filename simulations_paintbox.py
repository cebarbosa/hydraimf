"""
Simulating stellar population analysis with Paintbox

"""
import os
import copy
import getpass

import numpy as np
from scipy import stats
from astropy.table import Table
from astropy.io import fits
import ppxf.ppxf_util as util
import zeus
import paintbox as pb

import context

def make_wave_array(velscale=200):
    outtable = os.path.join(os.getcwd(), "wave_velscale{}.fits".format(
                                                                velscale))
    if os.path.exists(outtable):
        t = Table.read(outtable)
        return t["wave"].data
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250/sci")
    specnames = sorted([_ for _ in sorted(os.listdir(wdir)) if
                        _.endswith(".fits")])
    data = Table.read(os.path.join(wdir, specnames[0]))
    wave_lin = data["wave"].data
    flam = data["flam"].data
    _, logwave, velscale = util.log_rebin([wave_lin[0], wave_lin[-1]], flam,
                                    velscale=velscale)
    wave = np.exp(logwave)[1:-1]
    t = Table([wave], names=["wave"])
    t.write("wave_velscale{}.fits".format(velscale))
    return wave

def build_sed_model(wave, w1=4500, w2=9400, velscale=200, sample=None,
                    fwhm=2.95, nssps=1, porder=0):
    """ Build model for simulations"""
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
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    # Creating a model including LOSVD
    sed = stars * poly
    limits["Rv"] = (2.5, 5.)
    limits["Av"] = (0., 0.2)
    limits["V"] = (3700, 3900)
    limits["sigma"] = (150, 350)
    limits["p0"] = (0.5, 1.5)
    return sed, limits

def create_simulations(sed, limits, sn=50, nsim=1000, redo=False):
    simfile = os.path.join(os.getcwd(), "simulations.fits")
    if os.path.exists(simfile) and not redo:
        return
    psim = np.zeros((nsim, len(sed.parnames)))
    for i, p in enumerate(sed.parnames):
        psim[:,i] = np.random.uniform(limits[p][0], limits[p][1], nsim)
    params = Table(psim, names=sed.parnames)
    simspec = np.zeros((nsim, len(sed.wave)))
    noises = np.zeros((nsim))
    for n in range(nsim):
        spec = sed(psim[n])
        signal = np.median(spec)
        noise = signal / sn
        simspec[n] = spec + np.random.normal(0, noise, len(sed.wave))
        noises[n] = noise
    hdu0 = fits.PrimaryHDU(simspec)
    hdu0.header["EXTNAME"] = "DATA"
    hdu1 = fits.BinTableHDU(params)
    hdu1.header["EXTNAME"] = "PARAMS"
    hdu2 = fits.BinTableHDU(Table([wave], names=["wave"]))
    hdu2.header["EXTNAME"] = "WAVE"
    hdu3 = fits.ImageHDU(noises)
    hdu3.header["EXTNAME"] = "NOISE"
    hdulist = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
    hdulist.writeto(simfile, overwrite=True)
    return

def run_zeus(sed, limits, loglike, db, redo=False, nsteps=1000):
    if os.path.exists(db) and not redo:
        return
    ndim = len(sed.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    priors = []
    for i, param in enumerate(sed.parnames):
        vmin, vmax = limits[param]
        prior = stats.uniform(vmin, vmax - vmin)
        priors.append(prior.logpdf)
        pos[:, i] = prior.rvs(nwalkers)
    def log_probability(theta):
        lp = np.sum([prior(val) for prior, val in zip(priors, theta)])
        if not np.isfinite(lp) or np.isnan(lp):
            return -np.inf
        return loglike(theta)
    sampler = zeus.sampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.get_chain(flat=True, discard=nsteps // 2, thin=50)
    table = Table(chain, names=sed.parnames)
    table.write(db, overwrite=True)
    return

def fit_simulations(sed, limits, start=0, limitto=100):
    specs = fits.getdata("simulations.fits")
    noise = fits.getdata("simulations.fits", extname="NOISE")
    zeus_dir = os.path.join(os.getcwd(), "zeus")
    if not os.path.exists(zeus_dir):
        os.mkdir(zeus_dir)
    for i, flam in enumerate(specs):
        if i < start or i > start + limitto:
            continue
        name = "{:04d}".format(i)
        print("Processing spectrum {} / {}".format(i+1, len(specs)))
        norm = np.median(flam)
        flam /= norm
        flamerr = np.full_like(flam, noise[i] / norm)
        loglike = pb.NormalLogLike(flam, sed, obserr=flamerr)
        db = os.path.join(zeus_dir, "{}_chain.fits".format(name))
        run_zeus(sed, limits, loglike, db)

if __name__ == "__main__":
    data_dir = os.path.join(context.data_dir, "pbsim")
    sspmodel = "emiles"
    sample, nsim, sn = "all", 1000, 50
    simname = "{}_{}_nsim{}_sn{}".format(sspmodel, sample, nsim, sn)
    wdir = os.path.join(data_dir, simname)
    for _dir in [data_dir, wdir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)
    os.chdir(wdir)
    wave = make_wave_array()
    sed, limits = build_sed_model(wave, sample=sample)
    create_simulations(sed, limits, nsim=nsim)
    fit_simulations(sed, limits, start=100)
