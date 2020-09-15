# -*- coding: utf-8 -*-
"""

Created on 03/08/2020

Author : Carlos Eduardo Barbosa

Calculate properties of NGC 3311 based on traces.
"""
from __future__ import print_function, division

import os
import itertools

import numpy as np
from astropy.table import Table, hstack, vstack
from scipy.interpolate import LinearNDInterpolator
import astropy.units as u
from tqdm import tqdm
import emcee
import speclite
import speclite.filters
from scipy.stats.distributions import chi2
from scipy.stats import pearsonr

import context
from run_painbox import build_sed_model

class Mass2Light:
    def __init__(self, imf="bi"):
        self.imf = imf
        self.tablefile = os.path.join(context.tables_dir,
                                      "sdss_{}_iTp0.00.MAG".format(imf))
        self.table = Table.read(self.tablefile, format="ascii.basic")
        imf = np.array([float(_[3:7]) for _ in self.table["model"]])
        z = np.array([float(_[8:13].replace("p", "+").replace("m", "-")) for
                      _ in self.table["model"]])
        t = np.array(([float(_[14:21]) for _ in self.table["model"]]))
        x = np.column_stack([imf, z, t])
        y = self.table["ML(r_SDSS)"].data
        self.f = LinearNDInterpolator(x, y, fill_value=0.)

    def __call__(self, p):
        return self.f(p)

def calc_mass2light(targetSN=250, dataset="MUSE", redo=False):
    Mg_sun = 4.65  # AB
    ps = (5.5555555555 * 1e-5 * u.degree).to(u.arcsec)
    re = 34 * u.arcsec
    reerr = 4 * u.arcsec
    D = 50 * u.Mpc
    Derr = 10 * u.Mpc
    pskpc = D.to(u.kpc) * np.tan(ps.to(u.radian))
    sdss_r = speclite.filters.load_filters('sdss2010-r')
    wdir = os.path.join(context.data_dir, dataset,
                        "voronoi/sn{}".format(targetSN))
    outtable = os.path.join(wdir, "mass2light.fits")
    if os.path.exists(outtable) and not redo:
        return
    emcee_dir = os.path.join(wdir, "EMCEE")
    spec_dir = os.path.join(wdir, "sci")
    dbs = sorted([_ for _ in os.listdir(emcee_dir) if _.endswith(".h5")])
    ts = []
    sed = build_sed_model(np.linspace(4500, 9000, 1000), sample="all")[0]
    params = np.array(sed.sspcolnames + ["sigma"])
    idx_trace = [sed.parnames.index(p) for p in params]
    m2l = Mass2Light()
    m2l_un = Mass2Light(imf="un")
    parnames = sed.sspcolnames + ["sigma", "M2L", "alpha", "logSigma"]
    idx = np.arange(len(parnames))
    idxs = list(itertools.permutations(idx, 2))
    pairs = list(itertools.permutations(parnames, 2))
    rs = np.zeros((len(pairs), len(dbs)))
    majaxis = np.zeros_like(rs)
    minaxis = np.zeros_like(rs)
    angs = np.zeros_like(rs)
    pvals = np.zeros_like(rs)
    s = chi2.ppf(0.68, df=2)
    for n, db in enumerate(tqdm(dbs, desc="Calculating M/L ratio")):
        t = Table()
        t["BIN"] = ["_".join(db.replace(".h5", "").split("_")[::2])]
        reader = emcee.backends.HDFBackend(os.path.join(emcee_dir, db))
        samples = reader.get_chain(discard=800, flat=True, thin=100).T[
            idx_trace].T
        chsize = len(samples)
        mls = m2l(samples[:,:3])
        ml = np.percentile(mls, 50)
        t["M2L"] = [ml]
        t["M2L_lerr"] = [ml - np.percentile(mls, 16)]
        t["M2L_uerr"] = [np.percentile(mls, 84) - ml]
        # Calculating alpha parameter
        samples_kroupa = np.copy(samples)
        samples_kroupa[:, 0] = 1.3
        mlk = m2l(samples_kroupa[:,:3])
        samples_salpeter = np.copy(samples)
        samples_salpeter[:, 0] = 1.35
        mlsalp = m2l_un(samples_salpeter[:,:3])
        alphas = mls / mlk
        alphas_salp = mlsalp / mlk
        a = np.median(alphas)
        b = np.median(alphas_salp)
        print(a, b)
        alpha = np.percentile(alphas, 50)
        t["alpha"] = [alpha]
        t["alpha_lerr"] = [alpha - np.percentile(alphas, 16)]
        t["alpha_uerr"] = [np.percentile(alphas, 84) - alpha]
        ts.append(t)
        # Calculating surface density
        data = Table.read(os.path.join(spec_dir, db.replace(".h5", ".fits")))
        flam = data["flam"].data
        wave = data["wave"].data * data["wave"].unit
        flamerr = data["flamerr"].data
        flams = np.random.normal(flam, flamerr,
                                 (chsize, len(flamerr))) * data["flam"].unit
        magr = sdss_r.get_ab_magnitudes(flams, wave)["sdss2010-r"].data
        Magr = magr - 5 * np.log10(
               np.random.normal(D.to(u.pc).value, Derr.to(u.pc).value, chsize) /
                                   10)
        L = np.power(10, -0.4 * (Magr - Mg_sun))
        Mstar = L * mls * u.M_sun
        mustar = Mstar / pskpc / pskpc
        logmustar = np.log10(mustar.value)
        lm = np.percentile(logmustar, 50)
        mustar_unit = u.M_sun / u.kpc / u.kpc
        t["logSigma"] = [lm] * mustar_unit
        t["logSigma_lerr"] = [lm - np.percentile(logmustar, 16)] * mustar_unit
        t["logSigma_uerr"] = [np.percentile(logmustar, 84) - lm] * mustar_unit
        # Calculating correlations between parameters
        trace = np.column_stack([samples, mls, alphas, logmustar]).T
        for k, (i, j) in enumerate(idxs):
            x = trace[i]
            y = trace[j]
            r, p = pearsonr(x, y)
            cov = np.cov(np.array([x, y]))
            w, v = np.linalg.eig(cov)
            imax, imin = np.argmax(w), np.argmin(w)
            v1, v2 = v[:, imax], v[:, imin]
            w1, w2 = w[imax], w[imin]
            ang = np.rad2deg(np.arctan2(v1[1], v1[0]))
            a = 2 * np.sqrt(s * w1)
            b = 2 * np.sqrt(s * w2)
            rs[k, n] = r
            majaxis[k, n] = a
            minaxis[k, n] = b
            angs[k, n] = ang
            pvals[k, n] = p
    table = vstack(ts)
    table.write(outtable, overwrite=True)
    # Saving correlations
    r = np.median(rs, axis=1)
    sort = np.argsort(np.abs(r))[::-1]
    pairs = [pairs[s] for s in sort]
    r = r[sort]
    rsd = np.std(rs, axis=1)[sort]
    a = np.median(majaxis, axis=1)[sort]
    b = np.median(minaxis, axis=1)[sort]
    asd = np.std(majaxis, axis=1)[sort]
    bsd = np.std(minaxis, axis=1)[sort]
    ang = np.median(angs, axis=1)[sort]
    angsd = np.std(angs, axis=1)[sort]
    p = np.median(pvals, axis=1)[sort]
    psd = np.std(pvals, axis=1)[sort]
    p1 = [p[0] for p in pairs]
    p2 = [p[1] for p in pairs]
    names = ["param1", "param2", "r", "rerr", "a", "aerr", "b", "berr",
             "ang", "angerr", "p", "perr"]
    tab = Table([p1, p2, r, rsd, a, asd, b, bsd, ang, angsd, p, psd],
                names=names)
    tab.write(os.path.join(wdir, "fit_stats.fits"), overwrite=True)

if __name__ == "__main__":
    calc_mass2light(redo=True)