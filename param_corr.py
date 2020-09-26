# -*- coding: utf-8 -*-
"""

Created on 29/07/2020

Author : Carlos Eduardo Barbosa

Checking average correlation between parameters in NGC 3311.
"""
from __future__ import print_function, division

import os
import itertools

import numpy as np
from astropy.table import Table
from scipy.stats.distributions import chi2
from scipy.stats import pearsonr
import emcee
from tqdm import tqdm

import context
from run_paintbox import build_sed_model

def calc_correlations(targetSN=250, dataset="MUSE"):
    wdir = os.path.join(context.data_dir, dataset,
                        "voronoi/sn{}".format(targetSN))
    emcee_dir = os.path.join(wdir, "EMCEE")
    dbs = sorted([_ for _ in os.listdir(emcee_dir) if _.endswith(".h5")])
    sed = build_sed_model(np.linspace(4500, 9000, 1000), sample="test")[0]
    params = np.array(sed.sspcolnames + ["sigma"])
    idx = [sed.parnames.index(p) for p in params]
    idxs = list(itertools.permutations(idx, 2))
    pairs = list(itertools.permutations(params, 2))
    rs = np.zeros((len(pairs), len(dbs)))
    majaxis = np.zeros_like(rs)
    minaxis = np.zeros_like(rs)
    angs = np.zeros_like(rs)
    pvals = np.zeros_like(rs)
    s = chi2.ppf(0.68, df=2)
    for n, db in enumerate(tqdm(dbs)):
        reader = emcee.backends.HDFBackend(os.path.join(emcee_dir, db))
        samples = reader.get_chain(discard=800, flat=True, thin=100)
        trace = samples.T
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
    tab.write(os.path.join(wdir, "fit_stats_only_sedpars.fits"), overwrite=True)
    # Make latex table
    labels = {"T": "Age (Gyr)", "Z": "[Z/H]", "alphaFe": "[$\\alpha$/Fe]",
              "NaFe": "[Na/Fe]", "sigma": "$\\sigma_*$ (km/s)", "imf":
                  "$\\Gamma_b$"}
    for i, line in enumerate(tab):
        if i%2 == 1:
            continue
        l = [labels[line["param1"]], labels[line["param2"]]]
        for p in ["r", "p", "a", "b", "ang"]:
            col = "${:.2f}\pm{:.2f}$".format(line[p], line["{}err".format(p)])
            l.append(col)
        print(" & ".join(l) + "\\\\")

if __name__ == "__main__":
    calc_correlations()