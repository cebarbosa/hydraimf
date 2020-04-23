# -*- coding: utf-8 -*-
"""

Created on 23/04/2020

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import platform

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator
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

class Spindex():
    """ Linearly interpolated line-strength indices."""
    def __init__(self, temptable, indnames, parnames):
        self.table = temptable
        self.indnames = indnames
        self.parnames = parnames
        self.nindices = len(indnames)
        self.nparams = len(parnames)
        # Interpolating models
        pdata = np.array([temptable[col].data for col in parnames]).T
        tdata =  np.array([temptable[col].data for col in indnames]).T
        nodes = []
        for param in parnames:
            x = np.unique(temptable[param]).data
            nodes.append(x)
        coords = np.meshgrid(*nodes, indexing='ij')
        dim = coords[0].shape + (self.nindices,)
        data = np.zeros(dim)
        with np.nditer(coords[0], flags=['multi_index']) as it:
            while not it.finished:
                multi_idx = it.multi_index
                x = np.array([coords[i][multi_idx] for i in range(len(coords))])
                idx = (pdata == x).all(axis=1).nonzero()[0]
                data[multi_idx] = tdata[idx]
                it.iternext()
        self.f = RegularGridInterpolator(nodes, data, fill_value=0)
        ########################################################################
        # Get grid points to handle derivatives
        inner_grid = []
        thetamin = []
        thetamax = []
        for par in self.parnames:
            thetamin.append(np.min(self.table[par].data))
            thetamax.append(np.max(self.table[par].data))
            inner_grid.append(np.unique(self.table[par].data)[1:-1])
        self.thetamin = np.array(thetamin)
        self.thetamax = np.array(thetamax)
        self.inner_grid = inner_grid

    def __call__(self, theta):
        return self.f(theta)[0]

    def gradient(self, theta, eps=1e-6):
        # Clipping theta to avoid border problems
        theta = np.maximum(theta, self.thetamin + 2 * eps)
        theta = np.minimum(theta, self.thetamax - 2 * eps)
        grads = np.zeros((self.nparams, self.nindices))
        for i,t in enumerate(theta):
            epsilon = np.zeros(self.nparams)
            epsilon[i] = eps
            # Check if data point is in inner grid
            in_grid = t in self.inner_grid[i]
            if in_grid:
                tp1 = theta + 2 * epsilon
                tm1 = theta + epsilon
                grad1 = (self.__call__(tp1) - self.__call__(tm1)) / (2 * eps)
                tp2 = theta - epsilon
                tm2 = theta - 2 * epsilon
                grad2 = (self.__call__(tp2) - self.__call__(tm2)) / (2 * eps)
                grads[i] = 0.5 * (grad1 + grad2)
            else:
                tp = theta + epsilon
                tm = theta - epsilon
                grads[i] = (self.__call__(tp) - self.__call__(tm)) / (2 * eps)
        return grads

def run_bsf(lick, lickerr, spindex, db, draws=500, redo=False):
    """Runs BSF on Lick indices. """
    summary = "{}.csv".format(db)
    if os.path.exists(summary) and not redo:
        return
    model = pm.Model()
    with model:
        theta = []
        for param in spindex.parnames:
            vmin = spindex.table[param].min()
            vmax = spindex.table[param].max()
            v = pm.Uniform(param, lower=vmin, upper=vmax)
            theta.append(v)
        nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
        theta.append(nu)
        theta = tt.as_tensor_variable(theta).T
        logl = bsf.LogLike(lick, spindex.indnames, lickerr, spindex,
                           loglike="studt")
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})
        trace = pm.sample(draws=draws, tune=draws)
        df = pm.stats.summary(trace)
        df.to_csv(summary)
    pm.save_trace(trace, db, overwrite=True)

if __name__ == "__main__":
    targetSN = 250
    w1 = 4500
    w2 = 10000
    sigma= 315
    dataset = "MUSE"
    licktype = "Ia"
    velscale = int(context.velscale)
    sample = "all"
    # Loading observed data
    home_dir =  os.path.join(context.get_data_dir(dataset),
                             "fieldA/sn{}".format(targetSN))
    wdir = os.path.join(home_dir, "lick")
    filenames = sorted([_ for _ in os.listdir(wdir) if
                        _.endswith("sigma{}.fits".format(sigma))])
    fields = ["name", licktype, "{}err".format(licktype)]
    tables = [Table.read(os.path.join(wdir, fname))[fields] for
            fname in filenames]
    lick = [np.array(t[t.colnames[1]].data) for t in tables]
    lickerr = [np.array(t[t.colnames[2]].data) for t in tables]
    # Setting indices to be used
    t0 = tables[0]
    idx = [i for i,_ in enumerate(t0[t0.colnames[1]]) if np.isfinite(_)]
    indnames = [t0[t0.colnames[0]][i] for i in idx]
    parnames = ["imf", "Z", "T", "alphaFe", "NaFe"]
    lick = [l[idx] for l in lick]
    lickerr = [l[idx] for l in lickerr]
    ############################################################################
    templates_file = os.path.join(context.home, "templates",
                                  "lick_vel{}_w{}_{}_{}_sig{}_{}.fits".format(
                                      velscale, w1, w2, sample, sigma,
                                      licktype))
    temptable = Table.read(templates_file)
    spindex = Spindex(temptable, indnames, parnames)
    ############################################################################
    outdir = os.path.join(home_dir, "bsf_lick")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for fname, l, lerr in zip(filenames, lick, lickerr):
        print(fname)
        outdb = os.path.join(outdir, fname.split(".")[0])
        run_bsf(l, lerr, spindex, outdb)