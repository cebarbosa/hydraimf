# -*- coding: utf-8 -*-
"""

Created on 23/04/2020

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import seaborn as sns

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

def build_model(lick, lickerr, spindex):
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
    return model

def run_bsf(lick, lickerr, spindex, db, draws=500, redo=False):
    """Runs BSF on Lick indices. """
    summary = "{}.csv".format(db)
    if os.path.exists(summary) and not redo:
        return
    model = build_model(lick, lickerr, spindex)
    with model:
        trace = pm.sample(draws=draws, tune=draws)
        df = pm.stats.summary(trace)
        df.to_csv(summary)
    pm.save_trace(trace, db, overwrite=True)
    return

def load_traces(db, params):
    if not os.path.exists(db):
        return None
    ntraces = len(os.listdir(db))
    data = [np.load(os.path.join(db, _, "samples.npz")) for _ in
            os.listdir(db)]
    traces = []
    for param in params:
        v = np.vstack([data[num][param] for num in range(ntraces)]).flatten()
        traces.append(v)
    traces = np.column_stack(traces)
    return traces

def make_table(trace, params, db, redo=False):
    outtab = "{}_results.fits".format(db)
    if os.path.exists(outtab) and not redo:
        tab = Table.read(outtab)
        return tab
    v = np.percentile(trace, 50, axis=0)
    vmax = np.percentile(trace, 84, axis=0)
    vmin = np.percentile(trace, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = Table()
    tab["BIN"] = [db.split("_")[3]]
    for i, param in enumerate(params):
        tab[param] = [round(v[i], 5)]
        tab["{}_lerr".format(param)] = [round(vlerr[i], 5)]
        tab["{}_uerr".format(param)] = [round(vuerr[i], 5)]
    tab.write(outtab, overwrite=True)
    return tab

def plot_corner(trace, params, db, redo=False):
    output = "{}_corner.png".format(db)
    if os.path.exists(output) and not redo:
        return
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
    N = len(params)
    v = np.percentile(trace, 50, axis=0)
    vmax = np.percentile(trace, 84, axis=0)
    vmin = np.percentile(trace, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    text = ["Spectrum {}".format(db.split("_")[3])]
    for i, param in enumerate(params):
        s = "{0}$={1:.2f}^{{+{2:.2f}}}_{{-{3:.2f}}}$".format(
            labels[param], v[i], vuerr[i], vlerr[i])
        text.append(s)
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
        x = trace[:,i1]
        if p1 == p2:
            sns.kdeplot(x, shade=True, ax=ax, color="C0")
        else:
            y = trace[:, i2]
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
    plt.text(0.6, 0.7, "\n".join(text), transform=plt.gcf().transFigure,
            size=8)
    plt.subplots_adjust(left=0.12, right=0.995, top=0.98, bottom=0.08,
                        hspace=0.04, wspace=0.04)
    fig.align_ylabels()
    for fmt in ["pdf", "png"]:
        output = "{}_corner.{}".format(db, fmt)
        plt.savefig(output, dpi=300)
    plt.close(fig)
    return

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
    print("Loading models...")
    templates_file = os.path.join(context.home, "templates",
                                  "lick_vel{}_w{}_{}_{}_sig{}_{}.fits".format(
                                      velscale, w1, w2, sample, sigma,
                                      licktype))
    temptable = Table.read(templates_file)
    spindex = Spindex(temptable, indnames, parnames)
    ############################################################################
    print("Processing data...")
    outdir = os.path.join(home_dir, "bsf_lick")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    ts = []
    for fname, l, lerr in zip(filenames, lick, lickerr):
        print(fname)
        outdb = os.path.join(outdir, fname.split(".")[0])
        run_bsf(l, lerr, spindex, outdb)
        traces = load_traces(outdb, parnames)
        plot_corner(traces, parnames, outdb, redo=True)
        t = make_table(traces, parnames, outdb)
        ts.append(t)
    ts = vstack(ts)
    outtab = os.path.join(home_dir, "stpop.fits")
    ts.write(outtab, overwrite=True)