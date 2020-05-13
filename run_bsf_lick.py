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
from matplotlib import cm
import pymc3 as pm
import theano.tensor as tt
import seaborn as sns
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

def build_model(lick, lickerr, spindex, loglike="studt"):
    model = pm.Model()
    with model:
        theta = []
        for param in spindex.parnames:
            vmin = spindex.table[param].min()
            vmax = spindex.table[param].max()
            v = pm.Uniform(param, lower=vmin, upper=vmax)
            theta.append(v)
        if loglike == "studt":
            nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(nu)
        if loglike == "normal2":
            x = pm.Normal("x", mu=0, sd=1)
            s = pm.Deterministic("S", 1. + pm.math.exp(x))
            theta.append(s)
        theta = tt.as_tensor_variable(theta).T
        logl = bsf.LogLike(lick, spindex.indnames, lickerr, spindex,
                           loglike=loglike)
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})

    return model

def run_bsf(lick, lickerr, spindex, db, loglike="studt", draws=500, redo=False):
    """Runs BSF on Lick indices. """
    summary = "{}.csv".format(db)
    if os.path.exists(summary) and not redo:
        return
    model = build_model(lick, lickerr, spindex, loglike=loglike)
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

def make_table(trace, params, db, binnum, redo=False):
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
    tab["BIN"] = [binnum]
    for i, param in enumerate(params):
        tab[param] = [round(v[i], 5)]
        tab["{}_lerr".format(param)] = [round(vlerr[i], 5)]
        tab["{}_uerr".format(param)] = [round(vuerr[i], 5)]
    tab.write(outtab, overwrite=True)
    return tab

def plot_corner(trace, params, db, title=None, redo=False):
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
    if title is None:
        title = ""
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
    plt.text(0.6, 0.7, "\n".join(title), transform=plt.gcf().transFigure,
             size=8)
    plt.subplots_adjust(left=0.12, right=0.995, top=0.98, bottom=0.08,
                        hspace=0.04, wspace=0.04)
    fig.align_ylabels()
    for fmt in ["pdf", "png"]:
        output = "{}_corner.{}".format(db, fmt)
        plt.savefig(output, dpi=300)
    plt.close(fig)
    return

def plot_fitting(lick, lickerr, spindex, traces, outfig, binnum,
                 object, redo=False):
    """ Produces plot for the fitting. """
    if os.path.exists("{}.png".format(outfig)) and not redo:
        return
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    Ia = np.zeros((len(traces), spindex.nindices))
    from tqdm import tqdm
    for i in tqdm(range(len(traces)), desc="Loading indices for plots and "
                                           "table..."):
        Ia[i] = spindex(traces[i])
    x = np.median(Ia, axis=0)
    fig = plt.figure(figsize=(context.fig_width, 3.5))
    ax = plt.subplot(121)
    plt.tick_params(axis="y", which="minor", left=False, right=False)
    names = [_.replace("_", "").replace("muse", "*") for _ in
             spindex.indnames]
    ax.errorbar(lick, names, xerr=lickerr, fmt="o", mec="w", mew=0.4,
                elinewidth=0.8)
    for c, per in zip(colors, percs):
        ax.fill_betweenx(names, np.percentile(Ia, per, axis=0),
                         np.percentile(Ia, per + 10, axis=0), color=c)
    ax.set_xlabel(r"Equivalent width (\r{A})")
    ax.invert_yaxis()
    ax = plt.subplot(122)
    plt.tick_params(axis="y", which="minor", left=False, right=False)
    ax.errorbar(lick - x, names, xerr=lickerr, fmt="o", mec="w", mew=0.4,
                elinewidth=0.8)
    for c, per in zip(colors, percs):
        ax.fill_betweenx(names, np.percentile(Ia, per, axis=0) - x,
                         np.percentile(Ia, per + 10, axis=0) - x, color=c)
    ax.set_xlabel(r"Residue (\r{A})")
    ax.yaxis.set_ticklabels([])
    ax.axvline(x=0, ls="--", c="k", lw=0.8)
    ax.invert_yaxis()
    plt.subplots_adjust(left=0.12, right=0.98, wspace=0.02, top=0.953,
                        bottom=0.095)
    plt.suptitle("{} spectrum {}".format(object, binnum), x=0.57,
                 y=0.99, fontsize=9)
    for fmt in ["png", "pdf"]:
        plt.savefig("{}.{}".format(outfig, fmt), dpi=250, overwrite=True)
    plt.close()
    # Saving predictions in a table
    xup = np.percentile(Ia, 84, axis=0) - x
    xlow = x - np.percentile(Ia, 16, axis=0)
    t = Table()
    t["BIN"] = [binnum]
    for i, idx in enumerate(spindex.indnames):
        t[idx] = [x[i]]
        t["{}_lowerr".format(idx)] = [xlow[i]]
        t["{}_upper".format(idx)] = [xup[i]]
    t.write(outfig.replace("fit", "lick.fits"), format="fits", overwrite=True)
    return

def load_model(w1=4500, w2=10000, sigma=315, licktype="Ia", velscale=None,
               sample="all", indnames=None):
    velscale = int(context.velscale) if velscale is None else velscale
    if indnames is None:
        indnames = ['bTiO_muse', 'H_beta', 'Fe5015', 'Mg_1', 'Mg_2', 'Mg_b',
                    'Fe5270', 'Fe5335', 'Fe5406', 'Fe5709', 'Fe5782', 'aTiO',
                    'Na_D', 'TiO_1', 'TiO_2_muse', 'CaH_1',
                    'CaH_2_muse', 'TiO_3', 'TiO_4', 'NaI', 'CaT1', 'CaT2',
                    'CaT3']
    print("Loading models...")
    templates_file = os.path.join(context.home, "templates",
                                  "lick_vel{}_w{}_{}_{}_sig{}_{}.fits".format(
                                      velscale, w1, w2, sample, sigma,
                                      licktype))
    temptable = Table.read(templates_file)
    parnames = ["imf", "Z", "T", "alphaFe", "NaFe"]
    return Spindex(temptable, indnames, parnames)

def run_ngc3311(targetSN=250, sigma=315, dataset="MUSE", licktype="Ia",
                loglike="normal2", useerr=True):
    spindex = load_model(sigma=sigma)
    # Loading observed data
    home_dir =  os.path.join(context.get_data_dir(dataset),
                             "fieldA/sn{}".format(targetSN))
    wdir = os.path.join(home_dir, "lick")
    noerr = "" if useerr else "_noerr"
    outdir = os.path.join(home_dir, "bsf_lick_{}{}".format(loglike, noerr))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filenames = sorted([_ for _ in os.listdir(wdir) if
                        _.endswith("sigma{}.fits".format(sigma))])
    fields = ["name", licktype, "{}err".format(licktype)]
    tables = [Table.read(os.path.join(wdir, fname))[fields] for
            fname in filenames]
    lick = [np.array(t[t.colnames[1]].data) for t in tables]
    if useerr:
        lickerr = [np.array(t[t.colnames[2]].data) for t in tables]
    else:
        lickerr = [np.ones_like(a) for a in lick]
    # Setting indices to be used
    t0 = tables[0]
    idx = [i for i,_ in enumerate(t0) if _["name"] in spindex.indnames]
    lick = [l[idx] for l in lick]
    lickerr = [l[idx] for l in lickerr]
    ts = []
    for fname, l, lerr in zip(filenames, lick, lickerr):
        print(fname)
        binnum = fname.split("_")[2]
        outdb = os.path.join(outdir, fname.split(".")[0])
        run_bsf(l, lerr, spindex, outdb, loglike=loglike)
        traces = load_traces(outdb, spindex.parnames)
        title = "Spectrum {}".format(binnum)
        plot_corner(traces, spindex.parnames, outdb, title=title, redo=False)
        outfig = "{}_fit".format(outdb)
        plot_fitting(l, lerr, spindex, traces, outfig, binnum,
                     "NGC 3311", redo=False)
        t = make_table(traces, spindex.parnames, outdb, binnum, redo=True)
        ts.append(t)
    ts = vstack(ts)
    outtab = os.path.join(home_dir, "stpop_{}{}.fits".format(loglike, noerr))
    ts.write(outtab, overwrite=True)

def run_m87(targetSN=500, sigma=410, licktype="Ia", loglike="normal"):
    spindex = load_model(sigma=sigma)
    # Loading observed data
    imgname, cubename = context.get_img_cube_m87()
    home_dir =  os.path.join(os.path.split(cubename)[0], "sn{}".format(targetSN))
    wdir = os.path.join(home_dir, "lick")
    outdir = os.path.join(home_dir, "bsf_lick_{}".format(loglike))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filenames = sorted([_ for _ in os.listdir(wdir) if
                        _.endswith("sigma{}.fits".format(sigma))])
    filenames = filenames[28:][::-1]
    fields = ["name", licktype, "{}err".format(licktype)]
    tables = [Table.read(os.path.join(wdir, fname))[fields] for
            fname in filenames]
    lick = [np.array(t[t.colnames[1]].data) for t in tables]
    lickerr = [np.array(t[t.colnames[2]].data) for t in tables]
    # Setting indices to be used
    t0 = tables[0]
    idx = [i for i,_ in enumerate(t0) if _["name"] in spindex.indnames]
    lick = [l[idx] for l in lick]
    lickerr = [l[idx] for l in lickerr]
    ts = []
    for fname, l, lerr in zip(filenames, lick, lickerr):
        print(fname)
        outdb = os.path.join(outdir, fname.split(".")[0])
        run_bsf(l, lerr, spindex, outdb, loglike=loglike)
        traces = load_traces(outdb, spindex.parnames)
        # title = "Spectrum {}".format(binnum)
        plot_corner(traces, spindex.parnames, outdb, redo=False)
        outfig = "{}_fit".format(outdb)
        plot_fitting(l, lerr, spindex, traces, outfig, redo=False)
        t = make_table(traces, spindex.parnames, outdb, binnum)
        ts.append(t)
    ts = vstack(ts)
    outtab = os.path.join(home_dir, "stpop.fits")
    ts.write(outtab, overwrite=True)

if __name__ == "__main__":
    run_ngc3311(useerr=False)
    # run_m87()