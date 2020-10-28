"""
Plot results from the Painbox simulations.
"""

import os

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from scipy.stats import gaussian_kde

import context

if __name__ == "__main__":
    data_dir = os.path.join(context.data_dir, "pbsim")
    sspmodel = "emiles"
    loglike = "studt"
    nssps_sim = 1
    nssps_fit = 1
    sample, nsim, sn = "all", 1000, 50
    sampler = "emcee"
    simname = "{}_{}_nsim{}_{}_sn{}".format(sspmodel, sample, nsim, loglike, sn)
    wdir = os.path.join(data_dir, simname)
    plots_dir = os.path.join(wdir, "plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    simulations = Table.read(os.path.join(wdir, "simulations.fits"))
    mcmc_dir = os.path.join(wdir, sampler)
    tables = []
    sims = []
    for i, sim in enumerate(simulations):
        db = os.path.join(mcmc_dir, "{:04d}_chain.fits".format(i))
        if not os.path.exists(db):
            continue
        tables.append(Table.read(db))
        sims.append(sim)
    params = ["Z", "T", "alphaFe", "NaFe", 'imf', "Av", "V", "sigma"]
    fig = plt.figure(1, figsize=(7,4))
    gs = GridSpec(4, 2)
    for j, p in enumerate(tqdm(params)):
        ax = plt.subplot(gs[j])
        x, y, lerr, uerr = [], [], [], []
        x = np.array([sim[p] for sim in sims])
        m = np.array([np.median(t[p]) for t in tables])
        uerr = np.array([np.percentile(t[p], 84) for t in tables]) - m
        lerr = m - np.array([np.percentile(t[p], 16) for t in tables])
        y = x - m
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        # ax.errorbar(x, y, yerr=[lerr, uerr], marker=".", color="none",
        #             ecolor="0.8")
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, s=10, marker="x", cmap="jet", zorder=1500,
                   linewidth=0.5)
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # x1 = np.min([xlim[0], ylim[0]])
        # x2 = np.max([ylim[1], xlim[1]])
        # ax.plot([x1, x2], [x1, x2], "--k")
        ax.axhline(y=0, ls="--", c="k", zorder=2000)
        ax.set_xlabel(context.labels[p])
        ax.set_ylabel("$\Delta${}".format(context.labels[p]))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "{}_{}.png".format(
                simname, sampler)), dpi=250)
    # plt.show()
    plt.close()