"""
Plot results from the Painbox simulations.
"""

import os

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

import context

if __name__ == "__main__":
    data_dir = os.path.join(context.data_dir, "pbsim")
    sspmodel = "emiles"
    sample, nsim, sn = "all", 1000, 50
    simname = "{}_{}_nsim{}_sn{}".format(sspmodel, sample, nsim, sn)
    wdir = os.path.join(data_dir, simname)
    plots_dir = os.path.join(wdir, "plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    simulations = Table.read(os.path.join(wdir, "simulations.fits"))
    zeus_dir = os.path.join(wdir, "zeus")
    params = ["Z", "T", "alphaFe", "NaFe", 'imf', "Av", "V", "sigma"]
    fig = plt.figure(1, figsize=(7,4))
    gs = GridSpec(4, 2)
    for i, sim in tqdm(enumerate(simulations)):
        db = os.path.join(zeus_dir, "{:04d}_chain.fits".format(i))
        if not os.path.exists(db):
            continue
        fit = Table.read(db)
        for j, p in enumerate(params):
            simval = sim[p]
            m = np.median(fit[p])
            uerr = np.percentile(fit[p], 84) - m
            lerr = m - np.percentile(fit[p], 16)
            std = np.std(fit[p])
            ax = plt.subplot(gs[j])
            ax.errorbar(simval, m - simval, yerr=[[lerr], [uerr]], marker="o",
                        color="tab:blue", ecolor="0.8")
    for j, p in enumerate(params):
        ax = plt.subplot(gs[j])
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # x1 = np.min([xlim[0], ylim[0]])
        # x2 = np.max([ylim[1], xlim[1]])
        # ax.plot([x1, x2], [x1, x2], "--k")
        ax.axhline(y=0, ls="--", c="k")
        ax.set_xlabel(context.labels[p])
        ax.set_ylabel("$\Delta${}".format(context.labels[p]))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "sim_results_sn{}.png".format(sn)),
                dpi=250)
    plt.show()