# -*- coding: utf-8 -*-
"""

Created on 21/07/2020

Author : Carlos Eduardo Barbosa

Calculating gradients in the stellar populations using a piecewise linear
function.

"""
import os
import string

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm

import context

def plot_radial_profiles(table, ypars, xpar="R", xlim=(0.07, 20), xscale="log"):
    global labels
    b16= Table.read(os.path.join(context.tables_dir, "barbosa16.fits"))
    b16 = b16[b16["S_N"] >= 50]
    b16["logT"] = np.log10(b16["Age"])
    b16["e_logT"] = np.abs(b16["e_Age"] / b16["Age"] / np.log(10))
    b16_fields = {"Z": "__Z_H_", "T": "T", "alphaFe": "__alpha_Fe_"}
    b16["T"] = np.power(10, b16["logT"])
    b16["e_T"] = np.abs(b16["T"] * np.log(10) * b16["e_logT"])

    x = table[xpar]
    fig = plt.figure(figsize=(context.fig_width, 8))
    # Plot weights
    t = Table.read(os.path.join(context.tables_dir,
                                "barbosa2018_weights.fits"))
    r = t["R"].data
    f = np.array([t[p].data for p in ["w_A", "w_B", "w_galhalo", "w_cd"]]).T
    rs = []
    for i in range(4):
        rismax = np.where(f[:,i] == np.max(f, axis=1), r, np.nan)
        rs.append(np.nanmin(rismax))
        rs.append(np.nanmax(rismax))
    rs = np.sort(rs)
    rtran = np.array([[xlim[0], rs[2]], [rs[3], rs[4]], [rs[5], rs[6]]])
    # fcs = ["skyblue", "g", "khaki", "orange", "coral"]
    fcs = ["red", "khaki", "forestgreen", "cornflowerblue"]
    names = ["A", "B", "C", "D"]
    ax = plt.subplot(len(ypars) + 1, 1, len(ypars)+1)
    ax.set_xscale(xscale)
    transition_colors = ["orange", "green", "slateblue"]
    for i, (r1, r2) in enumerate(rtran):
        ax.axvspan(r1, r2, facecolor=transition_colors[i], alpha=0.15)
    legend_elements = []
    for i in range(4):
        ax.plot(r, f[:,i], ".", ms=0.5, c=fcs[i], rasterized=True)
        legend_elements.append(Patch(facecolor=fcs[i], edgecolor=fcs[i],
                                                          label=names[i]))
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1.0)
    # ax.axvline(x=1.08 * 0.262, color="k", ls="--", lw=0.5, dashes=(10, 10))
    fwhm = 1.08 * 0.262
    hw = [0.01, 0.02]
    for n in range(2):
        ax.arrow((n+1) * fwhm / 2, 0.75, 0, 0.2, head_width=hw[n],
                 head_length=0.05, fc='k',
                 ec='k')
        ax.text((n+1) * fwhm / 2 + hw[n], 0.77, "{:.2f}''".format((n+1) *
                                                                  1.08 / 2),
                fontsize=6)
    plt.legend(handles=legend_elements, ncol=4, frameon=True, loc=1,
               prop={"size": 5})
    ax.set_ylabel("light fraction", labelpad=-1)
    ax.set_xlabel(labels["R"])
    ax.text(0.05, 0.8, "({})".format(string.ascii_lowercase[len(ypars)]),
            transform=ax.transAxes, fontsize=10)
    # Plot SSP parameters
    for i, p in enumerate(ypars):
        ax = plt.subplot(len(ypars) + 1, 1, i + 1)
        for j, (r1, r2) in enumerate(rtran):
            ax.axvspan(r1, r2, facecolor=transition_colors[j], alpha=0.15)
        y = table[p].data
        ylower = table["{}_lerr".format(p)].data
        yupper = table["{}_uerr".format(p)].data

        ax.set_xscale(xscale)
        label = "This work" if i ==0 else None
        ax.errorbar(x, y, yerr=[ylower, yupper], ecolor=None, fmt="o",
                    mew=0.5, elinewidth=0.5, mec="w", ms=5, c="tab:blue",
                    label=label)
        ax.set_ylabel(labels[p], labelpad=-1)
        ax.set_xlim(xlim)
        ax.xaxis.set_ticklabels([])
        if p in b16_fields:
            label = "Barbosa et al. (2016)" if i==0 else None

            ax.errorbar(b16["Rad"], b16[b16_fields[p]],
                        yerr=b16["e_{}".format(b16_fields[p]).replace("___",
                                                                      "__")],
                        fmt="x", mew=0.5, label=label,
                        elinewidth=0.5, ms=4.5, c="k")
        if i == 0:
            plt.legend(frameon=True, prop={"size": 6})
        ax.text(0.05, 0.8,
                "({})".format(string.ascii_lowercase[i]),
                transform=ax.transAxes, fontsize=10)
        if i == 0:
            ax.set_ylim(3, None)
    fig.align_ylabels()
    plt.subplots_adjust(left=0.13, right=0.99, top=0.995, bottom=0.037,
                        hspace=0.06)
    out = os.path.join(wdir, "plots/radial_profiles")
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(out, fmt), dpi=350)
    plt.close()
    return

if __name__ == "__main__":
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma_*$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "logT": "$\log $ Age (Gyr)",
              "M2L": "$\\Upsilon_*^r$ ($M_\odot/L_\odot$)",
              "logSigma": "$\\log \\Sigma$ (M$_\\odot$/kpc$^2$)"}
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    table = Table.read(os.path.join(wdir, "results.fits"))
    params = ["T", "Z", "alphaFe", "NaFe", "imf", "sigma", "M2L", "logSigma"]
    plot_radial_profiles(table, params)