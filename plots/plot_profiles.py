# -*- coding: utf-8 -*-
"""

Created on 24/04/2020

Author : Carlos Eduardo Barbosa

"""
import os

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

import context

def plot_profiles(t, output, xfield, yfields):
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
    fig = plt.figure(figsize=(context.fig_width, 6))
    for i, field in enumerate(yfields):
        yerr = [t["{}_lerr".format(field)], t["{}_uerr".format(field)]]
        xerr = [t["{}_lerr".format(xfield)], t["{}_uerr".format(xfield)]]
        ax = plt.subplot(len(yfields), 1, i+1)
        # ax.set_xscale("log")
        ax.errorbar(t[xfield], t[field],
                     yerr=yerr, xerr=xerr, fmt="o", ecolor="C0", mec="w",
                    mew=0.5, elinewidth=0.5)
        plt.ylabel(labels[field])
        if i+1 < len(yfields):
            ax.xaxis.set_ticklabels([])
    plt.xlabel(labels[xfield])
    plt.subplots_adjust(left=0.14, right=0.985, top=0.995, bottom=0.052,
                        hspace=0.06)
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(output, fmt), dpi=250)
    plt.close()

def plot_single(t, output, xfield, yfield, return_ax=False, label=None,
                figsize=None):
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
    figsize = (context.fig_width, 2.8) if figsize is None else figsize
    fig = plt.figure(figsize=figsize)
    yerr = [t["{}_lerr".format(yfield)], t["{}_uerr".format(yfield)]]
    xerr = [t["{}_lerr".format(xfield)], t["{}_uerr".format(xfield)]]
    ax = plt.subplot(1, 1, 1)
    # ax.set_xscale("log")
    ax.errorbar(t[xfield], t[yfield],
                 yerr=yerr, xerr=xerr, fmt="o", ecolor="C0", mec="w",
                mew=0.5, elinewidth=0.5, label=label)
    plt.ylabel(labels[yfield])
    plt.xlabel(labels[xfield])
    plt.subplots_adjust(left=0.11, right=0.985, top=0.98, bottom=0.12,
                        hspace=0.06)
    if return_ax:
        return ax
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(output, fmt), dpi=250)
    plt.close()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval,
                                            b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_sigma_imf(t):
    # Producing plot similar to Spiniello+ 2014
    output = os.path.join(outdir, "sigma_imf")
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
    xfield = "sigma"
    yfield = "imf"
    label = "NGC 3311"
    figsize=(5,3)
    fig = plt.figure(figsize=figsize)
    xs = t[xfield]
    ys = t[yfield]
    xerrs = np.array([t["{}_lerr".format(xfield)], t["{}_uerr".format(
        xfield)]]).T
    yerrs = np.array([t["{}_lerr".format(yfield)], t["{}_uerr".format(
        yfield)]]).T
    R = t["R"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(R),
                                       clip=True)
    cmap = plt.get_cmap('Blues_r')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    mapper = cm.ScalarMappable(norm=norm, cmap=new_cmap)
    colors = np.array([(mapper.to_rgba(v)) for v in R])
    ax = plt.subplot(1, 1, 1)
    for x, y, xerr, yerr, c in zip(xs, ys, xerrs, yerrs, colors):
        ax.errorbar(x, y, yerr=np.atleast_2d(yerr).T,
                    xerr=np.atleast_2d(xerr).T, fmt="o",
                    ecolor=c, mec="w", color=c,
                    mew=0.5, elinewidth=0.5)
    plt.ylabel(labels[yfield])
    plt.xlabel(labels[xfield])
    a = np.random.normal(2.3, 0.1, 1000)
    b = np.random.normal(2.1, 0.2, 1000)
    sigma = np.linspace(150, 310, 100)
    y = a * np.log10(sigma / 200)[:, np.newaxis] + b
    ax.plot(sigma, y.mean(axis=1), "-", c="C1", label="Spiniello et al. (2014)")
    ax.plot(sigma, np.percentile(y, 16, axis=1), "--", c="C1")
    ax.plot(sigma, np.percentile(y, 84, axis=1), "--", c="C1")
    # Plot other authors
    a = [4.87, 3.4]
    b = [2.33, 2.3]
    labels = ["Ferreras et al. (2013)", "La Barbera et al. (2013)"]
    colors = ["C2", "C3"]
    for i in range(2):
        y = a[i] * np.log10(sigma / 200) + b[i]
        ax.plot(sigma, y, "-", c=colors[i], label=labels[i])
    plt.legend(loc=4, frameon=False)
    ax.set_xlim(140, 350)
    ax.axhline(y=1.8, c="k", ls="--", lw=0.8)
    ax.axhline(y=2.35, c="k", ls="--", lw=0.8)
    ax.text(325, 1.825, "Chabrier")
    ax.text(325, 2.375, "Salpeter")
    plt.subplots_adjust(left=0.08, right=0.98, top=0.99, bottom=0.105,
                        hspace=0.06)
    cbar_pos = [0.14, 0.18, 0.18, 0.05]
    cbaxes = fig.add_axes(cbar_pos)
    cbar = plt.colorbar(mapper, cax=cbaxes, orientation="horizontal")
    cbar.set_ticks([0, 2, 4, 6, 8])
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label("R (kpc)")
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(output, fmt), dpi=250)
    plt.close()

if __name__ == "__main__":
    dataset = "MUSE"
    targetSN = 250
    wdir =  os.path.join(context.get_data_dir(dataset),
                         "fieldA/sn{}".format(targetSN))
    outdir = os.path.join(wdir, "plots")
    tfile = os.path.join(wdir, "results.fits")
    t = Table.read(tfile)
    t["sigma_lerr"] = t["sigmaerr"]
    t["sigma_uerr"] = t["sigmaerr"]
    t["V_lerr"] = t["Verr"]
    t["V_uerr"] = t["Verr"]
    t["R_uerr"] = 0
    t["R_lerr"] = 0
    output = os.path.join(outdir, "radial_profiles")
    plot_profiles(t, output, "R",
                  ["sigma", "imf", "Z", "T", "alphaFe", "NaFe"])
    output = os.path.join(outdir, "sigma_profiles")
    # plot_profiles(t, output, "sigma", ["imf", "Z", "T", "alphaFe", "NaFe"])
    plot_sigma_imf(t)
    # plot_single(t, output, "Z", "imf")