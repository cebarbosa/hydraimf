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
import matplotlib.gridspec as gridspec

import context

def plot_profiles(t, output, xfield, yfields, redo=False):
    if os.path.exists(output) and not redo:
        return
    global labels
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
    global labels
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

def plot_sigma_imf(t, figsize=(5,3)):
    # Producing plot similar to Spiniello+ 2014
    global labels
    output = os.path.join(outdir, "sigma_imf")
    xfield = "sigma"
    yfield = "imf"
    label = "NGC 3311"
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
    plabels = ["Ferreras et al. (2013)", "La Barbera et al. (2013)"]
    colors = ["C2", "C3"]
    for i in range(2):
        y = a[i] * np.log10(sigma / 200) + b[i]
        ax.plot(sigma, y, "-", c=colors[i], label=plabels[i])
    plt.legend(loc=4, frameon=False)
    ax.set_xlim(140, 350)
    ax.axhline(y=1.35, c="k", ls="--", lw=0.8)
    ax.axhline(y=1.8, c="k", ls="--", lw=0.8)
    ax.axhline(y=2.35, c="k", ls="--", lw=0.8)
    ax.text(325, 1.375, "Kroupa")
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

def plot_sarzi(t, figsize=(7.24, 2.5)):
    global labels
    output = os.path.join(outdir, "imf_Z-alphafe-sigma")
    R = t["R"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(R),
                                       clip=True)
    cmap = plt.get_cmap('Blues_r')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    mapper = cm.ScalarMappable(norm=norm, cmap=new_cmap)
    colors = np.array([(mapper.to_rgba(v)) for v in R])
    yfield = "imf"
    ys = t[yfield]
    yerrs =  np.array([t["{}_lerr".format(yfield)], t["{}_uerr".format(
        yfield)]]).T
    xfields = ["Z", "alphaFe", "sigma"]
    fig = plt.figure(figsize=figsize)
    widths = [1, 1, 1, 0.08]
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=widths)
    gs.update(left=0.06, right=0.955, bottom=0.13, top=0.98, wspace=0.02,
              hspace=0.00)
    xlims = [[-0.35, 0.42], [-0.02, 0.38], [170, 360]]
    for i, xfield in enumerate(xfields):
        xs = t[xfield]
        xerrs = np.array([t["{}_lerr".format(xfield)], t["{}_uerr".format(
            xfield)]]).T
        ax = plt.subplot(gs[i])
        for x, y, xerr, yerr, c in zip(xs, ys, xerrs, yerrs, colors):
            ax.errorbar(x, y, yerr=np.atleast_2d(yerr).T,
                        xerr=np.atleast_2d(xerr).T, fmt="o",
                        ecolor="0.8", mec="w", color=c,
                        mew=0.5, elinewidth=0.5)
        ax.set_xlabel(labels[xfield])
        if i == 0:
            ax.set_ylabel(labels[yfield])
        else:
            ax.yaxis.set_ticklabels([])
        # IMF lines
        ax.axhline(y=1.3, c="k", ls="--", lw=0.8)
        # ax.axhline(y=1.8, c="k", ls="--", lw=0.8)
        # ax.axhline(y=2.35, c="k", ls="--", lw=0.8)
        if i == 0:
            ax.text(-0.15, 1.32, "Kroupa", size=6)
            # ax.text(-0.32, 1.82, "Chabrier", size=5)
            # ax.text(-0.32, 2.37, "Salpeter", size=5)
        ax.set_xlim(xlims[i])
        ax.set_ylim(0.45, 3.7)
        # Specific details for each plot
        if xfield == "sigma":
            sigma = np.linspace(180, 320, 100)
            # Plot other authors
            a = [4.87, 3.4]
            b = [2.33, 2.3]
            plabels = ["Ferreras et al. (2013)", "La Barbera et al. (2013)"]
            colors = ["C2", "C3"]
            for i in [0]:
                y = a[i] * np.log10(sigma / 200) + b[i]
                ax.plot(sigma, y, "-", c=colors[i], label=plabels[i])
            # Spiniello 2014
            # a = np.random.normal(2.3, 0.1, len(sigma))
            # b = np.random.normal(2.1-1, 0.2, len(sigma))
            # y = a * np.log10(sigma / 200)[:, np.newaxis] + b
            # ax.plot(sigma, y.mean(axis=1), "-", c="C1",
            #         label="Spiniello et al. (2014)")
            # ax.plot(sigma, np.percentile(y, 16, axis=1), "--", c="C1")
            # ax.plot(sigma, np.percentile(y, 84, axis=1), "--", c="C1")
            # La Barbera 2013
            b = np.random.normal(2.4, 0.1, len(sigma))
            a = np.random.normal(5.4, 0.9, len(sigma))
            y = a * np.log10(sigma / 200.)[:, np.newaxis] + b
            ax.plot(sigma, y.mean(axis=1), "-", c="C1",
                    label="La Barbera et al. (2013)")
            ax.plot(sigma, np.percentile(y, 16, axis=1), "--", c="C1")
            ax.plot(sigma, np.percentile(y, 84, axis=1), "--", c="C1")
            plt.legend(loc=4, frameon=False, prop={'size': 6})
        if xfield == "Z":
            z = np.linspace(-0.4, 0.45, 50)
            # Martin-Navarro 2015
            a = np.random.normal(3.1, 0.5, len(z))
            b = np.random.normal(2.2, 0.1, len(z))
            y = a * z[:, np.newaxis] + b
            ax.plot(z, y.mean(axis=1), "-", c="C4",
                    label="Martín-Navarro et al.(2015)")
            ax.plot(z, np.percentile(y, 16, axis=1), "--", c="C4")
            ax.plot(z, np.percentile(y, 84, axis=1), "--", c="C4")
            plt.legend(loc=3, frameon=False, prop={'size': 6})
        if xfield == "alphaFe":
            ax.plot([0.29, 0.42], [2, 2.9], "-", c="C5",
                    label="Sarzi et al. (2018)")
            plt.legend(loc=4, frameon=False, prop={'size': 6})
    cax = fig.add_subplot(gs[3])
    cbar = fig.colorbar(mapper, cax=cax, orientation="vertical")
    cbar.set_label("R (kpc)")
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(output, fmt), dpi=300)
    plt.close()

if __name__ == "__main__":
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma_*$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]"}
    dataset = "MUSE"
    targetSN = 250
    wdir =  os.path.join(context.data_dir, dataset, "voronoi",
                         "sn{}".format(targetSN))
    outdir = os.path.join(wdir, "plots")
    ############################################################################
    # Loading and preparing data
    tfile = os.path.join(wdir, "results.fits")
    t = Table.read(tfile)[:-1]
    # t["sigma_lerr"] = t["sigmaerr"]
    # t["sigma_uerr"] = t["sigmaerr"]
    # t["V_lerr"] = t["Verr"]
    # t["V_uerr"] = t["Verr"]
    t["R_uerr"] = 0
    t["R_lerr"] = 0
    t["sigma"] = np.where(t["sigma"] < 500, t["sigma"], np.nan)
    ############################################################################
    output = os.path.join(outdir, "radial_profiles")
    plot_profiles(t, output, "R",
                  ["T", "Z", "imf", "alphaFe", "NaFe", "sigma"], redo=True)
    ############################################################################
    output = os.path.join(outdir, "sigma_profiles")
    plot_profiles(t, output, "sigma", ["imf", "Z", "T", "alphaFe", "NaFe"],
                  redo=True)
    ############################################################################
    output = os.path.join(outdir, "metal_imf")
    plot_single(t, output, "Z", "imf")
    plt.close()
    plot_sigma_imf(t)
    plt.close()
    plot_sarzi(t)
