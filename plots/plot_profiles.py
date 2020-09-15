# -*- coding: utf-8 -*-
"""

Created on 24/04/2020

Author : Carlos Eduardo Barbosa

"""
import os
import itertools

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Ellipse

import context

def plot_profiles(t, xfield, yfields, output=None, xfracs=None, yfracs=None,
                  xlim=None, return_axis=False):
    global labels
    corr = Table.read(os.path.join(wdir, "fit_stats.fits"))
    fig = plt.figure(figsize=(context.fig_width, 2. * len(yfields)))
    xfracs = [0.2] * len(yfields) if xfracs is None else xfracs
    yfracs = [0.2] * len(yfields) if yfracs is None else yfracs
    xlim = [None, None] if xlim is None else xlim
    gs = gridspec.GridSpec(len(yfields), 1, figure=fig)
    gs.update(left=0.12, right=0.99, bottom=0.055, top=0.99, wspace=0.02,
              hspace=0.02)
    for i, yfield in enumerate(yfields):
        print(yfield)
        yerr = [t["{}_lerr".format(yfield)], t["{}_uerr".format(yfield)]]
        xerr = [t["{}_lerr".format(xfield)], t["{}_uerr".format(xfield)]]
        ax = plt.subplot(gs[i])
        # ax.set_xscale("log")
        ax.errorbar(t[xfield], t[yfield],
                     yerr=yerr, xerr=xerr, fmt="o", ecolor="0.8", mec="w",
                    mew=0.5, elinewidth=0.5)
        ax.set_xlim(xlim)
        plt.ylabel(labels[yfield])
        if i+1 < len(yfields):
            ax.xaxis.set_ticklabels([])
        # plot parameter correlations
        idx = np.where((corr["param1"]==xfield) & (corr["param2"]==yfield))[0]
        if idx:
            a = float(corr["a"][idx])
            b = float(corr["b"][idx])
            ang = float(corr["ang"][idx])
            print(xfield, yfield, a, b, ang)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xel = xmin + xfracs[i] * (xmax - xmin)
            yel = ymin + yfracs[i] * (ymax - ymin)
            ellipse = Ellipse((xel, yel), a, b, ang,
                              facecolor="none", edgecolor="r", linestyle="--")
            ax.text(xel - 0.03 * (xmax - xmin),
                    yel - 0.02 * (ymax - ymin), "$1\sigma$", size=5.5,
                    c="r")
            ax.add_patch(ellipse)
    if return_axis:
        return gs
    plt.xlabel(labels[xfield])
    if output is not None:
        for fmt in ["pdf", "png"]:
            plt.savefig("{}.{}".format(output, fmt), dpi=250)
    plt.close()

def plot_single(t, xfield, yfield, return_ax=True, label=None,
                figsize=None, output=None):
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
    if output is not None:
        for fmt in ["pdf", "png"]:
            plt.savefig("{}.{}".format(output, fmt), dpi=250)
    plt.close()
    return

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

def get_colors(R):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.ceil(np.max(R)),
                                       clip=True)
    cmap = plt.get_cmap('Blues_r')
    new_cmap = truncate_colormap(cmap, 0.0, 0.8)
    mapper = cm.ScalarMappable(norm=norm, cmap=new_cmap)
    return mapper, np.array([(mapper.to_rgba(v)) for v in R])

def plot_imf_relations(t, figsize=(7.24, 4.5)):
    global labels, wdir
    corr = Table.read(os.path.join(wdir, "fit_stats.fits"))

    mapper, colors = get_colors(t["R"])
    yfields= ["imf", "alpha"]
    xlim = {"T": [None, None], "Z": [-0.2, 0.25], "alphaFe": [0, 0.45],
            "NaFe": [None, 0.7], "sigma": [80, 380], "Re": [-0.1, 1.1]}
    xelf1 = {"T": 0.8, "Z": 0.85, "alphaFe": 0.85, "NaFe": 0.75, "sigma": 0.85,
            "Re": 0.3}
    xelf2 = {"T": 0.8, "Z": 0.85, "alphaFe": 0.85, "NaFe": 0.75, "sigma": 0.85,
            "Re": 0.3}
    xelfs = {"alpha" : xelf2, "imf": xelf1}
    ylims = {"imf": (0.0, 3.5), "alpha": (0.5, 2.0)}
    xfig = [["sigma", "Z", "alphaFe"], ["NaFe", "T", "Re"]]
    yfields = ["imf", "alpha"]
    for k, xfields in enumerate(xfig):
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        gs.update(left=0.058, right=0.91, bottom=0.07, top=0.99, wspace=0.03,
                  hspace=0.05)
        for i, (yfield, xfield) in enumerate(itertools.product(yfields,
                                                               xfields)):
            xs = t[xfield]
            ys = t[yfield]
            xerrs = np.array([t["{}_lerr".format(xfield)], t["{}_uerr".format(
                xfield)]]).T
            yerrs = np.array([t["{}_lerr".format(yfield)], t["{}_uerr".format(
                yfield)]]).T
            xelf = xelfs[yfield]
            ax = plt.subplot(gs[i])
            # ax.text(0.05, 0.95, "({})".format(letters[i]), transform=ax.transAxes,
            #         fontsize=10, va='top')
            for x, y, xerr, yerr, c in zip(xs, ys, xerrs, yerrs, colors):
                ax.errorbar(x, y, yerr=np.atleast_2d(yerr).T,
                            xerr=np.atleast_2d(xerr).T, fmt="o",
                            ecolor="0.8", mec="w", color=c,
                            mew=0.5, elinewidth=0.5)
            if i in [0,3]:
                ax.set_ylabel(labels[yfield])
            else:
                ax.yaxis.set_ticklabels([])
            if i < 3:
                ax.xaxis.set_ticklabels([])
            else:
                ax.set_xlabel(labels[xfield])
            ax.set_xlim(xlim[xfield])
            ax.set_ylim(ylims[yfield])
            # plot parameter correlations
            idx = np.where((corr["param1"]==xfield) & (corr["param2"]==yfield))[0]
            if idx:
                a = corr["a"][idx]
                b = corr["b"][idx]
                ang = corr["ang"][idx]
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                xel = xmin + xelf[xfield] * (xmax - xmin)
                yel = ymin + 0.2 * (ymax - ymin)
                ellipse = Ellipse((xel, yel), a, b, ang,
                                  facecolor="none", edgecolor="r", linestyle="--")
                ax.text(xel - 0.03 * (xmax - xmin),
                        yel - 0.02 * (ymax - ymin), "$1\sigma$", size=5.5,
                        c="r")
                ax.add_patch(ellipse)
            ####################################################################
            # alpha relations for Kroupa and Salpeter
            xmin, xmax = ax.get_xlim()
            if yfield == "alpha":
                ax.axhline(y=1, c="k", ls="--", lw=0.5)
                ax.axhline(y=1.55, c="k", ls="--", lw=0.5)
                if i == 3:
                    ax.text(xmin + 0.04 * (xmax - xmin), 1.03, "Kroupa",
                            size=5.5, c="k")
                    ax.text(xmin + 0.04 * (xmax - xmin), 1.58, "Salpeter",
                            size=5.5, c="k")
            ####################################################################
            add_literature_results(ax, xfield, yfield)
            plt.legend(loc=3, frameon=False, prop={"size": 5})


        cax = inset_axes(ax,  # here using axis of the lowest plot
                           width="10%",  # width = 5% of parent_bbox width
                           height="180%",  # height : 340% good for a (4x4) Grid
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0.25, 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0)
        cbar = fig.colorbar(mapper, cax=cax, orientation="vertical")
        cbar.set_label("R (kpc)")
        output = os.path.join(wdir, "plots/imf_relations")
        print(output)
        for fmt in ["pdf", "png"]:
            plt.savefig("{}_{}.{}".format(output, k+1, fmt), dpi=300)
        plt.close()

def plot_imf_individual(t, figsize=(3.54, 4.5)):
    global labels, wdir
    corr = Table.read(os.path.join(wdir, "fit_stats.fits"))
    mapper, colors = get_colors(t["R"])
    xlim = {"T": [None, None], "Z": [-0.2, 0.25], "alphaFe": [0, 0.45],
            "NaFe": [None, 0.7], "sigma": [80, 380], "Re": [-0.1, 1.1],
            "logSigma": [None, 11]}
    xelf1 = {"T": 0.8, "Z": 0.85, "alphaFe": 0.85, "NaFe": 0.75, "sigma": 0.85,
            "Re": 0.3, "logSigma": 0.8}
    xelf2 = {"T": 0.8, "Z": 0.85, "alphaFe": 0.85, "NaFe": 0.75, "sigma": 0.85,
            "Re": 0.3, "logSigma": 0.85}
    xelfs = {"alpha" : xelf2, "imf": xelf1}
    ylims = {"imf": (0.5, 3.6), "alpha": (0.5, 2.2)}
    xfields= ["sigma", "Z", "alphaFe", "NaFe", "T", "Re", "logSigma"]
    yfields = ["imf", "alpha"]
    for k, xfield in enumerate(xfields):
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, figure=fig)
        gs.update(left=0.11, right=0.985, bottom=0.07, top=0.99, wspace=0.03,
                  hspace=0.05)
        for i, yfield in enumerate(yfields):
            xs = t[xfield]
            ys = t[yfield]
            xerrs = np.array([t["{}_lerr".format(xfield)],
                              t["{}_uerr".format(xfield)]]).T
            yerrs = np.array([t["{}_lerr".format(yfield)],
                              t["{}_uerr".format(yfield)]]).T
            xelf = xelfs[yfield]
            ax = plt.subplot(gs[i])
            for x, y, xerr, yerr, c in zip(xs, ys, xerrs, yerrs, colors):
                ax.errorbar(x, y, yerr=np.atleast_2d(yerr).T,
                            xerr=np.atleast_2d(xerr).T, fmt="o",
                            ecolor="0.8", mec="w", color=c,
                            mew=0.5, elinewidth=0.5)
            ax.set_ylabel(labels[yfield])
            if i == 0:
                ax.xaxis.set_ticklabels([])
            else:
                ax.set_xlabel(labels[xfield])
            ax.set_xlim(xlim[xfield])
            ax.set_ylim(ylims[yfield])
            # plot parameter correlations
            idx = np.where((corr["param1"] == xfield) & (corr["param2"] == yfield))[
                0]
            if idx:
                a = corr["a"][idx]
                b = corr["b"][idx]
                ang = corr["ang"][idx]
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                xel = xmin + xelf[xfield] * (xmax - xmin)
                yel = ymin + 0.2 * (ymax - ymin)
                ellipse = Ellipse((xel, yel), a, b, ang,
                                  facecolor="none", edgecolor="0.3",
                                  linestyle="--")
                ax.text(xel - 0.02 * (xmax - xmin),
                        yel - 0.02 * (ymax - ymin), "$1\sigma$", size=5.5,
                        c="0.3")
                ax.add_patch(ellipse)
            ####################################################################
            # alpha values for Kroupa and Salpeter
            xmin, xmax = ax.get_xlim()
            if yfield == "alpha":
                ax.axhline(y=1, c="k", ls="--", lw=0.5)
                ax.axhline(y=1.55, c="k", ls="--", lw=0.5)
                if i == 1:
                    ax.text(xmin + 0.04 * (xmax - xmin), 1.03, "Kroupa",
                            size=5.5, c="k")
                    ax.text(xmin + 0.04 * (xmax - xmin), 1.58, "Salpeter",
                            size=5.5, c="k")
            ####################################################################
            add_literature_results(ax, xfield, yfield)
            plt.legend(loc=2, frameon=False, prop={"size": 6}, ncol=2)
        cbar_pos=[0.16, 0.12, 0.25, 0.025]
        cbaxes = fig.add_axes(cbar_pos)
        cbar = plt.colorbar(mapper, cax=cbaxes, orientation="horizontal")
        cbar.set_ticks(np.linspace(0, 16, 5))
        # cbar.ax.tick_params(labelsize=labelsize-1)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('bottom')
        # cl = plt.getp(cbar.ax, 'ymajorticklabels')
        # plt.setp(cl, fontsize=labelsize+2)
        cbar.set_label("R (kpc)")
        output = os.path.join(wdir, "plots/imf_{}".format(xfield))
        for fmt in ["pdf", "png"]:
            plt.savefig("{}.{}".format(output, fmt), dpi=300)
        plt.close()

def add_literature_results(ax, xfield, yfield, posacki=False,
                           mcdermid=False, labarbera=True, ferreras=False):
    global labels
    xmin, xmax = ax.get_xlim()
    ####################################################################
    # Plot results from Parikh et al. (2018)
    xtable = os.path.join(context.home,
                          "tables/parikh2018_{}.txt".format(xfield))
    if os.path.exists(xtable) and yfield == "imf":
        ytable = os.path.join(context.home,
                              "tables/parikh2018_{}.txt".format(yfield))
        y = np.loadtxt(ytable).ravel()[::2].reshape(3, 10)
        x = np.loadtxt(xtable).ravel()[::2].reshape(3, 10)
        colors = ["lightgreen", "limegreen", "green"]
        for j in range(3):
            lparikh = "Parikh et al. (2018)" if (j == 2) else \
                None
            ax.plot(x[j], y[j], "o", c=colors[j], label=lparikh, mec="w")
            ax.plot(x[j][0], y[j][0], "o", c=colors[j], label=None, mec="k")
    ####################################################################
    # Sarzi et al. 2017
    stable = os.path.join(context.home,
                          "tables/sarzi2017_{}_imf.csv".format(xfield))
    if os.path.exists(stable) and yfield == "imf":
        x, y = np.loadtxt(stable, delimiter=",", unpack=True)
        lsarzi = "Sarzi et al. (2017)"  # if i==5 else None
        ax.plot(x, y, "-", c="r", label=None)
        ax.plot(x[0], y[0], "o-", c="r", label=lsarzi)
        ax.plot(x[0], y[0], "o-", c="r", label=None, mec="k")
    ####################################################################
    if xfield == "Z" and yfield == "imf":
        z = np.linspace(-0.3, 0.2, 50)
        # Martin-Navarro 2015
        a = np.random.normal(3.1, 0.5, len(z))
        b = np.random.normal(2.2, 0.1, len(z))
        y = a * z[:, np.newaxis] + b
        ax.plot(z, y.mean(axis=1), "-", c="C4",
                label="Martín-Navarro et al.(2015)")
        # ax.plot(z, np.percentile(y, 16, axis=1), "--", c="C4")
        # ax.plot(z, np.percentile(y, 84, axis=1), "--", c="C4")
    # velocity dispersion relation
    if yfield == "imf" and xfield == "sigma":
        sigma = np.linspace(100, 300, 100)
        gamma = 2.4 + 5.4 * np.log10(sigma/200)
        ax.plot(sigma, gamma, "--", c="violet", 
                label="La Barbera et al. (2013)")
    ####################################################################
    # van Dokkum 2016
    if yfield == "alpha" and xfield == "Re":
        re = np.linspace(0, 1, 100)
        alpha_vd = np.clip(2.48 - 3.6 * re, 1.1, np.infty)
        ax.plot(re, alpha_vd, "-", c="coral",
                label="van Dokkum et al. (2016)")
    ####################################################################
    # Posacki et al (2015)
    if yfield == "alpha" and xfield == "sigma" and posacki:
        sigma = np.linspace(xmin, xmax, 100)
        p0 = np.random.normal(0.4, 0.15, 100)
        p1 = np.random.normal(0.49, 0.05, 100)
        p3 = np.random.normal(-0.07, 0.01, 100)
        s = np.log10(sigma / 200)
        loga = np.outer(p0, s ** 2) + np.outer(p1, s) + p3[:, np.newaxis]
        apos = np.power(10, loga)
        ax.fill_between(sigma, 1.55 * np.percentile(apos, 16, axis=0),
                        1.55 * np.percentile(apos, 84, axis=0),
                        color="0.8", label="Posacki et al. (2015)")
    ####################################################################
    # Barber et al (2019)
    if yfield == "alpha" and xfield == "alphaFe":
        x = np.array([-0.4, 0.2])
        y = np.array([2, 0.8])
        ax.plot(x + 0.18, y, "-", c="gold", label="LoM - Barber et al. (2019)")
        x = np.array([0, 0.4])
        y = [1.1, 1.3]
        ax.plot(x + 0.18, y, "-", c="orange", label="HiM - Barber et al. ("
                                                    "2019)")
    ############################################################################
    # McDermid et al. (2014)
    if yfield == "alpha" and xfield in ["alphaFe", "Z", "T"] and mcdermid:
        if xfield == "alphaFe":
            x = np.linspace(-0.05, 0.45, 100)
            a = -0.257
            b = 0.71
            eps = 0.07
            xp = x
        elif xfield == "T":
            x = np.linspace(0.3, 1.2, 100)
            a = -0.237
            b = 0.126
            eps = 0.069
            xp = np.power(10, x)
        elif xfield == "Z":
            x = np.linspace(-0.3, 0.3)
            a = -0.1181
            b = -0.13
            xp = x
            eps = 0.07
        ax.plot(xp, 1.55 * np.power(10, (a + b * x)), "-",
                c="olive", lw=0.7, label="McDermid et al. (2014)")
        ax.plot(xp, 1.55 * np.power(10, (a + b * x - eps)), "--",
                c="olive", lw=0.7)
        ax.plot(xp, 1.55 * np.power(10, (a + b * x + eps)), "--",
                c="olive", lw=0.7)
    ############################################################################
    if yfield == "alpha" and xfield == "sigma":
        sigma = np.linspace(150, 400, 100)
        aa = [1.31, 0.9, 1.05]
        bb = [-3.1, -2.2, -2.5]
        cc = ["lightblue", "turquoise", "goldenrod"]
        ll = ["Treu et al. (2010)", "Conroy et al. (2012)",
              "Spiniello et al. (2014)"]
        for a, b, l, c in zip(aa, bb, ll, cc):
            y = np.power(10, a * np.log10(sigma) + b) * 1.54
            ax.plot(sigma, y, "--", c=c, label=l)
    if xfield == "logSigma":
        x = np.linspace(-1, 1, 100)
        if yfield == "imf":
            y = 1.3 + 1.84 / (1 + np.exp(-x / 0.24))
        else:
            y = 1. + 0.98 / (1 + np.exp(-x / 0.24))
        ax.plot(x + 10, y, "-", c="green", label="La Barbera et al. (2019)")

    return



if __name__ == "__main__":
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma_*$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "Re" : "$R / R_e$",  "M2L": "$M_*/L_r$",
              "alpha": "$\\alpha=(M_*/L_r) / (M_*/L_r)_{\\rm MW}$",
              "logSigma": "$\\log \\Sigma$ (M$_\\odot$ / kpc$^2$)"}
    dataset = "MUSE"
    targetSN = 250
    wdir =  os.path.join(context.data_dir, dataset, "voronoi",
                         "sn{}".format(targetSN))
    outdir = os.path.join(wdir, "plots")
    ###########################################################################
    # Loading and preparing data
    tfile = os.path.join(wdir, "results.fits")
    t = Table.read(tfile)
    t["R_uerr"] = 0
    t["R_lerr"] = 0
    t["Re"] = t["R"] / 8.4
    t["Re_uerr"] = 0
    t["Re_lerr"] = 0
    ############################################################################
    profiles = False
    if profiles:
        plot_profiles(t, "R", ["imf", "M2L", "alpha"],
                      output=os.path.join(outdir, "R_imf-M2L-alpha"))
        plot_profiles(t, "Re", ["imf", "M2L", "alpha"],
                      output=os.path.join(outdir, "Re_imf-M2L-alpha"),
                      xlim=[None, 1.1])
        xfracs = [0.75, 0.75, 0.25]
        yfracs = [0.2, 0.2, 0.7]
        plot_profiles(t, "logSigma", ["imf", "M2L", "alpha"],
                      output=os.path.join(outdir, "logSigma_imf-M2L-alpha"),
                      xfracs=xfracs, yfracs=yfracs)
        xfracs = [0.85, 0.85, 0.85]
        yfracs = [0.3, 0.3, 0.3]
        plot_profiles(t, "sigma", ["imf", "M2L", "alpha"],
                      output=os.path.join(outdir, "sigma_imf-M2L-alpha"),
                      xfracs=xfracs, yfracs=yfracs)
    ############################################################################
    # plot_imf_relations(t)
    plot_imf_individual(t)
    ############################################################################