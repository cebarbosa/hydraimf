# -*- coding: utf-8 -*-
"""

Created on 21/07/2020

Author : Carlos Eduardo Barbosa

Calculating gradients in the stellar populations using a piecewise linear
function.

"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pymc3 as pm
import theano.tensor as tt

import context
from plots.plot_maps import make_table

def piecewise_linear_function(x, x0, x1, b, k1, k2, k3, pkg=np):
    s1 = k1 * x + b
    s2 = pkg.where(x >= x0, k2 * (x - x0), 0)
    s3 = pkg.where(x >= x1, k3 * (x - x1), 0)
    return s1 + s2 + s3

def residue(p, x, y, yerr):
    x0, x1, b, k1, k2, k3 = p
    return (piecewise_linear_function(x, x0, x1, b, k1, k2, k3) - y) / yerr

def fit_model1(y, ysd, ylower, yupper, wdir):
    db = os.path.join(wdir, "gradients_model1")
    model1 = pm.Model()
    with model1:
        x0 = pm.Uniform("x0", lower=0, upper=10)
        delta = pm.Uniform("delta", lower=0, upper=10)
        x1 = x0 + delta
        lx0 = pm.math.log(x0) / pm.math.log(10)
        lx1 = pm.math.log(x1) / pm.math.log(10)
        for i, p in enumerate(params):
            b = pm.Uniform("b_{}".format(p), lower=-10, upper=10)
            k1 = pm.Normal("k1_{}".format(p), mu=0, sd=0.3)
            k2 = pm.Normal("k2_{}".format(p), mu=0, sd=0.3)
            k3 = pm.Normal("k3_{}".format(p), mu=0, sd=0.3)
            nu = pm.HalfNormal("nu_{}".format(p), sd=1)
            fun = piecewise_linear_function(x, lx0, lx1, b, k1, k2, k3,
                                            pkg=pm.math)
            like = pm.StudentT("l_{}".format(p), mu=fun, sigma=ysd[i],
                               observed=y[i], nu=nu)
        if os.path.exists(db):
            trace = pm.load_trace(db)
        else:
            trace = pm.sample(n_init=2000)
            pm.save_trace(trace, db)
    print(pm.summary(trace))
    # Making plot
    x0 = trace["x0"]
    dx = trace["delta"]
    x1 = x0 + dx
    lx0 = np.log10(x0)
    lx1 = np.log10(x1)
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma_*$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "logT": "$\log $ Age (Gyr)"}
    fig = plt.figure(figsize=(context.fig_width, 6))
    x_plot = np.linspace(r[0], r[-1], 100)
    lx_plot = np.log10(x_plot)
    for i, p in enumerate(params):
        b = trace["b_{}".format(p)]
        k1 = trace["k1_{}".format(p)]
        k2 = trace["k2_{}".format(p)]
        k3 = trace["k3_{}".format(p)]
        pars = np.column_stack([lx0, lx1, b, k1, k2, k3])
        # Making table
        pars2 = np.column_stack([x0, x1, b, k1, k2, k3])
        m = np.percentile(pars2, 50, axis=0)
        splus = np.percentile(pars2, 84, axis=0) - m
        sminus = m - np.percentile(pars2, 16, axis=0)
        line = [labels[p]]
        for k in range(6):
            if k < 2 and i != 2:
                s = ""
            else:
                s = "${:.3f}^{{+{:.3f}}}_{{-{:.3f}}}$".format(m[k], splus[k],
                                                       sminus[k])
            line.append(s)
        line = " & ".join(line) + "\\\\\n"
        print(line)
        ymodel = np.zeros((len(x0), len(x_plot)))
        for j in range(len(x0)):
            ymodel[j] = piecewise_linear_function(lx_plot, *pars[j])
        ax = plt.subplot(len(params)+ 1, 1, i + 1)
        ax.set_xscale("log")
        ax.errorbar(r, y[i], yerr=[ylower[i], yupper[i]], ecolor=None, fmt="o",
                    mew=0.5, elinewidth=0.5, mec="w", ms=5)
        ymin, ymax = ax.get_ylim()
        for c, per in zip(colors, percs):
            ax.fill_between(x_plot, np.percentile(ymodel, per, axis=0),
                             np.percentile(ymodel, per + 10, axis=0),
                            color=c, alpha=0.8, ec="none", lw=0)
        ax.axvline(x=np.median(x0), c="g", ls="-", lw=0.5)
        ax.axvline(x=np.percentile(x0, 16), c="g", ls="--", lw=0.5)
        ax.axvline(x=np.percentile(x0, 84), c="g", ls="--", lw=0.5)
        ax.axvline(x=np.median(x1), c="r", ls="-", lw=0.5)
        ax.axvline(x=np.percentile(x1, 16), c="r", ls="--", lw=0.5)
        ax.axvline(x=np.percentile(x1, 84), c="r", ls="--", lw=0.5)
        plt.ylabel(labels[p])
        ax.xaxis.set_ticklabels([])
    ax = plt.subplot(len(params) + 1, 1, 6)
    sigma = table["sigma"].data
    ylower = table["sigma_lerr"].data
    yupper= table["sigma_uerr"].data
    ax.errorbar(r, sigma, yerr=[ylower, yupper], ecolor=None, fmt="o",
                mew=0.5, elinewidth=0.5, mec="w", ms=5)
    ax.axvline(x=np.median(x0), c="g", ls="-", lw=0.5)
    ax.axvline(x=np.percentile(x0, 16), c="g", ls="--", lw=0.5)
    ax.axvline(x=np.percentile(x0, 84), c="g", ls="--", lw=0.5)
    ax.axvline(x=np.median(x1), c="r", ls="-", lw=0.5)
    ax.axvline(x=np.percentile(x1, 16), c="r", ls="--", lw=0.5)
    ax.axvline(x=np.percentile(x1, 84), c="r", ls="--", lw=0.5)
    ax.set_ylabel(labels["sigma"])
    plt.xlabel(labels["R"])
    ax.set_xscale("log")
    plt.subplots_adjust(left=0.15, right=0.985, top=0.995, bottom=0.052,
                        hspace=0.06)
    out = os.path.join(wdir, "plots/ssp_grads1")
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(out, fmt), dpi=250)
    plt.close()
    return

def fit_model2(y, ysd, ylower, yupper, wdir):
    db = os.path.join(wdir, "gradients_model2")
    model2 = pm.Model()
    with model2:
        for i, p in enumerate(params):
            x0 = pm.Uniform("x0_{}".format(p), lower=0, upper=10)
            delta = pm.Uniform("delta_{}".format(p), lower=0, upper=10)
            x1 = x0 + delta
            lx0 = pm.math.log(x0) / pm.math.log(10)
            lx1 = pm.math.log(x1) / pm.math.log(10)
            b = pm.Uniform("b_{}".format(p), lower=-10, upper=10)
            k1 = pm.Normal("k1_{}".format(p), mu=0, sd=0.3)
            k2 = pm.Normal("k2_{}".format(p), mu=0, sd=0.3)
            k3 = pm.Normal("k3_{}".format(p), mu=0, sd=0.3)
            nu = pm.HalfNormal("nu_{}".format(p), sd=1)
            fun = piecewise_linear_function(x, lx0, lx1, b, k1, k2, k3,
                                            pkg=pm.math)
            like = pm.StudentT("l_{}".format(p), mu=fun, sigma=ysd[i],
                               observed=y[i], nu=nu)
        if os.path.exists(db):
            trace = pm.load_trace(db)
        else:
            trace = pm.sample(n_init=2000)
            pm.save_trace(trace, db)
    print(pm.summary(trace))
    # Making plot
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma_*$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "logT": "$\log $ Age (Gyr)"}
    fig = plt.figure(figsize=(context.fig_width, 6))
    x_plot = np.linspace(r[0], r[-1], 100)
    lx_plot = np.log10(x_plot)
    for i, p in enumerate(params):
        x0 = trace["x0_{}".format(p)]
        dx = trace["delta_{}".format(p)]
        x1 = x0 + dx
        lx0 = np.log10(x0)
        lx1 = np.log10(x1)
        b = trace["b_{}".format(p)]
        k1 = trace["k1_{}".format(p)]
        k2 = trace["k2_{}".format(p)]
        k3 = trace["k3_{}".format(p)]
        pars = np.column_stack([lx0, lx1, b, k1, k2, k3])
        # Making table
        pars2 = np.column_stack([x0, x1, b, k1, k2, k3])
        m = np.percentile(pars2, 50, axis=0)
        splus = np.percentile(pars2, 84, axis=0) - m
        sminus = m - np.percentile(pars2, 16, axis=0)
        line = [labels[p]]
        for k in range(6):
            s = "${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(m[k], splus[k],
                                                       sminus[k])
            line.append(s)
        line = " & ".join(line) + "\\\\\n"
        print(line)
        ymodel = np.zeros((len(x0), len(x_plot)))
        for j in range(len(x0)):
            ymodel[j] = piecewise_linear_function(lx_plot, *pars[j])
        ax = plt.subplot(len(params)+ 1, 1, i + 1)
        ax.set_xscale("log")
        ax.errorbar(r, y[i], yerr=[ylower[i], yupper[i]], ecolor=None, fmt="o",
                    mew=0.5, elinewidth=0.5, mec="w", ms=5)
        ymin, ymax = ax.get_ylim()
        for c, per in zip(colors, percs):
            ax.fill_between(x_plot, np.percentile(ymodel, per, axis=0),
                             np.percentile(ymodel, per + 10, axis=0),
                            color=c, alpha=0.8, ec="none", lw=0)
        ax.axvline(x=np.median(x0), c="g", ls="-", lw=0.5)
        ax.axvline(x=np.percentile(x0, 16), c="g", ls="--", lw=0.5)
        ax.axvline(x=np.percentile(x0, 84), c="g", ls="--", lw=0.5)
        ax.axvline(x=np.median(x1), c="r", ls="-", lw=0.5)
        ax.axvline(x=np.percentile(x1, 16), c="r", ls="--", lw=0.5)
        ax.axvline(x=np.percentile(x1, 84), c="r", ls="--", lw=0.5)
        plt.ylabel(labels[p])
        ax.xaxis.set_ticklabels([])
    ax = plt.subplot(len(params) + 1, 1, 6)
    sigma = table["sigma"].data
    ylower = table["sigma_lerr"].data
    yupper= table["sigma_uerr"].data
    ax.errorbar(r, sigma, yerr=[ylower, yupper], ecolor=None, fmt="o",
                mew=0.5, elinewidth=0.5, mec="w", ms=5)
    ax.axvline(x=np.median(x0), c="g", ls="-", lw=0.5)
    ax.axvline(x=np.percentile(x0, 16), c="g", ls="--", lw=0.5)
    ax.axvline(x=np.percentile(x0, 84), c="g", ls="--", lw=0.5)
    ax.axvline(x=np.median(x1), c="r", ls="-", lw=0.5)
    ax.axvline(x=np.percentile(x1, 16), c="r", ls="--", lw=0.5)
    ax.axvline(x=np.percentile(x1, 84), c="r", ls="--", lw=0.5)
    ax.set_ylabel(labels["sigma"])
    plt.xlabel(labels["R"])
    ax.set_xscale("log")
    plt.subplots_adjust(left=0.15, right=0.985, top=0.995, bottom=0.052,
                        hspace=0.06)
    out = os.path.join(wdir, "plots/ssp_grads2")
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(out, fmt), dpi=250)
    plt.close()
    return


if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    table = make_table(targetSN=250)
    table["logT"] = np.log10(table["T"].data)
    table["logT_lerr"] = np.abs(table["T_lerr"].data / table["T"] / np.log(10))
    table["logT_uerr"] = np.abs(table["T_uerr"].data / table["T"] / np.log(10))
    params = ["Z", "logT", "alphaFe", "imf", "NaFe"]
    r = table["R"].data
    x = np.log10(r)
    rm = np.linspace(r.min(), r.max(), 100)
    xm = np.linspace(x.min(), x.max(), 100)
    y = np.zeros((len(params), len(table)))
    ylower, yupper = np.zeros_like(y), np.zeros_like(y)
    for i, p in enumerate(params):
            y[i] = table[p].data
            ylower[i] = table["{}_lerr".format(p)].data
            yupper[i]= table["{}_uerr".format(p)].data
    ysd = np.mean([ylower, yupper], axis=0)

    # Calculating gradients
    # fit_model1(y, ysd, ylower, yupper, wdir)
    fit_model2(y, ysd, ylower, yupper, wdir)





