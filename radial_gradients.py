# -*- coding: utf-8 -*-
"""

Created on 21/07/2020

Author : Carlos Eduardo Barbosa

Calculating gradients in the stellar populations using a piecewise linear
function.

"""
import os

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import pymc3 as pm
import theano.tensor as tt

import context
from plots.plot_maps import make_table

def piecewise_linear_function0(x, x0, b, k1, k2, pkg=np):
    x8 = np.log10(1.05)
    return pkg.where(x <= x0, k1 * (x - x8) + b, k1 * (x0 - x8) + b + k2 * (
            x - x0))

def piecewise_linear_function1(x, x0, x1, b, k1, k2, k3, pkg=np):
    if pkg == np:
        if x1 < x0:
            x0, x1 = x1, x0
    if pkg == pm:
        x0, x1 = tt.sort([x0, x1])
    s1 = pkg.where(x <= x0, k1 * x + b, 0)
    s2 = pkg.where((x > x0) & (x <= x1 ), k1 * x0 + b + k2 * (x - x0), 0)
    s3 = pkg.where(x > x1, k1 * x0 + b + k2 * (x1 - x0) + k3 * (x - x1), 0)
    return s1 + s2 + s3

def piecewise_linear_function2(x, x0, x1, x2, b, k1, k2, k3, k4, pkg=np):
    if x1 < x0:
        x0, x1 = x1, x0
    s1 = k1 * x + b
    s2 = pkg.where(x >= x0, k2 * (x - x0), 0)
    s3 = pkg.where(x >= x1, k3 * (x - x1), 0)
    s4 = pkg.where(x >= x2, k4 * (x - x2), 0)
    return s1 + s2 + s3 + s4

def plot_table(table, ypars, xpar="R", xlim=(0.07, 20), xscale="log"):
    global labels
    b16= Table.read(os.path.join(context.tables_dir, "barbosa16.fits"))
    b16 = b16[b16["S_N"] >= 50]
    b16["logT"] = np.log10(b16["Age"])
    b16["e_logT"] = np.abs(b16["e_Age"] / b16["Age"] / np.log(10))
    b16_fields = {"Z": "__Z_H_", "logT": "logT", "alphaFe": "__alpha_Fe_"}
    x = table[xpar]
    fig = plt.figure(figsize=(context.fig_width, 7))
    # Plot weights
    t = Table.read(os.path.join(context.tables_dir, "barbosa2018_weights.fits"))
    r = t["R"].data
    f = np.array([t[p].data for p in ["w_core", "w_galhalo", "w_cd"]]).T
    rs = []
    for i in range(3):
        rismax = np.where(f[:,i] == np.max(f, axis=1), r, np.nan)
        rs.append(np.nanmin(rismax))
        rs.append(np.nanmax(rismax))
    rs = np.sort(rs)
    fcs = ["skyblue", "g", "khaki", "orange", "coral"]
    ax = plt.subplot(len(ypars) + 1, 1, 7)
    ax.set_xscale(xscale)
    for j in [1, 3]:
        ax.axvspan(rs[j], rs[j + 1], facecolor=fcs[j], alpha=0.15)
    ax.plot(r, f[:,0], ".", ms=0.5, c=fcs[0])
    ax.plot(r, f[:,1], ".", ms=0.5, c=fcs[2])
    ax.plot(r, f[:,2], ".", ms=0.5, c=fcs[4])
    ax.set_xlim(xlim)
    ax.set_ylim(-0.1, 1.2)
    legend_elements = []
    photlabels = ["core", None, "gal. halo", None, "cD halo"]
    for j in [0, 2, 4]:
        legend_elements.append(Patch(facecolor=fcs[j], edgecolor=fcs[j],
                         label=photlabels[j]))
    plt.legend(handles=legend_elements, ncol=3, frameon=False, loc=2)
    ax.set_ylabel("$\\omega$")
    ax.set_xlabel(labels["R"])
    # Plot SSP parameters
    for i, p in enumerate(ypars):
        ax = plt.subplot(len(ypars) + 1, 1, i + 1)
        for j in range(5):
            if j in [0, 2, 4]:
                continue
            ax.axvspan(rs[j], rs[j+1], facecolor=fcs[j], alpha=0.15)
        y = table[p].data
        ylower = table["{}_lerr".format(p)].data
        yupper = table["{}_uerr".format(p)].data

        ax.set_xscale(xscale)
        label = "This work" if i ==0 else None
        ax.errorbar(x, y, yerr=[ylower, yupper], ecolor=None, fmt="o",
                    mew=0.5, elinewidth=0.5, mec="w", ms=5, c="C0", label=label)
        ax.set_ylabel(labels[p])
        ax.set_xlim(xlim)
        ax.xaxis.set_ticklabels([])
        if p in b16_fields:
            label = "Barbosa et al. (2016)" if i==0 else None
            ax.errorbar(b16["Rad"], b16[b16_fields[p]],
                        yerr=b16["e_{}".format(b16_fields[p]).replace("___",
                                                                      "__")],
                        fmt="x", mew=0.5, label=label,
                        elinewidth=0.5, ms=4.5, c="k")
        plt.legend(frameon=False, prop={"size": 6})
    plt.subplots_adjust(left=0.15, right=0.99, top=0.995, bottom=0.045,
                        hspace=0.06)
    out = os.path.join(wdir, "plots/ssp_gradients")
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(out, fmt), dpi=250)
    plt.close()
    return

def plot_model(x, y, ylower, yupper, trace, fun, xpars, fpars, xbreaks, out):
    global labels
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]

    fig = plt.figure(figsize=(context.fig_width, 6))
    xplot = np.linspace(x[0], x[-1], 100)
    lxplot = np.log10(xplot)
    b16= Table.read(os.path.join(context.tables_dir, "barbosa16.fits"))
    b16 = b16[b16["S_N"] >= 50]
    b16["logT"] = np.log10(b16["Age"])
    b16["e_logT"] = np.abs(b16["e_Age"] / b16["Age"] / np.log(10))
    b16_fields = {"Z": "__Z_H_", "logT": "logT", "alphaFe": "__alpha_Fe_"}
    xbc = ["r", "y", "k"]
    for i, p in enumerate(params):
        pars = []
        for xp in xpars:
            pars.append(trace[xp])
        for fp in fpars:
            pars.append(trace["{}_{}".format(fp, p)])
        pars = np.column_stack(pars)
        ymodel = np.zeros((len(pars), len(lxplot)))
        for j in range(len(pars)):
            ymodel[j] = fun(lxplot, *pars[j])
        ax = plt.subplot(len(params)+ 1, 1, i + 1)
        ax.set_xscale("log")
        ax.errorbar(x, y[i], yerr=[ylower[i], yupper[i]], ecolor=None, fmt="o",
                    mew=0.5, elinewidth=0.5, mec="w", ms=5, label="This work")
        # Plot previous results
        if p in b16_fields:
            ax.errorbar(b16["Rad"], b16[b16_fields[p]],
                        yerr=b16["e_{}".format(b16_fields[p]).replace("___",
                                                                      "__")],
                        fmt="s", mew=0.5, label="Barbosa et al (2016)",
                        elinewidth=0.5, mec="w", ms=4.5, c="0.6")
        for c, per in zip(colors, percs):
            ax.fill_between(xplot, np.percentile(ymodel, per, axis=0),
                             np.percentile(ymodel, per + 10, axis=0),
                            color=c, alpha=0.8, ec="none", lw=0)
        for j, xb in enumerate(xbreaks):
            x0 = trace[xb]
            ax.axvline(x=np.median(x0), c=xbc[j], ls="-", lw=0.5)
            ax.axvline(x=np.percentile(x0, 16), c=xbc[j], ls="--", lw=0.5)
            ax.axvline(x=np.percentile(x0, 84), c=xbc[j], ls="--", lw=0.5)
        plt.ylabel(labels[p])
        ax.xaxis.set_ticklabels([])
        ax.set_xlim(None, 40)
    ax = plt.subplot(len(params) + 1, 1, 6)
    sigma = table["sigma"].data
    ylower = table["sigma_lerr"].data
    yupper= table["sigma_uerr"].data
    ax.errorbar(r, sigma, yerr=[ylower, yupper], ecolor=None, fmt="o",
                mew=0.5, elinewidth=0.5, mec="w", ms=5, label="This work")
    for j, xb in enumerate(xbreaks):
        x0 = trace[xb]
        ax.axvline(x=np.median(x0), c=xbc[j], ls="-", lw=0.5)
        ax.axvline(x=np.percentile(x0, 16), c=xbc[j], ls="--", lw=0.5)
        ax.axvline(x=np.percentile(x0, 84), c=xbc[j], ls="--", lw=0.5)
    ax.set_ylabel(labels["sigma"])
    plt.xlabel(labels["R"])
    ax.set_xscale("log")
    ax.set_xlim(None, 40)
    ax.errorbar(100, 300, yerr=1,
                fmt="s", mew=0.5, label="Barbosa et al (2016)",
                elinewidth=0.5, mec="w", ms=4.5, c="0.6")
    plt.legend(frameon=False, prop={"size": 5})
    plt.subplots_adjust(left=0.11, right=0.99, top=0.995, bottom=0.052,
                        hspace=0.06)
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(out, fmt), dpi=250)
    plt.close()
    return

def print_table(trace, xbreaks, fpars):
    global labels
    for i, p in enumerate(params):
        pars = []
        for xb in xbreaks:
            pars.append(trace[xb])
        for fp in fpars:
            pars.append(trace["{}_{}".format(fp, p)])
        pars = np.column_stack(pars)
        m = np.percentile(pars, 50, axis=0)
        splus = np.percentile(pars, 84, axis=0) - m
        sminus = m - np.percentile(pars, 16, axis=0)
        line = [labels[p]]
        for k in range(len(xbreaks) + len(fpars)):
            s = "${:.3f}^{{+{:.3f}}}_{{-{:.3f}}}$".format(m[k], splus[k],
                                                       sminus[k])
            line.append(s)
        line = " & ".join(line) + "\\\\\n"
        print(line)

def fit_model0(x, y, ysd, params, wdir):
    db = os.path.join(wdir, "gradients_model0")
    model1 = pm.Model()
    with model1:
        x0 = pm.Uniform("x0", lower=0, upper=10)
        lx0 = pm.Deterministic("logx0", pm.math.log(x0) / pm.math.log(10))
        for i, p in enumerate(params):
            b = pm.Uniform("b_{}".format(p), lower=-10, upper=10)
            k1 = pm.Normal("k1_{}".format(p), mu=0, sd=0.3)
            k2 = pm.Normal("k2_{}".format(p), mu=0, sd=0.3)
            nu = pm.HalfNormal("nu_{}".format(p), sd=1)
            fun = piecewise_linear_function0(x, lx0, b, k1, k2, pkg=pm.math)
            like = pm.StudentT("l_{}".format(p), mu=fun, sigma=ysd[i],
                               observed=y[i], nu=nu)
        if os.path.exists(db):
            trace = pm.load_trace(db)
        else:
            trace = pm.sample(n_init=2000)
            pm.save_trace(trace, db)
    return trace

def fit_model1(x, y, ysd, params, wdir):
    db = os.path.join(wdir, "gradients_model1")
    model1 = pm.Model()
    with model1:
        x0 = pm.Uniform("x0", lower=0, upper=20)
        x1 = pm.Uniform("x1", lower=0, upper=20)
        lx0 = pm.math.log(x0) / pm.math.log(10)
        lx1 = pm.math.log(x1) / pm.math.log(10)
        for i, p in enumerate(params):
            b = pm.Uniform("b_{}".format(p), lower=-10, upper=10)
            k1 = pm.Normal("k1_{}".format(p), mu=0, sd=0.3)
            k2 = pm.Normal("k2_{}".format(p), mu=0, sd=0.3)
            k3 = pm.Normal("k3_{}".format(p), mu=0, sd=0.3)
            nu = pm.HalfNormal("nu_{}".format(p), sd=1)
            fun = piecewise_linear_function1(x, lx0, lx1, b, k1, k2, k3,
                                             pkg=pm.math)
            like = pm.StudentT("l_{}".format(p), mu=fun, sigma=ysd[i],
                               observed=y[i], nu=nu)
        if os.path.exists(db):
            trace = pm.load_trace(db)
        else:
            trace = pm.sample(n_init=2000)
            pm.save_trace(trace, db)
    print(pm.summary(trace))
    return trace

def fit_model2(x, y, ysd, params, wdir):
    db = os.path.join(wdir, "gradients_model2")
    model2 = pm.Model()
    with model2:
        for i, p in enumerate(params):
            x0 = pm.Uniform("x0_{}".format(p), lower=0, upper=20)
            x1 = pm.Uniform("x1_{}".format(p), lower=0, upper=20)
            lx0 = pm.Deterministic("logx0_{}".format(p), pm.math.log(x0) /
                                   pm.math.log(10))
            lx1 = pm.Deterministic("logx1_{}".format(p), pm.math.log(x1) /
                                   pm.math.log(10))
            b = pm.Uniform("b_{}".format(p), lower=-10, upper=10)
            k1 = pm.Normal("k1_{}".format(p), mu=0, sd=0.3)
            k2 = pm.Normal("k2_{}".format(p), mu=0, sd=0.3)
            k3 = pm.Normal("k3_{}".format(p), mu=0, sd=0.3)
            nu = pm.HalfNormal("nu_{}".format(p), sd=1)
            fun = piecewise_linear_function1(x, lx0, lx1, b, k1, k2, k3,
                                             pkg=pm.math)
            like = pm.StudentT("l_{}".format(p), mu=fun, sigma=ysd[i],
                               observed=y[i], nu=nu)
        if os.path.exists(db):
            trace = pm.load_trace(db)
        else:
            trace = pm.sample(n_init=2000)
            pm.save_trace(trace, db)
    print(pm.summary(trace))
    return trace

def fit_model3(y, ysd, ylower, yupper, wdir):
    db = os.path.join(wdir, "gradients_model3")
    model1 = pm.Model()
    with model1:
        x0 = pm.Uniform("x0", lower=0, upper=10)
        delta1 = pm.Uniform("delta1", lower=0, upper=10)
        delta2 = pm.Uniform("delta2", lower=0, upper=10)
        x1 = x0 + delta1
        x2 = x1 + delta2
        lx0 = pm.math.log(x0) / pm.math.log(10)
        lx1 = pm.math.log(x1) / pm.math.log(10)
        lx2 = pm.math.log(x2) / pm.math.log(10)
        for i, p in enumerate(params):
            b = pm.Uniform("b_{}".format(p), lower=-10, upper=10)
            k1 = pm.Normal("k1_{}".format(p), mu=0, sd=0.3)
            k2 = pm.Normal("k2_{}".format(p), mu=0, sd=0.3)
            k3 = pm.Normal("k3_{}".format(p), mu=0, sd=0.3)
            k4 = pm.Normal("k4_{}".format(p), mu=0, sd=0.3)
            nu = pm.HalfNormal("nu_{}".format(p), sd=1)
            fun = piecewise_linear_function2(x, lx0, lx1, lx2, b, k1, k2, k3,
                                             k4, pkg=pm.math)
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
    dx1 = trace["delta1"]
    dx2 = trace["delta2"]
    x1 = x0 + dx1
    x2 = x1 + dx2
    lx0 = np.log10(x0)
    lx1 = np.log10(x1)
    lx2 = np.log10(x2)
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
        k4 = trace["k4_{}".format(p)]
        pars = np.column_stack([lx0, lx1, lx2, b, k1, k2, k3, k4])
        # Making table
        pars2 = np.column_stack([x0, x1, x2, b, k1, k2, k3, k4])
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
            ymodel[j] = piecewise_linear_function2(lx_plot, *pars[j])
        ax = plt.subplot(len(params)+ 1, 1, i + 1)
        ax.set_xscale("log")
        ax.errorbar(r, y[i], yerr=[ylower[i], yupper[i]], ecolor=None, fmt="o",
                    mew=0.5, elinewidth=0.5, mec="w", ms=5)
        ymin, ymax = ax.get_ylim()
        for c, per in zip(colors, percs):
            ax.fill_between(x_plot, np.percentile(ymodel, per, axis=0),
                             np.percentile(ymodel, per + 10, axis=0),
                            color=c, alpha=0.8, ec="none", lw=0)

        colors2 = ["r", "g", "b"]
        for k, xb in enumerate([x0, x1, x2]):
            ax.axvline(x=np.median(xb), c=colors2[k], ls="-", lw=0.5)
            ax.axvline(x=np.percentile(xb, 16), c=colors2[k], ls="--", lw=0.5)
            ax.axvline(x=np.percentile(xb, 84), c=colors2[k], ls="--", lw=0.5)
        plt.ylabel(labels[p])
        ax.xaxis.set_ticklabels([])
    ax = plt.subplot(len(params) + 1, 1, 6)
    sigma = table["sigma"].data
    ylower = table["sigma_lerr"].data
    yupper= table["sigma_uerr"].data
    ax.errorbar(r, sigma, yerr=[ylower, yupper], ecolor=None, fmt="o",
                mew=0.5, elinewidth=0.5, mec="w", ms=5)
    colors = ["r", "g", "b"]
    for i, xb in enumerate([x0, x1, x2]):
        ax.axvline(x=np.median(xb), c=colors[i], ls="-", lw=0.5)
        ax.axvline(x=np.percentile(xb, 16), c=colors[i], ls="--", lw=0.5)
        ax.axvline(x=np.percentile(xb, 84), c=colors[i], ls="--", lw=0.5)
    ax.set_ylabel(labels["sigma"])
    plt.xlabel(labels["R"])
    ax.set_xscale("log")
    plt.subplots_adjust(left=0.15, right=0.985, top=0.995, bottom=0.052,
                        hspace=0.06)
    out = os.path.join(wdir, "plots/ssp_grads3")
    for fmt in ["pdf", "png"]:
        plt.savefig("{}.{}".format(out, fmt), dpi=250)
    plt.close()
    return

def fit_model4(x, y, ysd, params, wdir):
    db = os.path.join(wdir, "gradients_model4")
    model1 = pm.Model()
    with model1:
        for i, p in enumerate(params):
            x0 = pm.Uniform("x0_{}".format(p), lower=0, upper=10)
            lx0 = pm.Deterministic("logx0_{}".format(p), pm.math.log(x0) /
                                   pm.math.log(10))
            b = pm.Uniform("b_{}".format(p), lower=-10, upper=10)
            k1 = pm.Normal("k1_{}".format(p), mu=0, sd=0.3)
            k2 = pm.Normal("k2_{}".format(p), mu=0, sd=0.3)
            nu = pm.HalfNormal("nu_{}".format(p), sd=1)
            fun = piecewise_linear_function0(x, lx0, b, k1, k2, pkg=pm.math)
            like = pm.StudentT("l_{}".format(p), mu=fun, sigma=ysd[i],
                               observed=y[i], nu=nu)
        if os.path.exists(db):
            trace = pm.load_trace(db)
        else:
            trace = pm.sample(n_init=2000)
            pm.save_trace(trace, db)
    print(pm.summary(trace))
    return trace

def make_model0():
    trace0 = fit_model0(x, y, ysd, params, wdir)
    x0pars = ["logx0"]
    f0pars = ["b", "k1", "k2"]
    xbreaks = ["x0"]
    out0 = os.path.join(wdir, "plots/ssp_grads0")
    plot_model(r, y, ylower, yupper, trace0, piecewise_linear_function0,
               x0pars, f0pars, xbreaks, out0)
    print_table(trace0, xbreaks, f0pars)

def make_model2():
    trace = fit_model2(x, y, ysd, params, wdir)
    x0pars = ["logx0", "logx1"]
    f0pars = ["b", "k1", "k2", "k2"]
    xbreaks = ["x0", "x1"]
    out = os.path.join(wdir, "plots/ssp_grads2")
    plot_model(r, y, ylower, yupper, trace, piecewise_linear_function0,
               x0pars, f0pars, xbreaks, out)
    print_table(trace, xbreaks, f0pars)

def make_model4():
    trace0 = fit_model4(x, y, ysd, params, wdir)
    x0pars = []
    f0pars = ["logx0", "b", "k1", "k2"]
    xbreaks = ["x0"]
    out0 = os.path.join(wdir, "plots/ssp_grads4")
    plot_model(r, y, ylower, yupper, trace0, piecewise_linear_function0,
               x0pars, f0pars, xbreaks, out0)
    print_table(trace0, xbreaks, f0pars)


if __name__ == "__main__":
    labels = {"R": "$R$ (kpc)", "sigma": r"$\sigma_*$ (km/s)",
              "V": "$V$ (km/s)", "imf": r"$\Gamma_b$", "Z": "[Z/H]",
              "T": "Age (Gyr)", "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "logT": "$\log $ Age (Gyr)"}
    wdir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    table = make_table(targetSN=250)
    table["logT"] = np.log10(table["T"].data)
    table["logT_lerr"] = np.abs(table["T_lerr"].data / table["T"] / np.log(10))
    table["logT_uerr"] = np.abs(table["T_uerr"].data / table["T"] / np.log(10))
    params = ["logT", "Z", "alphaFe", "imf", "NaFe"]
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
    plot_table(table, params + ["sigma"])
    # Calculating gradients
    make_model0()
    # make_model4()

    # fit_model1(y, ysd, ylower, yupper, wdir)
    # fit_model2(y, ysd, ylower, yupper, wdir)
    # fit_model3(y, ysd, ylower, yupper, wdir)



