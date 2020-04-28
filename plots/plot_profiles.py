# -*- coding: utf-8 -*-
"""

Created on 24/04/2020

Author : Carlos Eduardo Barbosa

"""
import os

from astropy.table import Table
import matplotlib.pyplot as plt

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
    plt.subplots_adjust(left=0.14, right=0.985, top=0.99, bottom=0.055,
                        hspace=0.03)
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
    plot_profiles(t, output, "sigma", ["imf", "Z", "T", "alphaFe", "NaFe"])