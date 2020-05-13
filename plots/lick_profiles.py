# -*- coding: utf-8 -*-
"""

Created on 09/05/2020

@author: Carlos Eduardo Barbosa

Radial profile of Lick indices

"""
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table, join, vstack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colorbar import Colorbar
import scipy.ndimage as ndimage
from tqdm import tqdm

import context

def make_table_obs(filenames, licktype=None, indnames=None):
    """ Join tables with Lick indices from observations. """
    licktype = "Ia" if licktype is None else licktype
    if indnames is None:
        indnames = ['bTiO_muse', 'H_beta', 'Fe5015', 'Mg_1', 'Mg_2', 'Mg_b',
                    'Fe5270', 'Fe5335', 'Fe5406', 'Fe5709', 'Fe5782', 'aTiO',
                    'Na_D', 'TiO_1', 'TiO_2_muse', 'CaH_1',
                    'CaH_2_muse', 'TiO_3', 'TiO_4', 'NaI', 'CaT1', 'CaT2',
                    'CaT3']
    ts = []
    for fname in tqdm(filenames, desc="Reading tables with indices"):
        table = Table.read(fname)
        t = Table()
        t["BIN"] = [fname.split("_")[2]]
        names = [_ for _ in table["name"]]
        for i in indnames:
            t[i] =  table[licktype].data[names.index(i)]
            t["{}err".format(i)] =  table["{}err".format(licktype,)].data[
                names.index(i)]
        ts.append(t)
    t = vstack(ts)
    return t

def make_table_bsf(filenames):
    """ Join tables with Lick indices from models. """
    ts = []
    for fname in tqdm(filenames, desc="Reading bsf tables:"):
        ts.append(Table.read(fname))
    t = vstack(ts)
    return t

def lick_profiles(table_obs, table_models, outimg, indnames=None,
                  figsize=(7.24, 5)):
    if indnames is None:
        indnames = ['bTiO_muse', 'H_beta', 'Fe5015', 'Mg_1', 'Mg_2', 'Mg_b',
                    'Fe5270', 'Fe5335', 'Fe5406', 'Fe5709', 'Fe5782', 'aTiO',
                    'Na_D', 'TiO_1', 'TiO_2_muse', 'CaH_1',
                    'CaH_2_muse', 'TiO_3', 'TiO_4', 'NaI', 'CaT1', 'CaT2',
                    'CaT3']
    temptable = templates_table()
    gs = gridspec.GridSpec(6, 4, hspace=0.04, left=0.05, right=0.99, top=0.995,
                           wspace=0.27, bottom=0.065)
    fig = plt.figure(figsize=figsize)
    ylabels = [_.replace("_", "").replace("muse", "*") for _ in indnames]
    for i, index in enumerate(indnames):
        ax = plt.subplot(gs[i])
        ax.errorbar(table_obs["R"], table_obs[index],
                    yerr=table_obs["{}err".format(
            index)], fmt="o", ecolor="C0", mec="w", mew=0.5, c="C0",
                    elinewidth=0.5, ms=4, label="NGC 3311")
        yerr = [table_models["{}_lowerr".format(index)].data,
                table_models["{}_upper".format(index)].data]
        ax.errorbar(table_models["R"], table_models[index],
                    yerr=yerr, fmt="o", ecolor="C1", mec="w", mew=0.5,
                    c="C1", elinewidth=0.5, ms=4, label="SSP Models")
        ax.set_ylabel("{} (\\r{{A}})".format(ylabels[i]))
        ax.axhline(temptable[index].min(), c="k", ls="--", lw=0.5)
        ax.axhline(temptable[index].max(), c="k", ls="--", lw=0.5,
                   label="Model limits")
        if i > 18:
            ax.set_xlabel("R (kpc)")
        else:
            ax.xaxis.set_ticklabels([])
    plt.legend(loc=(1.2, -0.2), prop={"size": 9})
    for ext in ["png", "pdf"]:
        plt.savefig("{}.{}".format(outimg, ext), dpi=250)
    plt.close()
    return

def templates_table(w1=4500, w2=10000, sigma=315, licktype="Ia", velscale=None,
               sample="all", indnames=None):
    velscale = int(context.velscale) if velscale is None else velscale
    if indnames is None:
        indnames = ['bTiO_muse', 'H_beta', 'Fe5015', 'Mg_1', 'Mg_2', 'Mg_b',
                    'Fe5270', 'Fe5335', 'Fe5406', 'Fe5709', 'Fe5782', 'aTiO',
                    'Na_D', 'TiO_1', 'TiO_2_muse', 'CaH_1',
                    'CaH_2_muse', 'TiO_3', 'TiO_4', 'NaI', 'CaT1', 'CaT2',
                    'CaT3']
    templates_file = os.path.join(context.home, "templates",
                                  "lick_vel{}_w{}_{}_{}_sig{}_{}.fits".format(
                                      velscale, w1, w2, sample, sigma,
                                      licktype))
    temptable = Table.read(templates_file)
    return temptable

def run_ngc3311(targetSN=250, licktype=None, sigma=315, redo=False,
                loglike=None):
    licktype = "Ia" if licktype is None else licktype
    loglike = "normal2" if loglike is None else loglike
    imgname, cubename = context.get_field_files("fieldA")
    wdir = os.path.join(os.path.split(cubename)[0], "sn{}".format(targetSN))
    geom = Table.read(os.path.join(wdir, "geom.fits"))
    # Make table with Lick measurements
    lick_dir = os.path.join(wdir, "lick")
    lick_table = os.path.join(wdir, "lick_{}.fits".format(licktype))
    filenames = sorted([os.path.join(lick_dir, _) for _ in os.listdir(
                        lick_dir) if _.endswith("sigma{}.fits".format(sigma))])
    if os.path.exists(lick_table) and not redo:
        tlick = Table.read(lick_table, format="fits")
    else:
        tlick = make_table_obs(filenames, licktype=licktype)
        tlick = join(geom, tlick, "BIN")
        tlick.write(lick_table, overwrite=True)
        tlick.write(lick_table, format="fits", overwrite=True)
    # Reading table with predictions of the models
    bsf_dir = os.path.join(wdir, "bsf_lick_{}".format(loglike))
    filenames = sorted([os.path.join(bsf_dir, _) for _ in os.listdir(
                        bsf_dir) if _.endswith("sigma{}_lick.fits".format(
                        sigma))])
    bsf_table = os.path.join(wdir, "bsf_lick_{}.fits".format(licktype))
    if os.path.exists(bsf_table) and not redo:
        blick = Table.read(bsf_table, format="fits")
    else:
        blick = make_table_bsf(filenames)
        blick = join(geom, blick, "BIN")
        blick.write(bsf_table, overwrite=True)
        blick.write(bsf_table, format="fits", overwrite=True)
    outimg = os.path.join(wdir, "plots/lick_profiles")
    lick_profiles(tlick, blick, outimg)



if __name__ == "__main__":
    run_ngc3311(redo=False)