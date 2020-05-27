"""
Plot spectra.
"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

import context

def plot_ngc3311(field=None, targetSN=250):
    """ Plot spectra of field A of NGC 3311"""
    # Reading indices file
    indnames = ['bTiO_muse', 'H_beta', 'Fe5015', 'Mg_1', 'Mg_2', 'Mg_b',
                'Fe5270', 'Fe5335', 'Fe5406', 'Fe5709', 'Fe5782', 'aTiO',
                'Na_D', 'TiO_1', 'TiO_2_muse', 'CaH_1',
                'CaH_2_muse', 'TiO_3', 'TiO_4', 'NaI', 'CaT1', 'CaT2',
                'CaT3']
    ylabels = [_.replace("_", "").replace("muse", "*") for _ in indnames]
    bandsfile = os.path.join(os.path.split(os.path.abspath(
                             context.__file__))[0],
                             "tables/spindex_CS.dat")
    names = np.loadtxt(bandsfile, usecols=(8,), dtype=str).tolist()
    idx = [names.index(index) for index in indnames]
    bandsz0 = np.loadtxt(bandsfile, usecols=(3,4,1,2,5,6))[idx]
    units_bin = np.loadtxt(bandsfile, usecols=(7,))[idx]
    units = np.where(units_bin, u.Unit("mag"), u.Unit("angstrom"))
    # Reading spectra
    field = "fieldA" if field is None else field
    imgname, cubename = context.get_field_files(field)
    wdir = os.path.join(os.path.split(cubename)[0], "sn{}".format(targetSN))
    outdir = os.path.join(wdir, "plots", "spec_with_indices")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Getting velocity of spectra
    ktable = Table.read(os.path.join(wdir,
                                  "ppxf_vel50_w4500_10000_kinematics.fits"))
    for spec in tqdm(ktable, desc="Producing figures:"):
        vel = spec["V"]
        c = const.c.to("km/s").value
        dwave = np.sqrt((1 + vel/ c) / (1 - vel/ c))
        data = Table.read(os.path.join(wdir, "ppxf_vel50_w4500_10000_kinematics",
                                       "{}_bestfit.fits".format(spec["spec"])))
        wave = data["lam"]
        norm = 1e19
        flux = (data["galaxy"] - data["gas_bestfit"]) * norm
        fluxerr = data["noise"] * norm
        bestfit = (data["bestfit"] - data["gas_bestfit"]) * norm
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(5, 6)
        ax0 = plt.subplot(gs[0,:])
        ax0.plot(wave, flux)
        ax0.plot(wave, bestfit)
        ax0.set_ylabel("$f_{\lambda}$")
        ax0.set_xlabel("$\lambda$ (\\r{A})")
        for i,w0 in enumerate(bandsz0):
            w = w0 * dwave
            axn = plt.subplot(gs[i+6])
            dw = 20
            idx = np.where((wave >= w[0] - dw) & (wave <= w[5] + dw))
            axn.plot(wave[idx], flux[idx])
            axn.plot(wave[idx], bestfit[idx])
            axn.fill_between(wave[idx], flux[idx] - fluxerr[idx], flux[idx] +
                             fluxerr[idx])
            axn.set_xlabel("$\lambda$ (\\r{A})")
            axn.set_ylabel("$f_{\lambda}$")
            for ax in [ax0, axn]:
                ax.axvspan(w[0], w[1], alpha=.3, color="b")
                ax.axvspan(w[2], w[3], alpha=.3, color="g")
                ax.axvspan(w[4], w[5], alpha=.3, color="r")
            axn.set_title(ylabels[i])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "{}.png".format(spec["spec"])),
                    dpi=250)
        plt.close()
if __name__ == "__main__":
    plot_ngc3311()
