"""
Testing what models with high alpha/Fe look like in practice
"""

import os

import numpy as np
from astropy.table import Table
import astropy.units as u
import astropy.constants as const
from spectres import spectres
import ppxf.ppxf_util as util
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

import paintbox
import context
from run_paintbox import build_sed_model

if __name__ == "__main__":
    velscale = 200
    data_dir = os.path.join(context.data_dir, "MUSE/voronoi/sn250")
    wdir = os.path.join(data_dir, "sci")
    outfile = os.path.join(data_dir, "plots/MgTiCa.pdf")
    table = Table.read(os.path.join(data_dir, "results.fits"))
    # Read first spectrum to set the dispersion
    specnames = sorted([_ for _ in sorted(os.listdir(wdir)) if _.endswith(
                         ".fits")])
    specnames = ['fieldA_sn250_0002.fits', "fieldA_sn250_0045.fits",
                 "fieldA_sn250_0065.fits"]
    data = Table.read(os.path.join(wdir, specnames[0]))
    wave_lin = data["wave"].data
    flam = data["flam"].data
    _, logwave, velscale = util.log_rebin([wave_lin[0], wave_lin[-1]], flam,
                                    velscale=velscale)
    wave = np.exp(logwave)[1:-1]
    # Masking wavelengths with sky lines/ residuals
    skylines = np.array([4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                         5895, 5888, 5990, 5895, 6300, 6363, 6386, 6562,
                         6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                         8747, 8757, 8767, 8777, 8787, 8797, 8827, 8836, 8919,
                         9310])
    dlam = 2
    skylines.sort()
    goodpixels = np.arange(len(wave))
    for line in skylines:
        sky = np.argwhere((wave <= line - dlam) | (wave >= line + dlam)).ravel()
        goodpixels = np.intersect1d(goodpixels, sky)
    # wave = wave[goodpixels]
    print("Interpoting model...")
    sed = build_sed_model(wave, sample="test")[0]
    mask = np.array([i for i in range(len(wave)) if i not in goodpixels])
    print("Done!")
    idxalpha = 3
    cmap = cm.get_cmap('Spectral')
    bands0 = np.array([[5069.125, 5366.125],
                       [5936.625, 6420.125],
                       [8450.000, 8590.000]])
    z = 0.012759
    # intervals = bands0 * (1 + 0.012759)
    intervals = np.array([[5150, 5350], [6000, 6400], [8450, 8750]])
    widths = np.diff(intervals, axis=1)
    fconst = 1e-19
    features = ["Mg", "Ti", "Ca"]
    # Lick indices shade
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                "tables/spindex_CS.dat")
    bandsz0 = np.loadtxt(bandsfile, usecols=(3,4,1,2,5,6))
    bands = bandsz0 * (1 + z)
    ibands = np.array([12, 20, 21, 26, 28, 29, 30])
    bands = bands[ibands, :]
    units_bin = np.loadtxt(bandsfile, usecols=(7,))
    units = np.where(units_bin, u.Unit("mag"), u.Unit("angstrom"))
    names = np.loadtxt(bandsfile, usecols=(8,), dtype=str)
    names = [_.replace("_", "\_") for _ in names]
    print(np.array(names)[ibands])
    # print(names)
    # input()
    # Starting plot
    fig = plt.figure(figsize=(7.245, 4.5),
                     constrained_layout=False)
    gs = fig.add_gridspec(ncols=len(widths), nrows=3, width_ratios=widths,
                          hspace=0.05, wspace=0.15, right=0.9, left=0.06,
                          top=0.99, bottom=0.07)
    ipol = np.array([i for i,p in enumerate(sed.parnames) if p.startswith("p")])
    ############################################################################
    for i, spec in enumerate(specnames):
        print(spec)
        name = spec.replace("_sn250", "").split(".")[0]
        idx = np.where(table["BIN"]==name)[0][0]
        R = "{:.1f}".format(table["R"][idx])
        t = table[idx]
        sci_file = os.path.join(wdir, spec)
        data = Table.read(sci_file)
        flam = data["flam"].data
        flamerr = data["flamerr"].data
        flam, flamerr = spectres(wave, wave_lin, flam, spec_errs=flamerr)
        norm = np.median(flam)
        flam /= norm
        flamerr /= norm
        theta = np.array([t[p] for p in sed.parnames])
        # Data
        tsky = np.copy(theta)
        tsky[ipol] = 0
        sky = sed(tsky)
        spec = norm * (flam - sky) / fconst
        y1 = norm * (flam - flamerr - sky) / fconst
        y2 = norm * (flam + flamerr  - sky) / fconst
        aFe = theta[3]
        alphaFes = np.linspace(0, 0.4, 20)
        ys = np.zeros((len(alphaFes), len(flam)))
        for k, alphaFe in enumerate(alphaFes):
            theta[3] = alphaFe
            model = sed(theta)
            ratio = flam / model
            z = np.polyfit(wave, ratio, 30)
            p = np.poly1d(z)
            res = flam
            ys[k] = model * p(wave) - sky
        for j, (w1, w2) in enumerate(intervals):
            idx = np.where((wave >= w1 - 10) & (wave <= w2 + 10))[0]

            ax = fig.add_subplot(gs[i, j])
            ax.set_xlim(w1, w2)
            data_label = "Spec {} R={} kpc [$\\alpha$/Fe]={:.2f}".format(
                                 name.split("_")[1], R, aFe)
            ax.fill_between(wave[idx], y1[idx], y2[idx], color="0.9")
            for k, y in enumerate(ys):
                plt.plot(wave[idx], norm * y[idx] / fconst,
                         c=cmap(k / (len(alphaFes)-1)), lw=1)
            ax.plot(wave[idx], spec[idx], "k-", label=data_label, lw=0.5)
            if j == 1:
                plt.legend(loc=1)
            if i < 2:
                ax.xaxis.set_ticklabels([])
            ax.text(0.1, 0.1, features[j], transform=ax.transAxes)

            ylim = ax.get_ylim()
            if j == 1:
                ax.set_ylim(None, ylim[1] + 0.15 * (ylim[1] - ylim[0]))
            if j == 0:
                ax.set_ylim(None, ylim[1] - 0.1 * (ylim[1] - ylim[0]))
            for band in bands:
                ax.axvspan(band[2], band[3], color="lightcyan")
            for skyline in skylines:
                ax.axvspan(skyline - 3, skyline + 3, color="0.9")
    cax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    normalize = mpl.colors.Normalize(vmin=0, vmax=0.4)
    cb1 = mpl.colorbar.ColorbarBase(cax, norm=normalize,
                                    cmap=mpl.cm.get_cmap("Spectral"))
    cb1.set_label(r"[$\alpha$/Fe]")
    # hide tick and tick label of the big axis
    fig.text(0.5, 0.015, "$\lambda$ (\\r{A})", ha='center', fontsize=10)
    label = "$f_\lambda$ ($10^{-19}$ erg cm$^{-2}$ s$^{-1}$ \\r{A}$^{-1}$)"
    fig.text(0.006, 0.5, label, va='center', rotation='vertical', fontsize=10)
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
    #                 right=False)
    # plt.xlabel("$\lambda$ (\\r{A})")
    # plt.ylabel("$f_\lambda$")
    plt.savefig(outfile, dpi=250)
