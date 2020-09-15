"""
Created on 24/07/20

Author : Carlos Eduardo Barbosa

Separates the skycalc model according to molecules.

"""
import os

import numpy as np
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from spectres import spectres
from astropy.io import fits

import context

def make_sky_templates():
    skydir = os.path.join(context.data_dir, "sky")
    table = Table.read(os.path.join(skydir, "skytable0.fits"))
    wave = (table["lam"].data * u.nm).to(u.AA).value
    emission = table["flux"].data
    # emission = emission[wave < 9300]
    # wave = wave[wave<9300]
    emission /= emission.max()
    emission[emission <= 0.01] = 0.
    ohtable = Table.read(os.path.join(skydir, 'oh_lines.fits'))
    a = ohtable["nu_"]
    b =  ohtable["nu__lc"]
    ab = ["{}-{}".format(x,y) for x,y in zip(a,b)]
    ohtable["ab"] = ab
    dw = 0.5
    idxs_used = []
    templates = []
    names = []
    # ohs = ["3-0", "4-0", "4-1", "5-0", "5-1", "6-0", "6-1", "6-2", "7-1", \
    #       "7-2", "7-3", "8-2", "8-3", "8-4", "9-2", "9-3", "9-4", "9-5"]
    ohs = ["3-0", "4-0", "4-1", "5-0", "5-1", "6-0", "6-1", "6-2", "7-1", \
          "7-2", "7-3", "8-3", "9-2", "9-3", "9-4", "9-5"]
    for i, tran in enumerate(ohs):
        idx = np.where(ohtable["ab"]==tran)[0]
        o = ohtable[idx]
        idxs=[]
        for line in o:
            lam = line["lambda"]
            l0 = lam - dw
            l1 = lam + dw
            idx = np.where((wave <= l1) & (wave >= l0))[0]
            idxs.append(idx)
        idxs = np.unique(np.hstack(idxs))
        idxs_used.append(idxs)
        if len(idxs) > 50:
            mask = np.zeros(len(wave))
            mask[idxs] = 1
            em = np.where(mask==1, emission, 0)
            templates.append(em)
            names.append("OH({})".format(tran))
    # Getting O_2
    o2table = Table.read(os.path.join(skydir, 'o2_lines.fits'))
    o2table = o2table[~np.isnan(o2table["lambda"])]
    idxs = []
    for line in o2table:
        lam = line["lambda"]
        l0 = lam - dw
        l1 = lam + dw
        idx = np.where((wave <= l1) & (wave >= l0))[0]
        idxs.append(idx)
    idxs = np.unique(np.hstack(idxs))
    idxs_used.append(idx)
    mask = np.zeros(len(wave))
    mask[idxs] = 1
    em = np.where(mask == 1, emission, 0)
    templates.append(em)
    names.append("O$_2$")
    # NaI lines
    add_NaI = False
    if add_NaI:
        idxs=[]
        dw=1
        for lam in [5889.99, 5895.59]:
            l0 = lam - dw
            l1 = lam + dw
            idx = np.where((wave <= l1) & (wave >= l0))[0]
            idxs.append(idx)
        idxs = np.unique(np.hstack(idxs))
        idxs_used.append(idx)
        mask = np.zeros(len(wave))
        mask[idxs] = 1
        em = np.where(mask == 1, emission, 0)
        templates.append(10 * em)
        names.append("NaI")
    # Processing remaining lines
    idxs_used = np.unique(np.hstack(idxs_used))
    add_other = False
    if add_other:
        mask = np.zeros(len(wave))
        mask[~idxs_used] = 1
        em_other = np.where(mask == 1, emission, 0)
        templates.append(em_other)
        names.append("Other")
        # dlams = [[6240, 6340], [6518, 6540], [7234, 7446], [7457, 7643],
        #          [7770, 7865], [7900, 8200], [8274, 8556], [8734, 9070]]
        # groups = ["OH(9-4)", "OH(6-1)", "OH(8-3)", "OH(5-0)", "OH(9-4)", "OH(5-1)",
        #          "OH(6-2)", "OH(7-3)"]
        # for dlam, group in zip(dlams, groups):
        #     idx = np.where((wave <= dlam[1]) & (wave >= dlam[0]))[0]
        #     mask = np.zeros(len(wave))
        #     mask[idx] = 1
        #     em = np.where(mask==1, em_other, 0)
        #     i = names.index(group)
        #     templates[i] += em
    # templates.append(em)
    # names.append("other")
    templates = np.array(templates)
    # Saving results
    FWHM = 29.5 # FWHM resolution in pixels, dw=0.1, fhwm_obs=2.95
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    templates = gaussian_filter1d(templates, sigma, axis=1)
    newwave = np.arange(4501, 9500)
    newtemplates = np.zeros((len(templates), len(newwave)))
    for i in range(len(templates)):
        newtemplates[i] = spectres(newwave, wave, templates[i])
    templates = newtemplates
    templates /= templates.max()
    wave = newwave
    w = Table([wave], names=["wave"])
    hdu0 = fits.PrimaryHDU(templates)
    hdu1 = fits.BinTableHDU(w)
    hdu2 = fits.BinTableHDU(Table([names], names=["skylines"]))
    hdulist = fits.HDUList([hdu0, hdu1, hdu2])
    hdulist.writeto(os.path.join(skydir, "sky_templates.fits"), overwrite=True)
    cm = plt.get_cmap('tab20')
    fig = plt.figure(figsize=(context.fig_width, 3.))
    ax = fig.add_subplot(111)
    for i, template in enumerate(templates):
        plt.plot(wave, template, label=names[i], c=cm((i+2)/20), lw=0.8)
    plt.legend(ncol=3, prop={"size":6}, frameon=False)
    plt.xlabel("$\lambda$ (\\r{A})")
    plt.ylabel("Flux (normalized)")
    plt.subplots_adjust(right=0.99, top=0.99, left=0.11)
    plt.xlim(5600, 9200)
    plt.savefig(os.path.join(skydir, "sky_templates.pdf"), dpi=300)
    plt.show()

if __name__ == "__main__":
    make_sky_templates()
