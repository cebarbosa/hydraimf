# -*- coding: utf-8 -*-
"""

Created on 24/10/18

Author : Carlos Eduardo Barbosa

ZAP does not allows ways of saving the models, so we have to work backwards
to obtain some models of the sky.

"""
import os

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy import signal
from scipy import sparse
from scipy.sparse.linalg import spsolve

import context
import misc

def calc_sky_model_residual(zap_table):
    outdir = os.path.join(os.path.join(context.data_dir), "MUSE/post-zap")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    zap_dir = os.path.join(context.data_dir, "MUSE/zap")
    for obs in zap_table:
        print(obs)
        output = os.path.join(outdir, obs["out_cube"])
        sci_cube = obs["sci_cube"]
        out_cube = os.path.join(zap_dir, obs["out_cube"])
        if not os.path.exists(sci_cube) or not os.path.exists(out_cube):
            continue
        if os.path.exists(output):
            continue
        sci = fits.getdata(sci_cube, hdu=1)
        out = fits.getdata(out_cube, hdu=1)
        sky = sci - out
        hdu = fits.PrimaryHDU(sky)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(output)


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).std(1)

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def make_sky_from_obs(nspec=2000, bin=10):
    zap_table = Table.read(os.path.join(context.data_dir,
                                  "MUSE/tables/zap_table.fits"))
    fields = [_.split("_")[1] for _ in zap_table["out_cube"]]
    idx = [i for i,f in enumerate(fields) if f=="FieldA"]
    zap_table = zap_table[idx]
    outdir = os.path.join(context.data_dir, "MUSE/sky")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sky_obs = []
    for obs in zap_table:
        skycube = obs["sky_cube"]
        name = obs["out_cube"]
        wave = misc.array_from_header(skycube)
        flat_file = os.path.join(outdir, name.replace(".", "_flat."))
        if not os.path.exists(flat_file):
            maskimg = os.path.join(context.data_dir, "MUSE/sky",
                      "skymask_{}".format(name.replace("NGC3311_", "")))
            mask = fits.getdata(maskimg)
            idx = np.where(mask==0)
            sky = fits.getdata(skycube, hdu=1)
            sky = sky[:, idx[0], idx[1]]
            hdulist = fits.HDUList(fits.PrimaryHDU(sky))
            hdulist.writeto(flat_file)
        else:
            sky = fits.getdata(flat_file)
        c = np.count_nonzero(np.isnan(sky), axis=0)
        sky = sky[:, c<10].T
        sky[np.isnan(sky)] = 0.
        idx = np.random.choice(sky.shape[0], nspec)
        sky = sky[idx,:]
        newshape = (int(nspec / bin), sky.shape[1])
        sky = rebin(sky, newshape)
        X = preprocessing.normalize(sky)
        mu = X.mean(0)
        z = baseline_als(mu, 1000000, 0.01)
        s = (mu - z)
        sky_obs.append(s/ np.max(s))
    return wave, np.array(sky_obs).sum(axis=0)


if __name__ == "__main__":
    wave1, sky1 = make_sky_from_obs()
    skymodel = Table.read(os.path.join(context.data_dir, "sky/skytable.fits"))
    wave2 = (skymodel["lam"] * u.nm).to(u.AA).value
    sky2 = skymodel["flux"].data
    sky2 /= sky2.max()
    plt.plot(wave1, sky1)
    plt.plot(wave2, sky2)
    plt.show()

