# -*- coding: utf-8 -*-
"""

Created on 28/10/2017

@author: Carlos Eduardo Barbosa

Detection of sources in data and separation of bins prior to Voronoi
tesselation

"""
from __future__ import division, print_function
import os

import pyregion
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.table import Table
import matplotlib.pyplot as plt

import sewpy

import context
from misc import array_from_header

def make_unsharp_mask(img, redo=False, output=None):
    """ Produces unsharp mask of a given image. """
    if output is None:
        output = "unsharp_mask.fits"
    if os.path.exists(output) and not redo:
        return output
    kernel = Gaussian2DKernel(5)
    data = fits.getdata(img)
    smooth = convolve(data, kernel, normalize_kernel=False)
    unsharp_mask = data - smooth
    fits.writeto(output, unsharp_mask, overwrite=True)
    return output

def mask_regions(img, redo=False):
    """ Mask regions marked in file mask.reg made in ds9. """
    filename = "mask.reg"
    outfile = "masked.fits"
    if not os.path.exists(filename):
        return img
    if os.path.exists(outfile) and not redo:
        return outfile
    r = pyregion.open(filename)
    data = fits.getdata(img)
    for i, region in enumerate(r.get_filter()):
        mask = region.mask(data.shape)
        data[mask] = np.nan
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(outfile, overwrite=True)
    return outfile

def run_sextractor(img, redo=False, outfile=None, segmentation_file=None,
                   save=False):
    """ Produces a catalogue of sources in a given field. """
    if outfile is None:
        outfile = "sexcat.fits"
    if segmentation_file is None:
        segmentation_file = "segmentation.fits"
    if os.path.exists(outfile) and not redo:
        return outfile
    params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "KRON_RADIUS", "ELLIPTICITY",
               "THETA_IMAGE", "A_IMAGE", "B_IMAGE", "MAG_AUTO"]
    config = {"CHECKIMAGE_TYPE": "SEGMENTATION",
                            "CHECKIMAGE_NAME": segmentation_file,
                            "DETECT_THRESH" : 1.5}
    sew = sewpy.SEW(config=config, sexpath="sextractor", params=params)
    cat = sew(img)
    cat["table"].write(outfile, format="fits", overwrite=True)
    return outfile

def mask_sources(img, cat, ignore=None, redo=False, output=None):
    """ Produces segmentation image with bins for detected sources using
    elliptical regions. """
    if output is None:
        output = "halo_only.fits"
    if os.path.exists(output) and not redo:
        return output
    data = fits.getdata(img)
    ydim, xdim = data.shape
    xx, yy = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))
    table = Table.read(cat, 1)
    if ignore is not None:
        idx = np.array([i for i,x in enumerate(table["NUMBER"]) if x not in
                        ignore])
        table = table[idx]
    axratio = table["B_IMAGE"] / table["A_IMAGE"]
    table = table[axratio > 0.4]
    segmentation = np.zeros_like(data)
    for source in table:
        R = calc_isophotes(xx, yy, source["X_IMAGE"], source["Y_IMAGE"], \
                           source["THETA_IMAGE"] - 90, source["B_IMAGE"] /
                           source["A_IMAGE"])
        Rmax = source["A_IMAGE"] * source["KRON_RADIUS"]
        segmentation += np.where(R <= Rmax, source["NUMBER"], 0.)
    d = np.copy(data)
    d[segmentation!=0] = np.nan
    hdu = fits.PrimaryHDU(d)
    hdu.writeto(output, overwrite=True)
    return output

def calc_isophotes(x, y, x0, y0, PA, q):
    """ Calculate isophotes for a given component. """
    x = np.copy(x) - x0
    y = np.copy(y) - y0
    shape = x.shape
    theta = np.radians(PA)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[s, c], [-c, s]])
    xy = np.dot(np.column_stack((x.flatten(), y.flatten())), rot).T
    x = np.reshape(xy[0], newshape=shape)
    y = np.reshape(xy[1], newshape=shape)
    return np.sqrt(np.power(x, 2) + np.power(y / q, 2))

def ignore_sources(field):
    if field == "fieldA":
        return [106, 107]
    else:
        return []

def simple_binning(img, field=None):
    """ Includes additional bins to halo before Voronoi. """
    outfile = "simple_binning.fits"
    data = fits.getdata(img)
    binning = np.zeros_like(data)
    # Adding sources if necessary
    newregions = "newsources.reg"
    if os.path.exists(newregions):
        r = pyregion.open(newregions)
        for i, region in enumerate(r.get_filter()):
            mask = region.mask(data.shape)
            binning[mask] = i + 1
    # Include radial bins in fields C and D
    if field in ["fieldC", "fieldD"]:
        refcube = context.get_field_files(field)[1]
        ra = array_from_header(refcube, axis=1)
        dec = array_from_header(refcube, axis=2)
        # Ofset to the center of NGC 3311
        ra -= context.ra0
        dec -= context.dec0
        # Convert to radians
        X = context.D * 1000 * np.deg2rad(ra)
        Y = context.D * 1000 * np.deg2rad(dec)
        xx, yy = np.meshgrid(X,Y)
        R = np.sqrt(xx**2 + yy**2)
        Rbins = 10 + 35 * np.logspace(0.3,1,4, base=10) / 10
        Rbins = np.hstack((10, Rbins))
        for i,rbin in enumerate(Rbins[:-1]):
            deltar = Rbins[i+1] - Rbins[i]
            newbin = binning.max() + 1
            idxbin = np.where((R > rbin) & (R <= rbin + deltar) & (binning==0))
            if i == 3:
                newbin = 0
            binning[idxbin] = newbin
    binning[np.isnan(data)] = np.nan
    hdu = fits.PrimaryHDU(binning)
    hdu.writeto(outfile, overwrite=True)
    return

def run_ngc3311():
    dataset = "MUSE"
    data_dir = os.path.join(context.data_dir, dataset, "combined")
    fields = ["fieldB"]
    for field in fields:
        f = field.replace("f", "F")
        os.chdir(os.path.join(data_dir, field))
        cubename = "NGC3311_{}_DATACUBE_COMBINED.fits".format(f)
        imgname = "NGC3311_{}_IMAGE_COMBINED.fits".format(f)
        imgum = make_unsharp_mask(imgname, redo=False)
        immasked = mask_regions(imgum, redo=True)
        sexcat = run_sextractor(immasked, redo=True)
        ignore = ignore_sources(field)
        imhalo = mask_sources(immasked, sexcat, ignore=ignore, redo=False)
        simple_binning(imhalo, field=field)

def run_m87():
    imgname, cubename = context.get_img_cube_m87()
    wdir = os.path.split(imgname)[0]
    os.chdir(wdir)
    imgum = make_unsharp_mask(imgname, redo=False)
    immasked = mask_regions(imgum, redo=False)
    sexcat = run_sextractor(immasked, redo=False)
    ignore = [86, 67, 70, 73, 79, 68, 1, 66]
    imhalo = mask_sources(immasked, sexcat, ignore=ignore, redo=True)
    simple_binning(imhalo)


if __name__ == "__main__":
    run_ngc3311()
    # run_m87()
