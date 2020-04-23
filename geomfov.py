# -*- coding: utf-8 -*-
""" 

Created on 04/12/17

Author : Carlos Eduardo Barbosa

Routines related to the geometry of the FoV of the observations.

"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

import context
from misc import array_from_header

def calc_extent(image, extension=1):
    h = fits.getheader(image)
    ra = array_from_header(image, axis=1, extension=extension)
    dec = array_from_header(image, axis=2, extension=extension)
    # Ofset to the center of NGC 3311
    ra -= context.ra0.value
    dec -= context.dec0.value
    # Convert to radians
    X = context.D * 1000 * np.deg2rad(ra)
    Y = context.D * 1000 * np.deg2rad(dec)
    # Scale to the distance of the cluster
    extent = np.array([X[0], X[-1], Y[0], Y[-1]])
    return extent


def offset_extent(extent, field):
    if field == "fieldB":
        extent[:2] -= 1.
        extent[2:] -= 0.15
    if field == "fieldC":
        extent[:2] -=1.7
        extent[2:] -= 0.15
    if field == "fieldD":
        extent[:2] -= 1.5
    return extent

def calc_geom(binfile, imgfile):
    """Calculate the location of bins for a given target S/N and field. """
    binimg = fits.getdata(binfile)
    extent = calc_extent(imgfile)
    #  TODO: check if geometry is correct in MUSE-DEEP data set
    # extent = offset_extent(extent, field)
    ydim, xdim = binimg.shape
    x = np.linspace(extent[0], extent[1], xdim)
    y = np.linspace(extent[2], extent[3], ydim)
    xx, yy = np.meshgrid(x, y)
    binimg = np.ma.array(binimg, mask=~np.isfinite(binimg))
    bins = np.arange(binimg.min(), binimg.max()+1)
    xcen = np.zeros_like(bins)
    ycen = np.zeros_like(xcen)
    bins = bins.astype(int)
    for i, bin in enumerate(bins):
        idx = np.where(binimg == bin)
        xcen[i] = np.mean(xx[idx])
        ycen[i] = np.mean(yy[idx])
    radius = np.sqrt(xcen**2 + ycen**2)
    pa = np.rad2deg(np.arctan2(xcen, ycen))
    specs = np.array(["{0:04d}".format(_) for _ in bins])
    table = Table(data=[specs, xcen * u.kpc, ycen * u.kpc, radius * u.kpc,
                        pa * u.degree], names=["BIN", "X", "Y", "R", "PA"])
    return table

def get_geom(field, targetSN, dataset="MUSE"):
    """ Obtain table with geometric parameters given only field and bin S/N"""
    binfile = os.path.join(context.get_data_dir(dataset), field,
                           "sn{0}/voronoi2d_sn{0}.fits".format(targetSN))
    imgname, cubename = context.get_field_files(field)
    table = calc_geom(binfile, imgname)
    return table


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

if __name__ == "__main__":
    pass