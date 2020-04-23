# -*- coding: utf-8 -*-
""" 

Created on 23/11/17

Author : Carlos Eduardo Barbosa

Project definitions

"""
import os
import platform

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.config import config
from dustmaps import sfd

def get_field_files(field, dataset="MUSE"):
    """ Returns the names of the image and cube associated with a given
    field. """
    if dataset == "MUSE-DEEP":
        wdir = os.path.join(home, "data/MUSE-DEEP", field)
        if field == "fieldA":
            img = "ADP.2017-03-27T12:49:43.628.fits"
            cube = "ADP.2017-03-27T12:49:43.627.fits"
        elif field == "fieldB":
            img = "ADP.2017-03-27T12:49:43.652.fits"
            cube = "ADP.2017-03-27T12:49:43.651.fits"
        elif field == "fieldC":
            img = "ADP.2017-03-27T12:49:43.644.fits"
            cube = "ADP.2017-03-27T12:49:43.643.fits"
        elif field == "fieldD":
            img = "ADP.2017-03-27T12:49:43.636.fits"
            cube = "ADP.2017-03-27T12:49:43.635.fits"
        return os.path.join(wdir, img), os.path.join(wdir, cube)
    elif dataset=="MUSE":
        wdir = os.path.join(home, "data/MUSE", "combined", field)
        img = os.path.join(wdir, "NGC3311_{}_IMAGE_COMBINED.fits".format(
            field.replace("f", "F")))
        cube = os.path.join(wdir, "NGC3311_{}_DATACUBE_COMBINED.fits".format(
            field.replace("f", "F")))
        return img, cube
    else:
        raise ValueError("Data set name not defined: {}".format(dataset))

def get_data_dir(dataset):
    if dataset == "MUSE-DEEP":
        return os.path.join(home, "data/MUSE-DEEP")
    elif dataset == "MUSE":
        return os.path.join(home, "data/MUSE/combined" )

# Emission lines used in the projects
def get_emission_lines():
    """ Returns dictionaries containing the emission lines to be used. """
    lines = (("Hbeta_4861", 4861.333), ("OIII_4959", 4958.91),
             ("OIII_5007", 5006.84), ("NII_6550", 6549.86),
             ("Halpha_6565", 6564.61), ("NII_6585", 6585.27),
             ("SII_6718", 6718.29), ("SII_6733", 6732.67))
    return lines

if platform.node() == "kadu-Inspiron-5557":
    home = "/home/kadu/Dropbox/hydraimf"
elif platform.node() in ["uv100", "alphacrucis"]:
    home = "/sto/home/cebarbosa/hydraimf"

data_dir = os.path.join(home, "data")

config['data_dir'] = os.path.join(data_dir, "dustmaps")
if not os.path.exists(config["data_dir"]): # Just to run once in my example
    sfd.fetch() # Specific for Schlafy and Finkbeiner (2011), which is an
    # updated version of the popular Schlegel, Finkbeiner & Davis (1998) maps

fields = ["fieldA", "fieldB", "fieldC", "fieldD"]

# Constants
D = 50.7 # Distance to the center of the Hydra I cluster in Mpc
DL = 55.5# Luminosity distance
velscale = 50. # Set velocity scale for pPXF related routines
V = 3800 # km/s
w1 = 4500
w2 = 10000

# Properties of the system
ra0 = 159.178471651 * u.degree
dec0 = -27.5281283035 * u.degree

# Get color excess
coords = SkyCoord(ra0, dec0)
sfq = sfd.SFDQuery()
ebv = sfq(coords)
Rv = 3.1  # Constant in our galaxy
Av = ebv * Rv

# VHELIO - radial velocities of the fields, have to be added from the
# observed velocities.
vhelio = {"fieldA" : 24.77, "fieldB" : 21.26, "fieldC" : 20.80,
          "fieldD" : 19.09} # km / s

# Matplotlib settings
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set tick width
width = 0.5
majsize = 4
minsize = 2
plt.rcParams['xtick.major.size'] = majsize
plt.rcParams['xtick.major.width'] = width
plt.rcParams['xtick.minor.size'] = minsize
plt.rcParams['xtick.minor.width'] = width
plt.rcParams['ytick.major.size'] = majsize
plt.rcParams['ytick.major.width'] = width
plt.rcParams['ytick.minor.size'] = minsize
plt.rcParams['ytick.minor.width'] = width
plt.rcParams['axes.linewidth'] = width

fig_width = 3.35 # inches

flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz