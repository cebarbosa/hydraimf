# -*- coding: utf-8 -*-
"""
@author: Carlos Eduardo Barbosa
Created on May 7, 2020
Produces table with all indices used in this project.
"""
from __future__ import print_function, division

import os

import numpy as np

if __name__ == "__main__":
    table = "tables/spindex_CS.dat"
    bands = np.loadtxt(table, usecols=(1,2,3,4,5,6))
    names = np.loadtxt(table, usecols=(8,), dtype=str)
    cenbands = ["{0[0]:.3f}-{0[1]:.3f}".format(band) for band in bands]
    redbands = ["{0[2]:.3f}-{0[3]:.3f}".format(band) for band in bands]
    bluebands = ["{0[4]:.3f}-{0[5]:.3f}".format(band) for band in bands]
    names = [n.replace("_", "").replace("muse", "*") for n in names]
    for name, cen, red, blue in zip(names, cenbands, bluebands, redbands):
        line = "{} & {} & {} & {} & \\\\".format(name, cen, blue, red)
        print(line)