"""
Use a complete list of lines to check which lines can affect Lick indices

"""
import os

import numpy as np

import context

if __name__ == "__main__":
    emlist = os.path.join(context.tables_dir, "emission_line_list.csv")
    wem = np.loadtxt(emlist, usecols=(0,), delimiter=",")
    nem = np.genfromtxt(emlist, usecols=(1,), delimiter=",", dtype="str")
    nem = np.array([_.strip() for _ in nem])
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                "tables/spindex_CS.dat")
    bands = np.loadtxt(bandsfile, usecols=(3, 4, 1, 2, 5, 6))
    names = np.loadtxt(bandsfile, usecols=(8,), dtype=str)
    all_lines = []
    for name, band in zip(names, bands):
        w0, w1 = np.min(band), np.max(band)
        lines = np.where((wem >= w0) & (wem <= w1))[0]
        print(name, nem[lines], wem[lines])
        all_lines.append(lines)
    print(np.unique(np.hstack(all_lines)).shape)