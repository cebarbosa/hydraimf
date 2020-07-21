from __future__ import print_function, division
import os

import numpy as np
from astropy.table import Table, vstack
from spectres import spectres
from tqdm import tqdm

import context
import misc

def measure_sn(targetSN=250, dataset="MUSE"):
    wdir = os.path.join(context.data_dir, dataset,
                        "voronoi/sn{}/sci".format(targetSN))
    ts = []
    specs = sorted([_ for _ in os.listdir(wdir) if _.endswith(".fits")])
    optical = False
    for spec in tqdm(specs):
        tab = Table.read(os.path.join(wdir, spec))
        wave = tab["wave"]
        flux = tab["flam"]
        if optical:
            idx = np.where(wave < 7000)[0]
            wave = wave[idx]
            flux = flux[idx]
        newwave = np.arange(np.ceil(wave[0]), np.floor(wave[-1]))
        newflux = spectres(newwave, wave, flux)
        newsn = misc.snr(newflux)[2]
        t = Table([[spec.split(".")[0]], [newsn]], names=["spec", "SN/Ang"])
        ts.append(t)
    ts = vstack(ts)
    output = os.path.join(os.path.split(wdir)[0], "measured_sn.fits")
    if optical:
        output = output.replace(".fits", "_optical.fits")
    ts.write(output, format="fits", overwrite=True)

if __name__ == "__main__":
    measure_sn()