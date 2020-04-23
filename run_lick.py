# -*- coding: utf-8 -*-
""" 

Created on 15/03/18

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import yaml

import numpy as np
import astropy.units as u
from astropy.table import Table, hstack, vstack
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter1d
from spectres import spectres
from tqdm import tqdm

import context
from basket.lick.lick import Lick

import context

def run_lick(w1, w2, targetSN, dataset="MUSE", redo=False, velscale=None,
             nsim=200, sigma=None):
    """ Calculates Lick indices and uncertainties based on pPXF fitting. """
    velscale = context.velscale if velscale is None else velscale
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                "tables/spindex_CS.dat")
    bandsz0 = np.loadtxt(bandsfile, usecols=(3,4,1,2,5,6)) * u.AA
    units_bin = np.loadtxt(bandsfile, usecols=(7,))
    units = np.where(units_bin, u.Unit("mag"), u.Unit("angstrom"))
    names = np.loadtxt(bandsfile, usecols=(8,), dtype=str)
    sigma_str = "" if sigma is None else "_sigma{}".format(sigma)
    for field in context.fields:
        wdir = os.path.join(context.get_data_dir(dataset), field,
                            "sn{}".format(targetSN))
        ppxf_dir = os.path.join(wdir, "ppxf_vel{}_w{}_{}_kinematics".format(int(
            velscale), w1, w2))
        sci_dir = os.path.join(wdir, "sci")
        if not os.path.exists(ppxf_dir):
            continue
        yamls = sorted([_ for _ in os.listdir(ppxf_dir) if _.endswith(".yaml")])
        outdir = os.path.join(wdir, "lick")
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for j, fyaml in enumerate(yamls):
            output = os.path.join(outdir, fyaml.replace(".yaml", "_nsim{"
                                  "}{}.fits".format(nsim, sigma_str)))
            if os.path.exists(output) and not redo:
                continue
            print("Working with file {} ({}/{})".format(fyaml, j + 1,
                                                        len(yamls)))
            with open(os.path.join(ppxf_dir, fyaml)) as f:
                ppxf = yaml.load(f)
            losvd = np.array([ppxf["V_0"], ppxf["sigma_0"]])
            losvderr = np.array([ppxf["Verr_0"], ppxf["sigmaerr_0"]])
            ppxf_tab = Table.read(os.path.join(ppxf_dir,
                            fyaml.replace(".yaml", "_bestfit.fits")))
            wave_log = ppxf_tab["lam"]
            flux = ppxf_tab["galaxy"] - ppxf_tab["gas_bestfit"]
            fluxerr = ppxf_tab["noise"]
            bestfit = ppxf_tab["bestfit"] - ppxf_tab["gas_bestfit"]
            if sigma is not None:
                if losvd[1] > sigma:
                    continue
                sigma_diff = np.sqrt(sigma ** 2 - losvd[1] ** 2) / velscale
                flux = gaussian_filter1d(flux, sigma_diff, mode="constant",
                                         cval=0.0)
                fluxerr = gaussian_filter1d(fluxerr, sigma_diff,
                                            mode="constant", cval=0.0)
                bestfit = gaussian_filter1d(bestfit, sigma_diff,
                                            mode="constant", cval=0.0)
            # Reading original data
            sci_tab = Table.read(os.path.join(sci_dir,
                                              fyaml.replace(".yaml", ".fits")))
            wave = sci_tab["wave"]
            wave = wave[(wave > wave_log[1]) & (wave < wave_log[-2])]
            flux = spectres(wave, wave_log, flux)
            fluxerr = spectres(wave, wave_log, fluxerr)
            bestfit = spectres(wave, wave_log, bestfit)
            lick = Lick(wave, flux, bandsz0, vel=losvd[0] * u.km / u.s,
                        units=units)
            lick.classic_integration()
            L = lick.classic
            Ia = lick.Ia
            Im = lick.Im
            R = lick.R
            veldist = np.random.normal(losvd[0], losvderr[0], nsim)
            Lsim= np.zeros((nsim, len(bandsz0)))
            Iasim = np.zeros((nsim, len(bandsz0)))
            Imsim = np.zeros((nsim, len(bandsz0)))
            Rsim = np.zeros((nsim, len(bandsz0)))
            for i, vel in tqdm(enumerate(veldist), ascii=True,
                               desc="MC simulations"):
                lsim = Lick(wave, bestfit + np.random.normal(0, fluxerr),
                            bandsz0, vel=vel * u.km / u.s)
                lsim.classic_integration()
                Lsim[i] = lsim.classic
                Iasim[i] = lsim.Ia
                Imsim[i] = lsim.Im
                Rsim[i] = lsim.R
            Lerr = np.std(Lsim, axis=0)
            Iaerr = np.std(Iasim, axis=0)
            Imerr = np.std(Imsim, axis=0)
            Rerr = np.std(Rsim, axis=0)
            table = Table([names, L, Lerr, Ia, Iaerr, Im, Imerr, R, Rerr],
                          names=["name", "lick", "lickerr", "Ia",
                                 "Iaerr", "Im", "Imerr", "R", "Rerr"])
            table.write(output, format="fits", overwrite=True)

def run_lick_templates(w1, w2, sigma, velscale=None, licktype=None,
                       sample=None, redo=False):
    """ Determine indices in SSP templates. """
    velscale = int(context.velscale) if velscale is None else velscale
    licktype = "Ia" if licktype is None else licktype
    sample = "all" if sample is None else sample
    templates_dir = os.path.join(context.home, "templates")
    templates_file = os.path.join(templates_dir,
                 "emiles_muse_vel{}_w{}_{}_{}_fwhm2.95.fits".format(
                     velscale, w1, w2, sample))
    output = os.path.join(templates_dir,
                          "lick_vel{}_w{}_{}_{}_sig{}_{}.fits".format(
                           velscale, w1, w2, sample, sigma, licktype))
    if os.path.exists(output) and not redo:
        return
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                "tables/spindex_CS.dat")
    bandsz0 = np.loadtxt(bandsfile, usecols=(3,4,1,2,5,6)) * u.AA
    units_bin = np.loadtxt(bandsfile, usecols=(7,))
    units = np.where(units_bin, u.Unit("mag"), u.Unit("angstrom"))
    names = np.loadtxt(bandsfile, usecols=(8,), dtype=str)
    data = fits.getdata(templates_file, hdu=0)
    params = Table.read(templates_file, hdu=1)
    wave = np.exp(Table.read(templates_file, hdu=2)["loglam"]) * u.AA
    newdata = gaussian_filter1d(data, sigma / velscale, axis=1, mode="constant",
                                         cval=0.0)
    ts = []
    for param, spec in tqdm(zip(params, newdata), total=len(params)):
        lick = Lick(wave, spec, bandsz0, vel=0 * u.km / u.s,
                        units=units)
        lick.classic_integration()
        t = hstack([param, Table(getattr(lick, licktype), names=names)])
        ts.append(t)
    ts = vstack(ts)
    ts.write(output, format="fits", overwrite=True)

if __name__ == "__main__":
    targetSN = 250
    w1 = 4500
    w2 = 10000
    sigma= 315
    run_lick(w1, w2, targetSN, nsim=200, sigma=sigma, redo=False)
    run_lick_templates(w1, w2, sigma, redo=True, sample="test")
