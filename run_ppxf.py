# -*- coding: utf-8 -*-
"""
Forked in Hydra IMF from Hydra/MUSE on Feb 19, 2018

@author: Carlos Eduardo Barbosa

Run pPXF in data
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import simps
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy import constants
from specutils.io.read_fits import read_fits_spectrum1d

from ppxf.ppxf import ppxf, reddening_curve
import ppxf.ppxf_util as util

import context
from misc import array_from_header, snr


class pPXF():
    """ Class to read pPXF pkl files """
    def __init__(self, pp, velscale):
        self.__dict__ = pp.__dict__.copy()
        self.dw = 1.25 # Angstrom / pixel
        self.calc_arrays()
        self.calc_sn()
        return

    def calc_arrays(self):
        """ Calculate the different useful arrays."""
        # Slice matrix into components
        self.m_poly = self.matrix[:,:self.degree + 1]
        self.matrix = self.matrix[:,self.degree + 1:]
        self.m_ssps = self.matrix[:,:self.ntemplates]
        self.matrix = self.matrix[:,self.ntemplates:]
        self.m_gas = self.matrix[:,:self.ngas]
        self.matrix = self.matrix[:,self.ngas:]
        self.m_sky = self.matrix
        # Slice weights
        if hasattr(self, "polyweights"):
            self.w_poly = self.polyweights
            self.poly = self.m_poly.dot(self.w_poly)
        else:
            self.poly = np.zeros_like(self.galaxy)
        if hasattr(self, "mpolyweights"):
            x = np.linspace(-1, 1, len(self.galaxy))
            self.mpoly = np.polynomial.legendre.legval(x, np.append(1,
                                                       self.mpolyweights))
        else:
            self.mpoly = np.ones_like(self.galaxy)
        if self.reddening is not None:
            self.extinction = reddening_curve(self.lam, self.reddening)
        else:
            self.extinction = np.ones_like(self.galaxy)
        self.w_ssps = self.weights[:self.ntemplates]
        self.weights = self.weights[self.ntemplates:]
        self.w_gas = self.weights[:self.ngas]
        self.weights = self.weights[self.ngas:]
        self.w_sky = self.weights
        # Calculating components
        self.ssps = self.m_ssps.dot(self.w_ssps)
        self.gas = self.m_gas.dot(self.w_gas)
        self.bestsky = self.m_sky.dot(self.w_sky)
        return

    def calc_sn(self):
        """ Calculates the S/N ratio of a spectra. """
        self.signal = np.nanmedian(self.galaxy)
        self.meannoise = np.nanstd(sigma_clip(self.galaxy - self.bestfit,
                                              sigma=5))
        self.sn = self.signal / self.meannoise
        return

    def mc_errors(self, nsim=200):
        """ Calculate the errors using MC simulations"""
        errs = np.zeros((nsim, len(self.error)))
        for i in range(nsim):
            y = self.bestfit + np.random.normal(0, self.noise,
                                                len(self.galaxy))

            noise = np.ones_like(self.galaxy) * self.noise
            sim = ppxf(self.bestfit_unbroad, y, noise, velscale,
                       [0, self.sol[1]],
                       goodpixels=self.goodpixels, plot=False, moments=4,
                       degree=-1, mdegree=-1,
                       vsyst=self.vsyst, lam=self.lam, quiet=True, bias=0.)
            errs[i] = sim.sol
        median = np.ma.median(errs, axis=0)
        error = 1.4826 * np.ma.median(np.ma.abs(errs - median), axis=0)
        # Here I am using always the maximum error between the simulated
        # and the values given by pPXF.
        self.error = np.maximum(error, self.error)
        return

    def plot(self, output=None, fignumber=1):
        """ Plot pPXF run in a output file"""
        if self.ncomp > 1:
            sol = self.sol[0]
            error = self.error[0]
            sol2 = self.sol[1]
            error2 = self.error[1]
        else:
            sol = self.sol
            error = self.error
            sol2 = sol
            error2 = error
        plt.figure(fignumber, figsize=(5,4.2))
        plt.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5,1])
        ax = plt.subplot(gs[0])
        ax.minorticks_on()
        ax.plot(self.w, self.galaxy - self.bestsky, "-k", lw=2.,
                label="Data (S/N={0})".format(np.around(self.sn,1)))
        ax.plot(self.w, self.bestfit - self.bestsky, "-", lw=2., c="r",
                label="SSPs: V={0:.0f} km/s, $\sigma$={1:.0f} km/s".format(
                    sol[0], sol[1]))
        ax.xaxis.set_ticklabels([])
        if self.ncomp == 2:
            ax.plot(self.w, self.gas, "-b",
                    lw=1.,
                    label="Emission: V={0:.0f} km/s, "
                          "$\sigma$={1:.0f} km/s".format(sol2[0],sol2[1]))
        # if self.sky != None:
        #     ax.plot(self.w[self.goodpixels], self.bestsky[self.goodpixels], \
        #             "-", lw=1, c="g", label="Sky")
        ax.set_xlim(self.w[0], self.w[-1])
        ax.set_ylim(None, 1.1 * self.bestfit.max())
        leg = plt.legend(loc=4, prop={"size":10}, title=self.title,
                         frameon=False)
        # leg.get_frame().set_linewidth(0.0)
        plt.axhline(y=0, ls="--", c="k")
        plt.ylabel(r"Flux ($10^{-20}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$)",
                   size=12)
        ax1 = plt.subplot(gs[1])
        ax1.minorticks_on()
        ax1.set_xlim(self.w[0], self.w[-1])
        ax1.plot(self.w[self.goodpixels], (self.galaxy[self.goodpixels] - \
                 self.bestfit[self.goodpixels]), "-k",
                 label="$\chi^2=${0:.2f}".format(self.chi2))
        ax1.plot(self.w, self.noise, "--g")
        ax1.plot(self.w, -self.noise, "--g")
        leg2 = plt.legend(loc=2, prop={"size":8})
        ax1.axhline(y=0, ls="--", c="k")
        ax1.set_ylim(-8 * self.meannoise, 8 * self.meannoise)
        ax1.set_xlabel(r"$\lambda$ ($\AA$)", size=12)
        ax1.set_ylabel(r"$\Delta$Flux", size=12)
        gs.update(hspace=0.075, left=0.11, bottom=0.11, top=0.98, right=0.98)
        if output is not None:
            plt.savefig(output, dpi=250)
        return

def ppsave(pp, outroot="logs/out"):
    """ Produces output files for a ppxf object. """
    arrays = ["matrix", "w", "bestfit", "goodpixels", "galaxy", "noise"]
    delattr(pp, "star_rfft")
    delattr(pp, "star")
    hdus = []
    for i,att in enumerate(arrays):
        if i == 0:
            hdus.append(fits.PrimaryHDU(getattr(pp, att)))
        else:
            hdus.append(fits.ImageHDU(getattr(pp, att)))
        delattr(pp, att)
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(os.path.join(outroot, "{}.fits".format(pp.name)),
                    overwrite=True)
    with open(os.path.join(outroot, "{}.pkl".format(pp.name)) , "w") as f:
        pickle.dump(pp, f)
    return

def ppload(pp, path):
    """ Read ppxf arrays. """
    with open(os.path.join(path, "{}.pkl".format(pp.name))) as f:
        pp = pickle.load(f)
    arrays = ["matrix", "w", "bestfit", "goodpixels", "galaxy", "noise"]
    for i, item in enumerate(arrays):
        setattr(pp, item, fits.getdata(os.path.join(path, "{}.fits".format(
            pp.name)), i))
    return pp

def plot_all():
    """ Make plot of all fits. """
    nights = sorted(os.listdir(data_dir))
    for night in nights:
        print "Working in run ", night
        wdir = os.path.join(data_dir, night)
        os.chdir(wdir)
        fits = [x for x in os.listdir(".") if x.endswith(".fits")]
        skies =  [x for x in fits if x.startswith("sky")]
        specs = sorted([x for x in fits if x not in skies])
        for i,spec in enumerate(specs):
            print "Working on spec {0} ({1}/{2})".format(spec, i+1, len(specs))
            pp = ppload("logs/{0}".format(spec.replace(".fits", "")))
            pp = pPXF(pp, velscale)
            pp.plot("logs/{0}".format(spec.replace(".fits", ".png")))

def run_ppxf(fields, w1, w2, targetSN, tempfile,
             velscale=None, redo=False, ncomp=2, only_halo=False, bins=None,
             dataset=None, **kwargs):
    """ New function to run pPXF. """
    if velscale is None:
        velscale = context.velscale
    if dataset is None:
        dataset = "MUSE-DEEP"
    logwave_temp = array_from_header(tempfile, axis=1, extension=0)
    wave_temp = np.exp(logwave_temp)
    stars = fits.getdata(tempfile, 0)
    emission = fits.getdata(tempfile, 1)
    params = fits.getdata(tempfile, 2)
    ngas = len(emission)
    nstars = len(stars)
    ##########################################################################
    # Set components
    if ncomp == 1:
        templates = stars.T
        components = np.zeros(nstars, dtype=int)
        kwargs["component"] = components
    elif ncomp == 2:
        templates = np.column_stack((stars.T, emission.T))
        components = np.hstack((np.zeros(nstars), np.ones(ngas))).astype(int)
        kwargs["component"] = components
    ##########################################################################
    for field in fields:
        print "Working on Field {0}".format(field[-1])
        os.chdir(os.path.join(context.data_dir, dataset, field, "spec1d"))
        logdir = os.path.join(context.data_dir, dataset, field,
                              "ppxf_vel{}_w{}_{}_sn{}".format(int(velscale),
                               w1, w2, targetSN))
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        filenames = sorted(os.listdir("."))
        ######################################################################
        for i, fname in enumerate(filenames):
            name = fname.replace(".fits", "")
            if os.path.exists(os.path.join(logdir, fname)) and not redo:
                continue
            print("=" * 80)
            print("PPXF run {0}/{1}".format(i+1, len(filenames)))
            print("=" * 80)
            spec = read_fits_spectrum1d(fname)
            ###################################################################
            # Trim spectrum according to templates
            idx = np.argwhere(np.logical_and(
                              spec.wavelength.value > wave_temp[0],
                              spec.wavelength.value < wave_temp[-1]))
            wave = spec.wavelength[idx].T[0].value
            flux = spec.flux[idx].T[0].value
            ###################################################################
            signal, noise, sn = snr(flux)
            galaxy, logLam, vtemp = util.log_rebin([wave[0], wave[-1]],
                                                   flux, velscale=velscale)
            dv = (logwave_temp[0]-logLam[0]) * constants.c.to("km/s").value
            noise =  np.ones_like(galaxy) * noise
            kwargs["lam"] = wave
            ###################################################################
            # Masking bad pixels
            skylines = np.array([4785, 5577,5889, 6300, 6863])
            goodpixels = np.arange(len(wave))
            for line in skylines:
                sky = np.argwhere((wave < line - 15) | (wave > line +
                                                        15)).ravel()
                goodpixels = np.intersect1d(goodpixels, sky)
            kwargs["goodpixels"] = goodpixels
            ###################################################################
            kwargs["vsyst"] = dv
            # First fitting
            pp = ppxf(templates, galaxy, noise, velscale, **kwargs)
            title = "Field {0} Bin {1}".format(field[-1], bin)
            pp.name = name
            # Adding other things to the pp object
            pp.has_emission = True
            pp.dv = dv
            pp.w = np.exp(logLam)
            pp.velscale = velscale
            pp.ngas = ngas
            pp.ntemplates = nstars
            pp.templates = 0
            pp.name = name
            pp.title = title
            ppsave(pp, outroot=logdir)
    return


def make_table(fields, logdir):
    """ Make table with results. """
    head = ("{0:<30}{1:<14}{2:<14}{3:<14}{4:<14}{5:<14}{6:<14}{7:<14}"
             "{8:<14}{9:<14}{10:<14}{11:<14}{12:<14}\n".format("# FILE",
             "V", "dV", "S", "dS", "h3", "dh3", "h4", "dh4", "chi/DOF",
             "S/N (/ pixel)", "ADEGREE", "MDEGREE"))
    for field in fields:
        print "Producing summary for Field {0}".format(field[-1])
        os.chdir(os.path.join(data_dir, "combined_{0}".format(field),
                 logdir))
        output = os.path.join(data_dir, "combined_{0}".format(field),
                   logdir.replace("logs", "ppxf") + ".txt")
        fits = sorted([x for x in os.listdir(".") if x.endswith("fits")])
        results = []
        for i, fname in enumerate(fits):
            print " Processing pPXF solution {0} / {1}".format(i+1, len(fits))
            pp = ppload(fname.replace(".fits", ""))
            pp = pPXF(pp, velscale)
            sol = pp.sol if pp.ncomp == 1 else pp.sol[0]
            sol[0] += vhelio[field]
            error = pp.error if pp.ncomp == 1 else pp.error[0]
            line = np.zeros((sol.size + error.size,))
            line[0::2] = sol
            line[1::2] = error
            line = np.append(line, [pp.chi2, pp.sn])
            line = ["{0:12.3f}".format(x) for x in line]
            num = int(fname.replace(".fits", "").split("bin")[1])
            name = "{0}_bin{1:04d}".format(field, num)
            line = ["{0:18s}".format(name)] + line + \
                   ["{0:12}".format(pp.degree), "{0:12}".format(pp.mdegree)]
            results.append(" ".join(line) + "\n")
        results = sorted(results)
        # Append results to outfile
        with open(output, "w") as f:
            f.write(head)
            f.write("".join(results))

def run_stellar_populations(fields, targetSN, w1, w2,
                            sampling=None, velscale=None, redo=False,
                            dataset=None):
    """ Run pPXF on binned data using stellar population templates"""
    if sampling is None:
        sampling = "salpeter_regular"
    if velscale is None:
        velscale = context.velscale
    if dataset is None:
        dataset = "MUSE-DEEP"
    tempfile = os.path.join(context.home, "templates",
               "emiles_muse_vel{}_w{}_{}_{}.fits".format(int(velscale), w1, w2,
                                                    sampling))
    logdir = "ppxf_sn{0}_w{1}_{2}".format(targetSN, w1, w2)
    bounds = np.array([[[1800., 5800.], [3., 800.], [-0.3, 0.3], [-0.3, 0.3]],
                       [[1800., 5800.], [3., 80.], [-0.3, 0.3], [-0.3, 0.3]]])
    kwargs = {"start" :  np.array([[3800, 50, 0., 0.], [3800, 50., 0, 0]]),
              "plot" : False, "moments" : [4, 4], "degree" : 12,
              "mdegree" : 0, "reddening" : None, "clean" : False,
              "bounds" : bounds}

    run_ppxf(fields, w1, w2, targetSN, tempfile, redo=redo,
             ncomp=2, **kwargs)
    make_table(fields, logdir)
    # linenames= ["HBeta_4861.3", "[OIII]_4958.9", "[OIII]_5006.8",
    #             "HAlpha_6564.6", "[NI]_5200.2", "[NII]_6585.2", "[NII]_6549.8",
    #             "[SII]_6718.2", "[SII]_6732.6"]
    # make_table_emission(fields, targetSN, logdir, w1=w1, w2=w2, ncomp=1,
    #                     lines=linenames)
    return

def make_table_emission(fields, targetSN, logdir, w1=4500, w2=5500, ncomp=1,
                        lines=None):
    """ Make a table with properties of the emission lines. """
    ######################################################################
    # Preparing header
    head = ["#Spec", "V", "Verr", "Sigma", "Sigerr", "E(B-V)"]
    snstr = len(lines) * ["A/N"]
    ls =  [item for pair in zip(lines,snstr) for item in pair]
    head += ls
    head = ["{0:15s}".format(x) for x in head]
    head[0] = "{0:20s}".format(head[0])
    head = "".join(head) + "\n"
    lines = np.array([4861.3, 4958.9, 5006.8, 6564.6, 5200.2, 6585.2,
                      6549.8, 6718.2, 6732.6])
    kccm89 = k_CCM89(lines)
    #######################################################################
    for field in fields:
        os.chdir(os.path.join(data_dir, "combined_{0}".format(field), logdir))
        output = os.path.join(data_dir, "combined_{0}".format(field), \
                "ppxf_emission_sn{0}_w{1}_{2}.txt".format(targetSN, w1, w2))
        fits = sorted([x for x in os.listdir(".") if x.endswith("fits")])
        results = []
        for i, fname in enumerate(fits):
            print "{2} ({0} / {1})".format(i+1, len(fits), fname)
            pp = ppload(fname.replace(".fits", ""))
            pp = pPXF(pp, velscale)
            line = [pp.sol[ncomp][0], pp.error[ncomp][0], pp.sol[ncomp][1],
                    pp.error[ncomp][1]]
            lfluxes = np.zeros_like(lines)
            ans = np.zeros_like(lines)
            for j in np.arange(pp.ngas):
                flux = pp.m_gas[:,j] * pp.w_gas[j]
                lfluxes[j] = simps(flux, pp.w)
                ans[j] = flux.max() / pp.noise
            ebv = 1.97 * np.log10(lfluxes[3] / lfluxes[0] / 2.86)
            ebv = ebv if np.isfinite(ebv) else 0
            lfluxes *= np.power(10, 0.4 * kccm89 * ebv)
            sflux = ["{0:10.5g} {1:10.3g}".format(x,y) for x,y in zip(
                     lfluxes,ans)]
            line = ["{0:15.5f}".format(x) for x in line] + \
                   ["{0:10.2g}".format(ebv)] + sflux
            num = int(fname.replace(".fits", "").split("bin")[1])
            name = "{0}_bin{1:04d}".format(field, num)
            line = ["{0:20s}".format(name)] + line
            results.append(" ".join(line) + "\n")
        results = sorted(results)
        # Write results to outfile
        with open(output, "w") as f:
            f.write(head)
            f.write("".join(results))
    return

if __name__ == '__main__':
    targetSN = 70
    w1 = 4500
    w2 = 5500
    ##########################################################################
    # Running stellar populations
    run_stellar_populations(context.fields, targetSN, w1, w2, redo = True)