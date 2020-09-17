# -*- coding: utf-8 -*-
"""

Created on 10/02/16
Adapted on 24/03/2018 to use in hydraimf project

@author: Carlos Eduardo Barbosa

"""


import os

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack, join, vstack
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colorbar import Colorbar
import scipy.ndimage as ndimage

import context
from geomfov import calc_extent, offset_extent, get_geom

class PlotVoronoiMaps():
    def __init__ (self, table, columns, outdir, labels=None, lims=None,
                  cmaps=None, cb_fmts=None, targetSN=70,
                  dataset="MUSE"):
        self.table = table
        self.columns = columns
        self.outdir = outdir
        self.labels = self.columns if labels is None else labels
        self.lims = lims
        if self.lims is None:
            self.lims = len(self.columns) * [[None, None]]
        self.cmaps = cmaps
        if self.cmaps is None:
            self.cmaps = len(self.columns) * [None]
        self.cb_fmts = cb_fmts
        if self.cb_fmts is None:
            self.cb_fmts = len(self.columns) * ["%i"]
        self.targetSN = targetSN
        self.table["field"] = [_.split("_")[0] for _ in self.table["BIN"]]
        self.fields = np.unique(self.table["field"])
        self.tables = []
        for field in np.unique(self.fields):
            idx = np.where(self.table["field"] == field)[0]
            self.tables.append(self.table[idx])
        self.dataset = dataset
        self.coords = SkyCoord(context.ra0, context.dec0)
        self.D = context.D * u.Mpc

    def plot(self, xylims=None, cbbox="regular", figsize=(2.8, 3),
             arrows=True, sigma=None, xloc=None, scale=1):
        """ Make the plots. """
        xloc = len(self.columns) * [-4] if xloc is None else xloc
        if xylims is None:
            xylims = [(25, -10), (-25, 20)]
        for j, col in enumerate(self.columns):
            print(col)
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1,1)
            gs.update(left=0.11, right=0.99, bottom=0.11, top=0.995)

            ax = plt.subplot(gs[0])
            ax.set_facecolor("0.85")
            plt.minorticks_on()
            self.make_contours(alpha=0.5)
            kmaps = []
            extents = []
            for i, (field, table) in enumerate(zip(self.fields, self.tables)):
                binsfile = os.path.join(context.get_data_dir(self.dataset),
                        field, "sn{0}/voronoi2d_sn{0}.fits".format(
                        self.targetSN))
                bins = np.array([float(_.split("_")[1]) for _ in table["BIN"]])
                vector = table[col].astype(np.float)
                image = context.get_field_files(field)[0]
                extent = calc_extent(image, self.coords, self.D)
                extent = offset_extent(extent, field)
                extents.append(extent)
                binimg = fits.getdata(binsfile)
                kmap = np.zeros_like(binimg)
                kmap[:] = np.nan
                for bin,v in zip(bins, vector):
                    if not np.isfinite(v):
                        continue
                    idx = np.where(binimg == bin)
                    kmap[idx] = v
                kmaps.append(kmap)

            automin = np.nanmin(self.table[col])
            automax = np.nanmax(self.table[col])
            vmin = self.lims[j][0] if self.lims[j][0] is not None else automin
            vmax = self.lims[j][1] if self.lims[j][1] is not None else automax
            for i in range(len(self.fields)):
                m = plt.imshow(kmaps[i], origin="bottom", cmap=self.cmaps[j],
                               vmin=vmin, vmax=vmax,
                               extent=extents[i],
                               aspect="equal", alpha=1)
            plt.xlim(*xylims[0])
            plt.ylim(*xylims[1])
            plt.xlabel("X (kpc)")
            plt.ylabel("Y (kpc)", labelpad=-3.5)
            if arrows:
                plt.arrow(-5, -20, 5, 0, linewidth=1.5, color="b", head_length=1,
                          head_width=0.5, alpha=1)
                plt.arrow(-5, -20, 0, 5, linewidth=1.5, color="b", head_length=1,
                          head_width=0.5, alpha=1)
                plt.text(1., -18, "E", color="b", fontsize=10, va='top')
                plt.text(-4.3, -12.5, "N", color="b", fontsize=10, va='top')
            ax.tick_params(axis="both",  which='major',
                           labelsize=context.MEDIUM_SIZE)
            if cbbox == "zoom":
                (x0, x1), (y0, y1) = xylims
                xsize = x1 - x0
                ysize = y1 - y0
                plt.gca().add_patch(Rectangle((x0 + 0.05 * xsize, y0 + 0.22 * ysize),
                                              0.25 * xsize, 0.48 * ysize,
                                              alpha=1, zorder=10, color="w"))
                self.draw_colorbar(fig, ax, m, orientation="vertical",
                                   cbar_pos=[0.24, 0.365, 0.05, 0.3],
                                   ticks=np.linspace(vmin, vmax, 5),
                                   cblabel=self.labels[j],
                                   cb_fmt=self.cb_fmts[j])
            elif cbbox == "regular":
                plt.gca().add_patch(Rectangle((14, -13), 9.5, 18, alpha=1,
                                              zorder=10, color="w"))
                self.draw_colorbar(fig, ax, m, orientation="vertical",
                                   cbar_pos=[0.25, 0.365, 0.05, 0.3],
                                   ticks=np.linspace(vmin, vmax, 5),
                                   cblabel=self.labels[j],
                                   cb_fmt=self.cb_fmts[j])
            elif cbbox == "horizontal":
                plt.gca().add_patch(Rectangle((2.25, -11.3), 9.2, 3.8, alpha=1,
                                              zorder=10, edgecolor="w",
                                              linewidth=1, facecolor="w"))
                self.draw_colorbar(fig, ax, m, orientation="horizontal",
                                   cbar_pos=[0.18, 0.18, 0.3, 0.05],
                                   ticks=np.linspace(vmin, vmax, 4),
                                   cblabel=self.labels[j],
                                   cb_fmt=self.cb_fmts[j])
                ax.text(xloc[j]+12.5, -8.4, self.labels[j], zorder=11)
            for fmt in ["pdf", "png"]:
                output = os.path.join(self.outdir,
                         "{}_sn{}.{}".format(col, self.targetSN, fmt))
                plt.savefig(output, dpi=250)
            plt.clf()
            plt.close()
        return

    def make_contours(self, plot_vband=False, nsigma=2, label=True, lw=0.8,
                      extent=None, fontsize=10.5, colors="k", contours=None,
                      vmax=25, vmin=20, alpha=0.5):
        vband = os.path.join(context.home, "images/hydra1.fits")
        vdata = fits.getdata(vband, verify=False)
        vdata = np.clip(vdata - 4900., 1., vdata)
        muv = -2.5 * np.log10(vdata/480./0.252/0.252) + 27.2
        if contours is None:
            contours = np.linspace(19.5,23.5,9)
        datasmooth = ndimage.gaussian_filter(muv, nsigma, order=0.)
        if extent is None:
            extent = np.array(calc_extent(vband, self.coords,
                                          self.D, extension=0))
            extent[:2] += 3
            extent[2:] -=0.2
        cs = plt.contour(datasmooth, contours, extent=extent,
                         colors=colors, linewidths=lw)
        if plot_vband:
            plt.imshow(muv, cmap="bone", extent=extent, vmax=vmax, vmin=vmin,
                       alpha=alpha, origin="bottom")
        if label:
            plt.clabel(cs, contours[0:-1:2], fmt="%d", fontsize=fontsize,
                       inline_spacing=-3,
                   manual=((-1.25, 1), (1.25, -4), (-6, 7)))
        return

    def draw_colorbar(self, fig, ax, coll, ticks=None, cblabel="",
                      cbar_pos=None, cb_fmt="%i", labelsize=context.MEDIUM_SIZE,
                      orientation="horizontal",
                      ylpos=0.5, xpos=-0.8, rotation=90):
        """ Draws the colorbar in a figure. """
        if cbar_pos is None:
            cbar_pos=[0.14, 0.13, 0.17, 0.04]
        cbaxes = fig.add_axes(cbar_pos)
        cbar = plt.colorbar(coll, cax=cbaxes, orientation=orientation,
                            format=cb_fmt)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=labelsize-1)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cl = plt.getp(cbar.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize=labelsize+2)
        # ax.text(xpos,ylpos,cblabel,rotation=rotation)
        return

def make_table(targetSN=250, dataset="MUSE", update=False):
    """ Produces a test plot using the geometry table. """
    wdir = os.path.join(context.data_dir, dataset, "voronoi",
                        "sn{}".format(targetSN))
    results_table = os.path.join(wdir, "results.fits")
    if not os.path.exists(results_table) or update:
        geomA = get_geom("fieldA", targetSN)
        geomB = get_geom("fieldB", targetSN)
        geomA["BIN"] = ["fieldA_{}".format(_) for _ in geomA["BIN"]]
        geomB["BIN"] = ["fieldB_{}".format(_) for _ in geomB["BIN"]]
        geom = vstack([geomA, geomB])
        sn_table = Table.read(os.path.join(wdir, "measured_sn.fits"))
        results = hstack([geom, sn_table])
        results.rename_column("SN/Ang", "SNR")
        m2l_table = Table.read(os.path.join(wdir, "mass2light.fits"))
        results = join(results, m2l_table, keys="BIN")
        # Adding stellar population table
        mcmc_dir = os.path.join(wdir, "EMCEE")
        tables = sorted([_ for _ in os.listdir(mcmc_dir) if _.endswith(
                "results.fits")])
        stpop = []
        for table in tables:
            binnum = "_".join(table.split("_")[0:4:2])
            tt = Table.read(os.path.join(mcmc_dir, table))
            newt = Table()
            newt["BIN"] = [binnum]
            for t in tt:
                newt[t["param"]] = [t["median"]]
                newt["{}_lerr".format(t["param"])] = [t["lerr"]]
                newt["{}_uerr".format(t["param"])] = [t["uerr"]]
            stpop.append(newt)
        stpop = vstack(stpop)
        results = join(results, stpop, keys="BIN")
        results.write(results_table, overwrite=True)
    else:
        results = Table.read(results_table)
    return results

def make_maps(results, targetSN=250, dataset="MUSE"):
    wdir = os.path.join(context.data_dir, dataset, "voronoi",
                        "sn{}".format(targetSN))
    outdir = os.path.join(wdir, "plots")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fields = ["SNR", "Z", "T", "imf", "alphaFe", "NaFe", "Av", "sigma",
              "M2L", "alpha", "logSigma"]
    labels = ["SNR (\\r{A}$^{-1}$)", "[Z/H]", "Age (Gyr)",
              "$\\Gamma_b$", r"[$\alpha$/Fe]", "[Na/Fe]", "$A_V$",
              "$\sigma_*$ (km/s)", "$M_*/L_r$",
              "$\\alpha$",
              "$\\log \\Sigma$ (M$_\\odot$ / kpc$^2$)"]
    cb_fmts = ["%i", "%.2f", "%i", "%.1f", "%.2f", "%.2f", "%.2f", "%i",
               "%.1f", "%.2f", "%.2f"]
    lims = [[None, None], [-.1, 0.2], [6, 14], [1.3, 2.3],
            [0.05, 0.20], [0.25, 0.5], [0, 0.05], [None, None], [None, None],
            [None, None], [None, None]]
    xloc = [-4, -4.5, -4, -5.0, -4.5, -4, -5, -3.5, -4.5, -5, -2.5]
    cmaps = ["viridis"] * len(xloc)
    idx = 0
    pvm = PlotVoronoiMaps(results, fields[idx:], outdir,
                          targetSN=targetSN, #lims=lims,
                          labels=labels[idx:], cb_fmts=cb_fmts[idx:],
                          cmaps=cmaps[idx:])

    xylims = [[12, -9.5], [-12, 12]]
    pvm.plot(xylims=xylims, arrows=False, cbbox="horizontal", xloc=xloc[idx:],
             scale=0.8)

if __name__ == "__main__":
    results = make_table(update=True)
    make_maps(results)