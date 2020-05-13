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
from astropy.table import Table, hstack, join
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colorbar import Colorbar
import scipy.ndimage as ndimage

import context
from geomfov import calc_extent, offset_extent, get_geom

class PlotVoronoiMaps():
    def __init__ (self, tables, columns, outdir, labels=None, lims=None,
                  cmaps=None, cb_fmts=None, targetSN=70, fields=None,
                  dataset="MUSE"):
        self.tables = tables
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
        self.fields = context.fields if fields is None else fields
        self.dataset = dataset
        self.coords = SkyCoord(context.ra0, context.dec0)
        self.D = context.D * u.Mpc

    def plot(self, xylims=None, cbbox="regular", figsize=(3.54, 3.5),
             arrows=True, sigma=None, xloc=None):
        """ Make the plots. """
        xloc = len(self.columns) * [-4] if xloc is None else xloc
        sigma_str = "" if sigma is None else "_sigma{}".format(sigma)
        if xylims is None:
            xylims = [(25, -10), (-25, 20)]
        for j, col in enumerate(self.columns):
            print(col)
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1,1)
            gs.update(left=0.12, right=0.97, bottom=0.1, top=0.99)
            ax = plt.subplot(gs[0])
            ax.set_facecolor("0.85")
            plt.minorticks_on()
            self.make_contours(alpha=0.5)
            kmaps = []
            for i, (field, table) in enumerate(zip(self.fields, self.tables)):
                binsfile = os.path.join(context.get_data_dir(self.dataset),
                        field, "sn{0}/voronoi2d_sn{0}.fits".format(
                        self.targetSN))
                bins = table["BIN"].astype(np.float)
                vector = table[col].astype(np.float)
                image = context.get_field_files(field)[0]
                extent = calc_extent(image, self.coords, self.D)
                extent = offset_extent(extent, field)
                binimg = fits.getdata(binsfile)
                kmap = np.zeros_like(binimg)
                kmap[:] = np.nan
                for bin,v in zip(bins, vector):
                    if not np.isfinite(v):
                        continue
                    idx = np.where(binimg == bin)
                    kmap[idx] = v
                kmaps.append(kmap)
            automin = np.nanmin(np.array(kmaps))
            automax = np.nanmax(np.array(kmaps))
            vmin = self.lims[j][0] if self.lims[j][0] is not None else automin
            vmax = self.lims[j][1] if self.lims[j][1] is not None else automax
            for i in range(len(self.fields)):
                m = plt.imshow(kmap, origin="bottom", cmap=self.cmaps[j],
                               vmin=vmin, vmax=vmax,
                               extent=extent,
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
                                   cbar_pos=[0.24, 0.365, 0.05, 0.3],
                                   ticks=np.linspace(vmin, vmax, 5),
                                   cblabel=self.labels[j],
                                   cb_fmt=self.cb_fmts[j])
            elif cbbox == "horizontal":
                plt.gca().add_patch(Rectangle((-9.7, -11.5), 8.4, 3.1, alpha=1,
                                              zorder=10, edgecolor="w",
                                              linewidth=1, facecolor="w"))
                self.draw_colorbar(fig, ax, m, orientation="horizontal",
                                   cbar_pos=[0.63, 0.16, 0.3, 0.05],
                                   ticks=np.linspace(vmin, vmax, 4),
                                   cblabel=self.labels[j],
                                   cb_fmt=self.cb_fmts[j])
                ax.text(xloc[j], -9.15, self.labels[j], zorder=11)
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
            plt.clabel(cs, contours[0::2], fmt="%d", fontsize=fontsize,
                       inline_spacing=-3)
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

def make_maps(targetSN=250, dataset="MUSE"):
    """ Produces a test plot using the geometry table. """
    geom = get_geom("fieldA", targetSN)
    geomtab = os.path.join(context.get_data_dir(dataset),
                           "fieldA/sn{}/geom.fits".format(targetSN))
    geom.write(geomtab, overwrite=True)
    wdir = os.path.join(context.get_data_dir(dataset), "fieldA",
                        "sn{}".format(targetSN))
    ppxftable = os.path.join(wdir, "ppxf_vel50_w4500_10000_kinematics.fits")
    results = Table.read(ppxftable)
    sn_table = Table.read(os.path.join(wdir, "measured_sn.fits"))
    results = hstack([geom, results])
    results = join(results, sn_table, keys="spec")
    results.rename_column("SN/Ang", "SNRperAng")
    # Adding stellar population table
    stpop = Table.read(os.path.join(wdir, "stpop.fits"))
    results = join(results, stpop, keys="BIN")
    results.write(os.path.join(wdir, "results.fits"), overwrite=True)
    outdir = os.path.join(wdir, "plots")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # cb_fmts = len(geom.colnames) * ["%i"]
    fields = ["SNRperAng", "Z", "T", "imf", "alphaFe", "NaFe"]
    labels = ["SNR (\\r{A}$^{-1}$)", "[Z/H]", "Age (Gyr)",
              "$\\Gamma_b$", r"[$\alpha$/Fe]", "[Na/Fe]"]
    cb_fmts = ["%i", "%.2f", "%i", "%.1f", "%.2f", "%.2f"]
    xloc = [-4, -4.5, -4, -5., -4.5, -4.5]
    cmaps = ["viridis"] * len(xloc)
    pvm = PlotVoronoiMaps([results], fields, outdir,
                          targetSN=targetSN, fields=["fieldA"],
                          labels=labels, cb_fmts=cb_fmts, cmaps=cmaps)

    xylims = [[10.1, -10.1], [-12, 9]]
    pvm.plot(xylims=xylims, arrows=False, cbbox="horizontal", xloc=xloc)

def make_triptychs(targetSN=250, dataset="MUSE"):
    wdir = os.path.join(context.get_data_dir(dataset), "fieldA",
                        "sn{}".format(targetSN))
    table= Table.read(os.path.join(wdir, "results.fits"))
    triptychs = [["SNRperAng", "Z", "NaFe"], ["T", "alphaFe", "imf"]]
    binsfile = os.path.join(context.get_data_dir(dataset),
                "fieldA", "sn{0}/voronoi2d_sn{0}.fits".format(targetSN))
    bins = table["BIN"].astype(np.float)
    image = context.get_field_files("fieldA")[0]
    extent = calc_extent(image)
    extent = offset_extent(extent, "fieldA")
    binimg = fits.getdata(binsfile)
    # Image for contours
    vband = os.path.join(context.home, "images/hydra1.fits")
    vdata = fits.getdata(vband, verify=False)
    vdata = np.clip(vdata - 4900., 1., vdata)
    muv = -2.5 * np.log10(vdata / 480. / 0.252 / 0.252) + 27.2
    contours = np.linspace(19.5, 23.5, 9)
    nsigma = 2
    datasmooth = ndimage.gaussian_filter(muv, nsigma, order=0.)
    extent_vband = np.array(calc_extent(vband, extension=0))
    extent_vband[:2] += 3
    extent_vband[2:] -= 0.2
    labels = {"SNRperAng": "SNR (\\r{A}$^{-1}$)",
              "Z": "[Z/H]", "T": "Age (Gyr)",
              "imf": "$\\Gamma_b$", "alphaFe": r"[$\alpha$/Fe]",
              "NaFe": "[Na/Fe]"}
    names = ["maps1", "maps2"]
    for j, triptych in enumerate(triptychs):
        fig = plt.figure(figsize=(6.5, 2.8))

        gs = gridspec.GridSpec(2, 3, height_ratios=[0.92, 0.08])
        gs.update(left=0.08, right=0.99, bottom=0.12, top=0.92, wspace=0.04,
                  hspace=0.00)
        for i, col in enumerate(triptych):
            vector = table[col].astype(np.float)
            kmap = np.zeros_like(binimg)
            kmap[:] = np.nan
            for bin, v in zip(bins, vector):
                if not np.isfinite(v):
                    continue
                idx = np.where(binimg == bin)
                kmap[idx] = v
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor("0.85")
            cs = ax.contour(datasmooth, contours, extent=extent_vband,
                             colors="k", linewidths=0.8)
            m = ax.imshow(kmap, origin="bottom", extent=extent,
                           aspect="equal", alpha=1)
            ax.set_xlabel("X (kpc)")
            ax.xaxis.set_label_position('top')
            ax.xaxis.set_ticks_position('top')
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,
                top=True,
                labelbottom=False)
            cbax = plt.subplot(gs[1, i])
            cbar = Colorbar(ax=cbax, mappable=m, orientation="horizontal",
                            ticklocation='bottom')
            cbar.set_label(labels[col])
            cbar.ax.xaxis.set_label_position('bottom')
            if i > 0:
                ax.yaxis.set_ticklabels([])
            else:
                ax.set_ylabel("Y (kpc)")
        for fmt in ["png", "pdf"]:
            plt.savefig(os.path.join(wdir, "plots/{}.{}".format(names[j], fmt)),
                        dpi=300)
        plt.close()




if __name__ == "__main__":
    make_maps(targetSN=250)
    # make_triptychs()