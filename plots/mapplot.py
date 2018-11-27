# -*- coding: utf-8 -*-
"""

Created on 10/02/16
Adapted on 24/03/2018 to use in hydraimf project

@author: Carlos Eduardo Barbosa

"""


import os

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
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
            self.cb_fmts = len(self.columns) * ["%.2f"]
        self.targetSN = targetSN
        self.fields = context.fields if fields is None else fields
        self.dataset = dataset

    def plot(self, xylims=None, cbbox="regular", figsize=(3.54, 4),
             arrows=True, sigma=None):
        """ Make the plots. """
        sigma_str = "" if sigma is None else "_sigma{}".format(sigma)
        if xylims is None:
            xylims = [(25, -10), (-25, 20)]
        for j, col in enumerate(self.columns):
            print(col)
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1,1)
            gs.update(left=0.13, right=0.98, bottom = 0.1, top=0.98)
            ax = plt.subplot(gs[0])
            ax.set_facecolor('1')
            plt.minorticks_on()
            self.make_contours(alpha=0.5)
            kmaps = []
            for i, (field, table) in enumerate(zip(self.fields, self.tables)):
                binsfile = os.path.join(context.data_dir, self.dataset,
                                        "combined", field,
                                    "voronoi2d_sn{}.fits".format(self.targetSN))
                bins = table["BIN"].astype(np.float)
                vector = table[col].astype(np.float)
                image = context.get_field_files(field)[0]
                extent = calc_extent(image)
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
            plt.ylabel("Y (kpc)")
            if arrows:
                plt.arrow(-5, -20, 5, 0, linewidth=1.5, color="b", head_length=1,
                          head_width=0.5, alpha=1)
                plt.arrow(-5, -20, 0, 5, linewidth=1.5, color="b", head_length=1,
                          head_width=0.5, alpha=1)
                plt.text(1., -18, "E", color="b", fontsize=10, va='top')
                plt.text(-4.3, -12.5, "N", color="b", fontsize=10, va='top')
            ax.tick_params(axis="both",  which='major', labelsize=10)
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
                                              zorder=10,
                                              color="w"))
                self.draw_colorbar(fig, ax, m, orientation="vertical",
                                   cbar_pos=[0.24, 0.365, 0.05, 0.3],
                                   ticks=np.linspace(vmin, vmax, 5),
                                   cblabel=self.labels[j],
                                   cb_fmt=self.cb_fmts[j])
            elif cbbox == "horizontal":
                plt.gca().add_patch(Rectangle((-9.5, -12), 7.5, 4.5, alpha=1,
                                              zorder=10,
                                              color="w"))
                self.draw_colorbar(fig, ax, m, orientation="horizontal",
                                   cbar_pos=[0.65, 0.2, 0.3, 0.05],
                                   ticks=np.linspace(vmin, vmax, 5),
                                   cblabel=self.labels[j],
                                   cb_fmt=self.cb_fmts[j], rotation=0,
                                   ylpos=1.2, xpos=.4)

            output = os.path.join(self.outdir, "{}_sn{}.png".format(col,
                                                                 self.targetSN))
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
            extent = np.array(calc_extent(vband, extension=0))
            extent[:2] += 3
            extent[2:] -=0.2
        cs = plt.contour(datasmooth, contours, extent=extent,
                         colors=colors, linewidths=lw)
        if plot_vband:
            plt.imshow(muv, cmap="bone", extent=extent, vmax=vmax, vmin=vmin,
                       alpha=alpha, origin="bottom")
        if label:
            plt.clabel(cs, contours[0::2], fmt="%d", fontsize=fontsize,
                       inline_spacing=-3, linewidths=lw)
        return

    def draw_colorbar(self, fig, ax, coll, ticks=None, cblabel="",
                      cbar_pos=None, cb_fmt="%i", labelsize=9.,
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
        ax = cbar.ax
        ax.text(xpos,ylpos,cblabel,rotation=rotation)
        return

def test_plot(targetSN=150):
    """ Produces a test plot using the geometry table. """
    geom = get_geom("fieldA", targetSN)
    geomtab = os.path.join(context.data_dir, "MUSE/combined/fieldA/geom.fits")
    geom.write(geomtab, overwrite=True)
    tables = [geom]
    outdir = os.path.join(context.home, "plots", "test")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    cb_fmts = len(geom.colnames) * ["%i"]
    pvm = PlotVoronoiMaps(tables, geom.colnames, outdir, targetSN=150, fields=[
        "fieldA"], cb_fmts=cb_fmts)
    xylims = [[10, -10], [-12.5, 10]]
    pvm.plot(xylims=xylims, arrows=False, cbbox="horizontal")


if __name__ == "__main__":
    test_plot(targetSN=150)
