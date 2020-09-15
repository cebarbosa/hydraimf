# -*- coding: utf-8 -*-
"""

Created on 16/05/16

@author: Carlos Eduardo Barbosa

Program to calculate lick indices

"""

import numpy as np
from scipy.integrate import simps, trapz
import astropy.units as u
import astropy.constants as const
from tqdm import tqdm

class Lick():
    """ Class to measure Lick indices.

    Computation of the Lick indices in a given spectrum. Position of the
    passbands are determined by redshifting the position of the bands
    to the systemic velocity of the galaxy spectrum.

    =================
    Input parameters:
    =================
        wave (array):
            Wavelength of the spectrum given.

        galaxy (array):
            Galaxy spectrum in arbitrary units.

        bands0 (array) :
            Definition of passbands for Lick indices at rest
            wavelengths. Units should be consistent with wavelength array.

        vel (float, optional):
            Systemic velocity of the spectrum in km/s. Defaults to zero.

        dw (float, optinal):
            Extra wavelength to be considered besides
            bands for interpolation. Defaults to 2 wavelength units.

    ===========
    Attributes:
    ===========
        bands (array):
            Wavelengths of the bands after shifting to the
            systemic velocity of the galaxy.

    """
    def __init__(self, bands0, units=None, wave_unit=None):
        self.wave_unit = u.AA if wave_unit is None else wave_unit
        self.bands0 = bands0.to(self.wave_unit).value
        self.units = units if units is not None else \
                                                np.ones(len(self.bands0)) * u.AA
        self.idx = [_ == u.AA for _ in self.units]
        self.c = const.c.to("km/s")

    def __call__(self, wave, data, vel=None):
        """ Calculation of Lick indices using spline integration.

        ===========
        Attributes:
        ===========
            R (array):
                Raw integration values for the Lick indices.

            Ia (array):
                Indices measured in equivalent widths.

            Im (array):
                Indices measured in magnitudes.
        """
        if hasattr(wave, "unit"):
            wave = wave.to(u.AA).value
        data = np.atleast_2d(data)
        vel = 0 * u.km / u.s if vel is None else vel
        bands = self.bands0 * np.sqrt((1 + vel.to("km/s")/ self.c)
                    / (1 - vel.to("km/s")/self.c))
        shape = (len(bands), len(data))
        R = np.full(shape, np.nan)
        Ia = np.full(shape, np.nan)
        Im = np.full(shape, np.nan)
        for i, w in tqdm(enumerate(bands), desc="Measuring indices:"):
            condition = np.array([w[0] > wave[0],
                                 w[-1] < wave[-1]])
            if not np.all(condition):
                continue
            xs, ys = [], []
            for j in range(3):
                w1 = w[2 * j]
                w2 = w[2 * j + 1]
                idx = np.where((wave >= w1) & (wave <= w2))[0]
                x = wave[idx]
                y = data[:, idx]
                x0 = wave[idx[0]-1]
                if x0 < w1:
                    dy = (data[:, idx[0]] - data[:, idx[0]-1]) / \
                         (wave[idx[0]] - wave[idx[0]-1])
                    y0 = data[:, idx[0]-1] + dy * (w1 - x0)
                    y = np.hstack([y0[:, None], y])
                    x = np.hstack([w1, x])
                x1 =  wave[idx[-1]+1]
                if x1 > w2:
                    dy = (data[:, idx[-1]+1] - data[:, idx[-1]]) / \
                         (wave[idx[-1]+1] - wave[idx[-1]])
                    y1 = data[:, idx[-1]] + dy * (x1 - w2)
                    y = np.hstack([y, y1[:, None], ])
                    x = np.hstack([x, w2])
                xs.append(x)
                ys.append(y)
            # Baseline flux for pseudo-contiuum
            fp1 = np.trapz(ys[0], xs[0]) / (w[1] - w[0])
            fp2 = np.trapz(ys[2], xs[2]) / (w[5] - w[4])
            # Making pseudocontinuum vector
            x1 = (w[0] + w[1])/2.
            x2 = (w[4] + w[5])/2.
            fc = (fp1 + (fp2 - fp1)/ (x2 - x1) * (xs[1]- x1)[:, None]).T
            # Calculating indices
            R[i] = np.trapz(ys[1] / fc, xs[1]) / (w[3]-w[2])
            Ia[i] = (1 - R[i]) * (w[3]-w[2])
            Im[i] = -2.5 * np.log10(R[i])
        return R.T, Ia.T, Im.T

def bands_shift(bands, vel):
    c = 299792.458  # Speed of light in km/s
    return  bands * np.sqrt((1 + vel/c)/(1 - vel/c))

if __name__ == "__main__":
    pass
