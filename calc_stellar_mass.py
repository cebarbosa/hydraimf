""" Calculating the stellar mass of NGC 3311 and its components. """
import os

import numpy as np

from calc_mass2light import Mass2Light

if __name__ == "__main__":
    Magr = np.array([-18.2, -19.56, -21.29, -23.15])
    t = np.array([11, 10, 8, 7])
    z = np.array([0.2, 0.1, 0.05, 0])
    mu = np.array([2.2, 2.2, 1.8, 1.5])
    p = np.column_stack([mu, z, t])
    Mg_sun = 4.5
    L = np.power(10, -0.4 * (Magr - Mg_sun))
    m2l_r = Mass2Light(imf="bi")
    m2l_g = Mass2Light(imf="bi", band="g")
    ml_R = m2l_g(p) - 0.59 * (m2l_g(p) - m2l_r(p)) - 0.01
    M = L * ml_R
    Mtot = M.sum()
    print(np.log10(M))