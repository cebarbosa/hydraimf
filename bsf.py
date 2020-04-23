# -*- coding: utf-8 -*-
"""

Created on 12/04/19

Author : Carlos Eduardo Barbosa

Bayesian spectrum fitting classes.

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u
from scipy.special import gamma, digamma
import theano.tensor as tt
from scipy.ndimage import convolve1d
from scipy.interpolate import LinearNDInterpolator
from spectres import spectres

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, data, x, sigma, stpop, loglike=None):
        self.loglike = "studt" if loglike is None else loglike
        self.data = data
        self.x = x
        self.sigma = sigma
        self.stpop = stpop
        self.loglike = loglike
        if self.loglike == "studt":
            self.likelihood = StudTLogLike(self.data, self.sigma, self.stpop)
        elif self.loglike == "normal":
            self.likelihood = NormalLogLike(self.data, self.sigma, self.stpop)
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood)

    def perform(self, node, inputs, outputs):
        theta, = inputs
        logl = self.likelihood(theta)
        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        theta, = inputs  # our parameters
        return [g[0]*self.logpgrad(theta)]


class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, likelihood):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.likelihood = likelihood

    def perform(self, node, inputs, outputs):
        theta, = inputs
        # calculate gradients
        grads = self.likelihood.gradient(theta)
        outputs[0][0] = grads

class StudTLogLike():
    def __init__(self, data, sigma, func):
        self.data = data
        self.sigma = sigma
        self.func = func
        self.N = len(data)
        self.nparams = self.func.nparams + 1

    def __call__(self, theta):
        nu = theta[-1]
        e_i = self.func(theta[:-1]) - self.data
        x = 1. + np.power(e_i / self.sigma, 2.) / (nu - 2)
        LLF = self.N * np.log(gamma(0.5 * (nu + 1)) /
                         np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu))  \
             - 0.5 * (nu + 1) * np.sum(np.log(x)) \
             - 0.5 * np.sum(np.log(self.sigma**2)) # Constant
        return float(LLF)

    def gradient(self, theta):
        grad = np.zeros(self.func.nparams + 1)
        nu = theta[-1]
        # d loglike / d theta
        e_i = self.func(theta[:-1]) - self.data
        x = np.power(e_i / self.sigma, 2.) / (nu - 2.)
        term1 = 1 / (1 + x)
        term2 = 2 * e_i / (self.sigma**2) / (nu-2)
        term12 = term1 * term2
        sspgrad = self.func.gradient(theta[:-1])
        grad[:-1] = -0.5 * (nu + 1) * np.sum(term12[np.newaxis, :] *
                                             sspgrad, axis=1)
        # d loglike / d nu
        nuterm1 = 0.5 * self.N * digamma(0.5 * (nu + 1))
        nuterm2 = - 0.5 * self.N / (nu - 2)
        nuterm3 = -0.5 * self.N * digamma(0.5 * nu)
        nuterm4 = -0.5 * np.sum(np.log(1 + x))
        nuterm5 = 0.5 * (nu + 1) * np.power(nu - 2, -2) * \
                  np.sum(np.power(e_i / self.sigma, 2) * term1)
        grad[-1] = nuterm1 + nuterm2 + nuterm3 + nuterm4 + nuterm5
        return grad

class NormalLogLike():
    def __init__(self, data, sigma, func):
        self.data = data
        self.sigma = sigma
        self.func = func
        self.N = len(data)
        self.nparams = self.func.nparams

    def __call__(self, theta):
        e_i = self.func(theta) - self.data
        LLF = - 0.5 * self.N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / self.sigma, 2)) \
              - 0.5 * np.sum(np.log(self.sigma ** 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = self.func(theta) - self.data
        grad = - np.sum(e_i / np.power(self.sigma, 2.)[np.newaxis, :] *
                        self.func.gradient(theta), axis=1)
        return grad

class NormalWithErrorsLogLike():
    def __init__(self, data, sigma, func):
        self.data = data
        self.sigma = sigma
        self.func = func
        self.N = len(data)
        self.nparams = self.func.nparams

    def __call__(self, theta):
        e_i = self.func(theta[:-1]) - self.data
        S = theta[-1]
        LLF = - 0.5 * self.N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / (S * self.sigma) , 2)) \
              - 0.5 * np.sum(np.log((S * self.sigma )** 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = self.func(theta[:-1]) - self.data
        S = theta[-1]
        A = e_i / np.power(S * self.sigma, 2.)
        B = self.func.gradient(theta[:-1])
        C = -np.sum(A[np.newaxis,:] * B, axis=1)
        grad = np.zeros(len(theta))
        grad[:-1] = C
        grad[-1] = - self.N / S + \
                   np.power(S, -3) * np.sum(np.power(e_i / self.sigma, 2))
        return grad

class SEDModel():
    """ Stellar population model. """
    def __init__(self, wave, params, templates, nssps=2,
                 velscale=None, wave_out=None, wave_unit=None,
                 em_templates=None, em_components=None,
                 em_names=None, extlaw=None):
        """ SED model based on combination of SSPs, emission lines, sky lines
        and their LOSVD.

        This is the main routine to produce a Spectral Energy Distribution
        model using a combination of stellar populations and emission lines.

        Parameters
        ----------
        wave: array_like
           Common wavelength array for the stellar population templates. For
           kinematic fitting, wave has to be binned at a fixed velocity
           scale prior to the modelling.

        params: astropy.table.Table
            Parameters of the SSPs.

        templates: array_like
            SED templates for the single stellar population models.

        nssps: int or 1D array_like, optional
            Number of stellar populations to be used. Negative values ignore
            the kinematics. Negative values ignore the LOSVD convolution.

            Examples:
                - 2 SSPs with common kinematics (default) nssps = 2 or
                nssps = np.array([2])
                - 2 SSPs with decoupled kinematics:
                  nssps = np.array([1,1])
                - 2 SSPS with 2 kinematic components
                  nssps = np.array([2,2])
                - Ignore LOSVD convolution in the fitting of 2 stellar
                populations (multi-band SED fitting)
                  nssps = -2 or nssps = np.array([2])

        velscale: astropy.unit, optional
            Determines the velocity difference between adjacent pixels in the
            wavelength array. Default value is 1 km/s/pixel.

        wave_out: array_like, optional
            Wavelength array of the output. In case the output is not
            determined, it is assumed that wave_out is the same as wave. If
            necessary, the rebbining if performed with spectres
            (https://spectres.readthedocs.io/en/latest/)

        wave_unit: astropy.units, optional
            Units of the wavelength array. Assumed to be Angstrom if not
            specified.

        em_templates: array_like, optional
            Emission line templates, which can be used either to model gas
            emission lines or sky lines. If

        em_components: int or 1D array_like, optional
            Similar to nssps, negative components do not make LOSVD convolution.


        em_names: array_like, optional
            Names of the emission lines to be used in the outputs.

        Returns
        -------
            "SEDModel" callable class


        Methods
        -------
            __call__(p): Returns the SED model of parameters p

            gradient(p): Gradient of the SED function with parameters p

         """
        ########################################################################
        # Setting wavelength units
        self.wave_unit = u.angstrom if wave_unit is None else wave_unit
        if hasattr(wave, "unit"):
            self.wave = wave
        else:
            self.wave = wave * self.wave_unit
        self.wave_out = self.wave if wave_out is None else wave_out
        if not hasattr(self.wave_out, "unit"):
            self.wave_out *= self.wave_unit
        ########################################################################
        # Verify if rebinning is necessary to modeling
        self.rebin = not np.array_equal(self.wave, self.wave_out)
        ########################################################################
        self.velscale = 1. * u.km / u.s if velscale is None else velscale
        self.params = params
        self.templates = templates
        self.nssps = np.atleast_1d(nssps).astype(np.int)
        self.em_templates = em_templates
        self.em_names = em_names
        self.extlaw = "CCM89" if extlaw is None else extlaw
        # Dealing with SSP components
        self.pops = []
        self.idxs = [0]
        self.parnames = []
        for i, comp in enumerate(self.nssps):
            if comp > 0:
                csp = NSSPSConv(self.wave, self.params,
                                self.templates, self.velscale, npop=comp)
            else:
                csp = NSSPs(self.wave, self.params, self.templates,
                            npop=abs(comp))
            parnames = []
            for p in csp.parnames:
                psplit = p.split("_")
                psplit[0] = "{}_{}".format(psplit[0], i)
                newp = "_".join(psplit)
                parnames.append(newp)
            self.idxs.append(csp.nparams + self.idxs[-1])
            self.pops.append(csp)
            self.parnames.append(parnames)
        #######################################################################
        # Dealing with emission line components
        if self.em_templates is not None:
            n_em = len(em_templates) # Number of emission lines
            em_components = np.ones(n_em) if em_components is None else \
                                 em_components
            self.em_components = np.atleast_1d(em_components).astype(np.int)
            for i, comp in enumerate(np.unique(self.em_components)):
                idx = np.where(self.em_components == comp)[0]
                if comp < 0:
                    em = SkyLines(self.wave, self.em_templates[idx],
                                  em_names=self.em_names[idx])
                else:
                    em = EmissionLines(self.wave, self.em_templates[idx],
                                       self.em_names[idx], self.velscale)
                parnames = []
                for p in em.parnames:
                    psplit = p.split("_")
                    psplit[0] = "{}_{}".format(psplit[0], i + len(self.nssps))
                    newp = "_".join(psplit)
                    parnames.append(newp)
                self.idxs.append(em.nparams + self.idxs[-1])
                self.pops.append(em)
                self.parnames.append(parnames)
        self.nparams = self.idxs[-1]
        self.parnames = [item for sublist in self.parnames for item in
                         sublist]


    def __call__(self, theta):
        sed = np.zeros(len(self.wave))
        for i in range(len(self.pops)):
            t = theta[self.idxs[i]: self.idxs[i+1]]
            s = self.pops[i](t)
            sed += s
        if not self.rebin:
            return sed
        sed = spectres(self.wave_out.to("AA").value,
                       self.wave.to("AA").value, sed)
        return sed

    def gradient(self, theta):
        grads = []
        for i, pop in enumerate(self.pops):
            t = theta[self.idxs[i]: self.idxs[i+1]]
            grads.append(pop.gradient(t))
        grads = np.vstack(grads)
        if not self.rebin:
            return grads
        grads = spectres(self.wave_out.to("AA").value,
                       self.wave.to("AA").value, grads)
        return grads

class SSP():
    """ Linearly interpolated SSP models."""
    def __init__(self, wave, params, templates):
        self.wave = wave
        self.params = params
        self.templates = templates
        self.nparams = len(self.params.colnames)
        ########################################################################
        # Interpolating models
        x = self.params.as_array()
        a = x.view((x.dtype[0], len(x.dtype.names)))
        self.f = LinearNDInterpolator(a, templates, fill_value=0.)
        ########################################################################
        # Get grid points to handle derivatives
        inner_grid = []
        thetamin = []
        thetamax = []
        for par in self.params.colnames:
            thetamin.append(np.min(self.params[par].data))
            thetamax.append(np.max(self.params[par].data))
            inner_grid.append(np.unique(self.params[par].data)[1:-1])
        self.thetamin = np.array(thetamin)
        self.thetamax = np.array(thetamax)
        self.inner_grid = inner_grid

    def __call__(self, theta):
        return self.f(theta)

    def gradient(self, theta, eps=1e-6):
        # Clipping theta to avoid border problems
        theta = np.maximum(theta, self.thetamin + 2 * eps)
        theta = np.minimum(theta, self.thetamax - 2 * eps)
        grads = np.zeros((self.nparams, self.templates.shape[1]))
        for i,t in enumerate(theta):
            epsilon = np.zeros(self.nparams)
            epsilon[i] = eps
            # Check if data point is in inner grid
            in_grid = t in self.inner_grid[i]
            if in_grid:
                tp1 = theta + 2 * epsilon
                tm1 = theta + epsilon
                grad1 = (self.__call__(tp1) - self.__call__(tm1)) / (2 * eps)
                tp2 = theta - epsilon
                tm2 = theta - 2 * epsilon
                grad2 = (self.__call__(tp2) - self.__call__(tm2)) / (2 * eps)
                grads[i] = 0.5 * (grad1 + grad2)
            else:
                tp = theta + epsilon
                tm = theta - epsilon
                grads[i] = (self.__call__(tp) - self.__call__(tm)) / (2 * eps)
        return grads

class NSSPs():
    """ Stellar population model. """
    def __init__(self, wave, params, templates, npop=2, extlaw=None):
        self.params = params
        self.wave = wave
        self.templates = templates
        self.npop = npop
        self.ssp = SSP(self.wave, self.params, self.templates)
        self.ncols = len(self.params.colnames)
        self.nparams = self.npop * (len(self.params.colnames) + 1)  + 2
        self.shape = (self.nparams, len(self.wave))
        self.extlaw = "C00" if extlaw is None else extlaw
        # Preparing array for redenning
        x = 1 / self.wave.to("micrometer") * u.micrometer
        if self.extlaw == "C00":
            self.Aw = C2000(self.wave)
        elif self.extlaw == "CMM89":
            self.Aw = CCM89(self.wave)
        # Set parameter names
        self.parnames = ["Av", "Rv"]
        for n in range(self.npop):
            for p in ["flux"] + self.params.colnames:
                self.parnames.append("{}_{}".format(p, n))

    def __call__(self, theta):
        p = theta[2:].reshape(self.npop, -1)
        return self.Aw(theta[:2]) * np.dot(p[:,0], self.ssp(p[:, 1:]))

    def gradient(self, theta):
        grad = np.zeros(self.shape)
        ps = theta[2:].reshape(self.npop, -1)
        F = np.dot(ps[:,0], self.ssp(ps[:, 1:]))
        grad[:2] = F * self.Aw.gradient(theta[:2])
        const = self.Aw(theta[:2])
        for i, p in enumerate(ps):
            idx = 2 + (i * (self.ncols + 1))
            # dF/dFi
            grad[idx] = const * self.ssp(p[1:])
            # dF / dSSPi
            grad[idx+1:idx+1+self.ncols] = const * p[0] * \
                                           self.ssp.gradient(p[1:])
        return grad

class CCM89:
    def __init__(self, wave):
        if hasattr(wave, "unit"):
            self.wave = wave
        else:
            self.wave = wave * u.AA
        x = 1 / wave.to(u.micrometer).value
        def anir(x):
            return 0.574 * np.power(x, 1.61)

        def bnir(x):
            return -0.527 * np.power(x, 1.61)

        def aopt(x):
            y = x - 1.82
            return 1 + 0.17699 * y - 0.50447 * np.power(y, 2) \
                   - 0.02427 * np.power(y, 3) + 0.7208 * np.power(y, 4) \
                   + 0.0197 * np.power(y, 5) - 0.7753 * np.power(y, 6) \
                   + 0.32999 * np.power(y, 7)

        def bopt(x):
            y = x - 1.82
            return 1.41338 * y + 2.28305 * np.power(y, 2) + \
                   1.07233 * np.power(y, 3) - 5.38434 * np.power(y, 4) - \
                   0.62251 * np.power(y, 5) + 5.30260 * np.power(y, 6) - \
                   2.09002 * np.power(y, 7)

        def auv(x):
            Fa = - 0.04473 * np.power(x - 5.9, 2) - 0.009779 * np.power(x - 5.9,
                                                                        3)
            a = 1.752 - 0.316 * x - 0.104 / (np.power(x - 4.67, 2) + 0.341)
            return np.where(x < 5.9, a, a + Fa)

        def buv(x):
            Fb = 0.2130 * np.power(x - 5.9, 2) + 0.1207 * np.power(x - 5.9, 3)
            b = -3.090 + 1.825 * x + 1.206 / (np.power(x - 4.62, 2) + 0.263)
            return np.where(x < 5.9, b, b + Fb)
        nir = (0.3 <= x) & (x <= 1.1)
        optical = (1.1 < x) & (x <= 3.3)
        uv = (3.3 < x) & (x <= 8)
        self.a = np.where(nir, anir(x),
                     np.where(optical, aopt(x),
                              np.where(uv, auv(x), 0)))
        self.b = np.where(nir, bnir(x),
                     np.where(optical, bopt(x),
                              np.where(uv, buv(x), 0)))

    def __call__(self, theta):
        """ theta = (Av, Rv)"""
        return np.power(10, -0.4 * theta[0] * (self.a + self.b / theta[1]))

    def gradient(self, theta):
        grad = np.zeros((2, len(self.wave)))
        A = self.__call__(theta)
        grad[0] = -0.4 * np.log(10) * (self.a + self.b / theta[1]) * A
        grad[1] = 0.4 * np.log(10) * theta[0] * self.b * \
                  np.power(theta[1], -2) * A
        return grad

class C2000():
    def __init__(self, wave):
        if hasattr(wave, "unit"):
            self.wave = wave
        else:
            self.wave = wave * u.AA
        x = 1 / wave.to(u.micrometer).value
        self.kappa = np.where(self.wave > 0.63 * u.micrometer,
                              2.659 * (-1.857 + 1.040 * x), \
                              2.659 * (-2.156 + 1.509 * x - 0.198 * x * x
                                       + 0.011 * (x * x * x)))

    def __call__(self, theta):
        return np.power(10, -0.4 * theta[0] * (1. + self.kappa / theta[1]))

    def gradient(self, theta):
        grad = np.zeros((2, len(self.wave)))
        A = self.__call__(theta)
        # dAw / dAv
        grad[0] = A * np.log(10) * (-0.4 * (1. + self.kappa / theta[1]))
        # dAw / dRv
        grad[1] = A * 0.4 * theta[0] * self.kappa * np.log(10) * \
                  np.power(theta[1], -2.)
        return grad

class NSSPSConv():
    def __init__(self, wave, params, templates, velscale, npop=1):
        self.params = params
        self.wave = wave
        self.templates = templates
        self.npop = npop
        self.velscale = velscale.to("km/s").value
        self.nssps = NSSPs(self.wave, self.params, self.templates,
                           npop=self.npop)
        self.nparams = self.nssps.nparams + 2
        self.shape = (self.nparams, len(self.wave))
        # Set parameter names
        self.parnames = ["Av", "Rv"]
        for n in range(self.npop):
            for p in ["flux"] + self.params.colnames:
                self.parnames.append("{}_{}".format(p, n))
        self.parnames.append("V")
        self.parnames.append("sigma")

    def kernel_arrays(self, p):
        x0, sigx = p / self.velscale
        dx = int(np.ceil(np.max(abs(x0) + 5 * sigx)))
        n = 2 * dx + 1
        x = np.linspace(-dx, dx, n)
        y = (x - x0) / sigx
        y2 = np.power(y, 2.)
        k = np.exp(-0.5 * y2) / (sigx * np.sqrt(2 * np.pi))
        return y, k

    def __call__(self, theta):
        p = theta[:self.nssps.nparams]
        sed = self.nssps(p)
        y, k = self.kernel_arrays(theta[self.nssps.nparams:])
        return convolve1d(sed, k)

    def gradient(self, theta):
        grad = np.zeros(self.shape)
        sspgrad = self.nssps.gradient(theta[:self.nssps.nparams])
        p = theta[self.nssps.nparams:]
        y, k = self.kernel_arrays(p)
        for i in range(len(sspgrad)):
            grad[i] = convolve1d(sspgrad[i], k)
        sed = self.nssps(theta[:self.nssps.nparams])
        grad[-2] = convolve1d(sed, y * k / p[1])
        grad[-1] = convolve1d(sed, (y * y - 1.) * k / p[1])
        return grad

class EmissionLines():
    def __init__(self, wave, templates, em_names, velscale):
        self.wave = wave
        self.templates = templates
        self.em_names = em_names
        self.n_em = len(templates)
        self.velscale = velscale
        self.shape = (self.n_em+2, len(self.wave))
        self.parnames = ["flux_{}".format(name) for name in em_names]
        self.parnames.append("V")
        self.parnames.append("sigma")
        self.nparams = len(self.parnames)


    def __call__(self, theta):
        g = theta[:self.n_em]
        p = theta[self.n_em:]
        return convolve1d(np.dot(g, self.templates),
                          self.kernel_arrays(p)[1])

    def gradient(self, theta):
        g = theta[:self.n_em]
        p = theta[-2:]
        y, k = self.kernel_arrays(p)
        grad = np.zeros(self.shape)
        for i in range(self.n_em):
            grad[i] = convolve1d(self.templates[i], k)
        gas = np.dot(g, self.templates)
        grad[-2] = convolve1d(gas, y * k / p[1])
        grad[-1] = convolve1d(gas, (y * y - 1.) * k / p[1])
        return grad

    def kernel_arrays(self, p):
        x0, sigx = p / self.velscale.to("km/s").value
        dx = int(np.ceil(np.max(abs(x0) + 5 * sigx)))
        n = 2 * dx + 1
        x = np.linspace(-dx, dx, n)
        y = (x - x0) / sigx
        y2 = np.power(y, 2.)
        k = np.exp(-0.5 * y2) / (sigx * np.sqrt(2 * np.pi))
        return y, k

class SkyLines():
    def __init__(self, wave, templates, em_names=None):
        self.wave = wave
        self.templates = templates
        self.nparams = len(self.templates)
        self.n_em = len(templates)
        self.em_names = ["sky{}".format(n) for n in range(self.n_em)] if \
            em_names is None else em_names
        self.parnames = self.em_names
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        return np.dot(theta, self.templates)

    def gradient(self, theta):
        return self.templates