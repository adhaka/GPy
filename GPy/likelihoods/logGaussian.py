# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from . import link_functions
from .likelihood import Likelihood



class LogGaussian(Likelihood):
    """
    Doc string for implementation of Log-gaussian!!!

    """
    def __init__(self,gp_link=None, sigma=1.):
        if gp_link is None:
            gp_link = link_functions.Identity()
            # gp_link = link_functions.Log()

        super(LogGaussian, self).__init__(gp_link, name='loggaussian')

        self.sigma = Param('sigma', sigma, Logexp())
        self.variance = Param('variance', sigma**2, Logexp())
        self.link_parameter(self.variance)
        # self.link_parameter()

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        :param link_f:
        :param y:
        :param Y_metadata:
        :return:
        """
        return np.exp(self.logpdf_link(link_f, y, Y_metadata=Y_metadata))

    def logpdf_link(self, link_f, y, Y_metadata=None):
        """

        :param link_f:
        :param y:
        :param Y_metadata:
        :return:
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        uncensored = (1-c)* (-0.5*np.log(2*np.pi*self.variance) - np.log(y) - (np.log(y)-link_f)**2 /(2*self.variance) )
        censored = c*np.log( 1 - stats.norm.cdf((np.log(y) - link_f)/np.sqrt(self.variance)) )
        logpdf = uncensored + censored
        return logpdf

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        derivative of logpdf wrt link_f param

        :param link_f:
        :param y:
        :param Y_metadata:
        :return:
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        uncensored = (1-c)*(val_scaled2)
        a = (1- stats.norm.cdf(val_scaled))
        # llg(z) = 1. / (1 - norm_cdf(r / sqrt(s2))). * (1 / sqrt(2 * pi * s2). * exp(-1 / (2. * s2). * r. ^ 2));
        censored = c*( 1./a) * (np.exp(-1.* val**2 /(2*self.variance)) / np.sqrt(2*np.pi*self.variance))
        # censored = c * (1. / (1 - stats.norm.cdf(val_scaled))) * (stats.norm.pdf(val_scaled))
        gradient = uncensored + censored
        return gradient

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """

        :param link_f:
        :param y:
        :param Y_metadata:
        :return:
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = (1-c) *(-1)/self.variance
        censored = c*(-np.exp(-val**2/self.variance) / ( 2*np.pi*self.variance*(a**2) ) +
                      val*np.exp(-(val**2)/(2*self.variance))/( np.sqrt(2*np.pi)*self.variance**(3/2.)*a) )
        hessian = censored + uncensored
        return hessian

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given f, w.r.t shape parameter

        .. math::

        :param inv_link_f: latent variables link(f)
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: float
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = 0
        censored = c *( 2*np.exp(-3*(val**2)/(2*self.variance)) / ((a**3)*(2*np.pi*self.variance)**(3/2.))
                        - val*np.exp(-(val**2)/self.variance)/ ( (a**2)*np.pi*self.variance**2)
                        - val*np.exp(-(val**2)/self.variance)/ ( (a**2)*2*np.pi*self.variance**2)
                        - np.exp(-(val**2)/(2*self.variance))/ ( a*(self.variance**(1.50))*np.sqrt(2*np.pi))
                        + (val**2)*np.exp(-(val**2)/(2*self.variance))/ ( a*np.sqrt(2*np.pi*self.variance)*self.variance**2 ) )
        d3pdf_dlink3 = uncensored + censored
        return d3pdf_dlink3

    def dlogpdf_link_dvar(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given f, w.r.t variance parameter

        .. math::

        :param inv_link_f: latent variables link(f)
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: float
        """

        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = (1-c)*(-0.5/self.variance + (val**2)/(2*(self.variance**2)) )
        censored = c *( val*np.exp(-val**2/ (2*self.variance)) / (a*np.sqrt(2*np.pi)*2*(self.variance**(1.5))) )
        dlogpdf_dvar = uncensored + censored
        # dlogpdf_dvar = dlogpdf_dvar*self.variance
        return dlogpdf_dvar

    def dlogpdf_dlink_dvar(self, link_f, y, Y_metadata=None):
        """
        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: Nx1 array
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = (1-c)*(-val/(self.variance**2))
        censored = c * (-val*np.exp(-val**2/self.variance)/( 4*np.pi*(self.variance**2)*(a**2)) +
                         (-1 + (val**2)/self.variance)*np.exp(-val**2/(2*self.variance) ) /
                        ( a*(np.sqrt(2.*np.pi)*2*self.variance**1.5)) )
        dlik_grad_dsigma = uncensored + censored
        # dlik_grad_dsigma = dlik_grad_dsigma*self.variance
        return dlik_grad_dsigma

    def d2logpdf_dlink2_dvar(self, link_f, y, Y_metadata=None):
        """

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: Nx1 array
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        val = np.log(y) - link_f
        val_scaled = val/np.sqrt(self.variance)
        val_scaled2 = val/self.variance
        a = (1 - stats.norm.cdf(val_scaled))
        uncensored = (1-c)*( 1./(self.variance**2) )
        # censored =c * ( (1./a**3.*r/(np.sqrt(8*np.pi**3)*s2**(2.5))*np.exp(-3/(2.*s2)*r**2)
        #            + 1./a**2 /(4.*np.pi*s2**2)*np.exp(-r**2./s2)
        #            - 1./a**2*r**2/(2*np.pi*s2**3)*np.exp(-r**2./s2)
        #            + 1./a**2.*1./(4*np.pi*s2**2)*np.exp(-r**2/s2)
        #            - 1./a*r/(np.sqrt(2*np.pi)*2*s2**(2.5))*np.exp(-r**2/(2*s2))
        #            - 1./a**2.*r**2./(4*np.pi*s2**3)*np.exp(-r**2/s2)
        #            - 1./a*r/(np.sqrt(2*np.pi)*s2**(2.5))*np.exp(-r**2/(2*s2))
        #            + 1./a*r**3./(np.sqrt(2*np.pi)*2*s2**(3.5))*np.exp(-r**2/(2*s2))) )
        censored = c*( val*np.exp(-3*(val**2)/(2*self.variance) )/ ((a**3)*np.sqrt(8*np.pi**3)*self.variance**(5/2.))
                       + np.exp(-val**2/self.variance)/((a**2)*4*np.pi*self.variance**2)
                       - np.exp(-val**2/self.variance)*val**2 / ((a**2)*2*np.pi*self.variance**3)
                       + np.exp(-val**2/self.variance)/ ( (a**2)*4*np.pi*self.variance**2)
                       - np.exp(-val**2/ (2*self.variance))*val / ( a*np.sqrt(2*np.pi)*2*self.variance**(5/2.))
                       - np.exp(-val**2/self.variance)*(val**2) / ((a**2)*4*np.pi*self.variance**3)
                       - np.exp(-val**2/ (2*self.variance))*val/ (a*np.sqrt(2*np.pi)*self.variance**(5/2.))
                       + np.exp(-val**2/ (2*self.variance))*(val**3) / (a*np.sqrt(2*np.pi)*2*self.variance**(7/2.)) )
        dlik_hess_dsigma = uncensored + censored
        return dlik_hess_dsigma

    # def d3logpdf_dlink3_dvar(self, link_f, y, Y_metadata=None):
    #     """
    #     :param link_f: latent variables link(f)
    #     :type link_f: Nx1 array
    #     :param y: data
    #     :type y: Nx1 array
    #     :param Y_metadata: Y_metadata not used in gaussian
    #     :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
    #     :rtype: Nx1 array
    #     """
    #     c = np.zeros_like(y)
    #     if Y_metadata is not None and 'censored' in Y_metadata.keys():
    #         c = Y_metadata['censored']
    #     val =np.log(y) -link_f
    #     val_scaled = val/np.sqrt(self.variance)
    #     val_scaled2 = val/self.variance
    #     a = (1 - stats.norm.cdf(val_scaled))
    #     uncensored = 0
    #     censored = 0
    #     dlik3_grad_sigma3 = uncensored + censored
    #     return dlik3_grad_sigma3

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        """
        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: Nx1 array
        """
        dlogpdf_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dtheta[0,:,:] = self.dlogpdf_link_dvar(f,y,Y_metadata=Y_metadata)
        return dlogpdf_dtheta

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        """
        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: Nx1 array
        """
        dlogpdf_dlink_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dlink_dtheta[0,:,:] = self.dlogpdf_dlink_dvar(f,y,Y_metadata=Y_metadata)
        return dlogpdf_dlink_dtheta

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        """
        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: Nx1 array
        :return:
        """
        d2logpdf_dlink2_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        d2logpdf_dlink2_dtheta[0,:,:] = self.d2logpdf_dlink2_dvar(f,y,Y_metadata=Y_metadata)
        return d2logpdf_dlink2_dtheta

    def update_gradients(self, grads):
        """

        :param grads:
        :return:
        """
        self.variance.gradient = grads[0]

    def samples(self, gp, Y_metadata=None):
        """

        :param gp:
        :param Y_metadata:
        :return:
        """
        orig_shape = gp.shape
        gp = gp.flatten()
