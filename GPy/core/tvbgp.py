# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .gp import GP, Model
from .parameterization.param import Param
from .. import likelihoods
from ..util import linalg
from ..inference.latent_function_inference.tvb import TVB as tvb_inf

import logging
logger = logging.getLogger("tvb gp")

class TVB(GP):
    """
    First stab at trying to code TVB for GPs.

    """

    def __init__(self, X, Y, kernel, likelihood, mean_function=None, name='tvbgp', Y_metadata=None):
        super(TVB, self).__init__(X, Y, kernel, likelihood, mean_function=mean_function, name='tvbgp', Y_metadata=None)

        assert X.ndim == 2
        assert Y.ndim == 2

        self.X = X.copy()
        self.Y = Y.copy()

        self.num_data, self.input_dim = self.X.shape
        self.num_out, self.output_dim = self.Y.shape
        assert self.num_data == self.num_out
        self.jitter = 0.1

        inf_method = tvb_inf()
        self.kernel = kernel
        self.likelihood = likelihood
        self.Y_metadata = Y_metadata
        self.inference_method = inf_method

        self.Ytilde = Param('Ytilde', np.zeros(self.num_data))
        self.beta = Param('beta', np.zeros(self.num_data)+self.jitter)
        self.link_parameter(self.Ytilde)
        self.link_parameter(self.beta)

    def parameters_changed(self):
        self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kernel, self.X, self.Y, self.likelihood, self.Ytilde, self.beta, Y_metadata=self.Y_metadata)
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        self.Ytilde.gradient = self.grad_dict['dL_dYtilde']
        self.beta.gradient = self.grad_dict['dL_dbeta']
        dL_dthetaL = self.grad_dict['dL_dthetaL'].flatten()

        print(self.grad_dict['dL_dthetaL'])
        self.likelihood.update_gradients(dL_dthetaL)

