from .posterior import PosteriorExact as Posterior
from ...util.linalg import pdinv, dpotrs, tdot, dtrtrs
from ...util import diag
from .expectation_propagation import QuadTilted
import numpy as np
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class TVB(LatentFunctionInference):
    def __init__(self):
        pass

    def reset_params(self, kern, X, Y, likelihood, Ytilde, beta, Y_metadata=None, K=None):
        self.K = K
        self.Ytilde = Ytilde
        self.beta = beta
        self.kern = kern
        self.X = X
        self.num_data = len(Y.flatten())
        if K is None:
            self.K = kern.K(X)

        A = self.K + np.diag(1. / beta)
        self.Ki,self.L, _, self.K_logdet = pdinv(self.K)

        self.Sigma_inv = self.Ki + np.diag(beta)
        self.Sigma,_,_,self.log_det_Sigma_inv = pdinv(self.Sigma_inv)
        self.diag_Sigma = np.diag(self.Sigma)
        self.mu = np.dot(self.Sigma, self.beta*self.Ytilde )
        self.tilted = QuadTilted(Y, likelihood)

        #compute cavity means, vars (all at once!)
        self.cavity_vars = 1./(1./self.diag_Sigma - self.beta)
        self.cavity_means = self.cavity_vars * (self.mu/self.diag_Sigma - self.Ytilde*self.beta)

        self.tilted.compute_moments(self.cavity_means, self.cavity_vars)

    def _log_marginal(self):
        tmp, _ = dtrtrs(self.L, self.tilted.mu_hat, lower=1)
        A = -0.5*self.K_logdet -0.5*np.sum(np.square(tmp)) - 0.5*np.sum(np.diag(self.Ki)*self.tilted.sigma2_hat)

        B = 0.5*np.sum(np.log(self.cavity_vars)) + 0.5*np.sum(np.square(self.cavity_means -
                                                                        self.tilted.mu_hat)/self.cavity_vars)

        C = np.sum(np.log(self.tilted.Z_hat))
        log_marginal = A + B + C
        return log_marginal

    def _log_likelihood_grads(self):
        dcav_vars_dbeta = -(self.Sigma**2/self.diag_Sigma**2 - np.eye(self.num_data))*self.cavity_vars**2
        dcav_means_dYtilde = (self.Sigma * self.beta[:,None]/self.diag_Sigma - np.diag(self.beta))*self.cavity_vars

        dcav_means_dbeta = dcav_vars_dbeta * (self.mu/ self.diag_Sigma - self.Ytilde*self.beta)
        tmp = self.Sigma / self.diag_Sigma
        dcav_means_dbeta += (tmp*(self.Ytilde[:,None] -self.mu[:,None]) + tmp**2*self.mu -
                             np.diag(self.Ytilde))*self.cavity_vars

        # gradient of A
        dA_dq_means = -np.dot(self.Ki, self.tilted.mu_hat)
        dA_dq_vars = -0.5*np.diag(self.Ki)

        dA_dcav_means = dA_dq_vars*self.tilted.dsigma2_dcav_mean + dA_dq_means*self.tilted.dmu_hat_dcav_mean
        dA_dcav_vars = dA_dq_vars*self.tilted.dsigma2_dcav_var
        dA_dcav_vars += dA_dq_means*self.tilted.dmu_hat_dcav_var

        # gradient of B
        dB_dq_means = (self.tilted.mu_hat - self.cavity_means)/self.cavity_vars
        dB_dq_vars = 0.5/self.cavity_vars
        dB_dcav_vars = 0.5/self.cavity_vars - 0.5*(np.square(self.cavity_means - self.tilted.mu_hat) +
                                                   self.tilted.sigma2_hat)/np.square(self.cavity_vars)
        dB_dcav_vars += dB_dq_means*self.tilted.dmu_hat_dcav_var
        dB_dcav_vars += dB_dq_vars*self.tilted.dsigma2_dcav_var
        dB_dcav_means = (self.cavity_means - self.tilted.mu_hat)/self.cavity_vars
        dB_dcav_means += dB_dq_vars*self.tilted.dsigma2_dcav_mean + dB_dq_means*self.tilted.dmu_hat_dcav_mean

        # C
        dC_dcav_means = self.tilted.dZ_dcav_mean/self.tilted.Z_hat
        dC_dcav_vars = self.tilted.dZ_dcav_var/self.tilted.Z_hat

        dL_dcav_vars = dA_dcav_vars + dB_dcav_vars + dC_dcav_vars
        dL_dcav_means = dA_dcav_means + dB_dcav_means + dC_dcav_means

        dL_dbeta = np.dot(dcav_means_dbeta, dL_dcav_means) + np.dot(dcav_vars_dbeta, dL_dcav_vars)
        dL_dYtilde = np.dot(dcav_means_dYtilde, dL_dcav_means)


        # gwt gradients for dL_dK
        KiSigma = np.dot(self.Ki, self.Sigma)

        tmp = dL_dcav_vars*np.square(self.cavity_vars/self.diag_Sigma)
        tmp += dL_dcav_means * (self.cavity_means - self.mu) * self.cavity_vars / np.square(self.diag_Sigma)
        KiSigma = np.dot(self.Ki, self.Sigma)

        tmp = KiSigma*tmp
        dL_dK = np.dot(tmp, KiSigma.T)
        dL_dK += (np.dot(self.Ki, self.mu)[:, None] * (dL_dcav_means * self.cavity_vars / self.diag_Sigma)[None, :]).dot(
            KiSigma.T)

        dL_dK -= 0.5* self.Ki
        Kim = np.dot(self.Ki, self.tilted.mu_hat)
        dL_dK += 0.5*Kim[:,None]*Kim[None,:]

        dL_dK += 0.5*np.dot(self.Ki*self.tilted.sigma2_hat, self.Ki)

        # gradients for likelihood parameters
        if self.tilted.num_params == 0:
            dL_dtheta_lik = np.zeros(0)
        else:
            dL_dtheta_lik = np.sum((dA_dq_means + dB_dq_means)*self.tilted.dmu_hat_dtheta, 1) + \
            np.sum((dA_dq_vars + dB_dq_vars)*self.tilted.dsigma2_dtheta, 1) + \
            np.sum(self.tilted.dZ_dtheta/self.tilted.Z_hat, 1)

        grads = np.hstack((dL_dYtilde, dL_dbeta, dL_dtheta_lik))
        grad_dict = {}
        grad_dict['dL_dK'] = dL_dK
        grad_dict['dL_dYtilde'] = dL_dYtilde
        grad_dict['dL_dbeta'] = dL_dbeta
        # grad_dict['dL_dtheta'] = dL_dtheta_kern
        grad_dict['dL_dthetaL'] = dL_dtheta_lik

        return grad_dict

    def inference(self, kern, X, Y, likelihood, Ytilde, beta, Y_metadata=None, K=None):
        self.reset_params(kern, X, Y, likelihood, Ytilde, beta, Y_metadata=Y_metadata, K=K)

        log_marginal = self._log_marginal()
        grad_dict = self._log_likelihood_grads()
        return log_marginal, grad_dict
