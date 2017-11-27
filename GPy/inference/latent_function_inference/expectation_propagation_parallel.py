import numpy as np
from ...util.linalg import jitchol, DSYR, dtrtrs, dtrtri, pdinv, dpotrs, tdot, symmetrify
from paramz import ObsAr
from . import ExactGaussianInference, VarDTC, VarDTC_minibatch
from ...util import diag
from .posterior import PosteriorEP as Posterior
from ...likelihoods import Gaussian
from . import LatentFunctionInference
from . import EPBase
from .expectation_propagation import cavityParams, gaussianApproximation, marginalMoments, posteriorParamsBase ,posteriorParams

log_2_pi = np.log(2*np.pi)


class posteriorParamsVar(posteriorParamsBase):
    def __init__(self, mu, Sigma_diag):
        super(posteriorParamsVar, self).__init__(mu, Sigma_diag)

    def _update_rank1(self, LLT, Kmn, delta_v, delta_tau, i, covCorr_trace):
        #DSYR(Sigma, Sigma[:,i].copy(), -delta_tau/(1.+ delta_tau*Sigma[i,i]))
        DSYR(LLT,Kmn[:,i].copy(),delta_tau)
        L = jitchol(LLT)
        V,info = dtrtrs(L,Kmn,lower=1)
        self.Sigma_diag = np.maximum(np.sum(V*V,-2), np.finfo(float).eps)  #diag(K_nm (L L^\top)^(-1)) K_mn
        si = np.sum(V.T*V[:,i],-1) #(V V^\top)[:,i]
        self.mu += (delta_v-delta_tau*self.mu[i])*si
        self.Sigma_diag += covCorr_trace

        #mu = np.dot(Sigma, v_tilde)

    @staticmethod
    def _recompute(LLT0, Kmn_local, psi2_full, ga_approx, covCorr_trace):
        LLT = LLT0 + psi2_full
        #LLT = LLT0 + np.dot(Kmn*ga_approx.tau[None,:],Kmn.T)
        L = jitchol(LLT)
        V, _ = dtrtrs(L,Kmn_local,lower=1)
        #Sigma_diag = np.sum(V*V,-2)
        #Knmv_tilde = np.dot(Kmn,v_tilde)
        #mu = np.dot(V2.T,Knmv_tilde)
        Sigma_local = np.dot(V.T,V)
        # Sigma += np.diag(covCorr_trace)
        mu_local = np.dot(Kmn_local.T, psi1Y_full)
        # mu_local = np.dot(Sigma, ga_approx.v)
        Sigma_diag_local = np.diag(Sigma_local).copy()
        # print(Sigma_diag.shape)
        # print(covCorr_trace.shape)
        Sigma_diag = Sigma_diag_local + covCorr_trace
        return posteriorParamsVar(mu_local, Sigma_diag_local), LLT



class EP_Var_Parallel(EPBase, VarDTC_minibatch):

    def gatherPsiStatsLocal(self, kern, X, Z, Y, beta, uncertain_inputs):

        num_data, output_dim = Y.shape
        # see whether we've got a different noise variance for each datum
        het_noise = beta.size > 1
        # VVT_factor is a matrix such that tdot(VVT_factor) = VVT...this is for efficiency!
        # self.YYTfactor = beta*self.get_YYTfactor(Y)
        if self.Y_speedup and not het_noise:
            YYT_factor = self.get_YYTfactor(Y)
        else:
            YYT_factor = Y

        n_start = self.batch_pos

        batchsize = num_data if self.batchsize is None else self.batchsize
        n_end = min(batchsize + n_start, num_data)
        if n_end == num_data:
            isEnd = True
            self.batch_pos = 0
        else:
            isEnd = False
            self.batch_pos = n_end

        if batchsize == num_data:
            Y_slice = YYT_factor
            X_slice = X
        else:
            Y_slice = YYT_factor[n_start:n_end]
            X_slice = X[n_start:n_end]

        if not uncertain_inputs:
            psi0 = kern.Kdiag(X_slice)
            psi1 = kern.K(X_slice, Z)
            psi2 = None
            betapsi1 = np.einsum('n,nm->nm', beta, psi1)
        elif het_noise:
            psi0 = kern.psi0(Z, X_slice)
            psi1 = kern.psi1(Z, X_slice)
            psi2 = kern.psi2(Z, X_slice)
            betapsi1 = np.einsum('n,nm->nm', beta, psi1)

        if het_noise:
            beta = beta[n_start]  # assuming batchsize==1

        betaY = beta * Y_slice

        return psi0, psi1, psi2, betaY

    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None, Lm=None, Knn_diag=None, dL_dKmm=None,
                  psi0=None, psi1=None, psi2=None):
        if self.always_reset:
            self.reset()

        num_data, output_dim = Y.shape
        assert output_dim == 1, "ep in 1D only (for now!)"
        beta = 1./np.fmax(likelihood.variance, 1e-6)


        if Lm is None:
            Kmm = kern.K(Z)
            Lm = jitchol(Kmm)

        if Knn_diag is None:
            Knn_diag = kern.Kdiag(X)

        psi0_local, psi1_local, psi2_local, betaY = self.gatherPsiStatsLocal(kern, X, Z, Y, beta)

        # if psi1 is None:
        #     try:
        #         Kmn = kern.K(Z, X)
        #     except TypeError:
        #         Kmn = kern.psi1(Z, X).T
        # else:
        #     Kmn = psi1.T

        if self.ep_mode == "nested":
            # Force EP at each step of the optimization
            self._ep_approximation = None
            post_params, ga_approx, log_Z_tilde = self._ep_approximation = self.expectation_propagation(psi0_local, Kmm, psi1_local,
                                                                                                        betaY, likelihood,
                                                                                                        Y_metadata)
        elif self.ep_mode == "alternated":
            if getattr(self, '_ep_approximation', None) is None:
                # if we don't yet have the results of runnign EP, run EP and store the computed factors in self._ep_approximation
                post_params, ga_approx, log_Z_tilde = self._ep_approximation = self.expectation_propagation(psi0_local, Kmm,
                                                                                                            psi1_local, betaY,
                                                                                                            likelihood,
                                                                                                            Y_metadata)
            else:
                # if we've already run EP, just use the existing approximation stored in self._ep_approximation
                post_params, ga_approx, log_Z_tilde = self._ep_approximation
        else:
            raise ValueError("ep_mode value not valid")

        # TODO: work landmark
        mu_tilde = ga_approx.v / ga_approx.tau.astype(float)

        return super(EP_Var_Parallel, self).inference(kern, X, Z, likelihood, ObsAr(mu_tilde[:, None]),
                                            mean_function=mean_function,
                                            Y_metadata=Y_metadata,
                                            precision=ga_approx.tau,
                                            Lm=Lm, dL_dKmm=dL_dKmm,
                                            psi0=psi0, psi1=psi1, psi2=psi2, Z_tilde=log_Z_tilde)

    def expectation_propagation(self, psi0_local, Kmm, psi1_local, betaY, likelihood, Y_metadata):
        num_data, output_dim = betaY.shape
        assert output_dim == 1, "This EP methods only works for 1D outputs"

        # Makes computing the sign quicker if we work with numpy arrays rather
        # than ObsArrays
        betaY = betaY.values.copy()

        # Initial values - Marginal moments, cavity params, gaussian approximation params and posterior params
        marg_moments = marginalMoments(num_data)
        cav_params = cavityParams(num_data)
        ga_approx, post_params, LLT0, LLT, covCorr_diag = self._init_approximations(psi0_local, Kmm, psi1_local, num_data)

        stop = False
        iterations = 0
        while not stop and (iterations < self.max_iters):
            self._local_updates(num_data, LLT0, LLT, psi1_local, cav_params, post_params, marg_moments, ga_approx, likelihood, betaY, Y_metadata, covCorr_diag)
            #(re) compute Sigma, Sigma_diag and mu using full Cholesky decompy
            post_params, LLT = posteriorParamsVar._recompute(LLT0, psi1_local, ga_approx, covCorr_diag)
            post_params.Sigma_diag = np.maximum(post_params.Sigma_diag, np.finfo(float).eps)

            #monitor convergence
            if iterations > 0:
                stop = self._stop_criteria(ga_approx)
            self.ga_approx_old = gaussianApproximation(ga_approx.v.copy(), ga_approx.tau.copy())
            iterations += 1
            # print iterations

        log_Z_tilde = self._log_Z_tilde(marg_moments, ga_approx, cav_params)

        return post_params, ga_approx, log_Z_tilde
        # posterior covariance matrix correction- "uncollapsing"= Knn - Qnn.
        # covCorr = Knn - Qnn


    def _init_approximations(self, psi0_local, Kmm, psi1_local, num_data):
        #initial values - Gaussian factors
        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        LLT0 = Kmm.copy()
        Lm = jitchol(LLT0) #K_m = L_m L_m^\top
        Vm,info = dtrtrs(Lm, psi1_local,lower=1)
        # Lmi = dtrtri(Lm)
        # Kmmi = np.dot(Lmi.T,Lmi)
        # KmmiKmn = np.dot(Kmmi,Kmn)
        # Knm = Kmn.T
        # KmnKmmiKmn = np.dot(Knm, KmmiKmn)
        # KnmKmmiKmn = np.dot(Vm, Vm.T)
        Qnn_diag = np.sum(Vm*Vm, -2)
        covCorr_diag = psi0_local - Qnn_diag

        # diag.add(LLT0, 1e-8)
        if self.ga_approx_old is None:
            #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
            LLT = LLT0.copy() #Sigma = K.copy()
            mu = np.zeros(num_data)
            Sigma_diag = Qnn_diag.copy() + 1e-8
            v_tilde, tau_tilde = np.zeros((2, num_data))
            ga_approx = gaussianApproximation(v_tilde, tau_tilde)
            post_params = posteriorParamsVar(mu, Sigma_diag)

        else:
            assert self.ga_approx_old.v.size == num_data, "data size mis-match: did you change the data? try resetting!"
            ga_approx_local = gaussianApproximation(self.ga_approx_old.v, self.ga_approx_old.tau)
            post_params, LLT = posteriorParamsVar._recompute(LLT0, psi1_local, ga_approx_local, covCorr_diag)
            post_params.Sigma_diag += 1e-8

            # TODO: Check the log-marginal under both conditions and choose the best one

        return (ga_approx, post_params, LLT0, LLT, covCorr_diag)

    def _log_Z_tilde(self, marg_moments, ga_approx, cav_params):
        mu_tilde = ga_approx.v/ga_approx.tau
        mu_cav = cav_params.v/cav_params.tau
        sigma2_sigma2tilde = 1./cav_params.tau + 1./ga_approx.tau

        return np.sum((np.log(marg_moments.Z_hat) + 0.5*np.log(2*np.pi) + 0.5*np.log(sigma2_sigma2tilde)
                         + 0.5*((mu_cav - mu_tilde)**2) / (sigma2_sigma2tilde)))

    def _local_updates(self, num_data, LLT0, LLT, Kmn, cav_params, post_params, marg_moments, ga_approx, likelihood, Y, Y_metadata, covCorr_trace, update_order=None):
        if update_order is None:
            update_order = np.random.permutation(num_data)
        for i in update_order:

            #Cavity distribution parameters
            cav_params._update_i(self.eta, ga_approx, post_params, i)


            if Y_metadata is not None:
                # Pick out the relavent metadata for Yi
                Y_metadata_i = {}
                for key in Y_metadata.keys():
                    Y_metadata_i[key] = Y_metadata[key][i, :]
            else:
                Y_metadata_i = None

            #Marginal moments
            marg_moments.Z_hat[i], marg_moments.mu_hat[i], marg_moments.sigma2_hat[i] = likelihood.moments_match_ep(Y[i], cav_params.tau[i], cav_params.v[i], Y_metadata_i=Y_metadata_i)
            #Site parameters update
            delta_tau, delta_v = ga_approx._update_i(self.eta, self.delta, post_params, marg_moments, i)

            #Posterior distribution parameters update
            if self.parallel_updates == False:
                post_params._update_rank1(LLT, Kmn, delta_v, delta_tau, i, covCorr_trace)