
from ...util.linalg import mdot, jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri, dpotri, dpotrs, symmetrify
from ...util import diag
from GPy.core.parameterization.variational import VariationalPosterior
import numpy as np
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)


class VarDTC_Hetero_minibatch(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.
    """
    const_jitter = 1e-6

    def __init__(self, batchsize=None, limit=3, mpi_comm=None):

        self.batchsize = batchsize
        self.mpi_comm = mpi_comm
        self.limit = limit
        self.midRes = {}
        self.batch_pos = 0 # the starting position of the current mini-batch
        self.Y_speedup = False # Replace Y with the cholesky factor of YY.T, but the computation of posterior object will be skipped.

    #  this function only needs to be called once otherwise it can be cached ..
    def get_trYYT(self, Y):
        assert Y.dim == 1
        YYT_tr = np.sum(np.square(Y))
        return YYT_tr

    #  the precision values will change, so we cannot really cache this function here ...
    def get_beta_trYYT(self, Y, precision=None):
        assert Y.dim == 1
        YYT_tr_scaled = np.sum(np.square(Y)*precision)
        return YYT_tr_scaled

    # this function can also be Cached
    def get_YYTfactor(self, Y, precision):
        N, D = Y.shape
        YYT_factor = tdot(Y)
        return YYT_factor

    # I am not sure what it does, optimisaition interms of rows and column ordering
    def get_beta_YYTfactor(self, Y, precision):
        N, D = Y.shape
        shp = tdot(Y).shape
        precmat = np.reshape(precision.flatten(), shp)
        return jitchol(precmat*tdot(Y))

    def get_VVTfactor(self, Y, prec):
        return Y*prec

    def gatherPsiStat(self, kern, X, Z, Y, precision):
        assert X.shape[0] == prec.shape[0]
        if precision.ndim == 1:
            precision = precision[:,None]

        precmat = np.reshape(precision, (1, self.num_inducing))
        psi0 = kern.Kdiag(X)
        psi1 = kern.K(X, Z)
        psi2 = psi1[:, :, None] * psi1[:, None, :]
        scaled_psi0 = prec*psi0
        scaled_psi1 = precmat*psi1
        scaled_psi2 = scaled_psi1[:,:,None]*scaled_psi1[:,None,:]
        return psi0, psi1, psi2, scaled_psi0, scaled_psi1, scaled_psi2

    def gatherPsiStatMiniBatch(self, kern, X, Z, Y, beta):

        het_noise = beta.size > 1

        trYYT = self.get_trYYT(Y)
        if self.Y_speedup and not het_noise:
            Y = self.get_YYTfactor(Y)

        num_inducing = Z.shape[0]
        num_data, output_dim = Y.shape
        assert output_dim == 1
        batchsize = num_data if self.batchsize is None else self.batchsize

        psi2_full = np.zeros((num_inducing, num_inducing))  # MxM
        psi1Y_full = np.zeros((output_dim, num_inducing))  # DxM
        psi0_full = 0.
        YRY_full = 0.
        psi0_scaled_full = 0.
        psi1Y_scaled_full = 0.

        for n_start in range(0, num_data, batchsize):
            n_end = min(batchsize + n_start, num_data)
            if batchsize == num_data:
                Y_slice = Y
                X_slice = X
                beta_slice = beta
            else:
                Y_slice = Y[n_start:n_end]
                X_slice = X[n_start:n_end]
                beta_slice = beta[n_start:n_end]

            YRY_full += np.inner(Y_slice*beta_slice, Y_slice)

            psi0 = kern.Kdiag(X_slice)
            psi1 = kern.K(X_slice, Z)
            psi2_full += np.dot(psi1.T, beta_slice*psi1)
            psi0_scaled = psi0*beta_slice

            psi0_full += psi0_scaled.sum()
            psi1Y_scaled_full += np.dot(Y_slice.T,  beta_slice*psi1)  #

        if self.mpi_comm != None:
            from mpi4py import MPI
            psi0_all = np.array(psi0_full)
            psi1Y_all = psi1Y_scaled_full.copy()
            psi2_all = psi2_full.copy()
            YRY_all = np.array(YRY_full)
            self.mpi_comm.Reduce([psi0_full, MPI.DOUBLE], [psi0_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([psi1Y_full, MPI.DOUBLE], [psi1Y_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([psi2_full, MPI.DOUBLE], [psi2_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([YRY_full, MPI.DOUBLE], [YRY_all, MPI.DOUBLE])
            return psi0_all, psi1Y_all, psi2_all, YRY_all

        return psi0_full, psi1Y_scaled_full, psi2_full, YRY_full

    def inference_first(self, kern, X, Z, Y, likelihood, precision=None, Y_metadata=None, Lm=None, dL_dKmm=None, Kuu_sigma=None):
        """
        This function will only calculate loglikelihood by summing over data partitions and dL/dKmm
        :param kern:
        :param X:
        :param Z:
        :param Y:
        :param likelihood:
        :param precision:
        :param Y_metadata:
        :param Lm:
        :param dL_dKmm:
        :param Kuu_sigma:
        :return:
        """

        self.input_dim = Z.shape[1]
        self.num_inducing = Z.shape[0]
        self.Y = Y

        self.hetero = True
        uncertain_inputs = isinstance(X, VariationalPosterior)
        self.beta = precision

        if len(self.beta)==1:
            self.hetero = False

        prectmp = self.precision.flatten()

        VVT_factor = precision*Y
        trYYT, trYYT_scaled = self.get_trYYT(Y)

        # prectmp[]
        if np.array_equal(np.repeat(np.min(prectmp), prectmp.size), prectmp):
            self.hetero = False

        psi0_full, psi1Y_full, psi2_full, YRY_full = self.gatherPsiStatMiniBatch(kern, X, Z, Y, beta, uncertain_inputs)
        Kmm = kern.K(Z).copy()
        diag.add(Kmm, self.const_jitter)
        if not np.isfinite(Kmm).all():
            print(Kmm)

        Lm = jitchol(Kmm)
        LmInv = dtrtri(Lm)

        LmInvPsi2LmInvT = LmInv.dot(psi2_full.dot(LmInv.T))
        LL = jitchol(Lambda)
        LLInv = dtrtri(LL)

        tmp, _ = dtrtrs(Lm, scaled_psi1.T, lower=1)

        logL_R = -np.log(beta).sum()
        logL = -((num_data*log_2_pi+logL_R+psi0_full-np.trace(LmInvPsi2LmInvT))+YRY_full-bbt)/2.-output_dim*logdet_L/2.

        dL_dKmm =  dL_dpsi2R - output_dim*LmInv.T.dot(LmInvPsi2LmInvT).dot(LmInv)/2.

        A = tdot(tmp)  # print A.sum()
        B = np.eye(self.num_inducing) + A
        LB = jitchol(B)
        psi1Vf = np.dot(psi1.T, VVT_factor)
        return logL, dL_dKmm


    def inference_minibatch(self, kern, X, Z, likelihood, Y, precision, Y_metadata):
        """
        The second phase of inference: Computing the derivatives over a minibatch of Y
        Compute: dL_dpsi0, dL_dpsi1, dL_dpsi2, dL_dthetaL
        return a flag showing whether it reached the end of Y (isEnd)
        """

        num_data, output_dim = Y.shape
        assert output_dim ==1

        n_start = self.batch_pos
        batchsize = num_data if self.batchsize is None else self.batchsize
        n_end = min(batchsize+n_start, num_data)

        beta = precision

        if n_end==num_data:
            isEnd = True
            self.batch_pos = 0
        else:
            isEnd = False
            self.batch_pos = n_end

        if batchsize==num_data:
            Y_slice = YYT_factor
            X_slice =X
        else:
            Y_slice = YYT_factor[n_start:n_end]
            X_slice = X[n_start:n_end]
            beta_slice = beta[n_start:n_end]

        if not uncertain_inputs:
            psi0 = kern.Kdiag(X_slice)
            psi1 = kern.K(X_slice, Z)
            psi2 = None
            betapsi1 = np.einsum('n,nm->nm',beta,psi1)
        elif het_noise:
            psi0 = kern.psi0(Z, X_slice)
            psi1 = kern.psi1(Z, X_slice)
            psi2 = kern.psi2(Z, X_slice)
            betapsi1 = np.einsum('n,nm->nm',beta,psi1)

        #======================================================================
        # Load Intermediate Results
        #======================================================================

        dL_dpsi2R = self.midRes['dL_dpsi2R']
        v = self.midRes['v']

        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        dL_dpsi0 = -output_dim * (beta * np.ones((n_end-n_start,)))/2.

        dL_dpsi1 = np.dot(betaY,v.T)

        if uncertain_inputs:
            dL_dpsi2 = beta* dL_dpsi2R
        else:
            dL_dpsi1 += np.dot(betapsi1,dL_dpsi2R)*2.
            dL_dpsi2 = None

        #======================================================================
        # Compute dL_dthetaL
        #======================================================================
