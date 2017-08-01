import numpy as np
import GPy

class TVBGP_regression(np.testing.TestCase):
    """
    Inference using tilted variational Bayes(Variational EP)

    """
    def setUp(self):
        X = np.linspace(0,10,100).reshape(-1,1)
        Y = np.sin(X) + np.random.randn(*X.shape)*0.1
        Y[50] += 3

        lik = GPy.likelihoods.StudentT(deg_free=4)
        k = GPy.kern.RBF(1, lengthscale=3.) + GPy.kern.White(1, 1e-5)

        self.m = GPy.core.TVB(X,Y,likelihood=lik, kernel=k)

    def test_grad(self):
        self.m.checkgrad(step=1e-4)
        self.m.optimize(optimizer='lbfgs')
        self.m.checkgrad(step=1e-4)
        # print self.m


class TVBGP_survival(np.testing.TestCase):
    """
    Inference using TVB GP for survival likelihood- loglogistic one ...
    """
    def setUp(self):
        X = np.linspace(0,10,50).reshape(-1,1)
        Y = np.sin(X) + np.random.randn(*X.shape)*0.1
        N = Y.flatten().size
        positive_Y = np.exp(Y.copy())*100
        int_pos_Y = np.rint(positive_Y)
        censored = np.zeros_like(Y)
        random_inds =np.random.choice(N, int(N/2), replace=True)
        censored[random_inds] = 1
        Y_metadata = {}
        Y_metadata['censored'] = censored
        lik = GPy.likelihoods.LogLogistic()
        k = GPy.kern.RBF(1, lengthscale=3.) + GPy.kern.White(1, 1e-5)

        self.m = GPy.core.TVB(X,int_pos_Y,likelihood=lik, kernel=k)

    def test_grad(self):
        self.m.checkgrad(step=1e-4)
        # self.m.checkgrad(self.m.checkgrad())
        # self.m.optimize()


class TVGP_classification(np.testing.TestCase):
    """
    Inference in TVGP with Bernoulli likelihood
    """
    def setUp(self):
        X = np.linspace(0,10,50).reshape(-1,1)
        Y = np.sin(X) + np.random.randn(*X.shape)*0.1
        N = Y.flatten().size
        binary_Y = Y.copy()
        binary_Y[binary_Y>0] = 1
        binary_Y[binary_Y<=0] = 0

        k = GPy.kern.RBF(1, lengthscale=3.) + GPy.kern.White(1, 1e-5)
        lik = GPy.likelihoods.Bernoulli()
        self.m = GPy.core.TVB(X,binary_Y,likelihood=lik,kernel=k)

    def test_grad(self):
        self.m.checkgrad(step=1e-4)
        self.m.optimize()
        self.m.checkgrad()
        # self.m.optimize()