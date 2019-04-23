import numpy as np

class SqExpIso:
    def __init__(self,sigma):
        self.sigma = sigma

    def kernel(self, x, y):

        assert ((x.shape==y.shape)), 'Check dimensions of kernel inputs'

        if (self.sigma==0):
            return (x==y).all()

        return np.exp(-np.sum(((x-y)/self.sigma)**2, axis=0))[:, np.newaxis]

class SqExpArd:
    def __init__(self, sigma_l, sigma_f):
        self.sigma_l = sigma_l
        self.sigma_f = sigma_f**2

    def kernel(self, x, y):

        assert ((x.shape==y.shape)), 'Check dimensions of kernel inputs'

        return (self.sigma_f)*np.exp(-np.sum(((x-y)/np.repeat(self.sigma_l, x.shape[1], axis=1))**2, axis=0))[:, np.newaxis]