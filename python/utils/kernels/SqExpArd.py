import numpy as np

class SqExpArd:
	def __init__(self, sigma_l, sigma_f):
		self.sigma_l = sigma_l
		self.sigma_f = sigma_f**2

	def kernel(self, x, y):
		assert ((len(x.shape)==len(y.shape)) and (x.shape[0]==y.shape[0]) and (x.shape[1]==y.shape[1])), 'Check dimensions of kernel inputs'
		
		return ((self.sigma_f)*np.exp(-np.sum(np.power((x-y)/np.repeat(self.sigma_l, x.shape[1], axis=1), 2), axis=0)))[:,np.newaxis]