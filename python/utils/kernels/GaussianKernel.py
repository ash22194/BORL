import numpy as np
from numpy.linalg import norm
from math import exp

class GaussianKernel:
	def __init__(self,sigma):
		self.sigma = sigma

	def kernel(self,x,y):
		assert ((x.shape[0]==y.shape[0])), 'Check kernel inputs'
		if (self.sigma==0):
			return (x==y).all()

		return exp(-norm(x-y)**2/self.sigma)