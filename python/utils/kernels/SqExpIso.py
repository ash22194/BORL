import numpy as np
from numpy.linalg import norm
from math import exp

class SqExpIso:
	def __init__(self,sigma):
		self.sigma = sigma

	def kernel(self,x,y):
		assert ((x.shape[0]==y.shape[0])), 'Check kernel inputs'
		if (self.sigma==0):
			return (x==y).all()

		return (np.exp(-np.sum(np.power((x-y)/self.sigma, 2), axis=0)))[:,np.newaxis]