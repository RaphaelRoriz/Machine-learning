import numpy as np 
import pandas as pd

class Perceptron(object):
	
	def __init__(self,eta = 0.01,n_iter = 10):
		self.eta = eta #learning range ,between 0.0 and 1.0
		self.n_iter = n_iter #passes over training dataset

	
	def fit(self,X,y):#fit training data
		self.w_ = np.zeros(1 + X.shape[1]) #training vectors
		self.errors_ = [] #target values

		for _ in range(self.n_iter):
			errors = 0
			for xi ,target in zip(X,y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors +=int(update != 0.0)
			self.errors_.append(errors)
		return self

	
	def net_input(self , X):#calculate net input
		return np.dot(X,self.w_[1:]) + self.w_[0]

	
	def predict(self,X):#return class label after unit step
		return np.where(self.net_input(X) >= 0.0,1,-1)



