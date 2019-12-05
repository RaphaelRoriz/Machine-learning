import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv() #dataset


class AdalineGD(object):
	
	def __init__(self,eta = 0.01,n_iter = 10):
		self.eta = eta #learning range ,between 0.0 and 1.0 , o quap rapido a rede aprende
		self.n_iter = n_iter #passes over training dataset

	
	def fit(self,X,y):#fit training data
		self.w_ = np.zeros(1 + X.shape[1]) #training vectors
		self.cost_ = [] #target values

		for _ in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = (errors **2).sum() / 2
			self.cost_.append(cost)
 
		return self

	
	def net_input(self , X):#calculate net input
		return np.dot(X,self.w_[1:]) + self.w_[0]

	
	def predict(self,X):#return class label after unit step
		return np.where(self.net_input(X) >= 0.0,1,-1)



#use yout data and plot your graphics below:
