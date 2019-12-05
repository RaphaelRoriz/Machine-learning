import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.data',header = None)


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


y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100 , [0,2]].values

fig,ax = plt.subplots(nrows = 1, ncols = 2 , figsize = (8,4))
ada1 = AdalineGD(n_iter = 10,eta = 0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_) + 1),np.log10(ada1.cost_), marker = 'o')
ax[0].set_ylabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('Adaline - learning rate 0.01')

ada2 = AdalineGD(n_iter = 10,eta = 0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_) + 1),ada2.cost_, marker = 'o')
ax[1].set_ylabel('Epochs')
ax[1].set_ylabel('log(sum-squared-error)')
ax[1].set_title('Adaline - learning rate 0.0001')
plt.show()

