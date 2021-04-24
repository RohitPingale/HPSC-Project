import pandas as pd
import numpy as np
from math import sqrt
import time

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt 

digits = load_digits()
print(digits.data.shape)


#image representatio of the data
# plt.gray() 
# plt.matshow(digits.images[0]) 
# plt.show() 


X = digits.data 
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)



class KNN:
    def __init__(self, K=3):
        self.K = K

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def euc_dist(self, x1, x2):
    	return np.sqrt(np.sum((x1-x2)**2))

    def predict(self, X_test):
	    predictions = [] 
	    for i in range(len(X_test)):
	        dist = np.array([self.euc_dist(X_test[i], x_t) for x_t in self.X_train])
	        dist_sorted = dist.argsort()[:self.K]
	        neigh_count = {}
	        for idx in dist_sorted:
	            if self.Y_train[idx] in neigh_count:
	                neigh_count[self.Y_train[idx]] += 1
	            else:
	                neigh_count[self.Y_train[idx]] = 1
	       
	        sorted_neigh_count = sorted(neigh_count.items(), reverse=True)
	        predictions.append(sorted_neigh_count[0][0]) 
	    return predictions

k = 3
start = time.time()
model = KNN(K = k)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
end = time.time()
print(f"Time Taken: {end-start} sec")
print("K = "+str(k)+"; Accuracy: "+str(acc))

