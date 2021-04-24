import pandas as pd
import numpy as np
from math import sqrt
import time

import numba
from numba import int32, float64
from numba.experimental import jitclass

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

 

@numba.jit(nopython=True)
def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

@numba.jit(nopython=True)
def predict(X_train, y_train, test, K):
    dist = np.array([euc_dist(test, x_t) for x_t in X_train])
    dist_sorted = dist.argsort()[:K]
    neigh_count = {}
    for idx in dist_sorted:
        if y_train[idx] in neigh_count:
            neigh_count[y_train[idx]] += 1
        else:
            neigh_count[y_train[idx]] = 1
       
        sorted_neigh_count = sorted(neigh_count.items(), reverse=True)
    return sorted_neigh_count[0][0]

def predict_numba(X_train, X_test, y_train,K):
    predictions = np.zeros(X_test.shape[0])
    for i in np.arange(X_test.shape[0]):
        predictions[i] = predict(X_train, y_train, X_test[i], K)
    return predictions


k = 3
start = time.time()
pred = predict_numba(X_train, X_test, y_train,k)
acc = accuracy_score(y_test, pred)
end = time.time()
print(f"Time Taken: {end-start} sec")
print("K = "+str(k)+"; Accuracy: "+str(acc))

