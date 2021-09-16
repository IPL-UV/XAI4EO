# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:57:01 2021

@author: Michele Ronco 
"""

from methods.feature_permutation import FeaturePermutation
import numpy as np 
from sklearn.linear_model import LinearRegression
from methods.partial_dependency import PDP


np.random.seed(10) 

N = 100
X = np.random.rand(N,5)
y = 3*X[:,3] + 6*X[:,1] + np.random.normal(0,0.5,N) + 5
model = LinearRegression().fit(X, y)

features = ["f"+str(i) for i in range(X.shape[1])]

config0 = {
    "model": model,
    "X": X, 
    "y": y,
    "features":features,
    "Np": 100,
    "cost":"mse",
}

fp = FeaturePermutation(config0)
fi = fp.fit()

predictor = "f0"

config1 = {
    "model": model,
    "X": X, 
    "features": features,
    "predictor":predictor,
}

pdep = PDP(config1)
pdep.fit()

