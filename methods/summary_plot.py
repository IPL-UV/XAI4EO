# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:12:14 2021

@author: Michele Ronco
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import sys
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import cm 


class Summary():
    """ Class for summarizing black-box model results """
    
    def __init__(self, params):
        self.params = params
        
        
    def fit(self):
        y_pred = self.params["model"].predict(self.params["X"])        
        accuracy = accuracy_score(self.params["y"], y_pred)
        print("Model accuracy = ", accuracy)
        f1 = f1_score(self.params["y"], y_pred)
        print("Model F1 = ", f1)
        
        return self.params["model"]
    
    def plot(self, model):
        plt.figure(figsize=(8,8))
        plt.title('Classified train set: ')
        probs = model.predict_proba(self.params["X"])[:,1]
        plt.scatter(self.params["X"][:,self.params["features"].index(self.params["2d_proj"][0])], 
                    self.params["X"][:,self.params["features"].index(self.params["2d_proj"][1])], c=probs, cmap=cm.coolwarm)
        plt.colorbar()
        plt.xlabel(self.params["2d_proj"][0])
        plt.ylabel(self.params["2d_proj"][1])
        plt.show()
        
    
    def run_all(self):
        model = self.fit()
        self.plot(model)
        