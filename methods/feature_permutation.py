# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:28:47 2021

@author: Michele Ronco 
"""
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from utils.utils import mse, cross_entropy 

random.seed(10)

class FeaturePermutation():
    """ Class that contains permutation techniques
    to rank the predictors of a model """
    
    def __init__(self, params):
        self.params = params
        cost_functions = {"mse":mse, "cross-entropy":cross_entropy}
        self.cf = cost_functions[self.params["cost"]]
        
    
    def fit(self):        
        fi = []
        fp = []
        if self.params["type"] =="predict":
            e0 = self.cf(self.params["y"],self.params["model"].predict(self.params["X"]))
        elif self.params["type"] =="predict_proba":
            e0 = self.cf(self.params["y"],self.params["model"].predict_proba(self.params["X"])[:,1])   
        else:
            sys.exit("Specify type: predict probabilities or values/class!")

        
        for n in range(self.params["Np"]):
            random.seed(n)
            for f in range(self.params["X"].shape[1]):
                Xp=self.params["X"].copy()
                random.shuffle(Xp[:,f])
                if self.params["type"] =="predict":
                    ep = self.cf(self.params["y"],self.params["model"].predict(Xp))
                elif self.params["type"] =="predict_proba":
                    ep = self.cf(self.params["y"],self.params["model"].predict_proba(Xp)[:,1])
                else:
                    sys.exit("Specify type: predict probabilities or values/class!")

                fp.append(ep/e0)
                
            fi.append(fp)
            fp = []
            
        fi_mean = np.array(fi).mean(axis=0)
        fi_std = np.array(fi).std(axis=0)
        
        return {"SCORE": fi_mean,
                "ERROR": fi_std,
                "RESULT": sorted(zip(fi_mean,self.params["features"]),reverse=True)
                }

    def plot(self, result):
        fi_vector, ordered_features = list(zip(*result))
        plt.figure(figsize=(10,10))
        plt.barh(ordered_features, fi_vector)
        plt.title('Feature importance')
        plt.ylabel('Features')
        plt.xlabel('Error permuted over original error')
        plt.savefig(self.params["path"])
        plt.show()

    
    def run_all(self):
        out_fit = self.fit()
        _, ordered_features = list(zip(*out_fit["RESULT"]))
        ranking = dict(zip(ordered_features, range(1,len(ordered_features)+1)))

        for i in range(len(self.params["features"])):
            print("Feature ", self.params["features"][i], 
                "is ranked number ",  ranking[self.params["features"][i]], " with a score of ", 
                out_fit["SCORE"][i],"+/-", out_fit["ERROR"][i])

        self.plot(out_fit["RESULT"])



 
               
        


