import numpy as np 
from lime.lime_tabular import LimeTabularExplainer

class Lime():
    """ Class for single instance explanation with local linear approximation of the model """

    def __init__(self, params):
        self.params = params
        
    def fit(self):
        expl = LimeTabularExplainer(self.params["X"], feature_names=self.params["features"], 
            class_names=self.params["target_names"], discretize_continuous=False)
        expi = expl.explain_instance(self.params["instance"], self.params["model"].predict_proba)
        
        return expi
        
    def plot(self, expi):
        expi.show_in_notebook()

    def run_all(self):
        instance_expl = self.fit()
        self.plot(instance_expl)
        

