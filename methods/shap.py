import numpy as np 
import shap as shp
import sys
from utils.utils import print_instance

shp.initjs()

class Shap():
    """ Class for single instance explanation from shapely values """

    def __init__(self, params):
        self.params = params
        
    def fit(self): 
        f = lambda x: self.params["model"].predict_proba(x)[:,1]
        med = self.params["X"].median().values.reshape((1,self.params["X"].shape[1]))
        print_instance(med[0,:], self.params["features"], "baseline")
        explainer = shp.Explainer(f, med)
        shap_values = explainer(self.params["Xtest"])
        plot_type = self.params["plot"]
        sample_id = self.params["instance"]

        return explainer, shap_values, plot_type, sample_id
                
    def plot(self, expl, shap_values, plot_type, sample_id):
        if plot_type[0] == "local" and plot_type[1] == "waterfall":
            shp.plots.waterfall(shap_values[sample_id])

        elif plot_type[0] == "global" and plot_type[1] == "scatter":
            shp.plots.scatter(shap_values[:,self.params["predictor"]])

        elif plot_type[0] == "global" and plot_type[1] == "beeswarm":
            shp.plots.beeswarm(shap_values)

        elif plot_type[0] == "global" and plot_type[1] == "bar":
            shp.plots.bar(shap_values)

        elif plot_type[0] == "global" and plot_type[1] == "heatmap":
            shp.plots.heatmap(shap_values)

        elif plot_type[0] == "global" and plot_type[1 =="summary"]:
            shp.summary_plot(shap_values, self.params["Xtest"])

        else:
            sys.exit("Explanation type and plot style specified are not valid!")


    def run_all(self):
        explainer, shap_values, plot_type , sample_id = self.fit()
        self.plot(explainer, shap_values, plot_type, sample_id)


