from corels import CorelsClassifier
from utils.utils import binarize_data


class Corels():
    """ Class for computing Certifiably Optimal RulE ListS given a dataset """

    def __init__(self, params):
        self.params = params
        self.binfeat, self.Xbin = binarize_data(self.params["X"], self.params["features"])

    def fit(self):
        C = CorelsClassifier(max_card=2, verbosity=["loud", "samples"])
        C.fit(self.Xbin, self.params["y"], features=self.binfeat, prediction_name="Fire Risk")
        return C

    def plot(self, classifier):
        print(classifier.rl())

    def run_all(self):
        c = self.fit()
        print("Rule list for Fire Risk:")
        self.plot(c)



