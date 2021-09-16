import numpy as np
import matplotlib.pyplot as plt
import sys
from utils.utils import unique_values
from utils.utils import print_instance


class PDP():
    """ Class for creating partial dependency plots
    showing the univariate dependence of the model wrt a given predictor """
    
    def __init__(self, params):
        self.params = params
        self.fdict = dict(list(zip(self.params["features"], range(self.params["X"].shape[1]))))
        self.feat = self.fdict[self.params["predictor"]]
        self.Xred = unique_values(self.params["X"],self.feat,self.params["decimal"])
        if self.Xred.shape[0]>self.params["nsample"]:
            self.Xred = self.Xred[np.random.choice(self.Xred.shape[0], self.params["nsample"], replace=False),:]

        print("Estimating ", self.params["predictor"], " PDP with ", self.Xred.shape[0], " samples!")
        
        
    def fit(self):
        if self.params["method"] == "PDP":
            yd = []
            for i in range(self.Xred.shape[0]):
                ydi = []
                for j in range(self.Xred.shape[0]):
                    Xp=self.Xred.copy()
                    p = Xp[j,:]
                    p[self.feat] = Xp[i,self.feat]
                    if self.params["type"] == "predict":
                        ydi.append(self.params["model"].predict(p.reshape(1,-1)))
                    elif self.params["type"] == "predict_proba":
                        ydi.append(self.params["model"].predict_proba(p.reshape(1,-1))[:,1])
                    else:
                        sys.exit("Specify type: predict probabilities or values/class!")
        
                yd.append(np.sum(ydi)/self.Xred.shape[0])
            
            return yd

        elif self.params["method"] == "ICE":
            yi = []
            fi = []
            for inst in self.params["instance"]:
                Xp = self.params["X"].copy()
                p = Xp[inst, :]
                print_instance(p, self.params["features"], inst)
                f = np.random.uniform(low=min(Xp[:,self.feat]), high=max(Xp[:,self.feat]), size=(self.params["nsample"],))
                f = np.append(f, p[self.feat])
                Xice = np.array([list(p),]*(1+self.params["nsample"]))
                Xice[:,self.feat] = f 
                if self.params["type"] == "predict":
                    yd = self.params["model"].predict(Xice)
                elif self.params["type"] == "predict_proba":
                    yd = self.params["model"].predict_proba(Xice)[:,1]
                
                yi.append(yd)
                fi.append(f)
            
            return (yi, fi)
            
        else: 
            sys.exit("Specify method: PDP or ICE!!")


    def plot(self, ypartial, x = []):
        if self.params["method"] =="PDP":
            plt.figure(figsize=(10,10))
            x, y = list(zip(*sorted(zip(self.Xred[:,self.feat], ypartial))))
            plt.plot(x, y)
            plt.title('Partial dependency plot')
            plt.ylabel('Probability')
            plt.xlabel(self.params["predictor"])
            plt.savefig(self.params["path"])
            plt.show()

        elif self.params["method"] =="ICE":
            plt.figure(figsize=(10,10))
            for i in range(len(ypartial)):
                x0, y0 = list(zip(*sorted(zip(x[i], ypartial[i]))))
                plt.plot(x0, y0)
            
            plt.title('Individual conditional expectation')
            plt.ylabel('Probability')
            plt.xlabel(self.params["predictor"])
            plt.legend(self.params["instance"], loc='best')
            plt.savefig(self.params["path"])
            plt.show()

        else: 
            sys.exit("Specify method: PDP or ICE!!")

    
    def run_all(self):
        if self.params["method"] =="PDP":
            out_fit = self.fit()
            if self.params["type"] == "predict":
                ytot = self.params["model"].predict(self.params["X"])
            elif self.params["type"] == "predict_proba":
                ytot = self.params["model"].predict_proba(self.params["X"])[:,1]
            else:
                sys.exit("Specify type: predict probabilities or values/class!")
            importance = abs(max(out_fit)-min(out_fit))/abs(max(ytot)-min(ytot))
            print("Feature ", self.params["predictor"], " importance: ", importance)
            self.plot(out_fit)
        
        elif self.params["method"] =="ICE":
            out_fit, x = self.fit()
            if self.params["type"] == "predict":
                ytot = self.params["model"].predict(self.params["X"])
            elif self.params["type"] == "predict_proba":
                ytot = self.params["model"].predict_proba(self.params["X"])[:,1]
            else:
                sys.exit("Specify type: predict probabilities or values/class!")
            out_tot = list(map(np.mean, zip(*out_fit)))
            importance = abs(max(out_tot)-min(out_tot))/abs(max(ytot)-min(ytot))
            print("Feature ", self.params["predictor"], " importance: ", importance)
            self.plot(out_fit, x=x)
        
        else:
            sys.exit("Specify method: PDP or ICE!!")
        



