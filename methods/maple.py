from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from utils.utils import print_instance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

class Maple():
 
    def __init__(self, params):
        self.params = params
        self.X_train, self.X_val, self.MR_train, self.MR_val = train_test_split( self.params["X"], self.params["model"].predict(self.params["X"]), test_size=0.05, random_state=42)
        self.fe_type = self.params["feature_selector"]
        self.n_estimators = self.params["n_estimators"]
        self.num_features = self.X_train.shape[1]
        self.num_train = self.X_train.shape[0]
        self.num_val = self.X_val.shape[0]
        
        
    def fit(self):
        if self.fe_type == "rf":
            fe = RandomForestClassifier(n_estimators = self.n_estimators, min_samples_leaf = 10, max_features =0.5)
        elif self.fe_type == "gbm":
            fe = GradientBoostingClassifier(n_estimators = self.n_estimators, min_samples_leaf = 10, max_features = 0.5, max_depth = None)
        else:
            print("Unknown FE type ", fe)
            import sys
            sys.exit(0)
        fe.fit(self.X_train, self.MR_train)
        print("Fitted ", self.fe_type, " to select features for local approximation of the black-box!")
        self.fe = fe
        
        train_leaf_ids = fe.apply(self.X_train)
        self.train_leaf_ids = train_leaf_ids
        self.val_leaf_ids_list = fe.apply(self.X_val)
                
        # Compute the feature importances: Non-normalized @ Root
        scores = np.zeros(self.num_features)
        if self.fe_type == "rf":
            for i in range(self.n_estimators):
                splits = fe[i].tree_.feature #-2 indicates leaf, index 0 is root
                if splits[0] != -2:
                    scores[splits[0]] += fe[i].tree_.impurity[0] #impurity reduction not normalized per tree
        elif self.fe_type == "gbm":
            for i in range(self.n_estimators):
                splits = fe[i, 0].tree_.feature #-2 indicates leaf, index 0 is root
                if splits[0] != -2:
                    scores[splits[0]] += fe[i, 0].tree_.impurity[0] #impurity reduction not normalized per tree
        self.feature_scores = scores
        mostImpFeats = np.argsort(-scores)
                
        # Find the number of features to use for MAPLE
        retain_best = 0
        acc_best = 0
        for retain in range(1, self.num_features + 1):
            print("Number of features to retain = ", retain_best)
            # Drop less important features for local regression
            self.X_train_p = np.delete(self.X_train, mostImpFeats[retain:], axis = 1)
            self.X_val_p = np.delete(self.X_val, mostImpFeats[retain:], axis = 1)
                        
            lr_predictions = np.empty([self.num_val], dtype=float)
            
            for i in range(self.num_val):
                weights = self.training_point_weights(self.val_leaf_ids_list[i])
                lr_model = LogisticRegression()
                lr_model.fit(self.X_train_p, self.MR_train, weights)
                lr_predictions[i] = lr_model.predict(self.X_val_p[i].reshape(1, -1))
            
            acc_curr = accuracy_score(lr_predictions, self.MR_val)          
            if acc_curr > acc_best and (retain+1)< self.num_features:
                acc_best = acc_curr
                retain_best = retain
                
        X = np.delete(self.X_train, mostImpFeats[retain_best:], axis = 1)
        
        return retain_best, X
                
    def training_point_weights(self, instance_leaf_ids):
        weights = np.zeros(self.num_train)
        for i in range(self.n_estimators):
            # Get the PNNs for each tree (ones with the same leaf_id)
            PNNs_Leaf_Node = np.where(self.train_leaf_ids[:, i] == instance_leaf_ids[i])
            weights[PNNs_Leaf_Node] += 1.0 / len(PNNs_Leaf_Node[0])
        return weights
        
    def run_all(self):
        
        x = self.params["X"][self.params["instance"],:].reshape(1, -1)
        nfeat = len(self.params["X"][self.params["instance"],:])
        retain, X = self.fit()
        
        mostImpFeats = np.argsort(-self.feature_scores)
        x_p = np.delete(x, mostImpFeats[retain:], axis = 1)
        selected_features = list( set(range(nfeat))-set(mostImpFeats[retain:]))
        print("\n")
        print("#################################")
        print("Selected features by MAPLE : ")
        feature_names = []
        for i in selected_features:
            feature_names.append(self.params["features"][i])
            print(self.params["features"][i])
        curr_leaf_ids = self.fe.apply(x)[0]
        weights = self.training_point_weights(curr_leaf_ids)
           
        # Local linear model
        lr_model = LogisticRegression()
        lr_model.fit(X, self.MR_train, weights)

        # Get the model coeficients
        coefs = np.zeros(self.num_features + 1)
        coefs[0] = lr_model.intercept_
        coefs[np.sort(mostImpFeats[0:retain]) + 1] = lr_model.coef_
        
        # Get the prediction at this point
        prediction = lr_model.predict(x_p.reshape(1, -1))
        
        out = {}
        out["weights"] = weights
        out["coefs"] = coefs
        out["pred"] = prediction
        print("#################################")
        print("\n")
        print("#################################")
        print_instance(self.params["X"][self.params["instance"],:], self.params["features"], self.params["instance"])
        print("#################################")
        print("\n")
        print("Linear approximation prediction = ",  out["pred"][0])
        
        result = sorted(list(zip(lr_model.coef_[0,:], feature_names)), reverse=True)
        print(result)
        self.plot(result)
        
        wgt, smpl = list(zip(*sorted(list(zip(out["weights"], range(len(out["weights"])))), reverse=True)))
        print("\n")
        print("#################################")
        print("Closest sample : ")
        print_instance(X[smpl[0],:], feature_names, smpl[0])
        print("Linear model pred = ", lr_model.predict(X[smpl[0],:].reshape(1, -1))[0])
        print("#################################")
        print("\n")
        print("#################################")
        print("Farhest sample : ")
        print_instance(X[smpl[-1],:], feature_names, smpl[-1])
        print("Linear model pred = ", lr_model.predict(X[smpl[-1],:].reshape(1, -1))[0])
        print("#################################")
        print("\n")
        
        
        return X, lr_model, out
    
    
    def plot(self, result):
        fi_vector, ordered_features = list(zip(*result))
        plt.figure(figsize=(10,10))
        plt.barh(ordered_features, fi_vector)
        plt.title('Feature importance')
        plt.ylabel('Features')
        plt.xlabel('Maple coefficients')
        plt.show()

