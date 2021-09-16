import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import sys
from sklearn.metrics import accuracy_score, f1_score
from utils.utils import print_instance


class WhiteBox():
    """ Class for fitting an interpretable (or somehow less complex) model that 
        approximates the original black-box model """

    def __init__(self, params):
        self.params = params
        self.params["y"] = self.params["model"].predict(self.params["X"])

    def fit(self):
        if self.params["approximator"] == "logistic":
            whitebox = LogisticRegression(penalty="none", class_weight="balanced")
            whitebox.fit(self.params["X"], self.params["y"])
        
        elif self.params["approximator"] == "tree":
            whitebox = []
            f1 = []
            for i in range(self.params["iterations"]):
                clf = DecisionTreeClassifier(max_depth=2, min_impurity_decrease=0.02, 
                                              min_samples_leaf=0.05, max_features=4, 
                                              class_weight="balanced")
                clf.fit(self.params["X"], self.params["y"])
                whitebox.append(clf)
                f1.append(f1_score(self.params["y"], clf.predict(self.params["X"])))
            
            whitebox = whitebox[f1.index(max(f1))]
            
        else: 
            sys.exit("Specify valid global approximaton method!")
            
        return whitebox

    def plot(self, model):
        if self.params["approximator"] == "logistic":
            fi = model.coef_[0,:]
            result = sorted(zip(fi,self.params["features"]),reverse=True)
            fi_vector, ordered_features = list(zip(*result))
            plt.figure(figsize=(10,10))
            plt.barh(ordered_features, fi_vector)
            plt.title('Feature importance')
            plt.ylabel('Features')
            plt.xlabel('Logistic regression coefficients')
            plt.show()
            
            
        elif self.params["approximator"] == "tree":
            n_nodes = model.tree_.node_count
            children_left = model.tree_.children_left
            children_right = model.tree_.children_right
            feature = model.tree_.feature
            threshold = model.tree_.threshold
            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, 0)]  
            while len(stack) > 0:
                node_id, depth = stack.pop()
                node_depth[node_id] = depth
                is_split_node = children_left[node_id] != children_right[node_id]
                if is_split_node:
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1))
                else:
                    is_leaves[node_id] = True
                    
            feature_names = []
            print("The binary tree structure has {n} nodes and has "
                "the following tree structure:\n".format(n=n_nodes))
            for i in range(n_nodes):
                if is_leaves[i]:
                    print("{space}node={node} is a leaf node.".format(
                        space=node_depth[i] * "\t", node=i))
                else:
                    print("{space}node={node} is a split node: "
                        "go to node {left} if {feature} <= {threshold} "
                        "else to node {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=self.params["features"][feature[i]],
                        threshold=threshold[i],
                        right=children_right[i])
                    )
                    feature_names.append(self.params["features"][feature[i]])
                    
            
            plt.figure(figsize=(15,15))
            plt.title('Tree structure: ')
            tree.plot_tree(model, proportion=True, feature_names = self.params["features"], 
                           class_names = self.params["classes"])
            plt.show()
            
            node_indicator = model.decision_path(self.params["X"])
            leaf_id = model.apply(self.params["X"])
            
            node_index = node_indicator.indices[node_indicator.indptr[self.params["instance"]]:
                                    node_indicator.indptr[self.params["instance"] + 1]]

            print('Rules used to predict sample {id}:\n'.format(id=self.params["instance"]))
            print_instance(self.params["X"][self.params["instance"],:],
                           self.params["features"],self.params["instance"])
            for node_id in node_index:
    # continue to the next node if it is a leaf node
                if leaf_id[self.params["instance"]] == node_id:
                    continue

    # check if value of the split feature for sample 0 is below threshold
                if (self.params["X"][self.params["instance"], feature[node_id]] <= threshold[node_id]):
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                print("decision node {node} : ({feature} = {value}) "
                    "{inequality} {threshold})".format(
                    node=node_id,
                    feature=self.params["features"][feature[node_id]],
                    value=self.params["X"][self.params["instance"], feature[node_id]],
                    inequality=threshold_sign,
                    threshold=threshold[node_id]))
            
        else: 
            sys.exit("Specify valid global approximaton method!")
            
  
    def run_all(self):
        model = self.fit()
        y_pred = model.predict(self.params["X"])
        accuracy = accuracy_score(self.params["y"], y_pred)
        print("White box model approximates black box with accuracy = ", accuracy)
        self.plot(model)



