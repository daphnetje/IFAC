from .BlackBoxClassifier import BlackBoxClassifier
from Dataset import Dataset
import pandas as pd
from .PD_itemset import PD_itemset, generate_potentially_discriminated_itemsets

class IFAC:

    def __init__(self, val1_ratio=0.1, val2_ratio=0.1, classifier="Random Forest"):
        self.val1_ratio = val1_ratio
        self.val2_ratio = val2_ratio
        self.classifier = classifier



    def fit(self, X, sensitive_attributes):
        # Generate potentially discriminated itemsets
        self.pd_itemsets = generate_potentially_discriminated_itemsets(X, sensitive_attributes)
        print(self.pd_itemsets)
        #Step 0: Split into train and two validation sets
        val1_n = int(self.val1_ratio * len(X.descriptive_data))
        val2_n = int(self.val1_ratio * len(X.descriptive_data))
        X_train_dataset, X_val1_dataset = X.split_into_train_test(val1_n)
        X_train_dataset, X_val1_dataset = X_train_dataset.split_into_train_test(val2_n)

        #Step 1: Train Black-Box Model    #TODO: consider if I want to use X_train_dataset (in Dataset format) or already extract one hot encoded
        BB = BlackBoxClassifier(self.classifier)
        BB.fit(X_train_dataset)

        #Step 2: Extract at-risk subgroups dict, each key is a potentially_discriminated itemset (can be intersectional!) and each value
        #is a list of rules that are problematic
        pred_val1 = BB.predict(X_val1_dataset)
        print(pred_val1)

        return


    def predict(self):
        return

