from .BlackBoxClassifier import BlackBoxClassifier
from Dataset import Dataset

class IFAC:

    #
    def __init__(self, val1_ratio=0.1, val2_ratio=0.1, classifier="Random Forest"):
        self.val1_ratio = val1_ratio
        self.val2_ratio = val2_ratio
        self.classifier = classifier


    def fit(self, X):
        #Step 0: Split into train and two validation sets
        val1_n = int(self.val1_ratio * len(X.descriptive_data))
        val2_n = int(self.val1_ratio * len(X.descriptive_data))
        X_train_dataset, X_val1_dataset = X.split_into_train_test(val1_n)
        X_train_dataset, X_val1_dataset = X_train_dataset.split_into_train_test(val2_n)

        #Step 1: Train Black-Box Model
        
        return


    def predict(self):
        return
