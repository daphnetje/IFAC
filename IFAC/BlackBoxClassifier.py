from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class BlackBoxClassifier:

    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        self.CLASSIFIER_MAPPING = {
        'Decision Tree': DecisionTreeClassifier,
        'Random Forest': RandomForestClassifier,
        'SVM': SVC}


    def get_classifier(self, **kwargs):
        if self.classifier_name not in self.CLASSIFIER_MAPPING:
            raise ValueError(
                f"Unsupported classifier type: {self.classifier_name}. Supported types are: {list(self.CLASSIFIER_MAPPING.keys())}")
        return self.CLASSIFIER_MAPPING[self.classifier_name](**kwargs)

    def fit(self, X_train_dataset, **kwargs):
        self.classifier = self.get_classifier(**kwargs)
        y_train = X_train_dataset.descriptive_data[X_train_dataset.decision_attribute]
        X_train = X_train_dataset.one_hot_encoded_data.loc[:, X_train_dataset.one_hot_encoded_data.columns != X_train_dataset.decision_attribute]
        self.classifier.fit(X_train, y_train)
        return self.classifier

    def predict(self, X_test_dataset):
        y_test = X_test_dataset.descriptive_data[X_test_dataset.decision_attribute]
        X_test = X_test_dataset.one_hot_encoded_data.loc[:,
                  X_test_dataset.one_hot_encoded_data.columns != X_test_dataset.decision_attribute]

        predictions = self.classifier.predict(X_test)
        print(accuracy_score(y_test, predictions))

        return predictions


    def predict_with_proba(self, X_dataset):
        X = X_dataset.one_hot_encoded_data.loc[:,
                 X_dataset.one_hot_encoded_data.columns != X_dataset.decision_attribute]
        predicted_labels = pd.Series(self.classifier.predict(X))

        # Predict the probabilities for each class
        predicted_probabilities = self.classifier.predict_proba(X)

        # Get the probability corresponding to the predicted label for each instance
        probabilities_for_labels = pd.Series(predicted_probabilities.max(axis=1))

        return predicted_labels, probabilities_for_labels


