from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

