from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class BlackBoxClassifier:

    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        self.CLASSIFIER_MAPPING = {
        'Decision Tree': DecisionTreeClassifier,
        'Random Forest': RandomForestClassifier,
        'SVM': SVC}


    def get_classifier(self, **kwargs):
        if self.name not in self.CLASSIFIER_MAPPING:
            raise ValueError(
                f"Unsupported classifier type: {self.name}. Supported types are: {list(self.CLASSIFIER_MAPPING.keys())}")
        return self.CLASSIFIER_MAPPING[self.name](**kwargs)

    def train_model(self, X_train, y_train, **kwargs):
        self.classifier = self.get_classifier(self.classifier_name, **kwargs)
        self.classifier.fit(X_train, y_train)
        return self.classifier

    def apply_model(self, X_test):
        predictions = self.classifier.predict(X_test)
        return predictions

