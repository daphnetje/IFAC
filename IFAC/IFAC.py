from .BlackBoxClassifier import BlackBoxClassifier

class IFAC:

    def __init__(self, classifier="Random Forest"):
        self.classifier = classifier

    def fit(self, X, y):
        #Step 1: Train Black Box Classifier
        BB = BlackBoxClassifier(self.classifier)
        bb_trained_model = BB.train_model(X, y)

        #Step 2: extract rules
        return

    def predict(self):
        return
    #fit function

    #predict function