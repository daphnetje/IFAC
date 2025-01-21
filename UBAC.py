from IFAC.BlackBoxClassifier import BlackBoxClassifier
import pandas as pd
from IFAC.Reject import create_uncertainty_based_reject

class UBAC:
    def __init__(self, coverage, val_ratio, base_classifier):
        self.coverage = coverage
        self.val_ratio = val_ratio
        self.base_classifier = base_classifier

    def fit(self, X):
        val_n = int(self.val_ratio * len(X.descriptive_data))
        X_train_dataset, X_val_dataset = X.split_into_train_test(val_n)

        n_to_reject = int((1-self.coverage) * val_n)

        # Step 1: Train Black-Box Model
        self.BB = BlackBoxClassifier(self.base_classifier)
        self.BB.fit(X_train_dataset)

        #Step 2: Apply on validation data
        pred_val, proba_val = self.BB.predict_with_proba(X_val_dataset)

        #Step 3: Learn threshold
        self.threshold = self.decide_on_probability_threshold(proba_val, n_to_reject)
        return


    def decide_on_probability_threshold(self, prediction_probabilities, n_instances_to_reject):
        ordered_prediction_probs = prediction_probabilities.sort_values(ascending=True)

        if (n_instances_to_reject > len(prediction_probabilities)):
            cut_off_probability = 0.5

        else:
            cut_off_probability = ordered_prediction_probs.iloc[n_instances_to_reject]

        return cut_off_probability


    def predict(self, X):
        predictions, probabilities = self.BB.predict_with_proba(X)

        indices_below_uncertainty_threshold = probabilities[probabilities < self.threshold].index

        all_uncertainty_based_rejects_df = pd.DataFrame({
            'prediction_without_reject': predictions[indices_below_uncertainty_threshold],
            'prediction probability': probabilities[indices_below_uncertainty_threshold],
        }, index=indices_below_uncertainty_threshold)

        all_uncertainty_based_rejects_series = all_uncertainty_based_rejects_df.apply(
            create_uncertainty_based_reject, axis=1)

        predictions.update(all_uncertainty_based_rejects_series)
        return predictions
