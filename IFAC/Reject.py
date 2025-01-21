
class Reject:
    def __init__(self, reject_threat, prediction_without_reject, prediction_probability):
        self.reject_threat = reject_threat
        self.prediction_without_reject = prediction_without_reject
        self.prediction_probability = prediction_probability

    def __str__(self):
        reject_str_pres = "\n______________________________\n"
        reject_str_pres += self.reject_threat + "-based Reject for this instance\n"
        reject_str_pres += "\nPrediction that would have been made: " + str(self.prediction_without_reject)
        reject_str_pres += "\nPrediction Probability: " + str(self.prediction_probability)
        return reject_str_pres


class UnfairnessReject(Reject):
    def __init__(self, prediction_without_reject, prediction_probability, rule_reject_is_based_upon, sit_test_summary):
        Reject.__init__(self, "Unfairness", prediction_without_reject, prediction_probability)
        self.sit_test_summary = sit_test_summary
        self.rule_reject_is_based_upon = rule_reject_is_based_upon


    def __str__(self):
        str_pres = Reject.__str__(self)
        str_pres += "\nRejection Based on this rule\n"
        str_pres += str(self.rule_reject_is_based_upon)

        if self.sit_test_summary != None:
            str_pres += "\nSituation Testing Score: " + str(self.sit_test_summary)
        return str_pres

class UncertaintyReject(Reject):

    def __init__(self, prediction_without_reject,  prediction_probability):
        Reject.__init__(self, "Uncertain Probability", prediction_without_reject,  prediction_probability)

    def __str__(self):
        str_pres = Reject.__str__(self)
        str_pres += "\nDecision will be deferred to human"
        return str_pres