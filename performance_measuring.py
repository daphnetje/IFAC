from IFAC.Reject import Reject
from sklearn.metrics import confusion_matrix
import pandas as pd
from IFAC.Rule import get_instances_covered_by_rule_base

def extract_performance_df_over_non_rejected_instances(classification_method, data, predictions, pd_itemsets):
    desirable_label = data.desirable_label
    undesirable_label = data.undesirable_label
    ground_truth = data.descriptive_data[data.decision_attribute]

    rejected_predictions = predictions[predictions.apply(lambda x: isinstance(x, Reject))]

    non_rejected_predictions = predictions[predictions.apply(lambda x: not isinstance(x, Reject))]
    non_rejected_part_of_data = data.descriptive_data.loc[non_rejected_predictions.index]

    performance_df = pd.DataFrame([])

    for protected_itemset in pd_itemsets:
        performance_entry = {"Classification Type": classification_method, "Group": protected_itemset.string_notation,
                             "Sensitive Features": protected_itemset.sensitive_features}
        indices_of_protected_itemsets_data = get_instances_covered_by_rule_base(protected_itemset.dict_notation, non_rejected_part_of_data).index
        predictions_for_protected_itemset = predictions[indices_of_protected_itemsets_data]
        ground_truth_for_protected_itemset = ground_truth[indices_of_protected_itemsets_data]

        conf_matrix = confusion_matrix(ground_truth_for_protected_itemset, predictions_for_protected_itemset,
                                       labels=[desirable_label, undesirable_label])

        performance_entry["Accuracy"] = calculate_accuracy_based_on_conf_matrix(conf_matrix)
        performance_entry["Positive Dec. Ratio"] = calculate_positive_decision_ratio_based_on_conf_matrix(conf_matrix)
        performance_entry["FNR"] = calculate_false_negative_rate_based_on_conf_matrix(conf_matrix)
        performance_entry["FPR"] = calculate_false_positive_rate_based_on_conf_matrix(conf_matrix)
        performance_entry["Number of instances"] = calculate_number_of_instances_based_on_conf_matrix(conf_matrix)

        performance_df = performance_df.append(performance_entry, ignore_index=True)
    return performance_df




def make_confusion_matrix_for_every_protected_itemset(desirable_label, undesirable_label, ground_truth, predicted_labels, protected_info, protected_itemsets, print_matrix=False):
    conf_matrix_dict = {}

    for protected_itemset in protected_itemsets:
        protected_itemset_dict = protected_itemset.dict_notation
        indices_belonging_to_this_pi = get_instances_covered_by_rule_base(protected_itemset_dict, protected_info).index
        ground_truth_of_indices = ground_truth.loc[indices_belonging_to_this_pi]
        predictions_for_indices = predicted_labels.loc[indices_belonging_to_this_pi]
        conf_matrix = confusion_matrix(ground_truth_of_indices, predictions_for_indices, labels=[desirable_label, undesirable_label])
        conf_matrix_dict[protected_itemset] = conf_matrix
        if print_matrix:
            print(protected_itemset)
            print(f"Total number of instances: {calculate_number_of_instances_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"Positive Decision Ratio: {calculate_positive_decision_ratio_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"False Positive Rate: {calculate_false_positive_rate_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"False Negative Rate: {calculate_false_negative_rate_based_on_conf_matrix(conf_matrix):.2f}")
            print(conf_matrix)
            print("_________")
    return conf_matrix_dict


def calculate_accuracy_based_on_conf_matrix(conf_matrix):
    number_true_negatives = conf_matrix[1][1]
    number_true_positives = conf_matrix[0][0]

    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]

    accuracy = (number_true_negatives + number_true_positives) / total
    return accuracy


def calculate_positive_decision_ratio_based_on_conf_matrix(conf_matrix):
    number_false_positives = conf_matrix[1][0]
    number_true_positives = conf_matrix[0][0]

    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]

    pos_ratio = (number_false_positives + number_true_positives) / total
    return pos_ratio


def calculate_false_positive_rate_based_on_conf_matrix(conf_matrix):
    number_false_positives = conf_matrix[1][0]
    number_true_negatives = conf_matrix[1][1]

    fpr = (number_false_positives) / (number_false_positives + number_true_negatives)
    return fpr


def calculate_false_negative_rate_based_on_conf_matrix(conf_matrix):
    number_false_negatives = conf_matrix[0][1]
    number_true_positives = conf_matrix[0][0]

    fnr = (number_false_negatives) / (number_false_negatives + number_true_positives)
    return fnr

def calculate_recall_based_on_conf_matrix(conf_matrix):
    number_true_positives = conf_matrix[0][0]
    number_false_negatives = conf_matrix[0][1]

    number_of_actual_positives = number_true_positives + number_false_negatives
    recall = number_true_positives/number_of_actual_positives
    return recall

def calculate_precision_based_on_conf_matrix(conf_matrix):
    number_true_positives = conf_matrix[0][0]
    number_false_positives = conf_matrix[1][0]

    number_of_predicted_positives = number_true_positives + number_false_positives
    precision = number_true_positives/number_of_predicted_positives
    return precision#


def calculate_number_of_instances_based_on_conf_matrix(conf_matrix):
    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]
    return total