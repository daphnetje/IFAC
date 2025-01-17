from .BlackBoxClassifier import BlackBoxClassifier
from .PD_itemset import generate_potentially_discriminated_itemsets
from .Rule import get_instances_covered_by_rule_base, remove_rules_that_are_subsets_from_other_rules, convert_to_apriori_format, initialize_rule, calculate_support_conf_slift_and_significance, calculate_support_and_conf_of_rule
from copy import deepcopy
from apyori import apriori
import pandas as pd

class IFAC:

    def __init__(self, sensitive_attributes, reference_group, val1_ratio=0.2, val2_ratio=0.05, base_classifier="Random Forest", max_pvalue_slift=0.01):
        self.sensitive_attributes = sensitive_attributes
        self.reference_group = reference_group
        self.val1_ratio = val1_ratio
        self.val2_ratio = val2_ratio
        self.base_classifier = base_classifier
        self.max_pvalue_slift = max_pvalue_slift

    def fit(self, X):
        # Generate potentially discriminated itemsets
        self.pd_itemsets = generate_potentially_discriminated_itemsets(X, self.sensitive_attributes)
        self.decision_attribute = X.decision_attribute
        self.positive_label = X.desirable_label
        self.class_items = frozenset([X.decision_attribute + " : " + X.undesirable_label, X.decision_attribute + " : " + X.desirable_label])

        #Step 0: Split into train and two validation sets
        val1_n = int(self.val1_ratio * len(X.descriptive_data))
        val2_n = int(self.val1_ratio * len(X.descriptive_data))
        X_train_dataset, X_val1_dataset = X.split_into_train_test(val1_n)
        X_train_dataset, X_val1_dataset = X_train_dataset.split_into_train_test(val2_n)

        #Step 1: Train Black-Box Model    #TODO: consider if I want to use X_train_dataset (in Dataset format) or already extract one hot encoded
        self.BB = BlackBoxClassifier(self.base_classifier)
        self.BB.fit(X_train_dataset)

        #Step 2: Extract at-risk subgroups dict, each key is a potentially_discriminated itemset (can be intersectional!) and each value
        #is a list of rules that are problematic
        class_rules_per_prot_itemset = self.learn_reject_rules(X_val1_dataset)
        final_reject_rules = self.construct_protected_itemset_and_their_reject_rules_dict(class_rules_per_prot_itemset)

        #Step 3
        #Learn uncertainty reject thresholds
        self.unfair_and_certain_limit = 0
        self.fair_and_uncertain_limit = 0
        return


    def learn_reject_rules(self, val_set):
        pred_val1 = self.BB.predict(val_set)
        val_descriptive = val_set.descriptive_data
        decision_label = val_set.decision_attribute

        val_data_with_preds = deepcopy(val_descriptive)
        val_data_with_preds = val_data_with_preds.drop(columns=[decision_label])
        val_data_with_preds[decision_label] = pred_val1
        val_data_with_preds.to_csv('val_data')

        disc_rules_per_prot_itemset = {}
        for prot_itemset in self.pd_itemsets:
            print(prot_itemset)
            disc_rules_for_prot_itemset = self.extract_disc_rules_for_one_prot_itemset(prot_itemset, val_data_with_preds)
            disc_rules_per_prot_itemset[prot_itemset] = disc_rules_for_prot_itemset

        return disc_rules_per_prot_itemset


    def extract_disc_rules_for_one_prot_itemset(self, prot_itemset, val_data):
        data_belonging_to_prot_itemset = get_instances_covered_by_rule_base(prot_itemset.dict_notation, val_data)
        data_belonging_to_prot_itemset = data_belonging_to_prot_itemset.drop(columns=self.sensitive_attributes)

        data_apriori_format = convert_to_apriori_format(data_belonging_to_prot_itemset)
        all_rules = list(apriori(transactions=data_apriori_format, min_support=0.01,
                               min_confidence=0.85, min_lift=1.0, min_length=2,
                               max_length=4))

        discriminatory_rules = []

        for rule in all_rules:
            if rule.items.isdisjoint(self.class_items):
                continue
            for ordering in rule.ordered_statistics:
                rule_base = ordering.items_base
                rule_consequence = ordering.items_add
                if (not rule_consequence.isdisjoint(self.class_items)) & (len(rule_consequence) == 1):
                    rule_base_with_prot_itemset = rule_base.union(prot_itemset.frozenset_notation)
                    myRule = initialize_rule(rule_base_with_prot_itemset, rule_consequence)
                    support_over_all_data, conf_over_all_data, slift, slift_p = calculate_support_conf_slift_and_significance(
                        myRule, val_data, prot_itemset)
                    myRule.set_support(support_over_all_data); myRule.set_confidence(conf_over_all_data)
                    myRule.set_slift(slift); myRule.set_slift_p_value(slift_p)
                    discriminatory_rules.append(myRule)
        return discriminatory_rules

    def construct_protected_itemset_and_their_reject_rules_dict(self, class_rules_per_prot_itemset):
        reject_rules_per_prot_itemset = {}
        for pd_itemset in self.pd_itemsets:
            if pd_itemset.dict_notation != {}:
                reject_rules_per_prot_itemset[pd_itemset] = []

        reject_rules_per_prot_itemset = {}
        #could do a more advanced thing here, also taking confidence and everything into account
        for pd_itemset, rules in class_rules_per_prot_itemset.items():
            if pd_itemset.dict_notation != {}:
                significant_rules_with_high_slift = [rule for rule in rules if (((rule.confidence - rule.slift) < 0.5) & (rule.slift_p_value < self.max_pvalue_slift))]
                rules_with_high_slift_no_subrules = remove_rules_that_are_subsets_from_other_rules(significant_rules_with_high_slift)
                reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_subrules
                #this part is written to only have 'favouritism' rules for reference group
                if pd_itemset.dict_notation not in self.reference_group:
                    rules_with_high_slift_no_subrules_no_favouritism_Rules = [rule for rule in rules_with_high_slift_no_subrules if (rule.rule_consequence[self.decision_attribute] != self.positive_label)]
                    reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_subrules_no_favouritism_Rules

        for pd_itemset, reject_rules in reject_rules_per_prot_itemset.items():
            print(pd_itemset)
            for reject_rule in reject_rules:
                print(reject_rule)
        return reject_rules_per_prot_itemset



    def predict(self):

        return

